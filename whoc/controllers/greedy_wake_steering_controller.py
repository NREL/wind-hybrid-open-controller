"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""
import numpy as np
import pandas as pd

from scipy.signal import lfilter

from whoc.controllers.controller_base import ControllerBase
from memory_profiler import profile
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from floris.tools.visualization import visualize_quiver2

# np.seterr(all="ignore")

class GreedyController(ControllerBase):
    def __init__(self, interface, wind_forecast, simulation_input_dict, verbose=False, **kwargs):
        # print("in GreedyController.__init__")
        super().__init__(interface, verbose=verbose)
        self.n_turbines = interface.n_turbines #simulation_input_dict["controller"]["num_turbines"]
        self.yaw_limits = simulation_input_dict["controller"]["yaw_limits"]
        self.yaw_rate = simulation_input_dict["controller"]["yaw_rate"]
        self.yaw_increment = simulation_input_dict["controller"]["yaw_increment"]
        self.simulation_dt = simulation_input_dict["simulation_dt"]
        self.controller_dt = simulation_input_dict["controller"]["controller_dt"]
        self.init_time = interface.init_time
        self.wind_forecast = wind_forecast
        self.wf_source = kwargs["wf_source"]
        
        self.turbine_signature = kwargs["turbine_signature"]
        self.tid2idx_mapping = kwargs["tid2idx_mapping"]
        self.idx2tid_mapping = dict([(i, k) for i, k in enumerate(self.tid2idx_mapping.keys())])
        self.target_turbine_indices = simulation_input_dict["controller"]["target_turbine_indices"] or "all"
        
        if self.target_turbine_indices != "all":
            self.sorted_tids = sorted(list(self.target_turbine_indices))
        else:
            self.sorted_tids = np.arange(len(self.tid2idx_mapping))
            
        
        self.uncertain = simulation_input_dict["controller"]["uncertain"]
        
        self.mean_ws_horz_cols = [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        self.mean_ws_vert_cols = [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            
        self.historic_measurements = pd.DataFrame(columns=["time"] 
                                                  + self.mean_ws_horz_cols 
                                                  + self.mean_ws_vert_cols
                                                  + [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
                                                  + [f"nd_sin_{tid}" for tid in self.tid2idx_mapping], dtype=pd.Float64Dtype())
        
        self.lpf_time_const = simulation_input_dict["controller"]["lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=simulation_input_dict["controller"]["lpf_start_time"])
        self.lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.deadband_thr = simulation_input_dict["controller"]["deadband_thr"]
        self.use_filt = simulation_input_dict["controller"]["use_filtered_wind_dir"]

        self.rated_turbine_power = simulation_input_dict["controller"]["rated_turbine_power"]

        self.wind_field_ts = kwargs["wind_field_ts"]

        self.is_yawing = np.array([False for _ in range(self.n_turbines)])

        self._last_measured_time = None

        self.yaw_norm_const = 360.0

        # Set initial conditions
        self.yaw_IC = simulation_input_dict["controller"]["initial_conditions"]["yaw"]
   
        if hasattr(self.yaw_IC, "__len__"):
            if len(self.yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": np.array(self.yaw_IC)}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": np.array([self.yaw_IC] * self.n_turbines)}
        
        self.previous_target_yaw_setpoints = self.controls_dict["yaw_angles"]
    
    # self.filtered_measurements["wind_direction"] = []
    
    def _first_ord_filter(self, x):
        b = [1 - self.lpf_alpha]
        a = [1, -self.lpf_alpha]
        return lfilter(b, a, x)
    
    def yaw_offsets_interpolant(self, wind_directions, wind_speeds):
        # return np.zeros((*wind_directions.shape, self.n_turbines))
        return np.array([[[wind_directions[i, j] for t in range(self.n_turbines)] for j in range(wind_directions.shape[1])] for i in range(wind_directions.shape[0])])
    
    # @profile
    def compute_controls(self):
        # print("in GreedyController.compute_controls")
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        # if self.verbose:
        #     logging.info(f"self._last_measured_time == {self._last_measured_time}")
        #     logging.info(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self.current_time = self._last_measured_time = self.measurements_dict["time"]

        # current_wind_directions = np.broadcast_to(self.wind_dir_ts[int(self.current_time // self.simulation_dt)], (self.n_turbines,))
        if self.wf_source == "floris":
            current_wind_directions = self.measurements_dict["wind_directions"]
            current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(current_wind_directions + 180.0)) 
            current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(current_wind_directions + 180.0))
        else:
            current_row = self.wind_field_ts.loc[self.wind_field_ts["time"] == self.current_time, :]
            current_ws_horz = np.hstack([current_row[f"ws_horz_{tid}"].values for tid in self.tid2idx_mapping])
            current_ws_vert = np.hstack([current_row[f"ws_vert_{tid}"].values for tid in self.tid2idx_mapping])
            current_wind_directions = 180.0 + np.rad2deg(
                np.arctan2(
                    current_ws_horz, 
                    current_ws_vert
                )
            )
        
        if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
            # yaw angles will be set to initial values
            if self.verbose:
                logging.info("Bad wind direction measurement received, reverting to previous measurement.")
        
        # pass greedy angles to all non target turbines
        current_nd_cos = np.cos(np.deg2rad(current_wind_directions))
        current_nd_sin = np.sin(np.deg2rad(current_wind_directions))
        current_nd_cos[self.sorted_tids] = np.cos(np.deg2rad(self.measurements_dict["yaw_angles"]))
        current_nd_sin[self.sorted_tids] = np.sin(np.deg2rad(self.measurements_dict["yaw_angles"]))
        
        current_measurements = pd.DataFrame(data={
                "ws_horz": current_ws_horz,
                "ws_vert": current_ws_vert,
                "nd_cos": current_nd_cos,
                "nd_sin": current_nd_sin
        })
        current_measurements = current_measurements.unstack().to_frame().reset_index(names=["data", "turbine_id"])
        current_measurements["turbine_id"] = current_measurements["turbine_id"].apply(lambda tidx: self.idx2tid_mapping[tidx])
        
        current_measurements = current_measurements\
            .assign(data=current_measurements["data"] + "_" + current_measurements["turbine_id"].astype(str), index=0)\
                    .pivot(index="index", columns="data", values=0)
                            # .droplevel(0, axis=0)
        current_measurements = current_measurements.assign(time=self.current_time)
        
        # only get wind_dirs corresponding to target_turbine_ids
        current_wind_directions = current_wind_directions[self.sorted_tids]
        
        if self.use_filt or self.wind_forecast:
            self.historic_measurements = pd.concat([self.historic_measurements, 
                                                    current_measurements], axis=0).iloc[-int(np.ceil(self.lpf_time_const // self.simulation_dt) * 1e3):]
        
        # NOTE: this is run every simulation_dt, not every controller_dt, because the yaw angle may be moving gradually towards the correct setpoint
        
        current_yaw_setpoints = self.controls_dict["yaw_angles"]

        reached_setpoints_cond = self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)
        if self.verbose and any(reached_setpoints_cond):
            logging.info(f"Greedy Controller turbines {np.where(reached_setpoints_cond)[0]} have reached their target setpoint of {self.previous_target_yaw_setpoints[reached_setpoints_cond]} at time {self.current_time}.")

        # flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
        self.is_yawing[reached_setpoints_cond] = False

        new_yaw_setpoints = np.array(current_yaw_setpoints)
        
        if self.wind_forecast:
            forecasted_wind_field = self.wind_forecast.predict_point(self.historic_measurements, self.current_time)
            single_forecasted_wind_field = forecasted_wind_field.loc[forecasted_wind_field["time"] == self.current_time + self.wind_forecast.prediction_timedelta, :].iloc[:1]
        
        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            
            # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
            if self.current_time < self.lpf_start_time or not self.use_filt:
                wind = single_forecasted_wind_field.iloc[-1] if self.wind_forecast else current_measurements.iloc[0]
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(
                    wind[self.mean_ws_horz_cols].values.astype(float), 
                    wind[self.mean_ws_vert_cols].values.astype(float)))
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"unfiltered forecasted wind directions = {wind_dirs[self.sorted_tids]}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                
            else:
                # use filtered wind direction and speed     
                if self.wind_forecast:
                    hist_meas = self.historic_measurements
                    wind = pd.concat([hist_meas, 
                                        forecasted_wind_field[self.mean_ws_horz_cols + self.mean_ws_vert_cols]], axis=0)[
                                            self.mean_ws_horz_cols+self.mean_ws_vert_cols]
                else:
                    wind = self.historic_measurements
                    
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(
                    wind[self.mean_ws_horz_cols].values.astype(float), 
                    wind[self.mean_ws_vert_cols].values.astype(float)))
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"unfiltered forecasted wind directions = {wind_dirs[-1, self.sorted_tids]}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                
                # filter the wind direction
                wind_dirs = np.array([self._first_ord_filter(wind_dirs[:, i])
                                                for i in range(wind_dirs.shape[1])]).T # [-int(self.controller_dt // self.simulation_dt), :]
                wind_dirs = wind_dirs[-1, :]
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"filtered forecasted wind directions = {wind_dirs[self.sorted_tids]}")
                    else:
                        logging.info(f"filtered current wind directions = {wind_dirs[self.sorted_tids]}")
                    
            # only get wind_dirs corresponding to target_turbine_ids
            wind_dirs = wind_dirs[self.sorted_tids]
            
            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            target_yaw_setpoints = np.mod(np.rint(wind_dirs / self.yaw_increment) * self.yaw_increment, 360.0)
            setpoint_change = target_yaw_setpoints - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)]) 
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0) # if == 0, need to change within 360 deg, otherwise if == 1 faster to cross 360/0 boundary
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(self.n_turbines)]
            is_target_changing = (abs_setpoint_change > self.deadband_thr) & ~self.is_yawing
            
            dir_setpoint_change = np.sign(setpoint_change)
            dir_setpoint_change[setpoint_change_idx == 1] = -dir_setpoint_change[setpoint_change_idx == 1]
            new_yaw_setpoints[is_target_changing] = new_yaw_setpoints[is_target_changing] + dir_setpoint_change[is_target_changing] * abs_setpoint_change[is_target_changing]
            self.is_yawing[is_target_changing] = True
            
            if self.verbose and any(is_target_changing):
                logging.info(f"Greedy Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} at time {self.current_time}")
        else:
            is_target_changing = np.zeros_like(self.is_yawing).astype(bool)
        
        reaching_setpoints_cond = self.is_yawing & ~is_target_changing 
        if self.verbose and any(reaching_setpoints_cond):
            logging.info(f"Greedy Controller continuing to yaw turbines {np.where(self.is_yawing)[0]} from {current_yaw_setpoints[self.is_yawing]} to {self.previous_target_yaw_setpoints[self.is_yawing]} at time {self.current_time}")
        
        new_yaw_setpoints[reaching_setpoints_cond] = self.previous_target_yaw_setpoints[reaching_setpoints_cond].copy()
        
        # stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
        self.previous_target_yaw_setpoints = np.mod(np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment, 360)
        
        lb, ub = current_yaw_setpoints - self.simulation_dt * self.yaw_rate, current_yaw_setpoints + self.simulation_dt * self.yaw_rate
        constrained_yaw_setpoints = np.mod(np.clip(new_yaw_setpoints, lb, ub), 360.0)
        
        # constrained_yaw_setpoints = np.clip(constrained_yaw_setpoints, *reversed([current_wind_directions - yl for yl in self.yaw_limits]))
        constrained_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        
        # self.init_sol = {"states": list(constrained_yaw_setpoints / self.yaw_norm_const)}
        # self.init_sol["control_inputs"] = (constrained_yaw_setpoints - self.controls_dict["yaw_angles"]) * (self.yaw_norm_const / (self.yaw_rate * self.controller_dt))

        if self.wind_forecast:
            
            newest_predictions = forecasted_wind_field.loc[
                forecasted_wind_field["time"] <= self.current_time + max(pd.Timedelta(self.controller_dt, unit="s"), self.wind_forecast.prediction_timedelta), :]
            self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints),
                                  "predicted_wind_speeds": newest_predictions[["time"] + self.mean_ws_horz_cols + self.mean_ws_vert_cols]
                                    # "predicted_time":  newest_predictions["time"].values,
                                    # "predicted_wind_speeds_horz": newest_predictions[self.mean_ws_horz_cols].values,
                                    # "predicted_wind_speeds_vert": newest_predictions[self.mean_ws_horz_cols].values
                                    }
        else:
            self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)} 

        return None