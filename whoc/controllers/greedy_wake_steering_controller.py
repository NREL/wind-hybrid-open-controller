"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""
import numpy as np
import pandas as pd

from scipy.signal import lfilter

from whoc.controllers.controller_base import ControllerBase

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from floris.tools.visualization import visualize_quiver2

# np.seterr(all="ignore")

class GreedyController(ControllerBase):
    def __init__(self, interface, wind_forecast, input_dict, verbose=False, **kwargs):
        # print("in GreedyController.__init__")
        super().__init__(interface, verbose=verbose)
        self.n_turbines = interface.n_turbines #input_dict["controller"]["num_turbines"]
        self.yaw_limits = input_dict["controller"]["yaw_limits"]
        self.yaw_rate = input_dict["controller"]["yaw_rate"]
        self.yaw_increment = input_dict["controller"]["yaw_increment"]
        self.simulation_dt = input_dict["simulation_dt"]
        self.dt = input_dict["controller"]["controller_dt"]
        self.init_time = interface.init_time
        self.wind_forecast = wind_forecast
        self.wf_source = kwargs["wf_source"]
        
        # TODO HIGH this needs to be mapped to turbine ids for wind_forecaster
        self.turbine_ids = np.arange(self.n_turbines) + 1
        self.historic_measurements = pd.DataFrame(columns=["time"] + [f"ws_horz_{tid}" for tid in self.turbine_ids] 
                                                  + [f"ws_vert_{tid}" for tid in self.turbine_ids], dtype=pd.Float64Dtype())
        
        self.lpf_time_const = input_dict["controller"]["lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=input_dict["controller"]["lpf_start_time"])
        self.lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["simulation_dt"])
        self.deadband_thr = input_dict["controller"]["deadband_thr"]
        self.use_filt = input_dict["controller"]["use_filtered_wind_dir"]

        self.rated_turbine_power = input_dict["controller"]["rated_turbine_power"]

        self.wind_field_ts = kwargs["wind_field_ts"]

        self.is_yawing = np.array([False for _ in range(self.n_turbines)])

        self._last_measured_time = None

        self.yaw_norm_const = 360.0

        # Set initial conditions
        if isinstance(input_dict["controller"]["initial_conditions"]["yaw"], (float, list)):
            self.yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        elif input_dict["controller"]["initial_conditions"]["yaw"] == "auto":
            self.yaw_IC = None
        else:
            raise Exception("must choose float or 'auto' for initial yaw value")
   
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
    
    def compute_controls(self):
        # print("in GreedyController.compute_controls")
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        if self.verbose:
            logging.info(f"self._last_measured_time == {self._last_measured_time}")
            logging.info(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self._last_measured_time = self.measurements_dict["time"]

        self.current_time = self.measurements_dict["time"]

        # if current_time < 2 * self.simulation_dt:
        
        if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
            # yaw angles will be set to initial values
            if self.verbose:
                logging.info("Bad wind direction measurement received, reverting to previous measurement.")
        
        elif (abs((self.current_time - self.init_time).total_seconds() % self.simulation_dt) == 0.0):
            # current_wind_directions = np.broadcast_to(self.wind_dir_ts[int(self.current_time // self.simulation_dt)], (self.n_turbines,))
            if self.wf_source == "floris":
                current_wind_directions = self.measurements_dict["wind_directions"]
                current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(current_wind_directions + 180.0)) 
                current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(current_wind_directions + 180.0))
            else:
                current_row = self.wind_field_ts.loc[self.wind_field_ts["time"] == self.current_time, :]
                current_ws_horz = np.hstack([current_row[f"ws_horz_{tid}"].values for tid in self.turbine_ids])
                current_ws_vert = np.hstack([current_row[f"ws_vert_{tid}"].values for tid in self.turbine_ids])
                current_wind_directions = 180.0 + np.rad2deg(
                    np.arctan2(
                        current_ws_horz, 
                        current_ws_vert
                    )
                ) 
            
            current_measurements = pd.DataFrame(data={
                    "ws_horz": current_ws_horz,
                    "ws_vert": current_ws_vert
            })
            current_measurements = current_measurements.unstack().to_frame().reset_index(names=["data", "turbine_id"])
            current_measurements["turbine_id"] += 1 # Change from zero index to one index TODO HIGH assumes that outputs of wind forecast will have suffixex = integers 1-index for each turbines 
            
            current_measurements = current_measurements\
                .assign(data=current_measurements["data"] + "_" + current_measurements["turbine_id"].astype(str), index=0)\
                        .pivot(index="index", columns="data", values=0)
                                # .droplevel(0, axis=0)
            current_measurements = current_measurements.assign(time=self.current_time)
            
            if self.verbose:
                logging.info(f"unfiltered wind directions = {current_wind_directions}")
            
            if self.use_filt or self.wind_forecast:
                self.historic_measurements = pd.concat([self.historic_measurements, current_measurements], axis=0).iloc[-int(np.ceil(self.lpf_time_const // self.simulation_dt) * 1e3):]
             
            if self.wind_forecast:
                # TODO HIGH check matching turbine_ids, might be an issue with machine learning models...
                forecasted_wind_field = self.wind_forecast.predict_point(self.historic_measurements, self.current_time)
                 
            # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
            if self.current_time < self.lpf_start_time or not self.use_filt:
                wind = forecasted_wind_field.iloc[-1] if self.wind_forecast else current_measurements
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(
                    wind[[col for col in wind.index if "ws_horz_" in col]].values.astype(float), 
                    wind[[col for col in wind.index if "ws_vert_" in col]].values.astype(float)))
            else:
                # use filtered wind direction and speed
                wind = pd.concat([self.historic_measurements, 
                                  forecasted_wind_field.iloc[-1:]], axis=0)[
                                           [col for col in forecasted_wind_field.columns if col.startswith("ws")]] \
                                               if self.wind_forecast else self.historic_measurements
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(
                    wind[[col for col in wind.columns if "ws_horz_" in col]].values.astype(float), 
                    wind[[col for col in wind.columns if "ws_vert_" in col]].values.astype(float)))
                
                wind_dirs = np.array([self._first_ord_filter(wind_dirs[:, i])
                                                for i in range(self.n_turbines)]).T[-int(self.dt // self.simulation_dt), :]
            
            if self.verbose:
                logging.info(f"{'filtered' if self.use_filt else 'unfiltered'} wind directions = {wind_dirs}")
                
            if np.any(np.isnan(wind_dirs)):
                print("hi")
                
            current_yaw_setpoints = self.controls_dict["yaw_angles"]

            if self.verbose and any(self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)):
                logging.info(f"Greedy Controller turbines {np.where(self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints))[0]} have reached their target setpoint")

            # flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
            self.is_yawing[self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)] = False

            new_yaw_setpoints = np.array(current_yaw_setpoints)

            target_yaw_setpoints = np.rint(wind_dirs / self.yaw_increment) * self.yaw_increment

            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            is_target_changing = (np.abs(target_yaw_setpoints - current_yaw_setpoints) > self.deadband_thr) & ~self.is_yawing

            if self.verbose and any(is_target_changing):
                logging.info(f"Greedy Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} at time {self.current_time}")
            
            if self.verbose and any(self.is_yawing):
                logging.info(f"Greedy Controller continuing to yaw turbines {np.where(self.is_yawing)[0]} from {current_yaw_setpoints[self.is_yawing]} to {self.previous_target_yaw_setpoints[self.is_yawing]} at time {self.current_time}")
            
    
            new_yaw_setpoints[is_target_changing] = target_yaw_setpoints[is_target_changing]
            new_yaw_setpoints[self.is_yawing] = self.previous_target_yaw_setpoints[self.is_yawing].copy()
            
            # stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
            self.previous_target_yaw_setpoints = np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment

            self.is_yawing[is_target_changing] = True
            
            constrained_yaw_setpoints = np.clip(new_yaw_setpoints, current_yaw_setpoints - self.simulation_dt * self.yaw_rate, current_yaw_setpoints + self.simulation_dt * self.yaw_rate)
            constrained_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
            
            self.init_sol = {"states": list(constrained_yaw_setpoints / self.yaw_norm_const)}
            self.init_sol["control_inputs"] = (constrained_yaw_setpoints - self.controls_dict["yaw_angles"]) * (self.yaw_norm_const / (self.yaw_rate * self.dt))

            if self.wind_forecast:
                self.controls_dict = {"yaw_angles": list(np.clip(constrained_yaw_setpoints, *reversed([current_wind_directions - yl for yl in self.yaw_limits]))), 
                                      "predicted_wind_speeds_horz": forecasted_wind_field.iloc[-1][[col for col in forecasted_wind_field.columns if col.startswith("ws_horz")]].values,
                                      "predicted_wind_speeds_vert": forecasted_wind_field.iloc[-1][[col for col in forecasted_wind_field.columns if col.startswith("ws_vert")]].values
                                      }
            else:
                self.controls_dict = {"yaw_angles": list(np.clip(constrained_yaw_setpoints, *reversed([current_wind_directions - yl for yl in self.yaw_limits])))} 

        return None