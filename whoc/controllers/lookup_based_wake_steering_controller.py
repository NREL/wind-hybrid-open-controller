# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import numpy as np
import pandas as pd
import os
import re
from memory_profiler import profile

from whoc.controllers.controller_base import ControllerBase
from floris.floris_model import FlorisModel
from floris.uncertain_floris_model import UncertainFlorisModel

from scipy.interpolate import LinearNDInterpolator
from scipy.signal import lfilter

from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LookupBasedWakeSteeringController(ControllerBase):
    def __init__(self, interface, wind_forecast, simulation_input_dict, verbose=False, **kwargs):
        super().__init__(interface, verbose=verbose)
        self.init_time = interface.init_time
        self.wind_forecast = wind_forecast
        self.simulation_dt = simulation_input_dict["simulation_dt"]
        self.controller_dt = simulation_input_dict["controller"]["controller_dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = interface.n_turbines #simulation_input_dict["controller"]["num_turbines"]
        
        # self.filtered_measurements = pd.DataFrame(columns=["time"] + [f"ws_horz_{tid}" for tid in range(self.n_turbines)] + [f"ws_vert_{tid}" for tid in range(self.n_turbines)], dtype=pd.Float64Dtype())
        # self.ws_lpf_alpha = np.exp(-simulation_input_dict["controller"]["ws_lpf_omega_c"] * simulation_input_dict["controller"]["lpf_T"])
        self.use_filt = simulation_input_dict["controller"]["use_lut_filtered_wind_dir"]
        self.lpf_time_const = simulation_input_dict["controller"]["lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=simulation_input_dict["controller"]["lpf_start_time"])        
        self.lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.deadband_thr = simulation_input_dict["controller"]["deadband_thr"]
        self.floris_input_file = simulation_input_dict["controller"]["floris_input_file"]
        self.yaw_limits = simulation_input_dict["controller"]["yaw_limits"]
        self.yaw_rate = simulation_input_dict["controller"]["yaw_rate"]
        self.yaw_increment = simulation_input_dict["controller"]["yaw_increment"]
        self.max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 16
        self.rated_turbine_power = simulation_input_dict["controller"]["rated_turbine_power"]
        self.wind_field_ts = kwargs["wind_field_ts"]
        self.wf_source = kwargs["wf_source"]
        
        self.turbine_signature = kwargs["turbine_signature"]
        self.tid2idx_mapping = kwargs["tid2idx_mapping"]
        self.idx2tid_mapping = dict([(i, k) for i, k in enumerate(self.tid2idx_mapping.keys())])
        self.target_turbine_indices = simulation_input_dict["controller"]["target_turbine_indices"]
        
        if self.target_turbine_indices != "all":
            self.sorted_tids = sorted(list(self.target_turbine_indices))
        else:
            self.sorted_tids = np.arange(len(self.tid2idx_mapping))
            
        self.uncertain = simulation_input_dict["controller"]["uncertain"]
        
        if self.wind_forecast and self.uncertain:
            self.mean_ws_horz_cols = [f"loc_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            self.mean_ws_vert_cols = [f"loc_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            self.sd_ws_horz_cols = [f"sd_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            self.sd_ws_vert_cols = [f"sd_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            # self.mean_nd_cos_cols = [f"loc_nd_cos_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            # self.mean_nd_sin_cols = [f"loc_nd_sin_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        else:
            self.mean_ws_horz_cols = [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            self.mean_ws_vert_cols = [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
            self.sd_ws_horz_cols = self.sd_ws_vert_cols = []
         
        # self.turbine_ids = np.arange(self.n_turbines) + 1
        self.historic_measurements = pd.DataFrame(columns=["time"] 
                                                + [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.tid2idx_mapping))]
                                                + [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.tid2idx_mapping))]
                                                + [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
                                                + [f"nd_sin_{tid}" for tid in self.tid2idx_mapping], 
                                                dtype=pd.Float64Dtype())

        self._last_measured_time = None
        self.is_yawing = np.array([False for _ in range(self.n_turbines)])

        # Handle yaw optimizer object
        if "df_yaw" in kwargs:
            self.wake_steering_interpolant = get_yaw_angles_interpolant(kwargs["df_yaw"])
        else:
            # optimize, unless passed existing lookup table
            # os.path.abspath(lut_path)
            # this is generated for new layout if len(target_turbine_ids) < n_turbines
            self.wake_steering_interpolant = LookupBasedWakeSteeringController._optimize_lookup_table(
                floris_config_path=self.floris_input_file, 
                lut_path=simulation_input_dict["controller"]["lut_path"], 
                yaw_limits=self.yaw_limits,
                generate_lut=simulation_input_dict["controller"]["generate_lut"],
                uncertain=self.uncertain, 
                sorted_target_tids=self.sorted_tids if self.target_turbine_indices != "all" else "all",
                parallel=kwargs["multiprocessor"] is not None)
        
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

        # For startup
        self.previous_target_yaw_setpoints = self.controls_dict["yaw_angles"]
        self.yaw_norm_const = 360.0
    
    def _first_ord_filter(self, x):
        
        b = [1 - self.lpf_alpha]
        a = [1, -self.lpf_alpha]
        return lfilter(b, a, x)
    
    
    @staticmethod
    def _optimize_lookup_table(floris_config_path, uncertain, yaw_limits, parallel=False, optimization="scipy", sorted_target_tids="all", lut_path=None, generate_lut=True):
        if not generate_lut and lut_path is not None and os.path.exists(lut_path):
            df_lut = pd.read_csv(lut_path, index_col=0)
            df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
        else:
            # if csv to load from is not given, optimize
            # LUT optimizer wind field options
            wind_directions_lut = np.arange(0.0, 360.0, 3.0)
            # wind_directions_lut = np.arange(0.0, 360.0, 60.0)
            wind_speeds_lut = np.arange(6.0, 22.0, 2.0)
            # wind_speeds_lut = np.arange(6.0, 22.0, 6.0)
            
            ## Get optimized AEP, with wake steering
            
            # Load a FLORIS object for yaw optimization, adapting to target_turbine_ids
            # for uncertain case, add extra dimension for standard deviation of wind dir
            if uncertain:
                # wd_stddevs_lut = np.arange(1.0, 10.0, 2.0)
                wd_stddevs_lut = np.arange(1.0, 10.0, 2.0)
                 
                fi_lut = UncertainFlorisModel(floris_config_path,
                                                wd_resolution=0.5,
                                                ws_resolution=0.5,
                                                ti_resolution=0.01,
                                                yaw_resolution=0.5,
                                                power_setpoint_resolution=100,
                                                wd_std=wd_stddevs_lut[0])
                # wd_grid, ws_grid, wds_grid = np.meshgrid(wind_directions_lut, wind_speeds_lut, wd_stddevs_lut, indexing="ij")
                
            else:
                fi_lut = FlorisModel(floris_config_path)  # GCH model matched to the default "legacy_gauss" of V2
            
            wd_grid, ws_grid = np.meshgrid(wind_directions_lut, wind_speeds_lut, indexing="ij")
            
            if sorted_target_tids != "all":
                fi_lut.set(layout_x=fi_lut.layout_x[sorted_target_tids], 
                           layout_y=fi_lut.layout_y[sorted_target_tids])
            
            if uncertain:
                fi_lut.set(
                    wind_directions=wd_grid.flatten(),
                    wind_speeds=ws_grid.flatten(),
                    wd_stddevs=wd_stddevs_lut,
                    turbulence_intensities=[fi_lut.core.flow_field.turbulence_intensities[0]] * len(ws_grid.flatten())
                )
            else:
                fi_lut.set(
                        wind_directions=wd_grid.flatten(),
                        wind_speeds=ws_grid.flatten(),
                        turbulence_intensities=[fi_lut.core.flow_field.turbulence_intensities[0]] * len(ws_grid.flatten())
                    )
                
            # fi_lut.run()
            # turbine_powers = fi_lut.get_turbine_powers(per_wd_sample=True)
             
            if optimization == "scipy":
                yaw_opt = YawOptimizationScipy(fi_lut, 
                                            minimum_yaw_angle=yaw_limits[0],
                                            maximum_yaw_angle=yaw_limits[1], parallel=parallel,
                                            include_wd_stddev=uncertain)
            elif optimization == "sr":
                # TODO update this based on MPC implementation
                yaw_opt = YawOptimizationSR(fi_lut, 
                                            minimum_yaw_angle=yaw_limits[0],
                                            maximum_yaw_angle=yaw_limits[1],
                                            include_wd_stddev=uncertain)
            else:
                raise TypeError("optimization argument must equal 'scipy' or 'sr'")
            df_lut = yaw_opt.optimize()
            
            # Assume linear ramp up at 5-6 m/s and ramp down at 13-14 m/s,
            # add to table for linear interpolant
            df_copy_lb = df_lut[df_lut["wind_speed"] == 6.0].copy()
            df_copy_ub = df_lut[df_lut["wind_speed"] == 13.0].copy()
            df_copy_lb["wind_speed"] = 5.0
            df_copy_ub["wind_speed"] = 14.0
            df_copy_lb["yaw_angles_opt"] *= 0.0
            df_copy_ub["yaw_angles_opt"] *= 0.0
            df_lut = pd.concat([df_copy_lb, df_lut, df_copy_ub], axis=0).reset_index(drop=True)
            
            # Deal with 360 deg wrapping: solutions at 0 deg are also solutions at 360 deg
            df_copy_360deg = df_lut[df_lut["wind_direction"] == 0.0].copy()
            df_copy_360deg["wind_direction"] = 360.0
            df_lut = pd.concat([df_lut, df_copy_360deg], axis=0).reset_index(drop=True)
            # ['wind_direction', 'wind_speed', 'turbulence_intensity',
            #        'yaw_angles_opt', 'farm_power_opt', 'farm_power_baseline']
            os.makedirs(os.path.dirname(lut_path), exist_ok=True)
            if lut_path is not None:
                df_lut.to_csv(lut_path)

        # pd.unique(df_lut.iloc[np.where(np.any(np.vstack(df_lut["yaw_angles_opt"].array) != 0, axis=1))[0]]["wind_direction"])
        # Derive linear interpolant from solution space
        if uncertain:
            return LinearNDInterpolator(
                points=df_lut[["wind_direction", "wind_speed", "wd_stddev"]].values,
                values=np.vstack(df_lut["yaw_angles_opt"].values),
                fill_value=0.0,
            )
        else:
            return LinearNDInterpolator(
                points=df_lut[["wind_direction", "wind_speed"]].values,
                values=np.vstack(df_lut["yaw_angles_opt"].values),
                fill_value=0.0,
            )
    
    # @profile
    def compute_controls(self):
        # TODO update LUT for turbine breakdown
        # TODO: move data collection for filtering purposes to another method that is called every simulation_dt, also move constraints on yaw angles and incremental updates as per yaw_rate to simulator, only job of compute_controls should be to compute new yaw angles for turbines that are not in motion
        
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            return

        # if self.verbose:
        #     logging.info(f"self._last_measured_time == {self._last_measured_time}")
        #     logging.info(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self.current_time = self._last_measured_time = self.measurements_dict["time"]

        if self.wf_source == "floris":
            current_wind_directions = self.measurements_dict["wind_directions"]
            # current_farm_wind_direction = self.measurements_dict["amr_wind_direction"]
            # current_farm_wind_speed = self.measurements_dict["amr_wind_speed"]
            current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
            current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
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

        # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
        if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
            # yaw angles will be set to initial values
            if self.verbose:
                logging.info("Bad wind direction measurement received, reverting to previous measurement.")
        
        # NOTE: current_measurements collects measurements corresponding to current time step, NOT since last controller call if wind_dt < controller_dt
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
        assert not pd.isna(current_measurements).values.any() 
        
        # only get wind_dirs corresponding to target_turbine_ids
        current_wind_directions = current_wind_directions[self.sorted_tids]
        
        # need historic measurements for filter or for wind forecast
        if self.use_filt or self.wind_forecast:
            self.historic_measurements = pd.concat([self.historic_measurements, 
                                                    current_measurements], axis=0).iloc[-int(np.ceil(self.lpf_time_const // self.simulation_dt) * 1e3):]
 
        current_yaw_setpoints = self.controls_dict["yaw_angles"]
        
        # flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
        reached_setpoints_cond = self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)
        if self.verbose and any(reached_setpoints_cond):
            logging.info(f"LUT Controller turbines {np.where(reached_setpoints_cond)[0]} have reached their target setpoint of {self.previous_target_yaw_setpoints[reached_setpoints_cond]} at time {self.current_time}")
        
        self.is_yawing[reached_setpoints_cond] = False

        new_yaw_setpoints = np.array(current_yaw_setpoints)
        
        # NOTE: this is run every simulation_dt, not every controller_dt, because the yaw angle may be moving gradually towards the correct setpoint
        if self.wind_forecast:
            if self.uncertain:
                # forecasted_wind_sample = self.wind_forecast.predict_sample(self.historic_measurements, self.current_time)
                forecasted_wind_field = self.wind_forecast.predict_distr(self.historic_measurements, self.current_time)
            else:
                forecasted_wind_field = self.wind_forecast.predict_point(self.historic_measurements, self.current_time)
            
            single_forecasted_wind_field = forecasted_wind_field.loc[forecasted_wind_field["time"] == self.current_time + self.wind_forecast.prediction_timedelta, :].iloc[:1]
                
        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            # if (abs((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            
            if self.current_time < self.lpf_start_time or not self.use_filt:
                wind = forecasted_wind_field.iloc[-1] if self.wind_forecast else current_measurements.iloc[0]
                
                wind_dirs = 180.0 + np.rad2deg(np.arctan2(
                    wind[self.mean_ws_horz_cols].values.astype(float), 
                    wind[self.mean_ws_vert_cols].values.astype(float)))
                wind_mags = (wind[self.mean_ws_horz_cols].values.astype(float)**2 + wind[self.mean_ws_vert_cols].values.astype(float)**2)**0.5
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"unfiltered forecasted wind directions = {wind_dirs[self.sorted_tids]}")
                    else:
                        logging.info(f"unfiltered current wind directions = {current_wind_directions}")
                
            else:
                # use filtered wind direction, NOTE historic_measurements includes controller_dt steps into the future such that we can run simulation in time batches
                # forecasted_wind_field.iloc[-1:].rename(columns={old_col: re.search("(?<=loc_)\\w+", old_col).group(0) for old_col in self.mean_ws_horz_cols+self.mean_ws_vert_cols})
                if self.wind_forecast:
                    hist_meas = self.historic_measurements.rename(columns={re.search("(?<=loc_)\\w+", new_col).group(0): new_col for new_col in self.mean_ws_horz_cols + self.mean_ws_vert_cols}) if self.uncertain else self.historic_measurements
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
                
                # pass the historic and forecasted values to the low pass filter
                wind_dirs = np.array([self._first_ord_filter(wind_dirs[:, i])
                                                for i in range(wind_dirs.shape[1])]).T # [-int(self.controller_dt // self.simulation_dt), :]
                wind_dirs = wind_dirs[-1, :]
                
                if self.verbose:
                    if self.wind_forecast:
                        logging.info(f"filtered forecasted wind directions = {wind_dirs[self.sorted_tids]}")
                    else:
                        logging.info(f"filtered current wind directions = {wind_dirs[self.sorted_tids]}")
                        
                # just use the latest forecasted value for the wind magnitude
                wind_mags = (wind.iloc[-1][self.mean_ws_horz_cols].values.astype(float)**2 
                                + wind.iloc[-1][self.mean_ws_vert_cols].values.astype(float)**2)**0.5
                # wind = wind.iloc[-1] # just get the last forecasted values
                
                if self.uncertain:
                    ws_horz_stddevs = single_forecasted_wind_field.iloc[0][self.sd_ws_horz_cols].values.astype(float)
                    ws_vert_stddevs = single_forecasted_wind_field.iloc[0][self.sd_ws_vert_cols].values.astype(float)
                    forecasted_wind_norm = (single_forecasted_wind_field.iloc[0][self.mean_ws_horz_cols].values.astype(float)**2 + single_forecasted_wind_field.iloc[0][self.mean_ws_vert_cols].values.astype(float)**2)
                    c1 = single_forecasted_wind_field.iloc[0][self.mean_ws_vert_cols].values.astype(float) / forecasted_wind_norm
                    c2 = -single_forecasted_wind_field.iloc[0][self.mean_ws_horz_cols].values.astype(float) / forecasted_wind_norm
                    wind_dir_stddevs = ((c1 * ws_horz_stddevs)**2 + (c2 * ws_vert_stddevs)**2)**0.5 
                
            # only get wind_dirs corresponding to target_turbine_ids
            wind_dirs = wind_dirs[self.sorted_tids]
            wind_mags = wind_mags[self.sorted_tids]
            if self.uncertain:
                wind_dir_stddevs = wind_dir_stddevs[self.sorted_tids]
           
            # TODO HIGH feed just upstream turbine or mean?
            if self.target_turbine_indices == "all":
                wd_inp, wm_inp, wd_stddev_inp = wind_dirs.mean(), wind_mags.mean(), wind_dir_stddevs.mean() if self.uncertain else None
            else:
                upstream_turbine_idx = np.argsort(self.target_turbine_indices)[0]
                wd_inp, wm_inp, wd_stddev_inp = wind_dirs[upstream_turbine_idx], wind_mags[upstream_turbine_idx], wind_dir_stddevs[upstream_turbine_idx] if self.uncertain else None
            if self.uncertain:
                target_yaw_offsets = self.wake_steering_interpolant(wd_inp, wm_inp, wd_stddev_inp)
            else:
                target_yaw_offsets = self.wake_steering_interpolant(wd_inp, wm_inp)
            
            target_yaw_setpoints = np.mod(np.rint((wind_dirs - target_yaw_offsets) / self.yaw_increment) * self.yaw_increment, 360.0)
        
            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            setpoint_change = target_yaw_setpoints - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)]) 
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0) # if == 0, need to change within 360 deg, otherwise if == 1 faster to cross 360/0 boundary
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(self.n_turbines)]
            is_target_changing = (abs_setpoint_change > self.deadband_thr) & ~self.is_yawing
            
            # setpoint_change[setpoint_change_idx == 0] = setpoint_change[setpoint_change_idx == 0]
            # if ((setpoint_change_idx == 1) & ((target_yaw_setpoints - current_yaw_setpoints) > 0))
            dir_setpoint_change = np.sign(setpoint_change)
            dir_setpoint_change[setpoint_change_idx == 1] = -dir_setpoint_change[setpoint_change_idx == 1]
            new_yaw_setpoints[is_target_changing] = new_yaw_setpoints[is_target_changing] + dir_setpoint_change[is_target_changing] * abs_setpoint_change[is_target_changing]
            self.is_yawing[is_target_changing] = True
            
            if self.verbose and any(is_target_changing):
                logging.info(f"LUT Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} at time {self.current_time}") 
        else:
            is_target_changing = np.zeros_like(self.is_yawing).astype(bool)
        
        reaching_setpoints_cond = self.is_yawing & ~is_target_changing
        if self.verbose and any(reaching_setpoints_cond):
            # from [351.15 320.2 ] to [20.65 12.4 ] 
            logging.info(f"LUT Controller continuing to yaw turbines {np.where(self.is_yawing)[0]} from {current_yaw_setpoints[self.is_yawing]} to {self.previous_target_yaw_setpoints[self.is_yawing]} at time {self.current_time}")
        
        # else:
        # 	logging.info(f"LUT Controller current_setpoints = {current_yaw_setpoints}, \n previous_target_yaw_setpoints = {self.previous_target_yaw_setpoints}, \n target_setpoints={target_yaw_setpoints}")
        
        new_yaw_setpoints[reaching_setpoints_cond] = self.previous_target_yaw_setpoints[reaching_setpoints_cond].copy()

        # stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
        self.previous_target_yaw_setpoints = np.mod(np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment, 360)
        
        lb, ub = current_yaw_setpoints - self.simulation_dt * self.yaw_rate, current_yaw_setpoints + self.simulation_dt * self.yaw_rate
        constrained_yaw_setpoints = np.mod(np.clip(new_yaw_setpoints, lb, ub), 360.0)
        
        # if np.all(np.diff(constrained_yaw_setpoints) == 0) and not np.all(np.diff(new_yaw_setpoints) == 0):
        # 	logging.info(f"Note: all yaw angles have been constrained by the yaw rate equally at time {self.current_time}")
        
        # if not len(is_target_changing):
        # 	logging.info(f"Note: no yaw angle setpoints surpass the deadband threshold at time {self.current_time}")

        constrained_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment
        # constrained_yaw_setpoints = np.clip(constrained_yaw_setpoints, *reversed([current_wind_directions - yl for yl in self.yaw_limits]))
        
        # self.init_sol = {"states": list(constrained_yaw_setpoints / self.yaw_norm_const)}
        # self.init_sol["control_inputs"] = (constrained_yaw_setpoints - self.controls_dict["yaw_angles"]) * (self.yaw_norm_const / (self.yaw_rate * self.controller_dt))
        
        if self.wind_forecast:
            # wf.filter(pl.col("time") < pl.col("time").first() + preview_forecast.controller_timedelta)
            newest_predictions = forecasted_wind_field.loc[
                forecasted_wind_field["time"] <= self.current_time + max(pd.Timedelta(self.controller_dt, unit="s"), self.wind_forecast.prediction_timedelta), :]
            self.controls_dict = {
                "yaw_angles": list(constrained_yaw_setpoints),
                "predicted_wind_speeds": newest_predictions[["time"] + self.mean_ws_horz_cols + self.mean_ws_vert_cols 
                                                            + ((self.sd_ws_horz_cols + self.sd_ws_vert_cols) if self.uncertain else [])]
                # "predicted_wind_speeds_vert": newest_predictions[self.mean_ws_vert_cols].values
            }
        else:
            self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)} 

            
        return None

def get_yaw_angles_interpolant(df_opt, ramp_up_ws=[4, 5], ramp_down_ws=[10, 12], minimum_yaw_angle=None, maximum_yaw_angle=None):
    """Create an interpolant for the optimal yaw angles from a dataframe
    'df_opt', which contains the rows 'wind_direction', 'wind_speed',
    'turbulence_intensity', and 'yaw_angles_opt'. This dataframe is typically
    produced automatically from a FLORIS yaw optimization using Serial Refine
    or SciPy. One can additionally apply a ramp-up and ramp-down region
    to transition between non-wake-steering and wake-steering operation.

    Args:
        df_opt (pd.DataFrame): Dataframe containing the rows 'wind_direction',
        'wind_speed', 'turbulence_intensity', and 'yaw_angles_opt'.
        ramp_up_ws (list, optional): List with length 2 depicting the wind
        speeds at which the ramp starts and ends, respectively, on the lower
        end. This variable defaults to [4, 5], meaning that the yaw offsets are
        zero at and below 4 m/s, then linearly transition to their full offsets
        at 5 m/s, and continue to be their full offsets past 5 m/s. Defaults to
        [4, 5].
        ramp_down_ws (list, optional): List with length 2 depicting the wind
        speeds at which the ramp starts and ends, respectively, on the higher
        end. This variable defaults to [10, 12], meaning that the yaw offsets are
        full at and below 10 m/s, then linearly transition to zero offsets
        at 12 m/s, and continue to be zero past 12 m/s. Defaults to [10, 12].

    Returns:
        LinearNDInterpolator: An interpolant function which takes the inputs
        (wind_directions, wind_speeds, turbulence_intensities), all of equal
        dimensions, and returns the yaw angles for all turbines. This function
        incorporates the ramp-up and ramp-down regions.
    """

    # Load data and set up a linear interpolant
    points = df_opt[["wind_direction", "wind_speed", "turbulence_intensity"]]
    values = np.vstack(df_opt["yaw_angles_opt"])

    # Derive maximum and minimum yaw angle (deg)
    if minimum_yaw_angle is None:
        minimum_yaw_angle = np.min(values)
    if maximum_yaw_angle is None:
        maximum_yaw_angle = np.max(values)

    # Expand wind direction range to cover 0 deg to 360 deg
    points_copied = points[points["wind_direction"] == 0.0].copy()
    points_copied.loc[points_copied.index, "wind_direction"] = 360.0
    values_copied = values[points["wind_direction"] == 0.0, :]
    points = np.vstack([points, points_copied])
    values = np.vstack([values, values_copied])

    # Copy lowest wind speed / TI solutions to -1.0 to create lower bound
    for col in [1, 2]:
        ids_to_copy_lb = points[:, col] == np.min(points[:, col])
        points_copied = np.array(points[ids_to_copy_lb, :], copy=True)
        values_copied = np.array(values[ids_to_copy_lb, :], copy=True)
        points_copied[:, col] = -1.0  # Lower bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

        # Copy highest wind speed / TI solutions to 999.0
        ids_to_copy_ub = points[:, col] == np.max(points[:, col])
        points_copied = np.array(points[ids_to_copy_ub, :], copy=True)
        values_copied = np.array(values[ids_to_copy_ub, :], copy=True)
        points_copied[:, col] = 999.0  # Upper bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

    # Now create a linear interpolant for the yaw angles
    interpolant = LinearNDInterpolator(
        points=points,
        values=values,
        fill_value=np.nan
    )

    # Now create a wrapper function with ramp-up and ramp-down
    def interpolant_with_ramps(wd_array, ws_array, ti_array=None):
        # Deal with missing ti_array
        if ti_array is None:
            ti_ref = float(np.median(interpolant.points[:, 2]))
            ti_array = np.ones(np.shape(wd_array), dtype=float) * ti_ref

        # Format inputs
        wd_array = np.array(wd_array, dtype=float)
        ws_array = np.array(ws_array, dtype=float)
        ti_array = np.array(ti_array, dtype=float)
        yaw_angles = interpolant(wd_array, ws_array, ti_array)
        yaw_angles = np.array(yaw_angles, dtype=float)

        # Define ramp down factor
        rampdown_factor = np.interp(
            x=ws_array,
            xp=[0.0, *ramp_up_ws, *ramp_down_ws, 999.0],
            fp=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        )

        # Saturate yaw offsets to threshold
        axis = len(np.shape(yaw_angles)) - 1
        nturbs = np.shape(yaw_angles)[-1]
        yaw_lb = np.expand_dims(
            minimum_yaw_angle * rampdown_factor, axis=axis
        ).repeat(nturbs, axis=axis)
        yaw_ub = np.expand_dims(
            maximum_yaw_angle * rampdown_factor, axis=axis
        ).repeat(nturbs, axis=axis)

        return np.clip(yaw_angles, yaw_lb, yaw_ub)

    return interpolant_with_ramps