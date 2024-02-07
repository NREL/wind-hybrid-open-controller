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

from flasc.wake_steering.lookup_table_tools import get_yaw_angles_interpolant
from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface

from scipy.interpolate import LinearNDInterpolator
from scipy.signal import lfilter

from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


class LookupBasedWakeSteeringController(ControllerBase):
    def __init__(self, interface, input_dict, max_workers=None, df_yaw=None, lut_path=None, verbose=False):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)
        self.historic_measurements = {"wind_directions": np.zeros((0, self.n_turbines)),
                                      "wind_speeds": np.zeros((0, self.n_turbines))}
        self.ws_lpf_alpha = np.exp(-input_dict["controller"]["ws_lpf_omega_c"] * input_dict["controller"]["lpf_T"])
        self.wd_lpf_alpha = np.exp(-input_dict["controller"]["wd_lpf_omega_c"] * input_dict["controller"]["lpf_T"])
        self.floris_input_file = input_dict["controller"]["floris_input_file"]
        self.yaw_limits = input_dict["controller"]["yaw_limits"]
        self.yaw_rate = input_dict["controller"]["yaw_rate"]
        self.max_workers = max_workers or 16
        self.use_filt = input_dict["controller"]["use_filtered_wind_measurements"]

        # Handle yaw optimizer object
        if df_yaw is not None:
            self.wake_steering_interpolant = get_yaw_angles_interpolant(df_yaw)
        else:
            # optimize, unless passed existing lookup table
            # os.path.abspath(lut_path)
            self._optimize_lookup_table(lut_path=lut_path, load_lut=lut_path is not None)
        # else:
            # if self.verbose:
            #     print("No offsets received; assuming nominal aligned control.")
            # self.wake_steering_interpolant = None

        # Set initial conditions
        yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        if hasattr(yaw_IC, "__len__"):
            if len(yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": yaw_IC}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": [yaw_IC] * self.n_turbines}

        # For startup
        self.wd_store = [270.]*self.n_turbines # TODO: update this?
    
    def _first_ord_filter(self, x, alpha):
        
        b = [1 - alpha]
        a = [1, -alpha]
        return lfilter(b, a, x)
    
    def _optimize_lookup_table(self, lut_path=None, load_lut=False):
        if load_lut is True and lut_path is not None and os.path.exists(lut_path):
            df_lut = pd.read_csv(lut_path, index_col=0)
            df_lut["yaw_angles_opt"] = [[float(f) for f in re.findall(r"-*\d+\.\d*", s)] for i, s in
                                        df_lut["yaw_angles_opt"].items()]
        else:
            # if csv to load from is not given, optimize
            # LUT optimizer wind field options
            wind_directions_lut = np.arange(0.0, 360.0, 3.0)
            wind_speeds_lut = np.arange(6.0, 22.0, 2.0)
            
            ## Get optimized AEP, with wake steering
            
            # Load a FLORIS object for yaw optimization
            
            fi_lut = ControlledFlorisInterface(max_workers=self.max_workers, yaw_limits=self.yaw_limits, dt=self.dt,
                                               yaw_rate=self.yaw_rate, floris_version='dev') \
                .load_floris(config_path=self.floris_input_file)
            
            fi_lut.env.reinitialize(
                wind_directions=wind_directions_lut,
                wind_speeds=wind_speeds_lut,
                turbulence_intensity=0.08  # Assume 8% turbulence intensity
            )
            
            # Pour this into a parallel computing interface
            fi_lut.parallelize()
            
            # Now optimize the yaw angles using the Serial Refine method
            # df_lut = fi_lut.optimize_yaw_angles(
            #     minimum_yaw_angle=self.yaw_limits[0],
            #     maximum_yaw_angle=self.yaw_limits[1],
            #     Ny_passes=[5, 4],
            #     exclude_downstream_turbines=True,
            #     exploit_layout_symmetry=True,
            # )
            yaw_opt = YawOptimizationSR(fi_lut.env, minimum_yaw_angle=self.yaw_limits[0],
                maximum_yaw_angle=self.yaw_limits[1],
                # yaw_angles_baseline=np.zeros((len(wind_directions_lut), len(wind_speeds_lut), self.n_turbines)),
                Ny_passes=[5, 4],
                exclude_downstream_turbines=True,
                exploit_layout_symmetry=True, verify_convergence=True)
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
            
            if lut_path is not None:
                df_lut.to_csv(lut_path)
        
        # Derive linear interpolant from solution space
        self.wake_steering_interpolant = LinearNDInterpolator(
            points=df_lut[["wind_direction", "wind_speed"]],
            values=np.vstack(df_lut["yaw_angles_opt"]),
            fill_value=0.0,
        )
    
    def compute_controls(self):
        # wind_directions = self.measurements_dict["wind_directions"]
        # wind_speeds = [8.0] * self.n_turbines  # TODO: enable extraction of wind speed in Hercules
        # if wind_directions is None or not len(wind_directions): # Recieved empty or None
        #     if self.verbose:
        #         print("Bad wind direction measurement received, reverting to previous measurement.")
        #     wind_directions = self.wd_store
        # else:
        #     self.wd_store = wind_directions

        if self.use_filt:
            self.historic_measurements["wind_directions"] = np.vstack([self.historic_measurements["wind_directions"],
                                                            self.measurements_dict["wind_directions"]])
            self.historic_measurements["wind_speeds"] = np.vstack([self.historic_measurements["wind_speeds"],
                                                                self.measurements_dict["wind_speeds"]])
    
        if self.measurements_dict["time"] < 60 or not self.use_filt:
            wind_dir = self.measurements_dict["wind_directions"][0]
            wind_speed = self.measurements_dict["wind_speeds"][0]
        else:
            if self.use_filt:
                # use filtered wind direction and speed
                wind_dir = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"][:, i],
                                                                    self.wd_lpf_alpha)
                                            for i in range(self.n_turbines)]).T[-1, 0]
                wind_speed = np.array([self._first_ord_filter(self.historic_measurements["wind_speeds"][:, i],
                                                                    self.ws_lpf_alpha)
                                                for i in range(self.n_turbines)]).T[-1, 0]
        
        # TODO shouldn't freestream wind speed/dir also be availalbe in measurements_dict, or just assume first row of turbines?
        # TODO filter wind speed and dir before certain time statpm?
        yaw_offsets = self.wake_steering_interpolant(wind_dir, wind_speed)
        # yaw_offsets = np.diag(interpolated_angles)
        # TODO don't think this should be subtracted? depends where wind direction is measured from
        # yaw_setpoints = (np.array(wind_dir) - yaw_offsets).tolist()
        yaw_setpoints = yaw_offsets.tolist()
        
        self.controls_dict = {"yaw_angles": np.clip(yaw_setpoints, *self.yaw_limits)}
        
        return None
    
    def compute_controls_old(self):
        self.wake_steering_angles()

    def wake_steering_angles(self):
        
        # Handle possible bad data
        # TODO from where are the wind directions measured?
        wind_directions = self.measurements_dict["wind_directions"]
        wind_speeds = [8.0]*self.n_turbines # TODO: enable extraction of wind speed in Hercules
        if not wind_directions: # Recieved empty or None
            if self.verbose:
                print("Bad wind direction measurement received, reverting to previous measurement.")
            wind_directions = self.wd_store
        else:
            self.wd_store = wind_directions
        
        # look up wind direction
        if self.wake_steering_interpolant is None:
            yaw_setpoint = wind_directions
        else:
            interpolated_angles = self.wake_steering_interpolant(
                wind_directions,
                wind_speeds,
                None
            )
            yaw_offsets = np.diag(interpolated_angles)
            yaw_setpoint = (np.array(wind_directions) - yaw_offsets).tolist()

        self.controls_dict = {"yaw_angles": yaw_setpoint}

        return None


if __name__ == "__main__":
    from whoc.wind_field.WindField import generate_multi_wind_ts
    from whoc.plotting import plot_power_vs_speed, plot_yaw_vs_dir, plot_power_vs_dir
    import glob
    import matplotlib.pyplot as plt
    from whoc.config import WIND_FIELD_CONFIG, DATA_SAVE_DIR, N_CASES, EPISODE_MAX_TIME

    from hercules.emulator import Emulator
    from hercules.py_sims import PySims
    from hercules.utilities import load_yaml
    
    from whoc.controllers.no_yaw_wake_steering_controller import NoYawController
    from whoc.interfaces.hercules_actuator_disk_yaw_interface import HerculesADYawInterface
    
    from hercules.floris_standin import launch_floris
    
    # TODO how to pass controller to floris here
    # # Clear old log files for clarity
    # rm loghercules logfloris
    # # Set up the helics broker
    # helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT &
    # python3 hercules_runscript.py hercules_input_000.yaml >> loghercules 2>&1 & # Start the controller center and pass in input file
    # python3 floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris 2>&1
    
    # Parallel options
    max_workers = 16

    # input_dict = load_yaml(sys.argv[1])
    input_dict = load_yaml("../../examples/hercules_input_001.yaml")
    interface = HerculesADYawInterface(input_dict)
    controller = LookupBasedWakeSteeringController(interface, input_dict, max_workers=max_workers)
    py_sims = PySims(input_dict)
    
    emulator = Emulator(controller, py_sims, input_dict)
    emulator.run_helics_setup()
    emulator.enter_execution(function_targets=[], function_arguments=[[]])
    
    amr_input_file = "amr_input.inp"
    amr_standin_data_file = "amr_standin_data.csv"
    launch_floris(amr_input_file, amr_standin_data_file)

    
    # results wind field options
    wind_directions_tgt = np.arange(0.0, 360.0, 1.0)
    wind_speeds_tgt = np.arange(1.0, 25.0, 1.0)
    
    # Load a dataframe containing the wind rose information
    df_windrose, windrose_interpolant \
        = ControlledFlorisInterface.load_windrose(
        windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
    
    ## First, get baseline AEP, without wake steering
    
    # Load a FLORIS object for AEP calculations
    fi_noyaw = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
                                         dt=input_dict["dt"],
                                         yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
    fi_noyaw.env.reinitialize(
        wind_directions=wind_directions_tgt,
        wind_speeds=wind_speeds_tgt,
        turbulence_intensity=0.08  # Assume 8% turbulence intensity
    )
    
    # Pour this into a parallel computing interface
    fi_noyaw.parallelize()
    
    # instantiate controller
    # input_dict = {"controller": {
    #     "num_turbines": fi_noyaw.n_turbines, "yaw_limits": fi_noyaw.yaw_limits, "yaw_rate": fi_noyaw.yaw_rate,
    #     "ws_lpf_omega_c": 0.005, "wd_lpf_omega_c": 0.005, "lpf_T": 60,
    #     "lut_path": os.path.join(os.path.dirname(__file__), "../../examples/lut.csv"),
    #     "initial_conditions": {
    #         "yaw": 0
    #     }}}
    ctrl_noyaw = NoYawController(fi_noyaw, input_dict=input_dict)
    
    farm_power_noyaw, farm_aep_noyaw, farm_energy_noyaw = ControlledFlorisInterface.compute_aep(fi_noyaw, ctrl_noyaw,
                                                                                                windrose_interpolant,
                                                                                                wind_directions_tgt,
                                                                                                wind_speeds_tgt)
    
    # instantiate interface
    fi_lut = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
                                       dt=input_dict["dt"],
                                       yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
    
    # instantiate controller, and load lut from csv if it exists
    input_dict["controller"]["floris_input_file"] = WIND_FIELD_CONFIG["floris_input_file"]
    ctrl_lut = LookupBasedWakeSteeringController(fi_lut, input_dict=input_dict)
    
    farm_power_lut, farm_aep_lut, farm_energy_lut = ControlledFlorisInterface.compute_aep(fi_lut, ctrl_lut,
                                                                                          windrose_interpolant,
                                                                                          wind_directions_tgt,
                                                                                          wind_speeds_tgt)
    aep_uplift = 100.0 * (farm_aep_lut / farm_aep_noyaw - 1)
    
    print(" ")
    print("===========================================================")
    print("Calculating optimized annual energy production (AEP)...")
    print(f"Optimized AEP: {farm_aep_lut / 1.0e9:.3f} GWh.")
    print(f"Relative AEP uplift by wake steering: {aep_uplift:.3f} %.")
    print("===========================================================")
    print(" ")
    
    # Now calculate helpful variables and then plot wind rose information
    wd_grid, ws_grid = np.meshgrid(wind_directions_tgt, wind_speeds_tgt, indexing="ij")
    freq_grid = windrose_interpolant(wd_grid, ws_grid)
    freq_grid = freq_grid / np.sum(freq_grid)
    df = pd.DataFrame({
        "wd": wd_grid.flatten(),
        "ws": ws_grid.flatten(),
        "freq_val": freq_grid.flatten(),
        "farm_power_baseline": farm_power_noyaw.flatten(),
        "farm_power_opt": farm_power_lut.flatten(),
        "farm_power_relative": farm_power_lut.flatten() / farm_power_noyaw.flatten(),
        "farm_energy_baseline": farm_energy_noyaw.flatten(),
        "farm_energy_opt": farm_energy_lut.flatten(),
        "energy_uplift": (farm_energy_lut - farm_energy_noyaw).flatten(),
        "rel_energy_uplift": farm_energy_lut.flatten() / np.sum(farm_energy_noyaw)
    })
    
    plot_power_vs_speed(df)
    plot_yaw_vs_dir(ctrl_lut.wake_steering_interpolant, ctrl_lut.n_turbines)
    plot_power_vs_dir(df, fi_lut.env.floris.flow_field.wind_directions)
    
    ## Simulate wind farm with interface and controller
    # instantiate wind field if files don't already exist
    wind_field_filenames = glob(f"{DATA_SAVE_DIR}/case_*.csv")
    if not len(wind_field_filenames):
        generate_multi_wind_ts(WIND_FIELD_CONFIG, N_CASES)
        wind_field_filenames = [f"case_{i}.csv" for i in range(N_CASES)]
    
    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(DATA_SAVE_DIR):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))
    
    # select wind field case
    case_idx = 0
    time_ts = wind_field_data[case_idx]["Time"].to_numpy()
    wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
    wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
    turbulence_intensity_ts = [0.08] * int(EPISODE_MAX_TIME // input_dict["dt"])
    yaw_angles_ts = []
    fi_lut.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
                               "wind_directions": [wind_dir_ts[0]],
                               "turbulence_intensity": turbulence_intensity_ts[0]})
    for k, t in enumerate(np.arange(0, EPISODE_MAX_TIME - input_dict["dt"], input_dict["dt"])):
        print(f'Time = {t}')
        
        # feed interface with new disturbances
        fi_lut.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
                                  "wind_directions": [wind_dir_ts[k]],
                                  "turbulence_intensity": turbulence_intensity_ts[k]})
        
        # receive measurements from interface, compute control actions, and send to interface
        ctrl_lut.step()
        
        print(f"Time = {ctrl_lut.measurements_dict['time']}",
              f"Freestream Wind Direction = {wind_dir_ts[k]}",
              f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
              f"Turbine Wind Directions = {ctrl_lut.measurements_dict['wind_directions']}",
              f"Turbine Wind Magnitudes = {ctrl_lut.measurements_dict['wind_speeds']}",
              f"Turbine Powers = {ctrl_lut.measurements_dict['powers']}",
              f"Yaw Angles = {ctrl_lut.measurements_dict['yaw_angles']}",
              sep='\n')
        yaw_angles_ts.append(ctrl_lut.measurements_dict['yaw_angles'])
    
    yaw_angles_ts = np.vstack(yaw_angles_ts)
    
    filt_wind_dir_ts = ctrl_lut._first_ord_filter(wind_dir_ts, ctrl_lut.wd_lpf_alpha)
    filt_wind_speed_ts = ctrl_lut._first_ord_filter(wind_mag_ts, ctrl_lut.ws_lpf_alpha)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_dir_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
    ax[0].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_dir_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
               label='filtered')
    ax[0].set(title='Wind Direction [deg]', xlabel='Time')
    ax[1].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_mag_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
    ax[1].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_speed_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
               label='filtered')
    ax[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
    # ax.set_xlim((time_ts[1], time_ts[-1]))
    ax[0].legend()
    ax[2].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], yaw_angles_ts)
    fig.show()