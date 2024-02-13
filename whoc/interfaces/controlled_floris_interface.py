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

from whoc.interfaces.interface_base import InterfaceBase
import numpy as np
from floris.tools import FlorisInterface, ParallelComputingInterface
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import floris.tools.visualization as wakeviz
import matplotlib.pyplot as plt



class ControlledFlorisInterface(InterfaceBase):
    def __init__(self, yaw_limits, dt, yaw_rate, max_workers=16, floris_version='v4'):
        super().__init__()
        self.max_workers = max_workers
        self.yaw_limits = yaw_limits
        self.yaw_rate = yaw_rate
        self.time = 0
        self.dt = dt
        self.floris_version = floris_version
    
    def load_floris(self, config_path):
        # Load the default example floris object
        if self.floris_version == 'v4':
            self.env = FlorisInterface(config_path)  # GCH model matched to the default "legacy_gauss" of V2
        elif self.floris_version == 'dev':
            from floris_dev.tools import FlorisInterface as FlorisInterfaceDev
            self.env = FlorisInterfaceDev(config_path)
        self.n_turbines = self.env.floris.farm.n_turbines
        
        return self
    
    def reset(self, disturbances, init_controls_dict):
        self.env.floris.farm.yaw_angles = disturbances["wind_directions"][0] - np.array(init_controls_dict["yaw_angles"])[np.newaxis, :]
        # self.env.floris.flow_field.u, self.env.floris.flow_field.v
        self.step(disturbances)
        
        return disturbances
    
    @classmethod
    def load_windrose(cls, windrose_path):
        # Grab a linear interpolant from this wind rose
        df = pd.read_csv(windrose_path)
        interp = LinearNDInterpolator(points=df[["wd", "ws"]], values=df["freq_val"], fill_value=0.0)
        return df, interp
    
    def parallelize(self):
        # Pour this into a parallel computing interface
        parallel_interface = "concurrent"
        self.par_env = ParallelComputingInterface(
            fi=self.env,
            max_workers=self.max_workers,
            n_wind_direction_splits=self.max_workers,
            n_wind_speed_splits=1,
            interface=parallel_interface,
            print_timings=True,
        )
    
    def get_measurements(self, hercules_dict=None):
        """ abstract method from Interface class """
        
        u_only_dirs = np.zeros_like(self.env.floris.flow_field.u)
        u_only_dirs[(self.env.floris.flow_field.v == 0) & (self.env.floris.flow_field.u >= 0)] = 270
        u_only_dirs[(self.env.floris.flow_field.v == 0) & (self.env.floris.flow_field.u < 0)] = 90
        u_only_dirs = (u_only_dirs - 180) * (np.pi / 180)
        
        dirs = np.arctan(np.divide(self.env.floris.flow_field.u, self.env.floris.flow_field.v,
                                   out=np.ones_like(self.env.floris.flow_field.u) * np.nan,
                         where=self.env.floris.flow_field.v != 0),
                         out=u_only_dirs,
                         where=self.env.floris.flow_field.v != 0)
        dirs[dirs < 0] = np.pi + dirs[dirs < 0]
        dirs = (dirs * (180/np.pi)) + 180
        # dirs = (np.arctan(dirs) * (180/np.pi)) + 180
        # dirs += u_only_dirs
        dirs = np.squeeze(np.mean(dirs.reshape(*dirs.shape[:2], -1), axis=2))
        

        dirs = np.repeat(self.env.floris.flow_field.wind_directions, (self.n_turbines,))
        
        mags = np.sqrt(self.env.floris.flow_field.u**2 + self.env.floris.flow_field.v**2 + self.env.floris.flow_field.w**2)
        mags = np.squeeze(np.mean(mags.reshape(*mags.shape[:2], -1), axis=2))
        
        measurements = {"time": self.time,
                        "wind_directions": dirs,
                        "wind_speeds": mags,# self.env.turbine_average_velocities,
                        "powers": np.squeeze(self.env.get_turbine_powers()),
                        "yaw_angles": np.squeeze(dirs - self.env.floris.farm.yaw_angles)}
        
        return measurements
    
    def check_controls(self, ctrl_dict):
        """ abstract method from Interface class """
        ctrl_dict["yaw_angles"] = np.float64(ctrl_dict["yaw_angles"])
        return ctrl_dict

    def step(self, disturbances):
        # reinitialize floris
        yaw_angles = self.env.floris.farm.yaw_angles
        self.env.reinitialize(
            wind_directions=disturbances["wind_directions"],
            wind_speeds=disturbances["wind_speeds"],
            turbulence_intensity=disturbances["turbulence_intensity"]
        )
        
        self.env.calculate_wake(yaw_angles)
        
        return disturbances
    def send_controls(self, hercules_dict, **controls):
        """ abstract method from Interface class """
        self.env.calculate_wake((self.env.floris.flow_field.wind_directions - controls["yaw_angles"])[np.newaxis, :])
        
        self.time += self.dt
        return controls
    
    @classmethod
    def compute_aep(cls, fi, controller, windrose_interpolant, wind_directions=None, wind_speeds=None):
        # Alternatively to below code, we could calculate AEP using
        # 'fi_aep_parallel.get_farm_AEP(...)' but then we would not have the
        # farm power productions, which we use later on for plotting.
        if wind_directions is None:
            wind_directions = np.arange(0.0, 360.0, 1.0)
        if wind_speeds is None:
            wind_speeds = np.arange(1.0, 25.0, 1.0)
        # if controller is None:
        #     yaw_angles = self.env.floris.farm.yaw_angles
        # else:
        # yaw_angles = controller.step()
        
        fi.env.reinitialize(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensity=0.08,  # Assume 8% turbulence intensity
            solver_settings={"turbine_grid_points": 1}
        )
        fi.parallelize()
        
        # Calculate frequency of occurrence for each bin and normalize sum to 1.0
        wd_grid, ws_grid = np.meshgrid(wind_directions, wind_speeds, indexing="ij")
        freq_grid = windrose_interpolant(wd_grid, ws_grid)
        freq_grid = freq_grid / np.sum(freq_grid)
        yaw_grid = controller.yaw_angles_interpolant(wd_grid, ws_grid)
        
        farm_power = fi.par_env.get_farm_power(yaw_grid)
        farm_aep = np.sum(24 * 365 * np.multiply(farm_power, freq_grid))
        farm_energy = np.multiply(freq_grid, farm_power)
        
        print(" ")
        print("===========================================================")
        print(f"Calculating {controller.__class__!r} annual energy production (AEP)...")
        print(f"{controller.__class__!r} AEP: {farm_aep / 1.0e9:.3f} GWh.")
        print("===========================================================")
        print(" ")
        
        return farm_power, farm_aep, farm_energy

if __name__ == '__main__':
    # Parallel options
    max_workers = 16
    
    yaw_limits = (-30, 30)
    
    # results wind field options
    wind_directions_tgt = [250]
    wind_speeds_tgt = [16]
    turbulence_intensity = 0.08
    
    # Load a dataframe containing the wind rose information
    df_windrose, windrose_interpolant \
        = ControlledFlorisInterface.load_windrose(
        windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
    
    # Load a FLORIS object for AEP calculations
    fi_greedy = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=yaw_limits)\
        .load_floris(config_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/emgauss.yaml',
                     wind_directions=wind_directions_tgt, wind_speeds=wind_speeds_tgt,
                     turbulence_intensity=turbulence_intensity)
    
    fi_greedy.send_controls(yaw_angles=np.array([[[float(wd) #+ np.random.randint(-30, 30)
                                             for t in range(fi_greedy.env.floris.farm.n_turbines)]
                                             for ws in wind_speeds_tgt]
                                             for wd in wind_directions_tgt]).flatten())
    
    # Plot a horizatonal slice of the initial configuration
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # axarr = axarr.flatten()
    horizontal_plane = fi_greedy.env.calculate_horizontal_plane(height=90.0)
                                                                # yaw_angles=np.array([[[-30, -20, -10, 0, 10, 20, 30, 20, 10]]]))
    #TODO yawa angles are adjusted to fit inertial frame of reference ...
    wakeviz.plot_turbines_with_fi(fi_greedy.env, ax=ax)
    wakeviz.visualize_cut_plane(
        horizontal_plane,
        ax=ax,
        title="Initial setup",
        min_speed=None,
        max_speed=None
    )
    
    plt.show()
    # 0 yaw angle points to west, positive yaw angle points to southwest, negative yaw angle points to northwest
    fi_greedy.env.floris.farm.yaw_angles
    fi_greedy.env.get_turbine_powers()
    # for yaw=-20 or 270 - wd, wind_dir=250:
    # array([[[5000006.24431751, 5000006.24431751, 5000006.24431751,
    #          5000006.24431403, 5000006.24431402, 5000006.24431402,
    #          5000006.24431403, 5000002.68691173, 5000002.68691173]]])
    # for yaw=0, wind_dir=250:
    # array([[[4999989.56317222, 4999989.56317222, 4999989.56317222,
    #          4999989.56323679, 4999989.5632368, 4999989.5632368,
    #          4999989.56323679, 4999998.55748623, 4999998.55748623]]])
    
    fi_greedy.get_measurements(None)
    
    # plot horizontal plane
    # fi_greedy.env.calculate_horizontal_plane()
    
    # compute AEP with greedy control
    # aep_greedy = fi_greedy.compute_aep(wind_directions=wind_directions_tgt, wind_speeds=wind_speeds_tgt,
    #                                    windrose_interpolant=windrose_interpolant,
    #                                    controller=GreedyController)
    
    # Alternatively to above code, we could calculate AEP using
    # 'fi_aep_parallel.get_farm_AEP(...)' but then we would not have the
    # farm power productions, which we use later on for plotting.
    
    # print(" ")
    # print("===========================================================")
    # print("Calculating greedy annual energy production (AEP)...")
    # print("Greedy AEP: {:.3f} GWh.".format(aep_greedy / 1.0e9))
    # print("===========================================================")
    # print(" ")