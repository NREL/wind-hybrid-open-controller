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


def compute_aep(self, interface, controller, windrose_interpolant, wind_directions=None, wind_speeds=None):
    if wind_directions is None:
        wind_directions = np.arange(0.0, 360.0, 1.0)
    if wind_speeds is None:
        wind_speeds = np.arange(1.0, 25.0, 1.0)
    # if controller is None:
    #     yaw_angles = self.env.floris.farm.yaw_angles
    # else:
    yaw_angles = controller.step()
    
    # Calculate frequency of occurrence for each bin and normalize sum to 1.0
    wd_grid, ws_grid = np.meshgrid(wind_directions, wind_speeds, indexing="ij")
    freq_grid = windrose_interpolant(wd_grid, ws_grid)
    freq_grid = freq_grid / np.sum(freq_grid)  # Normalize to 1.0
    
    # Calculate farm power greedy control
    self.env.reinitialize(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensity=turbulence_intensity  # Assume 8% turbulence intensity
    )
    fi_greedy.env.calculate_wake(np.array([[[yaw_angles[t]
                                             for t in range(interface.env.floris.farm.n_turbines)]
                                            for ws in range(len(wind_speeds))]
                                           for wd in range(len(wind_directions))]))
    farm_power_greedy = self.env.get_farm_power()
    aep = np.sum(24 * 365 * np.multiply(farm_power_greedy, freq_grid))
    return aep

class ControlledFlorisInterface(InterfaceBase):
    def __init__(self, max_workers, yaw_limits, dt):
        super().__init__()
        self.max_workers = max_workers
        self.yaw_limits = yaw_limits
        self.time = 0
        self.dt = dt
    
    def load_floris(self, config_path, wind_directions, wind_speeds, turbulence_intensity):
        # Load the default example floris object
        self.env = FlorisInterface(config_path)  # GCH model matched to the default "legacy_gauss" of V2
        self.n_turbines = self.env.floris.farm.n_turbines
        # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model
        
        # Specify wind farm layout and update in the floris object
        # N = 3  # number of turbines per row and per column
        # X, Y = np.meshgrid(
        #     5.0 * self.env.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
        #     5.0 * self.env.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
        # )
        # self.env.reinitialize(layout_x=X.flatten(), layout_y=Y.flatten())
        self.env.reinitialize(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensity=turbulence_intensity  # Assume 8% turbulence intensity
        )
        # np.array([[[ctrl_dict['yaw_angles'][t]
        #             for t in range(self.env.floris.farm.n_turbines)]
        #            for ws in self.env.floris.flow_field.wind_speeds]
        #           for wd in self.env.floris.flow_field.wind_directions])
        self.env.calculate_wake()
        
        return self
    
    @classmethod
    def load_windrose(cls, windrose_path):
        # Grab a linear interpolant from this wind rose
        df = pd.read_csv(windrose_path)
        interp = LinearNDInterpolator(points=df[["wd", "ws"]], values=df["freq_val"], fill_value=0.0)
        return df, interp
    
    def parallelize(self):
        # Pour this into a parallel computing interface
        parallel_interface = "concurrent"
        self.env = ParallelComputingInterface(
            fi=self.env,
            max_workers=self.max_workers,
            n_wind_direction_splits=self.max_workers,
            n_wind_speed_splits=1,
            interface=parallel_interface,
            print_timings=True,
        )
    
    def get_measurements(self, obs_args):
        # reinitialize floris
        self.env.reinitialize(
            wind_directions=obs_args["wind_directions"],
            wind_speeds=obs_args["wind_speeds"],
            turbulence_intensity=obs_args["turbulence_intensity"]  # Assume 8% turbulence intensity
        )
        # np.array([[[ctrl_dict['yaw_angles'][t]
        #             for t in range(self.env.floris.farm.n_turbines)]
        #            for ws in self.env.floris.flow_field.wind_speeds]
        #           for wd in self.env.floris.flow_field.wind_directions])
        self.env.calculate_wake()
        
        dirs = (np.arctan(self.env.floris.flow_field.u / self.env.floris.flow_field.v) * (180/np.pi)) + 180
        dirs = dirs.reshape(*dirs.shape[:3], -1)
        dirs = np.mean(dirs, axis=3)
        mags = np.sqrt(self.env.floris.flow_field.u**2 + self.env.floris.flow_field.v**2 + self.env.floris.flow_field.w**2)
        mags = mags.reshape(*mags.shape[:3], -1)
        mags = np.mean(mags, axis=3)
        measurements = {"time": self.time,
                        "wind_directions": dirs,
                        "wind_speeds": mags,# self.env.turbine_average_velocities,
                        "powers": self.env.get_turbine_powers(),
                        "yaw_angles": self.env.floris.farm.yaw_angles}
        # np.arctan(self.env.floris.flow_field.u / self.env.floris.flow_field.v)
        # measurements = meas_dict
        
        return measurements
    
    def check_controls(self, ctrl_dict):
        return ctrl_dict

    def send_controls(self, other_args=None, **kwargs):
        
        self.env.calculate_wake(np.array([[[kwargs['yaw_angles'][t]
                                                 for t in range(self.env.floris.farm.n_turbines)]
                                                 for ws in self.env.floris.flow_field.wind_speeds]
                                                 for wd in self.env.floris.flow_field.wind_directions]))
        
        self.time += self.dt
        return kwargs

if __name__ == '__main__':
    # Parallel options
    max_workers = 16
    
    # Yaw options TODO put this in config file
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
    
    # Pour this into a parallel computing interface
    # fi_greedy.parallelize()
    
    # calculate wake
    # greedy_ctrl.measurements_dict["turbine_wind_directions"] =
    # greedy_ctrl.compute_controls(wd, ws, fi_greedy.env.yaw_angles[t])
    # fi_greedy.env.calculate_wake(yaw_angles=[[[270 - wd
    #                                          for wd in wind_directions_tgt]
    #                                          for ws in wind_speeds_tgt]
    #                                          for t in range(fi_greedy.env.floris.farm.n_turbines)])
    # TODO check controls should turn yaw angles into floats
    fi_greedy.send_controls(yaw_angles=np.array([[[270. - float(wd) #+ np.random.randint(-30, 30)
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