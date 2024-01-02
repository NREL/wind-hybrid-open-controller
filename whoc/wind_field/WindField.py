# %matplotlib inline
'''
Generate 'true' wake field raw_data for use in GP learning procedure
Inputs: Yaw Angles, Freestream Wind Velocity, Freestream Wind Direction, Turbine Topology
Need csv containing 'true' wake characteristics at each turbine (variables) at each time-step (rows).
'''

# git add . & git commit -m "updates" & git push origin
# ssh ahenry@eagle.hpc.nrel.gov
# cd ...
# sbatch ...

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from floris import tools as wfct
import pandas as pd
from multiprocessing import Pool
# from CaseGen_General import CaseGen_General
from postprocessing import plot_wind_farm
from init import *
from array import array
from scipy.interpolate import interp1d, LinearNDInterpolator

# **************************************** Initialization **************************************** #

# Initialize
fi_sim = wfct.floris_interface.FlorisInterface(WAKE_FIELD_CONFIG["floris_input_file"])
# fi_model = wfct.floris_interface.FlorisInterface(floris_model_dir)

# for fi_temp in [fi_sim, fi_model]:
#     assert fi_temp.get_model_parameters()["Wake Deflection Parameters"]["use_secondary_steering"] == False
#     assert "use_yaw_added_recovery" not in fi_temp.get_model_parameters()["Wake Deflection Parameters"] or fi_temp.get_model_parameters()["Wake Deflection Parameters"]["use_yaw_added_recovery"] == False
#     assert "calculate_VW_velocities" not in fi_temp.get_model_parameters()["Wake Deflection Parameters"] or fi_temp.get_model_parameters()["Wake Deflection Parameters"]["calculate_VW_velocities"] == False

# **************************************** GENERATE TIME-VARYING FREESTREAM WIND SPEED/DIRECTION, YAW ANGLE, TURBINE TOPOLOGY SWEEP **************************************** #
print(f'Simulating {N_CASES} total wake field cases...')

# **************************************** CLASS ********************************************* #
class WakeField:
    def __init__(self, **config: dict):
        self.episode_time_step = None
        self.offline_probability = (
            config["offline_probability"] if "offline_probability" in config else 0.001
        )
        
        self.floris_input_file = (
            config["floris_input_file"]
        )
        
        self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
        self.n_turbines = self.wind_farm.floris.farm.n_turbines
        self.wind_farm.turbine_indices = list(range(self.n_turbines))
        max_downstream_dist = max(self.wind_farm.floris.farm.coordinates[t].x1
                                  for t in range(self.n_turbines))
        min_downstream_dist = min(self.wind_farm.floris.farm.coordinates[t].x1
                                  for t in range(self.n_turbines))
        # exclude most downstream turbine
        upstream_turbine_indices = [t for t in range(self.n_turbines) if
                                    self.wind_farm.floris.farm.coordinates[t].x1 < max_downstream_dist]
        n_upstream_turbines = len(upstream_turbine_indices)
        self.downstream_turbine_indices = [t for t in range(self.n_turbines) if
                                      self.wind_farm.floris.farm.coordinates[t].x1 > min_downstream_dist]
        self.n_downstream_turbines = len(self.downstream_turbine_indices)
        
        # set wind speed/dir change probabilities and variability parameters
        self.wind_speed_change_probability = config["wind_speed_change_probability"]  # 0.1
        self.wind_dir_change_probability = config["wind_dir_change_probability"]  # 0.1
        self.yaw_angle_change_probability = config["yaw_angle_change_probability"]
        self.ai_factor_change_probability = config["ai_factor_change_probability"]
        
        self.wind_speed_var = config["wind_speed_var"]  # 0.5
        self.wind_dir_var = config["wind_dir_var"]  # 5.0
        self.yaw_angle_var = config["yaw_angle_var"]
        self.ai_factor_var = config["ai_factor_var"]
        
        self.wind_speed_turb_std = config["wind_speed_turb_std"]  # 0.5
        self.wind_dir_turb_std = config["wind_dir_turb_std"]  # 5.0
        self.yaw_angle_turb_std = config["yaw_angle_turb_std"]  # 0
        self.ai_factor_turb_std = config["ai_factor_turb_std"]  # 0
        
        self.episode_max_time_steps = config["episode_max_time_steps"]
        self.wind_speed_sampling_time_step = config["wind_speed_sampling_time_step"]
        self.wind_dir_sampling_time_step = config["wind_dir_sampling_time_step"]
        self.yaw_angle_sampling_time_step = config["yaw_angle_sampling_time_step"]
        self.ai_factor_sampling_time_step = config["ai_factor_sampling_time_step"]
        
        self.yaw_angle_roc = config["yaw_angle_roc"]
        
    def _generate_online_bools_ts(self):
        return np.random.choice(
                [0, 1], size=(self.episode_max_time_steps, self.n_turbines),
                p=[self.offline_probability, 1 - self.offline_probability])
    
    def _generate_change_ts(self, val_range, val_var, change_prob, noise_std, sample_time_step, roc=None):
        # initialize at random wind speed
        init_val = np.random.choice(
            np.arange(val_range[0], val_range[1], val_var)
        )
        
        # randomly increase or decrease mean wind speed or keep static
        random_vals = np.random.choice(
                    [-val_var, 0, val_var],
                    size=(int(self.episode_max_time_steps // sample_time_step)),
                    p=[ change_prob / 2,
                        1 - change_prob,
                        change_prob / 2,
                    ]
                )
        if roc is None:
            # if we assume instantaneous change (ie over a single DT)
            a = array('d', [
                y
                for x in random_vals
                for y in (x,) * sample_time_step]) # repeat random value x sample_time_step times
            delta_vals = array('d',
                               interp1d(np.arange(0, self.episode_max_time_steps, sample_time_step), random_vals,
                                        fill_value='extrapolate', kind='previous')(
                                       np.arange(0, self.episode_max_time_steps, 1)))
        
        else:
            # else we assume a linear change between now and next sample time, considering roc as max slope allowed
            for i in range(len(random_vals) - 1):
                diff = random_vals[i + 1] - random_vals[i]
                if abs(diff) > roc * sample_time_step:
                    random_vals[i + 1] = random_vals[i] + (diff / abs(diff)) \
                                         * (roc * sample_time_step)
                    
            assert (np.abs(np.diff(random_vals)) <= roc * sample_time_step).all()
            delta_vals = array('d',
                               interp1d(np.arange(0, self.episode_max_time_steps, sample_time_step),
                                       random_vals, kind='linear', fill_value='extrapolate')(
                                       np.arange(0, self.episode_max_time_steps, 1)))
            
            
        
        noise_vals = np.random.normal(scale=noise_std, size=(self.episode_max_time_steps,))
        
        # add mean and noise and bound the wind speed to given range
        ts = array('d', [init_val := np.clip(init_val + delta + n, val_range[0], val_range[1])
                                       for delta, n in zip(delta_vals, noise_vals)])
        
        return ts
    def _generate_freestream_wind_speed_ts(self):
        freestream_wind_speed_ts = self._generate_change_ts(WIND_SPEED_RANGE, self.wind_speed_var,
                                                                 self.wind_speed_change_probability,
                                                                 self.wind_speed_turb_std,
                                                                 self.wind_speed_sampling_time_step)
        return freestream_wind_speed_ts
        
    def _generate_freestream_wind_dir_ts(self):
        freestream_wind_dir_ts = self._generate_change_ts(WIND_DIR_RANGE, self.wind_dir_var,
                                                                 self.wind_dir_change_probability,
                                                                 self.wind_dir_turb_std,
                                                                 self.wind_dir_sampling_time_step)
        return freestream_wind_dir_ts
    
    def _generate_yaw_angle_ts(self):
        yaw_angle_ts = [self._generate_change_ts(YAW_ANGLE_RANGE, self.yaw_angle_var,
                                                      self.yaw_angle_change_probability,
                                                      self.yaw_angle_turb_std,
                                                      self.yaw_angle_sampling_time_step,
                                                      roc=self.yaw_angle_roc)
                        for i in range(self.n_turbines)]
        yaw_angle_ts = np.array(yaw_angle_ts).T
        
        return yaw_angle_ts
    
    def _generate_ai_factor_ts(self):
        online_bool_ts = self._generate_online_bools_ts()
        
        set_ai_factor_ts = [self._generate_change_ts(AI_FACTOR_RANGE, self.ai_factor_var,
                                                      self.ai_factor_change_probability,
                                                      self.ai_factor_var,
                                                      self.ai_factor_sampling_time_step)
                            for _ in range(self.n_turbines)]
        set_ai_factor_ts = np.array(set_ai_factor_ts).T
        
        effective_ai_factor_ts = np.array(
            [
                [set_ai_factor_ts[i][t] if online_bool_ts[i][t] else EPS
                 for t in range(self.n_turbines)]
                for i in range(self.episode_max_time_steps)
            ]
        )
        
        return effective_ai_factor_ts
    
def plot_ts(wf):
    # Plot vs. time
    fig_ts, ax_ts = plt.subplots(2, 2, sharex=True) #len(case_list), 5)
    if hasattr(ax_ts, '__len__'):
        ax_ts = ax_ts.flatten()
    else:
        ax_ts = [ax_ts]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(wf.downstream_turbine_indices)))

    # for case_idx in range(n_cases):
        # if case_idx == 0 or n_cases < 5:
        
    time = wf.df['Time']
    freestream_wind_speed = wf.df['FreestreamWindSpeed'].to_numpy()
    freestream_wind_dir = wf.df['FreestreamWindDir'].to_numpy()
    ds_turbine_wind_speeds = np.hstack(
        [wf.df[f'TurbineWindSpeeds_{t}'].to_numpy()[:, np.newaxis]
         for t in wf.downstream_turbine_indices])
    # ds_turbine_wind_dirs = np.hstack(
    #     [wf.df[f'TurbineWindDirs_{t}'].to_numpy()[:, np.newaxis]
    #      for t in wf.downstream_turbine_indices])
    yaw_angles = np.hstack(
        [wf.df[f'YawAngles_{t}'].to_numpy()[:, np.newaxis]
         for t in range(wf.n_turbines)])
    ai_factors = np.hstack(
        [wf.df[f'AxIndFactors_{t}'].to_numpy()[:, np.newaxis]
         for t in range(wf.n_turbines)])
    
    # ds_turbine_wind_speeds_model = np.hstack(
    #     [wf.df[f'TurbineWindSpeedsModel_{t}'].to_numpy()[:, np.newaxis]
    #      for t in wf.downstream_turbine_indices])
    
    ax_ts[0].plot(time, freestream_wind_speed)
    ax_ts[1].plot(time, freestream_wind_dir)
    
    for t_idx, t in enumerate(wf.downstream_turbine_indices):
        ax_ts[2].plot(time, ds_turbine_wind_speeds[:, t_idx], label=f'DS Turbine {t}',
                             c=colors[t_idx])
        # ax_ts[2].plot(time, ds_turbine_wind_dirs[:, t_idx], label=f'DS Turbine {t}',
        #                  c=colors[t_idx])
        ax_ts[3].plot(time, yaw_angles[:, t_idx], label=f'Turbine {t}',
                         c=colors[t_idx])
        ax_ts[4].plot(time, ai_factors[:, t_idx], label=f'Turbine {t}',
                         c=colors[t_idx])
    
    ax_ts[0].set(title='Freestream Wind Speed [m/s]')
    ax_ts[1].set(title='Freestream Wind Direction [deg]')
    ax_ts[2].set(title='Turbine Effective Rotor Wind Speed [m/s]')
    # ax_ts[3].set(title='Turbine Wind Direction [deg]')
    ax_ts[3].set(title='Turbine Yaw Angle [deg]')
    ax_ts[4].set(title='Turbine AI Factor [-]')

        # ax_ts[case_idx].plot(time, ds_turbine_wind_speeds_model[:, t_idx], label=f'DS Turbine {t} Case {case_idx}',
        #                      linestyle='--', c=colors[t_idx])
    
    for ax in ax_ts:
        ax.set(xticks=time[0:-1:int(60 // DT)], xlabel='Time [s]')

    fig_ts.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wake_field_ts.png'))
    # fig_ts.show()

def generate_wake_ts(config, case_idx):
    wf = WakeField(**config)
    print(f'Simulating case #{case_idx}')
    
    # Initialize
    fi_sim = wfct.floris_interface.FlorisInterface(config["floris_input_file"])
    # fi_model = wfct.floris_interface.FlorisInterface(floris_model_dir)
    
    # define yaw angle time series
    yaw_angles = np.array(wf._generate_yaw_angle_ts())
    # ai_factors = np.array(wf._generate_ai_factor_ts())
    freestream_wind_speeds = wf._generate_freestream_wind_speed_ts()
    freestream_wind_dirs = wf._generate_freestream_wind_dir_ts()
    fi_sim.reinitialize(wind_speeds=[freestream_wind_speeds[0]], wind_directions=[freestream_wind_dirs[0]])
    # fi_sim.calculate_wake(yaw_angles=yaw_angles[0, :], axial_induction=ai_factors[0, :])
    fi_sim.calculate_wake(yaw_angles=yaw_angles[0, :][np.newaxis, np.newaxis, :])
    # fi_model.reinitialize_flow_field(wind_speed=freestream_wind_speeds[0], wind_direction=freestream_wind_dirs[0])
    # fi_model.calculate_wake(yaw_angles=yaw_angles[0, :], axial_induction=ai_factors[0, :])
    
    # lists that will be needed for visualizationsd
    turbine_wind_speeds_sim = []
    # turbine_wind_speeds_model = [[] for t in range(wf.n_turbines)]
    # TODO create floris instance for each turbine with stochastic 'freestream' wind speed and direction,
    #  use calculated wind speed at turbine immediately downstream as 'freestream' wind speed for downstream turbine?
    # turbine_wind_dirs_sim = [[] for t in range(wf.n_turbines)]
    ai_factors = []
    
    # turbine_turb_intensities_sim = [[] for t in range(wf.n_turbines)]
    # turbine_wind_dirs_model = [[] for t in range(wf.n_turbines)]
    # turbine_turb_intensities_model = [[] for t in range(wf.n_turbines)]
    # turbine_powers_sim = [[] for t in range(wf.n_turbines)]
    # turbine_powers_model = [[] for t in range(wf.n_turbines)]
    
    time = np.arange(0, wf.episode_max_time_steps) * DT
    horizontal_planes = []
    y_planes = []
    cross_planes = []
    
    for tt, sim_time in enumerate(time):
        if sim_time % 100 == 0:
            print("Simulation Time:", sim_time, "For Case:", case_idx)
        
        # fi_sim.floris.farm.flow_field.mean_wind_speed = freestream_wind_speeds[tt]
        # fi_model.floris.farm.flow_field.mean_wind_speed = freestream_wind_speeds[tt]
        
        fi_sim.reinitialize(wind_speeds=[freestream_wind_speeds[tt]],
                                       wind_directions=[freestream_wind_dirs[tt]])
        # fi_model.reinitialize_flow_field(wind_speed=freestream_wind_speeds[tt],
        #                                  wind_direction=freestream_wind_dirs[tt],
        #                                  sim_time=sim_time)
        
        # calculate dynamic wake computationally
        fi_sim.calculate_wake(yaw_angles=yaw_angles[tt, :][np.newaxis, np.newaxis, :])
        # fi_model.calculate_wake(yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt], sim_time=sim_time)
        
        if case_idx == 0 and False:
            fi_sim.turbine_indices = np.arange(wf.n_turbines)
            farm_fig = plot_wind_farm(fi_sim)
            farm_fig.show()
            horizontal_planes.append(fi_sim.get_hor_plane(x_resolution=200, y_resolution=100,
                                                          height=90.0))  # horizontal plane at hub-height
            y_planes.append(fi_sim.get_y_plane(x_resolution=200, z_resolution=100,
                                               y_loc=0.0))  # vertical plane parallel to freestream wind direction
            cross_planes.append(fi_sim.get_cross_plane(y_resolution=100, z_resolution=100,
                                                       x_loc=630.0))  # vertical plane parallel to turbine disc plane
        
        # for t in range(wf.n_turbines):
            # QUESTION effective vs average velocities? former takes yaw misalignment into consideration?
        turbine_wind_speeds_sim.append(fi_sim.turbine_effective_velocities[0, 0, :])
            # turbine_wind_speeds_model[t].append(fi_model.floris.farm.turbines[t].average_velocity)
            # turbine_wind_dirs_sim[t].append(fi_sim.floris.farm.wind_map.turbine_wind_direction[t])
        ai_factors.append(fi_sim.get_turbine_ais()[0, 0, :])
            # turbine_wind_dirs_model[t].append(fi_model.floris.farm.wind_map.turbine_wind_direction[t])
            # turbine_turb_intensities_sim[t].append(fi_sim.floris.farm.turbulence_intensity[t])
            # turbine_turb_intensities_model[t].append(fi_model.floris.farm.turbulence_intensity[t])
            # turbine_powers_sim[t].append(fi_sim.floris.farm.turbines[t].power / 1e6)
            # turbine_powers_model[t].append(fi_model.floris.farm.turbines[t].power / 1e6)
        
        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
        
        # calculate steady-state wake computationally
        # fi.calculate_wake(yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt])
        
        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
    
    turbine_wind_speeds_sim = np.vstack(turbine_wind_speeds_sim)
    # turbine_wind_dirs_sim = np.array(turbine_wind_dirs_sim).T
    # turbine_wind_speeds_model = np.array(turbine_wind_speeds_model).T
    # yaw_angles = np.vstack(yaw_angles)
    ai_factors = np.vstack(ai_factors)
    
    # turbine_wind_dirs_sim = np.array(turbine_wind_dirs_sim).T
    # turbine_wind_dirs_model = np.array(turbine_wind_dirs_model).T
    # turbine_turb_intensities_sim = np.array(turbine_turb_intensities_sim).T
    # turbine_turb_intensities_model = np.array(turbine_turb_intensities_model).T
    # turbine_powers_sim = np.array(turbine_powers_sim).T
    # turbine_powers_model = np.array(turbine_powers_model).T
    
    # save case raw_data as dataframe
    wake_field_data = {
        'Time': time,
        'FreestreamWindSpeed': freestream_wind_speeds,
        'FreestreamWindDir': freestream_wind_dirs
    }
    
    for t in range(wf.n_turbines):
        wake_field_data = {**wake_field_data,
                           f'TurbineWindSpeeds_{t}': turbine_wind_speeds_sim[:, t],
                           # f'TurbineWindSpeedsModel_{t}': turbine_wind_speeds_model[:, t],
                           # f'TurbineWindDirs_{t}': turbine_wind_dirs_sim[:, t],
                           # f'TurbineWindDirsModel_{t}': turbine_wind_dirs_model[:, t],
                           # f'TurbineTI_{t}': turbine_turb_intensities_sim[:, t],
                           # f'TurbineTIModel_{t}': turbine_turb_intensities_model[:, t],
                           # f'TurbinePowers_{t}': turbine_powers_sim[:, t],
                           # f'TurbinePowersModel_{t}': turbine_powers_model[:, t],
                           f'YawAngles_{t}': yaw_angles[:, t],
                           f'AxIndFactors_{t}': ai_factors[:, t]
                           }
    
    wake_field_df = pd.DataFrame(data=wake_field_data)
    
    # export case raw_data to csv
    wake_field_df.to_csv(os.path.join(DATA_SAVE_DIR, f'case_{case_idx}.csv'))
    wf.df = wake_field_df
    wf.horizontal_planes = horizontal_planes
    wf.y_planes = y_planes
    wf.cross_planes = cross_planes
    
    return wf


if __name__ == '__main__':
    if N_CASES == 1:
        wake_field_data = []
        for i in range(N_CASES):
            wake_field_data.append(generate_wake_ts(WAKE_FIELD_CONFIG, i))
        plot_ts(wake_field_data[0])
    else:
        pool = Pool()
        res = pool.map(partial(generate_wake_ts, WAKE_FIELD_CONFIG), range(N_CASES))
        pool.close()
    