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
import pickle
import gc

import whoc
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.legend as mlegend
import seaborn as sns
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
import scipy
import os

from memory_profiler import profile

import yaml
from array import array
from scipy.interpolate import interp1d
from collections import defaultdict
from itertools import cycle, chain
from glob import glob
from scipy.signal import lfilter

from whoc.wind_field.plotting import plot_ts, plot_distribution_samples, plot_distribution_ts

factor = 1.5 # double column
plt.rc('font', size=12*factor)          # controls default text sizes
plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
plt.rc('legend', fontsize=12*factor)    # legend fontsize
plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize

class WindField:
    def __init__(self, **config: dict):

        self.fig_dir = config["fig_dir"]
        self.data_save_dir = config["data_save_dir"]

        self.episode_time_step = None
        self.offline_probability = (
            config["offline_probability"] if "offline_probability" in config else 0.001
        )
        
        self.simulation_dt = config["simulation_sampling_time"]
        self.time_series_dt = config["time_series_dt"]

        # self.num_turbines = config["num_turbines"]
        
        # set wind speed/dir change probabilities and variability parameters
        self.wind_speed_change_probability = config["wind_speed"]["change_probability"]  # 0.1
        self.wind_dir_change_probability = config["wind_dir"]["change_probability"]  # 0.1
        self.yaw_angle_change_probability = config["yaw_angles"]["change_probability"]
        self.ai_factor_change_probability = config["ai_factors"]["change_probability"]
        
        self.wind_speed_range = config["wind_speed"]["range"]
        self.wind_speed_u_range = config["wind_speed"]["u_range"]
        self.wind_speed_v_range = config["wind_speed"]["v_range"]
        self.wind_dir_range = config["wind_dir"]["range"]
        self.yaw_offsets_range = config["yaw_angles"]["range"]
        self.ai_factor_range = config["ai_factors"]["range"]

        self.wind_dir_turb_std = config["wind_dir"]["turb_std"]
        
        self.wind_speed_var = config["wind_speed"]["var"]  # 0.5
        self.wind_dir_var = config["wind_dir"]["var"]  # 5.0
        self.yaw_angle_var = config["yaw_angles"]["var"]
        self.ai_factor_var = config["ai_factors"]["var"]
        
        self.wind_speed_turb_std = config["wind_speed"]["turb_std"]  # 0.5
        self.wind_dir_turb_std = config["wind_dir"]["turb_std"]  # 5.0
        self.yaw_angle_turb_std = config["yaw_angles"]["turb_std"]  # 0
        self.ai_factor_turb_std = config["ai_factors"]["turb_std"]  # 0
        noise_func_parts = config["wind_speed"]["noise_func"].split(".")
        func = globals()[noise_func_parts[0]]
        for i in range(1, len(noise_func_parts)):
            func = getattr(func, noise_func_parts[i])
        self.wind_speed_noise_func = func
        self.wind_speed_u_noise_args = config["wind_speed"]["u_noise_args"]
        self.wind_speed_v_noise_args = config["wind_speed"]["v_noise_args"]
        
        self.simulation_max_time = config["simulation_max_time"]
        
        self.yaw_rate = config["yaw_angles"]["roc"]

        self.simulation_max_time_steps = int(self.simulation_max_time // self.simulation_dt)

        # time interval between each horizon step in preview
        self.preview_dt = config["preview_dt"]

        # time steps in preview
        self.n_preview_steps = config["n_preview_steps"]
        self.n_samples_per_init_seed = config["n_samples_per_init_seed"]

        # # total time steps including simulation max time plus preview time
        # self.n_total_time_steps = self.simulation_max_time_steps + self.n_preview_steps
        self.distribution_params_path = config["distribution_params_path"]
        self.wind_preview_distribution_params = self._generate_wind_preview_distribution_params(regenerate_params=config["regenerate_distribution_params"])
    
    def _generate_online_bools_ts(self):
        return np.random.choice(
            [0, 1], size=(self.simulation_max_time_steps, self.num_turbines),
            p=[self.offline_probability, 1 - self.offline_probability])
    
    
    def _generate_wind_preview_distribution_params(self, regenerate_params=False):
        # TODO just compute this on the fly...
        if os.path.exists(self.distribution_params_path) and not regenerate_params:
            with open(self.distribution_params_path, "rb") as fp:
                wind_preview_distribution_params = pickle.load(fp)

        if not os.path.exists(self.distribution_params_path) or regenerate_params or len(wind_preview_distribution_params["mean_u"]) < (self.n_preview_steps + self.preview_dt):
            # compute mean, variance, covariance ahead of time
            mean_u = [self.wind_speed_u_range[0] + ((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) / 2)] * (self.n_preview_steps + self.preview_dt)
            mean_v = [self.wind_speed_v_range[0] + ((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) / 2)] * (self.n_preview_steps + self.preview_dt)

            # noise_args["mean"] = [(self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) / 2 for j in range(n_preview_steps + preview_dt)] \
            # 					+ [(self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) / 2 for j in range(n_preview_steps + preview_dt)]


            # variance of u[j], and v[j], grows over the course of the prediction horizon (by 1+j*0.05, or 5% for each step), and has an initial variance of 0.25 the value of the current measurement
            # prepend 0 variance values for the deterministic measurements at the current (0) time-step
            # o = 0.2
            # we want it to be very unlikely (3 * standard deviations) that the value will stray outside of the desired range
            
            # p = 0.1
            # var_u = np.array([(((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) * p) / 3)**2 * (1. + j*0.02) for j in range(0, n_preview_steps + preview_dt)])
            # var_v = np.array([(((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) * p) / 3)**2 * (1. + j*0.02) for j in range(0, n_preview_steps + preview_dt)])
            # QUESTION: we want growing uncertainty in prediction further along in the prediction horizon, not growing variation - should variance remain the same?
            p = 0.4 # 0.4
            q = 0.000
            # var_u = np.array([(((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) * p * (2. - np.exp(-q * j))) / 3)**2  for j in range(0, self.n_preview_steps + self.preview_dt)])
            # var_v = np.array([(((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) * p * (2. - np.exp(-q * j))) / 3)**2 for j in range(0, self.n_preview_steps + self.preview_dt)])

            var_u = np.array([(((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) * p) / 3)**2  for j in range(0, self.n_preview_steps + self.preview_dt)])
            var_v = np.array([(((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) * p) / 3)**2 for j in range(0, self.n_preview_steps + self.preview_dt)])

            # cov = np.diag(np.concatenate([var_u, var_v]))
            
            cov_u = np.diag(var_u)
            cov_v = np.diag(var_v)
            # covariance of u[j], v[j] 
            # is zero for u and v elements (no correlation),
            # positive and greater for adjacent elements u[j], u[j +/- 1], and v[j], v[j +/- 1] 
            # more off-diagonal covariance elements are farther apart in time over the prediction horizon (i.e. the row number is farther from the column number)
            # for the covariance matrix to be positive definite, off-diagonal elements should be less than diagonal elements, so p should be a fraction
            # requirement on the variance such that covariance does not become negative, (plus safety factor of 5%)
            # greater multiple leads to greater changes between time-steps
            # p = 1.0 1 = 0.005
            p = 1.0
            q = 0.001
            # a = 1.0 * (1 - (1 / (n_preview_steps)))**0.5
            for i in range(1, self.n_preview_steps + self.preview_dt):
            # for i in range(1, n_preview_steps):
                # off-diagonal elements
                # b = var_u[:n_preview_steps - i + preview_dt] * p
                # b = var_u[:n_preview_steps + preview_dt - i] * p
                b = np.array([var_u[0]] * (self.n_preview_steps + self.preview_dt - i)) * p
                # x = b * (a - ((i - 1) / (a * (n_preview_steps + preview_dt))))
                x = b * np.exp(-q * (i * self.simulation_dt)) 
                cov_u += np.diag(x, k=i)
                cov_u += np.diag(x, k=-i)

                # b = var_v[:n_preview_steps - i + preview_dt] * p
                # b = var_v[:n_preview_steps + preview_dt - i] * p
                b = np.array([var_v[0]] * (self.n_preview_steps + self.preview_dt - i)) * p
                x = b * np.exp(-q * (i * self.simulation_dt))
                # x = b * (a - ((i - 1) / (a * (n_preview_steps + preview_dt))))
                cov_v += np.diag(x, k=i)
                cov_v += np.diag(x, k=-i)

            wind_preview_distribution_params = {"mean_u": mean_u, "mean_v": mean_v, "cov_u": cov_u, "cov_v": cov_v}
            with open(self.distribution_params_path, "wb") as fp:
                pickle.dump(wind_preview_distribution_params, fp)
        
        if False:
            # visualize covariance matrix for testing purposes
            import seaborn as sns
            import matplotlib.pyplot as plt     

            fig, ax = plt.subplots(1, 2, figsize=(15,15)) 
            sns.heatmap(cov_u, annot=True, fmt='g', ax=ax[0], annot_kws={'size': 10})
            sns.heatmap(cov_v, annot=True, fmt='g', ax=ax[1], annot_kws={'size': 10})
            #annot=True to annotate cells, ftm='g' to disable scientific notation
            # annot_kws si size  of font in heatmap
            # labels, title and ticks
            ax[1].set_xlabel('Columns') 
            ax[0].set_ylabel('Rows')
            ax[0].set_title('Covariance Matrix: NN') 
            # ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 10)
            # ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 10)
            plt.show()
        
        short_wind_preview_distribution_params = {}
        for key in ["mean_u", "mean_v"]:
            # short_wind_preview_distribution_params[key] = np.array(wind_preview_distribution_params[key][:(self.n_preview_steps + self.preview_dt)])
            short_wind_preview_distribution_params[key] = np.array(wind_preview_distribution_params[key][:(self.n_preview_steps + self.preview_dt):self.time_series_dt])
            # del wind_preview_distribution_params[key]
            # gc.collect(wind_preview_distribution_params[key])
            
        for key in ["cov_u", "cov_v"]:
            # time_series_dt is the number of simulation_dt time steps in controller_dt
            # n_preview_steps is the number of simulation_d ttime steps in the horizon length 
            short_wind_preview_distribution_params[key] = np.array(wind_preview_distribution_params[key][:(self.n_preview_steps + self.preview_dt):self.time_series_dt, :(self.n_preview_steps + self.preview_dt):self.time_series_dt])
            # del wind_preview_distribution_params[key]
        
        del wind_preview_distribution_params
        gc.collect()

        return short_wind_preview_distribution_params
    
    def _sample_wind_preview(self, current_measurements, noise_func=np.random.multivariate_normal, noise_args=None, return_params=True, seed=None):
        """
        corr(X) = (diag(Kxx))^(-0.5)Kxx(diag(Kxx))^(-0.5)
        low variance and high covariance => high correlation
        """
        # mean_u = self.wind_preview_distribution_params["mean_u"][:self.n_preview_steps + self.preview_dt:self.time_series_dt]
        # mean_v = self.wind_preview_distribution_params["mean_v"][:self.n_preview_steps + self.preview_dt:self.time_series_dt]
        # cov_u = self.wind_preview_distribution_params["cov_u"][:self.n_preview_steps + self.preview_dt:self.time_series_dt, :self.n_preview_steps + self.preview_dt:self.time_series_dt]
        # cov_v = self.wind_preview_distribution_params["cov_v"][:self.n_preview_steps + self.preview_dt:self.time_series_dt, :self.n_preview_steps + self.preview_dt:self.time_series_dt]
        mean_u = self.wind_preview_distribution_params["mean_u"]
        mean_v = self.wind_preview_distribution_params["mean_v"]
        cov_u = self.wind_preview_distribution_params["cov_u"]
        cov_v = self.wind_preview_distribution_params["cov_v"]

        # assert len(mean_u) == self.n_preview_steps + self.preview_dt

        cond_mean_u = mean_u[1:] + cov_u[1:, :1] @ np.linalg.inv(cov_u[:1, :1]) @ (current_measurements[0] - mean_u[:1])
        cond_mean_v = mean_v[1:] + cov_v[1:, :1] @ np.linalg.inv(cov_v[:1, :1]) @ (current_measurements[1] - mean_v[:1])

        cond_cov_u = cov_u[1:, 1:] - cov_u[1:, :1] @ np.linalg.inv(cov_u[:1, :1]) @ cov_u[:1, 1:]
        cond_cov_v = cov_v[1:, 1:] - cov_v[1:, :1] @ np.linalg.inv(cov_v[:1, :1]) @ cov_v[:1, 1:]

        if return_params:
            return cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v

        noise_args = {}
        noise_args["mean"] = np.concatenate([cond_mean_u, cond_mean_v])
        noise_args["cov"] = scipy.linalg.block_diag(cond_cov_u, cond_cov_v)
        noise_args["size"] = self.n_samples_per_init_seed
        
        np.random.seed(seed)
        preview = noise_func(**noise_args)
        
        preview = np.hstack([np.broadcast_to(current_measurements[0], (self.n_samples_per_init_seed, 1)), preview[:, :cond_mean_u.shape[0]], 
                       np.broadcast_to(current_measurements[1], (self.n_samples_per_init_seed, 1)), preview[:, cond_mean_v.shape[0]:]])
        
        return preview
    
    def _generate_change_ts(self, val_range, val_var, change_prob, sample_time_step,
                            noise_func=None, noise_args=None, roc=None):
        # initialize at random wind speed
        init_val = np.random.choice(
            np.arange(val_range[0], val_range[1], val_var)
        )
        
        # randomly increase or decrease mean wind speed or keep static
        random_vals = np.random.choice(
            [-val_var, 0, val_var],
            size=(int(self.simulation_max_time // sample_time_step)),
            p=[change_prob / 2,
               1 - change_prob,
               change_prob / 2,
               ]
        )
        if roc is None:
            # if we assume instantaneous change (ie over a single DT)
            a = array('d', [
                y
                for x in random_vals
                for y in (x,) * sample_time_step])  # repeat random value x sample_time_step times
            delta_vals = array('d',
                               interp1d(np.arange(0, self.simulation_max_time, sample_time_step), random_vals,
                                        fill_value='extrapolate', kind='previous')(
                                   np.arange(0, self.simulation_max_time, self.simulation_dt)))
        
        else:
            # else we assume a linear change between now and next sample time, considering roc as max slope allowed
            for i in range(len(random_vals) - 1):
                diff = random_vals[i + 1] - random_vals[i]
                if abs(diff) > roc * sample_time_step:
                    random_vals[i + 1] = random_vals[i] + (diff / abs(diff)) \
                                         * (roc * sample_time_step)
            
            assert (np.abs(np.diff(random_vals)) <= roc * sample_time_step).all()
            delta_vals = array('d',
                               interp1d(np.arange(0, self.simulation_max_time, sample_time_step),
                                        random_vals, kind='linear', fill_value='extrapolate')(
                                   np.arange(0, self.simulation_max_time, self.simulation_dt)))
        
        if noise_func is None:
            noise_vals = np.zeros_like(delta_vals)
        else:
            noise_args["size"] = (int(self.simulation_max_time // self.wind_speed_sampling_time_step),)
            noise_vals = noise_func(**noise_args)
        
        # add mean and noise and bound the wind speed to given range
        ts = array('d', [init_val := np.clip(init_val + delta + n, *val_range)
                         for delta, n in zip(delta_vals, noise_vals)])
        
        return ts
    
    def _generate_stochastic_wind_speed_ts(self, init_seed=None, return_params=False):
        """
        n_preview_steps = n_horizon * controller_dt/simulation_dt = number of simulation steps in the prediction control horizon
        """
        # np.random.seed(init_seed)
        # initialize at random wind speed
        # init_val = [
        #     np.random.choice(np.arange(self.wind_speed_u_range[0], self.wind_speed_u_range[1], self.wind_speed_var)),
        #     np.random.choice(np.arange(self.wind_speed_v_range[0], self.wind_speed_v_range[1], self.wind_speed_var))
        # ]
        # TODO no longer using init_seed
        init_val = np.array([
            self.wind_speed_u_range[0] + ((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) / 2), 
            self.wind_speed_v_range[0] + ((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) / 2)])

        # n_time_steps = int(self.simulation_max_time // self.simulation_dt) + self.n_preview_steps
        generate_incrementally = False

        if return_params:
            cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v = self._sample_wind_preview(init_val, noise_func=np.random.multivariate_normal, noise_args=None, return_params=return_params)
            return cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v
        else:
            if generate_incrementally:
                
                full_u_ts = []
                full_v_ts = []
                for ts_subset_i in range(int(np.ceil(n_time_steps / self.n_preview_steps))):
                    i = 0
                    while 1:
                        wind_sample = self._sample_wind_preview(init_val, noise_func=np.random.multivariate_normal, noise_args=None, return_params=return_params)
                        # u_ts, v_ts = wind_sample[0, :n_preview_steps + preview_dt], wind_sample[0, n_preview_steps + preview_dt:]
                        u_ts, v_ts = wind_sample[0, :self.n_preview_steps + preview_dt], wind_sample[0, self.n_preview_steps + preview_dt:]
                        if (np.all(u_ts <= self.wind_speed_u_range[1]) and np.all(u_ts >= self.wind_speed_u_range[0]) 
                            and np.all(v_ts <= self.wind_speed_v_range[1]) and np.all(v_ts >= self.wind_speed_v_range[0])):
                            break
                        i += 1

                    full_u_ts.append(u_ts)
                    full_v_ts.append(v_ts)
                    init_val = [u_ts[-1], v_ts[-1]]

                full_u_ts = np.concatenate(full_u_ts)[:n_time_steps + self.preview_dt]
                full_v_ts = np.concatenate(full_v_ts)[:n_time_steps + self.preview_dt]
            else:
                wind_sample = self._sample_wind_preview(init_val, noise_func=np.random.multivariate_normal, noise_args=None, return_params=return_params)
                full_u_ts, full_v_ts = wind_sample[:, :self.n_preview_steps + self.preview_dt], wind_sample[:, self.n_preview_steps + self.preview_dt:]

        return full_u_ts, full_v_ts

    def _generate_freestream_wind_speed_u_ts(self):
        
        u_ts = self._generate_change_ts(self.wind_speed_u_range, self.wind_speed_var,
                                        self.wind_speed_change_probability,
                                        self.wind_speed_sampling_time_step,
                                        self.wind_speed_noise_func,
                                        self.wind_speed_u_noise_args)
        return u_ts
    
    def _generate_freestream_wind_speed_v_ts(self):

        v_ts = self._generate_change_ts(self.wind_speed_v_range, self.wind_speed_var,
                                    self.wind_speed_change_probability,
                                    self.wind_speed_sampling_time_step,
                                    self.wind_speed_noise_func,
                                    self.wind_speed_v_noise_args)
        return v_ts
    
    def _generate_freestream_wind_dir_ts(self):
        freestream_wind_dir_ts = self._generate_change_ts(self.wind_dir_range, self.wind_dir_var,
                                                          self.wind_dir_change_probability,
                                                          self.wind_dir_turb_std,
                                                          self.wind_dir_sampling_time_step)
        return freestream_wind_dir_ts

    def _generate_yaw_angle_ts(self):
        yaw_angle_ts = [self._generate_change_ts(self.yaw_offsets_range, self.yaw_angle_var,
                                                 self.yaw_angle_change_probability,
                                                 self.yaw_angle_turb_std,
                                                 self.yaw_angle_sampling_time_step,
                                                 roc=self.yaw_rate)
                        for i in range(self.num_turbines)]
        yaw_angle_ts = np.array(yaw_angle_ts).T
        
        return yaw_angle_ts
    
    def _generate_ai_factor_ts(self):
        online_bool_ts = self._generate_online_bools_ts()
        
        set_ai_factor_ts = [self._generate_change_ts(self.ai_factor_range, self.ai_factor_var,
                                                     self.ai_factor_change_probability,
                                                     self.ai_factor_var,
                                                     self.ai_factor_sampling_time_step)
                            for _ in range(self.num_turbines)]
        set_ai_factor_ts = np.array(set_ai_factor_ts).T
        
        effective_ai_factor_ts = np.array(
            [
                [set_ai_factor_ts[i][t] if online_bool_ts[i][t] else 0.0
                 for t in range(self.num_turbines)]
                for i in range(self.simulation_max_time_steps)
            ]
        )
        
        return effective_ai_factor_ts

def generate_wind_ts(wf, from_gaussian, case_idx, save_dir, save_name="", init_seed=None, return_params=False):
    # x = pd.read_csv(os.path.join(save_dir, f"{save_name}case_{case_idx}.csv"), index_col=0)
    # dir_tmp = np.arctan2(x["FreestreamWindSpeedV"], x["FreestreamWindSpeedU"])
    # dir_tmp = (270.0 - (dir_tmp * 180.0 / np.pi)) % 360.0
    # x["FreestreamWindDir"] = dir_tmp
    # (x.loc[x["FreestreamWindDir"] > 270.0, "FreestreamWindSpeedV"] < 0).all()
    # x.to_csv(os.path.join(save_dir, f"{save_name}case_{case_idx}.csv"))
    print(f'Simulating case #{case_idx}')
    # define freestream time series
    if from_gaussian:
        freestream_wind_speed_u, freestream_wind_speed_v = wf._generate_stochastic_wind_speed_ts(init_seed=init_seed, return_params=return_params)
    else:
        freestream_wind_speed_u = np.array(wf._generate_freestream_wind_speed_u_ts())
        freestream_wind_speed_v = np.array(wf._generate_freestream_wind_speed_v_ts())
    
    time = np.arange(freestream_wind_speed_u.shape[1]) * wf.simulation_dt
    time = pd.to_timedelta(time, unit="s") + pd.to_datetime("2025-01-01")
    # define noise preview
    # compute freestream wind direction angle from above, clockwise from north
    dir_preview = (180.0 + np.rad2deg(np.arctan2(freestream_wind_speed_u, freestream_wind_speed_v))) % 360.0
    mag_preview = np.linalg.norm(np.dstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=2)
    
    # save case raw_data as dataframe
    wind_field_data = {
        "time": np.tile(time, wf.n_samples_per_init_seed),
        "WindSeed": [init_seed] * len(time) * wf.n_samples_per_init_seed,
        "WindSample": np.repeat(np.arange(wf.n_samples_per_init_seed), len(time)),
        "FreestreamWindSpeedU": freestream_wind_speed_u.flatten(),
        "FreestreamWindSpeedV": freestream_wind_speed_v.flatten(),
        "FreestreamWindMag": mag_preview.flatten(),
        "FreestreamWindDir": dir_preview.flatten()
    }
    wind_field_df = pd.DataFrame(data=wind_field_data)
    
    # export case raw_data to csv
    wind_field_df.to_csv(os.path.join(save_dir, f"{save_name}case_{case_idx}.csv"))
    wf.df = wind_field_df
    return wf

def generate_wind_preview(wf, current_freestream_measurements, simulation_time_step, *, wind_preview_generator, return_params=False, include_uv=False, seed=None):
    
    # define noise preview
    # noise_func = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)

    # wind_preview_data = defaultdict(list)
    
    if return_params:
            cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v = wind_preview_generator(current_measurements=current_freestream_measurements, return_params=return_params, seed=seed)
            return (cond_mean_u, 
                    cond_mean_v,
                    cond_cov_u, 
                    cond_cov_v)
            # return cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v
    else:
        
        noise_preview = wind_preview_generator(current_measurements=current_freestream_measurements, return_params=return_params, seed=seed)
        
        u_preview = noise_preview[:, :(wf.n_preview_steps + wf.preview_dt) // wf.time_series_dt]
        v_preview = noise_preview[:, (wf.n_preview_steps + wf.preview_dt) // wf.time_series_dt:]
        mag_preview = np.linalg.norm(np.stack([u_preview, v_preview], axis=2), axis=2)
        
        # compute directions
        # compute freestream wind direction angle from above, clockwise from north
        dir_preview = (180.0 + np.rad2deg(np.arctan2(u_preview, v_preview))) % 360.0
        wind_preview_data = {"FreestreamWindMag": mag_preview, 
                             "FreestreamWindDir": dir_preview}
        if include_uv:
            wind_preview_data["FreestreamWindSpeedU"] = u_preview
            wind_preview_data["FreestreamWindSpeedV"] = v_preview
        
        # for j in range(int((wf.n_preview_steps + wf.preview_dt) // wf.time_series_dt)):
        #     wind_preview_data[f"FreestreamWindSpeedU_{j}"] += list(u_preview[:, j])
        #     wind_preview_data[f"FreestreamWindSpeedV_{j}"] += list(v_preview[:, j])
        #     wind_preview_data[f"FreestreamWindMag_{j}"] += list(mag_preview[:, j])
        #     wind_preview_data[f"FreestreamWindDir_{j}"] += list(dir_preview[:, j])
        
        return wind_preview_data

def generate_wind_preview_ts(config, case_idx, wind_field_data):
    wf = WindField(**config)
    print(f'Generating noise preview for case #{case_idx}')
    
    time = np.arange(0, wf.simulation_max_time, wf.simulation_dt)
    mean_freestream_wind_speed_u = wind_field_data[case_idx]['FreestreamWindSpeedU'].to_numpy()
    mean_freestream_wind_speed_v = wind_field_data[case_idx]['FreestreamWindSpeedV'].to_numpy()
    
    # save case raw_data as dataframe
    wind_preview_data = defaultdict(list)
    wind_preview_data["time"] = time
    
    for u, v in zip(mean_freestream_wind_speed_u, mean_freestream_wind_speed_v):
        noise_preview = wf._sample_wind_preview(current_measurements=[u, v], 
                                          noise_func=np.random.multivariate_normal, noise_args=None)
        u_preview = noise_preview[0, :config["n_preview_steps"] + 1].squeeze()
        v_preview = noise_preview[0, config["n_preview_steps"] + 1:].squeeze()
        mag = np.linalg.norm(np.vstack([u_preview, v_preview]), axis=0)
        
        direction = (180.0 + np.rad2deg(np.arctan2(u_preview, v_preview))) % 360.0
        
        for i in range(config["n_preview_steps"]):
            wind_preview_data[f"FreestreamWindSpeedU_{i}"].append(u_preview[i])
            wind_preview_data[f"FreestreamWindSpeedV_{i}"].append(v_preview[i])
            wind_preview_data[f"FreestreamWindMag_{i}"].append(mag[i])
            wind_preview_data[f"FreestreamWindDir_{i}"].append(direction[i])
    
    wind_preview_df = pd.DataFrame(data=wind_preview_data)
    
    # export case raw_data to csv
    wind_preview_df.to_csv(os.path.join(wf.data_save_dir, f'preview_case_{case_idx}.csv'))
    wf.df = wind_preview_df
    return wf



def generate_multi_wind_ts(wf, save_dir, save_name="", init_seeds=None, return_params=False, parallel=True):

    if parallel:		
        with ProcessPoolExecutor() as generate_wind_fields:
            futures = [generate_wind_fields.submit(generate_wind_ts, 
                                              wf=wf, from_gaussian=True, save_dir=save_dir, save_name=save_name, return_params=return_params, 
                                              case_idx=case_idx, init_seed=init_seeds[case_idx]) 
                       for case_idx in range(len(init_seeds))]
        wait(futures)
        wind_field_data = [fut.result() for fut in futures]
    else:
        wind_field_data = []
        for i, seed in enumerate(init_seeds):
            wind_field_data.append(generate_wind_ts(wf=wf, from_gaussian=True, case_idx=i, save_dir=save_dir, save_name=save_name, init_seed=seed, return_params=return_params))
        # plot_ts(wind_field_data[0].df, config["fig_dir"])
        
    return wind_field_data

def write_abl_velocity_timetable(dfs, save_path, boundary_starttime=7200.0):
    for d, df in enumerate(dfs):
        df = df[["time", "FreestreamWindMag", "FreestreamWindDir"]]
        df["FreestreamWindDir"] = (270.0 - df["FreestreamWindDir"]) % 360.0
        # df.loc[df["FreestreamWindDir"] > 180.0, "FreestreamWindDir"] = df.loc[df["FreestreamWindDir"] > 180.0, "FreestreamWindDir"] - 360.0
        # df.loc[df["FreestreamWindDir"] < 0.0, "FreestreamWindDir"] = df.loc[df["FreestreamWindDir"] < 0.0, "FreestreamWindDir"] + 360.0
        # dt = df["time"].iloc[1] - df["time"].iloc[0]
        # init_time = np.arange(0, boundary_starttime, dt)
        # mean_df = pd.DataFrame({"time": init_time,
        #                         "FreestreamWindMag": [df["FreestreamWindMag"].iloc[0]] * len(init_time),
        #                         "FreestreamWindDir": [df["FreestreamWindDir"].iloc[0]] * len(init_time)})
        df["time"] = df["time"] + boundary_starttime
        # pd.concat([mean_df, df])
        
        df.to_csv(os.path.join(save_path, f"abl_velocity_timetable_{d}.txt"), index=False, sep=" ")

def get_amr_timeseries(case_folders=None, abl_stats_files=None):
    from moa_python.post_abl_stats import Post_abl_stats
    if case_folders is None:
        case_folders = ['/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples']#,'/lustre/eaglefs/projects/ssc/jfrederi/amr-wind-runs/precursor-new/baseline_8ms_dt002_dx2p5']#,'/projects/ssc/jfrederi/precursors-amr-wind/neutral_highti_8ms/precursor','/projects/ssc/jfrederi/precursors-amr-wind/neutral_highti_rthedin']#,'/projects/ssc/jfrederi/precursors-amr-wind/neutral_lowti_8ms/precursor','/projects/ssc/jfrederi/precursors-amr-wind/neutral_lowti_9ms/precursor','/projects/ssc/jfrederi/precursors-amr-wind/neutral_lowti_10ms/precursor','/projects/ssc/jfrederi/precursors-amr-wind/neutral_lowti_12ms','/projects/ssc/jfrederi/precursors-amr-wind/neutral_lowti_12ms']

    if abl_stats_files is None:
        abl_stats_files = ['post_processing/abl_statistics00000.nc']#,'post_processing/abl_statistics14400.nc','abl_statistics129531.nc']

    abl = []
    
    full_filename_abl_stats = []
    for n in range(len(case_folders)):
        full_filename_abl_stats = os.path.join(case_folders[n],abl_stats_files[n])
        abl.append(Post_abl_stats(full_filename_abl_stats))

    # Ncases = len(case_folders)

    settling_time = 7200
    height = 90

    # print(abl[0].time[abl[0].time>=settling_time])
    settled_time_idx = abl[0].time >= settling_time
    settled_time = abl[0].time[settled_time_idx]
    settled_u = abl[0].get_time_series_at_height('u', height)[settled_time_idx]
    settled_v = abl[0].get_time_series_at_height('v', height)[settled_time_idx]
    return settled_time, settled_u, settled_v

def fit_amr_distribution(distribution_params_path, case_folders=None, abl_stats_files=None):

    settled_time, settled_u, settled_v = get_amr_timeseries(case_folders, abl_stats_files)

    u_preview_samples = []
    v_preview_samples = []

    n_horizon = 12
    controller_dt = 60
    simulation_dt = 0.5
    n_preview_steps = int(n_horizon * controller_dt // simulation_dt)
    preview_dt = int(controller_dt // simulation_dt)
    time_series_dt = 1
    for k in range(int(len(settled_time) - (n_preview_steps + preview_dt))):
        # print(k, len(settled_u[k:k+n_preview_steps + preview_dt:time_series_dt]))
        u_preview_samples.append(settled_u[k:k+n_preview_steps + preview_dt:time_series_dt])
        v_preview_samples.append(settled_v[k:k+n_preview_steps + preview_dt:time_series_dt])

    u_preview_samples = np.vstack(u_preview_samples)
    v_preview_samples = np.vstack(v_preview_samples)

    mean_u = np.mean(settled_u)
    cov_u = np.cov(u_preview_samples, rowvar=False)
    mean_v = np.mean(settled_v)
    cov_v = np.cov(v_preview_samples, rowvar=False)

    wind_preview_distribution_params = {"mean_u": mean_u, "mean_v": mean_v, "cov_u": cov_u, "cov_v": cov_v}
    with open(distribution_params_path, "wb") as fp:
        pickle.dump(wind_preview_distribution_params, fp)

    # print(int(n_preview_steps + preview_dt))
    # print(preview_dt)
    # print(settled_time, settled_time.shape)

    fig, ax = plt.subplots(1, 2) 
    sns.heatmap(cov_u, annot=True, fmt='g', ax=ax[0], annot_kws={'size': 10})
    sns.heatmap(cov_v, annot=True, fmt='g', ax=ax[1], annot_kws={'size': 10})
    #annot=True to annotate cells, ftm='g' to disable scientific notation
    # annot_kws si size  of font in heatmap
    # labels, title and ticks
    ax[0].set_title('Covariance Matrix for Wind Speed U') 
    ax[1].set_title('Covariance Matrix for Wind Speed V') 
    ax[0].set_xticks(np.arange(0, int(n_preview_steps + preview_dt), preview_dt))
    ax[1].set_xticks(np.arange(0, int(n_preview_steps + preview_dt), preview_dt))
    ax[0].set_yticks(np.arange(0, int(n_preview_steps + preview_dt), preview_dt))
    ax[1].set_yticks(np.arange(0, int(n_preview_steps + preview_dt), preview_dt))
    ax[0].set_xticklabels((ax[0].get_xticks() * simulation_dt).astype(int))
    ax[1].set_xticklabels((ax[1].get_xticks() * simulation_dt).astype(int))
    ax[0].set_yticklabels((ax[0].get_yticks() * simulation_dt).astype(int))
    ax[1].set_yticklabels((ax[1].get_yticks() * simulation_dt).astype(int))
    fig.tight_layout()
    # ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 10)
    # ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 10)

    fig, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].plot(settled_time, settled_u, label="True")
    axarr[0].plot(settled_time, [mean_u] * len(settled_time), linestyle=":", label="Mean")
    axarr[0].set(title="Wind Speed U [m/s]")
    axarr[0].legend()
    axarr[1].plot(settled_time, settled_v, label="v")
    axarr[1].plot(settled_time, [mean_v] * len(settled_time), linestyle=":", label="mean_v")
    axarr[1].set(title="Wind Speed V [m/s]", xlabel="Time [s]")
    axarr[1].set_xticks(settled_time[::n_preview_steps])
    #axarr[1].set_xticklabels((axarr[1].get_xticks() * simulation_dt).astype(int))


def generate_multi_wind_preview_ts(config, wind_field_data):
    if config["n_wind_field_cases"] == 1:
        wind_field_preview_data = []
        for i in range(config["n_wind_field_cases"]):
            wind_field_preview_data.append(generate_wind_preview_ts(config, i, wind_field_data))
        plot_distribution_ts(wind_field_preview_data[0])
        return wind_field_preview_data
    
    else:
        with ProcessPoolExecutor() as generate_wind_fields:
            futures = [generate_wind_fields.submit(generate_wind_preview_ts, 
                                              config=config, 
                                              wind_field_data=wind_field_data,
                                              case_idx=case_idx) 
                       for case_idx in range(config["n_wind_field_cases"])]
        wait(futures)
        wind_field_data = [fut.result() for fut in futures]

def first_ord_filter(x, alpha=np.exp(-(1 / 35) * 0.5)):
    b = [1 - alpha]
    a = [1, -alpha]
    return lfilter(b, a, x)

if __name__ == '__main__':
    # with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml")) as fp:
    # 	wind_field_config = yaml.safe_load(fp)
    # wind_field_data = generate_multi_wind_ts(wind_field_config)
    # generate_multi_wind_preview_ts(wind_field_config, wind_field_data)
    from hercules.utilities import load_yaml
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    regenerate_wind_field = False
    
    with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    input_dict = load_yaml(os.path.join(os.path.dirname(whoc.__file__), "../examples/hercules_input_001.yaml"))
    input_dict["controller"]["n_wind_preview_samples"] = 100
    input_dict["controller"]["controller_dt"] = 15
    input_dict["controller"]["n_horizon"] = 24
    
    wind_field_config["simulation_max_time"] = 3600
    wind_field_config["simulation_sampling_time"] = input_dict["simulation_dt"]
    wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / input_dict["simulation_dt"]) + input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["controller_dt"] / input_dict["simulation_dt"])
    wind_field_config["preview_dt"] = int(input_dict["controller"]["controller_dt"] / input_dict["simulation_dt"])
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["distribution_params_path"] = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "floris_case_studies", "wind_field_data", "wind_preview_distribution_params.pkl")  

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "floris_case_studies", "wind_field_data", "raw_data")        
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    n_wind_field_cases = 1
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    seed = 0
    if not len(wind_field_filenames) or regenerate_wind_field:
        # generate_multi_wind_ts(wind_field_config, save_name="short_", seed=seed)
        wind_field_config["regenerate_distribution_params"] = True
        # wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / input_dict["simulation_dt"]) # TODO remove
        wind_field_config["n_samples_per_init_seed"] = 1
        wind_field_config["time_series_dt"] = 1
        full_wf = WindField(**wind_field_config)
        generate_multi_wind_ts(full_wf, wind_field_dir, save_name="", init_seeds=[seed])
        wind_field_filenames = [f"case_{i}.csv" for i in range(n_wind_field_cases)]
        regenerate_wind_field = True

    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(os.path.join(wind_field_dir, fn), index_col=0, parse_dates=["time"]))
    
    plot_ts(pd.concat(wind_field_data), wind_field_dir)
    # plt.savefig(os.path.join(wind_field_config["fig_dir"], "wind_field_ts.png"))
    # true wind disturbance time-series
    case_idx = 0
    wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
    wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
    wind_u_ts = wind_field_data[case_idx]["FreestreamWindSpeedU"].to_numpy()
    wind_v_ts = wind_field_data[case_idx]["FreestreamWindSpeedV"].to_numpy()

    wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["controller_dt"] / input_dict["simulation_dt"])
    wind_field_config["time_series_dt"] = int(input_dict["controller"]["controller_dt"] / input_dict["simulation_dt"])
    wind_field_config["n_samples_per_init_seed"] = input_dict["controller"]["n_wind_preview_samples"]
    wind_field_config["regenerate_distribution_params"] = False
    
    # if False:
    preview_wf = WindField(**wind_field_config)
    stochastic_wind_preview_func = partial(generate_wind_preview, wf=preview_wf, wind_preview_generator=preview_wf._sample_wind_preview)
    # else:
    #     wind_field_config["time_series_dt"] = 1
    #     preview_wf = WindField(**wind_field_config)
    #     stochastic_wind_preview_func = partial(generate_wind_preview, wf=preview_wf, wind_preview_generator=preview_wf._sample_wind_preview)
            
    def persistent_wind_preview_func(current_freestream_measurements, time_step):
        wind_preview_data = defaultdict(list)
        for j in range(input_dict["controller"]["n_horizon"] + 1):
            wind_preview_data[f"FreestreamWindMag_{j}"] += [wind_mag_ts[time_step]]
            wind_preview_data[f"FreestreamWindDir_{j}"] += [wind_dir_ts[time_step]]
            wind_preview_data[f"FreestreamWindSpeedU_{j}"] += [wind_u_ts[time_step]]
            wind_preview_data[f"FreestreamWindSpeedV_{j}"] += [wind_v_ts[time_step]]
        return wind_preview_data
    
    def perfect_wind_preview_func(current_freestream_measurements, time_step):
        wind_preview_data = defaultdict(list)
        for j in range(input_dict["controller"]["n_horizon"] + 1):
            delta_k = j * int(input_dict["controller"]["controller_dt"] // input_dict["simulation_dt"])
            wind_preview_data[f"FreestreamWindMag_{j}"] += [wind_mag_ts[time_step + delta_k]]
            wind_preview_data[f"FreestreamWindDir_{j}"] += [wind_dir_ts[time_step + delta_k]]
            wind_preview_data[f"FreestreamWindSpeedU_{j}"] += [wind_u_ts[time_step + delta_k]]
            wind_preview_data[f"FreestreamWindSpeedV_{j}"] += [wind_v_ts[time_step + delta_k]]
        return wind_preview_data
    
    idx = 0
    current_freestream_measurements = [
        wind_mag_ts[idx] * np.sin(np.deg2rad(wind_dir_ts[idx] + 180.)),
        wind_mag_ts[idx] * np.cos(np.deg2rad(wind_dir_ts[idx] + 180.))
    ]

    n_time_steps = (input_dict["controller"]["n_horizon"] + 1) * int(input_dict["controller"]["controller_dt"] // input_dict["simulation_dt"])
    preview_dt = int(input_dict["controller"]["controller_dt"] // input_dict["simulation_dt"])

    tmp = perfect_wind_preview_func(current_freestream_measurements, idx)
    perfect_preview = {}
    perfect_preview["WindSeed"] = [1] * 2 * n_time_steps
    perfect_preview["Wind Speed"] \
        = np.concatenate([tmp[f"FreestreamWindSpeedU_{j}"] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for j in range(input_dict["controller"]["n_horizon"] + 1)] \
        + [tmp[f"FreestreamWindSpeedV_{j}"] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                   for j in range(input_dict["controller"]["n_horizon"] + 1)])
    perfect_preview["Wind Component"] = ["U"] * n_time_steps + ["V"] * n_time_steps

    tmp = persistent_wind_preview_func(current_freestream_measurements, idx)
    persistent_preview = {}
    persistent_preview["WindSeed"] = [1] * 2 * n_time_steps
    persistent_preview["Wind Speed"] \
        = np.concatenate([tmp[f"FreestreamWindSpeedU_{j}"] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for j in range(input_dict["controller"]["n_horizon"] + 1)] \
        + [tmp[f"FreestreamWindSpeedV_{j}"] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                   for j in range(input_dict["controller"]["n_horizon"] + 1)])
    persistent_preview["Wind Component"] = ["U"] * n_time_steps + ["V"] * n_time_steps

    
    tmp = stochastic_wind_preview_func(current_freestream_measurements=current_freestream_measurements, simulation_time_step=idx, include_uv=True)
    stochastic_preview = {}
    stochastic_preview["WindSeed"] = np.repeat(np.arange(input_dict["controller"]["n_wind_preview_samples"]) + 1, (2 * (n_time_steps),))
    # stochastic_preview["FreestreamWindSpeedU"] = [tmp[f"FreestreamWindSpeedU_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(input_dict["controller"]["n_horizon"] + 1)]
    # stochastic_preview["FreestreamWindSpeedV"] = [tmp[f"FreestreamWindSpeedV_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"])for j in range(input_dict["controller"]["n_horizon"] + 1)]
    # stochastic_preview["Wind Speed"] = [tmp[f"FreestreamWindSpeedU_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(n_time_steps)] \
    # 	+ [tmp[f"FreestreamWindSpeedV_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(n_time_steps)]
    stochastic_preview["Wind Speed"] \
        = np.concatenate([np.concatenate([[tmp[f"FreestreamWindSpeedU"][m, j]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                   for j in range(input_dict["controller"]["n_horizon"] + 1)] \
        + [[tmp[f"FreestreamWindSpeedV"][m, j]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                     for j in range(input_dict["controller"]["n_horizon"] + 1)]) for m in range(input_dict["controller"]["n_wind_preview_samples"])])
    
    stochastic_preview["Wind Component"] = np.concatenate([["U"] * n_time_steps + ["V"] * n_time_steps for m in range(input_dict["controller"]["n_wind_preview_samples"])])
    # stochastic_preview = pd.DataFrame(stochastic_preview)

    perfect_preview["time"] = persistent_preview["time"] = np.tile(np.arange(n_time_steps), (2, )) * input_dict["simulation_dt"]
    
    stochastic_preview["time"] = np.tile(np.arange(n_time_steps) * input_dict["simulation_dt"], (2 * input_dict["controller"]["n_wind_preview_samples"],))
    
    perfect_preview = pd.DataFrame(perfect_preview)
    perfect_preview["Data Type"] = ["Preview"] * len(perfect_preview.index)
    tmp = pd.DataFrame(perfect_preview)
    tmp["Data Type"] = ["True"] * len(tmp.index)
    # tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
    tmp["Wind Speed"] \
        = np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)] \
        + [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)])
    perfect_preview = pd.concat([perfect_preview, tmp])

    persistent_preview = pd.DataFrame(persistent_preview)
    persistent_preview["Data Type"] = ["Preview"] * len(persistent_preview.index)
    tmp = pd.DataFrame(persistent_preview)
    tmp["Data Type"] = ["True"] * len(tmp.index)
    # tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
    tmp["Wind Speed"] \
        = np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)] \
        + [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)])
    persistent_preview = pd.concat([persistent_preview, tmp])

    stochastic_preview = pd.DataFrame(stochastic_preview)
    stochastic_preview["Data Type"] = ["Preview"] * len(stochastic_preview.index)
    tmp = pd.DataFrame(stochastic_preview.loc[stochastic_preview["WindSeed"] == 1])
    tmp["Data Type"] = ["True"] * len(tmp.index)
    # tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
    tmp["Wind Speed"] \
        = np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)] \
        + [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["controller_dt"] - input_dict["simulation_dt"]) // input_dict["simulation_dt"])) 
                                  for k in range(0, n_time_steps, preview_dt)])
    
    stochastic_preview = pd.concat([stochastic_preview, tmp])

    perfect_preview.reset_index(inplace=True, drop=True)
    persistent_preview.reset_index(inplace=True, drop=True)
    stochastic_preview.reset_index(inplace=True, drop=True)
    
    # stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Wind Component"] == "U"), "Wind Speed"].mean()
    # stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Sample"] == 1), "time"]
    
    assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy() == persistent_preview.loc[persistent_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy())
    assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy() == stochastic_preview.loc[stochastic_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy())
    assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "time"].dropna().to_numpy() == stochastic_preview.loc[stochastic_preview["Data Type"] == "True", "time"].dropna().to_numpy())

    # perfect_preview["TrueWindSpeed"] = persistent_preview["TrueWindSpeed"] \
    # 	= [wind_u_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)] + [wind_v_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)]
    # stochastic_preview["TrueWindSpeed"] \
    # 	= np.tile([wind_u_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)] + [wind_v_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)], (input_dict["controller"]["n_wind_preview_samples"], ))

    # different hues for u vs k, different style for true vs preview
    # TODO
    fig = plt.figure(figsize=(7.7, 5.86))
    ax = sns.lineplot(data=perfect_preview.loc[perfect_preview["Data Type"] == "True", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
    ax = sns.lineplot(data=perfect_preview.loc[perfect_preview["Data Type"] == "Preview", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o")
    ax.set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(0, int(wind_field_config["n_preview_steps"] * input_dict["simulation_dt"])))
    ax.set_xticks(np.arange(0, int(n_time_steps * input_dict["simulation_dt"]), int(60)))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:5] + h[9:], l[:5] + l[9:])
    fig.savefig(os.path.join(wind_field_dir, f'perfect_preview.png'))
    
    # plt.legend(labels=["Preview, U", "Preview, V", "True, U", "True, V"])
    
    fig = plt.figure(figsize=(7.7, 5.86))
    ax = sns.lineplot(data=persistent_preview.loc[persistent_preview["Data Type"] == "True", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
    ax = sns.lineplot(data=persistent_preview.loc[persistent_preview["Data Type"] == "Preview", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o")
    ax.set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(0, int(wind_field_config["n_preview_steps"] * input_dict["simulation_dt"])))
    ax.set_xticks(np.arange(0, int(n_time_steps * input_dict["simulation_dt"]), int(60)))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:5] + h[9:], l[:5] + l[9:])
    fig.savefig(os.path.join(wind_field_dir, f'persistent_preview.png'))
    # sns.scatterplot(data=persistent_preview.loc[perfect_preview["Data Type"] == "Preview", :], x="time", y="Wind Speed", zorder=7)
    # plt.legend(labels=["Preview, U", "Preview, V", "True, U", "True, V"])

    # stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Wind Component"] == "U"), "Wind Speed"].dropna()
    fig = plt.figure(figsize=(7.7, 5.86))
    # sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Sample"] == 1, :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4], [1, 0]])
    ax = sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Data Type"] == "True", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
    # ax = sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Data Type"] == "Preview", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o", errorbar= lambda x: (x.min(), x.max()))
    
    ax = sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Data Type"] == "Preview", :], x="time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o", errorbar=("sd", 2))
    ax.lines[-6].remove()
    ax.lines[-6].remove()
    ax.get_children()[9].set_label("U Preview")
    ax.get_children()[10].set_label("V Preview")
    ax.set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(0, int(wind_field_config["n_preview_steps"] * input_dict["simulation_dt"])))
    ax.set_xticks(np.arange(0, int(n_time_steps * input_dict["simulation_dt"]), int(60)))
    h, l = ax.get_legend_handles_labels()
    desired_labels = ["Wind Component", "U", "V", "Data Type", "True", "U Preview", "V Preview"]
    ax.legend(h[:7], l[:7])
    fig.savefig(os.path.join(wind_field_dir, f'stochastic_preview.png'))