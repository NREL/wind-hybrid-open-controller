import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp, ttest_ind
from pingouin import multivariate_ttest
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import yaml
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as cf_wait

import whoc
from whoc.case_studies.initialize_case_studies import STORAGE_DIR
from whoc.wind_field.WindField import WindField, generate_wind_preview
from whoc.wind_field.WindField import generate_multi_wind_ts

from hercules.utilities import load_yaml


# for each time-step in freestream time-series, for each seed generate the preview
def generate_preview(wf, input_dict, wind_field_data, time_step, seed):
    print(f"generating preview for seed {seed} time-step {time_step}")
    current_freestream_measurements = wind_field_data[seed].loc[(wind_field_data[seed]["Time"] // input_dict["dt"] == time_step) & (wind_field_data[seed]["WindSeed"] == 0), ["FreestreamWindSpeedU", "FreestreamWindSpeedV"]].to_numpy()[0, :]
    preview_dt = int(input_dict["controller"]["dt"] // input_dict["dt"])
    n_preview_steps = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] // input_dict["dt"])
    preview = generate_wind_preview(wf, current_freestream_measurements, time_step,
                    wind_preview_generator=wf._sample_wind_preview, return_params=False)
    
    # preview_params = generate_wind_preview( 
    #                     current_freestream_measurements, k,
    #                     wind_preview_generator=wf._sample_wind_preview, 
    #                     n_preview_steps=input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] // input_dict["dt"]),
    #                     preview_dt=int(input_dict["controller"]["dt"] // input_dict["dt"]),
    #                     n_samples=input_dict["controller"]["n_wind_preview_samples"],
    #                     return_params=True)
    freestream_samples = wind_field_data[seed].loc[(wind_field_data[seed]["Time"] // input_dict["dt"]).isin(np.arange(time_step, time_step + n_preview_steps + preview_dt, wf.time_series_dt)), ["Time", "WindSample", "FreestreamWindSpeedU", "FreestreamWindSpeedV"]]
    freestream_samples = np.hstack([
        freestream_samples[["Time", "WindSample", "FreestreamWindSpeedU"]].pivot(index="WindSample", columns="Time", values="FreestreamWindSpeedU").to_numpy(),
        freestream_samples[["Time", "WindSample", "FreestreamWindSpeedV"]].pivot(index="WindSample", columns="Time", values="FreestreamWindSpeedV").to_numpy()
        ])
    # .to_numpy().T.flatten()
    # preview_samples = np.vstack([np.vstack([[preview[f"FreestreamWindSpeedU_{k}"][m], preview[f"FreestreamWindSpeedV_{k}"][m]] for k in range(input_dict["controller"]["n_horizon"] + 1)]).T.flatten() for m in range(wf.n_samples_per_init_seed)])
    preview_samples = np.vstack([np.vstack([[preview[f"FreestreamWindSpeedU_{k}"][m], preview[f"FreestreamWindSpeedV_{k}"][m]] for k in range(int((n_preview_steps + preview_dt) // wf.time_series_dt))]).T.flatten() for m in range(wf.n_samples_per_init_seed)])
    return freestream_samples, preview_samples

if __name__ == "__main__":
    if 0:
        # generate probabilities for known inputs from univariate gaussian pdf
        mu = 8
        sigma = 4.0
        
        lb_angle = 60 * np.pi / 180.0
        ub_angle = 120 * np.pi / 180.0
        # lb_angle = -45 * np.pi / 180.0
        # ub_angle = 45 * np.pi / 180.0

        lb_tan = min(np.tan(lb_angle), np.tan(ub_angle))
        ub_tan = max(np.tan(lb_angle), np.tan(ub_angle))

        err = []
        sample_sizes = np.linspace(100, 10000, 50)
        for n_samples in sample_sizes:
            ratio_range = np.linspace(-20, 20, int(n_samples))
            dratio = ratio_range[1] - ratio_range[0]

            # pdf_ratio = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
            # probs_ratio = pdf_ratio(ratio_range)
            probs_ratio = norm.pdf(ratio_range, loc=mu, scale=sigma)
            
            # compute nonlinear function of inputs
            angle_range = np.arctan(ratio_range)
            probs_angle = probs_ratio[np.argsort(angle_range)]
            angle_range = np.sort(angle_range)
            dangle = np.diff(angle_range)

            # compute cdf for angular bounds and for tranfsormed bounds
            angle_shift = -(np.pi/2) - lb_angle
            lb_angle += angle_shift
            ub_angle += angle_shift
            # angle_range += angle_shift
            dangle = np.diff(angle_range)


            lb_angle_ratio = np.argmin(abs(angle_range - lb_angle))
            ub_angle_ratio = np.argmin(abs(angle_range - ub_angle))
            cdf_angle = sum(probs_angle[lb_angle_ratio:ub_angle_ratio+1] * dangle[lb_angle_ratio:ub_angle_ratio+1])
            
            lb_tan_ratio = np.argmin(abs(ratio_range - lb_tan))
            ub_tan_ratio = np.argmin(abs(ratio_range - ub_tan))
            cdf_tan = sum(probs_ratio[lb_tan_ratio:ub_tan_ratio+1] * dratio)
            # cdf_tan_true = norm.cdf(ub_tan, loc=mu, scale=sigma) - norm.cdf(lb_tan, loc=mu, scale=sigma)
            err.append(abs(cdf_angle - cdf_tan))


        plt.plot(sample_sizes, err)
        plt.show()
        # print(cdf_angle)
        # print(cdf_tan)
        # print(cdf_tan_true)

    if 1:
        # generate 6 seeds of freestream wind time series
        n_seeds = 1
        regenerate_wind_field = False
        input_dict = load_yaml(os.path.join(os.path.dirname(whoc.__file__), "../examples/hercules_input_001.yaml"))

        with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
            wind_field_config = yaml.safe_load(fp)

        # instantiate wind field if files don't already exist
        wind_field_dir = os.path.join(STORAGE_DIR, "wind_field_data", "raw_data")        
        wind_field_filenames = glob(os.path.join(wind_field_dir, "debug_case_*.csv"))
        distribution_params_path = os.path.join(STORAGE_DIR, "wind_field_data", "wind_preview_distribution_params.pkl")    
        
        if not os.path.exists(wind_field_dir):
            os.makedirs(wind_field_dir)

        seed = 0

        input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = 300
        wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
        wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
        wind_field_config["preview_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
        wind_field_config["simulation_sampling_time"] = input_dict["dt"]
        wind_field_config["n_preview_steps"] =  int(input_dict["hercules_comms"]["helics"]["config"]["stoptime"] / input_dict["dt"]) + input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
        wind_field_config["n_samples_per_init_seed"] = 5
        wind_field_config["regenerate_distribution_params"] = False
        wind_field_config["distribution_params_path"] = os.path.join(STORAGE_DIR, "wind_field_data", "wind_preview_distribution_params.pkl")  
        wind_field_config["time_series_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])

        if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
            # generate_multi_wind_ts(wf, wind_field_config, seeds=[seed + i for i in range(n_seeds)], save_name="short_")
            wind_field_config["regenerate_distribution_params"] = True
            full_wf = WindField(**wind_field_config)
            generate_multi_wind_ts(full_wf, wind_field_dir, init_seeds=range(n_seeds), save_name="debug_")
            wind_field_filenames = [os.path.join(wind_field_dir, f"debug_case_{i}.csv") for i in range(n_seeds)]
            regenerate_wind_field = True
        
        wind_field_data = []
        if os.path.exists(wind_field_dir):
            for fn in wind_field_filenames:
                wind_field_data.append(pd.read_csv(fn, index_col=0))

        # set significance level - probability of rejecting null hypotheisis (no significant different between distributions) when it is true
        alpha = 0.05
        wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
        # wind_field_config["n_samples"] = input_dict["controller"]["n_wind_preview_samples"]
        wind_field_config["n_samples_per_init_seed"] = 500
        wind_field_config["regenerate_distribution_params"] = False
        wind_field_config["time_series_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
        preview_wf = WindField(**wind_field_config)
        with ProcessPoolExecutor() as generate_previews_exec:
            # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
            futures = [generate_previews_exec.submit(generate_preview, wf=preview_wf, input_dict=input_dict, wind_field_data=wind_field_data, time_step=k, seed=s) 
                # for k in range(0, int(wind_field_config["simulation_max_time"] // wind_field_config["simulation_sampling_time"]), int(input_dict["controller"]["dt"] / input_dict["dt"])) for s in range(n_seeds)]
                for k in range(1) for s in range(n_seeds)]
        cf_wait(futures)
        results = [fut.result() for fut in futures]

        # np.cov(preview_samples, rowvar=False).shape

        ttest_stats = []
        for res in results:
            ttest_stats.append(multivariate_ttest(res[0], res[1], paired=False))

        ttest_stats = pd.concat(ttest_stats)
        
        freestream_samples = np.vstack([res[0] for res in results])
        preview_samples = np.vstack([res[1] for res in results])
        np.save(os.path.join(STORAGE_DIR, "wind_field_data", "freestream_samples.npy"), freestream_samples)
        np.save(os.path.join(STORAGE_DIR, "wind_field_data", "preview_samples.npy"), preview_samples)

        # ttest_stats = multivariate_ttest(freestream_samples, preview_samples, paired=False)

        # compare the windows of data from the freestream time-series to the preview in a statistical t-test
        # p-value = probability of obtaining a t-stat as extreme or more exptreme than the observed value, assuming that the null hypothesis is true
        # t_stat, p_value = ttest_1samp()

        # Interpret the results
        if np.any(ttest_stats["pval"] < alpha):
            print("Reject the null hypothesis; there is a significant difference between the sample mean and the hypothesized population mean.")
        else:
            print("Fail to reject the null hypothesis; there is no significant difference between the sample mean and the hypothesized population mean.")
