import sys
import yaml
import os
from glob import glob
import pandas as pd

from hercules.utilities import load_yaml

import whoc
from whoc.wind_field.WindField import generate_multi_wind_ts, WindField


def generate_freestream_wind(hercules_input_dict, wind_field_config, wind_field_dir, save_path, n_seeds, regenerate_wind_field=False):
    # instantiate wind field if files don't already exist
    wind_field_filenames = glob(os.path.join(f"{wind_field_dir}", "case_*.csv"))
    # distribution_params_path = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "wind_field_data", "wind_preview_distribution_params.pkl")    
    
    os.makedirs(wind_field_dir, exist_ok=True)

    seed = 0
    wind_field_config["num_turbines"] = hercules_input_dict["controller"]["num_turbines"]
    wind_field_config["preview_dt"] = int(hercules_input_dict["controller"]["dt"] / hercules_input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = hercules_input_dict["dt"]

    wind_field_config["n_preview_steps"] = int(hercules_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] / hercules_input_dict["dt"]) + hercules_input_dict["controller"]["n_horizon"] * int(hercules_input_dict["controller"]["dt"] / hercules_input_dict["dt"])
    
    wind_field_config["simulation_sampling_time"] = hercules_input_dict["dt"]
    wind_field_config["distribution_params_path"] = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "wind_field_data", "wind_preview_distribution_params.pkl")  
    wind_field_config["time_series_dt"] = hercules_input_dict["dt"]
    wind_field_config["n_samples_per_init_seed"] = 1
    
    # if not os.path.exists(distribution_params_path):
    #     wind_preview_distribution_params = wf._generate_wind_preview_distribution_params(int(wind_field_config["simulation_max_time"] // wind_field_config["simulation_sampling_time"]) + wind_field_config["n_preview_steps"], wind_field_config["preview_dt"], regenerate_params=False)
    
    if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
        wind_field_config["regenerate_distribution_params"] = True
        wf = WindField(**wind_field_config)
        generate_multi_wind_ts(wf, wind_field_config, seeds=[seed + i for i in range(n_seeds)])
        wind_field_filenames = [f"case_{i}.csv" for i in range(n_seeds)]
        regenerate_wind_field = True

    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(fn, index_col=0))

    # true wind disturbance time-series
    amr_standin_data = []
    for case_idx in range(len(wind_field_data)):
        amr_standin_data.append({
                                    "time": wind_field_data[case_idx]["Time"].to_numpy(),
                                    "amr_wind_speed": wind_field_data[case_idx]["FreestreamWindMag"].to_numpy(),
                                    "amr_wind_direction": wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
                                })
        df = pd.DataFrame(amr_standin_data[-1])
        df.to_csv(os.path.join(save_path, f"amr_standin_data_{case_idx}.csv"))
    
    return amr_standin_data