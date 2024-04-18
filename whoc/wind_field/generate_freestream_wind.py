import sys
import yaml
import os
from glob import glob
import pandas as pd

from hercules.utilities import load_yaml

import whoc
from whoc.wind_field.WindField import generate_multi_wind_ts, WindField


def generate_freestream_wind(save_path, n_seeds, regenerate_wind_field=False):

    input_dict = load_yaml(sys.argv[1])

    with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "wind_field_data", "raw_data")
    wind_field_filenames = glob(os.path.join(f"{wind_field_dir}", "case_*.csv"))
    distribution_params_path = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "wind_field_data", "wind_preview_distribution_params.pkl")    
    
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    seed = 0
    wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * input_dict["controller"]["dt"]
    wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
    wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
    wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["preview_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = input_dict["dt"]
    
    wf = WindField(**wind_field_config)
    if not os.path.exists(distribution_params_path):
        wind_preview_distribution_params = wf._generate_wind_preview_distribution_params(int(wind_field_config["simulation_max_time"] // wind_field_config["simulation_sampling_time"]) + wind_field_config["n_preview_steps"], wind_field_config["preview_dt"], regenerate_params=False)
    
    if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
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
                                    "time": wind_field_data[case_idx]["Time"],
                                    "amr_wind_speed": wind_field_data[case_idx]["FreestreamWindMag"].to_numpy(),
                                    "amr_wind_direction": wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
                                })
        df = pd.DataFrame(amr_standin_data[-1])
        df.to_csv(os.path.join(save_path, f"amr_standin_data_{case_idx}.csv"))
    
    return amr_standin_data