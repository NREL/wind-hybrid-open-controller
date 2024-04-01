import sys
import yaml
import os
from glob import glob
import pandas as pd

from hercules.utilities import load_yaml

import whoc
from whoc.wind_field.WindField import generate_multi_wind_ts


def generate_freestream_wind(save_path, regenerate_wind_field=False):

    input_dict = load_yaml(sys.argv[1])

    with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join('/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/wind_field_data/raw_data')        
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    n_wind_field_cases = 1
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    # TODO make sure this is the same as the wind field from amr_standin_data
    # TODO how can we make hercules wait for controller response?s
    seed = 0
    wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * input_dict["controller"]["dt"]
    wind_field_config["preview_dt"] = input_dict["controller"]["dt"]
    if not len(wind_field_filenames) or regenerate_wind_field:
        generate_multi_wind_ts(wind_field_config, seed=seed)
        wind_field_filenames = [f"case_{i}.csv" for i in range(n_wind_field_cases)]
        regenerate_wind_field = True

    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(os.path.join(wind_field_dir, fn)))

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