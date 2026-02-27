import argparse
import os

import numpy as np
import pandas as pd
import yaml
from floris import FlorisModel
from hycon.design_tools.wake_steering_design import build_simple_wake_steering_lookup_table

with open(os.path.join('inputs','gch_whoc_example.yaml'),'r') as f:
    floris_dict = yaml.safe_load(f)

if __name__ == "__main__":
    # Handle inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaw_offset_filename", default="yaw_offsets.pkl")
    parser.add_argument("--input_wind_filename", default="amr_standin_data.csv")

    args = parser.parse_args()

    fmodel = FlorisModel(floris_dict)

    df_opt = build_simple_wake_steering_lookup_table(
        fmodel,
        wd_resolution=3.0,
        ws_resolution=1.0,
        ws_min=2.0,
        ws_max=17.0,
        minimum_yaw_angle=-25.0,
        maximum_yaw_angle=25.0,
    )

    print("Optimization results:")
    print(df_opt)

    df_opt.to_pickle(args.yaw_offset_filename)

    # Also, build an example external data file
    total_time = 100 # seconds
    dt = 0.5
    np.random.seed(0)
    wind_directions = np.concatenate((
        260*np.ones(60),
        np.linspace(260., 270., 80),
        270. + 5.*np.random.randn(round(total_time/dt)-60-80)
    ))
    df_data = pd.DataFrame(data={
        "time": np.arange(0, total_time, dt),
        "amr_wind_speed": 8.0*np.ones_like(wind_directions),
        "amr_wind_direction": wind_directions
    })

    df_data.to_csv(args.input_wind_filename, index=False)
