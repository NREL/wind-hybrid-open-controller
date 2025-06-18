import argparse

import numpy as np
import pandas as pd
from floris import FlorisModel
from whoc.design_tools.wake_steering_design import build_simple_wake_steering_lookup_table

floris_dict = {
    "logging": {
        "console": {"enable": True, "level": "WARNING"},
        "file": {"enable": False, "level": "WARNING"},
    },
    "solver": {"type": "turbine_grid", "turbine_grid_points": 3},
    "wake": {
        "model_strings": {
            "combination_model": "sosfs",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "velocity_model": "gauss",
        },
        "enable_secondary_steering": True,
        "enable_yaw_added_recovery": True,
        "enable_transverse_velocities": True,
        "enable_active_wake_mixing": False,
        "wake_deflection_parameters": {
            "gauss": {
                "ad": 0.0,
                "alpha": 0.58,
                "bd": 0.0,
                "beta": 0.077,
                "dm": 1.0,
                "ka": 0.38,
                "kb": 0.004,
            },
        },
        "wake_turbulence_parameters": {
            "crespo_hernandez": {"initial": 0.1, "constant": 0.5, "ai": 0.8, "downstream": -0.32}
        },
        "wake_velocity_parameters": {
            "gauss": {"alpha": 0.58, "beta": 0.077, "ka": 0.38, "kb": 0.004},
        },
    },
    "farm": {
        "layout_x": [0.0, 1000.0],
        "layout_y": [0.0, 0.0],
        "turbine_type": ["nrel_5MW"],
    },
    "flow_field": {
        "wind_speeds": [8.0],
        "wind_directions": [270.0],
        "turbulence_intensities": [0.06],
        "wind_veer": 0.0,
        "wind_shear": 0.12,
        "air_density": 1.225,
        "reference_wind_height": 90.0,
    },
    "name": "GCH_for_FlorisStandin",
    "description": "FLORIS Gauss Curl Hybrid model standing in for AMR-Wind",
    "floris_version": "v4.x",
}

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
