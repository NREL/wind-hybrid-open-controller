# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import pandas as pd
from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

"""
NOTE: Currently, FLORIS v3 is required to run this! Will fix when v4 more mature.
"""

optimize_yaw_offsets = True
build_external_data = True

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
            "jimenez": {"ad": 0.0, "bd": 0.0, "kd": 0.05},
        },
        "wake_turbulence_parameters": {
            "crespo_hernandez": {"initial": 0.1, "constant": 0.5, "ai": 0.8, "downstream": -0.32}
        },
        "wake_velocity_parameters": {
            "gauss": {"alpha": 0.58, "beta": 0.077, "ka": 0.38, "kb": 0.004},
            "jensen": {"we": 0.05},
        },
    },
    "farm": {
        "layout_x": [0.0],
        "layout_y": [0.0],
        "turbine_type": ["nrel_5MW"],
    },
    "flow_field": {
        "wind_speeds": [8.0],
        "wind_directions": [270.0],
        "wind_veer": 0.0,
        "wind_shear": 0.12,
        "air_density": 1.225,
        "turbulence_intensity": 0.06,
        "reference_wind_height": 90.0,
    },
    "name": "GCH_for_FlorisStandin",
    "description": "FLORIS Gauss Curl Hybrid model standing in for AMR-Wind",
    "floris_version": "v4.x",
}

if optimize_yaw_offsets:
    fi = FlorisInterface(floris_dict)

    fi.reinitialize(
        layout_x=[0.0, 1000.0],
        layout_y=[0.0, 0.0],
        wind_directions=np.arange(0.0, 360.0, 3.0),
        wind_speeds=np.arange(2.0, 18.0, 1.0),
    )

    yaw_opt = YawOptimizationSR(fi, verify_convergence=True)
    df_opt = yaw_opt.optimize()

    print("Optimization results:")
    print(df_opt)

    df_opt.to_pickle("yaw_offsets.pkl")

if build_external_data:
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

    df_data.to_csv("amr_standin_data.csv")
