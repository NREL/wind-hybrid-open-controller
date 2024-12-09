"""
This script is not run as part of the main simulation procedure, but demonstrates
various capabilities of the wake_steering_design toolbox in WHOC by designing
a range of offset lookup tables and comparing them to one-another.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from floris import FlorisModel

from whoc.design_tools import wake_steering_design as wsd
from whoc.design_tools import wake_steering_visualization as wsv

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
    # Specify various settings for the lookup table designs
    wd_resolution = 3.0
    ws_resolution = 1.0
    ws_min = 2.0
    ws_max = 17.0
    minimum_yaw_angle = 0.0
    maximum_yaw_angle = 25.0
    wd_std = 3.0
    ws_main = 8.0
    plot_turbine = 0

    # Plotting
    col_simple = "black"
    col_unc = "C0"
    
    fmodel = FlorisModel(floris_dict)

    print("Building simple lookup table.")
    df_opt_simple = wsd.build_simple_wake_steering_lookup_table(
        fmodel,
        wd_resolution=wd_resolution,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    print("\nBuilding lookup table with 3 degrees of wind direction uncertainty.")
    df_opt_unc = wsd.build_uncertain_wake_steering_lookup_table(
        fmodel,
        wd_std=wd_std,
        wd_resolution=wd_resolution,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    # Plot various designs
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)

    wsv.plot_offsets_wd(
        df_opt_simple,
        plot_turbine,
        ws_plot=ws_main,
        color=col_simple,
        label="Simple",
        ax=ax[0]
    )

    wsv.plot_offsets_wd(
        df_opt_unc,
        plot_turbine,
        ws_plot=ws_main,
        color=col_unc,
        label="Uncertain",
        ax=ax[0]
    )

    ax[0].set_ylabel("Yaw offset [deg]")
    ax[0].set_xlabel("")
    ax[0].legend()
    ax[0].grid()
    ax[1].set_ylabel("Yaw offset [deg]")
    ax[1].grid()
    ax[-1].set_xlabel("Wind direction [deg]")
    ax[-1].set_xlim([240, 300])

    plt.show()

    # Plot the wake steering misalignments for the first turbine
    # We have tools for this---where are they?
