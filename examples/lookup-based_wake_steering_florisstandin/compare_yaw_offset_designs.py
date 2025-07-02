"""
This script is not run as part of the main simulation procedure, but demonstrates
various capabilities of the wake_steering_design toolbox in WHOC by designing
a range of offset lookup tables and comparing them to one-another.
"""


import matplotlib.pyplot as plt
from floris import FlorisModel
from whoc.design_tools import wake_steering_design as wsd, wake_steering_visualization as wsv

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
    ti_resolution = 0.02
    ti_min = 0.06
    ti_max = 0.08
    minimum_yaw_angle = -25.0
    maximum_yaw_angle = 25.0
    wd_std = 3.0
    ws_main = 8.0
    wd_rate_limit = 3.0
    ws_rate_limit = 100.0 # No rate limit on wind speed
    ti_rate_limit = 1e3 # No rate limit on turbulence intensity
    plot_turbine = 0
    plot_wd_lims = (240, 300)

    # Plotting
    col_simple = "black"
    col_unc = "C0"
    col_rate_limited = "C1"
    col_ws_ramps = "C2"
    
    fmodel = FlorisModel(floris_dict)

    print("Building simple lookup table.")
    df_opt_simple = wsd.build_simple_wake_steering_lookup_table(
        fmodel,
        wd_resolution=wd_resolution,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        ti_resolution=ti_resolution,
        ti_min=ti_min,
        ti_max=ti_max,
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
        ti_resolution=ti_resolution,
        ti_min=ti_min,
        ti_max=ti_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    print("\nApplying rate limits to simple lookup table (3 deg/deg).")
    df_opt_rate_limited = wsd.apply_static_rate_limits(
        df_opt_simple,
        wd_rate_limit=wd_rate_limit,
        ws_rate_limit=ws_rate_limit,
        ti_rate_limit=ti_rate_limit,
    )

    print("\nGenerating offsets using single wind speed.")
    df_opt_single_ws = wsd.build_simple_wake_steering_lookup_table(
        fmodel,
        wd_resolution=wd_resolution,
        ws_resolution=ws_resolution,
        ws_min=ws_main,
        ws_max=ws_main,
        ti_resolution=ti_resolution,
        ti_min=ti_min,
        ti_max=ti_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    df_opt_ws_ramps = wsd.apply_wind_speed_ramps(
        df_opt_single_ws,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        ws_wake_steering_cut_in=3.0,
        ws_wake_steering_fully_engaged_low=4.0,
        ws_wake_steering_fully_engaged_high=11.0,
        ws_wake_steering_cut_out=13.0
    )

    # Plot various designs
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True, sharey=True)

    wsv.plot_offsets_wd(
        df_opt_simple,
        plot_turbine,
        ws_plot=ws_main,
        ti_plot=ti_min,
        color=col_simple,
        label="Simple",
        ax=ax
    )

    wsv.plot_offsets_wd(
        df_opt_unc,
        plot_turbine,
        ws_plot=ws_main,
        ti_plot=ti_min,
        color=col_unc,
        label="Uncertain",
        ax=ax
    )

    wsv.plot_offsets_wd(
        df_opt_rate_limited,
        plot_turbine,
        ws_plot=ws_main,
        ti_plot=ti_min,
        color=col_rate_limited,
        label="Rate limited",
        ax=ax
    )

    wsv.plot_offsets_wd(
        df_opt_ws_ramps,
        plot_turbine,
        ws_plot=ws_main,
        ti_plot=ti_min,
        color=col_ws_ramps,
        label="Single wind speed",
        linestyle="dotted",
        ax=ax
    )

    ax.set_title("Yaw offsets at {} m/s".format(ws_main))
    ax.set_ylabel("Yaw offset [deg]")
    ax.grid()
    ax.set_xlabel("Wind direction [deg]")
    ax.set_xlim(plot_wd_lims)
    ax.legend()

    # Also, plot heatmap of offsets for Simple design
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))
    _, cbar = wsv.plot_offsets_wdws_heatmap(
        df_opt_simple,
        plot_turbine,
        ti_plot=ti_min,
        vmax=maximum_yaw_angle,
        vmin=minimum_yaw_angle,
        ax=ax[0,0]
    )
    cbar.set_label("Yaw offset [deg]")
    ax[0,0].set_title("Simple")
    _, cbar = wsv.plot_offsets_wdws_heatmap(
        df_opt_unc,
        plot_turbine,
        ti_plot=ti_min,
        vmax=maximum_yaw_angle,
        vmin=minimum_yaw_angle,
        ax=ax[0,1]
    )
    cbar.set_label("Yaw offset [deg]")
    ax[0,1].set_title("Uncertain")
    _, cbar = wsv.plot_offsets_wdws_heatmap(
        df_opt_rate_limited,
        plot_turbine,
        ti_plot=ti_min,
        vmax=maximum_yaw_angle,
        vmin=minimum_yaw_angle,
        ax=ax[1,0]
    )
    cbar.set_label("Yaw offset [deg]")
    ax[1,0].set_title("Rate limited")
    _, cbar = wsv.plot_offsets_wdws_heatmap(
        df_opt_ws_ramps,
        plot_turbine,
        ti_plot=ti_min,
        vmax=maximum_yaw_angle,
        vmin=minimum_yaw_angle,
        ax=ax[1,1]
    )
    cbar.set_label("Yaw offset [deg]")
    ax[1,1].set_title("Single wind speed heuristic")

    for ax_ in ax[:,0]:
        ax_.set_ylabel("Wind speed [m/s]")
    for ax_ in ax[-1,:]:
        ax_.set_xlabel("Wind direction [deg]")
    ax[0,0].set_xlim(plot_wd_lims)

    plt.show()
