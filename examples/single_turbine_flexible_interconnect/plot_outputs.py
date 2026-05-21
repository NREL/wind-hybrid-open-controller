import argparse

import matplotlib.pyplot as plt
import numpy as np
from hercules import HerculesOutput


def plot_outputs():
    # Plot settings
    lw = 1  # linewidth
    wind_cl = "C0"
    batt_cl = "C1"
    fi_cl = "red"
    pow_cl = "black"
    av_cl = "gray"
    fi_ls = "dotted"  # linestyle
    av_alpha = 1  # opacity

    df_wind = HerculesOutput("outputs/hercules_output_wind_only.h5").df
    df_batt = HerculesOutput("outputs/hercules_output_with_battery.h5").df
    df_base = HerculesOutput("outputs/hercules_output_baseline.h5").df

    print(df_batt["battery.power"].head())

    pow_col = "distributed_wind.turbine_powers.000"
    ref_col = "external_signals.plant_power_reference"
    batt_col = "battery.power"
    ws_col = "distributed_wind.wind_speeds_withwakes.000"

    # Create plots
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(10, 8)

    # Extract data from larger array
    time = df_wind["time"].to_numpy()
    powers_wind_only = df_wind[pow_col].to_numpy()
    powers_with_batt = df_batt[pow_col].to_numpy()
    battery_power = df_batt[batt_col]  # Discharging positive
    powers_base = df_base[pow_col].to_numpy()
    flexible_interconnect = df_wind[ref_col].to_numpy()

    ax[0].plot(time / 3600, df_wind[ws_col], color=wind_cl, label="Wind speed", linewidth=lw)
    ax[0].set_ylabel("Wind speed [m/s]")
    ax[0].grid()

    # Power output wind only
    ax[1].plot(
        time / 3600,
        powers_base,
        color=av_cl,
        alpha=av_alpha,
        linewidth=lw,
        label="Available wind power",
    )
    ax[1].fill_between(time / 3600, powers_wind_only, color=wind_cl, label="Wind power")
    ax[1].plot(time / 3600, powers_wind_only, color=pow_cl, linewidth=lw, label="Plant power")
    ax[1].plot(
        time / 3600,
        flexible_interconnect,
        color=fi_cl,
        linestyle=fi_ls,
        linewidth=lw,
        label="Flexible interconnect limit",
    )

    # Power output wind + battery
    ax[2].plot(
        time / 3600,
        powers_base,
        color=av_cl,
        alpha=av_alpha,
        linewidth=lw,
        label="Available wind power",
    )
    ax[2].fill_between(time / 3600, battery_power, color=batt_cl, label="Battery power")
    ax[2].fill_between(
        time / 3600,
        np.maximum(battery_power, 0),
        powers_with_batt + np.maximum(battery_power, 0),
        color=wind_cl,
        label="Wind power",
    )
    ax[2].plot(
        time / 3600,
        powers_with_batt + battery_power,
        color=pow_cl,
        linewidth=lw,
        label="Plant power",
    )
    ax[2].plot(
        time / 3600,
        flexible_interconnect,
        color=fi_cl,
        linestyle=fi_ls,
        linewidth=lw,
        label="Flexible interconnect limit",
    )

    # Plot aesthetics
    ax[1].grid()
    ax[1].set_ylabel("Power [kW]\n(wind only case)")
    ax[1].set_xlim([time[0] / 3600, time[-1] / 3600])
    ax[1].set_ylim([-200, 2000])
    ax[1].legend(loc="lower center")
    ax[2].grid()
    ax[2].set_ylabel("Power [kW]\n(wind + battery case)")
    ax[2].set_xlim([time[0] / 3600, time[-1] / 3600])
    ax[2].set_ylim([-200, 2000])
    ax[2].legend(loc="lower center")
    ax[-1].set_xlabel("Time [hr]")

    # fig.savefig("../../docs/graphics/flexible-interconnect.png", dpi=300, format="png")

    # Report output results. Wind only case, then wind + battery
    dt = time[1] - time[0]
    available_energy = powers_base.sum() * dt / 3600
    curtailed_energy = (powers_base - powers_wind_only).sum() * dt / 3600
    percentage_curtailed = curtailed_energy / available_energy * 100
    # 10 kW threshold for "curtailed" when computing time curtailed
    curtailed_hrs = ((powers_base - powers_wind_only) > 10).sum() * dt / 3600
    print("\nResults for wind only case")
    print(
        "Curtailed energy: {0:.2f} kWh ({1:.1f}% of available)".format(
            curtailed_energy, percentage_curtailed
        )
    )
    print("Total time curtailed: {0:.1f} hours".format(curtailed_hrs))

    curtailed_energy = (powers_base - powers_with_batt).sum() * dt / 3600
    percentage_curtailed = curtailed_energy / available_energy * 100
    curtailed_hrs = ((powers_base - powers_with_batt) > 10).sum() * dt / 3600
    print("\nResults for wind + battery case")
    print(
        "Curtailed energy: {0:.2f} kWh ({1:.1f}% of available)".format(
            curtailed_energy, percentage_curtailed
        )
    )
    print("Total time curtailed: {0:.1f} hours".format(curtailed_hrs))

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot outputs of battery market example")

    parser.add_argument(
        "--save_plots", type=bool, default=False, help="Whether to save the generated plots"
    )

    args = parser.parse_args()

    fig = plot_outputs()

    if args.save_plots:
        fig.savefig("../../docs/graphics/flexible-interconnect.png", dpi=300, format="png")

    plt.show()
