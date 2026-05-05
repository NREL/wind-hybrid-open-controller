import argparse

import matplotlib.pyplot as plt
import numpy as np
from hercules import HerculesOutput


def plot_outputs():
    # Read the Hercules output file using HerculesOutput
    ho = HerculesOutput("outputs/hercules_output.h5")

    df = ho.df
    print("Available columns in the output DataFrame:")
    for c in df.columns.tolist():
        print(c)

    # Get high-level signals
    power_output = df["plant.power"]
    time = df["time"] / 60  # minutes
    power_ref_input = df["external_signals.plant_power_reference"]

    # Extract individual components powers as well as total power
    if "solar_farm.power" in df.columns:
        solar_power = df["solar_farm.power"]
    else:
        solar_power = np.zeros(len(df))
    n_wind_turbines = 10
    wind_power = (
        df[["wind_farm.turbine_powers.{0:03d}".format(t) for t in range(n_wind_turbines)]]
        .to_numpy()
        .sum(axis=1)
    )
    if "battery.power" in df.columns:
        battery_power = df["battery.power"]  # discharging positive
    else:
        battery_power = np.zeros(len(df))

    # Set plotting aesthetics
    wind_col = "C0"
    solar_col = "C1"
    battery_col = "C2"
    plant_col = "C3"

    # Plotting power outputs from each technology as well as the total power output (top)
    # Plotting the SOC of the battery (bottom)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5))
    ax.plot(time, wind_power / 1e3, label="Wind", color=wind_col)
    ax.plot(time, solar_power / 1e3, label="Solar PV", color=solar_col)
    ax.plot(time, battery_power / 1e3, label="Battery", color=battery_col)
    ax.plot(time, power_output / 1e3, label="Plant output", color=plant_col)
    ax.plot(time, power_ref_input / 1e3, "k--", label="Reference")
    ax.set_ylabel("Power [MW]")
    ax.set_xlabel("Time [mins]")
    ax.grid()
    ax.legend(loc="lower right")
    ax.set_xlim([0, 120])

    # Plot the battery power and state of charge, if battery component included
    if not (battery_power == 0).all():
        battery_soc = df["battery.soc"]
        figb, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
        ax[0].plot(time, battery_power / 1e3, color=battery_col)
        ax[1].plot(time, battery_soc, color=battery_col)
        ax[0].set_ylabel("Battery power [MW]")
        ax[0].grid()
        ax[1].set_ylabel("Battery SOC")
        ax[1].set_xlabel("Time [mins]")
        ax[1].grid()

    # Plot the solar data, if solar component included in Hercules.
    if not (solar_power == 0).all():
        angle_of_incidence = df["solar_farm.aoi"]
        direct_normal_irradiance = df["solar_farm.dni"]
        figs, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
        ax[0].plot(time, solar_power / 1e3, color="C1")
        ax[0].set_ylabel("Solar power [MW]")
        ax[0].grid()

        ax[1].plot(time, direct_normal_irradiance, color="black")
        ax[1].set_ylabel("DNI [W/m$^2$]")
        ax[1].grid()

        ax[2].plot(time, angle_of_incidence, color="black")
        ax[2].set_ylabel("AOI [deg]")
        ax[-1].set_xlabel("Time [mins]")
        ax[2].grid()

    # Plot the wind data
    wind_power_individuals = df[
        ["wind_farm.turbine_powers.{0:03d}".format(t) for t in range(n_wind_turbines)]
    ].to_numpy()
    figw, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax[0].plot(time, wind_power / 1e3, color=wind_col)
    for i in range(n_wind_turbines):
        ax[1].plot(
            time, wind_power_individuals[:, i] / 1e3, label="WT" + str(i), alpha=0.7, color=wind_col
        )
    ax[0].set_ylabel("Total wind power [MW]")
    ax[1].set_ylabel("Individual turbine power [MW]")
    ax[0].grid()
    ax[1].grid()
    ax[1].set_xlabel("Time [mins]")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot outputs of battery market example")

    parser.add_argument(
        "--save_plots", type=bool, default=False, help="Whether to save the generated plots"
    )

    args = parser.parse_args()

    fig = plot_outputs()

    if args.save_plots:
        fig.savefig("../../docs/graphics/simple-hybrid-example-plot.png", dpi=300, format="png")

    plt.show()
