# Plot the outputs of the simulation for the wind and storage example
import argparse

import matplotlib.pyplot as plt
import numpy as np
from hercules import HerculesOutput


def plot_outputs():
    # Read the Hercules output file using HerculesOutput
    ho = HerculesOutput("outputs/hercules_output.h5")

    # Create a shortcut to the dataframe
    df = ho.df

    fig, axarr = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 8)

    # Plot grid signals as well as battery limits
    ax = axarr[0]

    ax.plot(
        df["time"],
        df["external_signals.lmp_da"],
        label="Day-ahead LMP",
        color="k",
        linestyle="--",
        linewidth=0.5,
    )
    ax.plot(
        df["time"],
        df["external_signals.lmp_rt"],
        label="Real-time LMP",
        color="k",
        linestyle="-",
        linewidth=0.5,
    )

    # Process and plot the day ahead top and bottom 4 and 1
    # for each midnight to midnight (UTC) period
    df_reduced = df[["time_utc", "external_signals.lmp_da"]].copy()

    # Create an hourly version for the DA LMP (drop all periods that aren't on an hour)
    df_hr = df_reduced[df_reduced["time_utc"].dt.minute == 0].copy(deep=True)

    # Extract date and hour for use in pivot_table
    df_hr["date"] = df_hr["time_utc"].dt.date
    df_hr["hour"] = df_hr["time_utc"].dt.hour

    df_hourly = df_hr.pivot_table(
        values="external_signals.lmp_da",
        index="date",
        columns="hour",
        aggfunc="first",  # Use first value if multiple entries per hour
    )
    day_hour = df_hourly.to_numpy()
    sorted_day_hour = np.sort(day_hour, axis=1)
    df_hourly["B4"] = sorted_day_hour[:, 3]
    df_hourly["T4"] = sorted_day_hour[:, -4]
    df_hourly["B1"] = sorted_day_hour[:, 0]
    df_hourly["T1"] = sorted_day_hour[:, -1]

    df = df.merge(
        df_hourly[["B4", "T4", "B1", "T1"]],
        left_on=df["time_utc"].dt.date,
        right_on=df_hourly.index,
        how="left",
    ).ffill()

    ax.fill_between(
        df["time"],
        df["T4"],
        500,
        label="Main discharge price (daily)",
        color="C2",
        alpha=0.3,
        edgecolor="None",
    )
    ax.fill_between(
        df["time"],
        df["T1"],
        500,
        label="High discharge price (daily)",
        color="C2",
        alpha=0.5,
        edgecolor="None",
    )
    ax.fill_between(
        df["time"],
        -500,
        df["B4"],
        label="Main charge price (daily)",
        color="C1",
        alpha=0.3,
        edgecolor="None",
    )
    ax.fill_between(
        df["time"],
        -500,
        df["B1"],
        label="Low charge price (daily)",
        color="C1",
        alpha=0.5,
        edgecolor="None",
    )
    ax.set_ylim([-50, 50])
    ax.set_ylabel("Price [$/MWh]")
    ax.set_xlim([0, df["time"].max()])

    # Plot the battery power and SOC
    ax = axarr[1]
    ax.plot(
        df["time"],
        df["battery.power"] / 1e3,  # Base unit: kW
        label="Battery output",
        color="k",
        linewidth=1.0,
    )
    ax.plot(
        df["time"],
        df["battery.power_setpoint"] / 1e3,  # Base unit: kW
        label="Battery setpoint",
        color="k",
        linestyle=":",
        linewidth=1.0,
    )
    ax.set_ylabel("Power [MW]")

    ax2 = ax.twinx()

    color = "C0"
    ax2.set_ylabel("State of charge [-]", color=color)
    ax2.plot(df["time"], df["battery.soc"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    for ax in axarr:
        ax.grid(True)
        ax.legend(loc="upper right")

    # Compute total revenue on real-time market
    df["revenue_rt"] = df["battery.power"] / 1e3 * df["external_signals.lmp_rt"] / 3600
    print("Real-time revenue over simulation: ${:.2f}".format(df["revenue_rt"].sum()))

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot outputs of battery market example")

    parser.add_argument(
        "--save_plots", type=bool, default=False, help="Whether to save the generated plots"
    )

    args = parser.parse_args()

    fig = plot_outputs()

    if args.save_plots:
        fig.savefig("../../docs/graphics/battery-market.png", dpi=300, format="png")

    plt.show()
