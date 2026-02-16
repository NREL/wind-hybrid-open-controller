import matplotlib.pyplot as plt
from hercules import HerculesOutput


def plot_outputs():
    dfs = [
        HerculesOutput("outputs/hercules_output_ol.h5").df,
        HerculesOutput("outputs/hercules_output_cl.h5").df,
    ]

    labels = ["Open-loop control", "Closed-loop control"]

    n_turbines = 2
    pow_cols = ["wind_farm.turbine_powers.{0:03d}".format(t) for t in range(n_turbines)]
    ref_col = "external_signals.wind_power_reference"

    # Create plots
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)

    for case, (df, label) in enumerate(zip(dfs, labels)):
        # Extract data from larger array
        time = df["time"].to_numpy()
        powers = df[pow_cols].to_numpy()
        ref = df[ref_col].to_numpy()

        # Direction
        for t in range(n_turbines):
            if t == 0:
                ax[case].fill_between(time, powers[:, t], label="T{0:03d} power".format(t))
            else:
                ax[case].fill_between(
                    time,
                    powers[:, : t + 1].sum(axis=1),
                    powers[:, :t].sum(axis=1),
                    label="T{0:03d} power".format(t),
                )
        ax[case].plot(time, powers.sum(axis=1), color="black", label="Farm power")
        ax[case].plot(time, ref, color="gray", linestyle="dashed", label="Ref. power")

        # Plot aesthetics
        ax[case].grid()
        ax[case].set_title(label)
        ax[case].set_ylabel("Power [kW]")
    ax[0].set_xlim([time[0], time[-1]])
    ax[0].legend(loc="lower left")
    ax[1].set_xlabel("Time [mins]")

    return fig


# In this example, the wind turbines are aligned with the oncoming wind, so T000 wakes T001.
# The farm power setpoint more than available to begin, so both
# turbines are at max power. Between 10s and 20s, the setpoint ramps down to 3000kW; the open-loop
# controller asks each turbine for 1500kW, but only the upstream turbine is able to meet the demand,
# so the total farm power is below the setpoint. The closed-loop controller is able to adjust the
# power of T000 to compensate for T001's underperformance, and the farm power tracks the setpoint.
# When the setpoint shifts to 2000kW, there is sufficient resource for T001 to produce 1000kW, and
# both controllers meet the setpoint.

if __name__ == "__main__":
    fig = plot_outputs()
    # fig.savefig("../../docs/graphics/wf-power-tracking-plot.png", dpi=300, format="png")
    plt.show()
