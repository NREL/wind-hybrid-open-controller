import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dfs = [pd.read_csv("hercules_output_ol.csv"), pd.read_csv("hercules_output_cl.csv")]
labels = ["Open-loop control", "Closed-loop control"]

n_turbines = 2
wf_str = "hercules_comms.amr_wind.wind_farm_0."
pow_cols = [wf_str+"turbine_powers.{0:03d}".format(t) for t in range(n_turbines)]
wd_cols = [wf_str+"turbine_wind_directions.{0:03d}".format(t) for t in range(n_turbines)]
yaw_cols = [wf_str+"turbine_yaw_angles.{0:03d}".format(t) for t in range(n_turbines)]
ref_col = "external_signals.wind_power_reference"

# Create plots
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

for case, (df, label) in enumerate(zip(dfs, labels)):
    # Extract data from larger array
    time = df.dt.values * np.arange(0, len(df), 1)
    powers = df[pow_cols].to_numpy()
    wds = df[wd_cols].to_numpy()
    yaws = df[yaw_cols].to_numpy()
    ref = df[ref_col].to_numpy()

    # Direction
    for t in range(n_turbines):
        if t == 0:
            line = ax[case].fill_between(time, powers[:,t], label="T{0:03d} power".format(t))
        else:
            ax[case].fill_between(time, powers[:,:t+1].sum(axis=1), powers[:,:t].sum(axis=1),
                label="T{0:03d} power".format(t))
    ax[case].plot(time, powers.sum(axis=1), color="black", label="Farm power")
    ax[case].plot(time, ref, color="gray", linestyle="dashed", label="Ref. power")

    # Plot aesthetics
    ax[case].grid()
    ax[case].set_title(label)
    ax[case].set_ylabel("Power [kW]")
ax[0].set_xlim([time[0], time[-1]])
ax[0].legend(loc="lower left")
ax[1].set_xlabel("Time [s]")

#fig.savefig("../../docs/graphics/lookup-table-example-plot.png", dpi=300, format="png")

# Almost equal power to begin with as turbines turbines are not aligned. As the wind direction
# shifts towards the aligned direction beginning at t = 30s, the downstream turbine (T001) begins to
# lose significant power, until the wake steering begins around t = 60s. At t = 70s, noise in the
# wind direction propagates instantaneously into the power signal (as steady-state FLORIS is used
# in place of the dynamic AMR-wind simulation.

# Note that in the upper plot, T000 dir., T001 dir., and T001 yaw are identical througout.

plt.show()