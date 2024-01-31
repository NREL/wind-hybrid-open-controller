import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("hercules_output.csv")

print(df.columns)

n_turbines = 2
wf_str = "hercules_comms.amr_wind.wind_farm_0."
pow_cols = [wf_str+"turbine_powers.{0:03d}".format(t) for t in range(n_turbines)]
wd_cols = [wf_str+"turbine_wind_directions.{0:03d}".format(t) for t in range(n_turbines)]
yaw_cols = [wf_str+"turbine_yaw_angles.{0:03d}".format(t) for t in range(n_turbines)]

# Extract data from larger array
time = df.dt.values * np.arange(0, len(df), 1)
powers = df[pow_cols].to_numpy()
wds = df[wd_cols].to_numpy()
yaws = df[yaw_cols].to_numpy()

# Plots
fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(10, 5)

# Direction
for t in range(n_turbines):
    line, = ax[0].plot(time, wds[:,t], label="T{0:03d} wind dir.".format(t))
    ax[0].plot(time, yaws[:,t], color=line.get_color(), label="T{0:03d} yaw pos.".format(t),
        linestyle=":")
    if t == 0:
        ax[1].fill_between(time, powers[:,t], color=line.get_color(),
            label="T{0:03d} power".format(t))
    else:
        ax[1].fill_between(time, powers[:,:t+1].sum(axis=1), powers[:,:t].sum(axis=1),
            color=line.get_color(), label="T{0:03d} power".format(t))
ax[1].plot(time, powers.sum(axis=1), color="black", label="Farm power")

# Plot aesthetics
ax[0].grid()
ax[0].set_xlim([time[0], time[-1]])
ax[0].set_ylim([240, 290])
ax[0].set_ylabel("Direction [deg]")
ax[0].legend(loc="lower left")

ax[1].grid()
ax[1].set_ylabel("Power [kW]")
ax[1].set_xlabel("Time [s]")
ax[1].legend(loc="lower left")

#fig.savefig("../../docs/graphics/lookup-table-example-plot.png", dpi=300, format="png")

# Almost equal power to begin with as turbines turbines are not aligned. As the wind direction
# shifts towards the aligned direction beginning at t = 30s, the downstream turbine (T001) begins to
# lose significant power, until the wake steering begins around t = 60s. At t = 70s, noise in the
# wind direction propagates instantaneously into the power signal (as steady-state FLORIS is used
# in place of the dynamic AMR-wind simulation.

# Note that in the upper plot, T000 dir., T001 dir., and T001 yaw are identical througout.

plt.show()