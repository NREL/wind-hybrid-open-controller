import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/hercules_output.csv")

n_turbines = 1
wf_str = "hercules_comms.amr_wind.wind_farm_0."
pow_cols = [wf_str+"turbine_powers.{0:03d}".format(t) for t in range(n_turbines)]
wd_cols = [wf_str+"turbine_wind_directions.{0:03d}".format(t) for t in range(n_turbines)]
yaw_cols = [wf_str+"turbine_yaw_angles.{0:03d}".format(t) for t in range(n_turbines)]
ref_col = "external_signals.wind_power_reference"

# Create plots
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# Extract data from larger array
time = df['time'].to_numpy()
powers = df[pow_cols].to_numpy()
wds = df[wd_cols].to_numpy()
yaws = df[yaw_cols].to_numpy()
ref = df[ref_col].to_numpy()

# Direction
for t in range(n_turbines):
    if t == 0:
        line = ax.fill_between(time/3600, powers[:,t], label="T{0:03d} power".format(t))
    else:
        ax.fill_between(time/3600, powers[:,:t+1].sum(axis=1), powers[:,:t].sum(axis=1),
            label="T{0:03d} power".format(t))
ax.plot(time/3600, powers.sum(axis=1), color="black", label="Farm power")
ax.plot(time/3600, ref, color="gray", linestyle="dashed", label="Ref. power")

# Plot aesthetics
ax.grid()
ax.set_ylabel("Power [kW]")
ax.set_xlim([time[0]/3600, time[-1]/3600])
ax.set_ylim([0, 2000])
ax.legend(loc="lower left")
ax.set_xlabel("Time [hr]")

# fig.savefig("../../docs/graphics/flexible-interconnect.png", dpi=300, format="png")

# In this example, the a single ~1.5MW wind turbine generates power as a distributed
# power plant but most follow a flexible interconnect amount. The turbine is controlled
# to generated maximum power under the time-varying interconnect limit, which differs each
# hour of the day for 24 hours.

plt.show()