import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/hercules_output.csv")

n_turbines = 1
wf_str = "hercules_comms.amr_wind.wind_farm_0."
pow_cols = [wf_str+"turbine_powers.{0:03d}".format(t) for t in range(n_turbines)]
wd_cols = [wf_str+"turbine_wind_directions.{0:03d}".format(t) for t in range(n_turbines)]
yaw_cols = [wf_str+"turbine_yaw_angles.{0:03d}".format(t) for t in range(n_turbines)]
ref_col = "external_signals.wind_power_reference"

ws_col = "hercules_comms.amr_wind.wind_farm_0.wind_speed"

# Create plots
fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(10, 5)

# Extract data from larger array
time = df['time'].to_numpy()
powers = df[pow_cols].to_numpy()
wds = df[wd_cols].to_numpy()
yaws = df[yaw_cols].to_numpy()
ref = df[ref_col].to_numpy()

ax[0].plot(time/3600, df[ws_col], color="C0", label="Wind speed")
ax[0].set_ylabel("Wind speed [m/s]")
ax[0].grid()

# Power output
line = ax[1].fill_between(time/3600, powers[:,0], label="Turbine power")
ax[1].plot(time/3600, powers.sum(axis=1), color="black", label="Total power")
ax[1].plot(time/3600, ref, color="gray", linestyle="dashed", label="Flexible interconnect limit")

# Plot aesthetics
ax[1].grid()
ax[1].set_ylabel("Power [kW]")
ax[1].set_xlim([time[0]/3600, time[-1]/3600])
ax[1].set_ylim([0, 2000])
ax[1].legend(loc="lower right")
ax[1].set_xlabel("Time [hr]")

# fig.savefig("../../docs/graphics/flexible-interconnect.png", dpi=300, format="png")

plt.show()
