import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the simulation
df = pd.read_csv("outputs/hercules_output_control.csv", index_col=False)
h2_ref_input = pd.read_csv("inputs/hydrogen_ref_signal.csv")

# Extract individual components powers as well as total power
n_wind_turbines = 9
wind_power = df[["hercules_comms.amr_wind.wind_farm_0.turbine_powers.{0:03d}".format(t)
                 for t in range(n_wind_turbines)]].to_numpy().sum(axis=1) / 1e3
hydrogen_output = df["py_sims.hydrogen_plant_0.outputs.H2_output"]
# power_output = (df["py_sims.inputs.available_power"]) / 1e3
time = df["hercules_comms.amr_wind.wind_farm_0.sim_time_s_amr_wind"] / 60 # minutes

# Set plotting aesthetics
wind_col = "C0"
h2_col = "C2"
plant_col = "C3"


wind_power_individuals = df[["hercules_comms.amr_wind.wind_farm_0.turbine_powers.{0:03d}".format(t)
                             for t in range(n_wind_turbines)]].to_numpy() / 1e3

# Plot the hydrogen output 
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12,8))
ax[0].plot(time, wind_power, color=plant_col, label="Total Plant Power")
ax[0].set_ylabel("Power [MW]")
ax[0].grid()
ax[0].legend()

turbines = range(0,9)
width = 12
alpha = 0.25
red_col = 0.5
blue_col = 1.0
for i in turbines:
    turb_num = i
    turb_str = str(turb_num)
    ax[1].plot(time,  wind_power_individuals[:,i], label="WT"+str(i), \
               color=(0.5, blue_col, red_col), linewidth=width, alpha = alpha)
    width = width - 1.25
    alpha = alpha+0.075
    red_col = red_col+0.05
    blue_col = blue_col-0.05
# for i in range (n_wind_turbines):
#     ax[1].plot(time, wind_power_individuals[:,i], label="WT"+str(i), alpha=0.7, color=wind_col)
ax[1].grid()
ax[1].legend()

ax[2].plot(h2_ref_input['time'] / 60, h2_ref_input['hydrogen_reference'],\
            'k--', label="Hydrogen Reference")
ax[2].plot(time, hydrogen_output, color=h2_col, label='Hydrogen Rate Output')
ax[2].set_ylabel("Hydrogen Production Rate [kg/s]")
ax[2].grid()
ax[2].legend()
ax[2].set_xlabel("Time [mins]")

# fig.savefig("../../docs/graphics/wind-hydrogen-example-plot.png", dpi=300, format="png")

plt.show()
