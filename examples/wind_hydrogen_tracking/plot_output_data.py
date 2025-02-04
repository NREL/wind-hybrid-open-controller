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
power_output = (df["py_sims.inputs.available_power"] - df["py_sims.battery_0.outputs.power"]) / 1e3
time = df["hercules_comms.amr_wind.wind_farm_0.sim_time_s_amr_wind"] / 60 # minutes

# Set plotting aesthetics
wind_col = "C0"
h2_col = "C1"
plant_col = "C3"

# Plotting power outputs from each technology as well as the total power output (top)
# Plotting the SOC of the battery (bottom)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,5))
ax.plot(time, wind_power, label="Wind", color=wind_col)
# ax.plot(time, solar_power, label="Solar PV", color=solar_col)
# ax.plot(time, battery_power, label="Battery", color=battery_col)
ax.set_ylabel("Power [MW]")
ax.set_xlabel("Time [mins]")
ax.grid()
ax.legend(loc="lower right")
ax.set_xlim([0, 5])

# fig.savefig("../../docs/graphics/simple-hybrid-example-plot.png", dpi=300, format="png")

# Plot the hydrogen output 
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,5))
ax.plot(time, hydrogen_output, color=h2_col)
ax.plot(h2_ref_input['time'] / 60, h2_ref_input['hydrogen_reference'],\
            'k--', label="Reference")
ax.set_ylabel("Hydrogen Output [kg/s]")
ax.grid()
ax.set_xlabel("Time [mins]")

# # Plot the solar data
# angle_of_incidence = df["py_sims.solar_farm_0.outputs.aoi"]
# direct_normal_irradiance = df["py_sims.solar_farm_0.outputs.dni"]
# fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,5))
# ax[0].plot(time, solar_power, color="C1")
# ax[0].set_ylabel("Solar power [MW]")
# ax[0].grid()

# ax[1].plot(time, direct_normal_irradiance, color="black")
# ax[1].set_ylabel("DNI [W/m$^2$]")
# ax[1].grid()

# ax[2].plot(time, angle_of_incidence, color="black")
# ax[2].set_ylabel("AOI [deg]")
# ax[-1].set_xlabel("Time [mins]")
# ax[2].grid()

# Plot the wind data
wind_power_individuals = df[["hercules_comms.amr_wind.wind_farm_0.turbine_powers.{0:03d}".format(t)
                             for t in range(n_wind_turbines)]].to_numpy() / 1e3
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,5))
ax[0].plot(time, wind_power, color=wind_col)
for i in range (n_wind_turbines):
    ax[1].plot(time, wind_power_individuals[:,i], label="WT"+str(i), alpha=0.7, color=wind_col)
ax[0].set_ylabel("Total wind power [MW]")
ax[1].set_ylabel("Individual turbine power [MW]")
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel("Time [mins]")

plt.show()
