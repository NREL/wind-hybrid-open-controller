import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data from the simulation
df = pd.read_csv("outputs/hercules_output.csv", index_col=False)
power_ref_input = pd.read_csv("inputs/plant_power_reference.csv")

# Extract individual components powers as well as total power
solar_power = df["py_sims.solar_farm_0.outputs.power_mw"]
n_wind_turbines = 10
wind_power = df[["hercules_comms.amr_wind.wind_farm_0.turbine_powers.{0:03d}".format(t)
                 for t in range(n_wind_turbines)]].to_numpy().sum(axis=1) / 1e3
battery_power = -df["py_sims.battery_0.outputs.power"] / 1e3 # discharging positive
power_output = (df["py_sims.inputs.available_power"] - df["py_sims.battery_0.outputs.power"]) / 1e3
time = df["hercules_comms.amr_wind.wind_farm_0.sim_time_s_amr_wind"] / 60 # minutes

# Set plotting aesthetics
wind_col = "C0"
solar_col = "C1"
battery_col = "C2"
plant_col = "C3"

# Plotting power outputs from each technology as well as the total power output (top)
# Plotting the SOC of the battery (bottom)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,5))
ax.plot(time, wind_power, label="Wind", color=wind_col)
ax.plot(time, solar_power, label="Solar PV", color=solar_col)
ax.plot(time, battery_power, label="Battery", color=battery_col)
ax.plot(time, power_output, label="Plant output", color=plant_col)
ax.plot(power_ref_input['time'] / 60, power_ref_input['plant_power_reference']/1e3,\
            'k--', label="Reference")
ax.set_ylabel("Power [MW]")
ax.set_xlabel("Time [mins]")
ax.grid()
ax.legend(loc="upper right")
ax.set_xlim([0, 1])

# Plot the battery power and state of charge
battery_soc = df["py_sims.battery_0.outputs.soc"]
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,5))
ax[0].plot(time, battery_power, color=battery_col)
ax[1].plot(time, battery_soc, color=battery_col)
ax[0].set_ylabel("Battery power [MW]")
ax[0].grid()
ax[1].set_ylabel("Battery SOC")
ax[1].set_xlabel("Time [mins]")
ax[1].grid()

# Plot the solar data
angle_of_incidence = df["py_sims.solar_farm_0.outputs.aoi"]
direct_normal_irradiance = df["py_sims.solar_farm_0.outputs.dni"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,5))
ax[0].plot(time, solar_power, color="C1")
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
