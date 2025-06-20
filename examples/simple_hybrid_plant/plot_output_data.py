import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the simulation
df = pd.read_csv("outputs/hercules_output.csv", index_col=False)
power_ref_input = pd.read_csv("inputs/plant_power_reference.csv")

# Extract individual components powers as well as total power
if "py_sims.solar_farm_0.outputs.power_kw" in df.columns:
    solar_power = df["py_sims.solar_farm_0.outputs.power_kw"]
elif "py_sims.solar_farm_0.outputs.power_mw" in df.columns:
    solar_power = df["py_sims.solar_farm_0.outputs.power_mw"] * 1e3
else:
    solar_power = [0] * len(df)
n_wind_turbines = 10
wind_power = df[["hercules_comms.amr_wind.wind_farm_0.turbine_powers.{0:03d}".format(t)
                 for t in range(n_wind_turbines)]].to_numpy().sum(axis=1)
if "py_sims.battery_0.outputs.power" in df.columns:
    battery_power = -df["py_sims.battery_0.outputs.power"] # discharging positive
else:
    battery_power = [0] * len(df)
power_output = df["py_sims.inputs.plant_outputs.electricity"]
time = df["hercules_comms.amr_wind.wind_farm_0.sim_time_s_amr_wind"] / 60 # minutes

# Set plotting aesthetics
wind_col = "C0"
solar_col = "C1"
battery_col = "C2"
plant_col = "C3"

# Plotting power outputs from each technology as well as the total power output (top)
# Plotting the SOC of the battery (bottom)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,5))
ax.plot(time, wind_power/1e3, label="Wind", color=wind_col)
ax.plot(time, solar_power/1e3, label="Solar PV", color=solar_col)
ax.plot(time, battery_power/1e3, label="Battery", color=battery_col)
ax.plot(time, power_output/1e3, label="Plant output", color=plant_col)
ax.plot(power_ref_input['time'] / 60, power_ref_input['plant_power_reference']/1e3,\
            'k--', label="Reference")
ax.set_ylabel("Power [MW]")
ax.set_xlabel("Time [mins]")
ax.grid()
ax.legend(loc="lower right")
ax.set_xlim([0, 5])

# fig.savefig("../../docs/graphics/simple-hybrid-example-plot.png", dpi=300, format="png")

# Plot the battery power and state of charge, if battery component included
if "py_sims.battery_0.outputs.power" in df.columns:
    battery_soc = df["py_sims.battery_0.outputs.soc"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,5))
    ax[0].plot(time, battery_power/1e3, color=battery_col)
    ax[1].plot(time, battery_soc, color=battery_col)
    ax[0].set_ylabel("Battery power [MW]")
    ax[0].grid()
    ax[1].set_ylabel("Battery SOC")
    ax[1].set_xlabel("Time [mins]")
    ax[1].grid()

# Plot the solar data, if solar component included
if "py_sims.solar_farm_0.outputs.power_mw" in df.columns:
    angle_of_incidence = df["py_sims.solar_farm_0.outputs.aoi"]
    direct_normal_irradiance = df["py_sims.solar_farm_0.outputs.dni"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,5))
    ax[0].plot(time, solar_power/1e3, color="C1")
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
                             for t in range(n_wind_turbines)]].to_numpy()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,5))
ax[0].plot(time, wind_power/1e3, color=wind_col)
for i in range (n_wind_turbines):
    ax[1].plot(time, wind_power_individuals[:,i]/1e3, label="WT"+str(i), alpha=0.7, color=wind_col)
ax[0].set_ylabel("Total wind power [MW]")
ax[1].set_ylabel("Individual turbine power [MW]")
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel("Time [mins]")

plt.show()
