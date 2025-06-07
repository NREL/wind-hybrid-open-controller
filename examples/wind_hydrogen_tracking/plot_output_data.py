import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the simulation
df = pd.read_csv("outputs/hercules_output_control.csv", index_col=False)
h2_ref_input = pd.read_csv("inputs/hydrogen_ref_signal.csv")

# Extract individual components powers as well as total power
n_wind_turbines = 9
hydrogen_output_rate = df["py_sims.hydrogen_plant_0.outputs.H2_mfr"]
time = df["hercules_comms.amr_wind.wind_farm_0.sim_time_s_amr_wind"]

# Set plotting aesthetics
wind_col = "C0"
h2_col = "k"
power_err_col = "red"
h2_ref_col = "red"

# Plot the hydrogen output 
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,8))
ax[0].plot(
    time/60,
    df["py_sims.inputs.locally_generated_power"]/1e3,
    color=wind_col,
    label="Power generated"
)
ax[0].plot(
    time/60,
    df["py_sims.hydrogen_plant_0.outputs.power_used_kw"]/1e3,
    color=h2_col,
    label="Electrolyzer power consumed",
    linestyle="--"
)
ax[0].fill_between(
    time/60,
    df["py_sims.inputs.plant_outputs.electricity"]/1e3,
    color=power_err_col,
    label="Generation/consumption mismatch"
)
ax[0].set_ylabel("Power [MW]")
ax[0].grid()
ax[0].legend(loc="lower right")

ax[1].plot(
    h2_ref_input["time"]/60,
    h2_ref_input["hydrogen_reference"],
    color=h2_ref_col,
    label="Hydrogen reference",
    linestyle=":"
)
ax[1].plot(time/60, hydrogen_output_rate, color=h2_col, label='Hydrogen output rate')
ax[1].set_ylabel("Hydrogen Production Rate [kg/s]")
ax[1].grid()
ax[1].legend(loc="lower right")
ax[1].set_xlabel("Time [mins]")

# fig.savefig("../../docs/graphics/wind-hydrogen-example-plot.png", dpi=300, format="png")
plt.show()
