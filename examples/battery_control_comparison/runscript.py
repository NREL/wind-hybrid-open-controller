import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hercules import HerculesOutput
from hercules.hercules_model import HerculesModel
from hercules.utilities import load_hercules_input
from hercules.utilities_examples import prepare_output_directory
from hycon.controllers import BatteryController, HybridSupervisoryControllerMultiRef
from hycon.interfaces import HerculesInterface

prepare_output_directory()

save_figs = False

# Generate the reference signal to track. We will simplify things by using an
# existing input file.
df = pd.read_csv("../example_inputs/lmp_rt.csv")
df = df.rename(columns={"interval_start_utc": "time_utc"}).drop(columns=["market", "lmp"])
# Create reference that steps up and down each five minutes
reference_input_sequence = np.tile(np.array([20000, 0]), int(len(df) / 2))
df["battery_power_reference"] = reference_input_sequence
# Add end of step info
df["time_utc"] = pd.to_datetime(df["time_utc"])
df_2 = df.copy(deep=True)
df_2["time_utc"] = df_2["time_utc"] + pd.Timedelta(seconds=299)
df = pd.merge(df, df_2, how="outer").sort_values("time_utc").reset_index(drop=True)
df.to_csv("power_reference.csv", index=False)


# Create some functions for simulating for simplicity
def simulate(soc_0, clipping_thresholds, gain):
    h_dict = load_hercules_input("hercules_input.yaml")
    h_dict["battery"]["initial_conditions"]["SOC"] = soc_0

    hmodel = HerculesModel(h_dict)

    # Establish the interface and controller, assign to the Hercules model
    interface = HerculesInterface(hmodel.h_dict)
    battery_controller = BatteryController(
        interface=interface,
        input_dict=hmodel.h_dict,
        controller_parameters={"k_batt": gain, "clipping_thresholds": clipping_thresholds},
    )
    controller = HybridSupervisoryControllerMultiRef(
        battery_controller=battery_controller, interface=interface, input_dict=hmodel.h_dict
    )
    

    hmodel.assign_controller(controller)

    # Run the simulation
    hmodel.run()

    # Extract the signals for plotting
    df_out = HerculesOutput("outputs/hercules_output.h5").df
    power_sequence = df_out["battery.power"].to_numpy()
    soc_sequence = df_out["battery.soc"].to_numpy()
    time = df_out["time"].to_numpy()
    reference_sequence = df_out["external_signals.battery_power_reference"].to_numpy()

    return time, power_sequence, soc_sequence, reference_sequence


def plot_results_soc(ax, color, time, power_sequence, soc_sequence):
    ax[0].plot(
        time, power_sequence, color=color, label="SOC initial: {:.3f}".format(soc_sequence[0])
    )
    ax[1].plot(time, soc_sequence, color=color, label="SOC")


def plot_results_gain(ax, color, time, power_sequence, soc_sequence, gain):
    ax[0].plot(time, power_sequence, color=color, label="Gain: {:.3f}".format(gain))
    ax[1].plot(time, soc_sequence, color=color, label="SOC")


### SOC clipping

# Establish simulation options for demonstrating SOC clipping
starting_socs = [0.15, 0.5, 0.85]
colors = ["C0", "C1", "C2"]
clipping_thresholds = [0.1, 0.2, 0.8, 0.9]

# Run simulations and create plots for SOC clipping
fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(10, 5)
for soc_0, col in zip(starting_socs, colors):
    time, pow, soc, ref = simulate(soc_0, clipping_thresholds, 0.01)
    plot_results_soc(ax, col, time / 60, pow, soc)

# Add references and plot aesthetics
ax[0].plot(time / 60, ref, color="black", linestyle="dashed", label="Reference")
ax[0].set_ylabel("Power [kW]")
ax[0].legend()

ax[1].set_ylabel("SOC [-]")
ax[1].set_xlabel("Time [min]")
ax[1].set_xlim([time[0] / 60, time[-1] / 60])
ax[0].grid()
ax[1].grid()
ax[0].plot([time[0] / 60, time[-1] / 60], [20000, 20000], color="black", linestyle="dotted")
ax[0].plot([time[0] / 60, time[-1] / 60], [-20000, -20000], color="black", linestyle="dotted")

# Add shading for the different clipping regions
ax[1].fill_between(time / 60, 0, clipping_thresholds[0], color="black", alpha=0.2, edgecolor=None)
ax[1].fill_between(
    time / 60,
    clipping_thresholds[0],
    clipping_thresholds[1],
    color="black",
    alpha=0.1,
    edgecolor=None,
)
ax[1].fill_between(
    time / 60,
    clipping_thresholds[2],
    clipping_thresholds[3],
    color="black",
    alpha=0.1,
    edgecolor=None,
)
ax[1].fill_between(time / 60, clipping_thresholds[3], 1, color="black", alpha=0.2, edgecolor=None)
ax[1].set_ylim([0, 1])
if save_figs:
    fig.savefig(
        "../../docs/graphics/battery-soc-clipping.png", format="png", bbox_inches="tight", dpi=300
    )

### k_batt gain

# Demonstrate different gains
gains = [0.001, 0.01, 0.1]
fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(10, 5)
for gain, col in zip(gains, colors):
    time, pow, soc, ref = simulate(0.5, clipping_thresholds, gain)
    plot_results_gain(ax, col, time / 60, pow, soc, gain)

# Add references and plot aesthetics
ax[0].plot(time / 60, ref, color="black", linestyle="dashed", label="Reference")
ax[0].set_ylabel("Power [kW]")
ax[0].legend()

ax[1].set_ylabel("SOC [-]")
ax[1].set_xlabel("Time [min]")
ax[1].set_xlim([0, 15])  # Show only the first 15 to highlight differences
ax[1].set_ylim([0.45, 0.51])
ax[0].grid()
ax[1].grid()
ax[0].plot([time[0] / 60, time[-1] / 60], [20000, 20000], color="black", linestyle="dotted")
ax[0].plot([time[0] / 60, time[-1] / 60], [-20000, -20000], color="black", linestyle="dotted")
if save_figs:
    fig.savefig(
        "../../docs/graphics/battery-varying-gains.png", format="png", bbox_inches="tight", dpi=300
    )

plt.show()
