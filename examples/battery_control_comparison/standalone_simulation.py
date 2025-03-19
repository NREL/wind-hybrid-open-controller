import matplotlib.pyplot as plt
import numpy as np
from hercules.python_simulators.battery import Battery
from whoc.controllers import BatteryController
from whoc.interfaces import HerculesBatteryInterface

dt = 0.5
input_dict = {
    "dt": dt,
    "py_sims": {
        "battery": {
            "py_sim_type": "LIB",
            "size": 20, # MW
            "energy_capacity": 80, # MWh
            "charge_rate": 20,
            "discharge_rate": 20,
            "max_SOC": 0.9,
            "min_SOC": 0.1,
            "initial_conditions": {"SOC": 0.5}
        }
    },
    "controller": {
        "k_p_max": 1.0, # These will be overwritten during simulation
        "k_p_min": 1.0, # These will be overwritten during simulation
    }
}

save_figs = False

# Make reference input sequence
np.random.seed(0)
hour = round(3600/dt)
reference_input_sequence = np.concatenate((
    np.repeat(0+10000*np.random.randn(round(hour/(12*60))), (12*60)),
))
reference_input_sequence = np.tile(
    np.concatenate(
        (20000*np.ones(round(hour/(12))), 0*np.ones(round(hour/(12))))
    ),
    8
)


# Create some functions for simulating for simplicity
def simulate(soc_0, clipping_thresholds, gain):
    k_batt = gain
    input_dict["controller"]["k_batt"] = k_batt
    input_dict["controller"]["clipping_thresholds"] = clipping_thresholds
    input_dict["py_sims"]["battery"]["initial_conditions"]["SOC"] = soc_0


    # Establish simulation components
    battery=Battery(input_dict["py_sims"]["battery"], dt)
    interface=HerculesBatteryInterface(input_dict)
    controller=BatteryController(interface, input_dict)


    runtime_dict = {
        "time": 0,
        "external_signals": {"plant_power_reference": 0},
        "py_sims": {
            "battery": {"outputs": {"power": 0, "soc": 0, "reject": 0}},
            "inputs": {"battery_signal": 0, "available_power": 20000}
        }
    }

    # Create inputs
    time = np.arange(0, len(reference_input_sequence)*dt, dt)
    power_sequence = np.zeros_like(reference_input_sequence)
    soc_sequence = np.zeros_like(reference_input_sequence)

    # Simulation loop
    for i, (t, r) in enumerate(zip(time, reference_input_sequence)):
        # Update time and reference signals
        runtime_dict["time"] = t
        runtime_dict["external_signals"]["plant_power_reference"] = r

        # Run controller
        runtime_dict = controller.step(runtime_dict)

        # Sign switch since battery assumes positive power is charging
        runtime_dict["py_sims"]["inputs"]["battery_signal"] *= -1

        # Run battery simulator
        runtime_dict["py_sims"]["battery"]["outputs"] = battery.step(runtime_dict)

        # Save outputs (switch sign back)
        power_sequence[i] = -runtime_dict["py_sims"]["battery"]["outputs"]["power"]
        soc_sequence[i] = runtime_dict["py_sims"]["battery"]["outputs"]["soc"]

    return time, power_sequence, soc_sequence

def plot_results_soc(ax, color, time, power_sequence, soc_sequence):
    # Plot
    #fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, power_sequence, color=color,
               label="SOC initial: {:.3f}".format(soc_sequence[0]))
    ax[1].plot(time, soc_sequence, color=color, label="SOC")

def plot_results_gain(ax, color, time, power_sequence, soc_sequence, gain):
    # Plot
    #fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, power_sequence, color=color,
               label="Gain: {:.3f}".format(gain))
    ax[1].plot(time, soc_sequence, color=color, label="SOC")

starting_socs = [0.15, 0.5, 0.85]
colors = ["C0", "C1", "C2"]

clipping_thresholds = [0.1, 0.2, 0.8, 0.9]

# Run simulations and plot
fig, ax = plt.subplots(2,1,sharex=True)
fig.set_size_inches(10,5)
for soc_0, col in zip(starting_socs, colors):
    time, power_sequence, soc_sequence = simulate(soc_0, clipping_thresholds, 0.01)
    plot_results_soc(ax, col, time/60, power_sequence, soc_sequence)

# Add references and plot aesthetics
ax[0].plot(time/60, reference_input_sequence, color="black", linestyle="dashed", label="Reference")
ax[0].set_ylabel("Power [kW]")
ax[0].legend()

ax[1].set_ylabel("SOC [-]")
ax[1].set_xlabel("Time [min]")
ax[1].set_xlim([time[0]/60, time[-1]/60])
ax[0].grid()
ax[1].grid()
ax[0].plot([time[0]/60, time[-1]/60], [20000, 20000], color="black", linestyle="dotted")
ax[0].plot([time[0]/60, time[-1]/60], [-20000, -20000], color="black", linestyle="dotted")

# Add shading for the different clipping regions
ax[1].fill_between(time/60, 0, clipping_thresholds[0], color="black", alpha=0.2, edgecolor=None)
ax[1].fill_between(time/60, clipping_thresholds[0], clipping_thresholds[1], color="black",
                   alpha=0.1, edgecolor=None)
ax[1].fill_between(time/60, clipping_thresholds[2], clipping_thresholds[3], color="black",
                   alpha=0.1, edgecolor=None)
ax[1].fill_between(time/60, clipping_thresholds[3], 1, color="black", alpha=0.2, edgecolor=None)
ax[1].set_ylim([0,1])
if save_figs:
    fig.savefig(
        "../../docs/graphics/battery-soc-clipping.png",
        format="png", bbox_inches="tight", dpi=300
    )

# Demonstrate different gains
reference_input_sequence = reference_input_sequence[:3*round(hour/(12))]
gains = [0.001, 0.01, 0.1]
fig, ax = plt.subplots(2,1,sharex=True)
fig.set_size_inches(10,5)
for gain, col in zip(gains, colors):
    time, power_sequence, soc_sequence = simulate(0.5, clipping_thresholds, gain)
    plot_results_gain(ax, col, time/60, power_sequence, soc_sequence, gain)


# Add references and plot aesthetics
ax[0].plot(time/60, reference_input_sequence, color="black", linestyle="dashed", label="Reference")
ax[0].set_ylabel("Power [kW]")
ax[0].legend()

ax[1].set_ylabel("SOC [-]")
ax[1].set_xlabel("Time [min]")
ax[1].set_xlim([time[0]/60, time[-1]/60])
ax[0].grid()
ax[1].grid()
ax[0].plot([time[0]/60, time[-1]/60], [20000, 20000], color="black", linestyle="dotted")
ax[0].plot([time[0]/60, time[-1]/60], [-20000, -20000], color="black", linestyle="dotted")
if save_figs:
    fig.savefig(
        "../../docs/graphics/battery-varying-gains.png",
        format="png", bbox_inches="tight", dpi=300
    )

plt.show()
