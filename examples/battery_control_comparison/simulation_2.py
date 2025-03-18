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
def simulate(soc_0):
    k_batt = 0.1
    input_dict["controller"]["k_batt"] = k_batt
    #input_dict["controller"]["clipping_thresholds"] = [0.0, 0.05, 0.95, 1.0]
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

def plot_results(ax, time, power_sequence, soc_sequence):
    # Plot
    #fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, power_sequence,
               label="SOC initial: {:.3f}".format(soc_sequence[0]))
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()
    ax[1].plot(time, soc_sequence, label="SOC")
    ax[1].set_ylabel("SOC [-]")
    ax[1].set_xlabel("Time [h]")
    ax[1].set_xlim([time[0], time[-1]])
    ax[0].grid()
    ax[1].grid()
    #ax[0].set_title("Initial SOC: {:.3f}".format(soc_sequence[0]))
    ax[0].plot([time[0], time[-1]], [20000, 20000], color="red", linestyle="dotted")
    ax[0].plot([time[0], time[-1]], [-20000, -20000], color="red", linestyle="dotted")

starting_socs = [0.15, 0.5, 0.85]

# Run simulations and plot
fig, ax = plt.subplots(2,1,sharex=True)
for soc_0 in starting_socs:
    time, power_sequence, soc_sequence = simulate(soc_0)
    plot_results(ax, time, power_sequence, soc_sequence)
ax[0].plot(time, reference_input_sequence, color="black", linestyle="dashed", label="Reference")

plt.show()
