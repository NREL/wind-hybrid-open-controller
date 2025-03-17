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
            "initial_conditions": {"SOC": 0.8}
        }
    },
    "controller": {
        "k_p_max": 1.0,#0.1,#1.0,
        "k_p_min": 0.01,#0.1,
    }
}

runtime_dict = {
    "time": 0,
    "external_signals": {"plant_power_reference": 0},
    "py_sims": {
        "battery": {"outputs": {"power": 0, "soc": 0, "reject": 0}},
        "inputs": {"battery_signal": 0, "available_power": 20000}
    }
}

# Establish simulation components
battery=Battery(input_dict["py_sims"]["battery"], dt)
interface=HerculesBatteryInterface(input_dict)
controller=BatteryController(interface, input_dict)

# Create inputs
seg_len = round(100*dt)
reference_input_sequence = np.concatenate((
    10000*np.ones(seg_len),
    5000*np.ones(seg_len),
    10000*np.ones(seg_len),
    0*np.ones(seg_len),
    -10000*np.ones(seg_len),
    -20000*np.ones(seg_len),
    -30000*np.ones(seg_len),
))
time = np.arange(0, len(reference_input_sequence)*dt, dt)
power_sequence = np.zeros_like(reference_input_sequence)
soc_sequence = np.zeros_like(reference_input_sequence)
battery_signal_sequence = np.zeros_like(reference_input_sequence)


# Run the simulation
for i, (t, r) in enumerate(zip(time, reference_input_sequence)):
    # Update time and reference signals
    runtime_dict["time"] = t
    runtime_dict["external_signals"]["plant_power_reference"] = r

    # Run controller
    runtime_dict = controller.step(runtime_dict)

    # Sign switch since battery assumes positive power is charging
    battery_signal_sequence[i] = runtime_dict["py_sims"]["inputs"]["battery_signal"]
    runtime_dict["py_sims"]["inputs"]["battery_signal"] *= -1

    # Run battery simulator
    runtime_dict["py_sims"]["battery"]["outputs"] = battery.step(runtime_dict)

    # Save outputs (switch sign back)
    power_sequence[i] = -runtime_dict["py_sims"]["battery"]["outputs"]["power"]
    soc_sequence[i] = runtime_dict["py_sims"]["battery"]["outputs"]["soc"]

# Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time, power_sequence, color="C0", label="Battery output")
ax[0].plot(time, reference_input_sequence, color="black", linestyle="dashed", label="Reference")
ax[0].plot(time, battery_signal_sequence, color="red", linestyle="dashed", label="Battery signal")
ax[0].set_ylabel("Power [kW]")
ax[0].legend()
ax[1].plot(time, soc_sequence, color="C0", label="SOC")
ax[1].set_ylabel("SOC [-]")
ax[1].set_xlabel("Time [s]")
ax[1].set_xlim([time[0], time[-1]])
ax[0].grid()
ax[1].grid()

plt.show()
