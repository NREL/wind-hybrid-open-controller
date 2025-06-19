import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import (
    BatteryController,
    HybridSupervisoryControllerBaseline,
    WindFarmPowerTrackingController,
)
from whoc.interfaces import HerculesHybridADInterface

input_dict = load_yaml(sys.argv[1])
input_dict["output_file"] = "outputs/hercules_output_with_battery.csv"
input_dict["py_sims"] = {
    "battery_0": {
        "py_sim_type": "LIB",
        "size": 0.5,
        "energy_capacity": 2,
        "charge_rate": 0.5,
        "discharge_rate": 0.5,
        "max_SOC": 0.9,
        "min_SOC": 0.1,
        "initial_conditions": {
            "SOC": 0.88,
        }
    }
}

interface = HerculesHybridADInterface(input_dict)

print("Running closed-loop controller...")
wind_controller = WindFarmPowerTrackingController(interface, input_dict)
battery_controller = BatteryController(interface, input_dict)
hybrid_controller = HybridSupervisoryControllerBaseline(
    interface=interface,
    input_dict=input_dict,
    wind_controller=wind_controller,
    solar_controller=None,
    battery_controller=battery_controller,
)

py_sims = PySims(input_dict)

emulator = Emulator(hybrid_controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("Finished running closed-loop controller.")