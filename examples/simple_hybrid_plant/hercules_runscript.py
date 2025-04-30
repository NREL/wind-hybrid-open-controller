import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import (
    BatteryPassthroughController,
    HybridSupervisoryControllerBaseline,
    SolarPassthroughController,
    WindFarmPowerTrackingController,
)
from whoc.interfaces.hercules_hybrid_actuator_disk_interface import HerculesHybridADInterface

# User options
include_solar = True
include_battery = True

# Load all inputs, remove solar and/or battery as desired
input_dict = load_yaml(sys.argv[1])
if not include_solar:
    del input_dict["py_sims"]["solar_farm_0"]
if not include_battery:
    del input_dict["py_sims"]["battery_0"]

print("Establishing simulators.")
py_sims = PySims(input_dict)

# Establish controllers based on options
interface = HerculesHybridADInterface(input_dict)
print("Setting up controller.")
wind_controller = WindFarmPowerTrackingController(interface, input_dict)
solar_controller = (
    SolarPassthroughController(interface, input_dict) if include_solar
    else None
)
battery_controller = (
    BatteryPassthroughController(interface, input_dict) if include_battery
    else None
)
controller = HybridSupervisoryControllerBaseline(
    interface,
    input_dict,
    wind_controller=wind_controller,
    solar_controller=solar_controller,
    battery_controller=battery_controller
)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("Finished running open-loop controller.")
