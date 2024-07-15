import sys

from hercules.controller_standin import ControllerStandin
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


input_dict = load_yaml(sys.argv[1])
# input_dict["output_file"] = "hercules_output_hybrid.csv"

interface = HerculesHybridADInterface(input_dict)

print("Setting up controller.")
wind_controller = WindFarmPowerTrackingController(interface, input_dict)
solar_controller = SolarPassthroughController(interface, input_dict)
battery_controller = BatteryPassthroughController(interface, input_dict)
controller = HybridSupervisoryControllerBaseline(
    interface,
    input_dict,
    wind_controller=wind_controller,
    solar_controller=solar_controller,
    battery_controller=battery_controller
)

print("Establishing simulators.")
py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("Finished running open-loop controller.")
