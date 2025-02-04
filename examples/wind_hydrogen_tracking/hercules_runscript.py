import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import (
    WindFarmPowerTrackingController,
    WindHydrogenController,
)
from whoc.interfaces.hercules_wind_hydrogen_interface import HerculesWindHydrogenInterface

# Check that command line arguments are provided
if len(sys.argv) != 3:
    raise Exception("Usage: python hercules_runscript.py <hercules_input_file> <helics_port>")


input_dict = load_yaml(sys.argv[1])
input_dict["output_file"] = "hercules_output_control.csv"

# Set the helics port
helics_port = int(sys.argv[2])
input_dict["hercules_comms"]["helics"]["config"]["helics"]["helicsport"] = helics_port
print(f"Running Hercules with helics_port {helics_port}")


interface = HerculesWindHydrogenInterface(input_dict)

print("Setting up controller.")
wind_controller = WindFarmPowerTrackingController(interface, input_dict)
controller = WindHydrogenController(
    interface,
    input_dict,
    wind_controller=wind_controller
)

print("Establishing simulators.")
py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("Finished running closed-loop controller.")