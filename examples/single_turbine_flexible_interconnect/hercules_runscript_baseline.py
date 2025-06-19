import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import WindFarmPowerTrackingController
from whoc.interfaces import HerculesADInterface

input_dict = load_yaml(sys.argv[1])
input_dict["output_file"] = "outputs/hercules_output_baseline.csv"
del input_dict["external_data_file"] # Remove wind reference

interface = HerculesADInterface(input_dict)

print("Running closed-loop controller...")
controller = WindFarmPowerTrackingController(interface, input_dict)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("Finished running closed-loop controller.")