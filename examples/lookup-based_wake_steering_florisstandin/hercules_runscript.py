import sys

import pandas as pd
from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.interfaces.hercules_actuator_disk_interface import HerculesADInterface

input_dict = load_yaml(sys.argv[1])

# Load the optimal yaw angle lookup table for controller us
df_opt = pd.read_pickle("yaw_offsets.pkl")

interface = HerculesADInterface(input_dict)
controller = LookupBasedWakeSteeringController(interface, input_dict, df_yaw=df_opt)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")