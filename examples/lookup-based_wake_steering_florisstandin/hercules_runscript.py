import sys

import pandas as pd
from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import LookupBasedWakeSteeringController
from whoc.design_tools.wake_steering_design import compute_hysteresis_zones
from whoc.interfaces import HerculesADInterface

input_dict = load_yaml(sys.argv[1])

use_hysteresis = False

# Load the optimal yaw angle lookup table for controller use.
df_opt = pd.read_pickle("inputs/yaw_offsets.pkl")

# Optionally, add hysteresis
if use_hysteresis:
    hysteresis_dict = compute_hysteresis_zones(df_opt, min_zone_width=8.0, verbose=True)
else:
    hysteresis_dict = None

interface = HerculesADInterface(input_dict)
controller = LookupBasedWakeSteeringController(
    interface, input_dict,
    df_yaw=df_opt,
    hysteresis_dict=hysteresis_dict,
    verbose=True
)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")