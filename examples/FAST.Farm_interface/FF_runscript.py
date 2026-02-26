import os

import pandas as pd
from hercules.utilities import load_yaml
from hycon.controllers import LookupBasedWakeSteeringController
from hycon.design_tools.wake_steering_design import compute_hysteresis_zones
from hycon.interfaces.rosco_zmq_interface2 import ROSCO_ZMQInterface, ROSCO_Emulator

input_dict = load_yaml(os.path.join("inputs", "hercules_input.yaml"))

use_hysteresis = False

# Load the optimal yaw angle lookup table for controller use.
df_opt = pd.read_pickle(os.path.join("inputs", "yaw_offsets.pkl"))

# Optionally, add hysteresis
if use_hysteresis:
    hysteresis_dict = compute_hysteresis_zones(df_opt, min_zone_width=8.0, verbose=True)
else:
    hysteresis_dict = None

interface = ROSCO_ZMQInterface(input_dict)
controller = LookupBasedWakeSteeringController(
    interface, input_dict, df_yaw=df_opt, hysteresis_dict=hysteresis_dict, verbose=True
)
# interface.addcontroller(controller)

emulator = ROSCO_Emulator(interface, controller)
emulator.startserverandsim()
