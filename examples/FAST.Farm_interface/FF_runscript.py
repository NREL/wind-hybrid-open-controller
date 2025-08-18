import os

import pandas as pd
from hercules.utilities import load_yaml
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.emulators.rosco_zmq_emulator import ROSCO_ZMQEmulator
from whoc.interfaces.rosco_zmq_interface2 import ROSCO_ZMQInterface

input_dict = load_yaml(os.path.join("inputs", "hercules_input.yaml"))

# Load the optimal yaw angle lookup table for controller us
df_opt = pd.read_pickle("yaw_offsets.pkl")

network_address = "tcp://*:5555"
timeout = 600.0
verbose = False

interface = ROSCO_ZMQInterface(network_address, timeout, verbose)
controller = LookupBasedWakeSteeringController(interface, input_dict, df_yaw=df_opt)
emulator = ROSCO_ZMQEmulator(controller,input_dict)

print("runscript complete.")
