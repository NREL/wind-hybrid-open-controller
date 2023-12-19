import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers.wake_steering_actuatordisk_standin import WakeSteeringADStandin
from whoc.interfaces.hercules_actuator_disk_yaw_interface import HerculesADYawInterface

input_dict = load_yaml(sys.argv[1])

interface = HerculesADYawInterface(input_dict)
controller = WakeSteeringADStandin(interface, input_dict)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")
