# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import sys

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml
from whoc.controllers import (
    HybridSupervisoryControllerSkeleton,
    WindFarmPowerTrackingController,
    BatteryPassthroughController,
    SolarPassthroughController
)
from whoc.interfaces.hercules_hybrid_actuator_disk_interface import HerculesHybridADInterface

input_dict = load_yaml(sys.argv[1])
input_dict["output_file"] = "hercules_output_hybrid.csv"

interface = HerculesHybridADInterface(input_dict)

print("Setting up controller.")
wind_controller = WindFarmPowerTrackingController(interface, input_dict)
solar_controller = SolarPassthroughController(interface, input_dict)
battery_controller = BatteryPassthroughController(interface, input_dict)
controller = HybridSupervisoryControllerSkeleton(
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
