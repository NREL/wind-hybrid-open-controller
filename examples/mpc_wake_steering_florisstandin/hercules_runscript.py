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
import yaml
import os

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.interfaces.hercules_actuator_disk_yaw_interface import HerculesADYawInterface
from whoc.wind_field.generate_freestream_wind import generate_freestream_wind

regenerate_wind_field = False
case_idx = 0

input_dict = load_yaml(sys.argv[1])

with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

amr_standin_data = generate_freestream_wind(".", regenerate_wind_field)[case_idx]

interface = HerculesADYawInterface(input_dict)

seed = 0
controller = MPC(interface, input_dict, 
                 wind_mag_ts=amr_standin_data["amr_wind_speed"], wind_dir_ts=amr_standin_data["amr_wind_direction"], 
                 lut_path=os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv"), 
                 generate_lut=False, 
                 seed=seed,
                 wind_field_config=wind_field_config)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")