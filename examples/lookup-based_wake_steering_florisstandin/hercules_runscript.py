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
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.interfaces.hercules_actuator_disk_interface import HerculesADInterface
from whoc.wind_field.generate_freestream_wind import generate_freestream_wind

n_seeds = 6
regenerate_wind_field = False

input_dict = load_yaml(sys.argv[1])
case_idx = int(sys.argv[2])

with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

amr_standin_data = generate_freestream_wind(".", n_seeds, regenerate_wind_field)[case_idx]

# Load the optimal yaw angle lookup table for controller us
# df_opt = pd.read_csv("lut.csv", index_col=0)
# df_opt_pk = pd.read_pickle("yaw_offsets.pkl", index_col=0)

interface = HerculesADInterface(input_dict)
controller = LookupBasedWakeSteeringController(interface, input_dict, 
                                               lut_path=f"lut_{input_dict['controller']['num_turbines']}.csv", generate_lut=False)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")