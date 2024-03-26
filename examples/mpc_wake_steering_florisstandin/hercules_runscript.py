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
from glob import glob
import pandas as pd

from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.interfaces.hercules_actuator_disk_yaw_interface import HerculesADYawInterface
from whoc.wind_field.WindField import generate_multi_wind_ts

regenerate_wind_field = False

input_dict = load_yaml(sys.argv[1])
wind_case_idx = int(sys.argv[2])

with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

# instantiate wind field if files don't already exist
wind_field_dir = os.path.join('/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/wind_field_data/raw_data')        
wind_field_filenames = glob(f"{wind_field_dir}/amr_case_*.csv")
n_wind_field_cases = 1
if not os.path.exists(wind_field_dir):
    os.makedirs(wind_field_dir)

# TODO make sure this is the same as the wind field from amr_standin_data
# TODO how can we make hercules wait for controller responses
seed = 0
wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
wind_field_config["preview_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
wind_field_config["simulation_sampling_time"] = input_dict["dt"]
wind_field_config["simulation_max_time"] = 600
if not len(wind_field_filenames) or regenerate_wind_field:
    generate_multi_wind_ts(wind_field_config, seed=seed, save_name="amr_")
    wind_field_filenames = [f"amr_case_{i}.csv" for i in range(n_wind_field_cases)]
    regenerate_wind_field = True

# if wind field data exists, get it
wind_field_data = []
if os.path.exists(wind_field_dir):
    for fn in wind_field_filenames:
        wind_field_data.append(pd.read_csv(os.path.join(wind_field_dir, fn), index_col=0))

# true wind disturbance time-series
wind_mag_ts = wind_field_data[wind_case_idx]["FreestreamWindMag"].to_numpy()
wind_dir_ts = wind_field_data[wind_case_idx]["FreestreamWindDir"].to_numpy()

interface = HerculesADYawInterface(input_dict)

seed = 0
controller = MPC(interface, input_dict, 
                 wind_mag_ts=wind_mag_ts, wind_dir_ts=wind_dir_ts, 
                 lut_path=os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv"), 
                 generate_lut=False, 
                 seed=seed,
                 wind_field_config=wind_field_config)

py_sims = PySims(input_dict)

emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])

print("runscript complete.")