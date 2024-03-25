import sys
import yaml
import os
import glob
import pandas as pd

# from hercules.controller_standin import ControllerStandin
from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.wind_field.WindField import generate_multi_wind_ts

regenerate_wind_field = False

input_dict = load_yaml(sys.argv[1])

with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

# instantiate wind field if files don't already exist
wind_field_dir = os.path.join('/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/wind_field_data/raw_data')        
wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
n_wind_field_cases = 1
if not os.path.exists(wind_field_dir):
    os.makedirs(wind_field_dir)

# TODO make sure this is the same as the wind field from amr_standin_data
# TODO how can we make hercules wait for controller response?s
seed = 0
wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * input_dict["controller"]["dt"]
wind_field_config["preview_dt"] = input_dict["controller"]["dt"]
if not len(wind_field_filenames) or regenerate_wind_field:
    generate_multi_wind_ts(wind_field_config, seed=seed)
    wind_field_filenames = [f"case_{i}.csv" for i in range(n_wind_field_cases)]
    regenerate_wind_field = True

# if wind field data exists, get it
WIND_TYPE = "stochastic"
wind_field_data = []
if os.path.exists(wind_field_dir):
    for fn in wind_field_filenames:
        wind_field_data.append(pd.read_csv(os.path.join(wind_field_dir, fn)))

# true wind disturbance time-series
case_idx = 0

wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()


# controller = ControllerStandin(input_dict)
seed = 0
controller = MPC(interface, input_dict, 
                 wind_mag_ts=wind_mag_ts, wind_dir_ts=wind_dir_ts, 
                 lut_path=input_dict["controller"]["lut_path"], 
                 generate_lut=input_dict["controller"]["generate_lut"], 
                 seed=seed,
                 wind_field_config=wind_field_config)

py_sims = PySims(input_dict)


emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])
