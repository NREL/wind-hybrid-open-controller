import sys
import yaml
import os
import numpy as np
from itertools import product

# from hercules.controller_standin import ControllerStandin
from hercules.emulator import Emulator
from hercules.py_sims import PySims
from hercules.utilities import load_yaml

import whoc
from whoc.interfaces.hercules_actuator_disk_interface import HerculesADInterface
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.wind_field.generate_freestream_wind import generate_freestream_wind
from whoc.wind_field.WindField import fit_amr_distribution, get_amr_timeseries

n_seeds = 6
regenerate_wind_field = False
input_dict = load_yaml(sys.argv[1])
case_idx = int(sys.argv[2])

# TODO ensure that time.stop_time in amr_input matches stop_time in hercules input
input_dict["controller"]["floris_input_file"] = "/home/ahenry/toolboxes/whoc_env/wind-hybrid-open-controller/examples/mpc_wake_steering_florisstandin/floris_gch_25.yaml"
input_dict["controller"]["num_turbines"] = 25

print(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"))
with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
    wind_field_config = yaml.safe_load(fp)

amr_standin_data = generate_freestream_wind(".", n_seeds, regenerate_wind_field)[case_idx]
amr_standin_data["time"] += input_dict["hercules_comms"]["helics"]["config"]["starttime"]
print(amr_standin_data["amr_wind_speed"])
print(amr_standin_data["amr_wind_direction"])

    # regenerate mean and cov matrices from the amr precursors driven by abl_forcing_velocity_timetabla
amr_case_folders = ['/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples']
amr_abl_stats_files = ['post_processing/abl_statistics00000.nc']

settled_time, settled_u, settled_v = get_amr_timeseries(amr_case_folders, amr_abl_stats_files)
settled_speed = (settled_u**2 + settled_v**2)**0.5
settled_direction = (360.0 - np.degrees(np.arctan2(settled_u / settled_v))) % 360.0
settled_direction = settled_direction[settled_direction > 180.0] - 360.0

if all(os.path.exists(os.path.join(dirname, filename) for dirname, filename in product(amr_case_folders, amr_abl_stats_files))):
    fit_amr_distribution(wind_field_config["distribution_params_path"].replace("wind_preview_distribution_params.pkl", "wind_preview_distribution_params_amr.pkl"), 
                            case_folders=amr_case_folders, 
                            abl_stats_files=amr_abl_stats_files) # change distribution params based on amr data if it exists

# controller = ControllerStandin(input_dict)
seed = 0
interface = HerculesADInterface(input_dict)
controller = GreedyController(interface, input_dict, 
                #  wind_mag_ts=amr_standin_data["amr_wind_speed"], wind_dir_ts=amr_standin_data["amr_wind_direction"], 
                 wind_mag_ts=settled_speed, wind_dir_ts=settled_direction, 
                 lut_path=os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv"), 
                 generate_lut=False, 
                 seed=seed,
                 wind_field_config=wind_field_config)

py_sims = PySims(input_dict)


emulator = Emulator(controller, py_sims, input_dict)
emulator.run_helics_setup()
emulator.enter_execution(function_targets=[], function_arguments=[[]])
