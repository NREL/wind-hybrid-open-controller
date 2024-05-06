import yaml
import os
import numpy as np
from itertools import product

import whoc
from whoc.wind_field.WindField import fit_amr_distribution, get_amr_timeseries

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

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
