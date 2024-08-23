import os
import pickle
import pandas as pd
import numpy as np

from glob import glob
import yaml
from itertools import product
import copy
import io
import sys
from functools import partial

from whoc import __file__ as whoc_file
from whoc.wind_field.WindField import plot_ts
from whoc.wind_field.WindField import generate_multi_wind_ts, WindField, write_abl_velocity_timetable, first_ord_filter
from whoc.case_studies.process_case_studies import plot_wind_field_ts
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel

from hercules.utilities import load_yaml

if sys.platform == "linux":
    N_COST_FUNC_TUNINGS = 21
    # if os.getlogin() == "ahenry":
    #     # Kestrel
    #     STORAGE_DIR = "/projects/ssc/ahenry/whoc/floris_case_studies"
    # elif os.getlogin() == "aohe7145":
    #     STORAGE_DIR = "/projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies"
elif sys.platform == "darwin":
    N_COST_FUNC_TUNINGS = 6
    # STORAGE_DIR = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"



# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers": {"seed": {"group": 0, "vals": [0]}, # case_families[0]
                                "dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          },
    "solver_type": {"seed": {"group": 0, "vals": [0]},  # case_families[1]
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 0, "vals": ["stochastic_interval"]}, 
                         "case_names": {"group": 1, "vals": ["SLSQP", "Sequential SLSQP", "Sequential Refine"]},
                              "solver": {"group": 1, "vals": ["slsqp", "sequential_slsqp", "serial_refine"]}
    },
    "wind_preview_type": {"seed": {"group": 0, "vals": [0]}, # case_families[2]
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                        "case_names": {"group": 1, "vals": ["Perfect", "Persistent", 
                                                            "Stochastic Interval 3", "Stochastic Interval 5", "Stochastic Interval 7", 
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100"]},
                        "n_wind_preview_samples": {"group": 1, "vals": [1, 1, 3, 5, 7, 25, 50, 100]},
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent"] + ["stochastic_interval"] * 3 + ["stochastic_sample"] * 3}
                          },
    "warm_start": {"seed": {"group": 0, "vals": [0]},  # case_families[3]
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "case_names": {"group": 1, "vals": ["Greedy", "LUT", "Previous"]},
                   "warm_start": {"group": 1, "vals": ["greedy", "lut", "previous"]}
                   },
    "horizon_length": {"seed": {"group": 0, "vals": [0]},  # case_families[4]
                       "controller_class": {"group": 0, "vals": ["MPC"]},
                       "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [12, 16, 20, 24, 28, 32]]},
                       "n_horizon": {"group": 1, "vals": [12, 16, 20, 24, 28, 32]}
                    },
    "breakdown_robustness":  # case_families[5]
        {"seed": {"group": 0, "vals": [0]},
         "controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
         "dt": {"group": 1, "vals": [30, 5, 5]},
         "use_filtered_wind_dir": {"group": 1, "vals": [False, True, True]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{25}.csv")]},
          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{25}.yaml")]},
        #   "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))]},
          "offline_probability": {"group": 2, "vals": list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))}
        },
    "scalability": {"seed": {"group": 0, "vals": [0]},  # case_families[6]
                    "controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
                    "dt": {"group": 1, "vals": [30, 5, 5]},
                    "use_filtered_wind_dir": {"group": 1, "vals": [False, True, True]},
                    # "case_names": {"group": 2, "vals": ["3 Turbines", "9 Turbines", "25 Turbines"]},
                    "num_turbines": {"group": 2, "vals": [3, 9, 25]},
                    "lut_path": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{nturb}.csv") for nturb in [3, 9, 25]]},
                    "floris_input_file": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3, 9, 25]]
                    },
    },
    "cost_func_tuning": {"seed": {"group": 0, "vals": [0]},  # case_families[7]
                         "controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 3)}" for f in list(np.linspace(0.001, 0.999, N_COST_FUNC_TUNINGS))]},
                         "alpha": {"group": 1, "vals": list(np.logspace(0.001, 0.999, N_COST_FUNC_TUNINGS))}
                          },
    "yaw_offset_study": {"seed": {"group": 0, "vals": [0]},  # case_families[8]
                          "controller_class": {"group": 1, "vals": ["MPC", "MPC", "LookupBasedWakeSteeringController", "MPC"]},
                          "case_names": {"group": 1, "vals":[f"StochasticInterval_1_3turb", f"StochasticInterval_5_3turb", f"LUT_3turb", f"StochasticSample_25_3turb"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 3 + ["stochastic_sample"]},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 5, 1, 25]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]}
    },
     "baseline_plus_controllers": {"seed": {"group": 0, "vals": [0]},
                                "dt": {"group": 1, "vals": [5, 5, 60.0, 60.0, 60.0, 60.0]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy", "MPC_with_Filter", "MPC_without_Filter", "MPC_without_state_cons", "MPC_without_dyn_state_cons"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController", "MPC", "MPC", "MPC", "MPC"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True, True, False, False, False]},
                          },
    "lut": {"seed": {"group": 0, "vals": [0]},
            "case_names": {"group": 0, "vals": ["LUT"]},
            "controller_class": {"group": 0, "vals": ["LookupBasedWakeSteeringController"]},
            "dt": {"group": 0, "vals": [5]},
            "use_filtered_wind_dir": {"group": 0, "vals": [True]},
            "lut_path": {"group": 0, "vals": 
                            [os.path.join(os.path.dirname(whoc_file), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
            "floris_input_file": {"group": 0, "vals": 
                                    [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", f"floris_gch_{3}.yaml")]},
            },
    "slsqp_solver_sweep": {"seed": {"group": 0, "vals": [0]},
                            "controller_class": {"group": 0, "vals": ["MPC"]},
                            "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 20 + ["stochastic_sample"] * 5},
                            "n_wind_preview_samples": {"group": 1, "vals": [1, 3, 5, 7, 9] * 4 + [25, 50, 100, 200, 300]},
                            "diff_type": {"group": 1, "vals": ["chain_cd"] * 5 + ["chain_fd"] * 5 + ["direct_cd"] * 5 + ["direct_fd"] * 5 + ["none"] * 5},
                            "nu": {"group": 2, "vals": [10**x for x in range(-3, -1, 1)]},
                            "use_filtered_wind_dir": {"group": 4, "vals": [False]},
                             "dt": {"group": 5, "vals": [15, 30, 60]},
                             "n_horizon": {"group": 5, "vals": [int(6 * 60 // 15), int(6 * 60 // 30), int(6 * 60 // 60)]},  
                             "alpha": {"group": 0, "vals": [1.0]},
                            "solver": {"group": 0, "vals": ["slsqp"]},
                            "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                            # "alpha": {"group": 3, "vals": list(np.linspace(0.005, 0.995, 12))}
                          },
    "generate_sample_figures": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval", "stochastic_sample"]},
                             "n_wind_preview_samples": {"group": 1, "vals": [5, 500]},
                            #  "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent", "stochastic_interval", "stochastic_interval", "stochastic_sample"]},
                            #  "n_wind_preview_samples": {"group": 1, "vals": [1, 1, 1, 9, 500]},
                            #  "nu": {"group": 2, "vals": [0.001, 0.01]},
                            # "diff_type": {"group": 3, "vals": ["custom_cd", "custom_fd"]},
                             "alpha": {"group": 0, "vals": [0.5]},
                             "solver": {"group": 0, "vals": ["slsqp"]},
                             "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                            #  "decay_type": {"group": 2, "vals": ["exp"] * 4 + ["cosine"] * 4 + ["linear"] * 4 + ["zero", "none"]},
                            # "decay_const": {"group": 2, "vals": [31, 45, 60, 90] * 3 + [90, 90]},
                            # "decay_all": {"group": 3, "vals": ["True", "False"]},
                            # "clip_value": {"group": 4, "vals": [30, 44]},
    },
    "test_gradient": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 0, "vals": ["stochastic_interval"]}, #, "stochastic_sample"]},
                             "n_wind_preview_samples": {"group": 0, "vals": [5]},#9, 100]},
                            #  "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent", "stochastic_interval", "stochastic_interval", "stochastic_sample"]},
                            #  "n_wind_preview_samples": {"group": 1, "vals": [1, 1, 1, 9, 500]},
                            #  "nu": {"group": 2, "vals": [0.001, 0.01]},
                              "diff_type": {"group": 2, "vals": ["chain_cd", "chain_fd", "direct_cd", "direct_fd"]},
                              "dt": {"group": 0, "vals": [30]},
                              "n_horizon": {"group": 0, "vals": [3]},  
                             "alpha": {"group": 0, "vals": [1]},
                             "solver": {"group": 0, "vals": ["slsqp"]},
                             "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                            #  "decay_type": {"group": 2, "vals": ["exp"] * 4 + ["cosine"] * 4 + ["linear"] * 4 + ["zero", "none"]},
                            # "decay_const": {"group": 2, "vals": [31, 45, 60, 90] * 3 + [90, 90]},
                            # "decay_all": {"group": 3, "vals": ["True", "False"]},
                            # "clip_value": {"group": 4, "vals": [30, 44]},
    },
    "sequential_slsqp_solver": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": ["Sequential SLSQP"]},
                          "solver": {"group": 1, "vals": ["sequential_slsqp"]}
                          },
    "serial_refine_solver": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 0, "vals": ["Sequential Refine"]},
                           "solver": {"group": 0, "vals": ["serial_refine"]}
                          },
     "solver_type_test": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval", "stochastic_sample"]}, 
                              "solver": {"group": 2, "vals": ["serial_refine"]} #, "sequential_slsqp", "slsqp"]}
    },
    "stochastic_preview_type": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": [f"StochasticInterval_{d}" for d in [1, 3, 5, 7]] + [f"StochasticSample_{d}" for d in [25, 50, 100]]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 4 + ["stochastic_sample"] * 3},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 3, 5, 7] + [25, 50, 100]}
    },
    "stochastic_preview_type_small": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": [f"StochasticSample_{d}" for d in [100, 250, 500, 1000]] + [f"StochasticInterval_{d}" for d in [3, 5, 9, 15]]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_sample"] * 4 + ["stochastic_interval"] * 4},
                           "n_wind_preview_samples": {"group": 1, "vals": [100, 250, 500, 1000] + [3, 5, 9, 15]},
                           "nu": {"group": 0, "vals": [0.01]},
                           "alpha": {"group": 0, "vals": [1.0]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                          },
    "persistent_preview_type": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": ["Persistent"]},
                          "wind_preview_type": {"group": 1, "vals": ["persistent"]}
                          },
    "perfect_preview_type": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": ["Perfect"]},
                          "wind_preview_type": {"group": 1, "vals": ["perfect"]}
                          },
    "lut_warm_start": {"seed": {"group": 0, "vals": [0]},
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "case_names": {"group": 0, "vals": ["LUT"]},
                   "warm_start": {"group": 0, "vals": ["lut"]}
                   },

    "test_nu_preview": { "seed": {"group": 0, "vals": [0]},
                         "controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": ["Perfect", "Stat-Based", "Sample-Based"]},
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "stochastic", "stochastic"]},
                         "alpha": {"group": 0, "vals": [1.0]},
                         "nu": {"group": 0, "vals": [0.01]},
                         "num_turbines": {"group": 0, "vals": [3]},
                         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{nturb}.csv") for nturb in [3]]},
                         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3]]},
                          "n_wind_preview_samples": {"group": 1, "vals": [1, 5, 1]}
                          }
}

def convert_str(val):
    def try_type(val, data_type):
        try:
            data_type(val)
            return True
        except:
            return False
#        return isinstance(val, data_type)  ### this doesn't work b/c of numpy data types; they're not instances of base types
    def try_list(val):
        try:
            val[0]
            return True
        except:
            return False

    if try_type(val, int) and int(val) == float(val):
        return int(val)
    elif try_type(val, float):
        return float(val)
    elif val=='True':
        return True
    elif val=='False':
        return False
    # elif type(val)!=str and try_list(val):
    #     return ", ".join(['{:}'.format(i) for i in val])
    else:
        return val

def case_naming(n_cases, namebase=None):
    # case naming
    case_name = [('%d'%i).zfill(len('%d'%(n_cases-1))) for i in range(n_cases)]
    if namebase:
        case_name = [namebase+'_'+caseid for caseid in case_name]

    return case_name

def CaseGen_General(case_inputs, namebase=''):
    """ Cartesian product to enumerate over all combinations of set of variables that are changed together"""

    # put case dict into lists
    change_vars = sorted(case_inputs.keys())
    change_vals = [case_inputs[var]['vals'] for var in change_vars]
    change_group = [case_inputs[var]['group'] for var in change_vars]

    # find number of groups and length of groups
    group_set = list(set(change_group))
    group_len = [len(change_vals[change_group.index(i)]) for i in group_set]

    # case matrix, as indices
    group_idx = [range(n) for n in group_len]
    matrix_idx = list(product(*group_idx))

    # index of each group
    matrix_group_idx = [np.where([group_i == group_j for group_j in change_group])[0].tolist() for group_i in group_set]

    # build final matrix of variable values
    matrix_out = []
    for i, row in enumerate(matrix_idx):
        row_out = [None]*len(change_vars)
        for j, val in enumerate(row):
            for g in matrix_group_idx[j]:
                row_out[g] = change_vals[g][val]
        matrix_out.append(row_out)
    try:
        matrix_out = np.asarray(matrix_out, dtype=str)
    except:
        matrix_out = np.asarray(matrix_out)
    n_cases = np.shape(matrix_out)[0]

    # case naming
    case_name = case_naming(n_cases, namebase=namebase)

    case_list = []
    for i in range(n_cases):
        case_list_i = {}
        for j, var in enumerate(change_vars):
            case_list_i[var] = convert_str(matrix_out[i,j])
        case_list.append(case_list_i)

    return case_list, case_name

def initialize_simulations(case_study_keys, regenerate_lut, regenerate_wind_field, n_seeds, stoptime, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_dict = load_yaml(os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml"))

    with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(save_dir, 'wind_field_data/raw_data')
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime

    if "slsqp_solver_sweep" not in case_studies or "dt" not in case_studies["slsqp_solver_sweep"]:
        max_controller_dt = input_dict["controller"]["dt"]
    else:
        max_controller_dt = max(case_studies["slsqp_solver_sweep"]["dt"]["vals"])
    
    if "horizon_length" not in case_studies or "n_horizon" not in case_studies["horizon_length"]:
        max_n_horizon = input_dict["controller"]["n_horizon"]
    else:
        max_n_horizon = max(case_studies["horizon_length"]["n_horizon"]["vals"])

    # wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
    wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
    wind_field_config["preview_dt"] = int(max_controller_dt / input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = input_dict["dt"]
    
    # wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / input_dict["dt"]) \
        + max_n_horizon * int(max_controller_dt/ input_dict["dt"])
    wind_field_config["n_samples_per_init_seed"] = 1
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["distribution_params_path"] = os.path.join(save_dir, "wind_field_data", "wind_preview_distribution_params.pkl")  
    wind_field_config["time_series_dt"] = 1
    
    # TODO check that wind field has same dt or interpolate...
    seed = 0
    if len(wind_field_filenames) < n_seeds or regenerate_wind_field:
        n_seeds = 6
        print("regenerating wind fields")
        wind_field_config["regenerate_distribution_params"] = True # set to True to regenerate from constructed mean and covaraicne
        full_wf = WindField(**wind_field_config)
        if not os.path.exists(wind_field_dir):
            os.makedirs(wind_field_dir)
        wind_field_data = generate_multi_wind_ts(full_wf, wind_field_dir, init_seeds=[seed + i for i in range(n_seeds)])
        write_abl_velocity_timetable([wfd.df for wfd in wind_field_data], wind_field_dir) # then use these timetables in amr precursor
        # write_abl_velocity_timetable(wind_field_data, wind_field_dir) # then use these timetables in amr precursor
        lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["dt"])
        plot_wind_field_ts(wind_field_data[0].df, wind_field_dir, filter_func=partial(first_ord_filter, alpha=lpf_alpha))
        plot_ts(pd.concat([wfd.df for wfd in wind_field_data]), wind_field_dir)
        wind_field_filenames = [os.path.join(wind_field_dir, f"case_{i}.csv") for i in range(n_seeds)]
        regenerate_wind_field = True
    
    # if wind field data exists, get it
    WIND_TYPE = "stochastic"
    wind_field_data = []
    if os.path.exists(wind_field_dir):
        for f, fn in enumerate(wind_field_filenames):
            wind_field_data.append(pd.read_csv(fn, index_col=0))
            
            if WIND_TYPE == "step":
                # n_rows = len(wind_field_data[-1].index)
                wind_field_data[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                wind_field_data[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                wind_field_data[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                wind_field_data[-1].loc[45:, f"FreestreamWindDir"] = 270.0
    
    # write_abl_velocity_timetable(wind_field_data, wind_field_dir)
    
    # true wind disturbance time-series
    #plot_wind_field_ts(pd.concat(wind_field_data), os.path.join(wind_field_fig_dir, "seeds.png"))
    wind_mag_ts = [wind_field_data[case_idx]["FreestreamWindMag"].to_numpy() for case_idx in range(n_seeds)]
    wind_dir_ts = [wind_field_data[case_idx]["FreestreamWindDir"].to_numpy() for case_idx in range(n_seeds)]

    # regenerate floris lookup tables for all wind farms included
    if regenerate_lut:
        lut_input_dict = dict(input_dict)
        for lut_path, floris_input_file in zip(case_studies["scalability"]["lut_path"]["vals"], 
                                                        case_studies["scalability"]["floris_input_file"]["vals"]):
            fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                            offline_probability=input_dict["controller"]["offline_probability"],
                                            dt=input_dict["dt"],
                                            yaw_rate=input_dict["controller"]["yaw_rate"],
                                            config_path=floris_input_file)
            lut_input_dict["controller"]["lut_path"] = lut_path
            lut_input_dict["controller"]["generate_lut"] = True
            ctrl_lut = LookupBasedWakeSteeringController(fi, lut_input_dict, wind_mag_ts=wind_mag_ts[0], wind_dir_ts=wind_dir_ts[0])

        input_dict["controller"]["generate_lut"] = False

    assert np.all([np.isclose(wind_field_data[case_idx]["Time"].iloc[1] - wind_field_data[case_idx]["Time"].iloc[0], input_dict["dt"]) for case_idx in range(n_seeds)]), "sampling time of wind field should be equal to simulation sampling time"

    input_dicts = []
    case_lists = []
    case_name_lists = []
    n_cases_list = []

    for case_study_key in case_study_keys:
        case_list, case_names = CaseGen_General(case_studies[case_study_key], namebase=case_study_key)
        case_lists = case_lists + case_list
        case_name_lists = case_name_lists + case_names
        n_cases_list.append(len(case_list))

        # make save directory
        results_dir = os.path.join(save_dir, case_study_key)
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Load default settings and make copies
        start_case_idx = len(input_dicts)
        input_dicts = input_dicts + [copy.deepcopy(input_dict) for i in range(len(case_list))]

        # make adjustements based on case study
        for c, case in enumerate(case_list):
            for property_name, property_value in case.items():
                if type(property_value) is np.str_:
                    input_dicts[start_case_idx + c]["controller"][property_name] = str(property_value)
                else:
                    input_dicts[start_case_idx + c]["controller"][property_name] = property_value
                    
            fn = f'input_config_case_{"_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_ or type(val) is bool) else np.round(val, 6)}" for key, val in case.items() if key not in ["wind_case_idx", "seed", "floris_input_file", "lut_path"]]) if "case_names" not in case else case["case_names"]}.yaml'.replace("/", "_")
            
            with io.open(os.path.join(results_dir, fn), 'w', encoding='utf8') as fp:
                yaml.dump(input_dicts[start_case_idx + c], fp, default_flow_style=False, allow_unicode=True)

    # instantiate controller and run_simulations simulation
    wind_field_config["regenerate_distribution_params"] = False

    with open(os.path.join(save_dir, "init_simulations.pkl"), "wb") as fp:
        pickle.dump({"case_lists": case_lists, "case_name_lists": case_name_lists, "input_dicts": input_dicts, "wind_field_config": wind_field_config,
                     "wind_mag_ts": wind_mag_ts, "wind_dir_ts": wind_dir_ts}, fp)

    return case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts

# 0, 1, 2, 3, 6
case_families = ["baseline_controllers", "solver_type",
                    "wind_preview_type", "warm_start", 
                    "horizon_length", "breakdown_robustness",
                    "scalability", "cost_func_tuning", 
                    "yaw_offset_study", "slsqp_solver_sweep",
                    "stochastic_preview_type", "stochastic_preview_type_small",
                    "perfect_preview_type", "generate_sample_figures",
                    "test_nu_preview", "serial_refine_solver", 
                    "sequential_slsqp_solver", "lut",
                    "solver_type_test", "test_gradient"]
