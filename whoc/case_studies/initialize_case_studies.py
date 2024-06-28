# TODO HIGH 1) Profile stochastic_interval, stochastic_sample and check gradients X
# TODO HIGH 2) Check debug version of all case studies on local machine
# TODO HIGH 3) Run full version of all case studies on Kestrel
# TODO HIGH 4) Generate figures and table for FLORIS results

# TODO MEDIUM 1) Generate 6 x AMR precursors, download outputs, and fit to Gaussian curve
# TODO MEDIUM 2) Run AMR + Greedy, LUT controllers
# TODO MEDIUM 3) Generate AMR + Greedy, LUT controller results
# TODO MEDIUM 4) Run AMR + MPC controller and generate results

import os
# from mpi4py import MPI
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

# from warnings import simplefilter
# simplefilter('error')
# N_COST_FUNC_TUNINGS = 21
N_COST_FUNC_TUNINGS = 6 # TODO HIGH increase for non-debug mode

if sys.platform == "linux":
    STORAGE_DIR = "/projects/ssc/ahenry/whoc/floris_case_studies"
elif sys.platform == "darwin":
    STORAGE_DIR = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers": {"seed": {"group": 0, "vals": [0]},
                            #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                                "dt": {"group": 1, "vals": [5, 5, 60.0, 60.0, 60.0, 60.0]},
                                # "dt": {"group": 1, "vals": [0.5, 0.5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy", "MPC_with_Filter", "MPC_without_Filter", "MPC_without_state_cons", "MPC_without_dyn_state_cons"]},
                                # "case_names": {"group": 1, "vals": ["MPC_with_Filter", "MPC_without_Filter"]},
                                # "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                # "controller_class": {"group": 1, "vals": ["MPC", "MPC"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController", "MPC", "MPC", "MPC", "MPC"]},
                                # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True, True, False, False, False]},
                                # "use_state_cons": {"group": 1, "vals": [True, True, True, True, False, True]},
                                # "use_dyn_state_cons": {"group": 1, "vals": [True, True, True, True, True, False]},
                                # "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                # "use_state_cons": {"group": 1, "vals": [True]},
                                # "use_dyn_state_cons": {"group": 1, "vals": [False]},
                                # "use_filt": {"group": 1, "vals": [True, True]}},
                          },
    "greedy": {"seed": {"group": 0, "vals": [0]},
            #    "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                                "case_names": {"group": 1, "vals": ["Greedy"]},
                                "controller_class": {"group": 1, "vals": ["GreedyController"]}
                          },
    "slsqp_solver_sweep": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                            "nu": {"group": 1, "vals": [10**x for x in range(-5, 0, 1)]},
                            "n_wind_preview_samples": {"group": 2, "vals": [3, 5, 7, 9]},
                            "alpha": {"group": 3, "vals": list(np.linspace(0.005, 0.995, 11))},
                        # "nu": {"group": 1, "vals": [10**x for x in [0.1]]},
                        # "n_wind_preview_samples": {"group": 2, "vals": [9]},
                        #  "alpha": {"group": 3, "vals": [0.1]},
                        #   "case_names": {"group": 3, "vals": [f"SLSQP_nu_{np.round(nu, 4)}_nsamples_{n_samples}_alpha_{alpha}" 
                        #                                       for nu in list(np.logspace(-4, 0, 5))
                        #                                       for n_samples in [10, 25, 50, 100, 200]
                        #                                       for alpha in list(np.linspace(0, 1.0, 11))]},
                           
                           
                           "solver": {"group": 0, "vals": ["slsqp"]}
                          },
    "slsqp_solver_sweep_small": {"seed": {"group": 0, "vals": [0]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 3 + ["stochastic_sample"] * 3},
                             "n_wind_preview_samples": {"group": 1, "vals": [3, 5, 7, 25, 50, 100]},
                             "nu": {"group": 2, "vals": [10**x for x in range(-3, 0, 1)]},
                             "alpha": {"group": 3, "vals": list(np.linspace(0.005, 0.995, 3))},
                        # "nu": {"group": 1, "vals": [10**x for x in [0.1]]},
                        # "n_wind_preview_samples": {"group": 2, "vals": [9]},
                        #  "alpha": {"group": 3, "vals": [0.1]},
                        #   "case_names": {"group": 3, "vals": [f"SLSQP_nu_{np.round(nu, 4)}_nsamples_{n_samples}_alpha_{alpha}" 
                        #                                       for nu in list(np.logspace(-4, 0, 5))
                        #                                       for n_samples in [10, 25, 50, 100, 200]
                        #                                       for alpha in list(np.linspace(0, 1.0, 11))]},
                           
                           
                           "solver": {"group": 0, "vals": ["slsqp"]}
                          },
    "sequential_slsqp_solver": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "n_horizon": {"group": 0, "vals": [5]}, 
                          "case_names": {"group": 1, "vals": ["Sequential SLSQP"]},
                          "solver": {"group": 1, "vals": ["sequential_slsqp"]}
                          },
    "serial_refine_solver": {"seed": {"group": 0, "vals": [0]},
                            #  "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": ["Sequential Refine"]},
                           "solver": {"group": 1, "vals": ["serial_refine"]}
                          },
    "solver_type": {"seed": {"group": 0, "vals": [0]},
                    # "wind_case_idx": {"group": 2, "vals": [i for i in range(N_SEEDS)]},
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "wind_preview_type": {"group": 0, "vals": ["stochastic"]}, 
                            #  "warm_start": {"group": 0, "vals": ["greedy"]}, 
                        #   "case_names": {"group": 1, "vals": ["ZSGD", "SLSQP", "Sequential SLSQP", "Sequential Refine"]},
                         "case_names": {"group": 1, "vals": ["SLSQP", "Sequential SLSQP", "Sequential Refine"]},
                        #    "solver": {"group": 1, "vals": ["zsgd", "slsqp", "sequential_slsqp", "serial_refine"]},
                              "solver": {"group": 1, "vals": ["slsqp", "sequential_slsqp", "serial_refine"]}
    },
    "yaw_offset_study": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 1, "vals": ["MPC", "MPC", "LookupBasedWakeSteeringController"]},
                          "case_names": {"group": 1, "vals":[f"StochasticInterval_1_3turb", f"StochasticInterval_3turb", f"LUT_3turb"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 3},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 3, 1]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{3}.csv")]}
    },
    "stochastic_preview_type": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": [f"StochasticInterval_{d}" for d in [1, 3, 5, 7]] + [f"StochasticSample_{d}" for d in [25, 50, 100]]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval"] * 4 + ["stochastic_sample"] * 3},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 3, 5, 7] + [25, 50, 100]}
    },
    "stochastic_preview_type_small": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                          "case_names": {"group": 1, "vals": [f"StochasticSample_{d}" for d in [100]] + [f"StochasticInterval_{d}" for d in [3]]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_sample"] * 1 + ["stochastic_interval"] * 1},
                           "n_wind_preview_samples": {"group": 1, "vals": [500] + [3]},
                           "nu": {"group": 0, "vals": [0.01]},
                           "alpha": {"group": 0, "vals": [0.9]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{3}.csv")]},
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
    "wind_preview_type": {"seed": {"group": 0, "vals": [0]},
                          "controller_class": {"group": 0, "vals": ["MPC"]},
                        "case_names": {"group": 1, "vals": ["Perfect", "Persistent"] + ["Stochastic Interval"] * 3 + ["Stochastic Sample"] * 3},
                        "n_wind_preview_samples": {"group": 1, "vals": [1, 1, 3, 5, 7, 25, 50, 100]},
                        #  "case_names": {"group": 1, "vals": ["Stochastic"]},
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent"] + ["stochastic_interval"] * 3 + ["stochastic_sample"] * 3}
                        #   "wind_preview_type": {"group": 1, "vals": ["stochastic"]}
                          },
    "lut_warm_start": {"seed": {"group": 0, "vals": [0]},
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "case_names": {"group": 0, "vals": ["LUT"]},
                   "warm_start": {"group": 0, "vals": ["lut"]}
                   },
    "warm_start": {"seed": {"group": 0, "vals": [0]},
                   "controller_class": {"group": 0, "vals": ["MPC"]},
                   "wind_preview_type": {"group": 0, "vals": ["stochastic"]},
                   "case_names": {"group": 1, "vals": ["Greedy", "LUT", "Previous"]},
                   "warm_start": {"group": 1, "vals": ["greedy", "lut", "previous"]}
                   },
    "cost_func_tuning": {"seed": {"group": 0, "vals": [0]},
                         "controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 2)}" for f in list(np.linspace(0.001, 0.999, N_COST_FUNC_TUNINGS))]},
                         "alpha": {"group": 1, "vals": list(np.logspace(0.001, 0.999, N_COST_FUNC_TUNINGS))}
                          },
    "breakdown_robustness": 
        {"seed": {"group": 0, "vals": [0]},
         "num_turbines": {"group": 0, "vals": [25]}, 
         "controller_class": {"group": 0, "vals": ["MPC"]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{25}.csv")]},
          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_25.yaml")]},
          "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))]},
          "offline_probability": {"group": 1, "vals": list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))}
        },
    "scalability": {"seed": {"group": 0, "vals": [0]},
                    "controller_class": {"group": 0, "vals": ["MPC"]},
                    "case_names": {"group": 1, "vals": ["3 Turbines", "9 Turbines", "25 Turbines", "100 Turbines"]},
                    "num_turbines": {"group": 1, "vals": [3, 9, 25, 100]},
                    "lut_path": {"group": 1, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{nturb}.csv") for nturb in [3, 9, 25, 100]]},
                    "floris_input_file": {"group": 1, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3, 9, 25, 100]]},
    },
    "horizon_length": {"seed": {"group": 0, "vals": [0]},
                       "controller_class": {"group": 0, "vals": ["MPC"]},
                       "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [6, 8, 10, 12, 14]]},
                       "n_horizon": {"group": 1, "vals": [6, 8, 10, 12, 14]}
                    },
    "test_nu_preview": { "seed": {"group": 0, "vals": [0]},
                         "controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": ["Perfect", "Stat-Based", "Sample-Based"]},
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "stochastic", "stochastic"]},
                         "alpha": {"group": 0, "vals": [1.0]},
                         "nu": {"group": 0, "vals": [0.01]},
                         "num_turbines": {"group": 0, "vals": [3]},
                         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{nturb}.csv") for nturb in [3]]},
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

def initialize_simulations(case_study_keys, regenerate_lut, regenerate_wind_field, n_seeds, debug):

    input_dict = load_yaml(os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml"))

    with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
        wind_field_config = yaml.safe_load(fp)

    # instantiate wind field if files don't already exist
    wind_field_dir = os.path.join(STORAGE_DIR, 'wind_field_data/raw_data')
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    if not os.path.exists(wind_field_dir):
        os.makedirs(wind_field_dir)

    if debug:
        input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = 480
    else:
        input_dict["hercules_comms"]["helics"]["config"]["stoptime"] = 3600

    # wind_field_config["simulation_max_time"] = input_dict["hercules_comms"]["helics"]["config"]["stoptime"]
    wind_field_config["num_turbines"] = input_dict["controller"]["num_turbines"]
    wind_field_config["preview_dt"] = int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["simulation_sampling_time"] = input_dict["dt"]
    
    # wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / input_dict["dt"]) \
        + max(case_studies["horizon_length"]["n_horizon"]["vals"]) * int(input_dict["controller"]["dt"] / input_dict["dt"])
    wind_field_config["n_samples_per_init_seed"] = 1
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["distribution_params_path"] = os.path.join(STORAGE_DIR, "wind_field_data", "wind_preview_distribution_params.pkl")  
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
        lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["dt"])
        plot_wind_field_ts(wind_field_data[0].df, wind_field_dir, filter_func=partial(first_ord_filter, alpha=lpf_alpha))
        plot_ts(wind_field_data[0].df, wind_field_dir)
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
        results_dir = os.path.join(STORAGE_DIR, case_study_key)
        
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
                    
            fn = f'input_config_case_{"_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_) else np.round(val, 5)}" for key, val in case.items() if key not in ["wind_case_idx", "seed"]]) if "case_names" not in case else case["case_names"]}.yaml'.replace("/", "_")
            
            with io.open(os.path.join(results_dir, fn), 'w', encoding='utf8') as fp:
                yaml.dump(input_dicts[start_case_idx + c], fp, default_flow_style=False, allow_unicode=True)

    # instantiate controller and run_simulations simulation
    # wind_field_config["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])
    # wind_field_config["n_samples_per_init_seed"] = input_dict["controller"]["n_wind_preview_samples"]
    wind_field_config["regenerate_distribution_params"] = False
    wind_field_config["time_series_dt"] = int(input_dict["controller"]["dt"] // input_dict["dt"])

    with open(os.path.join(STORAGE_DIR, "init_simulations.pkl"), "wb") as fp:
        pickle.dump({"case_lists": case_lists, "case_name_lists": case_name_lists, "input_dicts": input_dicts, "wind_field_config": wind_field_config,
                     "wind_mag_ts": wind_mag_ts, "wind_dir_ts": wind_dir_ts}, fp)

    return case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts
# 0, 1, 2, 3, 4, 5, 6, 7
# 0, 2, 4, 7, 8, 9
# 0, 2, 10
# 8
# 1, 2, 3, 5, 6 7
# 0, 2, 11
# 0, 8
case_families = ["baseline_controllers", "solver_type",
                    "wind_preview_type", "warm_start", 
                    "horizon_length", "breakdown_robustness",
                    "scalability", "cost_func_tuning", 
                    "stochastic_preview_type", "stochastic_preview_type_small",
                    "perfect_preview_type", "slsqp_solver_sweep_small",
                    "test_nu_preview", "serial_refine_solver", 
                    "sequential_slsqp_solver", "yaw_offset_study"]
    
if __name__ == "__main__":
    REGENERATE_WIND_FIELD = True
    REGENERATE_LUT = False
    # TODO replace these with proper command line args
    # comm_rank = MPI.COMM_WORLD.Get_rank()
    if sys.argv[2].lower() == "mpi":
        MULTI = "mpi"
    else:
        MULTI = "cf"

    PARALLEL = sys.argv[3].lower() == "parallel"
    DEBUG = sys.argv[1].lower() == "debug"
    if len(sys.argv) > 4:
        CASE_FAMILY_IDX = [int(i) for i in sys.argv[4:]]
    else:
        CASE_FAMILY_IDX = list(range(len(case_families)))

    if DEBUG:
        N_SEEDS = 1
    else:
        N_SEEDS = 6
    # if (MULTI == "mpi" and comm_rank == 0) or (MULTI != "mpi"):
    if True:
        for case_family in case_families:
            case_studies[case_family]["wind_case_idx"] = {"group": 2, "vals": [i for i in range(N_SEEDS)]}

        # MISHA QUESTION how to make AMR-Wind wait for control solution?
        print([case_families[i] for i in CASE_FAMILY_IDX])
        case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in CASE_FAMILY_IDX], regenerate_wind_field=REGENERATE_WIND_FIELD, regenerate_lut=REGENERATE_LUT, n_seeds=N_SEEDS, debug=DEBUG)
