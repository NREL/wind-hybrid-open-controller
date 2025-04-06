import os
import pickle
import yaml
import copy
import sys
from glob import glob
from itertools import product
from functools import partial
from memory_profiler import profile
#from line_profiler import profile
# from datetime import timedelta


import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np

from whoc import __file__ as whoc_file
from whoc.case_studies.process_case_studies import plot_wind_field_ts
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController

if sys.platform == "linux":
    N_COST_FUNC_TUNINGS = 21
    # if os.getlogin() == "ahenry":
    #     # Kestrel
    #     STORAGE_DIR = "/projects/ssc/ahenry/whoc/floris_case_studies"
    # elif os.getlogin() == "aohe7145":
    #     STORAGE_DIR = "/projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies"
elif sys.platform == "darwin":
    N_COST_FUNC_TUNINGS = 21
    # STORAGE_DIR = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"
elif sys.platform == "win32" or sys.platform == "cygwin":  # Add Windows check
    N_COST_FUNC_TUNINGS = 21

# sequential_pyopt is best solver, stochastic is best preview type
case_studies = {
    "baseline_controllers_preview_flasc_perfect": {
                                    # "controller_dt": {"group": 1, "vals": [120, 120]},
                                    # # "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                    "target_turbine_indices": {"group": 1, "vals": ["6,4", "6,"]},
                                    # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                    # "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                    # "use_lut_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                    "controller_dt": {"group": 1, "vals": [180]},
                                    #"case_names": {"group": 1, "vals": ["LUT"]},
                                    "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController"]},
                                    "use_filtered_wind_dir": {"group": 1, "vals": [True]},
                                    "use_lut_filtered_wind_dir": {"group": 1, "vals": [True]},
                                    "simulation_dt": {"group": 0, "vals": [60]},
                                    "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm.yaml"]},
                                    # "lut_path": {"group": 0, "vals": ["../../examples/inputs/lut_smarteole_farm_(1, 2)_uncertainFalse.csv"]},
                                    "uncertain": {"group": 3, "vals": [False, False]},
                                    "wind_forecast_class": {"group": 3, "vals": ["KalmanFilterForecast", "PerfectForecast"]},
                                    "prediction_timedelta": {"group": 4, "vals": [240]},
                                    "yaw_limits": {"group": 0, "vals": ["-15,15"]}
                                    },
    "baseline_controllers_forecasters_flasc": {"controller_dt": {"group": 0, "vals": [5]},
                                               "simulation_dt": {"group": 0, "vals": [1]},
                                               "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm.yaml"]},
                                                # "lut_path": {"group": 0, "vals": ["../../examples/inputs/smarteole_farm_lut.csv"]},
                                               "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                                "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
                                                "yaw_limits": {"group": 0, "vals": ["-15,15"],
                                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                                "target_turbine_indices": {"group": 1, "vals": ["4,6", "4,"]},
                                                "uncertain": {"group": 2, "vals": [False, False, False, True, False,
                                                                                   True, False,
                                                                                   True, False,
                                                                                   True, False,
                                                                                   True, False]},
                                                "wind_forecast_class": {"group": 2, "vals": ["PerfectForecast", "PersistenceForecast", "PreviewForecast", "KalmanFilterForecast", "SVRForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast", 
                                                                                             "MLForecast", "MLForecast"]},
                                                "model_key": {"group": 2, "vals": [None, None, None, None, None,
                                                                                   "informer", "informer", 
                                                                                   "autoformer", "autoformer", 
                                                                                   "spacetimeformer", "spacetimeformer", 
                                                                                   "tactis", "tactis"]},
                                                "prediction_timedelta": {"group": 3, "vals": [60, 120, 180]},
                                                }
                                    },
    "baseline_controllers_perfect_forecaster_awaken": {
        "controller_dt": {"group": 0, "vals": [60]},
        "use_filtered_wind_dir": {"group": 0, "vals": [True]},
        "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
        "simulation_dt": {"group": 0, "vals": [30]},
        "floris_input_file": {"group": 0, "vals": ["../../examples/inputs/gch_KP_v4.yaml"]},
        "lut_path": {"group": 0, "vals": ["../../examples/inputs/gch_KP_v4_lut.csv"]},
        "yaw_limits": {"group": 0, "vals": ["-15,15"]},
        # "controller_class": {"group": 1, "vals": ["GreedyController", "LookupBasedWakeSteeringController"]},
        # "target_turbine_indices": {"group": 1, "vals": ["4,", "74,73"]},
        # "uncertain": {"group": 1, "vals": [False, False]},
        # "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast", "PerfectForecast"]},
        # "controller_class": {"group": 1, "vals": ["GreedyController"]},
        # "target_turbine_indices": {"group": 1, "vals": ["4,"]},
        # "uncertain": {"group": 1, "vals": [False]},
        # "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast"]},
        "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController"]},
        "target_turbine_indices": {"group": 1, "vals": ["74,73", "74,73"]},
        "uncertain": {"group": 1, "vals": [False, True]},
        "wind_forecast_class": {"group": 1, "vals": ["PerfectForecast", "PerfectForecast"]},
        # "prediction_timedelta": {"group": 2, "vals": [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]},
        },
    "baseline_controllers_forecasters_awaken": {"controller_dt": {"group": 0, "vals": [5]},
                                    "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController", 
                                                                              "LookupBasedWakeSteeringController", "LookupBasedWakeSteeringController",
                                                                              "GreedyController", "GreedyController"]},
                                    "target_turbine_indices": {"group": 1, "vals": ["74,73", "74,73", 
                                                                                    "74,73", "74,73",
                                                                                    "4,",  "4,"]},
                                    "uncertain": {"group": 1, "vals": [True, False, 
                                                                       True, False, 
                                                                       False, False]},
                                    "wind_forecast_class": {"group": 1, "vals": ["MLForecast", "MLForecast",
                                                                                 "KalmanFilterForecast", "KalmanFilterForecast",
                                                                                 "MLForecast", "KalmanFilterForecast"]},
                                    "model_key": {"group": 1, "vals": ["informer", "informer",
                                                                        None, None,
                                                                        "informer", None]},
                                    
                                    "prediction_timedelta": {"group": 2, "vals": [100]},
                                    # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController"]},
                                    # "target_turbine_indices": {"group": 1, "vals": ["74,73"]},
                                    # "uncertain": {"group": 1, "vals": [True]}, 
                                    "use_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    "use_lut_filtered_wind_dir": {"group": 0, "vals": [True]},
                                    # "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController"]},
                                    # "target_turbine_indices": {"group": 1, "vals": ["74,73"]},
                                    # "use_filtered_wind_dir": {"group": 1, "vals": [True]},
                                    # "use_lut_filtered_wind_dir": {"group": 1, "vals": [True]},
                                    "simulation_dt": {"group": 0, "vals": [1]},
                                    "floris_input_file": {"group": 0, "vals": [
                                        "../../examples/inputs/gch_KP_v4.yaml"
                                                                            ]},
                                    "lut_path": {"group": 0, "vals": [
                                        "../../examples/inputs/gch_KP_v4_lut.csv",
                                                                    ]},
                                    "yaw_limits": {"group": 0, "vals": ["-15,15"]}
                                    },
    "baseline_controllers": { "controller_dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "use_lut_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          },
    "solver_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    # "alpha": {"group": 0, "vals": [1.0]},
                    # "max_std_dev": {"group": 0, "vals": [2]},
                    #  "warm_start": {"group": 0, "vals": ["lut"]},
                    #     "controller_dt": {"group": 0, "vals": [15]},
                    #      "decay_type": {"group": 0, "vals": ["exp"]},
                    #     "wind_preview_type": {"group": 0, "vals": ["stochastic_sample"]},
                    #     "n_wind_preview_samples": {"group": 0, "vals": [9]},
                    #     "n_horizon": {"group": 0, "vals": [12]},
                    #     "diff_type": {"group": 0, "vals": ["direct_cd"]},
                        # "nu": {"group": 0, "vals": [0.0001]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                         "case_names": {"group": 1, "vals": ["Sequential SLSQP", "SLSQP", "Sequential Refine"]},
                        "solver": {"group": 1, "vals": ["sequential_slsqp", "slsqp", "serial_refine"]}
    },
    "wind_preview_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                          "case_names": {"group": 1, "vals": [
                                                            "Perfect", "Persistent",
                                                            "Stochastic Interval Elliptical 3", "Stochastic Interval Elliptical 5", "Stochastic Interval Elliptical 11", 
                                                            "Stochastic Interval Rectangular 3", "Stochastic Interval Rectangular 5", "Stochastic Interval Rectangular 11",
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100"
                                                            ]},
                         "n_wind_preview_samples": {"group": 1, "vals": [1, 1] + [3, 5, 11] * 2 + [25, 50, 100]},
                         "decay_type": {"group": 1, "vals": [None] * 2 + ["exp"] * 3 + ["none"] * 3 + ["cosine"] * 3},
                         "max_std_dev": {"group": 1, "vals": [None] * 2 + [2] * 3 + [2] * 3 + [1] * 3},  
                         "wind_preview_type": {"group": 1, "vals": ["perfect", "persistent"] + ["stochastic_interval_elliptical"] * 3 + ["stochastic_interval_rectangular"] * 3 + ["stochastic_sample"] * 3}
                          },
    "warm_start": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                   "case_names": {"group": 1, "vals": ["Greedy", "LUT", "Previous"]},
                   "warm_start": {"group": 1, "vals": ["greedy", "lut", "previous"]}
                   },
    "horizon_length": {"controller_class": {"group": 0, "vals": ["MPC"]},
                        "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                    f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                    f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                    #    "case_names": {"group": 1, "vals": [f"N_p = {n}" for n in [6, 12, 24, 36]]},
                        "controller_dt": {"group": 1, "vals": [15, 30, 45, 60]},
                       "n_horizon": {"group": 2, "vals": [6, 12, 18, 24]}
                    },
    "breakdown_robustness":  # case_families[5]
        {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
         "controller_dt": {"group": 1, "vals": [15, 5, 5]},
         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{25}.yaml")]},
         "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{25}.csv")]},
        #   "case_names": {"group": 1, "vals": [f"{f*100:04.1f}% Chance of Breakdown" for f in list(np.linspace(0, 0.5, N_COST_FUNC_TUNINGS))]},
          "offline_probability": {"group": 2, "vals": list(np.linspace(0, 0.1, N_COST_FUNC_TUNINGS))}
        },
    "scalability": {"controller_class": {"group": 1, "vals": ["MPC", "LookupBasedWakeSteeringController", "GreedyController"]},
                    "controller_dt": {"group": 1, "vals": [15, 5, 5]},
                    # "case_names": {"group": 2, "vals": ["3 Turbines", "9 Turbines", "25 Turbines"]},
                    "num_turbines": {"group": 2, "vals": [3, 9, 25]},
                    "floris_input_file": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), "../examples/mpc_wake_steering_florisstandin", 
                                                             f"floris_gch_{i}.yaml") for i in [3, 9, 25]]},
                    "lut_path": {"group": 2, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                    f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{nturb}.csv") for nturb in [3, 9, 25]]},
    },
    "cost_func_tuning": {"controller_class": {"group": 0, "vals": ["MPC"]},
                         "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 3)}" for f in list(np.concatenate([np.linspace(0, 0.8, int(N_COST_FUNC_TUNINGS//2)), 0.801 + (1-np.logspace(-3, 0, N_COST_FUNC_TUNINGS - int(N_COST_FUNC_TUNINGS//2)))*0.199]))]},
                         "alpha": {"group": 1, "vals": list(np.concatenate([np.linspace(0, 0.8, int(N_COST_FUNC_TUNINGS//2)), 0.801 + (1-np.logspace(-3, 0, N_COST_FUNC_TUNINGS - int(N_COST_FUNC_TUNINGS//2)))*0.199]))},
                         "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
    },
    "yaw_offset_study": {"controller_class": {"group": 1, "vals": ["MPC", "MPC", "MPC", "LookupBasedWakeSteeringController", "MPC", "MPC"]},
                          "case_names": {"group": 1, "vals":[f"StochasticIntervalRectangular_1_3turb", f"StochasticIntervalRectangular_11_3turb", f"StochasticIntervalElliptical_11_3turb", 
                                                             f"LUT_3turb", f"StochasticSample_25_3turb", f"StochasticSample_100_3turb"]},
                          "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular"] * 2 + ["stochastic_interval_elliptical"] + ["none"] + ["stochastic_sample"] * 2},
                           "n_wind_preview_samples": {"group": 1, "vals": [1, 11, 11, 1, 25, 100]},
                           "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                            "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]}
    },
    "baseline_plus_controllers": {"controller_dt": {"group": 1, "vals": [5, 5, 60.0, 60.0, 60.0, 60.0]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy", "MPC_with_Filter", "MPC_without_Filter", "MPC_without_state_cons", "MPC_without_dyn_state_cons"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController", "MPC", "MPC", "MPC", "MPC"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True, True, False, False, False]},
    },
    "baseline_controllers_3": { "controller_dt": {"group": 1, "vals": [5, 5]},
                                "case_names": {"group": 1, "vals": ["LUT", "Greedy"]},
                                "controller_class": {"group": 1, "vals": ["LookupBasedWakeSteeringController", "GreedyController"]},
                                "use_filtered_wind_dir": {"group": 1, "vals": [True, True]},
                                "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                                "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
    },
    "gradient_type": {"controller_class": {"group": 0, "vals": ["MPC"]},
                    "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                    "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                    "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular"] * 4 + ["stochastic_interval_elliptical"] * 4 + ["stochastic_sample"] * 6},
                    "n_wind_preview_samples": {"group": 1, "vals": [10] * 4 + [33] * 4 + [100] * 6},
                    "diff_type": {"group": 1, "vals": ["direct_cd", "direct_fd", "chain_cd", "chain_fd"] * 2 + ["direct_cd", "direct_fd", "direct_zscg", "chain_cd", "chain_fd", "chain_zscg"]},
                    "nu": {"group": 2, "vals": [0.0001, 0.001, 0.01]},
                    "decay_type": {"group": 3, "vals": ["none", "exp", "cosine", "linear", "zero"]},
                    # "decay_const": {"group": 2, "vals": [31, 45, 60, 90] * 3 + [90, 90]},
                    # "decay_all": {"group": 3, "vals": ["True", "False"]},
                    # "clip_value": {"group": 4, "vals": [30, 44]},
                    "max_std_dev": {"group": 4, "vals": [1, 1.5, 2]}
    },
    "n_wind_preview_samples": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
                          "case_names": {"group": 1, "vals": [
                                                            "Stochastic Interval Elliptical 11", "Stochastic Interval Elliptical 21", "Stochastic Interval Elliptical 33", 
                                                            "Stochastic Interval Rectangular 5", "Stochastic Interval Rectangular 7", "Stochastic Interval Rectangular 11", 
                                                            "Stochastic Sample 25", "Stochastic Sample 50", "Stochastic Sample 100",
                                                            "Perfect", "Persistent"]},
                        "nu": {"group": 1, "vals": [0.001] * 4 + [0.0001] * 4 + [0.001] * 4},
                        "max_std_dev": {"group": 1, "vals": [1.5] * 8 + [2] * 4 + [2, 2]},
                        "decay_type": {"group": 1, "vals": ["exp"] * 4 + ["cosine"] * 4 + ["exp"] * 4 + ["none", "none"]},
                        "n_wind_preview_samples": {"group": 1, "vals": [11, 21, 33] + [5, 7, 11] + [25, 50, 100] + [1, 1]},
                        "diff_type": {"group": 1, "vals": ["direct_cd"] * 3 + ["chain_cd"] * 3 + ["chain_cd"] * 3 + ["chain_cd", "chain_cd"]},
                         "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_elliptical"] * 3 + ["stochastic_interval_rectangular"] * 3 + ["stochastic_sample"] * 3 + ["perfect", "persistent"]}
     },
    "generate_sample_figures": {
                             "controller_class": {"group": 0, "vals": ["MPC"]},
                             "n_horizon": {"group": 0, "vals": [24]},
                             "wind_preview_type": {"group": 1, "vals": ["stochastic_interval_rectangular", "stochastic_interval_elliptical", "stochastic_sample"]},
                             "n_wind_preview_samples": {"group": 1, "vals": [5, 8, 500]},
                             "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{1}.yaml")]},
                             "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{1}.csv")]}
    },
    "cost_func_tuning_small": {
        "controller_class": {"group": 0, "vals": ["MPC"]},
        "n_horizon": {"group": 0, "vals": [6]},
        # "wind_preview_type": {"group": 2, "vals": ["stochastic_sample", "stochastic_interval_rectangular", "stochastic_interval_elliptical"]},
        # "n_wind_preview_samples": {"group": 2, "vals": [100, 10, 10]},
        "case_names": {"group": 1, "vals": [f"alpha_{np.round(f, 3)}" for f in [0.0, 0.001, 0.5, 0.999, 1]]},
        "alpha": {"group": 1, "vals": [0.0, 0.001, 0.5, 0.999, 1]},
        "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                    f"../examples/mpc_wake_steering_florisstandin/floris_gch_{3}.yaml")]},
        "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                       f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv")]},
    },
    "sr_solve": {"controller_class": {"group": 0, "vals": ["MPC"]},
                          "floris_input_file": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/floris_gch_{9}.yaml")]},
                          "lut_path": {"group": 0, "vals": [os.path.join(os.path.dirname(whoc_file), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{9}.csv")]},
                         "case_names": {"group": 0, "vals": ["Serial Refine"]},
                        "solver": {"group": 0, "vals": ["serial_refine"]}
    },
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

#@profile
def initialize_simulations(case_study_keys, regenerate_lut, regenerate_wind_field, 
                           n_seeds, stoptime, save_dir, wf_source, multiprocessor,
                           whoc_config, model_config=None, data_config=None):
    """_summary_

    Args:
        case_study_keys (_type_): _description_
        regenerate_lut (_type_): _description_
        regenerate_wind_field (_type_): _description_
        n_seeds (_type_): _description_
        stoptime (_type_): _description_
        save_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    os.makedirs(save_dir, exist_ok=True)
    
    simulation_dt = set(np.concatenate([case_studies[k]["simulation_dt"]["vals"] for k in case_study_keys]))
    assert len(simulation_dt) == 1, "There may only be a single value of 'simulation_dt'."
    simulation_dt = list(simulation_dt)[0]
    # simulation_timedelta = pd.Timedelta(seconds=list(simulation_dt)[0])
    
    if stoptime != "auto": 
        whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = int(stoptime)
    
    if "slsqp_solver_sweep" not in case_studies or "controller_dt" not in case_studies["slsqp_solver_sweep"]:
        max_controller_dt = whoc_config["controller"]["controller_dt"]
    else:
        max_controller_dt = max(case_studies["slsqp_solver_sweep"]["controller_dt"]["vals"])
    
    if "horizon_length" not in case_studies or "n_horizon" not in case_studies["horizon_length"]:
        max_n_horizon = whoc_config["controller"]["n_horizon"]
    else:
        max_n_horizon = max(case_studies["horizon_length"]["n_horizon"]["vals"])
    
    if (whoc_config["controller"]["target_turbine_indices"] is not None) and ((num_target_turbines := len(whoc_config["controller"]["target_turbine_indices"])) < whoc_config["controller"]["num_turbines"]):
        # need to change num_turbines, floris_input_file, lut_path
        whoc_config["controller"]["num_turbines"] = num_target_turbines
        whoc_config["controller"]["lut_path"] = os.path.join(os.path.dirname(whoc_config["controller"]["lut_path"]), 
                                                            f"lut_{num_target_turbines}.csv")
        whoc_config["controller"]["target_turbine_indices"] = tuple(whoc_config["controller"]["target_turbine_indices"])
        
    if whoc_config["controller"]["target_turbine_indices"] is None:
         whoc_config["controller"]["target_turbine_indices"] = "all"

    if wf_source == "floris":
        from whoc.wind_field.WindField import plot_ts
        from whoc.wind_field.WindField import generate_multi_wind_ts, WindField, write_abl_velocity_timetable, first_ord_filter
    
        with open(os.path.join(os.path.dirname(whoc_file), "wind_field", "wind_field_config.yaml"), "r") as fp:
            wind_field_config = yaml.safe_load(fp)

        # instantiate wind field if files don't already exist
        wind_field_dir = os.path.join(save_dir, 'wind_field_data/raw_data')
        wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
        os.makedirs(wind_field_dir, exist_ok=True)

        # wind_field_config["simulation_max_time"] = whoc_config["hercules_comms"]["helics"]["config"]["stoptime"]
        wind_field_config["num_turbines"] = whoc_config["controller"]["num_turbines"]
        wind_field_config["preview_dt"] = int(max_controller_dt / whoc_config["simulation_dt"])
        wind_field_config["simulation_sampling_time"] = whoc_config["simulation_dt"]
        
        # wind_field_config["n_preview_steps"] = whoc_config["controller"]["n_horizon"] * int(whoc_config["controller"]["controller_dt"] / whoc_config["simulation_dt"])
        wind_field_config["n_preview_steps"] = int(wind_field_config["simulation_max_time"] / whoc_config["simulation_dt"]) \
            + max_n_horizon * int(max_controller_dt/ whoc_config["simulation_dt"])
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
            os.makedirs(wind_field_dir, exist_ok=True)
            wind_field_data = generate_multi_wind_ts(full_wf, wind_field_dir, init_seeds=[seed + i for i in range(n_seeds)])
            write_abl_velocity_timetable([wfd.df for wfd in wind_field_data], wind_field_dir) # then use these timetables in amr precursor
            # write_abl_velocity_timetable(wind_field_data, wind_field_dir) # then use these timetables in amr precursor
            lpf_alpha = np.exp(-(1 / whoc_config["controller"]["lpf_time_const"]) * whoc_config["simulation_dt"])
            plot_wind_field_ts(wind_field_data[0].df, wind_field_dir, filter_func=partial(first_ord_filter, alpha=lpf_alpha))
            plot_ts(pd.concat([wfd.df for wfd in wind_field_data]), wind_field_dir)
            wind_field_filenames = [os.path.join(wind_field_dir, f"case_{i}.csv") for i in range(n_seeds)]
            regenerate_wind_field = True
        
        wind_field_config["regenerate_distribution_params"] = False
        
        # if wind field data exists, get it
        WIND_TYPE = "stochastic"
        wind_field_data = []
        if os.path.exists(wind_field_dir):
            for f, fn in enumerate(wind_field_filenames):
                wind_field_data.append(pd.read_csv(fn, index_col=0, parse_dates=["time"]))
                
                # wind_field_data[f]["time"] = pd.to_timedelta(wind_field_data[-1]["time"], unit="s") + pd.to_datetime("2025-01-01")
                # wind_field_data[f].to_csv(fn)
                
                if WIND_TYPE == "step":
                    # n_rows = len(wind_field_data[-1].index)
                    wind_field_data[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                    wind_field_data[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                    wind_field_data[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                    wind_field_data[-1].loc[45:, f"FreestreamWindDir"] = 270.0
        
        # write_abl_velocity_timetable(wind_field_data, wind_field_dir)
        
        # true wind disturbance time-series
        #plot_wind_field_ts(pd.concat(wind_field_data), os.path.join(wind_field_fig_dir, "seeds.png"))
        # wind_mag_ts = [wind_field_data[case_idx]["FreestreamWindMag"].to_numpy() for case_idx in range(n_seeds)]
        # wind_dir_ts = [wind_field_data[case_idx]["FreestreamWindDir"].to_numpy() for case_idx in range(n_seeds)]
        
        wind_field_ts = [wind_field_data[case_idx][["time", "FreestreamWindMag", "FreestreamWindDir"]] for case_idx in range(n_seeds)] 
        
        assert np.all([np.isclose((wind_field_data[case_idx]["time"].iloc[1] - wind_field_data[case_idx]["time"].iloc[0]).total_seconds(), whoc_config["simulation_dt"]) for case_idx in range(n_seeds)]), "sampling time of wind field should be equal to simulation sampling time"
        
        if stoptime == "auto": 
            durations = [(df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() for df in wind_field_data]
            whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = min([d.total_seconds() if hasattr(d, 'total_seconds') else d for d in durations])

    elif wf_source == "scada":
        # pull ws_horz, ws_vert, nacelle_direction, normalization_consts from awaken data and run for ML, SVR
        wind_field_ts = pl.scan_parquet(model_config["dataset"]["data_path"])
        wind_field_norm_consts = pd.read_csv(model_config["dataset"]["normalization_consts_path"], index_col=None)
        norm_min_cols = [col for col in wind_field_norm_consts if "_min" in col]
        norm_max_cols = [col for col in wind_field_norm_consts if "_max" in col]
        data_min = wind_field_norm_consts[norm_min_cols].values.flatten()
        data_max = wind_field_norm_consts[norm_max_cols].values.flatten()
        norm_min_cols = [col.replace("_min", "") for col in norm_min_cols]
        norm_max_cols = [col.replace("_max", "") for col in norm_max_cols]
        feature_range = (-1, 1)
        norm_scale = ((feature_range[1] - feature_range[0]) / (data_max - data_min))
        norm_min = feature_range[0] - (data_min * norm_scale)
        
        wind_field_ts = [df.to_pandas() for df in wind_field_ts.with_columns([(cs.starts_with(col) - norm_min[c]) 
                                                    / norm_scale[c] 
                                                    for c, col in enumerate(norm_min_cols)])\
                                         .collect().partition_by("continuity_group")]
        
        wind_field_ts = sorted(wind_field_ts, reverse=True, key=lambda df: df["time"].iloc[-1] - df["time"].iloc[0])
        wind_field_ts = wind_field_ts[:n_seeds]
        
        print(f"Loaded and normalized SCADA wind field from {model_config['dataset']['data_path']} with dt = {wind_field_ts[0]['time'].diff().iloc[1]}")
        
        # make sure wind_dt == simulation_dt
        if simulation_dt != wind_field_ts[0]["time"].diff().iloc[1].total_seconds():
            print(f"Resampling to {simulation_dt} seconds.")
            wind_field_ts = [wf.set_index("time").resample(f"{simulation_dt}s").mean().reset_index(names=["time"]) for wf in wind_field_ts]
        
        wind_field_config = {}
        
        if stoptime == "auto": 
            durations = [df["time"].iloc[-1] - df["time"].iloc[0] for df in wind_field_ts]
            whoc_config["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime = min([d.total_seconds() for d in durations])
    
    input_dicts = []
    case_lists = []
    case_name_lists = []
    n_cases_list = []
    lut_cases = set()
    input_filenames = []
    for case_study_key in case_study_keys:
        case_list, case_names = CaseGen_General(case_studies[case_study_key], namebase=case_study_key)
        case_lists = case_lists + case_list
        case_name_lists = case_name_lists + case_names
        n_cases_list.append(len(case_list))
        
        # Load default settings and make copies
        start_case_idx = len(input_dicts)
        input_dicts = input_dicts + [copy.deepcopy(whoc_config) for i in range(len(case_list))]

        # make adjustements based on case study
        for c, case in enumerate(case_list):
            print(f"Processing case: {case['wind_forecast_class']}")
            for property_name, property_value in case.items():
                if property_name in input_dicts[start_case_idx + c]["controller"]:
                    property_group = "controller"
                elif ((property_name in input_dicts[start_case_idx + c]["wind_forecast"]) 
                      or (input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"] and property_name in input_dicts[start_case_idx + c]["wind_forecast"].get(input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"], {}))):
                    property_group = "wind_forecast"
                else:
                    property_group = None
                
                if property_group:
                    if property_name == "yaw_limits":
                        input_dicts[start_case_idx + c][property_group][property_name] = tuple(int(v) for v in str(property_value).split(","))
                    elif property_name == "target_turbine_indices":
                        if property_value != "all":
                            # need to preserve order, taking first as upstream
                            target_turbine_indices = np.array([int(v) for v in property_value.split(",") if len(v)])
                            _, order_idx = np.unique(target_turbine_indices, return_index=True)
                            target_turbine_indices = target_turbine_indices[np.sort(order_idx)]
                            input_dicts[start_case_idx + c]["controller"][property_name] = tuple(target_turbine_indices)
                        else:
                            input_dicts[start_case_idx + c]["controller"][property_name] = "all"
                            
                    elif property_name == "uncertain":
                        if (controller_class := case.setdefault("controller_class", whoc_config["controller"]["controller_class"])) == "GreedyController":
                            input_dicts[start_case_idx + c]["controller"]["uncertain"] = False
                        else:
                            input_dicts[start_case_idx + c]["controller"]["uncertain"] = property_value
                    elif isinstance(property_value, np.str_):
                        input_dicts[start_case_idx + c][property_group][property_name] = str(property_value)
                    else:
                        input_dicts[start_case_idx + c][property_group][property_name] = property_value
                else:
                    input_dicts[start_case_idx + c][property_name] = property_value
            
            assert input_dicts[start_case_idx + c]["controller"]["controller_dt"] <= stoptime
            
            if input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"] or "wind_forecast_class" in case: 
                input_dicts[start_case_idx + c]["wind_forecast"] \
                    = {**{
                        "measurements_timedelta": wind_field_ts[0]["time"].iloc[1] - wind_field_ts[0]["time"].iloc[0],
                        "context_timedelta": pd.Timedelta(seconds=input_dicts[start_case_idx + c]["wind_forecast"]["context_timedelta"]),
                        "prediction_timedelta": pd.Timedelta(seconds=input_dicts[start_case_idx + c]["wind_forecast"]["prediction_timedelta"]),
                        "controller_timedelta": pd.Timedelta(seconds=input_dicts[start_case_idx + c]["controller"]["controller_dt"])
                        }, 
                    **input_dicts[start_case_idx + c]["wind_forecast"].setdefault(input_dicts[start_case_idx + c]["controller"]["wind_forecast_class"], {}),
                    }
                
            # need to change num_turbines, floris_input_file, lut_path
            if (target_turbine_indices := input_dicts[start_case_idx + c]["controller"]["target_turbine_indices"])  != "all":
                target_turbine_indices = tuple(target_turbine_indices)
                num_target_turbines = len(target_turbine_indices)
                input_dicts[start_case_idx + c]["controller"]["num_turbines"] = num_target_turbines
                # NOTE: lut tables should be regenerated for different yaw limits
                uncertain_flag = input_dicts[start_case_idx + c]["controller"]["uncertain"]
                lut_path = input_dicts[start_case_idx + c]["controller"]["lut_path"]
                floris_input_file = os.path.splitext(os.path.basename(input_dicts[start_case_idx + c]["controller"]["floris_input_file"]))[0]
                yaw_limits = (input_dicts[start_case_idx + c]["controller"]["yaw_limits"])[1]
                input_dicts[start_case_idx + c]["controller"]["lut_path"] = os.path.join(
                    os.path.dirname(lut_path), 
                    f"lut_{floris_input_file}_{target_turbine_indices}_uncertain{uncertain_flag}_yawlimits{yaw_limits}.csv")
            # **{k: v for k, v in input_dicts[start_case_idx + c]["wind_forecast"].items() if isinstance(k, str) and "_kwargs" in k} 
            assert input_dicts[start_case_idx + c]["controller"]["controller_dt"] >= input_dicts[start_case_idx + c]["simulation_dt"], "controller_dt must be greater than or equal to simulation_dt"
             
            # regenerate floris lookup tables for all wind farms included
            # generate LUT for combinations of lut_path/floris_input_file, yaw_limits, uncertain, and target_turbine_indices that arise together
            if regenerate_lut:
                
                floris_input_file = input_dicts[start_case_idx + c]["controller"]["floris_input_file"]
                lut_path = input_dicts[start_case_idx + c]["controller"]["lut_path"] 
                uncertain_flag = input_dicts[start_case_idx + c]["controller"]["uncertain"] 
                yaw_limits = tuple(input_dicts[start_case_idx + c]["controller"]["yaw_limits"])
                target_turbine_indices = input_dicts[start_case_idx + c]["controller"]["target_turbine_indices"]
                if (new_case := tuple([floris_input_file, lut_path, uncertain_flag, yaw_limits, target_turbine_indices])) in lut_cases:
                    continue
                
                print(f"Regenerating LUT {lut_path}")
                LookupBasedWakeSteeringController._optimize_lookup_table(
                    floris_config_path=floris_input_file, uncertain=uncertain_flag, yaw_limits=yaw_limits, 
                    parallel=multiprocessor is not None,
                    sorted_target_tids=sorted(target_turbine_indices) if target_turbine_indices != "all" else "all", lut_path=lut_path, generate_lut=True)
                
                lut_cases.add(new_case)

                input_dicts[start_case_idx + c]["controller"]["generate_lut"] = False
            
            # TODO rename this by index with only config updates from case inside
            fn = f'input_config_case_{"_".join(
                [f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case.items() \
                    if key not in ["simulation_dt", "use_filtered_wind_dir", "use_lut_filtered_wind_dir", "yaw_limits", "wind_case_idx", "seed", "floris_input_file", "lut_path"]]) \
                    if "case_names" not in case else case["case_names"]}.pkl'.replace("/", "_")

            input_filenames.append((case_study_key, fn)) 

    prediction_timedelta = max(inp["wind_forecast"]["prediction_timedelta"] for inp in input_dicts if inp["controller"]["wind_forecast_class"]) \
            if any(inp["controller"]["wind_forecast_class"] for inp in input_dicts) else pd.Timedelta(seconds=0)
    horizon_timedelta = max(pd.Timedelta(seconds=inp["controller"]["n_horizon"] * inp["controller"]["controller_dt"]) for inp in input_dicts if inp["controller"]["n_horizon"]) \
            if any(inp["controller"]["controller_class"] == "MPC" for inp in input_dicts) else pd.Timedelta(seconds=0)
    # stoptime -= prediction_timedelta.total_seconds()
    # assert stoptime > 0, "increase stoptime parameter and/or decresease prediction_timedetla, as stoptime < prediction_timedelta"

    # assert all([(df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() >= stoptime + prediction_timedelta + horizon_timedelta for df in wind_field_ts])
    wind_field_ts = [df.loc[(df["time"] - df["time"].iloc[0]).dt.total_seconds() 
                        <= stoptime + prediction_timedelta.total_seconds() + horizon_timedelta.total_seconds()] 
                    for df in wind_field_ts]
    stoptime = max(min([((df["time"].iloc[-1] - df["time"].iloc[0]) - prediction_timedelta - horizon_timedelta).total_seconds() for df in wind_field_ts]), stoptime)
    for (case_study_key, fn), inp in zip(input_filenames, input_dicts):
        inp["hercules_comms"]["helics"]["config"]["stoptime"] = stoptime
        results_dir = os.path.join(save_dir, case_study_key)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, fn), 'wb') as fp:
            pickle.dump(inp, fp)
    
    # instantiate controller and run_simulations simulation
    with open(os.path.join(save_dir, "init_simulations.pkl"), "wb") as fp:
        pickle.dump({"case_lists": case_lists, "case_name_lists": case_name_lists, "input_dicts": input_dicts, "wind_field_config": wind_field_config,
                    "wind_field_ts": wind_field_ts}, fp)

    return case_lists, case_name_lists, input_dicts, wind_field_config, wind_field_ts

# 0, 1, 2, 3, 6
case_families = ["baseline_controllers", "solver_type", # 0, 1
                    "wind_preview_type", "warm_start", # 2, 3
                    "horizon_length", "cost_func_tuning",  # 4, 5
                    "yaw_offset_study", "scalability", # 6, 7
                    "breakdown_robustness", # 8
                    "gradient_type", "n_wind_preview_samples", # 9, 10
                    "generate_sample_figures", "baseline_controllers_3", # 11, 12
                    "cost_func_tuning_small", "sr_solve", # 13, 14
                    "baseline_controllers_forecasters_flasc", "baseline_controllers_forecasters_awaken", # 15, 16
                    "baseline_controllers_preview_flasc_perfect", "baseline_controllers_perfect_forecaster_awaken"] # 18
