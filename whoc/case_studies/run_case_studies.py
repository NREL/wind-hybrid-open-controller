import os
from concurrent.futures import ProcessPoolExecutor
import warnings
import re
import argparse

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import numpy as np
import pandas as pd
import yaml
import pickle
from memory_profiler import profile

import whoc
try:
    from whoc.controllers.mpc_wake_steering_controller import MPC
except Exception:
    print("Cannot import MPC controller in current environment.")
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import (read_time_series_data, write_case_family_time_series_data, read_case_family_time_series_data, 
                                                    aggregate_time_series_data, read_case_family_agg_data, write_case_family_agg_data, 
                                                    generate_outputs, plot_simulations, plot_wind_farm, plot_breakdown_robustness, plot_horizon_length,
                                                    plot_cost_function_pareto_curve, plot_yaw_offset_wind_direction, plot_parameter_sweep, plot_power_increase_vs_prediction_time)
try:
    from whoc.wind_forecast.WindForecast import PerfectForecast, PersistenceForecast, MLForecast, SVRForecast, KalmanFilterForecast, PreviewForecast
except ModuleNotFoundError:
    print("Cannot import wind forecast classes in current environment.")
# np.seterr("raise")

warnings.simplefilter('error', pd.errors.DtypeWarning)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_case_studies.py", description="Run FLORIS case studies for WHOC module.")
    parser.add_argument("case_ids", metavar="C", nargs="+", choices=[str(i) for i in range(len(case_families))])
    parser.add_argument("-gwf", "--generate_wind_field", action="store_true")
    parser.add_argument("-glut", "--generate_lut", action="store_true")
    parser.add_argument("-rs", "--run_simulations", action="store_true")
    parser.add_argument("-rrs", "--rerun_simulations", action="store_true")
    parser.add_argument("-ps", "--postprocess_simulations", action="store_true")
    parser.add_argument("-rps", "--reprocess_simulations", action="store_true")
    parser.add_argument("-ras", "--reaggregate_simulations", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-st", "--stoptime", default="auto")
    parser.add_argument("-ns", "--n_seeds", type=int, default=1)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"])
    parser.add_argument("-sd", "--save_dir", type=str, default=os.path.join(os.getcwd(), "simulation_results"))
    parser.add_argument("-wf", "--wf_source", type=str, choices=["floris", "scada"], required=True)
    parser.add_argument("-mcnf", "--model_config", type=str, required=False, default="")
    parser.add_argument("-dcnf", "--data_config", type=str, required=False, default="")
    parser.add_argument("-wcnf", "--whoc_config", type=str, required=True)
     
    # "/projects/ssc/ahenry/whoc/floris_case_studies" on kestrel
    # "/projects/aohe7145/whoc/floris_case_studies" on curc
    # "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies" on mac
    # python run_case_studies.py 0 1 2 3 4 5 6 7 -rs -p -st 480 -ns 1 -m cf -sd "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"
    args = parser.parse_args()
    args.case_ids = [int(i) for i in args.case_ids]

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": max(d["group"] for d in case_studies[case_family].values()) + 1, "vals": [i for i in range(args.n_seeds)]}

    # os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "false"
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    PLOT = True #sys.platform != "linux"
    if args.run_simulations or args.generate_lut or args.generate_wind_field:
        # run simulations
        
        if RUN_ONCE:
            print(f"running initialize_simulations for case_ids {[case_families[i] for i in args.case_ids]}")
            # os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml")
            with open(args.whoc_config, 'r') as file:
                whoc_config  = yaml.safe_load(file)
                
            if args.wf_source == "scada":
                with open(args.model_config, 'r') as file:
                    model_config  = yaml.safe_load(file)
                with open(args.data_config, 'r') as file:
                    data_config  = yaml.safe_load(file)
                    
                if len(data_config["turbine_signature"]) == 1:
                    tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
                else:
                    tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
                
                turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
                # optuna_args = model_config.setdefault("optuna", None)
     
            else:
                model_config = None
                data_config = None
                turbine_signature = None
                tid2idx_mapping = None
                
            case_lists, case_name_lists, input_dicts, wind_field_config, wind_field_ts \
                = initialize_simulations(case_study_keys=[case_families[i] for i in args.case_ids], 
                                         regenerate_wind_field=args.generate_wind_field, 
                                         regenerate_lut=args.generate_lut, 
                                         n_seeds=args.n_seeds, 
                                         stoptime=args.stoptime, 
                                         save_dir=args.save_dir, 
                                         wf_source=args.wf_source,
                                         multiprocessor=args.multiprocessor, 
                                         whoc_config=whoc_config, model_config=model_config, data_config=data_config)
        
        if args.multiprocessor is not None:
            if args.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif args.multiprocessor == "cf":
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                if args.multiprocessor == "mpi":
                    run_simulations_exec.max_workers = comm_size
                  
                # print(f"run_simulations line 64 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[d["controller"]["controller_class"]], 
                                                wind_forecast_class=globals()[d["controller"]["wind_forecast_class"]] if d["controller"]["wind_forecast_class"] else None,
                                                simulation_input_dict=d,
                                                wf_source=args.wf_source, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_field_ts=wind_field_ts[case_lists[c]["wind_case_idx"]],
                                                case_name="_".join([f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["simulation_dt", "use_filtered_wind_dir", "use_lut_filtered_wind_dir", "yaw_limits", "wind_case_idx", "seed", "floris_input_file", "lut_path"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), 
                                                verbose=args.verbose, 
                                                save_dir=args.save_dir, 
                                                rerun_simulations=args.rerun_simulations,
                                                multiprocessor=False, 
                                                turbine_signature=turbine_signature, 
                                                tid2idx_mapping=tid2idx_mapping,
                                                use_tuned_params=True, 
                                                model_config=model_config, wind_field_config=wind_field_config, )

                        for c, d in enumerate(input_dicts)]
                
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[d["controller"]["controller_class"]], 
                                    wind_forecast_class=globals()[d["controller"]["wind_forecast_class"]] if d["controller"]["wind_forecast_class"] else None, 
                                    simulation_input_dict=d, 
                                    wf_source=args.wf_source,
                                    wind_case_idx=case_lists[c]["wind_case_idx"], wind_field_ts=wind_field_ts[case_lists[c]["wind_case_idx"]],
                                    case_name="_".join([f"{key}_{val if (isinstance(val, str) or isinstance(val, np.str_) or isinstance(val, bool)) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["simulation_dt", "use_filtered_wind_dir", "use_lut_filtered_wind_dir", "yaw_limits", "wind_case_idx", "seed", "floris_input_file", "lut_path"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                    case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                    multiprocessor=False, 
                                    wind_field_config=wind_field_config, verbose=args.verbose, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations,
                                    turbine_signature=turbine_signature, tid2idx_mapping=tid2idx_mapping,
                                    use_tuned_params=True, model_config=model_config)
    
    if args.postprocess_simulations:
        # if (not os.path.exists(os.path.join(args.save_dir, f"time_series_results.csv"))) or (not os.path.exists(os.path.join(args.save_dir, f"agg_results.csv"))):
        # regenerate some or all of the time_series_results_all and agg_results_all .csv files for each case family in case ids
        if args.reprocess_simulations \
            or not all(os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")) for i in args.case_ids) \
                or not all(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv") for i in args.case_ids):
            if RUN_ONCE:
                # make a list of the time series csv files for all case_names and seeds in each case family directory
                case_family_case_names = {}
                for i in args.case_ids:
                    case_family_case_names[case_families[i]] = [fn for fn in os.listdir(os.path.join(args.save_dir, case_families[i])) if ".csv" in fn and "time_series_results_case" in fn]

                # case_family_case_names["slsqp_solver_sweep"] = [f"time_series_results_case_alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval_seed_{s}" for s in range(6)]

            # if using multiprocessing
            if args.multiprocessor is not None:
                if args.multiprocessor == "mpi":
                    comm_size = MPI.COMM_WORLD.Get_size()
                    executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                elif args.multiprocessor == "cf":
                    executor = ProcessPoolExecutor()
                with executor as run_simulations_exec:
                    if args.multiprocessor == "mpi":
                        run_simulations_exec.max_workers = comm_size
                    
                    print(f"run_simulations line 107 with {run_simulations_exec._max_workers} workers")
                    # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)

                    # if args.reaggregate_simulations is true, or for any case family where doesn't time_series_results_all.csv exist, 
                    # read the time-series csv files for all case families, case names, and wind seeds
                    read_futures = [run_simulations_exec.submit(
                                                    read_time_series_data, 
                                                    results_path=os.path.join(args.save_dir, case_families[i], fn))
                        for i in args.case_ids 
                        for fn in case_family_case_names[case_families[i]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))
                    ]

                    new_time_series_df = [fut.result() for fut in read_futures]
                    # if there are new resulting dataframes, concatenate them from a list into a dataframe
                    if new_time_series_df:
                        new_time_series_df = pd.concat(new_time_series_df)

                    read_futures = [run_simulations_exec.submit(read_case_family_time_series_data, 
                                                                case_family=case_families[i], save_dir=args.save_dir)
                                    for i in args.case_ids
                                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                    existing_time_series_df = [fut.result() for fut in read_futures]

                    if len(new_time_series_df):
                        write_futures = [run_simulations_exec.submit(write_case_family_time_series_data, 
                                                                     case_family=case_families[i], 
                                                                     new_time_series_df=new_time_series_df, 
                                                                     save_dir=args.save_dir)
                                        for i in args.case_ids
                                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv"))]
                        _ = [fut.result() for fut in write_futures]
                    
                    time_series_df = pd.concat(existing_time_series_df + [new_time_series_df])

                    # if args.reaggregate_simulations is true, or for any case family where doesn't agg_results_all.csv exist, compute the aggregate stats for each case families and case name, over all wind seeds
                    futures = [run_simulations_exec.submit(aggregate_time_series_data,
                                                             time_series_df=time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i]) & (time_series_df.index.get_level_values("CaseName") == case_name), :],
                                                                input_dict_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.pkl"),
                                                                n_seeds=args.n_seeds)
                        for i in args.case_ids
                        for case_name in pd.unique(time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i])].index.get_level_values("CaseName"))
                        # for case_name in [re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], 
                                                                                           "agg_results_all.csv"))
                    ]

                    new_agg_df = [fut.result() for fut in futures]
                    new_agg_df = [df for df in new_agg_df if df is not None]
                    if len(new_agg_df):
                        new_agg_df = pd.concat(new_agg_df)
                    else:
                        new_agg_df = pd.DataFrame()
                    # if args.reaggregate_simulations is false, read the remaining aggregate data from each agg_results_all csv file
                    read_futures = [run_simulations_exec.submit(read_case_family_agg_data, 
                                                                case_family=case_families[i], save_dir=args.save_dir)
                                    for i in args.case_ids 
                                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv"))]
                    existing_agg_df = [fut.result() for fut in read_futures]

                    if len(new_agg_df):
                        write_futures = [run_simulations_exec.submit(write_case_family_agg_data,
                                                                     case_family=case_families[i],
                                                                     new_agg_df=new_agg_df,
                                                                     save_dir=args.save_dir)
                                        for i in args.case_ids
                                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv"))]
                        _ = [fut.result() for fut in write_futures]

                    agg_df = pd.concat(existing_agg_df + [new_agg_df])
                    
            # else, run sequentially
            else:
                new_time_series_df = []
                existing_time_series_df = []
                for i in args.case_ids:
                    # all_ts_df_path = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        existing_time_series_df.append(read_case_family_time_series_data(case_families[i], save_dir=args.save_dir))
                
                new_case_family_time_series_df = [] 
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")):
                        # TODO HIGH check that we are only reading time series files corresponding to the number of seeds, stop times used
                        for fn in case_family_case_names[case_families[i]]:
                            new_case_family_time_series_df.append(read_time_series_data(results_path=os.path.join(args.save_dir, case_families[i], fn)))

                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if new_case_family_time_series_df:
                        new_time_series_df.append(pd.concat(new_case_family_time_series_df))
                        write_case_family_time_series_data(case_families[i], new_time_series_df[-1], args.save_dir)
                
                time_series_df = pd.concat(existing_time_series_df + new_time_series_df)
                
                new_agg_df = []
                for i in args.case_ids:
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        # for case_name in set([re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]):
                        case_family_df = time_series_df.iloc[time_series_df.index.get_level_values("CaseFamily") == case_families[i], :]
                        for case_name in pd.unique(case_family_df.index.get_level_values("CaseName")):
                            case_name_df = case_family_df.iloc[case_family_df.index.get_level_values("CaseName") == case_name, :]
                            res = aggregate_time_series_data(
                                                            time_series_df=case_name_df,
                                                            input_dict_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.pkl"),
                                                            # results_path=os.path.join(args.save_dir, case_families[i], f"agg_results_{case_name}.csv"),
                                                            n_seeds=args.n_seeds)
                            if res is not None:
                                new_agg_df.append(res)

                new_agg_df = pd.concat(new_agg_df)

                existing_agg_df = []
                for i in args.case_ids:
                    if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        existing_agg_df.append(read_case_family_agg_data(case_families[i], save_dir=args.save_dir))
                
                for i in args.case_ids:
                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")):
                        write_case_family_agg_data(case_families[i], new_agg_df, args.save_dir)
                
                agg_df = pd.concat(existing_agg_df + [new_agg_df])

        elif RUN_ONCE:
            time_series_df = []
            for i in args.case_ids:
                warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "time_series_results_all.csv")
                if os.path.exists(filepath):
                    try:
                        time_series_df.append(pd.read_csv(filepath, index_col=[0, 1]))
                    except pd.errors.DtypeWarning as w:
                        print(f"DtypeWarning with combined time series file {filepath}: {w}")
                        warnings.simplefilter('ignore', pd.errors.DtypeWarning)
                        bad_df = pd.read_csv(filepath, index_col=[0, 1])
                        bad_cols = [bad_df.columns[int(s) - len(bad_df.index.names)] for s in re.findall(r"(?<=Columns \()(.*)(?=\))", w.args[0])[0].split(",")]
                        bad_df.loc[bad_df[bad_cols].isna().any(axis=1)]
            time_series_df = pd.concat(time_series_df)
            
            agg_df = []
            for i in args.case_ids:
                warnings.simplefilter('error', pd.errors.DtypeWarning)
                filepath = os.path.join(args.save_dir, case_families[i], "agg_results_all.csv")
                if os.path.exists(filepath):
                    try:
                        agg_df.append(pd.read_csv(filepath, header=[0,1], index_col=[0, 1], skipinitialspace=True))
                    except pd.errors.DtypeWarning as w:
                        print(f"DtypeWarning with combined time series file {filepath}: {w}")
                        warnings.simplefilter('ignore', pd.errors.DtypeWarning)
                        bad_df = pd.read_csv(filepath, header=[0,1], index_col=[0, 1], skipinitialspace=True)
                        bad_cols = [bad_df.columns[int(s) - len(bad_df.index.names)] for s in re.findall(r"(?<=Columns \()(.*)(?=\))", w.args[0])[0].split(",")]
                        bad_df.loc[bad_df[bad_cols].isna().any(axis=1)]

            agg_df = pd.concat(agg_df)

        if RUN_ONCE and PLOT:
            if case_families.index("baseline_controllers_preview_flasc_perfect") in args.case_ids:
                 pass
             
            if case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:
                from whoc.wind_forecast.WindForecast import WindForecast
                from wind_forecasting.preprocessing.data_inspector import DataInspector
                
                forecaster_case_fam = "baseline_controllers_perfect_forecaster_awaken"
                baseline_time_df = time_series_df.loc[time_series_df.index.get_level_values("CaseFamily") == forecaster_case_fam, :] #.reset_index(level="CaseFamily", drop=True)
                baseline_agg_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily") == forecaster_case_fam, :] #.reset_index(level="CaseFamily", drop=True)
                
                config_cols = ["controller_class", "wind_forecast_class", "prediction_timedelta"]
                
                for (case_family, case_name), _ in baseline_agg_df.iterrows():
                # for case_name, _ in baseline_time_df.iterrows():    
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.pkl"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='rb') as fp:
                        input_config = pickle.load(fp)
                        
                    full_config = {**input_config["controller"], **input_config["wind_forecast"]}
                    for col in config_cols:
                        baseline_time_df.loc[(baseline_time_df.index.get_level_values("CaseFamily") == case_family) & 
                                            (baseline_time_df.index.get_level_values("CaseName") == case_name), col] = full_config[col]
                        baseline_agg_df.loc[(baseline_agg_df.index.get_level_values("CaseFamily") == case_family) & 
                                            (baseline_agg_df.index.get_level_values("CaseName") == case_name), col] = full_config[col]

                # Filter data for the two forecast types
                forecasters_agg_df = baseline_agg_df.loc[baseline_agg_df["wind_forecast_class"] != "PerfectForecast", :]
                perfect_agg_df = baseline_agg_df.loc[baseline_agg_df["wind_forecast_class"] == "PerfectForecast", :]
                
                # PLOT 1) Farm power of perfect forecaster vs prediction timedela for different controllers
                import seaborn as sns
                import matplotlib.pyplot as plt
                perfect_agg_df["prediction_timedelta"] = perfect_agg_df["prediction_timedelta"].dt.total_seconds()
                perfect_agg_df[("FarmPowerMean", "mean")] = perfect_agg_df[("FarmPowerMean", "mean")] / 1e6
                controller_labels = {"GreedyController": "Greedy", "LookupBasedWakeSteeringController": "LUT"}
                controllers = pd.unique(perfect_agg_df["controller_class"])
                x_vals = pd.unique(perfect_agg_df["prediction_timedelta"])
                xlim = (x_vals.min(), x_vals.max())
                fig, ax = plt.subplots(1, len(controllers))
                for c, ctrl in enumerate(controllers):
                    sns.lineplot(perfect_agg_df.loc[perfect_agg_df["controller_class"] == ctrl, :], 
                                x="prediction_timedelta", y=("FarmPowerMean", "mean"), ax=ax[c])
                    ax[c].set_ylabel("")
                    ax[c].set_xlabel("Prediction Horizon (s)")
                    ax[c].set_title(f"{controller_labels[ctrl]} Mean Farm Power (MW)")
                    ax[c].set_xlim(xlim)

                # PLOT 2) Farm power ratio of other forecasters relative to perfect forecaster vs prediction timedela for different controllers (diff plots)
                forecasters_agg_df["power_ratio"] = (forecasters_agg_df[("FarmPowerMean", "mean")] / perfect_agg_df[("FarmPowerMean", "mean")]) * 100
                plot_power_increase_vs_prediction_time(forecasters_agg_df, args.save_dir)

                # PLOT 3) Mean of True vs Predicted values of Turbine wind speeds vs. time for different controllers (diff plots) and forecasters (diff colors)
                
                # forecasters_time_df = baseline_time_df.loc[baseline_time_df["wind_forecast_class"] != "PerfectForecast", :]
                # perfect_time_df = baseline_time_df.loc[baseline_time_df["wind_forecast_class"] == "PerfectForecast", :]
                baseline_time_df["TrueTurbineWindSpeedHorzMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "TrueTurbineWindSpeedHorz_" in col]].mean(axis=1)
                baseline_time_df["TrueTurbineWindSpeedVertMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "TrueTurbineWindSpeedVert_" in col]].mean(axis=1)
                baseline_time_df["PredictedTurbineWindSpeedHorzMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "PredictedTurbineWindSpeedHorz_" in col]].mean(axis=1)
                baseline_time_df["PredictedTurbineWindSpeedVertMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "PredictedTurbineWindSpeedVert_" in col]].mean(axis=1)
                baseline_time_df["StddevTurbineWindSpeedHorzMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "StddevTurbineWindSpeedHorz_" in col]].mean(axis=1)
                baseline_time_df["StddevTurbineWindSpeedVertMean"] = baseline_time_df[[col for col in baseline_time_df.columns if "StddevTurbineWindSpeedVert_" in col]].mean(axis=1)
                
                fig, ax = plt.subplots(2, len(controllers))
                xlim = (baseline_time_df[["Time", "TrueTurbineWindSpeedHorzMean", "TrueTurbineWindSpeedVertMean"]].dropna()["Time"].min(),
                        baseline_time_df[["Time", "TrueTurbineWindSpeedHorzMean", "TrueTurbineWindSpeedVertMean"]].dropna()["Time"].max())
                for c, ctrl in enumerate(controllers):
                    cond = (baseline_time_df["controller_class"] == ctrl) & (baseline_time_df["WindSeed"] == 0)
                    sns.lineplot(baseline_time_df.reset_index(level=["CaseFamily", "CaseName"], drop=True).loc[cond.values, :],
                                 x="Time", y="TrueTurbineWindSpeedHorzMean", 
                                 hue="prediction_timedelta",
                                 style="wind_forecast_class", ax=ax[0, c])
                    sns.lineplot(baseline_time_df.reset_index(level=["CaseFamily", "CaseName"], drop=True).loc[cond.values, :],
                                 x="Time", y="TrueTurbineWindSpeedVertMean", 
                                 hue="prediction_timedelta",
                                 style="wind_forecast_class", ax=ax[1, c])
                    ax[0, c].set_title(f"{controller_labels[ctrl]}")
                    ax[0, c].set_ylabel("")
                    ax[1, c].set_ylabel("")
                    ax[0, c].legend([], [], frameon=False)
                    ax[1, c].legend([], [], frameon=False)
                    ax[0, c].set_xlabel("")
                    ax[1, c].set_xlabel("Time (s)")
                    ax[0, c].set_xlim(xlim)
                    ax[1, c].set_xlim(xlim)
                    ax[0, c].set_xticks([])
                
                ax[0, 0].set_ylabel("Horizontal \nWind \nSpeed \n(m/s)", rotation=0, labelpad=50)
                ax[1, 0].set_ylabel("Vertical \nWind \nSpeed \n(m/s)", rotation=0, labelpad=50)
                # ax[0, 1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.tight_layout()
                
                # PLOT 4) Yaw angles/power for persistent vs. other forecasters for best lead times
                best_forecaster_prediction_delta = forecasters_agg_df.groupby("wind_forecast_class", group_keys=False).apply(lambda x: x.sort_values(by=("FarmPowerMean", "mean"), ascending=False).head(10)) #[("FarmPowerMean", "mean")] 
                best_perfect_prediction_delta = perfect_agg_df.groupby("wind_forecast_class", group_keys=False).apply(lambda x: x.sort_values(by=("FarmPowerMean", "mean"), ascending=False).head(10))
                plotting_cases = [(forecaster_case_fam, best_perfect_prediction_delta.iloc[0]._name[1])]
                plot_simulations(
                        baseline_time_df, plotting_cases, args.save_dir, include_power=True, 
                        legend_loc="outer", single_plot=False) 

                
                forecast_wf = time_series_df.iloc[
                    (time_series_df.index.get_level_values("CaseFamily") == forecaster_case_name)]\
                        .reset_index(level=["CaseFamily", "CaseName"], drop=True)[
                           ["WindSeed", "PredictedTime", "controller_class", "wind_forecast_class", "prediction_timedelta"] 
                           + [col for col in time_series_df.columns if "PredictedTurbine" in col] 
                        ].rename(columns={"PredictedTime": "time"})
                        
                true_wf = time_series_df.iloc[
                    (time_series_df.index.get_level_values("CaseFamily") == forecaster_case_name)]\
                        .reset_index(level=["CaseFamily", "CaseName"], drop=True)[
                           ["WindSeed", "Time", "controller_class", "wind_forecast_class", "prediction_timedelta"] 
                           + [col for col in time_series_df.columns if col.startswith("TurbineWind")] 
                           + [col for col in time_series_df.columns if col.startswith("TurbineYawAngle_")] 
                        ].rename(columns={"Time": "time"})
                
                id_vars = ["WindSeed", "time", "controller_class", "wind_forecast_class", "prediction_timedelta"]
                value_vars = set([re.match(".*(?=_\\d+)", col).group(0) for col in time_series_df.columns if col.startswith("PredictedTurbine")])
                # first unpivot makes long on turbine ids, second makes long on feature type, for the purposes of seaborn plot
                forecast_wf = DataInspector.unpivot_dataframe(forecast_wf, 
                                                            value_vars=value_vars, 
                                                            turbine_signature="_(\\d+)$")\
                                                    .melt(id_vars=["turbine_id"] + id_vars, 
                                                          value_vars=value_vars, 
                                                          var_name="feature", value_name="value")
                forecast_wf = forecast_wf.assign(data_type="Forecast")
                forecast_wf["time"] = pd.to_datetime(forecast_wf["time"], unit="s")
                forecast_wf.loc[forecast_wf["feature"] == "PredictedTurbineWindSpeedVert", "feature"] = "ws_vert"
                forecast_wf.loc[forecast_wf["feature"] == "PredictedTurbineWindSpeedHorz", "feature"] = "ws_horz"
                forecast_wf.loc[forecast_wf["feature"] == "PredictedTurbineWindDir", "feature"] = "wd"
                forecast_wf.loc[forecast_wf["feature"] == "PredictedTurbineWindMag", "feature"] = "wm"
                
                value_vars = set([re.match(".*(?=_\\d+)", col).group(0) for col in time_series_df.columns if (col.startswith("TurbineWind") or col.startswith("TurbineYawAngle_"))]) 
                true_wf = DataInspector.unpivot_dataframe(true_wf, 
                                                    value_vars=value_vars, 
                                                turbine_signature="_(\\d+)$")\
                                                .melt(id_vars=["turbine_id"] + id_vars, 
                                            value_vars=value_vars, 
                                            var_name="feature", value_name="value")
                true_wf = true_wf.assign(data_type="True")
                true_wf["time"] = pd.to_datetime(true_wf["time"], unit="s")
                true_wf.loc[true_wf["feature"] == "TurbineWindSpeedVert", "feature"] = "ws_vert"
                true_wf.loc[true_wf["feature"] == "TurbineWindSpeedHorz", "feature"] = "ws_horz"
                true_wf.loc[true_wf["feature"] == "TurbineWindDir", "feature"] = "wd"
                true_wf.loc[true_wf["feature"] == "TurbineWindMag", "feature"] = "wm"
                true_wf.loc[true_wf["feature"] == "TurbineYawAngle", "feature"] = "nc"
                # TODO HIGH clean this code up
                
                wind_seed = 0
                wind_forecast_class = "KalmanFilterForecast" # "PerfectForecast" # 
                # controller_class = "GreedyController"
                controller_class = "LookupBasedWakeSteeringController"
                # TODO WHY DO GREEDY AND LUT LOOK THE SAME, WHY IS ONLY ONE TURBINE VISIBLE
                WindForecast.plot_forecast(
                    forecast_wf=forecast_wf.loc[
                        (forecast_wf["WindSeed"] == wind_seed) & (forecast_wf["wind_forecast_class"] == wind_forecast_class) & (forecast_wf["controller_class"] == controller_class), 
                        ["data_type", "time", "feature", "value", "turbine_id"]],
                    true_wf=true_wf.loc[
                        (true_wf["WindSeed"] == wind_seed) & (true_wf["wind_forecast_class"] == wind_forecast_class) & (true_wf["controller_class"] == controller_class), 
                        ["data_type", "time", "feature", "value", "turbine_id"]]
                )
                import seaborn as sns
                import matplotlib.pyplot as plt
                import polars as pl
                fig, ax = plt.subplots(1, 1)
                #  & (true_wf["time"].between(forecast_wf["time"].min(), forecast_wf["time"].max(), closed="both"))
                sns.lineplot(data=true_wf.loc[
                        (true_wf["WindSeed"] == wind_seed) & (true_wf["wind_forecast_class"] == wind_forecast_class) & (true_wf["controller_class"] == controller_class), 
                        ["data_type", "time", "feature", "value", "turbine_id"]]\
                                        .loc[(true_wf["feature"] == "nc"), :],
                                 x="time", y="value", ax=ax, style="data_type", hue="turbine_id")
            
                 
            
            if ((case_families.index("baseline_controllers") in args.case_ids)):
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") != "baseline_controllers"]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) 
                                                & (mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), 
                                                [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]\
                                                    .sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True)\
                                                        .reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), 
                                                   [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]\
                                                    .sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)\
                                                        .reset_index(level="CaseFamily", drop=True)

                100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - lut_df.iloc[0]["FarmPowerMean"]) / lut_df.iloc[0]["FarmPowerMean"]
                100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - greedy_df.iloc[0]["FarmPowerMean"]) / greedy_df.iloc[0]["FarmPowerMean"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - lut_df.iloc[0]["YawAngleChangeAbsMean"]) / lut_df.iloc[0]["YawAngleChangeAbsMean"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - greedy_df.iloc[0]["YawAngleChangeAbsMean"]) / greedy_df.iloc[0]["YawAngleChangeAbsMean"]
                
                if True:
                    plotting_cases = [("wind_preview_type", better_than_lut_df.iloc[0]._name),   
                                        ("baseline_controllers", "LUT"),
                                        ("baseline_controllers", "Greedy")
                        ]
                    plot_simulations(
                        time_series_df, plotting_cases, args.save_dir, include_power=True, legend_loc="outer", single_plot=False) 

            if ((case_families.index("baseline_controllers") in args.case_ids)) and (case_families.index("cost_func_tuning") in args.case_ids):
                
                mpc_alpha_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning"]

                if case_families.index("baseline_controllers") in args.case_ids:
                    lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                elif case_families.index("baseline_controllers_3") in args.case_ids:
                    lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName") == "Greedy")]

                mpc_alpha_df[[("RelativeTotalRunningOptimizationCostMean", "mean"), ("RelativeRunningOptimizationCostTerm_0", "mean"), ("RelativeRunningOptimizationCostTerm_1", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]\
                        .sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True) 
            

                # better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_alpha_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)

                # plot_simulations(time_series_df=time_series_df, 
                #                  plotting_cases=[("cost_func_tuning", "alpha_0.001"),
                #                                   ("cost_func_tuning", "alpha_0.999")], save_dir=args.save_dir)

                
                # x = agg_df.loc[(agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning") 
                #            & ((agg_df.index.get_level_values("CaseName") == "alpha_0.001") 
                #               | (agg_df.index.get_level_values("CaseName") == "alpha_0.999")), 
                #            [('YawAngleChangeAbsMean', 'mean'), ('FarmPowerMean', 'mean'), 
                #             ('RelativeRunningOptimizationCostTerm_0', 'mean'), ('RelativeRunningOptimizationCostTerm_1', 'mean')]
                #             ].sort_values(by=('FarmPowerMean', 'mean'), ascending=False).reset_index(level="CaseFamily", drop=True)
                # x.columns = x.columns.droplevel(1)
                better_than_lut_df = better_than_lut_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)
                100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - lut_df.iloc[0]["FarmPowerMean"]) / lut_df.iloc[0]["FarmPowerMean"]
                100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - greedy_df.iloc[0]["FarmPowerMean"]) / greedy_df.iloc[0]["FarmPowerMean"]
                
                better_than_lut_df = better_than_lut_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - lut_df.iloc[0]["YawAngleChangeAbsMean"]) / lut_df.iloc[0]["YawAngleChangeAbsMean"]
                100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - greedy_df.iloc[0]["YawAngleChangeAbsMean"]) / greedy_df.iloc[0]["YawAngleChangeAbsMean"]

                100 * (lut_df.iloc[0]["FarmPowerMean"] - greedy_df.iloc[0]["FarmPowerMean"]) / greedy_df.iloc[0]["FarmPowerMean"]
                100 * (lut_df.iloc[0]["YawAngleChangeAbsMean"] - greedy_df.iloc[0]["YawAngleChangeAbsMean"]) / greedy_df.iloc[0]["YawAngleChangeAbsMean"]

                plot_cost_function_pareto_curve(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("scalability") in args.case_ids):
                floris_input_files = case_studies["scalability"]["floris_input_file"]["vals"]
                lut_paths = case_studies["scalability"]["lut_path"]["vals"]
                plot_wind_farm(floris_input_files, lut_paths, args.save_dir)
            
            if case_families.index("breakdown_robustness") in args.case_ids:
                plot_breakdown_robustness(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("horizon_length") in args.case_ids):
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "horizon_length"][
                    [("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]
                    ].sort_values(by=("FarmPowerMean", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)

                config_cols = ["controller_dt", "n_horizon"]
                for (case_family, case_name), _ in mpc_df.iterrows():
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.pkl"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='r') as fp:
                        input_config = pickle.load(fp)
                    
                    for col in config_cols:
                        mpc_df.loc[(mpc_df.index.get_level_values("CaseFamily") == case_family) & (mpc_df.index.get_level_values("CaseName") == case_name), col] = input_config["controller"][col]

                lut_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                greedy_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]]
                 
                plot_horizon_length(pd.concat([mpc_df, lut_df, greedy_df]), args.save_dir)

            if case_families.index("yaw_offset_study") in args.case_ids:
                
                mpc_alpha_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (~agg_df.index.get_level_values("CaseName").str.contains("LUT"))]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "yaw_offset_study") & (agg_df.index.get_level_values("CaseName").str.contains("LUT"))]
                
                if "baseline_controllers_3" in agg_df.index.get_level_values("CaseFamily"):
                    greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_3") & (agg_df.index.get_level_values("CaseName").str.contains("Greedy"))]  
                    
                    better_than_lut_df = mpc_alpha_df.loc[((mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0])
                                                        & (mpc_alpha_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0])), 
                                                        [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]\
                                                            .sort_values(by=("FarmPowerMean", "mean"), ascending=False)\
                                                                .reset_index(level="CaseFamily", drop=True)
                    plotting_cases = [("yaw_offset_study", better_than_lut_df.iloc[0]._name),   
                                                    ("baseline_controllers_3", "LUT_3turb"),
                                                    ("baseline_controllers_3", "Greedy")
                    ]
                    # NOTE USE THIS CALL TO GENERATE TIME SERIES PLOTS
                    plot_simulations(
                        time_series_df, plotting_cases, args.save_dir, include_power=True, legend_loc="outer", single_plot=False) 


                # plot yaw vs wind dir
                # set(time_series_df.loc[time_series_df.index.get_level_values("CaseFamily") == "yaw_offset_study", :].index.get_level_values("CaseName").values)
                case_names = ["LUT_3turb", "StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_11_3turb", 
                              "StochasticIntervalElliptical_11_3turb", "StochasticSample_25_3turb", "StochasticSample_100_3turb"]
                case_labels = ["LUT", "MPC\n1 RI Samples", "MPC\n11 RI Samples", "MPC\n11 EI Samples", "MPC\n25 * S Samples", "MPC\n100 S Samples"]
                plot_yaw_offset_wind_direction(time_series_df, case_names, case_labels,
                                            os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                            os.path.join(args.save_dir, "yaw_offset_study", "yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], 
                                            include_yaw=True, include_power=True, scatter=False)
                
                for sub_case_names, sub_case_labels, filename in zip([["LUT_3turb"], 
                                                                      ["StochasticIntervalRectangular_1_3turb", "StochasticIntervalRectangular_11_3turb", "StochasticIntervalElliptical_11_3turb"],
                                                                      ["StochasticSample_25_3turb", "StochasticSample_100_3turb"]], 
                                                           [["LUT"], ["MPC\n1 * RI Samples", "MPC\n11 * RI Samples", "MPC\n11 * EI Samples"], 
                                                            ["MPC\n25 * Stochastic Samples", "MPC\n100 * Stochastic Samples"]],
                                                           ["lut", "stochastic_interval", "stochastic_sample"]):
                    plot_yaw_offset_wind_direction(time_series_df, sub_case_names, sub_case_labels,
                                                os.path.join(os.path.dirname(whoc.__file__), 
                                                             f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"),
                                                os.path.join(args.save_dir, "yaw_offset_study", 
                                                             f"yawoffset_winddir_{filename}_ts.png"), 
                                                             plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True, scatter=False)

            if (case_families.index("baseline_controllers_3") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids or case_families.index("n_wind_preview_samples") in args.case_ids):
                # find best diff_type, nu, and decay for each sampling type
                 
                if case_families.index("gradient_type") in args.case_ids:
                    MPC_TYPE = "gradient_type"
                elif case_families.index("n_wind_preview_samples") in args.case_ids:
                    MPC_TYPE = "n_wind_preview_samples"

                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == MPC_TYPE][
                    [("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]
                    ].sort_values(by=("FarmPowerMean", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)

                config_cols = ["n_wind_preview_samples", "wind_preview_type", "diff_type", "nu", "decay_type", "max_std_dev", "n_horizon"]
                for (case_family, case_name), _ in mpc_df.iterrows():
                    # input_fn = [fn for fn in os.listdir(os.path.join(args.save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
                    input_fn = f"input_config_case_{case_name}.yaml"
                    with open(os.path.join(args.save_dir, case_family, input_fn), mode='r') as fp:
                        input_config = yaml.safe_load(fp)
                    
                    for col in config_cols:
                        mpc_df.loc[(mpc_df.index.get_level_values("CaseFamily") == case_family) & (mpc_df.index.get_level_values("CaseName") == case_name), col] = input_config["controller"][col]
            
                mpc_df["diff_direction"] = mpc_df["diff_type"].apply(lambda s: s.split("_")[1] if s != "none" else None)
                mpc_df["diff_steps"] = mpc_df["diff_type"].apply(lambda s: s.split("_")[0] if s != "none" else None)
                mpc_df["n_wind_preview_samples_index"] = None

                unique_sir_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples"]))
                unique_sie_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples"]))
                unique_ss_n_samples = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples"]))
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples"].apply(lambda n: np.where(unique_sir_n_samples == n)[0][0]).astype("int").values
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples"].apply(lambda n: np.where(unique_sie_n_samples == n)[0][0]).astype("int").values
                mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples_index"] = mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples"].apply(lambda n: np.where(unique_ss_n_samples == n)[0][0]).astype("int").values
                mpc_df.columns = mpc_df.columns.droplevel(1)

                # read params from input configs rather than CaseName
                lut_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                lut_df.columns = lut_df.columns.droplevel(1)
                greedy_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]]
                greedy_df.columns = greedy_df.columns.droplevel(1)

                # better_than_lut_df = mpc_df.loc[(mpc_df["FarmPowerMean"] > lut_df["FarmPowerMean"].iloc[0]), ["YawAngleChangeAbsMean", "OptimizationConvergenceTime", "FarmPowerMean"] + config_cols].sort_values(by="FarmPowerMean", ascending=False).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df["FarmPowerMean"] > lut_df["FarmPowerMean"].iloc[0]) 
                                                & (mpc_df["YawAngleChangeAbsMean"] < lut_df["YawAngleChangeAbsMean"].iloc[0]), 
                                                ["YawAngleChangeAbsMean", "OptimizationConvergenceTime", "FarmPowerMean"] + config_cols]\
                                                    .sort_values(by="FarmPowerMean", ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut_df.groupby("wind_preview_type").head(3)[["n_wind_preview_samples", "wind_preview_type", "diff_type", "nu", "decay_type", "max_std_dev"]]
                   # better_than_lut_df = better_than_lut_df.reset_index(level="CaseName", drop=True)

                # better_than_lut_df = better_than_lut_df.sort_values("FarmPowerMean", ascending=False)
                # 100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - lut_df.iloc[0]["FarmPowerMean"]) / lut_df.iloc[0]["FarmPowerMean"]
                # 100 * (better_than_lut_df.iloc[0]["FarmPowerMean"] - greedy_df.iloc[0]["FarmPowerMean"]) / greedy_df.iloc[0]["FarmPowerMean"]

                # better_than_lut_df = better_than_lut_df.sort_values("YawAngleChangeAbsMean", ascending=True)
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - lut_df.iloc[0]["YawAngleChangeAbsMean"]) / lut_df.iloc[0]["YawAngleChangeAbsMean"]
                # 100 * (better_than_lut_df.iloc[0]["YawAngleChangeAbsMean"] - greedy_df.iloc[0]["YawAngleChangeAbsMean"]) / greedy_df.iloc[0]["YawAngleChangeAbsMean"]             

                # best_case_names = better_than_lut_df.groupby(["wind_preview_type"])["diff_type"].idxmax()
                # better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)
                better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)\
                                  .loc[better_than_lut_df.index.get_level_values("CaseName").isin(
                                      better_than_lut_df.groupby(["wind_preview_type"])["FarmPowerMean"].idxmax()),
                                      ["wind_preview_type", "diff_type", "decay_type", "max_std_dev", "nu"]]

                for param in ["diff_type", "decay_type", "max_std_dev", "nu"]:
                    for agg_type in ["mean", "max"]: 
                        print(f"\nFor parameter {param}, taking the {agg_type} of FarmPowerMean over all other parameters, the best parameter for each wind_preview_type is:")
                        print(better_than_lut_df.drop(["n_wind_preview_samples", "n_horizon"], axis=1)\
                                        .groupby(["wind_preview_type", param])["FarmPowerMean"].agg(agg_type)\
                                        .groupby("wind_preview_type").idxmax().values)
                
                                #   .loc[better_than_lut_df.index.get_level_values("CaseName").isin(
                                #       better_than_lut_df.groupby(["wind_preview_type"])["FarmPowerMean"].idxmax()),
                                #       ["wind_preview_type", "diff_type", "decay_type", "max_std_dev", "nu"]]

                if True:
                    plot_parameter_sweep(pd.concat([mpc_df, lut_df, greedy_df]), MPC_TYPE, args.save_dir, 
                                         plot_columns=["FarmPowerMean", "diff_type", "decay_type", "max_std_dev", "n_wind_preview_samples", "wind_preview_type", "nu"],
                                         merge_wind_preview_types=False, estimator="mean")
                
                plotting_cases = [(MPC_TYPE, better_than_lut_df.sort_values(by="FarmPowerMean", ascending=False).iloc[0]._name),   
                                                ("baseline_controllers_3", "LUT"),
                                                ("baseline_controllers_3", "Greedy")
                ]

                plot_simulations(
                    time_series_df, plotting_cases, args.save_dir, include_power=True, legend_loc="outer", single_plot=False) 


                # find best power decay type
                # power_decay_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "power_decay_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)

            if case_families.index("wind_preview_type") in args.case_ids:
                # TODO get best parameters from each sweep and add to other sweeps, then rerun to compare with LUT
                # find best wind_preview_type and number of samples, if best is on the upper end, increase n_wind_preview_samples in wind_preview_type sweep
                wind_preview_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "wind_preview_type"][[("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)


            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("gradient_type") in args.case_ids):
               
                mpc_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "gradient_type", :]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                
                # get mpc configurations for which the generated farm power is greater than lut, and the resulting yaw actuation lesser than lut
                # better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), [("YawAngleChangeAbsMean", "mean"), ("OptimizationConvergenceTime", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)
                # better_than_lut = pd.read_csv(os.path.join(args.save_dir, "better_than_lut.csv"), header=[0,1], index_col=[0], skipinitialspace=True)
                better_than_lut_df.to_csv(os.path.join(args.save_dir, "better_than_lut.csv"))
                # better_than_lut_df = mpc_df.loc[(mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("YawAngleChangeAbsMean", "mean"), ("RelativeTotalRunningOptimizationCostMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                
                # get mpc configurations for which the generated farm power is greater than greedy
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                # better_than_greedy_df = better_than_greedy_df.loc[better_than_greedy_df.index.isin(better_than_lut_df.index)]
                # better_than_lut_df.loc[better_than_lut_df.index.isin(better_than_greedy_df.index)]
                # greedy warm start better,
                
                # lut_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                # greedy_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                # mpc_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].reset_index(level="CaseFamily", drop=True)
                # mpc_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].iloc[0]
                # print(better_than_lut_df.iloc[0]._name)
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                # 100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                # 100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                # 100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                # plot multibar of farm power vs. stochastic interval n_wind_preview_samples, stochastic sample n_wind_preview_samples
                # 

                # alpha_1.0_controller_class_MPC_diff_type_chain_cd_dt_15_n_horizon_24_n_wind_preview_samples_7_nu_0.001_


            if all(case_families.index(cf) in args.case_ids for cf in ["baseline_controllers", "solver_type",
             "wind_preview_type", "warm_start"]):
                generate_outputs(agg_df, args.save_dir)       

            if case_families.index("baseline_controllers_preview_flasc_perfect") in args.case_ids \
                or case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:

                if case_families.index("baseline_controllers_preview_flasc_perfect") in args.case_ids:
                    mpc_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_preview_flasc_perfect", :]
                elif case_families.index("baseline_controllers_perfect_forecaster_awaken") in args.case_ids:
                    mpc_df = agg_df.loc[agg_df.index.get_level_values("CaseFamily") == "baseline_controllers_perfect_forecaster_awaken", :]

                config_cols = ["wind_forecast_class", "prediction_timedelta"]

                for (case_family, case_name), _ in mpc_df.iterrows():
                    input_fn = f"input_config_case_{case_name}.pkl"
                    input_path = os.path.join(args.save_dir, case_family, input_fn)

                    with open(input_path, mode='rb') as fp:
                        input_config = pickle.load(fp)

                    controller_config = input_config.get("controller", {})
                    wind_forecast_config = input_config.get("wind_forecast", {})
                    full_config = {**controller_config, **wind_forecast_config}
                    
                    for col in config_cols:
                        mpc_df.loc[
                            (mpc_df.index.get_level_values("CaseFamily") == case_family) & 
                            (mpc_df.index.get_level_values("CaseName") == case_name), 
                            col
                        ] = full_config[col]  

                # Filter data for the two forecast types
                forecasters_df = mpc_df.loc[mpc_df["wind_forecast_class"] != "PerfectForecast", :]
                perfect_df = mpc_df.loc[mpc_df["wind_forecast_class"] == "PerfectForecast", :]

                if "prediction_timedelta" in forecasters_df.columns and "prediction_timedelta" in perfect_df.columns:
                    merged_df = forecasters_df.merge(
                        perfect_df,
                        on=["CaseFamily", "prediction_timedelta"],
                        suffixes=("_kalman", "_perfect")
                    )


                merged_df["power_ratio"] = (merged_df["FarmPowerMean_kalman", "mean"] / merged_df["FarmPowerMean_perfect", "mean"]) * 100

                plot_df = merged_df[["prediction_timedelta", "power_ratio"]]

                # Display the prepared data (for debugging)
                print(plot_df.head())
                plot_power_increase_vs_prediction_time(plot_df, args.save_dir)