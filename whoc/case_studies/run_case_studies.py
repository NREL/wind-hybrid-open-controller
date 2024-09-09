import os
import numpy as np
import pandas as pd
import re
import sys

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import read_time_series_data, aggregate_time_series_data, generate_outputs, plot_simulations, plot_wind_farm, plot_breakdown_robustness, plot_cost_function_pareto_curve, plot_yaw_offset_wind_direction, plot_parameter_sweep

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

import argparse
# np.seterr("raise")

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
    parser.add_argument("-st", "--stoptime", type=float, default=3600)
    parser.add_argument("-ns", "--n_seeds", type=int, default=6)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"])
    parser.add_argument("-sd", "--save_dir", type=str)
   
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
    if args.run_simulations:
        # run simulations
        
        if RUN_ONCE:
            print(f"running initialize_simulations for case_ids {[case_families[i] for i in args.case_ids]}")
            
            case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in args.case_ids], regenerate_wind_field=args.generate_wind_field, regenerate_lut=args.generate_lut, n_seeds=args.n_seeds, stoptime=args.stoptime, save_dir=args.save_dir)
        
        if args.multiprocessor is not None:
            if args.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                # comm_rank = MPI.COMM_WORLD.Get_rank()
                # node_name = MPI.Get_processor_name()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                # executor = MPIPoolExecutor(max_workers=mp.cpu_count(), root=0)
            elif args.multiprocessor == "cf":
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                if args.multiprocessor == "mpi":
                    run_simulations_exec.max_workers = comm_size
                  
                print(f"run_simulations line 64 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]],
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_ or type(val) is bool) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
                        for c, d in enumerate(input_dicts)]
                
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_ or type(val) is bool) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                                wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
    
    if args.postprocess_simulations:
        # if (not os.path.exists(os.path.join(args.save_dir, f"time_series_results.csv"))) or (not os.path.exists(os.path.join(args.save_dir, f"agg_results.csv"))):
        # regenerate some or all of the time_series_results_all and agg_results_all .csv files for each case family in case ids
        if args.reprocess_simulations:
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
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv"))
                    ]
                    
                    new_time_series_df = [fut.result() for fut in read_futures]
                    existing_time_series_df = []
                    for i in args.case_ids:
                        all_ts_df_path = os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv")                        # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                        
                        if not args.reaggregate_simulations and os.path.exists(os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv")):
                            existing_time_series_df.append(pd.read_csv(all_ts_df_path, index_col=[0, 1]))
                    
                    time_series_df = pd.concat(existing_time_series_df + new_time_series_df)

                    # if args.reaggregate_simulations is true, or for any case family where doesn't agg_results_all.csv exist, compute the aggregate stats for each case families and case name, over all wind seeds
                    futures = [run_simulations_exec.submit(aggregate_time_series_data,
                                                             time_series_df=time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i]) & (time_series_df.index.get_level_values("CaseName") == case_name), :],
                                                                # time_series_path=os.path.join(args.save_dir, case_families[i], fn),
                                                                # results_path=os.path.join(args.save_dir, case_families[i], f"agg_results_{case_name}.csv"), 
                                                                yaml_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.yaml"),
                                                                n_seeds=args.n_seeds)
                        for i in args.case_ids 
                        for case_name in pd.unique(time_series_df.iloc[(time_series_df.index.get_level_values("CaseFamily") == case_families[i])].index.get_level_values("CaseName"))
                        # for case_name in [re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]
                        if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv"))
                    ]

                    new_agg_df = [fut.result() for fut in futures]
                    # for i in args.case_ids:
                    #     if len(new_agg_df) and not all([df is None for df in new_agg_df]):
                    #         new_agg_df = pd.concat([df for df in new_agg_df if df is not None])
                    #         all_agg_df_path = os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv")
                    #         new_agg_df.loc[new_agg_df.index.get_level_values("CaseFamily") == case_families[i], :].to_csv(all_agg_df_path)
                    #     else:
                    #         new_agg_df = pd.DataFrame()
                    
            # else, run sequentially
            else:
                new_time_series_df = []
                existing_time_series_df = []
                for i in args.case_ids:
                    all_ts_df_path = os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv")
                    new_case_family_time_series_df = []

                    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv")):
                        for fn in case_family_case_names[case_families[i]]:
                            new_case_family_time_series_df.append(read_time_series_data(results_path=os.path.join(args.save_dir, case_families[i], fn)))
                    else:
                        existing_time_series_df.append(pd.read_csv(all_ts_df_path, index_col=[0, 1]))
                    
                    # if any new time series data has been read, add it to the new_time_series_df list and save the aggregated time-series data
                    if len(new_time_series_df):
                        new_time_series_df.append(pd.concat(new_case_family_time_series_df))
                        new_time_series_df[-1].to_csv(all_ts_df_path)   
                
                time_series_df = pd.concat(existing_time_series_df + new_time_series_df)
                
                new_agg_df = []
                for i in args.case_ids:
                    if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv")):
                        # for case_name in set([re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]):
                        case_family_df = time_series_df.iloc[time_series_df.index.get_level_values("CaseFamily") == case_families[i], :]
                        for case_name in pd.unique(case_family_df.index.get_level_values("CaseName")):
                            case_name_df = case_family_df.iloc[time_series_df.index.get_level_values("CaseName") == case_name, :]
                            res = aggregate_time_series_data(
                                                            time_series_df=case_name_df,
                                                             yaml_path=os.path.join(args.save_dir, case_families[i], f"input_config_case_{case_name}.yaml"),
                                                            # results_path=os.path.join(args.save_dir, case_families[i], f"agg_results_{case_name}.csv"),
                                                            n_seeds=args.n_seeds)
                            if res is not None:
                                new_agg_df.append(res)

            if RUN_ONCE:
                if len(new_agg_df) and not all([df is None for df in new_agg_df]):
                    new_agg_df = pd.concat([df for df in new_agg_df if df is not None])
                else:
                    new_agg_df = pd.DataFrame()
                 # if args.reaggregate_simulations is false, read the remaining aggregate data from each agg_results_all csv file
                existing_agg_df = []
                for i in args.case_ids:
                    # if args.reaggregate_simulations or not os.path.exists(os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv")):
                    all_agg_df_path = os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv")
                    if not args.reaggregate_simulations and os.path.exists(all_agg_df_path):
                        existing_agg_df.append(pd.read_csv(all_agg_df_path, header=[0,1], index_col=[0, 1], skipinitialspace=True))
                    else:
                        new_agg_df.loc[new_agg_df.index.get_level_values("CaseFamily") == case_families[i], :].to_csv(all_agg_df_path)   

                agg_df = pd.concat(existing_agg_df + [new_agg_df])
                
                # for each case family, 
                # for i in args.case_ids:
                #     # save the updated time_series_results_all.csv
                #     all_ts_results_path = os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv")
                #     # if not len(time_series_df):
                #     #     time_series_df = pd.read_csv(all_ts_results_path, index_col=[0, 1]) 
                #     # elif case_families[i] not in time_series_df.index.get_level_values("CaseFamily") and os.path.exists(all_ts_results_path):
                #     #     time_series_df = pd.concat([time_series_df, 
                #     #                         pd.read_csv(all_ts_results_path, index_col=[0, 1])]) 
                #     # else:
                #     time_series_df.loc[time_series_df.index.get_level_values("CaseFamily") == case_families[i], :].to_csv(all_ts_results_path)    
                    
                #     # if any agg_results_all files were not newly generated above, add them to the agg_df dataframe now 
                #     all_agg_results_path = os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv")
                #     if not len(agg_df):
                #         agg_df = pd.read_csv(all_agg_results_path, header=[0,1], index_col=[0, 1], skipinitialspace=True) 
                #     elif case_families[i] not in agg_df.index.get_level_values("CaseFamily") and os.path.exists(all_agg_results_path):
                #         agg_df = pd.concat([agg_df, 
                #                             pd.read_csv(all_agg_results_path, header=[0,1], index_col=[0, 1], skipinitialspace=True)])
                #     else:
                #         agg_df.loc[agg_df.index.get_level_values("CaseFamily") == case_families[i], :].to_csv(all_agg_results_path)

        elif RUN_ONCE:
            time_series_df = pd.concat([pd.read_csv(os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv"), index_col=[0, 1]) for i in args.case_ids if os.path.exists(os.path.join(args.save_dir, case_families[i], f"time_series_results_all.csv"))])
            agg_df = pd.concat([pd.read_csv(os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv"), header=[0,1], index_col=[0, 1], skipinitialspace=True) for i in args.case_ids if os.path.exists(os.path.join(args.save_dir, case_families[i], f"agg_results_all.csv"))])

        if RUN_ONCE and PLOT:
            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("cost_func_tuning") in args.case_ids):
                
                mpc_alpha_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning"]
                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")]
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_alpha_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)

                plot_simulations(time_series_df, [("cost_func_tuning", "alpha_0.001"),
                                                  ("cost_func_tuning", "alpha_0.999")], args.save_dir)
                
                agg_df.loc[agg_df.index.get_level_values("CaseFamily") == "cost_func_tuning", [('YawAngleChangeAbsMean', 'mean'), ('FarmPowerMean', 'mean')]].sort_values(by=('FarmPowerMean', 'mean'), ascending=False)
                plot_cost_function_pareto_curve(agg_df, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("scalability") in args.case_ids):
                floris_input_files = case_studies["scalability"]["floris_input_file"]["vals"]
                lut_paths = case_studies["scalability"]["lut_path"]["vals"]
                plot_wind_farm(floris_input_files, lut_paths, args.save_dir)
            
            if case_families.index("breakdown_robustness") in args.case_ids:
                plot_breakdown_robustness(agg_df, args.save_dir)

            if case_families.index("yaw_offset_study") in args.case_ids:
                # plot yaw vs wind dir
                case_names = ["LUT_3turb", "StochasticInterval_1_3turb", "StochasticInterval_5_3turb", "StochasticSample_25_3turb", "StochasticSample_100_3turb"]
                case_labels = ["LUT", "MPC\n1 * Interval Samples", "MPC\n5 * Interval Samples", "MPC\n25 * Stochastic Samples", "MPC\n100 * Stochastic Samples"]
                plot_yaw_offset_wind_direction(time_series_df, case_names, case_labels,
                                            os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                            os.path.join(args.save_dir, "yaw_offset_study", f"yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)
                
                for sub_case_names, sub_case_labels, filename in zip([["LUT_3turb"], ["StochasticInterval_1_3turb", "StochasticInterval_5_3turb"], ["StochasticSample_25_3turb", "StochasticSample_100_3turb"]], 
                                                           [["LUT"], ["MPC\n1 * Interval Samples", "MPC\n5 * Interval Samples"], ["MPC\n25 * Stochastic Samples", "MPC\n100 * Stochastic Samples"]],
                                                           ["lut", "stochastic_interval", "stochastic_sample"]):
                    plot_yaw_offset_wind_direction(time_series_df, sub_case_names, sub_case_labels,
                                                os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                                os.path.join(args.save_dir, "yaw_offset_study", f"yawoffset_winddir_{filename}_ts.png"), plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)

            if case_families.index("baseline_controllers") in args.case_ids and case_families.index("gradient_type") in args.case_ids:
                # find best diff_type, nu, and decay for each sampling type
                gradient_type_df = agg_df.iloc[agg_df.index.get_level_values("CaseFamily")  == "gradient_type"][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False) #.reset_index(level="CaseFamily", drop=True)
                gradient_type_df["WindPreviewType"] = [re.findall(r"(?<=wind_preview_type_)(.*?)(?=$)", s)[0] for s in gradient_type_df.index.get_level_values("CaseName")]

                lut_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "LUT")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]] 
                greedy_df = agg_df.iloc[(agg_df.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_df.index.get_level_values("CaseName") == "Greedy")][[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]]

                better_than_lut_df = gradient_type_df.loc[(gradient_type_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), [("YawAngleChangeAbsMean", "mean"), ("OptimizationConvergenceTime", "mean"), ("FarmPowerMean", "mean"), ("WindPreviewType", "")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True).groupby("WindPreviewType").head(3)
                better_than_lut_df = better_than_lut_df.reset_index(level="CaseName", drop=False)
                better_than_lut_df["nu"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=nu_)(.*?)(?=_solver)", s)[0]).astype("float")
                better_than_lut_df["n_wind_preview_samples"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=n_wind_preview_samples_)(.*?)(?=_nu)", s)[0]).astype("int")
                better_than_lut_df["preview_type"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=wind_preview_type_)(.*?)(?=$)", s)[0])
                better_than_lut_df["decay_type"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=decay_type_)(.*?)(?=_diff)", s)[0])
                better_than_lut_df["diff_type"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=diff_type_)(.*?)(?=_dt)", s)[0])
                better_than_lut_df["max_std_dev"] = better_than_lut_df["CaseName"].apply(lambda s: re.findall(r"(?<=max_std_dev_)(.*?)(?=_n_horizon)", s)[0])

                if False:
                    plot_parameter_sweep(pd.concat([gradient_type_df, lut_df, greedy_df]), args.save_dir)
                
                plotting_cases = [("gradient_type", better_than_lut_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False).iloc[0]._name),   
                                                ("baseline_controllers", "LUT"),
                                                ("baseline_controllers", "Greedy")
                ]

                plot_simulations(
                    time_series_df, plotting_cases, args.save_dir, include_power=False, legend_loc="outer", single_plot=False) 


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
                better_than_lut_df.loc[better_than_lut_df.index.isin(better_than_greedy_df.index)]
                # greedy warm start better,
                
                lut_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                greedy_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                mpc_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].reset_index(level="CaseFamily", drop=True)
                mpc_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean"), ("OptimizationConvergenceTime", "mean")]].iloc[0]
                print(better_than_lut_df.iloc[0]._name)
                100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                100 * (better_than_lut_df.loc[better_than_lut_df.index == "alpha_1.0_controller_class_MPC_diff_type_custom_cd_dt_30_n_horizon_24_n_wind_preview_samples_5_nu_0.01_solver_slsqp_use_filtered_wind_dir_False_wind_preview_type_stochastic_interval", ("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - lut_df.iloc[0][("FarmPowerMean", "mean")]) / lut_df.iloc[0][("FarmPowerMean", "mean")]
                100 * (better_than_lut_df.iloc[0][("FarmPowerMean", "mean")] - greedy_df.iloc[0][("FarmPowerMean", "mean")]) / greedy_df.iloc[0][("FarmPowerMean", "mean")]
                
                # plot multibar of farm power vs. stochastic interval n_wind_preview_samples, stochastic sample n_wind_preview_samples
                # 

                # alpha_1.0_controller_class_MPC_diff_type_chain_cd_dt_15_n_horizon_24_n_wind_preview_samples_7_nu_0.001_


            if all(case_families.index(cf) in args.case_ids for cf in ["baseline_controllers", "solver_type",
             "wind_preview_type", "warm_start", 
              "horizon_length", "scalability"]):
                generate_outputs(agg_df, args.save_dir)