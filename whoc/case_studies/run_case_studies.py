import os
import numpy as np
import pandas as pd
import re

import whoc
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import read_time_series_data, aggregate_time_series_data, generate_outputs, plot_simulations, plot_wind_farm, plot_breakdown_robustness, plot_cost_function_pareto_curve, plot_yaw_offset_wind_direction

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

import argparse
np.seterr("raise")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_case_studies.py", description="Run FLORIS case studies for WHOC module.")
    parser.add_argument("case_ids", metavar="C", nargs="+", choices=[str(i) for i in range(len(case_families))])
    parser.add_argument("-gwf", "--generate_wind_field", action="store_true")
    parser.add_argument("-glut", "--generate_lut", action="store_true")
    parser.add_argument("-rs", "--run_simulations", action="store_true")
    parser.add_argument("-rrs", "--rerun_simulations", action="store_true")
    parser.add_argument("-ps", "--postprocess_simulations", action="store_true")
    parser.add_argument("-rps", "--reprocess_simulations", action="store_true")
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
                #TODO check that case_family is being fetched properly    
                print(f"run_simulations line 64 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_ or type(val) is bool) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file", "use_filtered_wind_dir", "dt"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
                        for c, d in enumerate(input_dicts)]
                
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_ or type(val) is bool) else np.round(val, 6)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed", "lut_path", "floris_input_file", "use_filtered_wind_dir", "dt"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"],
                                                wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir, rerun_simulations=args.rerun_simulations)
    
    if args.postprocess_simulations:

        if args.reprocess_simulations or (not os.path.exists(os.path.join(args.save_dir, f"time_series_results.csv"))) or (not os.path.exists(os.path.join(args.save_dir, f"agg_results.csv"))):
            if RUN_ONCE:
                case_family_case_names = {}
                for i in args.case_ids:
                    case_family_case_names[case_families[i]] = [fn for fn in os.listdir(os.path.join(args.save_dir, case_families[i])) if ".csv" in fn]

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

                    read_futures = [run_simulations_exec.submit(
                                                    read_time_series_data, 
                                                    results_path=os.path.join(args.save_dir, case_families[i], fn))
                        for i in args.case_ids 
                        for fn in case_family_case_names[case_families[i]]
                    ]
                    
                    time_series_df = pd.concat([fut.result() for fut in read_futures])
                    
                    agg_futures = [run_simulations_exec.submit(aggregate_time_series_data,
                                                            case_df=time_series_df.loc[(time_series_df["CaseFamily"] == case_families[i]) & (time_series_df["CaseName"] == case_name), :],
                                                                save_dir=args.save_dir)
                        for i in args.case_ids  
                        for case_name in [re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]
                    ]
                    
                    agg_dfs = pd.concat([fut.result() for fut in agg_futures])

            else:
                time_series_df = []
                for i in args.case_ids:
                    for fn in case_family_case_names[case_families[i]]:
                        time_series_df.append(read_time_series_data(results_path=os.path.join(args.save_dir, case_families[i], fn)))
                time_series_df = pd.concat(time_series_df)

                agg_dfs = []
                for i in args.case_ids:
                    for case_name in [re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] for fn in case_family_case_names[case_families[i]]]:
                        agg_dfs.append(aggregate_time_series_data(case_df=time_series_df.loc[(time_series_df["CaseFamily"] == case_families[i]) & (time_series_df["CaseName"] == case_name), :],
                                                            save_dir=args.save_dir))
                agg_dfs = pd.concat(agg_dfs)

            if RUN_ONCE:
                
                time_series_df = time_series_df.reset_index(drop=True) 
                time_series_df.to_csv(os.path.join(args.save_dir, f"time_series_results.csv"))
                
                agg_dfs = agg_dfs.reset_index(drop=True)
                agg_dfs = agg_dfs.groupby(by=["CaseFamily", "CaseName"])[[col for col in agg_dfs.columns if col not in ["CaseFamily", "CaseName", "WindSeed"]]].agg(["min", "max", "mean"])
                agg_dfs.to_csv(os.path.join(args.save_dir, f"agg_results.csv"))

        else:
            if RUN_ONCE:
                time_series_df = pd.read_csv(os.path.join(args.save_dir, f"time_series_results.csv"), index_col=0)
                agg_dfs = pd.read_csv(os.path.join(args.save_dir, f"agg_results.csv"), header=[0,1], index_col=[0, 1], skipinitialspace=True)

        if RUN_ONCE:
            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("cost_func_tuning") in args.case_ids):
                
                mpc_alpha_df = agg_dfs.iloc[agg_dfs.index.get_level_values("CaseFamily") == "cost_func_tuning"]
                lut_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_dfs.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_dfs.index.get_level_values("CaseName") == "Greedy")]
                better_than_lut_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_alpha_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_greedy_df = mpc_alpha_df.loc[(mpc_alpha_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)

                plot_simulations(time_series_df, [("cost_func_tuning", "alpha_0.001"),
                                                  ("cost_func_tuning", "alpha_0.999")], args.save_dir)
                
                agg_dfs.loc[agg_dfs.index.get_level_values("CaseFamily") == "cost_func_tuning", [('YawAngleChangeAbsMean', 'mean'), ('FarmPowerMean', 'mean')]].sort_values(by=('FarmPowerMean', 'mean'), ascending=False)
                plot_cost_function_pareto_curve(agg_dfs, args.save_dir)

            if (case_families.index("baseline_controllers") in args.case_ids) and (case_families.index("scalability") in args.case_ids):
                floris_input_files = case_studies["scalability"]["floris_input_file"]["vals"]
                lut_paths = case_studies["scalability"]["lut_path"]["vals"]
                plot_wind_farm(floris_input_files, lut_paths, args.save_dir)
            
            if case_families.index("breakdown_robustness") in args.case_ids:
                plot_breakdown_robustness(agg_dfs, args.save_dir)

            if case_families.index("yaw_offset_study") in args.case_ids:
                # plot yaw vs wind dir
                case_names = ["LUT_3turb", "StochasticInterval_1_3turb", "StochasticInterval_5_3turb", "StochasticSample_25_3turb"]
                case_labels = ["LUT", "MPC, mean wind preview", "Stochastic, 5 interval wind preview", "Stochastic, 25 sample wind preview"]
                plot_yaw_offset_wind_direction(time_series_df, case_names, case_labels,
                                            os.path.join(os.path.dirname(whoc.__file__), f"../examples/mpc_wake_steering_florisstandin/lookup_tables/lut_{3}.csv"), 
                                            os.path.join(args.save_dir, "yaw_offset_study", f"yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)
            
            if (case_families.index("baseline_controllers") in args.case_ids) and ((case_families.index("slsqp_solver_sweep") in args.case_ids) or (case_families.index("slsqp_solver_sweep_small") in args.case_ids)):
                
                mpc_df = agg_dfs.iloc[agg_dfs.index.get_level_values("CaseFamily")  == "slsqp_solver_sweep_small", :] # TODO 
                lut_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_dfs.index.get_level_values("CaseName") == "LUT")] 
                greedy_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily") == "baseline_controllers") & (agg_dfs.index.get_level_values("CaseName") == "Greedy")]
                # get mpc configurations for which the generated farm power is greater than lut, and the resulting yaw actuation lesser than lut
                # better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("FarmPowerMean", "mean"), ascending=False).reset_index(level="CaseFamily", drop=True)
                better_than_lut_df = mpc_df.loc[(mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                # print(mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < greedy_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), ("RelativeTotalRunningOptimizationCostMean", "mean")].sort_values(ascending=True))
                # get mpc configurations for which the generated farm power is greater than greedy
                better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
                # better_than_greedy_df = better_than_greedy_df.loc[better_than_greedy_df.index.isin(better_than_lut_df.index)]
                better_than_lut_df.loc[better_than_lut_df.index.isin(better_than_greedy_df.index)]
                # greedy warm start better,
                time_series_df.loc[(time_series_df["CaseFamily"] == "slsqp_solver_sweep_small") & (time_series_df["CaseName"] == "PerfectCDNormCost") & (time_series_df["WindSeed"] == 0) & (time_series_df["Time"] >= 180.0), [f"TurbineYawAngle_{i}" for i in range(3)]].min(axis=0)
                time_series_df.loc[(time_series_df["CaseFamily"] == "slsqp_solver_sweep_small") & (time_series_df["CaseName"] == "PerfectCDNormCost") & (time_series_df["WindSeed"] == 0) & (time_series_df["Time"] >= 180.0), [f"TurbineYawAngle_{i}" for i in range(3)]].max(axis=0)
                lut_df[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                mpc_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]]
                mpc_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[[("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].iloc[0]
                print(better_than_lut_df.iloc[0]._name)

                plot_simulations(time_series_df, [
                    # ("slsqp_solver_sweep_small", "PerfectCDSimpleCost"),

                                                #   ("slsqp_solver_sweep_small", "PerfectCDNormCost"),
                    ("slsqp_solver_sweep_small", better_than_lut_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False).iloc[0]._name),
                    # ("slsqp_solver_sweep_small", "alpha_1.0_controller_class_MPC_decay_type_linear_n_wind_preview_samples_1_nu_0.01_solver_slsqp_warm_start_lut_wind_preview_type_perfect"),
                                                  ("baseline_controllers", "LUT"),
                                                  ("baseline_controllers", "Greedy")], args.save_dir)

            if all(case_families.index(cf) in args.case_ids for cf in ["baseline_controllers", "solver_type",
             "wind_preview_type", "warm_start", 
              "horizon_length", "scalability"]):
                generate_outputs(agg_dfs, args.save_dir)