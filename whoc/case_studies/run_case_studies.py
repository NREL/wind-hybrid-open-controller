import os
import numpy as np

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import process_simulations, plot_simulations

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_case_studies.py", description="Run FLORIS case studies for WHOC module.")
    parser.add_argument("case_ids", metavar="C", nargs="+", choices=[str(i) for i in range(len(case_families))])
    parser.add_argument("-gwf", "--generate_wind_field", action="store_true")
    parser.add_argument("-glut", "--generate_lut", action="store_true")
    parser.add_argument("-rs", "--run_simulations", action="store_true")
    parser.add_argument("-ps", "--postprocess_simulations", action="store_true")
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("-st", "--stoptime", type=float, default=3600)
    parser.add_argument("-ns", "--n_seeds", type=int, default=6)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"], default="mpi")
    parser.add_argument("-sd", "--save_dir", type=str)
    # "/projects/ssc/ahenry/whoc/floris_case_studies" on kestrel
    # "/projects/aohe7145/whoc/floris_case_studies" on curc
    # "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies" on mac
    # python run_case_studies.py 0 1 2 3 4 5 6 7 -rs -p -st 480 -ns 1 -m cf -sd "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies"
    args = parser.parse_args()
    args.case_ids = [int(i) for i in args.case_ids]

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": max(d["group"] for d in case_studies[case_family].values()) + 1, "vals": [i for i in range(args.n_seeds)]}

    os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "true"
    
    print([case_families[i] for i in args.case_ids])
    if args.run_simulations:
        # print(55)
        # run simulations
        
        if (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or not args.parallel:
            print(f"running initialize_simulations")
        
            case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in args.case_ids], regenerate_wind_field=args.generate_wind_field, regenerate_lut=args.generate_lut, n_seeds=args.n_seeds, stoptime=args.stoptime, save_dir=args.save_dir)
        
        # print(62)
        if args.parallel:
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
                    
                print(f"run_simulations line 78 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_) else np.round(val, 5)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir)
                        for c, d in enumerate(input_dicts)]
                # wait(futures)
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_) else np.round(val, 5)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"],
                                                wind_field_config=wind_field_config, verbose=False, save_dir=args.save_dir)
        
    # print(97)
    if (args.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multi_processor != "mpi"):
        if args.postprocess_simulations:
            results_dirs = [os.path.join(args.save_dir, case_families[i]) for i in args.case_ids]
            
            # compute stats over all seeds
            process_simulations(results_dirs, case_families, args.save_dir)
            
            plot_simulations(results_dirs, args.save_dir)