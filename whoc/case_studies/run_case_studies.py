import os
import sys

import numpy as np

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies, STORAGE_DIR
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import process_simulations, plot_simulations

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
# import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    # TODO make these true command line arguments
    REGENERATE_WIND_FIELD = False
    REGENERATE_LUT = False
    RUN_SIMULATIONS = False
    POST_PROCESS = False

    DEBUG = sys.argv[1].lower() == "debug"
    if sys.argv[2].lower() == "mpi":
        MULTI = "mpi"
    else:
        MULTI = "cf"

    PARALLEL = sys.argv[3].lower() == "parallel"
    if len(sys.argv) > 4:
        CASE_FAMILY_IDX = [int(i) for i in sys.argv[4:]]
    else:
        CASE_FAMILY_IDX = list(range(len(case_families)))

    if DEBUG:
        N_SEEDS = 1
    else:
        N_SEEDS = 6

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": max(d["group"] for d in case_studies[case_family].values()) + 1, "vals": [i for i in range(N_SEEDS)]}

    os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "true"
    
    print([case_families[i] for i in CASE_FAMILY_IDX])
    if RUN_SIMULATIONS:
        # print(55)
        # run simulations
        
        if (MULTI == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTI != "mpi") or not PARALLEL:
            print(f"running initialize_simulations")
        
            case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in CASE_FAMILY_IDX], regenerate_wind_field=REGENERATE_WIND_FIELD, regenerate_lut=REGENERATE_LUT, n_seeds=N_SEEDS, debug=DEBUG)
        
        # print(62)
        if PARALLEL:
            if MULTI == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                # comm_rank = MPI.COMM_WORLD.Get_rank()
                # node_name = MPI.Get_processor_name()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                # executor = MPIPoolExecutor(max_workers=mp.cpu_count(), root=0)
            elif MULTI == "cf":
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                if MULTI == "mpi":
                    run_simulations_exec.max_workers = comm_size
                    
                print(f"run_simulations line 78 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_) else np.round(val, 5)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
                        for c, d in enumerate(input_dicts)]
                # wait(futures)
                _ = [fut.result() for fut in futures]

        else:
            for c, d in enumerate(input_dicts):
                simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name="_".join([f"{key}_{val if (type(val) is str or type(val) is np.str_) else np.round(val, 5)}" for key, val in case_lists[c].items() if key not in ["wind_case_idx", "seed"]]) if "case_names" not in case_lists[c] else case_lists[c]["case_names"], 
                                                case_family="_".join(case_name_lists[c].split("_")[:-1]), seed=case_lists[c]["seed"],
                                                wind_field_config=wind_field_config, verbose=False)
        
    # print(97)
    if (MULTI == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTI != "mpi"):
        if POST_PROCESS:
            results_dirs = [os.path.join(STORAGE_DIR, case_families[i]) for i in CASE_FAMILY_IDX]
            
            # compute stats over all seeds
            process_simulations(results_dirs, case_families, STORAGE_DIR)
            
            plot_simulations(results_dirs, STORAGE_DIR)