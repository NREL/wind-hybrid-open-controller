from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

import pickle
import pandas as pd
import numpy as np
import os
from time import perf_counter
import sys

from whoc import __file__ as whoc_file
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_studies, STORAGE_DIR, case_families

def simulate_controller(controller_class, input_dict, **kwargs):
    print(f"Running instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    # Load a FLORIS object for AEP calculations
    greedy_fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                          offline_probability=input_dict["controller"]["offline_probability"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
    fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                        offline_probability=input_dict["controller"]["offline_probability"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
    
    kwargs["wind_field_config"]["n_preview_steps"] = input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] / input_dict["dt"])

    ctrl = controller_class(fi, input_dict=input_dict, **kwargs)
    # TODO use coroutines or threading for hercules interfaces
    # optionally warm-start with LUT solution
    
    yaw_angles_ts = []
    yaw_angles_change_ts = []
    turbine_powers_ts = []
    turbine_wind_mag_ts = []
    turbine_wind_dir_ts = []
    turbine_offline_status_ts = []

    greedy_turbine_powers_ts = []

    convergence_time_ts = []
    # opt_codes_ts = []

    opt_cost_ts = []
    opt_cost_terms_ts = []

    n_time_steps = int(kwargs["wind_field_config"]["simulation_max_time"] / input_dict["dt"])
    greedy_fi.env.set(wind_speeds=kwargs["wind_mag_ts"][:n_time_steps], 
                    wind_directions=kwargs["wind_dir_ts"][:n_time_steps],
                    turbulence_intensities=[greedy_fi.env.core.flow_field.turbulence_intensities[0]] * n_time_steps,
                    yaw_angles=np.zeros((1, ctrl.n_turbines)))
    greedy_fi.env.run()
    greedy_turbine_powers_ts = np.max(greedy_fi.env.get_turbine_powers(), axis=1)

    n_future_steps = int(ctrl.dt // input_dict["dt"]) - 1
    
    # for k, t in enumerate(np.arange(0, kwargs["wind_field_config"]["simulation_max_time"], input_dict["dt"])):
    t = 0
    k = 0
    
    while t < kwargs["wind_field_config"]["simulation_max_time"]:

        # recompute controls and step floris forward by ctrl.dt
        # reiniitialize FLORIS interface with for current disturbances and disturbance up to (and excluding) next controls computation
        
        # consider yaw angles as most recently sent from last time-step
        fi.step(disturbances={"wind_speeds": kwargs["wind_mag_ts"][k:k + n_future_steps + 1],
                            "wind_directions": kwargs["wind_dir_ts"][k:k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=None if t > 0 else {"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines},
                            seed=k)
        
        ctrl.current_freestream_measurements = [
                kwargs["wind_mag_ts"][k] * np.sin((kwargs["wind_dir_ts"][k] - 180.) * (np.pi / 180.)),
                kwargs["wind_mag_ts"][k] * np.cos((kwargs["wind_dir_ts"][k] - 180.) * (np.pi / 180.))
        ]
        
        start_time = perf_counter()
        # get measurements from FLORIS int, then compute controls in controller class, set controls_dict, then send controls to FLORIS interface (calling calculate_wake)
        fi.time = np.arange(t, t + ctrl.dt, input_dict["dt"])
        # only step yaw angles by up to yaw_rate * input_dict["dt"] for each time-step
        ctrl.step()
        end_time = perf_counter()

        # convergence_time_ts.append((end_time - start_time) if ((t % ctrl.dt) == 0.0) else np.nan)
        convergence_time_ts += ([end_time - start_time] + [np.nan] * n_future_steps)

        # opt_codes_ts.append(ctrl.opt_code)
        if hasattr(ctrl, "opt_cost"):
            opt_cost_terms_ts += ([ctrl.opt_cost_terms] + [[np.nan] * 2] * n_future_steps)
            opt_cost_ts += ([ctrl.opt_cost] + [np.nan] * n_future_steps)
        else:
            opt_cost_terms_ts += [[np.nan] * 2] * (n_future_steps + 1)
            opt_cost_ts += [np.nan] * (n_future_steps + 1)
        
        if hasattr(ctrl, "init_sol"):
            init_states = np.array(ctrl.init_sol["states"]) * ctrl.yaw_norm_const
            init_ctrl_inputs = ctrl.init_sol["control_inputs"]
        else:
            init_states = [np.nan] * ctrl.n_turbines

            init_ctrl_inputs = [np.nan] * ctrl.n_turbines

        # assert np.all(ctrl.controls_dict['yaw_angles'] == ctrl.measurements_dict["wind_directions"] - fi.env.floris.farm.yaw_angles)
        # TODO add freestream wind mags/dirs provided to controller, yaw angles computed at this time-step, resulting turbine powers, wind mags, wind dirs
        # yaw_angles_ts += list(fi.env.floris.flow_field.wind_directions[:, np.newaxis] - fi.env.floris.farm.yaw_angles)
         # Note these are results from previous time step
        yaw_angles_ts += list(ctrl.measurements_dict["yaw_angles"])

        # Note these are results from previous time step
        turbine_powers_ts += list(ctrl.measurements_dict["turbine_powers"])

        # Note these are results from previous time step
        turbine_wind_mag_ts += list(ctrl.measurements_dict["wind_speeds"])
        # turbine_wind_mag_ts += list(fi.env.floris.flow_field.wind_speeds)

        # Note these are results from previous time step
        turbine_wind_dir_ts += list(ctrl.measurements_dict["wind_directions"])
        # turbine_wind_dir_ts += list(fi.env.floris.flow_field.wind_directions)

        # turbine_offline_status_ts.append(fi.offline_status) # TODO this should be included in measurements?
        turbine_offline_status_ts += list(fi.offline_status)

        print(f"\nTime = {t}",
            f"Measured Freestream Wind Direction = {kwargs['wind_dir_ts'][k]}",
            f"Measured Freestream Wind Magnitude = {kwargs['wind_mag_ts'][k]}",
            f"Measured Turbine Wind Directions = {ctrl.measurements_dict['wind_directions'][0, :] if ctrl.measurements_dict['wind_directions'].ndim == 2 else ctrl.measurements_dict['wind_directions']}",
            f"Measured Turbine Wind Magnitudes = {ctrl.measurements_dict['wind_speeds'][0, :] if ctrl.measurements_dict['wind_speeds'].ndim == 2 else ctrl.measurements_dict['wind_speeds']}",
            f"Measured Yaw Angles = {ctrl.measurements_dict['yaw_angles'][0, :] if ctrl.measurements_dict['yaw_angles'].ndim == 2 else ctrl.measurements_dict['yaw_angles']}",
            f"Measured Turbine Powers = {ctrl.measurements_dict['turbine_powers'][0, :] if ctrl.measurements_dict['turbine_powers'].ndim == 2 else ctrl.measurements_dict['turbine_powers']}",
            f"Distance from Initial Yaw Angle Solution = {np.linalg.norm(ctrl.controls_dict['yaw_angles'] - init_states[:ctrl.n_turbines])}",
            f"Distance from Initial Yaw Angle Change Solution = {np.linalg.norm((ctrl.controls_dict['yaw_angles'] - yaw_angles_ts[-(n_future_steps + 1)]) - init_ctrl_inputs[:ctrl.n_turbines])}",
            # f"Optimizer Output = {ctrl.opt_code['text']}",
            # f"Optimized Yaw Angle Solution = {ctrl.opt_sol['states'] * ctrl.yaw_norm_const}",
            # f"Optimized Yaw Angle Change Solution = {ctrl.opt_sol['control_inputs']}",
            f"Optimized Yaw Angles = {ctrl.controls_dict['yaw_angles']}",
            f"Optimized Yaw Angle Changes = {ctrl.controls_dict['yaw_angles'] - yaw_angles_ts[-(n_future_steps + 1)]}",
            # f"Optimized Power Cost = {opt_cost_terms_ts[-1][0]}",
            # f"Optimized Yaw Change Cost = {opt_cost_terms_ts[-1][1]}",
            f"Convergence Time = {convergence_time_ts[-(n_future_steps + 1)]}",
            sep='\n')
        
        t += ctrl.dt
        k += int(ctrl.dt / input_dict["dt"])
    else:
        last_measurements = fi.get_measurements()
        yaw_angles_ts = np.vstack([yaw_angles_ts, last_measurements["yaw_angles"]])
        yaw_angles_change_ts = np.diff(yaw_angles_ts, axis=0)
        yaw_angles_change_ts = yaw_angles_change_ts[n_future_steps:, :]
        yaw_angles_ts = yaw_angles_ts[n_future_steps + 1:, :]

        turbine_powers_ts = np.vstack([turbine_powers_ts, last_measurements["turbine_powers"]])
        turbine_powers_ts = turbine_powers_ts[n_future_steps + 1:, :]

        turbine_wind_mag_ts = np.vstack([turbine_wind_mag_ts, last_measurements["wind_speeds"]])
        turbine_wind_mag_ts = turbine_wind_mag_ts[n_future_steps + 1:, :]

        turbine_wind_dir_ts = np.vstack([turbine_wind_dir_ts, last_measurements["wind_directions"]])
        turbine_wind_dir_ts = turbine_wind_dir_ts[n_future_steps + 1:, :]
    
    # yaw_angles_change_ts = np.vstack(yaw_angles_change_ts)
    
    turbine_offline_status_ts = np.vstack(turbine_offline_status_ts)

    # greedy_turbine_powers_ts = np.vstack(greedy_turbine_powers_ts)
    # opt_cost_terms_ts = np.vstack(opt_cost_terms_ts)
    # running_opt_cost_terms_ts = np.vstack(running_opt_cost_terms_ts)

    running_opt_cost_terms_ts = np.zeros_like(opt_cost_terms_ts)
    Q = input_dict["controller"]["alpha"]
    R = (1 - input_dict["controller"]["alpha"])

    norm_turbine_powers = np.divide(turbine_powers_ts, greedy_turbine_powers_ts[:, np.newaxis],
                                    where=greedy_turbine_powers_ts[:, np.newaxis]!=0,
                                    out=np.zeros_like(turbine_powers_ts))
    norm_yaw_angle_changes = yaw_angles_change_ts / (ctrl.dt * ctrl.yaw_rate)
    
    running_opt_cost_terms_ts[:, 0] = np.sum(np.stack([-0.5 * (norm_turbine_powers[:, i])**2 * Q for i in range(ctrl.n_turbines)], axis=1), axis=1)
    running_opt_cost_terms_ts[:, 1] = np.sum(np.stack([0.5 * (norm_yaw_angle_changes[:, i])**2 * R for i in range(ctrl.n_turbines)], axis=1), axis=1)

    results_df = pd.DataFrame(data={
        "CaseName": [kwargs["case_name"]] *  int(kwargs["wind_field_config"]["simulation_max_time"] // input_dict["dt"]),
        "WindSeed": [kwargs["wind_case_idx"]] * int(kwargs["wind_field_config"]["simulation_max_time"] // input_dict["dt"]),
        "Time": np.arange(0, kwargs["wind_field_config"]["simulation_max_time"], input_dict["dt"]),
        "FreestreamWindMag": kwargs["wind_mag_ts"][:yaw_angles_ts.shape[0]],
        "FreestreamWindDir": kwargs["wind_dir_ts"][:yaw_angles_ts.shape[0]],
        **{
            f"TurbineYawAngle_{i}": yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        **{
            f"TurbineYawAngleChange_{i}": yaw_angles_change_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbinePower_{i}": turbine_powers_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindMag_{i}": turbine_wind_mag_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindDir_{i}": turbine_wind_dir_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineOfflineStatus_{i}": turbine_offline_status_ts[:, i] for i in range(ctrl.n_turbines)
        },
        "FarmYawAngleChangeAbsSum": np.sum(np.abs(yaw_angles_change_ts), axis=1),
        "RelativeFarmYawAngleChangeAbsSum": (np.sum(np.abs(yaw_angles_change_ts) * ~turbine_offline_status_ts, axis=1)) / (np.sum(~turbine_offline_status_ts, axis=1)),
        "FarmPower": np.sum(turbine_powers_ts, axis=1),
        "RelativeFarmPower": np.sum(turbine_powers_ts, axis=1) / (np.sum(~turbine_offline_status_ts * np.array([max(fi.env.core.farm.turbine_definitions[i]["power_thrust_table"]["power"]) for i in range(ctrl.n_turbines)]), axis=1)),
        # **{
        #     f"OptimizationCostTerm_{i}": opt_cost_terms_ts[:, i] for i in range(opt_cost_terms_ts.shape[1])
        # },
        # "TotalOptimizationCost": np.sum(opt_cost_terms_ts, axis=1),
        "OptimizationConvergenceTime": convergence_time_ts,
        **{
            f"RunningOptimizationCostTerm_{i}": running_opt_cost_terms_ts[:, i] for i in range(running_opt_cost_terms_ts.shape[1])
        },
        "TotalRunningOptimizationCost": np.sum(running_opt_cost_terms_ts, axis=1),
    })

    results_dir = os.path.join(STORAGE_DIR, kwargs['case_family'])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_df.to_csv(os.path.join(results_dir, f"time_series_results_case_{kwargs['case_name']}_seed_{kwargs['wind_case_idx']}.csv"))
    
    return results_df

if __name__ == "__main__":
    comm_rank = MPI.COMM_WORLD.Get_rank()
    if sys.argv[2].lower() == "mpi":
        MULTI = "mpi"
    else:
        MULTI = "cf"

    # intialization code
    if (MULTI == "mpi" and comm_rank == 0) or (MULTI != "mpi"):

        DEBUG = sys.argv[1].lower() == "debug"
        # if sys.argv[2].lower() == "dask":
        #     MULTI = "dask"
        #     initialize()
        #     client = Client()


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
            case_studies[case_family]["wind_case_idx"] = {"group": 2, "vals": [i for i in range(N_SEEDS)]}

        os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "true"
        # run_simulations(["perfect_preview_type"], REGENERATE_WIND_FIELD)
        print([case_families[i] for i in CASE_FAMILY_IDX])

        if os.path.exists("init_simulations.pkl"):
            with open("init_simulations.pkl", "rb") as fp:
                tmp = pickle.load(fp)
                case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = tuple(tmp.values())
        else:
            raise FileNotFoundError("run initialize_case_studies.py to generate init_simulation.pkl")

        # run simulations
        print(f"about to submit calls to simulate_controller")
        
    if PARALLEL:
        if MULTI == "mpi":
            comm_size = MPI.COMM_WORLD.Get_size()
            # comm_rank = MPI.COMM_WORLD.Get_rank()
            # node_name = MPI.Get_processor_name()
            executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
        else:
            executor = ProcessPoolExecutor()
        with executor as run_simulations_exec:
            if MULTI == "mpi":
                run_simulations_exec.max_workers = comm_size
            print(f"run_simulations line 618 with {run_simulations_exec._max_workers} workers")
            # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
            futures = [run_simulations_exec.submit(simulate_controller, 
                                            controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                            wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                            case_name=case_lists[c]["case_names"], case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                            lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
                    for c, d in enumerate(input_dicts)]
            # cf_wait(futures)
            results = [fut.result() for fut in futures]

        print("run_simulations line 626")

    else:
        results = []
        for c, d in enumerate(input_dicts):
            results.append(simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name=case_lists[c]["case_names"], case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                                lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"],
                                                wind_field_config=wind_field_config, verbose=False))