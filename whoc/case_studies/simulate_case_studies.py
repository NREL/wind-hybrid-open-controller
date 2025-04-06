import pandas as pd
import numpy as np
import os
from time import perf_counter
# from memory_profiler import profile
import re

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.wind_field.WindField import first_ord_filter


def simulate_controller(controller_class, wind_forecast_class, simulation_input_dict, **kwargs):
    print(f'Entering simulate_controller function')
    results_dir = os.path.join(kwargs["save_dir"], kwargs['case_family'])
    os.makedirs(results_dir, exist_ok=True)
    
    temp_storage_dir = os.path.join(results_dir, "temp")
    os.makedirs(temp_storage_dir, exist_ok=True)

    fn = f"time_series_results_case_{kwargs['case_name']}_seed_{kwargs['wind_case_idx']}.csv".replace("/", "_")
    # print(f'rerun_simulations = {kwargs["rerun_simulations"]}')
    # print(f'does {os.path.join(results_dir, fn)} exist = {os.path.exists(os.path.join(results_dir, fn))}')
    print(f'reaches the first if statement')
    if not kwargs["rerun_simulations"] and os.path.exists(os.path.join(results_dir, fn)):
        results_df = pd.read_csv(os.path.join(results_dir, fn))
        print(f"Loaded existing {fn} since rerun_simulations argument is false")
        return results_df
    elif not kwargs["rerun_simulations"] and os.path.exists(os.path.join(results_dir, fn.replace("results", f"chk"))):
        # TODO load checkpoint if exists
        pass
    

    print(f"Running instance of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
    # Load a FLORIS object for power calculations
    fi = ControlledFlorisModel(t0=kwargs["wind_field_ts"]["time"].iloc[0],
                               yaw_limits=simulation_input_dict["controller"]["yaw_limits"],
                                offline_probability=simulation_input_dict["controller"]["offline_probability"],
                                simulation_dt=simulation_input_dict["simulation_dt"],
                                yaw_rate=simulation_input_dict["controller"]["yaw_rate"],
                                config_path=simulation_input_dict["controller"]["floris_input_file"],
                                target_turbine_indices=simulation_input_dict["controller"]["target_turbine_indices"] or "all",
                                uncertain=simulation_input_dict["controller"]["uncertain"],
                                turbine_signature=kwargs["turbine_signature"],
                                tid2idx_mapping=kwargs["tid2idx_mapping"])
     
    if simulation_input_dict["controller"]["target_turbine_indices"] != "all":
        fi_full = ControlledFlorisModel(t0=kwargs["wind_field_ts"]["time"].iloc[0],
                               yaw_limits=simulation_input_dict["controller"]["yaw_limits"],
                                offline_probability=simulation_input_dict["controller"]["offline_probability"],
                                simulation_dt=simulation_input_dict["simulation_dt"],
                                yaw_rate=simulation_input_dict["controller"]["yaw_rate"],
                                config_path=simulation_input_dict["controller"]["floris_input_file"],
                                target_turbine_indices="all",
                                uncertain=simulation_input_dict["controller"]["uncertain"],
                                turbine_signature=kwargs["turbine_signature"],
                                tid2idx_mapping=kwargs["tid2idx_mapping"])
    else:
        fi_full = fi
    
    if not kwargs["tid2idx_mapping"]:
        kwargs["tid2idx_mapping"] = {i: i for i in np.arange(fi_full.n_turbines)}
    idx2tid_mapping = dict([(v, k) for k, v in kwargs["tid2idx_mapping"].items()])
    
    kwargs["wind_field_config"]["preview_dt"] = int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"]) 
    kwargs["wind_field_config"]["n_preview_steps"] = simulation_input_dict["controller"]["n_horizon"] * int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"])
    kwargs["wind_field_config"]["time_series_dt"] = int(simulation_input_dict["controller"]["controller_dt"] // simulation_input_dict["simulation_dt"])
    
    if simulation_input_dict["controller"]["initial_conditions"]["yaw"] == "auto":
        if "FreestreamWindDir" in kwargs["wind_field_ts"].columns:
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = np.array([kwargs["wind_field_ts"]["FreestreamWindDir"].iloc[0]] * fi.n_turbines)
        else:
            sorted_tids = np.arange(fi_full.n_turbines) if simulation_input_dict["controller"]["target_turbine_indices"] == "all" else sorted(simulation_input_dict["controller"]["target_turbine_indices"])
            u = kwargs["wind_field_ts"].iloc[0][[f"ws_horz_{idx2tid_mapping[i]}" for i in sorted_tids]].values.astype(float)
            v = kwargs["wind_field_ts"].iloc[0][[f"ws_vert_{idx2tid_mapping[i]}" for i in sorted_tids]].values.astype(float)
            simulation_input_dict["controller"]["initial_conditions"]["yaw"] = 180.0 + np.rad2deg(np.arctan2(u, v))
     
    # pl.DataFrame(kwargs["wind_field_ts"])
    # simulation_input_dict["wind_forecast"]["measurement_layout"] = np.vstack([fi.env.layout_x, fi.env.layout_y]).T
    if wind_forecast_class:
        wind_forecast = wind_forecast_class(true_wind_field=kwargs["wind_field_ts"],
                                            fmodel=fi_full.env, 
                                            tid2idx_mapping=kwargs["tid2idx_mapping"],
                                            turbine_signature=kwargs["turbine_signature"],
                                            use_tuned_params=kwargs["use_tuned_params"],
                                            model_config=kwargs["model_config"],
                                            **{k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" in k},
                                            kwargs={k: v for k, v in simulation_input_dict["wind_forecast"].items() if "timedelta" not in k},
                                            temp_save_dir=temp_storage_dir)
    else:
        wind_forecast = None
    ctrl = controller_class(fi, wind_forecast=wind_forecast, simulation_input_dict=simulation_input_dict, **kwargs)
    
    yaw_angles_ts = []
    init_yaw_angles_ts = []
    yaw_angles_change_ts = []
    turbine_powers_ts = []
    turbine_wind_mag_ts = []
    turbine_wind_dir_ts = []
    turbine_offline_status_ts = []
    predicted_wind_speeds_ts = []
    # predicted_time_ts = []
    # predicted_turbine_wind_speed_horz_ts = []
    # predicted_turbine_wind_speed_vert_ts = []
    # stddev_turbine_wind_speed_horz_ts = []
    # stddev_turbine_wind_speed_vert_ts = []
    
    convergence_time_ts = []

    opt_cost_ts = []
    opt_cost_terms_ts = []
    
    if hasattr(ctrl, "state_cons_activated"):
        lower_state_cons_activated_ts = []
        upper_state_cons_activated_ts = []

    n_future_steps = int(ctrl.controller_dt // simulation_input_dict["simulation_dt"]) - 1
    
    t = 0
    k = 0
    print(f'simulation_input_dict reached')
    # input to floris should be from first in target_turbine_indices (most upstream one), or mean over whole farm if no target_turbine_indices
    if kwargs["wf_source"] == "scada":
        if simulation_input_dict["controller"]["target_turbine_indices"] == "all":
            simulation_u = kwargs["wind_field_ts"][[f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(idx2tid_mapping))]].mean(axis=1)
            simulation_v = kwargs["wind_field_ts"][[f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(idx2tid_mapping))]].mean(axis=1)
        else:
            use_upstream_wind = True
            if use_upstream_wind:
                upstream_tidx = simulation_input_dict["controller"]["target_turbine_indices"][0]
                simulation_u = kwargs["wind_field_ts"][f"ws_horz_{idx2tid_mapping[upstream_tidx]}"]
                simulation_v = kwargs["wind_field_ts"][f"ws_vert_{idx2tid_mapping[upstream_tidx]}"]
            else:
                # use mean
                simulation_u = kwargs["wind_field_ts"][[f"ws_horz_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]].mean(axis=1)
                simulation_v = kwargs["wind_field_ts"][[f"ws_vert_{idx2tid_mapping[t_idx]}" for t_idx in simulation_input_dict["controller"]["target_turbine_indices"]]].mean(axis=1)
            
        simulation_mag = (simulation_u**2 + simulation_v**2)**0.5
        simulation_dir = 180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v))
        simulation_dir[simulation_dir < 0] = 360. + simulation_dir[simulation_dir < 0]
        simulation_dir[simulation_dir > 360] = np.mod(simulation_dir[simulation_dir > 360], 360.) 
    else:
        simulation_mag = kwargs["wind_field_ts"]["FreestreamWindMag"].to_numpy()
        simulation_dir = kwargs["wind_field_ts"]["FreestreamWindDir"].to_numpy()
        simulation_u = simulation_mag * np.sin(np.deg2rad(180 + simulation_dir))
        simulation_v = simulation_mag * np.cos(np.deg2rad(180 + simulation_dir))
        
    # recompute controls and step floris forward by ctrl.controller_dt
    while t < simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"]:

        # reiniitialize and run FLORIS interface with current disturbances and disturbance up to (and excluding) next controls computation
        # using yaw angles as most recently sent from last time-step i.e. initial yaw conditions for first time step
        fi.step(disturbances={"wind_speeds": simulation_mag[k:k + n_future_steps + 1],
                            "wind_directions": simulation_dir[k:k + n_future_steps + 1], 
                            "turbulence_intensities": [fi.env.core.flow_field.turbulence_intensities[0]] * (n_future_steps + 1)},
                            ctrl_dict=None if t > 0 else {"yaw_angles": [ctrl.yaw_IC] * ctrl.n_turbines if isinstance(ctrl.yaw_IC, float) else ctrl.yaw_IC},
                            seed=k)
        
        ctrl.current_freestream_measurements = [
                simulation_u[k],
                simulation_v[k]
        ]
         
        start_time = perf_counter()
        # get measurements from FLORIS int, then compute controls in controller class, set controls_dict, then send controls to FLORIS interface (calling calculate_wake)
        
        fi.run_floris = False
        # only step yaw angles by up to yaw_rate * simulation_input_dict["simulation_dt"] for each time-step
        # in ctrl.step(), get simulator measurements from FLORIS, update controls dict every simulation_dt seconds,
        # but only compute new yaw setpoints and run FLORIS with setpoints from full controllet_dt interval every controller_dt in ControllerFlorisInterface
        for tt in np.arange(t, t + ctrl.controller_dt, simulation_input_dict["simulation_dt"]):
            
            if tt == (t + ctrl.controller_dt - simulation_input_dict["simulation_dt"]):
                fi.run_floris = True
            
            ctrl.step()        
            
            # Note these are results from previous time step
            yaw_angles_ts += [ctrl.measurements_dict["yaw_angles"]]
            init_yaw_angles_ts += [[ctrl.init_sol["states"][i] * ctrl.yaw_norm_const for i in range(ctrl.n_turbines)]]
            turbine_powers_ts += [ctrl.measurements_dict["turbine_powers"]]
            turbine_wind_mag_ts += [ctrl.measurements_dict["wind_speeds"]]
            turbine_wind_dir_ts += [ctrl.measurements_dict["wind_directions"]]
            
            if wind_forecast_class:
                predicted_wind_speeds_ts += [ctrl.controls_dict["predicted_wind_speeds"]]
                # predicted_time_ts += [ctrl.controls_dict["predicted_time"]]
                # predicted_turbine_wind_speed_horz_ts += [ctrl.controls_dict["predicted_wind_speeds_horz"]]
                # predicted_turbine_wind_speed_vert_ts += [ctrl.controls_dict["predicted_wind_speeds_vert"]]
                # if ctrl.uncertain:
                #     stddev_turbine_wind_speed_horz_ts += [ctrl.controls_dict["stddev_wind_speeds_horz"]]
                #     stddev_turbine_wind_speed_vert_ts += [ctrl.controls_dict["stddev_wind_speeds_vert"]]
                # else:
                #     stddev_turbine_wind_speed_horz_ts += [[np.nan] * fi_full.n_turbines]
                #     stddev_turbine_wind_speed_vert_ts += [[np.nan] * fi_full.n_turbines]
            # turbine_offline_status_ts += [fi.offline_status[tt, :]]
            turbine_offline_status_ts += [np.isclose(ctrl.measurements_dict["turbine_powers"], 0, atol=1e-3)]
            
            if hasattr(ctrl, "state_cons_activated"):
                lower_state_cons_activated_ts += [ctrl.state_cons_activated["lower"]]
                upper_state_cons_activated_ts += [ctrl.state_cons_activated["upper"]]
             
            fi.time += pd.Timedelta(seconds=simulation_input_dict["simulation_dt"])
        
        # zero turbine power could be due to low wind speed as well as formally set offline 
        # assert np.all(np.vstack(turbine_offline_status_ts)[-int(ctrl.controller_dt // simulation_input_dict["simulation_dt"]):, :] == fi.offline_status), "collected turbine_offline_status_ts should be equal to fi.offline_status in simulate_controllers"

        end_time = perf_counter()

        # convergence_time_ts.append((end_time - start_time) if ((t % ctrl.controller_dt) == 0.0) else np.nan)
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
        # add freestream wind mags/dirs provided to controller, yaw angles computed at this time-step, resulting turbine powers, wind mags, wind dirs

        print(f"\nTime = {t} of {controller_class.__name__} - {kwargs['case_name']} with wind seed {kwargs['wind_case_idx']}")
        if ctrl.verbose and False:
            print(f"Measured Freestream Wind Direction = {simulation_dir[k]}",
                f"Measured Freestream Wind Magnitude = {simulation_mag[k]}",
                f"Measured Turbine Wind Directions = {ctrl.measurements_dict['wind_directions'] if ctrl.measurements_dict['wind_directions'].ndim == 2 else ctrl.measurements_dict['wind_directions']}",
                f"Measured Turbine Wind Magnitudes = {ctrl.measurements_dict['wind_speeds'] if ctrl.measurements_dict['wind_speeds'].ndim == 2 else ctrl.measurements_dict['wind_speeds']}",
                f"Measured Yaw Angles = {ctrl.measurements_dict['yaw_angles'] if ctrl.measurements_dict['yaw_angles'].ndim == 2 else ctrl.measurements_dict['yaw_angles']}",
                f"Measured Turbine Powers = {ctrl.measurements_dict['turbine_powers'] if ctrl.measurements_dict['turbine_powers'].ndim == 2 else ctrl.measurements_dict['turbine_powers']}",
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
        
        t += ctrl.controller_dt
        k += int(ctrl.controller_dt / simulation_input_dict["simulation_dt"])
    else:
        for tt in np.arange(t, t + ctrl.controller_dt, simulation_input_dict["simulation_dt"]):
            last_measurements = fi.get_measurements()
            
            # Note these are results from previous time step
            yaw_angles_ts += [last_measurements["yaw_angles"]]
            fi.time += pd.Timedelta(seconds=ctrl.controller_dt)

        turbine_wind_mag_ts = np.vstack(turbine_wind_mag_ts)
        turbine_wind_dir_ts = np.vstack(turbine_wind_dir_ts)
        turbine_offline_status_ts = np.vstack(turbine_offline_status_ts)

        yaw_angles_ts = np.vstack(yaw_angles_ts)
        init_yaw_angles_ts = np.vstack(init_yaw_angles_ts)
        yaw_angles_change_ts = np.diff(yaw_angles_ts, axis=0)[:-n_future_steps, :]

        yaw_angles_ts = yaw_angles_ts[:-(n_future_steps + 1), :]
        turbine_powers_ts = np.vstack(turbine_powers_ts)
        
    n_truncate_steps = (int(ctrl.controller_dt - (simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] % ctrl.controller_dt)) % ctrl.controller_dt) // simulation_input_dict["simulation_dt"]
    turbine_wind_mag_ts = turbine_wind_mag_ts[:(-n_truncate_steps) or None, :]
    turbine_wind_dir_ts = turbine_wind_dir_ts[:(-n_truncate_steps) or None, :]
    turbine_offline_status_ts = turbine_offline_status_ts[:(-n_truncate_steps) or None, :]
    yaw_angles_change_ts = yaw_angles_change_ts[:(-n_truncate_steps) or None, :]
    yaw_angles_ts = yaw_angles_ts[:(-n_truncate_steps) or None, :]
    turbine_powers_ts = turbine_powers_ts[:(-n_truncate_steps) or None, :]
    
    running_opt_cost_terms_ts = np.zeros_like(opt_cost_terms_ts)
    Q = simulation_input_dict["controller"]["alpha"]
    R = (1 - simulation_input_dict["controller"]["alpha"]) 
    # TODO greatest farm power should not occur for alpha = 0, and should not coincide with greatest cost term 0
    # TODO cost term 1 should not = 0 for alpha = 0
    # TODO cost term 1 should never be negative

    # norm_turbine_powers = np.divide(turbine_powers_ts, greedy_turbine_powers_ts[:, np.newaxis],
    #                                 where=greedy_turbine_powers_ts[:, np.newaxis]!=0,
    #                                 out=np.zeros_like(turbine_powers_ts))
    norm_turbine_powers = turbine_powers_ts / ctrl.rated_turbine_power
    norm_yaw_angle_changes = yaw_angles_change_ts / (ctrl.controller_dt * ctrl.yaw_rate)
    
    running_opt_cost_terms_ts[:, 0] = np.sum(np.stack([-0.5 * (norm_turbine_powers[:, i])**2 * Q for i in range(ctrl.n_turbines)], axis=1), axis=1)
    running_opt_cost_terms_ts[:, 1] = np.sum(np.stack([0.5 * (norm_yaw_angle_changes[:, i])**2 * R for i in range(ctrl.n_turbines)], axis=1), axis=1)
    
    # may be longer than following: int(simulation_input_dict["hercules_comms"]["helics"]["config"]["stoptime"] // simulation_input_dict["simulation_dt"]), if controller step goes beyond
    results_data = {
        "CaseFamily": [kwargs["case_family"]] * yaw_angles_ts.shape[0], 
        "CaseName": [kwargs["case_name"]] * yaw_angles_ts.shape[0],
        "WindSeed": [kwargs["wind_case_idx"]] * yaw_angles_ts.shape[0],
        "Time": np.arange(0, yaw_angles_ts.shape[0]) * simulation_input_dict["simulation_dt"],
        "FreestreamWindMag": simulation_mag[:yaw_angles_ts.shape[0]],
        "FreestreamWindDir": simulation_dir[:yaw_angles_ts.shape[0]],
        "FilteredFreestreamWindDir": first_ord_filter(simulation_dir[:yaw_angles_ts.shape[0]], 
                                                      alpha=np.exp(-(1 / simulation_input_dict["controller"]["lpf_time_const"]) * simulation_input_dict["simulation_dt"])),
        **{
            f"InitTurbineYawAngle_{idx2tid_mapping[i]}": init_yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        **{
            f"TurbineYawAngle_{idx2tid_mapping[i]}": yaw_angles_ts[:, i] for i in range(ctrl.n_turbines)
        }, 
        **{
            f"TurbineYawAngleChange_{idx2tid_mapping[i]}": yaw_angles_change_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbinePower_{idx2tid_mapping[i]}": turbine_powers_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindMag_{idx2tid_mapping[i]}": turbine_wind_mag_ts[:, i] for i in range(ctrl.n_turbines)
        },
        **{
            f"TurbineWindDir_{idx2tid_mapping[i]}": turbine_wind_dir_ts[:, i] for i in range(ctrl.n_turbines)
        },
        # **{
        #     f"PredictedTurbineWindMag_{idx2tid_mapping[i]}": predicted_turbine_wind_mag_ts[:, i] for i in range(fi_full.n_turbines)
        # },
        # **{
        #     f"PredictedTurbineWindDir_{idx2tid_mapping[i]}": predicted_turbine_wind_dir_ts[:, i] for i in range(fi_full.n_turbines)
        # },
        **{
            f"TurbineOfflineStatus_{idx2tid_mapping[i]}": turbine_offline_status_ts[:, i] for i in range(ctrl.n_turbines)
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
    }
    
    if kwargs["wf_source"] == "scada":
        results_data.update({
            **{
                f"TrueTurbineWindSpeedHorz_{idx2tid_mapping[i]}": 
                kwargs["wind_field_ts"][f"ws_horz_{idx2tid_mapping[i]}"].iloc[:len(results_data["Time"])]
                for i in range(fi_full.n_turbines)
            },
            **{
                f"TrueTurbineWindSpeedVert_{idx2tid_mapping[i]}": 
                kwargs["wind_field_ts"][f"ws_vert_{idx2tid_mapping[i]}"].iloc[:len(results_data["Time"])]
                for i in range(fi_full.n_turbines)
            },
        })

    if hasattr(ctrl, "state_cons_activated"):
        results_data.update({
            "StateConsActivatedLower": lower_state_cons_activated_ts,
            "StateConsActivatedUpper": upper_state_cons_activated_ts,
        })

    results_df = pd.DataFrame(results_data)
    
    if wind_forecast_class:
        predicted_wind_speeds_ts = pd.concat(predicted_wind_speeds_ts, axis=0).groupby("time").agg("last").reset_index(names=["time"])
        predicted_wind_speeds_ts["time"] = (predicted_wind_speeds_ts["time"] - ctrl.init_time).dt.total_seconds().astype(int)
        # results_df = pd.concat([results_df, predicted_wind_speeds_ts], axis=1)
        # sd_ws_vert_cols
        cols = ["time"] + ctrl.mean_ws_horz_cols + ctrl.mean_ws_vert_cols + ((ctrl.sd_ws_horz_cols + ctrl.sd_ws_vert_cols) if ctrl.uncertain else [])
        predicted_wind_speeds_ts = predicted_wind_speeds_ts[cols].rename(columns={
            src: f"PredictedTurbineWindSpeed{re.search('(?<=ws_)\\w+(?=_\\d+)', src).group().capitalize()}_{re.search('(?<=_)\\d+$', src).group()}"
            for src in ctrl.mean_ws_horz_cols + ctrl.mean_ws_vert_cols})
        predicted_wind_speeds_ts = predicted_wind_speeds_ts.rename(columns={"time": "Time"})
        if ctrl.uncertain:
            predicted_wind_speeds_ts = predicted_wind_speeds_ts[cols].rename(columns={
                src: f"StddevTurbineWindSpeed{re.search('(?<=ws_)\\w+(?=_\\d+)', src).group().capitalize()}_{re.search('(?<=_)\\d+$', src).group()}"
                for src in ctrl.sd_ws_horz_cols + ctrl.sd_ws_vert_cols})
        predicted_wind_speeds_ts[["CaseFamily", "CaseName", "WindSeed"]] = results_df[["CaseFamily", "CaseName", "WindSeed"]].iloc[0]
        results_df = results_df.merge(predicted_wind_speeds_ts, on=["CaseFamily", "CaseName", "WindSeed", "Time"], how="outer")
    
    results_df.to_csv(os.path.join(results_dir, fn))
    print(f"Saved {fn}")
    
    return results_df