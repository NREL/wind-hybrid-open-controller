from time import perf_counter
from functools import partial
import copy
from collections import defaultdict
import os
# TODO update LUT for turbine breakdown, update mpc model with disabled turbines
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyoptsparse import Optimization, SLSQP, SNOPT
from scipy.optimize import linprog, basinhopping
from scipy.integrate import dblquad

import whoc
from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.wind_field.WindField import WindField, generate_wind_preview

from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

# from floris_dev.tools import FlorisModel as FlorisModelDev
# from floris_dev.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

# TODO change constraints to be probabilistic and use freestream wind speed
# TODO can we run amr wind with a fidelity of 1 second, or change code to allow for other time-steps

optimizer_idx = 0

class YawOptimizationSRRHC(YawOptimizationSR):
    def __init__(
        self,
        fmodel,
        yaw_rate,
        dt,
        alpha,
        n_wind_preview_samples,
        n_horizon,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[10, 8, 6, 4],  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
    ):
        """
        Instantiate YawOptimizationSR object with a FlorisModel object
        and assign parameter values.
        """

        # Initialize base class
        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            Ny_passes=Ny_passes,
            turbine_weights=turbine_weights,
            # calc_baseline_power=False,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
        )
        self.yaw_rate = yaw_rate
        self.dt = dt
        self.Q = alpha
        self.R = (1 - alpha) # need to scale control input componet to contend with power component
        self.n_wind_preview_samples = n_wind_preview_samples
        self.n_horizon = n_horizon

        self._turbine_power_opt_subset = np.zeros_like(self._minimum_yaw_angle_subset)
        self._cost_opt_subset = np.ones((1)) * 1e6
        self._cost_terms_opt_subset = np.ones((*self._cost_opt_subset.shape, 2)) * 1e6

    def _calculate_turbine_powers(
            self, 
            yaw_angles=None, 
            wd_array=None, ws_array=None, ti_array=None, 
            turbine_weights=None,
            heterogeneous_speed_multipliers=None,
            current_offline_status=None
    ):
        """
        Calculate the wind farm power production assuming the predefined
        probability distribution (self.unc_options/unc_pmf), with the
        appropriate weighing terms, and for a specific set of yaw angles.

        Args:
            yaw_angles ([iteratible]): Array or list of yaw angles in degrees.

        Returns:
            farm_power (float): Weighted wind farm power.
        """
        # Unpack all variables, whichever are defined.
        fmodel_subset = copy.deepcopy(self.fmodel_subset)
        if wd_array is None:
            wd_array = fmodel_subset.core.flow_field.wind_directions
        if ws_array is None:
            ws_array = fmodel_subset.core.flow_field.wind_speeds
        if ti_array is None:
            ti_array = fmodel_subset.core.flow_field.turbulence_intensities
        if yaw_angles is None:
            yaw_angles = self._yaw_angles_baseline_subset
        if turbine_weights is None:
            turbine_weights = self._turbine_weights_subset
        if heterogeneous_speed_multipliers is not None:
            fmodel_subset.core.flow_field.\
                heterogenous_inflow_config['speed_multipliers'] = heterogeneous_speed_multipliers

        # Ensure format [incompatible with _subset notation]
        yaw_angles = self._unpack_variable(yaw_angles, subset=True)

        # Calculate solutions
        # turbine_powers = np.zeros_like(yaw_angles[:, 0, :])
        # fmodel_subset.core.farm.set_yaw_angles(fmodel_subset.core.farm.yaw_angles)
        # fmodel_subset.set_operation(
        #     yaw_angles=yaw_angles,
        #     disable_turbines=self.offline_status,
        # )
        fmodel_subset.set(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array, 
                      yaw_angles=yaw_angles, disable_turbines=current_offline_status)
        
        fmodel_subset.run()
        turbine_powers = fmodel_subset.get_turbine_powers()

        # Multiply with turbine weighing terms
        turbine_power_weighted = np.multiply(turbine_weights, turbine_powers)
        return turbine_power_weighted

    def _calculate_baseline_turbine_powers(self, current_offline_status):
        """
        Calculate the weighted wind farm power under the baseline turbine yaw
        angles.
        """
        if self.calc_baseline_power:
            P = self._calculate_turbine_powers(self._yaw_angles_baseline_subset, current_offline_status=current_offline_status)
            self._turbine_powers_baseline_subset = P
            self.turbine_powers_baseline = P
        else:
            self._turbine_powers_baseline_subset = None
            self.turbine_powers_baseline = None

    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        # farm_power_opt_subset = self._farm_power_opt_subset
        turbine_power_opt_subset = self._turbine_power_opt_subset
        wd_array_subset = self.fmodel_subset.core.flow_field.wind_directions
        ws_array_subset = self.fmodel_subset.core.flow_field.wind_speeds
        ti_array_subset = self.fmodel_subset.core.flow_field.turbulence_intensities
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 3)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1))
            # farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny, 1))
            turbine_power_opt_subset = np.tile(turbine_power_opt_subset, (Ny, 1))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            ws_array_subset = np.tile(ws_array_subset, Ny)
            ti_array_subset = np.tile(ti_array_subset, Ny)
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1))

        # Initialize empty matrix for floris farm power outputs
        # farm_powers = np.zeros((yaw_angles_subset.shape[0], yaw_angles_subset.shape[1]))
        turbine_powers = np.zeros(yaw_angles_subset.shape)

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            # idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=2).all(axis=1)
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=1)
            # farm_powers[idx, :] = farm_power_opt_subset[idx, :]
            turbine_powers[idx, :] = turbine_power_opt_subset[idx, :]
            if self.print_progress:
                self.logger.info(
                    "Skipping {:d}/{:d} calculations: already in memory.".format(
                        np.sum(idx), len(idx))
                )
        else:
            idx = np.zeros(yaw_angles_subset.shape[0], dtype=bool)

        if not np.all(idx):
            # Now calculate farm powers for conditions we haven't yet evaluated previously
            start_time = perf_counter()
            if (hasattr(self.fmodel.core.flow_field, 'heterogenous_inflow_config') and
                self.fmodel.core.flow_field.heterogenous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fmodel.core.flow_field.heterogenous_inflow_config['speed_multipliers']
                )
                het_sm = np.tile(het_sm_orig, (Ny, 1))[~idx, :]
            else:
                het_sm = None
            
            turbine_powers[~idx, :] = self._calculate_turbine_powers(
                wd_array=wd_array_subset[~idx],
                ws_array=ws_array_subset[~idx],
                ti_array=ti_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :],
                yaw_angles=yaw_angles_subset[~idx, :],
                heterogeneous_speed_multipliers=het_sm
            )
            self.time_spent_in_floris += (perf_counter() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            turbine_powers = np.reshape(
                turbine_powers,
                (
                    Ny,
                    self.fmodel_subset.core.flow_field.n_findex,
                    self.nturbs
                )
            )

        return turbine_powers
        
    def optimize(self, current_yaw_offsets, current_offline_status, constrain_yaw_dynamics=True, print_progress=False):
        
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """        
        self.print_progress = print_progress
        # compute baseline (no yaw) powers instead
        self._calculate_baseline_turbine_powers(current_offline_status)
        greedy_turbine_powers = np.max(self.turbine_powers_baseline, axis=1)[np.newaxis, :, np.newaxis]

        self._yaw_lbs = copy.deepcopy(self._minimum_yaw_angle_subset)
        self._yaw_ubs = copy.deepcopy(self._maximum_yaw_angle_subset)
        # wd_tmp = np.reshape(self.fi.core.flow_field.wind_directions, (self.n_wind_preview_samples, self.n_horizon))
        # _yaw_angles_opt_subset_original = np.array(self._yaw_angles_opt_subset)
        
        # self._turbine_power_opt_subset = self._turbine_power_opt_subset[:self.n_horizon, :]
        self._n_findex_subset = self.n_horizon

        # For each pass, from front to back
        ii = 0
        for Nii in range(len(self.Ny_passes)):
            # Disturb yaw angles for one turbine at a time, from front to back
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / (len(self.Ny_passes) * self.nturbs)
                ii += 1
                if self.print_progress:
                    print(
                        f"[Serial Refine] Processing pass={Nii}, "
                        f"turbine_depth={turbine_depth} ({p:.1f}%)"
                    )

                # Create grid to evaluate yaw angles for one turbine == turbine_depth
                
                # norm_current_yaw_angles + self.dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                # clip for control input values between -1 and 1
                
                # if we are solving an optimization problem constrainted by the dynamic state equation, constrain the alowwable range of yaw angles accordingly
                if constrain_yaw_dynamics:
                    current_yaw_offsets = self.fmodel.core.flow_field.wind_directions - current_yaw_setpoints
                    self._yaw_lbs = np.max([current_yaw_offsets - (self.dt * self.yaw_rate), self._yaw_lbs], axis=0)
                    self._yaw_ubs = np.min([current_yaw_offsets + (self.dt * self.yaw_rate), self._yaw_ubs], axis=0)

                # dimensions = (SR algorithm iteration, wind field, turbine)
                # assert np.sum(np.diff(np.reshape(self._yaw_angles_opt_subset, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=0)) == 0.
                self._yaw_angles_opt_subset = self._yaw_angles_opt_subset[:self.n_horizon, :]
                evaluation_grid = self._generate_evaluation_grid(
                    pass_depth=Nii,
                    turbine_depth=turbine_depth
                )
                evaluation_grid = np.tile(evaluation_grid, (1, self.n_wind_preview_samples, 1))
                self._yaw_angles_opt_subset = np.tile(self._yaw_angles_opt_subset, (self.n_wind_preview_samples, 1))
                
                self._yaw_evaluation_grid = evaluation_grid

                # Evaluate grid of yaw angles, get farm powers and find optimal solutions
                turbine_powers = self._process_evaluation_grid()

                # If farm powers contains any nans, then issue a warning
                if np.any(np.isnan(turbine_powers)):
                    err_msg = (
                        "NaNs found in farm powers during SerialRefine "
                        "optimization routine. Proceeding to maximize over yaw "
                        "settings that produce valid powers."
                    )
                    self.logger.warning(err_msg, stack_info=True)

                # Find optimal solutions in new evaluation grid
                # integrate linear cost function alpha, normalized powers and normalized yaw rate of changes
                # just selecting one (and only) wind speed
                    
                norm_turbine_powers = np.divide(turbine_powers, greedy_turbine_powers,
                                                out=np.zeros_like(turbine_powers),
                                                where=greedy_turbine_powers!=0) # choose power not in wake to normalize with
                # for each value in farm_powers, get corresponding next_yaw_angles from evaluation grid
                
                # just selecting one (and only) wind speed, negative because the actual control inputs measure change in absolute yaw angle, not offset
                
                evaluation_grid_tmp = np.reshape(evaluation_grid, (self.Ny_passes[Nii], self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                
                init_yaw_setpoint_change = -(evaluation_grid_tmp[:, :, 0, :] - current_yaw_offsets[np.newaxis, : ,:])[:, :, np.newaxis, :]
                subsequent_yaw_setpoint_changes = -np.diff(evaluation_grid_tmp, axis=2)

                assert np.isclose(np.sum(np.diff(init_yaw_setpoint_change, axis=1)), 0.0)
                assert np.isclose(np.sum(np.diff(subsequent_yaw_setpoint_changes, axis=1)), 0.0)


                control_inputs = np.concatenate([init_yaw_setpoint_change[:, 0, :, :], subsequent_yaw_setpoint_changes[:, 0, :, :]], axis=1) * (1 / (self.yaw_rate * self.dt))
                
                norm_turbine_powers = np.reshape(norm_turbine_powers, (self.Ny_passes[Nii], self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                cost_state = np.sum(-0.5 * np.mean(norm_turbine_powers**2, axis=1) * self.Q, axis=(1, 2))[:, np.newaxis]
                cost_control_inputs = np.sum(0.5 * control_inputs**2 * self.R, axis=(1, 2))[:, np.newaxis]
                cost_terms = np.stack([cost_state, cost_control_inputs], axis=2) # axis=3
                cost = cost_state + cost_control_inputs
                # optimum index is based on average over all wind directions supplied at second index
                args_opt = np.expand_dims(np.nanargmin(cost, axis=0), axis=0)

                cost_terms_opt_new = np.squeeze(
                    np.take_along_axis(cost_terms, 
                                       np.expand_dims(args_opt, axis=2),
                                       axis=0),
                    axis=0,
                )

                cost_opt_new = np.squeeze(
                    np.take_along_axis(cost, args_opt, axis=0),
                    axis=0,
                )

                # take turbine powers over for each wind direction sample passed
                # turbine_powers = np.mean(turbine_powers, axis=1)[:, 0, :]
                turbine_powers_opt_new = np.squeeze(
                    np.take_along_axis(turbine_powers, 
                                       np.expand_dims(args_opt, axis=2), 
                                       axis=0),
                    axis=0,
                )
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(np.sum(turbine_powers, axis=2), args_opt, axis=0),
                    axis=0,
                )
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=2),
                        axis=0
                    ),
                    axis=0
                )
                assert np.sum(np.diff(np.reshape(yaw_angles_opt_new, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=0)) == 0.0

                cost_terms_opt_prev = self._cost_terms_opt_subset
                cost_opt_prev = self._cost_opt_subset

                farm_powers_opt_prev = self._farm_power_opt_subset
                turbine_powers_opt_prev = self._turbine_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                # Now update optimal farm powers if better than previous
                ids_better = (cost_opt_new < cost_opt_prev)
                cost_opt = cost_opt_prev
                cost_opt[ids_better] = cost_opt_new[ids_better]

                cost_terms_opt = cost_terms_opt_prev
                cost_terms_opt[*ids_better, :] = cost_terms_opt_new[*ids_better, :]

                # Now update optimal yaw angles if better than previous
                turbs_sorted = self.turbines_ordered_array_subset
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (np.where(np.tile(ids_better, (yaw_angles_opt_prev.shape[0],)))[0], turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                turbine_powers_opt = turbine_powers_opt_prev
                turbine_powers_opt[ids] = turbine_powers_opt_new[ids]

                # ids = (*np.where(ids_better), 0)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids[0]] = farm_powers_opt_new[ids[0]]

                # Update bounds for next iteration to close proximity of optimal solution
                dx = (
                    evaluation_grid[1, :, :] -
                    evaluation_grid[0, :, :]
                )[ids]
                self._yaw_lbs[ids] = np.clip(
                    yaw_angles_opt[ids] - 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                self._yaw_ubs[ids] = np.clip(
                    yaw_angles_opt[ids] + 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )

                # Save results to self
                self._cost_terms_opt_subset = cost_terms_opt
                self._cost_opt_subset = cost_opt
                self._farm_power_opt_subset = farm_power_opt
                self._turbine_power_opt_subset = turbine_powers_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

                assert np.sum(np.diff(np.reshape(yaw_angles_opt, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=0)) == 0.

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()

        df_opt = df_opt.iloc[0:self.n_horizon] # only want a single row for all samples
        # np.diff(np.vstack(df_opt.iloc[::self.n_horizon]["yaw_angles_opt"].to_numpy()), axis=0)

        df_opt["cost_states"] = cost_terms_opt[0][0]
        df_opt["cost_control_inputs"] = cost_terms_opt[0][1]
        df_opt["cost"] = cost_opt[0]
        return df_opt

class MPC(ControllerBase):

    # SLSQP, NSGA2, ParOpt, CONMIN, ALPSO
    max_iter = 15
    acc = 1e-6
    # optimizers = [
    #     SLSQP(options={"IPRINT": 0, "MAXIT": max_iter, "ACC": acc}),
    #     # NSGA2(options={"xinit": 1, "PrintOut": 0, "maxGen": 50})
    #     # CONMIN(options={"IPRINT": 1, "ITMAX": max_iter})
    #     # ALPSO(options={}) #"maxOuterIter": 25})
    #     ]

    def __init__(self, interface, input_dict, wind_field_config, verbose=False, **kwargs):
        
        super().__init__(interface, verbose=verbose)
        
        self.optimizer_idx = optimizer_idx
        # TODO set time-limit
        self.optimizer = SLSQP(options={"IPRINT": 0 if verbose else -1, 
                                        "MAXIT": kwargs["max_iter"] if "max_iter" in kwargs else MPC.max_iter, 
                                        "ACC": kwargs["acc"] if "acc" in kwargs else MPC.acc})
        # self.optimizer = SNOPT(options={"iPrint": 1 if verbose else 0, 
        #                                 "Time limit": input_dict["controller"]["dt"]})

        self.dt = input_dict["controller"]["dt"]
        self.simulation_dt = input_dict["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]
        # assert self.n_turbines == interface.n_turbines
        self.turbines = range(self.n_turbines)
        self.yaw_limits = input_dict["controller"]["yaw_limits"]
        self.yaw_rate = input_dict["controller"]["yaw_rate"]
        self.yaw_increment = input_dict["controller"]["yaw_increment"]
        self.alpha = input_dict["controller"]["alpha"]
        self.beta = input_dict["controller"]["beta"]
        self.n_horizon = input_dict["controller"]["n_horizon"]
        self.wind_mag_ts = kwargs["wind_mag_ts"]
        self.wind_dir_ts = kwargs["wind_dir_ts"]

        if input_dict["controller"]["wind_preview_type"] != "stochastic":
            self.state_con_type = "check_all_samples"
        elif input_dict["controller"]["state_con_type"].lower() in ["check_all_samples", "extreme", "probabilistic"]:
            self.state_con_type = input_dict["controller"]["state_con_type"].lower()
        else:
            raise TypeError("state_con_type must be have value of 'check_all_samples', 'extreme' or 'extreme'")

        self.seed = kwargs["seed"] if "seed" in kwargs else None
        np.random.seed(seed=self.seed)

        wf = WindField(**wind_field_config)
        # wind_preview_generator = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
        
        if input_dict["controller"]["wind_preview_type"] == "stochastic":
            def wind_preview_func(current_freestream_measurements, time_step, return_statistical_values=False): 
                # returns cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v
                if return_statistical_values:
                    distribution_params = generate_wind_preview( 
                                    wf, current_freestream_measurements, time_step,
                                    wind_preview_generator=wf._sample_wind_preview, 
                                    # n_preview_steps=input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] // input_dict["dt"]),
                                    # preview_dt=int(input_dict["controller"]["dt"] // input_dict["dt"]),
                                    # n_samples=input_dict["controller"]["n_wind_preview_samples"],
                                    return_params=True)
                    wind_preview_data = defaultdict(list)

                    mag = np.linalg.norm([current_freestream_measurements[0], current_freestream_measurements[1]])
                    wind_preview_data[f"FreestreamWindMag_{0}"] = {"mean": mag, "min": mag, "max": mag}
                    
                    # compute angle of arctan(u/v)
                    if current_freestream_measurements[1] != 0.0:
                        direction = np.arctan((abs(current_freestream_measurements[0]) / abs(current_freestream_measurements[1])))
                    elif current_freestream_measurements[0] >= 0:
                        direction = np.pi / 2
                    else:
                        direction = np.pi
                    
                    # if current_freestream_measurements[1] >= 0 and current_freestream_measurements[0] >= 0:
                    #     # first quadrant
                    #     angle = angle
                    if current_freestream_measurements[1] < 0 and current_freestream_measurements[0] >= 0:
                        # second quadrant
                        direction = np.pi - direction
                    elif current_freestream_measurements[1] < 0 and current_freestream_measurements[0] < 0:
                        # third quadrant
                        direction = np.pi + direction
                    else:
                        # fourth quadrant
                        direction = 2*np.pi - direction
                    # compute freestream wind direction angle from above, clockwise from north
                    direction = (direction + np.pi) * (180 / np.pi)

                    wind_preview_data[f"FreestreamWindDir_{0}"] = {"mean": direction, "min": direction, "max": direction}
                    
                    dev_u = 2 * np.sqrt(np.diag(distribution_params[2]))
                    dev_v = 2 * np.sqrt(np.diag(distribution_params[3]))
                    min_u = distribution_params[0] - dev_u
                    max_u = distribution_params[0] + dev_u
                    min_v = distribution_params[1] - dev_v
                    max_v = distribution_params[1] + dev_v

                    min_mags = np.linalg.norm([min_u, min_v], axis=0)
                    mean_mags = np.linalg.norm([distribution_params[0], distribution_params[1]], axis=0)
                    max_mags = np.linalg.norm([max_u, max_v], axis=0)

                    min_pos_u = [min_u[j] if min_u[j] > 0 else (0.0 if max_u[j] > 0 else None) for j in range(self.n_horizon)]
                    min_pos_v = [min_v[j] if min_v[j] > 0 else (0.0 if max_v[j] > 0 else None) for j in range(self.n_horizon)]
                    max_pos_u = [max_u[j] if max_u[j] > 0 else None for j in range(self.n_horizon)]
                    max_pos_v = [max_v[j] if max_v[j] > 0 else None for j in range(self.n_horizon)]

                    min_neg_u = [max_u[j] if max_u[j] < 0 else (0.0 if min_u[j] < 0 else None) for j in range(self.n_horizon)]
                    min_neg_v = [max_v[j] if max_v[j] < 0 else (0.0 if min_v[j] < 0 else None) for j in range(self.n_horizon)]
                    max_neg_u = [min_u[j] if min_u[j] < 0 else None for j in range(self.n_horizon)]
                    max_neg_v = [min_v[j] if min_v[j] < 0 else None for j in range(self.n_horizon)]
                    # TODO vectorize
#                     u_vals = np.vstack([min_pos_u, max_pos_u, min_neg_u, max_neg_u]).T
#                     v_vals = np.vstack([min_pos_v, max_pos_v, min_neg_v, max_neg_v]).T
#                     u_vals, v_vals = np.meshgrid(u_vals, v_vals)
#                     # compute directions
#                     u_only_dir = np.zeros_like(u_vals)
#                     u_only_dir[(v_vala == 0) & (u_preview >= 0)] = (np.pi / 2)
#                     u_only_dir[(v_preview == 0) & (u_preview < 0)] = np.pi
#                     # TODO below does not capture all quadrants of angle...
# å
#                     dir_preview = np.arctan(np.divide(abs(u_preview), abs(v_preview),
#                                             out=np.ones_like(u_preview) * np.nan,
#                                             where=v_preview != 0),
#                                     out=u_only_dir,
#                                     where=v_preview != 0)
#                     dir_preview[dir_preview < 0] = np.pi + dir_preview[dir_preview < 0]
#                     # dir_preview[(u_preview >= 0) & (v_preview >= 0)] = dir_preview[(u_preview >= 0) & (v_preview >= 0)] # first quadrant
#                     dir_preview[(u_preview >= 0) & (v_preview < 0)] = np.pi - dir_preview[(u_preview >= 0) & (v_preview < 0)] # second quadrant
#                     dir_preview[(u_preview < 0) & (v_preview < 0)] = np.pi + dir_preview[(u_preview < 0) & (v_preview < 0)] # third quadrant
#                     dir_preview[(u_preview < 0) & (v_preview >= 0)] = 2*np.pi - dir_preview[(u_preview < 0) & (v_preview >= 0)] # fourth quadrant

                    all_dirs = []
                    for j in range(self.n_horizon):
                        all_dirs.append([])
                        for u in [min_pos_u, max_pos_u, min_neg_u, max_neg_u]:
                            if u[j] is None:
                                continue
                            for v in [min_pos_v, max_pos_v, min_neg_v, max_neg_v]:
                                if v[j] is None:
                                    continue

                                if v[j] != 0.0:
                                    alpha = np.arctan((abs(u[j]) / abs(v[j])))
                                elif u[j] > 0:
                                    alpha = np.pi / 2
                                else:
                                    alpha = np.pi
                                
                                if v[j] >= 0 and u[j] >= 0:
                                    # first quadrant
                                    angle = alpha
                                elif v[j] < 0 and u[j] >= 0:
                                    # second quadrant
                                    angle = np.pi - alpha
                                elif v[j] < 0 and u[j] < 0:
                                    # third quadrant
                                    angle = np.pi + alpha
                                else:
                                    # fourth quadrant
                                    angle = 2*np.pi - alpha
                                
                                all_dirs[-1].append((angle + np.pi) * (180 / np.pi) % 360.0)

                    min_dirs = [min(all_dirs[j]) for j in range(self.n_horizon)]
                    max_dirs = [max(all_dirs[j]) for j in range(self.n_horizon)]

                    mean_dirs = []
                    for u, v in zip(distribution_params[0], distribution_params[1]):
                        if v != 0.0:
                            alpha = np.arctan((abs(u) / abs(v)))
                        elif u > 0:
                            alpha = np.pi / 2
                        else:
                            alpha = np.pi
                        
                        if v >= 0 and u >= 0:
                            # first quadrant
                            angle = alpha
                        elif v < 0 and u >= 0:
                            # second quadrant
                            angle = np.pi - alpha
                        elif v < 0 and u < 0:
                            # third quadrant
                            angle = np.pi + alpha
                        else:
                            # fourth quadrant
                            angle = 2*np.pi - alpha

                        mean_dirs.append((angle + np.pi) * (180 / np.pi) % 360.0)

                    for j in range(input_dict["controller"]["n_horizon"]):
                        wind_preview_data[f"FreestreamWindMag_{j + 1}"] = {"mean": mean_mags[j], "min": min_mags[j], "max": max_mags[j]}
                        wind_preview_data[f"FreestreamWindDir_{j + 1}"] = {"mean": mean_dirs[j], "min": min_dirs[j], "max": max_dirs[j]}

                else:
                    return generate_wind_preview(wf, current_freestream_measurements, time_step,
 								wind_preview_generator=wf._sample_wind_preview, 
 								# n_preview_steps=input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] // input_dict["dt"]),
 								# preview_dt=int(input_dict["controller"]["dt"] // input_dict["dt"]),
 								# n_samples=input_dict["controller"]["n_wind_preview_samples"]
                                return_params=False)
                
                return wind_preview_data
        
        elif input_dict["controller"]["wind_preview_type"] == "persistent":
            def wind_preview_func(current_freestream_measurements, time_step, return_statistical_values=False):
                if return_statistical_values:
                    wind_preview_data = {}
                    for j in range(input_dict["controller"]["n_horizon"] + 1):
                        wind_preview_data[f"FreestreamWindMag_{j}"] = {"mean": kwargs["wind_mag_ts"][time_step], "min": kwargs["wind_mag_ts"][time_step], "max": kwargs["wind_mag_ts"][time_step]}
                        wind_preview_data[f"FreestreamWindDir_{j}"] = {"mean": kwargs["wind_dir_ts"][time_step], "min": kwargs["wind_dir_ts"][time_step], "max": kwargs["wind_dir_ts"][time_step]}
                else:
                    wind_preview_data = defaultdict(list)
                    for j in range(input_dict["controller"]["n_horizon"] + 1):
                        wind_preview_data[f"FreestreamWindMag_{j}"] += [kwargs["wind_mag_ts"][time_step]]
                        wind_preview_data[f"FreestreamWindDir_{j}"] += [kwargs["wind_dir_ts"][time_step]]
                return wind_preview_data
        
        elif input_dict["controller"]["wind_preview_type"] == "perfect":
            def wind_preview_func(current_freestream_measurements, time_step, return_statistical_values=False):
                if return_statistical_values:
                    wind_preview_data = {}
                    for j in range(input_dict["controller"]["n_horizon"] + 1):
                        delta_k = j * int(input_dict["controller"]["dt"] // input_dict["dt"])
                        wind_preview_data[f"FreestreamWindMag_{j}"] = {"mean": kwargs["wind_mag_ts"][time_step + delta_k], "min": kwargs["wind_mag_ts"][time_step + delta_k], "max": kwargs["wind_mag_ts"][time_step + delta_k]}
                        wind_preview_data[f"FreestreamWindDir_{j}"] = {"mean": kwargs["wind_dir_ts"][time_step + delta_k], "min": kwargs["wind_dir_ts"][time_step + delta_k], "max": kwargs["wind_dir_ts"][time_step + delta_k]}
                else:
                    wind_preview_data = defaultdict(list)
                    for j in range(input_dict["controller"]["n_horizon"] + 1):
                        delta_k = j * int(input_dict["controller"]["dt"] // input_dict["dt"])
                        wind_preview_data[f"FreestreamWindMag_{j}"] += [kwargs["wind_mag_ts"][time_step + delta_k]]
                        wind_preview_data[f"FreestreamWindDir_{j}"] += [kwargs["wind_dir_ts"][time_step + delta_k]]
                return wind_preview_data
        
        self.wind_preview_func = wind_preview_func

        self.wind_preview_type = input_dict["controller"]["wind_preview_type"]
        if self.wind_preview_type == "stochastic":
            self.n_wind_preview_samples = input_dict["controller"]["n_wind_preview_samples"] + 1 # add one for the mean value
            # self.n_wind_preview_samples = 3 # min, mean, max
        else:
            self.n_wind_preview_samples = 1

        self.n_wind_preview_cost_func_samples = 1
        # self.n_wind_preview_samples = 1
        self.warm_start = input_dict["controller"]["warm_start"]

        if self.warm_start == "lut":
            fi_lut = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
                        .load_floris(config_path=input_dict["controller"]["floris_input_file"])

            lut_input_dict = dict(input_dict)
            lut_input_dict["controller"]["use_filtered_wind_dir"] = False
            ctrl_lut = LookupBasedWakeSteeringController(fi_lut, input_dict=lut_input_dict, 
                                                    lut_path=os.path.join(os.path.dirname(whoc.__file__), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{fi_lut.n_turbines}.csv"), 
                                                    generate_lut=False)
            def warm_start_func(measurements: dict):
                # feed interface with new disturbances
                fi_lut.step(disturbances={"wind_speeds": self.wind_preview_samples[f"FreestreamWindMag_{0}"][:1],
                                        "wind_directions": self.wind_preview_samples[f"FreestreamWindDir_{0}"][:1],
                                        "turbulence_intensities": fi_lut.env.core.flow_field.turbulence_intensities[:1]},
                                        seed=self.seed,
                                        ctrl_dict=None if fi_lut.time > 0 else {"yaw_angles": [ctrl_lut.yaw_IC] * ctrl_lut.n_turbines})
                
                # receive measurements from interface, compute control actions, and send to interface
                ctrl_lut.step()
                # must return yaw setpoints
                return ctrl_lut.controls_dict["yaw_angles"]

            # TODO define descriptor class to ensure that this function takes current_measurements
            self.warm_start_func = warm_start_func

        self.Q = self.alpha
        self.R = (1 - self.alpha)
        self.nu = input_dict["controller"]["nu"]
        # self.sequential_neighborhood_solve = input_dict["controller"]["sequential_neighborhood_solve"]
        self.yaw_norm_const = 360.0
        self.basin_hop = input_dict["controller"]["basin_hop"]

        if input_dict["controller"]["solver"].lower() in ['slsqp', 'sequential_slsqp', 'serial_refine']:
            self.solver = input_dict["controller"]["solver"].lower()
        else:
            raise TypeError("solver must be have value of 'slsqp', 'sequential_slsqp' or 'serial_refine'")
        
        self.n_solve_turbines = self.n_turbines if self.solver != "sequential_slsqp" else 1
        self.n_solve_states = self.n_solve_turbines * self.n_horizon
        self.n_solve_control_inputs = self.n_solve_turbines * self.n_horizon

        self.dyn_state_jac, self.state_jac = self.con_sens_rules(self.n_solve_turbines)
        
        # Set initial conditions
        self.yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        if hasattr(self.yaw_IC, "__len__"):
            if len(self.yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": self.yaw_IC}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": [self.yaw_IC] * self.n_turbines}
        
        self.initial_state = np.array([self.yaw_IC / self.yaw_norm_const] * self.n_turbines)
        
        self.opt_sol = {"states": np.tile(self.initial_state, self.n_horizon), 
                        "control_inputs": np.zeros((self.n_horizon * self.n_turbines,))}
        
        if input_dict["controller"]["control_input_domain"].lower() in ['discrete', 'continuous']:
            self.control_input_domain = input_dict["controller"]["control_input_domain"].lower()
        else:
            raise TypeError("control_input_domain must be have value of 'discrete' or 'continuous'")
        
        self.wind_ti = 0.08

        self.fi = ControlledFlorisModel(yaw_limits=self.yaw_limits, dt=self.dt, yaw_rate=self.yaw_rate) \
            .load_floris(config_path=input_dict["controller"]["floris_input_file"])
        
        if self.solver == "serial_refine":
            # self.fi_opt = FlorisModelDev(input_dict["controller"]["floris_input_file"]) #.replace("floris", "floris_dev"))
            if self.warm_start == "lut":
                print("Can't warm-start FLORIS SR solver, setting self.warm_start to none")
                # self.warm_start = "greedy"
        elif self.solver == "slsqp":
            self.pyopt_prob = self.setup_slsqp_solver(np.arange(self.n_turbines))
        elif self.solver == "zsgd":
            pass
        
    def con_sens_rules(self, n_solve_turbines):
        
        n_solve_states = n_solve_turbines * self.n_horizon
        n_solve_control_inputs = n_solve_turbines * self.n_horizon
        dyn_state_con_sens = {"states": [], "control_inputs": []}
        state_con_sens = {"states": [], "control_inputs": []}

        for j in range(self.n_horizon):
            for i in range(n_solve_turbines):
                current_idx = (n_solve_turbines * j) + i
                
                # scaled by yaw limit
                dyn_state_con_sens["control_inputs"].append([
                    -(self.dt * (self.yaw_rate / self.yaw_norm_const)) if idx == current_idx else 0
                    for idx in range(n_solve_control_inputs)
                ])
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    
                    dyn_state_con_sens["states"].append([
                        1 if idx == current_idx else 0
                        for idx in range(n_solve_states)
                    ])
                    
                    
                else:
                    prev_idx = (n_solve_turbines * (j - 1)) + i
                    
                    dyn_state_con_sens["states"].append([
                        1 if idx == current_idx else (-1 if idx == prev_idx else 0)
                        for idx in range(n_solve_states)
                    ])
                
                state_con_sens["states"].append([
                        -1 if idx == current_idx else 0
                        for idx in range(n_solve_states)
                    ])
                state_con_sens["control_inputs"].append([0] * n_solve_control_inputs)
        
        # TODO this should depend on constraint type
        if self.state_con_type == "check_all_samples":
            state_con_sens["states"] = state_con_sens["states"] * self.n_wind_preview_samples
            state_con_sens["control_inputs"] = state_con_sens["control_inputs"] * self.n_wind_preview_samples
        elif self.state_con_type == "extreme":
            state_con_sens["states"] = state_con_sens["states"] * 2
            state_con_sens["control_inputs"] = state_con_sens["control_inputs"] * 2
        elif self.state_con_type == "probabilistic":
            # TODO
            state_con_sens["states"] = "FD"
            state_con_sens["control_inputs"] = "FD"

        return dyn_state_con_sens, state_con_sens
    
    def dyn_state_rules(self, opt_var_dict, solve_turbine_ids):
        n_solve_turbines = len(solve_turbine_ids)
        # define constraints
        dyn_state_cons = []
        for j in range(self.n_horizon):
            for opt_var_i, turbine_i in enumerate(solve_turbine_ids):
                current_idx = (n_solve_turbines * j) + opt_var_i
                delta_yaw = self.dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[turbine_i] + delta_yaw)]
                    
                    
                else:
                    prev_idx = (n_solve_turbines * (j - 1)) + opt_var_i
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [
                        opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]
                    
        return dyn_state_cons
    
    def state_rules(self, opt_var_dict, disturbance_dict, yaw_setpoints, solve_turbine_ids):
        # define constraints
        state_cons = []
        # n_solve_turbines = len(solve_turbine_ids)
        if self.state_con_type == "check_all_samples":
            for m in range(self.n_wind_preview_samples):
                for j in range(self.n_horizon):
                    for turbine_i in solve_turbine_ids:
                        # current_idx = (self.n_solve_turbines * j) + i
                        # state_cons = state_cons + [(disturbance_dict["wind_direction"][(m * self.n_horizon) + j] / self.yaw_norm_const) - opt_var_dict["states"][current_idx]]
                        state_cons = state_cons + [(disturbance_dict["wind_direction"][(m * self.n_horizon) + j] - yaw_setpoints[j, turbine_i]) / self.yaw_norm_const]
        elif self.state_con_type == "extreme":
            # rather than including every sample, could only include most 'extreme' wind directions...
            # max_wd = [np.max([disturbance_dict["wind_direction"][(m * self.n_horizon) + j] for m in range(self.n_wind_preview_samples)]) for j in range(self.n_horizon)]
            # min_wd = [np.min([disturbance_dict["wind_direction"][(m * self.n_horizon) + j] for m in range(self.n_wind_preview_samples)]) for j in range(self.n_horizon)]
            max_wd = [disturbance_dict["wind_direction"][j]["max"] for j in range(self.n_horizon)]
            min_wd = [disturbance_dict["wind_direction"][j]["min"] for j in range(self.n_horizon)]
            
            for wd in [max_wd, min_wd]:    
                for j in range(self.n_horizon):
                    for turbine_i in solve_turbine_ids:
                        state_cons = state_cons + [(wd[j] - yaw_setpoints[j, turbine_i]) / self.yaw_norm_const]
        elif self.state_con_type == "probabilistic":
            current_time = np.atleast_1d(self.measurements_dict["time"])[0] 
            mean_u, mean_v, cov_u, cov_v = self.wind_preview_func(self.current_freestream_measurements, 
                                                               int(current_time // self.simulation_dt), 
                                                               return_params=True)
            for j in range(self.n_horizon):
                joint_probability = 1
                for turbine_i in solve_turbine_ids:
                    joint_probability *= self.compute_constraint_probability(yaw_setpoints[j, turbine_i], mean_u[j], mean_v[j], cov_u[j, j], cov_v[j, j])
                state_cons.append(joint_probability)
            # create joint chance constraint
            # state_cons = [np.prod(state_cons[j * n_solve_turbines])]
                    
        return state_cons
        

    def setup_slsqp_solver(self, solve_turbine_ids):
        n_solve_turbines = len(solve_turbine_ids)
        n_solve_states = n_solve_turbines * self.n_horizon
        n_solve_control_inputs = n_solve_turbines * self.n_horizon

        # initialize optimization object
        opt_rules = self.generate_opt_rules(solve_turbine_ids)
        dyn_state_jac, state_jac = self.con_sens_rules(n_solve_turbines)
        sens_rules = self.generate_sens_rules(solve_turbine_ids, dyn_state_jac, state_jac)
        pyopt_prob = Optimization("Wake Steering MPC", opt_rules, sens=sens_rules)
        
        # add design variables
        pyopt_prob.addVarGroup("states", n_solve_states,
                                    varType="c",  # continuous variables
                                    lower=[0] * n_solve_states,
                                    upper=[1] * n_solve_states,
                                    value=[0] * n_solve_states)
                                    # scale=(1 / self.yaw_norm_const))
        
        if self.control_input_domain == 'continuous':
            pyopt_prob.addVarGroup("control_inputs", n_solve_control_inputs,
                                        varType="c",
                                        lower=[-1] * n_solve_control_inputs,
                                        upper=[1] * n_solve_control_inputs,
                                        value=[0] * n_solve_control_inputs)
        else:
            pyopt_prob.addVarGroup("control_inputs", n_solve_control_inputs,
                                        varType="i",
                                        lower=[-1] * n_solve_control_inputs,
                                        upper=[1] * n_solve_control_inputs,
                                        value=[0] * n_solve_control_inputs)
        
        # # add dynamic state equation constraints
        # jac = self.con_sens_rules()
        pyopt_prob.addConGroup("dyn_state_cons", n_solve_states, lower=0.0, upper=0.0)
                        #   linear=True, wrt=["states", "control_inputs"], # NOTE supplying fixed jac won't work because value of initial_state changes
                        #   jac=jac)
        if self.state_con_type == "check_all_samples":
            pyopt_prob.addConGroup("state_cons", n_solve_states * self.n_wind_preview_samples, lower=self.yaw_limits[0] / self.yaw_norm_const, upper=self.yaw_limits[1] / self.yaw_norm_const)
            
        elif self.state_con_type == "extreme":
            pyopt_prob.addConGroup("state_cons", n_solve_states * 2, lower=self.yaw_limits[0] / self.yaw_norm_const, upper=self.yaw_limits[1] / self.yaw_norm_const)
            
        elif self.state_con_type == "probabilistic":
            pyopt_prob.addConGroup("state_cons", self.n_horizon, lower=self.beta, upper=None)

        # pyopt_prob.addConGroup("norm_turbine_powers_states_drvt", self.n_wind_preview_samples * self.n_horizon * self.n_turbines * n_solve_turbines)
        # pyopt_prob.addConGroup("norm_turbine_powers", self.n_wind_preview_samples * self.n_horizon * self.n_turbines)
        
        # add objective function
        pyopt_prob.addObj("cost")
        # pyopt_prob.constraints["state_cons"].ncon
        return pyopt_prob
    
    def compute_controls(self):
        """
        solve OCP to minimize objective over future horizon
        """
        
        current_time = np.atleast_1d(self.measurements_dict["time"])[0]
        if np.all(self.measurements_dict["wind_directions"] == 0):
            pass # will be set to initial values
        # TODO MISHA this is a patch up for AMR wind initialization problem
        elif (abs(current_time % self.dt) == 0.0) or (current_time == self.simulation_dt * 2):
            
            if current_time > 0.:
                # update initial state self.mi_model.initial_state
                # TODO MISHA should be able to get this from measurements dict
                current_yaw_angles = np.atleast_2d(self.controls_dict["yaw_angles"])[0, :]
                self.initial_state = current_yaw_angles / self.yaw_norm_const # scaled by yaw limits
            
            # current_wind_speeds = np.atleast_1d(self.measurements_dict["amr_wind_speed"])[0]
            # current_wind_directions = np.atleast_1d(self.measurements_dict["amr_wind_direction"])[0]
            current_wind_speeds = self.wind_mag_ts[int(current_time // self.simulation_dt)]
            current_wind_directions = self.wind_dir_ts[int(current_time // self.simulation_dt)]
            
            self.current_freestream_measurements = [
                    current_wind_speeds * np.sin((current_wind_directions - 180.) * (np.pi / 180.)),
                    current_wind_speeds * np.cos((current_wind_directions - 180.) * (np.pi / 180.))
            ]
            
            # returns n_preview_samples of horizon preview realiztions in the case of stochastic preview type, 
            # else just returns single values for persistent or perfect preview type
            self.wind_preview_samples = self.wind_preview_func(self.current_freestream_measurements, 
                                                               int(current_time // self.simulation_dt),
                                                               return_statistical_values=False)
            
            # returns dictionary of mean, min, max value expected from distribution, in the cahse of stochastic preview type
            self.wind_preview_stats = self.wind_preview_func(self.current_freestream_measurements, 
                                                                int(current_time // self.simulation_dt),
                                                                return_statistical_values=True)
            
            if self.wind_preview_type == "stochastic":
                # include mean value to compute cost function with
                for j in range(self.n_horizon + 1):
                    self.wind_preview_samples[f"FreestreamWindDir_{j}"].append(self.wind_preview_stats[f"FreestreamWindDir_{j}"]["mean"])
                    self.wind_preview_samples[f"FreestreamWindMag_{j}"].append(self.wind_preview_stats[f"FreestreamWindMag_{j}"]["mean"])

            self.fi.env.set(
                wind_directions=[self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)],
                wind_speeds=[self.wind_preview_samples[f"FreestreamWindMag_{j + 1}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)],
                turbulence_intensities=[self.fi.env.core.flow_field.turbulence_intensities[0]] * self.n_wind_preview_samples * self.n_horizon
            )

            # compute 
            # TODO is this a valid way to check
            current_powers = np.atleast_2d(self.measurements_dict["turbine_powers"])[0, :]
            self.offline_status = np.isclose(current_powers, 0.0)
            self.offline_status = np.broadcast_to(self.offline_status, (self.fi.env.core.flow_field.n_findex, self.n_turbines))
        
            if self.solver == "slsqp":
                yaw_star = self.slsqp_solve()
            elif self.solver == "sequential_slsqp":
                yaw_star = self.sequential_slsqp_solve()
            elif self.solver == "serial_refine":
                yaw_star = self.sr_solve()
            elif self.solver == "zsgd":
                yaw_star = self.zsgd_solve()
            
            # check constraints
            # assert np.isclose(sum(self.opt_sol["states"][:self.n_turbines] - (self.initial_state + self.opt_sol["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_norm_const) * self.dt)), 0, atol=1e-2)
            init_dyn_state_cons = (sum(self.opt_sol["states"][:self.n_turbines] - (self.initial_state + self.opt_sol["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_norm_const) * self.dt)))
            
            # this can sometimes not be satisfied if the iteration limit is exceeded
            if not np.isclose(init_dyn_state_cons, 0.0): #and not np.any(["Successfully" in c["Text"] for c in np.atleast_1d(self.opt_code)]):
                print(init_dyn_state_cons)
            
            # assert np.isclose(sum(self.opt_sol["states"][self.n_turbines:] - (self.opt_sol["states"][:-self.n_turbines] + self.opt_sol["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_norm_const) * self.dt)), 0)
            subsequent_dyn_state_cons = (sum(
                self.opt_sol["states"][self.n_turbines:] - (self.opt_sol["states"][:-self.n_turbines] + self.opt_sol["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_norm_const) * self.dt)
                ))

            if not np.isclose(subsequent_dyn_state_cons, 0.0): # this can sometimes not be satisfied if the iteration limit is exceeded
                print(subsequent_dyn_state_cons) # self.pyopt_sol_obj

            # assert np.isclose(sum((np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j}"]) / self.yaw_norm_const) - self.opt_sol["states"][(j * self.n_turbines) + i] for j in range(self.n_horizon) for i in range(self.n_turbines)), 0)
            # x = [(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] / self.yaw_norm_const) for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon) for i in range(self.n_solve_turbines)]
            # x = [self.opt_sol["states"][(j * self.n_turbines) + i] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon) for i in range(self.n_turbines)]
            # m = 0 is location of mean value
            state_cons = [(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][0] / self.yaw_norm_const) - self.opt_sol["states"][(j * self.n_turbines) + i] for j in range(self.n_horizon) for i in range(self.n_turbines)]
            
            # x = [(c > (self.yaw_limits[1] / self.yaw_norm_const) + 0.025) or (c < (self.yaw_limits[0] / self.yaw_norm_const) - 0.025) for c in state_cons]
            # np.where([(c > (self.yaw_limits[1] / self.yaw_norm_const) + 0.025) or (c < (self.yaw_limits[0] / self.yaw_norm_const) - 0.025) for c in state_cons])
            state_con_bools = (all([(c <= (self.yaw_limits[1] / self.yaw_norm_const) + 0.025) and (c >= (self.yaw_limits[0] / self.yaw_norm_const) - 0.025) for c in state_cons]))

            if not state_con_bools: # this can sometimes not be satisfied if the iteration limit is exceeded
                print(state_con_bools)

            self.controls_dict = {"yaw_angles": list(yaw_star)}
            
            # [c1 == c2 for c1, c2 in zip(self.pyopt_sol_obj.constraints["state_cons"].value, state_cons)]

    def zsgd_solve(self):

        self.warm_start_opt_vars()

        bounds = (-1, 1)
        A_eq = np.zeros(((self.n_horizon * self.n_turbines), self.n_horizon * self.n_turbines * 2))
        b_eq = np.zeros(((self.n_horizon * self.n_turbines), ))
        
        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (self.n_turbines * j) + i
                # delta_yaw = self.dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                A_eq[current_idx, (self.n_horizon * self.n_turbines) + current_idx] = -self.dt * (self.yaw_rate / self.yaw_norm_const)
                A_eq[current_idx, current_idx] = 1

                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    # scaled by yaw limit
                    # dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[i] + delta_yaw)]
                    
                    b_eq[current_idx] = self.initial_state[i]
                    
                else:
                    prev_idx = (self.n_turbines * (j - 1)) + i
                    # scaled by yaw limit
                    # dyn_state_cons = dyn_state_cons + [
                    #     opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]\
                    A_eq[current_idx, prev_idx] = -1
                    b_eq[current_idx] = 0

        i = 0
        step_size = 1 / (MPC.max_iter)
        acc = MPC.acc

        z_next = np.concatenate([self.init_sol["states"], self.init_sol["control_inputs"]])
        opt_var_dict = dict(self.init_sol)
        while i < MPC.max_iter:
            
            funcs, fail = self.opt_rules(opt_var_dict)
            sens = self.sens_rules(opt_var_dict, {})

            c = sens["cost"]["states"] + sens["cost"]["control_inputs"]
            # A_ub = np.vstack([1, -1] * np.ones((len(c),)))
            # b_ub = ([1] * (self.n_horizon * self.n_turbines) * 4)

            res = linprog(c=c, bounds=bounds, A_eq=A_eq, b_eq=b_eq)
            x_next = res.x

            # check Frank-Wolfe Gap 
            fw_gap = np.dot(c, z_next - x_next)
            if fw_gap < acc:
                break

            z_next = (1 - step_size) * z_next + (step_size * x_next)
            
            opt_var_dict = {"states": z_next[:self.n_horizon * self.n_turbines], "control_inputs": z_next[self.n_horizon * self.n_turbines:]}

            i += 1
        
        
        self.opt_sol = opt_var_dict
        self.opt_code = {"text": None}
        self.opt_cost = funcs["cost"]
        self.opt_cost_terms = [funcs["cost_states"], funcs["cost_control_inputs"]]

        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment

    def sr_solve(self):
        
        # warm-up with previous solution
        self.warm_start_opt_vars()
        if not self.opt_sol:
            self.opt_sol = {k: v.copy() for k, v in self.init_sol.items()}
        
        opt_yaw_setpoints = np.zeros((self.n_horizon, self.n_turbines))
        opt_cost = np.zeros((self.n_horizon))
        opt_cost_terms = np.zeros((self.n_horizon, 2))

        unconstrained_solve = True
        if unconstrained_solve:
            
            yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                    + (self.yaw_rate * self.dt * np.sum(self.opt_sol["control_inputs"][i:(self.n_turbines * j) + i:self.n_turbines])))
                                    for i in range(self.n_turbines)] for j in range(self.n_horizon)])
            
            current_yaw_offsets = (np.array(self.wind_preview_samples[f"FreestreamWindDir_{0}"][0]) - yaw_setpoints[0, :][np.newaxis, :])
        
            # optimize yaw angles
            yaw_offset_opt = YawOptimizationSRRHC(self.fi.env, 
                            self.yaw_rate, self.dt, self.alpha,
                            n_wind_preview_samples=self.n_wind_preview_samples,
                            n_horizon=self.n_horizon,
                            minimum_yaw_angle=self.yaw_limits[0],
                            maximum_yaw_angle=self.yaw_limits[1],
                            yaw_angles_baseline=np.zeros((self.n_turbines,)),
                            Ny_passes=[12, 8, 6],
                            verify_convergence=False)

            opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets=current_yaw_offsets, 
                                                         current_offline_status=self.offline_status,
                                                         constrain_yaw_dynamics=False, print_progress=True)
            
            # opt_yaw_setpoints = np.vstack([np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"]) - opt_yaw_offsets_df["yaw_angles_opt"].iloc[j] for j in range(self.n_horizon)])\
            # mean value
            opt_yaw_setpoints = np.vstack([self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["mean"] - opt_yaw_offsets_df["yaw_angles_opt"].iloc[j] for j in range(self.n_horizon)])
            # opt_yaw_setpoints_tmp = np.zeros_like(opt_yaw_setpoints)
            # for j in range(self.n_horizon):
            #     opt_yaw_setpoints_tmp[j, :] = np.clip(opt_yaw_setpoints[j, :],
            #                             self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["max"] - self.yaw_limits[1],
            #                             self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["min"] - self.yaw_limits[0],
            #                             )
            
            assert np.all(np.isclose(self.fi.env.core.flow_field.wind_directions - np.array([self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)]), 0.0))
            opt_cost = opt_yaw_offsets_df["cost"].to_numpy()
            opt_cost_terms[:, 0] = opt_yaw_offsets_df["cost_states"].to_numpy()
            opt_cost_terms[:, 1] = opt_yaw_offsets_df["cost_control_inputs"].to_numpy()

            # check that all yaw offsets are within limits, possible issue above
            assert np.all(np.vstack(opt_yaw_offsets_df["yaw_angles_opt"].to_numpy()) <= self.yaw_limits[1]) and np.all(np.vstack(opt_yaw_offsets_df["yaw_angles_opt"].to_numpy()) >= self.yaw_limits[0])
            
            # assert np.all(
            #     (((self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) <= self.yaw_limits[1]) 
            #       and ((self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) >= self.yaw_limits[0]))
            #         for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)
            
            assert np.all([(self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["mean"] - opt_yaw_setpoints[j, :]) <= self.yaw_limits[1] + 1e-12 for j in range(self.n_horizon)])

            # assert np.all([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) <= self.yaw_limits[1] + 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            assert np.all([(self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["mean"] - opt_yaw_setpoints[j, :]) >= self.yaw_limits[0] - 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            # check if solution adheres to dynamic state equaion
            # ensure that the rate of change is not greater than yaw_rate
            # clipped_opt_yaw_setpoints = np.zeros_like(opt_yaw_setpoints)
            for j in range(self.n_horizon):
                # gamma(k+1) in [gamma(k) - gamma_dot delta_t, gamma(k) + gamma_dot delta_t]
                init_gamma = self.initial_state * self.yaw_norm_const if j == 0 else opt_yaw_setpoints[j - 1, :]
                # delta_gamma = opt_yaw_setpoints[j, :] - init_gamma
                # if np.any(np.abs(delta_gamma) > self.yaw_rate * self.dt):
                #     pass
            
                opt_yaw_setpoints[j, :] = np.clip(opt_yaw_setpoints[j, :], init_gamma - self.yaw_rate * self.dt, init_gamma + self.yaw_rate * self.dt)
            
            # assert np.all([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) <= self.yaw_limits[1] + 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            # assert np.all([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) >= self.yaw_limits[0] - 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            
            opt_cost = np.sum(opt_cost)
            opt_cost_terms = np.sum(opt_cost_terms, axis=0)
        else:
            for j in range(self.n_horizon):
                for m in range(self.n_wind_preview_samples):
                    # solve at each time-step, checking that new yaw angles are feasible given last yaw angles, then average solutions over all samples
                    # optimize yaw angles
                    yaw_offset_opt = YawOptimizationSRRHC(self.fi.env, 
                                    self.yaw_rate, self.dt, self.alpha,
                                    n_wind_preview_samples=self.n_wind_preview_samples,
                                    n_horizon=self.n_horizon,
                                    minimum_yaw_angle=self.yaw_limits[0],
                                    maximum_yaw_angle=self.yaw_limits[1],
                                #  yaw_angles_baseline=np.zeros((1, 1, self.n_turbines)),
                                yaw_angles_baseline=np.zeros((self.n_turbines,)),
                                #  normalize_control_variables=True,
                                    Ny_passes=[12, 8, 6],
                                # x0=(self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] 
                                #     - self.init_sol["states"][j * self.n_turbines:(j + 1) * self.n_turbines] * self.yaw_norm_const)[np.newaxis, np.newaxis, :],
                                    verify_convergence=False)#, exploit_layout_symmetry=False)
                    if j == 0:
                        # self.initial_state = np.array([self.yaw_IC / self.yaw_norm_const] * self.n_turbines)
                        # current_yaw_offsets = self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - (self.initial_state * self.yaw_norm_const)
                        current_yaw_offsets = np.zeros((self.n_turbines, ))
                    else:
                        current_yaw_offsets = self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - opt_yaw_setpoints[m, j - 1, :]
                        
                    opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets, self.offline_status, print_progress=True)
                    opt_yaw_setpoints[m, j, :] = self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_offsets_df["yaw_angles_opt"].iloc[0]
                    opt_cost[m, j] = opt_yaw_offsets_df["cost"].iloc[0]
                    opt_cost_terms[m, j, 0] = opt_yaw_offsets_df["cost_states"].iloc[0]
                    opt_cost_terms[m, j, 1] = opt_yaw_offsets_df["cost_control_inputs"].iloc[0]

            # TODO change this     
            # opt_yaw_setpoints = np.mean(opt_yaw_setpoints, axis=0)
            # opt_cost = np.sum(np.mean(opt_cost, axis=0))
            # opt_cost_terms = np.sum(np.mean(opt_cost_terms, axis=0), axis=0)
            opt_yaw_setpoints = opt_yaw_setpoints
            opt_cost = np.sum(opt_cost)
            opt_cost_terms = np.sum(opt_cost_terms, axis=0)
                    
        self.opt_sol = {
            "states": np.array([opt_yaw_setpoints[j, i] / self.yaw_norm_const for j in range(self.n_horizon) for i in range(self.n_turbines)]), 
            "control_inputs": np.array([(opt_yaw_setpoints[j, i] - (opt_yaw_setpoints[j - 1, i] if j > 0 else self.initial_state[i] * self.yaw_norm_const)) * (1 / (self.yaw_rate * self.dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)])
        }
        

        self.opt_code = {"text": None}
        self.opt_cost = opt_cost
        # funcs, _ = self.opt_rules(self.opt_sol)
        self.opt_cost_terms = opt_cost_terms

        return np.rint(opt_yaw_setpoints[0, :] / self.yaw_increment) * self.yaw_increment
            # set the floris object with the predicted wind magnitude and direction at this time-step in the horizon

    def warm_start_opt_vars(self):
        self.init_sol = {"states": [], "control_inputs": []}
        # TODO should be able to get this from measurements_dict
        current_yaw_setpoints = np.atleast_2d(self.controls_dict["yaw_angles"])[0, :]
        
        if self.warm_start == "previous":
            current_time = np.atleast_1d(self.measurements_dict["time"])[0]
            if current_time > 0:
                self.init_sol = {
                    "states": np.clip(np.concatenate([
                     self.opt_sol["states"][self.n_turbines:], self.opt_sol["states"][-self.n_turbines:]
                     ]), 0.0, 1),
                    "control_inputs": np.clip(np.concatenate([
                        self.opt_sol["control_inputs"][self.n_turbines:], self.opt_sol["control_inputs"][-self.n_turbines:]
                        ]), -1, 1)
                }
            else:
                next_yaw_setpoints = np.array([self.yaw_IC / self.yaw_norm_const] * (self.n_horizon * self.n_turbines))
                current_control_inputs = np.array([0] * (self.n_horizon * self.n_turbines))
                self.init_sol["states"] = next_yaw_setpoints
                self.init_sol["control_inputs"] = current_control_inputs

        elif self.warm_start == "lut":
            # delta_yaw = self.dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][prev_idx]
            # TODO let this neglect wind direction filtering?
            self.init_sol = {
                    "states": [],
                    "control_inputs": []
            }
            for j in range(self.n_horizon):
                # compute yaw angle setpoints from warm_start_func
                
                # if self.wind_preview_type == "stochastic":
                next_yaw_setpoints = self.warm_start_func(
                    {
                        "wind_speeds": [self.wind_preview_stats[f"FreestreamWindMag_{j + 1}"]["mean"]], 
                        "wind_directions": [self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"]["mean"]]
                    })
                
                current_control_inputs = (next_yaw_setpoints - current_yaw_setpoints) * (1 / (self.yaw_rate * self.dt))
                
                self.init_sol["states"] = self.init_sol["states"] + list(next_yaw_setpoints / self.yaw_norm_const)
                self.init_sol["control_inputs"] = self.init_sol["control_inputs"] + list(current_control_inputs)
                
                current_yaw_setpoints = next_yaw_setpoints

            self.init_sol["states"] = np.array(self.init_sol["states"])
            self.init_sol["control_inputs"] = np.array(self.init_sol["control_inputs"])

        elif self.warm_start == "greedy":
            self.init_sol["states"] = (self.wind_preview_samples[f"FreestreamWindDir_{0}"][0] / self.yaw_norm_const) * np.ones((self.n_turbines * self.n_horizon,))
            self.init_sol["control_inputs"] = (self.init_sol["states"] - self.opt_sol["states"][:]) * (self.yaw_norm_const / (self.yaw_rate * self.dt))

        if self.basin_hop:
            def basin_hop_obj(opt_var_arr):
                funcs, _ = self.opt_rules({"states": opt_var_arr[:self.n_horizon * self.n_turbines],
                                "control_inputs": opt_var_arr[self.n_horizon * self.n_turbines:]},
                                compute_derivatives=False, compute_constraints=False)
                return funcs["cost"]
            
            basin_hop_init_sol = basinhopping(basin_hop_obj, np.concatenate([self.init_sol["states"], self.init_sol["control_inputs"]]), niter=20, stepsize=0.2, disp=True)
            self.init_sol["states"] = basin_hop_init_sol.x[:self.n_horizon * self.n_turbines]
            self.init_sol["control_inputs"] = basin_hop_init_sol.x[self.n_horizon * self.n_turbines:]
        
    def sequential_slsqp_solve(self):
        
        # set self.opt_sol to initial solution
        self.warm_start_opt_vars()
        self.opt_sol = {k: v.copy() for k, v in self.init_sol.items()}
        self.opt_cost = 0
        self.opt_cost_terms = [0, 0]
        self.opt_code = []

        # rotate turbine coordinates based on most recent wind direction measurement
        # order turbines based on order of wind incidence
        layout_x = self.fi.env.layout_x
        layout_y = self.fi.env.layout_y
        # turbines_ordered_array = []
        wd = self.wind_preview_samples["FreestreamWindDir_0"][0]
        # wd = 250.0
        layout_x_rot = (
            np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
            - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
        )
        # layout_y_rot = (
        #     np.cos((wd - 270.0) * np.pi / 180.0) * layout_y
        #     - np.sin((wd - 270.0) * np.pi / 180.0) * layout_x
        # )
        turbines_ordered = np.argsort(layout_x_rot)
        # turbines_ordered_array.append(turbines_ordered)
        # self.turbines_ordered_array = np.vstack(turbines_ordered_array)

        grouped_turbines_ordered = []
        t = 0
        while t < self.n_turbines:
            grouped_turbines_ordered.append([turbines_ordered[t]])
            tt = t + 1
            while tt < self.n_turbines:
                if np.abs(layout_x_rot[turbines_ordered[t]] - layout_x_rot[turbines_ordered[tt]]) < 2 * self.fi.env.core.farm.turbine_definitions[0]["rotor_diameter"]:
                    # or np.abs(layout_y_rot[turbines_ordered[t]] - layout_y_rot[turbines_ordered[tt]]) > 6 * self.fi.env.core.farm.turbine_definitions[0]["rotor_diameter"]):
                    grouped_turbines_ordered[-1].append(turbines_ordered[tt])
                    tt += 1
                else:
                    break
            t = tt
            grouped_turbines_ordered[-1].sort()

        # for each turbine in sorted array
        n_solve_turbine_groups = len(grouped_turbines_ordered)

        solutions = []
        for turbine_group_idx in range(n_solve_turbine_groups):
            solve_turbine_ids = grouped_turbines_ordered[turbine_group_idx]
            n_solve_turbines = len(solve_turbine_ids)
            solutions.append(self.solve_turbine_group(solve_turbine_ids))

            # update self.opt_sol with most recent solutions for all turbines
            for opt_var_idx, solve_turbine_idx in enumerate(solve_turbine_ids):
                self.opt_sol["states"][solve_turbine_idx::self.n_turbines] = np.array(solutions[turbine_group_idx].xStar["states"])[opt_var_idx::n_solve_turbines]
                self.opt_sol["control_inputs"][solve_turbine_idx::self.n_turbines] = np.array(solutions[turbine_group_idx].xStar["control_inputs"])[opt_var_idx::n_solve_turbines]
            self.opt_code.append(solutions[turbine_group_idx].optInform)
            self.opt_cost = ((self.opt_cost * turbine_group_idx) + solutions[turbine_group_idx].fStar) / (turbine_group_idx + 1)
            
        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment

    def slsqp_solve(self):
        
        # warm start Vars by reinitializing the solution from last time-step self.mi_model.states, set self.init_sol
        self.warm_start_opt_vars()

        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (j * self.n_turbines) + i
                # next_idx = ((j + 1) * self.n_turbines) + i
                self.pyopt_prob.variables["states"][current_idx].value \
                    = self.init_sol["states"][current_idx]
                
                self.pyopt_prob.variables["control_inputs"][current_idx].value \
                    = self.init_sol["control_inputs"][current_idx]
                
        # dyn_state_jac, state_jac = self.con_sens_rules(self.n_turbines)
        # sens_rules = self.generate_sens_rules(np.arange(self.n_turbines), dyn_state_jac, state_jac)
        sol = self.optimizer(self.pyopt_prob,) # timeLimit=self.dt) #, sens=sens_rules) #, sensMode='pgc')
        # sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens="FD")
        self.pyopt_sol_obj = sol
        self.opt_sol = {k: v[:] for k, v in sol.xStar.items()}
        self.opt_code = sol.optInform
        self.opt_cost = sol.fStar
        assert sum(self.opt_cost_terms) == self.opt_cost

        # solution is scaled by yaw limit
        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment
        # return (self.initial_state * self.yaw_norm_const) \
        # 	+ self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]  # solution is scaled by yaw limit

    def solve_turbine_group(self, solve_turbine_ids):
        # solve_turbine_ids = grouped_turbines_ordered[turbine_group_idx]
        n_solve_turbines = len(solve_turbine_ids)

        pyopt_prob = self.setup_slsqp_solver(solve_turbine_ids)
        # sens_rules = self.generate_sens_rules(solve_turbine_ids)
        
        # setup pyopt problem to consider 2 optimization variables for this turbine and set all others as fixed parameters

        for j in range(self.n_horizon):
            for opt_var_idx, solve_turbine_idx in enumerate(solve_turbine_ids):
                pyopt_prob.variables["states"][(j * n_solve_turbines) + opt_var_idx].value \
                    = self.init_sol["states"][(j * self.n_turbines) + solve_turbine_idx]
                
                pyopt_prob.variables["control_inputs"][(j * n_solve_turbines) + opt_var_idx].value \
                    = self.init_sol["control_inputs"][(j * self.n_turbines) + solve_turbine_idx]
        
        # solve problem based on self.opt_sol
        sol = self.optimizer(pyopt_prob) #, timeLimit=self.dt) #, sens=sens_rules) #, sensMode='pgc')
        return sol

    def generate_opt_rules(self, solve_turbine_ids):
        n_solve_turbines = len(solve_turbine_ids)
        def opt_rules(opt_var_dict, compute_constraints=True, compute_derivatives=True):

            funcs = {}
            if self.solver == "sequential_slsqp":
                control_inputs = np.array(self.opt_sol["control_inputs"])
                
                for opt_var_id, turbine_id in enumerate(solve_turbine_ids):
                    control_inputs[turbine_id::self.n_turbines] = opt_var_dict["control_inputs"][opt_var_id::n_solve_turbines]

                # the yaw setpoints for the future horizon (current/iniital state is known and not an optimization variable)
                yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                        + (self.yaw_rate * self.dt * np.sum(control_inputs[i:(self.n_turbines * (j + 1)) + i:self.n_turbines])))
                                        for i in range(self.n_turbines)] for j in range(self.n_horizon)])
            
            else:
                # the yaw setpoints for the future horizon (current/iniital state is known and not an optimization variable)
                yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                        + (self.yaw_rate * self.dt * np.sum(opt_var_dict["control_inputs"][i:(self.n_turbines * (j + 1)) + i:self.n_turbines])))
                                        for i in range(self.n_turbines)] for j in range(self.n_horizon)])
            
            # plot_distribution_samples(pd.DataFrame(wind_preview_samples), self.n_horizon)
            # derivative of turbine power output with respect to yaw angles
            
            if compute_constraints:
                if self.state_con_type == "extreme":
                    funcs["state_cons"] = self.state_rules(opt_var_dict, 
                                                        {
                                                            "wind_direction": [self.wind_preview_stats[f"FreestreamWindDir_{j + 1}"] for j in range(self.n_horizon)]
                                                            }, yaw_setpoints, solve_turbine_ids)
                else:
                    funcs["state_cons"] = self.state_rules(opt_var_dict, 
                                                        {
                                                            "wind_direction": [self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] 
                                                                            for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)]
                                                            }, yaw_setpoints, solve_turbine_ids)
                    
                # if len(funcs["state_cons"]) == 90:
                #     print("oh")
                
                funcs["dyn_state_cons"] = self.dyn_state_rules(opt_var_dict, solve_turbine_ids)

            # np.unique(np.where(np.abs(current_yaw_offsets) > 90)[0])
            # send yaw angles 

            # greedily yaw directly into wind for normalization constant
            yaw_offsets = np.zeros((self.n_wind_preview_samples * self.n_horizon, self.n_turbines))
            self.fi.env.set_operation(
                yaw_angles=yaw_offsets,
                disable_turbines=self.offline_status,
            )
            self.fi.env.run()
            all_greedy_yaw_turbine_powers = self.fi.env.get_turbine_powers()
            # greedy_yaw_turbine_powers = np.reshape(greedy_yaw_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
            greedy_yaw_turbine_powers = np.max(all_greedy_yaw_turbine_powers, axis=1)[:, np.newaxis] # choose unwaked turbine for normalization constant
            
            # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
            
            current_yaw_offsets = np.vstack([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - yaw_setpoints[j, :]) for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            # current_yaw_offsets = np.vstack([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][0] - yaw_setpoints[j, :]) for j in range(self.n_horizon)])
            
            yaw_offsets = np.clip(current_yaw_offsets, *self.yaw_limits)
            self.fi.env.set_operation(
                yaw_angles=yaw_offsets,
                disable_turbines=self.offline_status,
            )
            
            # self.fi.env.set(yaw_angles=np.clip(current_yaw_offsets, *self.yaw_limits), disable_turbines=self.offline_status)
            self.fi.env.run()
            yawed_turbine_powers = self.fi.env.get_turbine_powers()

            # TODO add negative power to paper
            decay_factor = -np.log(1e-6) / ((90 - np.max(np.abs(self.yaw_limits))) / self.yaw_norm_const)
            neg_decay = np.exp(-decay_factor * (self.yaw_limits[0] - current_yaw_offsets[current_yaw_offsets < self.yaw_limits[0]]) / self.yaw_norm_const)
            pos_decay = np.exp(-decay_factor * (current_yaw_offsets[current_yaw_offsets > self.yaw_limits[1]] - self.yaw_limits[1]) / self.yaw_norm_const)
            yawed_turbine_powers[(current_yaw_offsets < self.yaw_limits[0])] = yawed_turbine_powers[current_yaw_offsets < self.yaw_limits[0]] * neg_decay
            yawed_turbine_powers[(current_yaw_offsets > self.yaw_limits[1])] = yawed_turbine_powers[current_yaw_offsets > self.yaw_limits[1]] * pos_decay
            assert not np.any(np.isnan(yawed_turbine_powers))
            
            # yawed_turbine_powers[np.isnan(yawed_turbine_powers)] = 0.0 # add negative power magnitude depends how close offsets are over 30
                
            # normalize power by no yaw output
            norm_turbine_powers = np.divide(yawed_turbine_powers, greedy_yaw_turbine_powers,
                                                    where=greedy_yaw_turbine_powers!=0,
                                                    out=np.zeros_like(yawed_turbine_powers))
            norm_turbine_powers = np.reshape(norm_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
            # norm_turbine_powers = np.reshape(norm_turbine_powers, (1, self.n_horizon, self.n_turbines))
            
            assert not np.any(np.isnan(norm_turbine_powers))

            # compute power based on sampling from wind preview
            # funcs["norm_turbine_powers"] = norm_turbine_powers.flatten()
            self.norm_turbine_powers = np.reshape(norm_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
            # self.norm_turbine_powers = np.reshape(norm_turbine_powers, (1, self.n_horizon, self.n_turbines))
            # funcs["cost_states"] = sum([-0.5*np.mean((norm_turbine_powers[:, j, i])**2) * self.Q
            #                     for j in range(self.n_horizon) for i in range(self.n_turbines)])
            # mean value is last element for stochastic case
            funcs["cost_states"] = sum([-0.5*(norm_turbine_powers[-1, j, i]**2) * self.Q
                                for j in range(self.n_horizon) for i in range(self.n_turbines)])
            
            funcs["cost_control_inputs"] = sum(0.5*(opt_var_dict["control_inputs"][(n_solve_turbines * j) + i])**2 * self.R
                                for j in range(self.n_horizon) for i in range(n_solve_turbines))
            
            assert not np.isnan(funcs["cost_states"])
            assert not np.isnan(funcs["cost_control_inputs"])

            if compute_derivatives:
                if self.wind_preview_type == "stochastic":
                    
                    u = np.random.normal(loc=0.0, scale=1.0, size=(self.n_wind_preview_samples * self.n_horizon, n_solve_turbines))
                    
                    # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
                    
                    if self.solver == "sequential_slsqp":
                        # mask = np.zeros((self.n_turbines,))
                        # mask[solve_turbine_ids] = 1
                        masked_u = np.zeros((self.n_wind_preview_samples * self.n_horizon, self.n_turbines))
                        # masked_u = np.zeros((1 * self.n_horizon, self.n_turbines))
                        masked_u[:, solve_turbine_ids] = u
                        plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * masked_u
                    else:
                        plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * u
                    
                    # self.fi.env.set(yaw_angles=, disable_turbines=self.offline_status)
                    yaw_offsets = np.clip(plus_yaw_offsets, *self.yaw_limits)
                    self.fi.env.set_operation(
                        yaw_angles=yaw_offsets,
                        disable_turbines=self.offline_status,
                    )
                    self.fi.env.run()
                    plus_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                    # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
                    
                    neg_decay = np.exp(-decay_factor * (self.yaw_limits[0] - plus_yaw_offsets[plus_yaw_offsets < self.yaw_limits[0]]) / self.yaw_norm_const)
                    pos_decay = np.exp(-decay_factor * (plus_yaw_offsets[plus_yaw_offsets > self.yaw_limits[1]] - self.yaw_limits[1]) / self.yaw_norm_const)
                    plus_perturbed_yawed_turbine_powers[(plus_yaw_offsets < self.yaw_limits[0])] = plus_perturbed_yawed_turbine_powers[plus_yaw_offsets < self.yaw_limits[0]] * neg_decay
                    plus_perturbed_yawed_turbine_powers[(plus_yaw_offsets > self.yaw_limits[1])] = plus_perturbed_yawed_turbine_powers[plus_yaw_offsets > self.yaw_limits[1]] * pos_decay
                    # plus_perturbed_yawed_turbine_powers[np.isnan(plus_perturbed_yawed_turbine_powers)] = 0.0
                    assert not np.any(np.isnan(plus_perturbed_yawed_turbine_powers))
                    
                    norm_turbine_power_diff = np.divide((plus_perturbed_yawed_turbine_powers - yawed_turbine_powers), greedy_yaw_turbine_powers,
                                                        where=greedy_yaw_turbine_powers!=0,
                                                        out=np.zeros_like(yawed_turbine_powers))

                    # should compute derivative of each power of each turbine wrt state (yaw angle) of each turbine
                    norm_turbine_powers_states_drvt = np.einsum("ia, ib->iab", norm_turbine_power_diff / self.nu, u)

                elif self.wind_preview_type == "persistent" or self.wind_preview_type == "perfect":

                    norm_turbine_powers_states_drvt = np.zeros((self.n_wind_preview_samples * self.n_horizon, self.n_turbines, n_solve_turbines))
                    # norm_turbine_powers_states_drvt = np.zeros((1 * self.n_horizon, self.n_turbines, n_solve_turbines))
                    # perturb each state (each yaw angle) by +/= nu to estimate derivative of all turbines power output for a variation in each turbines yaw offset
                    # if any yaw offset are out of the [-90, 90] range, then the power output of all turbines will be nan. clip to avoid this

                    # TODO parallelize across solve turbine perturbations
                    # u = np.tile(np.eye(self.n_solve_turbines), (self.n_solve_turbines * self.n_horizon, 1))

                    for i in range(n_solve_turbines):
                        mask = np.zeros((self.n_turbines,))
                        # if self.solver == "sequential_slsqp":
                        mask[solve_turbine_ids[i]] = 1
                        # else:
                        #     mask[i] = 1
                        
                        # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
                        plus_yaw_offsets = current_yaw_offsets - mask * self.nu * self.yaw_norm_const
                        
                        yaw_offsets = np.clip(plus_yaw_offsets, *self.yaw_limits)
                        # self.fi.env.set(yaw_angles=np.clip(plus_yaw_offsets, *self.yaw_limits), disable_turbines=self.offline_status)
                        self.fi.env.set_operation(
                            yaw_angles=yaw_offsets,
                            disable_turbines=self.offline_status,
                        )
                        self.fi.env.run()
                        plus_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                        # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
                        
                        neg_decay = np.exp(-decay_factor * (self.yaw_limits[0] - plus_yaw_offsets[plus_yaw_offsets < self.yaw_limits[0]]) / self.yaw_norm_const)
                        pos_decay = np.exp(-decay_factor * (plus_yaw_offsets[plus_yaw_offsets > self.yaw_limits[1]] - self.yaw_limits[1]) / self.yaw_norm_const)
                        plus_perturbed_yawed_turbine_powers[(plus_yaw_offsets < self.yaw_limits[0])] = plus_perturbed_yawed_turbine_powers[plus_yaw_offsets < self.yaw_limits[0]] * neg_decay
                        plus_perturbed_yawed_turbine_powers[(plus_yaw_offsets > self.yaw_limits[1])] = plus_perturbed_yawed_turbine_powers[plus_yaw_offsets > self.yaw_limits[1]] * pos_decay
                        
                        # plus_perturbed_yawed_turbine_powers[np.isnan(plus_perturbed_yawed_turbine_powers)] = 0.0 
                        assert not np.any(np.isnan(plus_perturbed_yawed_turbine_powers))
                        # we add negative since current_yaw_offsets = wind dir - yaw setpoints
                        neg_yaw_offsets = current_yaw_offsets + mask * self.nu * self.yaw_norm_const
                        yaw_offsets = np.clip(neg_yaw_offsets, *self.yaw_limits)
                        # self.fi.env.core.farm.set_yaw_angles(self.fi.env.core.farm.yaw_angles)
                        self.fi.env.set_operation(
                            yaw_angles=yaw_offsets,
                            disable_turbines=self.offline_status,
                        )
                        # self.fi.env.set(yaw_angles=np.clip(neg_yaw_offsets, *self.yaw_limits), disable_turbines=self.offline_status)
                        self.fi.env.run()
                        neg_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                        # TODO negative coefficient of thrust ... if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
                        
                        neg_decay = np.exp(-decay_factor * (self.yaw_limits[0] - neg_yaw_offsets[neg_yaw_offsets < self.yaw_limits[0]]) / self.yaw_norm_const)
                        pos_decay = np.exp(-decay_factor * (neg_yaw_offsets[neg_yaw_offsets > self.yaw_limits[1]] - self.yaw_limits[1]) / self.yaw_norm_const)
                        neg_perturbed_yawed_turbine_powers[(neg_yaw_offsets < self.yaw_limits[0])] = neg_perturbed_yawed_turbine_powers[neg_yaw_offsets < self.yaw_limits[0]] * neg_decay
                        neg_perturbed_yawed_turbine_powers[(neg_yaw_offsets > self.yaw_limits[1])] = neg_perturbed_yawed_turbine_powers[neg_yaw_offsets > self.yaw_limits[1]] * pos_decay
                        
                        # neg_perturbed_yawed_turbine_powers[np.isnan(neg_perturbed_yawed_turbine_powers)] = 0.0 
                        assert not np.any(np.isnan(neg_perturbed_yawed_turbine_powers))
                        
                        norm_turbine_power_diff = np.divide((plus_perturbed_yawed_turbine_powers - neg_perturbed_yawed_turbine_powers), greedy_yaw_turbine_powers,
                                                        where=greedy_yaw_turbine_powers!=0,
                                                        out=np.zeros_like(plus_perturbed_yawed_turbine_powers))
                        # TODO why should changing yaw angles in same row influence last row of turbines
                        # should compute derivative of each power of each turbine wrt state (yaw angle) of each turbine
                        # self.norm_turbine_powers_states_drvt = (norm_turbine_power_diff / self.nu).T @ u
                        norm_turbine_powers_states_drvt[:, :, i] = norm_turbine_power_diff / (2 * self.nu)
                        # np.einsum("ia, ib->iab", norm_turbine_power_diff / self.nu, u)
                    # norm_turbine_powers_states_drvt = np.reshape(norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines, n_solve_turbines))

                # funcs["norm_turbine_powers_states_drvt"] = norm_turbine_powers_states_drvt.flatten()
                self.norm_turbine_powers_states_drvt = np.reshape(norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines, n_solve_turbines))
                
                assert not np.any(np.isnan(self.norm_turbine_powers_states_drvt))
            
            funcs["cost"] = funcs["cost_states"] + funcs["cost_control_inputs"]
            
            if self.solver == "sequential_slsqp":
                self.opt_cost_terms[0] += funcs["cost_states"]
                self.opt_cost_terms[1] += funcs["cost_control_inputs"]
            else:
                self.opt_cost_terms = [funcs["cost_states"], funcs["cost_control_inputs"]]
            
            fail = False
            
            return funcs, fail
        
        # opt_rules.__qualname__ = "opt_rules"
        return opt_rules
        
    def generate_sens_rules(self, solve_turbine_ids, dyn_state_jac, state_jac):
        n_solve_turbines = len(solve_turbine_ids)
        def sens_rules(opt_var_dict, obj_con_dict):
            sens = {"cost": {"states": [], "control_inputs": []},
                    "dyn_state_cons": {"states": [], "control_inputs": []},
                    "state_cons": {"states": [], "control_inputs": []}}
            
            # norm_turbine_powers =  np.reshape(obj_con_dict["norm_turbine_powers"], (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
            # norm_turbine_powers_states_drvt = np.reshape(obj_con_dict["norm_turbine_powers_states_drvt"], (self.n_wind_preview_samples, self.n_horizon, self.n_turbines, n_solve_turbines))

            # if self.wind_preview_type == "stochastic":
            for j in range(self.n_horizon):
                for i in range(n_solve_turbines):
                    current_idx = (n_solve_turbines * j) + i
                    # TODO check
                    # compute power derivative based on sampling from wind preview with respect to changes to the state/control input of this turbine/horizon step
                    # using derivative: power of each turbine wrt each turbine's yaw setpoint, summing over terms for each turbine
                    sens["cost"]["states"].append(
                                np.mean(np.sum(-(self.norm_turbine_powers[:, j, :]) * self.Q * self.norm_turbine_powers_states_drvt[:, j, :, i], axis=1))
                            )
                    
                    sens["cost"]["control_inputs"].append(
                        (opt_var_dict["control_inputs"][current_idx] * self.R) + (sens["cost"]["states"][-1] * self.dt * (self.yaw_rate / self.yaw_norm_const))
                    )
                    
            
            sens["dyn_state_cons"] = dyn_state_jac
            sens["state_cons"] = state_jac
            return sens
        
        # sens_rules.__qualname__ = "sens_rules"
        return sens_rules
    
    def compute_constraint_probability(self, yaw_setpoint, mean_u, mean_v, cov_u, cov_v):
        # for j in range(self.n_horizon):
        #     for i in range(self.n_turbines):
        ub_ratio = (self.yaw_limits[1] + yaw_setpoint - 180.) # % 360.
        lb_ratio = (self.yaw_limits[0] + yaw_setpoint - 180.) # % 360.

        # if ub_ratio > 90. and ub_ratio < 180.:
        #     ub_ratio = 180. - ub_ratio
        # elif ub_ratio > 270. and ub_ratio < 360.:
        #     ub_ratio = 360. - ub_ratio
        
        # if lb_ratio > 90. and lb_ratio < 180.:
        #     lb_ratio = 180. - lb_ratio
        # elif lb_ratio > 270. and lb_ratio < 360.:
        #     lb_ratio = 360. - lb_ratio

        ub = np.tan(ub_ratio * np.pi / 180.0)
        lb = np.tan(lb_ratio * np.pi / 180.0)
        assert ub >= lb
        return MPC.ratio_cdf(ub, mean_u, mean_v, cov_u, cov_v) - MPC.ratio_cdf(lb, mean_u, mean_v, cov_u, cov_v)

    @staticmethod
    def ratio_cdf(z, mean_u, mean_v, cov_u, cov_v):
        return MPC._L((mean_u - mean_v * z) / (cov_u * cov_v * MPC._a(z, cov_u, cov_v)), -mean_v / cov_v, z / (cov_u * MPC._a(z, cov_u, cov_v))) + MPC._L((mean_v * z - mean_u) / (cov_u * cov_v * MPC._a(z, cov_u, cov_v)), mean_v / cov_v, z / (cov_u * MPC._a(z, cov_u, cov_v)))

    @staticmethod
    def _a(z, cov_u, cov_v):
        return np.sqrt(cov_u**(-2) * z**2 + cov_v**(-2))
    
    @staticmethod
    def _b(z, mean_u, mean_v, cov_u, cov_v):
        return mean_u * cov_u**(-2) * z + mean_v * cov_v**(-2)
    
    @staticmethod
    def _c(mean_u, mean_v, cov_u, cov_v):
        return mean_u**2 * cov_u**(-2) + mean_v**2 * cov_v**(-2)
    
    @staticmethod
    def _d(z, mean_u, mean_v, cov_u, cov_v):
        return np.exp((MPC._b(z, mean_u, mean_v, cov_u, cov_v)**2 - MPC._c(mean_u, mean_v, cov_u, cov_v) * MPC._a(z, cov_u, cov_v)**2) / (2 * MPC._a(z, cov_u, cov_v)**2))
    
    @staticmethod
    def _L(h, k, rho):
        # bivariate_normal_integral
        y, _ = dblquad(lambda x, y: np.exp(-(x**2 - (2 * rho * x * y) + y**2) / (2 * np.sqrt(1 - rho**2))), h, 1e+16, k, 1e+16)
        
        return (1 / (2 * np.pi * np.sqrt(1 - rho**2))) * y
    
# if __name__ == "__main__":


    # fi = ControlledFlorisModel(yaw_limits=input_dict["controller"]["yaw_limits"],
    #                                     offline_probability=input_dict["controller"]["offline_probability"],
    #                                     dt=input_dict["dt"],
    #                                     yaw_rate=input_dict["controller"]["yaw_rate"]) \
    #     .load_floris(config_path=input_dict["controller"]["floris_input_file"])

    # ctrl = MPC(fi, input_dict=input_dict, wind_mag_ts=wind_mag_ts, wind_dir_ts=wind_dir_ts, 
    #                                    lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"],
    #                                    wind_field_config=wind_field_config)

