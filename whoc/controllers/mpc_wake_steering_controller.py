from time import perf_counter, time, strftime
from glob import glob
from functools import partial
import copy
from collections import defaultdict
import yaml
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyoptsparse import Optimization, SLSQP
from scipy.optimize import linprog, basinhopping

import whoc
from whoc.config import *
from whoc.controllers.controller_base import ControllerBase, simulate_controller
from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.wind_field.WindField import generate_multi_wind_ts, generate_multi_wind_preview_ts, WindField, generate_wind_preview
from whoc.postprocess_case_studies import plot_wind_field_ts, plot_opt_var_ts, plot_opt_cost_ts

from floris.tools import FlorisInterface as FlorisInterfaceDev
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.tools.optimization.yaw_optimization.yaw_optimization_base import derive_downstream_turbines

# from floris_dev.tools import FlorisInterface as FlorisInterfaceDev
# from floris_dev.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

# TODO pass input_dict and floris_input_file

optimizer_idx = 0

class YawOptimizationSRRHC(YawOptimizationSR):
    def __init__(
        self,
        fi,
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
        Instantiate YawOptimizationSR object with a FlorisInterface object
        and assign parameter values.
        """

        # Initialize base class
        super().__init__(
            fi=fi,
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

        # self.fi.floris.flow_field.n_findex = self.n_horizon
        # self._n_findex_subset = self.n_horizon

        # if yaw_angles_baseline is not None:
        #     yaw_angles_baseline = self._unpack_variable(yaw_angles_baseline)
        #     self.yaw_angles_baseline = yaw_angles_baseline
        # else:
        #     b = self.fi.floris.farm.yaw_angles
        #     self.yaw_angles_baseline = self._unpack_variable(b)
        #     if np.any(np.abs(b) > 0.0):
        #         print(
        #             "INFO: Baseline yaw angles were not specified and "
        #             "were derived from the floris object."
        #         )
        #         print(
        #             "INFO: The inherent yaw angles in the floris object "
        #             "are not all 0.0 degrees."
        #         )

        # self.minimum_yaw_angle = self._unpack_variable(minimum_yaw_angle)
        # self.maximum_yaw_angle = self._unpack_variable(maximum_yaw_angle)
        
        # minimum_yaw_angle_subset = copy.deepcopy(self.minimum_yaw_angle)
        # maximum_yaw_angle_subset = copy.deepcopy(self.maximum_yaw_angle)

        # if x0 is not None:
        #     self.x0 = self._unpack_variable(x0)
        # else:
        #     self.x0 = self._unpack_variable(0.0)
        #     for ti in range(self.nturbs):
        #         yaw_lb = self.minimum_yaw_angle[:, ti]
        #         yaw_ub = self.maximum_yaw_angle[:, ti]
        #         idx = (yaw_lb > 0.0) | (yaw_ub < 0.0)
        #         self.x0[idx, ti] = (yaw_lb[idx] + yaw_ub[idx]) / 2.0

        # # Define turbine weighing terms
        # if turbine_weights is None:
        #     self.turbine_weights = self._unpack_variable(1.0)
        # else:
        #     self.turbine_weights = self._unpack_variable(turbine_weights)
        
        # wd_tmp = np.array(self.fi.floris.flow_field.wind_directions)
        # ws_tmp = np.array(self.fi.floris.flow_field.wind_speeds)
        # self.fi.floris.flow_field.wind_directions = np.mean(np.reshape(self.fi.floris.flow_field.wind_directions, (self.n_wind_preview_samples, self.n_horizon)), axis=0)
        
        # try:
        #     self.fi.floris.flow_field.wind_speeds = np.mean(np.reshape(self.fi.floris.flow_field.wind_speeds, (self.n_wind_preview_samples, self.n_horizon)), axis=0)
        # except ValueError:
        #     pass
        # self._reduce_control_problem(np.mean(np.reshape(self.fi.floris.flow_field.wind_directions, (self.n_wind_preview_samples, self.n_horizon)), axis=0))
        # self.fi.floris.flow_field.wind_directions = np.array(wd_tmp)
        # self.fi.floris.flow_field.wind_speeds = np.array(ws_tmp)

        # self.turbs_to_opt
        # self._turbs_to_opt_subset
        # self._turbine_weights_subset
        # self._yaw_angles_template_subset
        # self._yaw_angles_baseline_subset


        self._turbine_power_opt_subset = np.zeros_like(self._minimum_yaw_angle_subset)
        self._cost_opt_subset = np.ones((1)) * 1e6
        self._cost_terms_opt_subset = np.ones((*self._cost_opt_subset.shape, 2)) * 1e6

    # def _reduce_control_problem(self, wind_directions=None):
    #     """
    #     This function reduces the control problem by eliminating turbines
    #     of which the yaw angles need not be optimized, either because of a
    #     user-specified set of bounds (where bounds[i][0] == bounds[i][1]),
    #     or alternatively turbines that are far downstream in the wind farm
    #     and of which the wake does not impinge other turbines, if
    #     exclude_downstream_turbines == True.
    #     """
    #     # Initialize which turbines to optimize for
    #     self.turbs_to_opt = (self.maximum_yaw_angle - self.minimum_yaw_angle >= 0.001)

    #     # Initialize subset variables as full set
    #     # self.fi_subset = self.fi.copy()
    #     # n_findex_subset = copy.deepcopy(self.fi.floris.flow_field.n_findex)
    #     # minimum_yaw_angle_subset = copy.deepcopy(self.minimum_yaw_angle)
    #     # maximum_yaw_angle_subset = copy.deepcopy(self.maximum_yaw_angle)
    #     # x0_subset = copy.deepcopy(self.x0)
    #     # turbs_to_opt_subset = copy.deepcopy(self.turbs_to_opt)
    #     # turbine_weights_subset = copy.deepcopy(self.turbine_weights)
    #     # yaw_angles_template_subset = self._unpack_variable(0.0)
    #     # yaw_angles_baseline_subset = copy.deepcopy(self.yaw_angles_baseline)

    #     # # Define which turbines to optimize for
    #     # if self.exclude_downstream_turbines:
    #     #     if wind_directions is None:
    #     #         wind_directions = self.fi.floris.flow_field.wind_directions
    #     #     for iw, wd in enumerate(wind_directions):
    #     #         # Remove turbines from turbs_to_opt that are downstream
    #     #         downstream_turbines = derive_downstream_turbines(self.fi, wd)
    #     #         downstream_turbines = np.array(downstream_turbines, dtype=int)
    #     #         self.turbs_to_opt[iw, downstream_turbines] = False
    #     #         turbs_to_opt_subset = copy.deepcopy(self.turbs_to_opt)  # Update

    #     # # Set up a template yaw angles array with default solutions. The default
    #     # # solutions are either 0.0 or the allowable yaw angle closest to 0.0 deg.
    #     # # This solution addresses both downstream turbines, minimizing their abs.
    #     # # yaw offset, and additionally fixing equality-constrained turbines to
    #     # # their appropriate yaw angle.
    #     # idx = (minimum_yaw_angle_subset > 0.0) | (maximum_yaw_angle_subset < 0.0)
    #     # if np.any(idx):
    #     #     # Find bounds closest to 0.0 deg
    #     #     combined_bounds = np.concatenate(
    #     #         (
    #     #             np.expand_dims(minimum_yaw_angle_subset, axis=3),
    #     #             np.expand_dims(maximum_yaw_angle_subset, axis=3)
    #     #         ),
    #     #         axis=3
    #     #     )
    #     #     # Overwrite all values that are not allowed to be 0.0 with bound value closest to zero
    #     #     ids_closest = np.expand_dims(np.argmin(np.abs(combined_bounds), axis=3), axis=3)
    #     #     yaw_mb = np.squeeze(np.take_along_axis(combined_bounds, ids_closest, axis=3))
    #     #     yaw_angles_template_subset[idx] = yaw_mb[idx]

    #     # # Save all subset variables to self
    #     # self._n_findex_subset = n_findex_subset
    #     # self._minimum_yaw_angle_subset = minimum_yaw_angle_subset
    #     # self._maximum_yaw_angle_subset = maximum_yaw_angle_subset
    #     # self._x0_subset = x0_subset
    #     # self._turbs_to_opt_subset = turbs_to_opt_subset
    #     # self._turbine_weights_subset = turbine_weights_subset
    #     # self._yaw_angles_template_subset = yaw_angles_template_subset
    #     # self._yaw_angles_baseline_subset = yaw_angles_baseline_subset


    def _calculate_turbine_powers(
            self, 
            yaw_angles=None, 
            wd_array=None, ws_array=None, ti_array=None, 
            turbine_weights=None,
            heterogeneous_speed_multipliers=None
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
        fi_subset = copy.deepcopy(self.fi_subset)
        if wd_array is None:
            wd_array = fi_subset.floris.flow_field.wind_directions
        if ws_array is None:
            ws_array = fi_subset.floris.flow_field.wind_speeds
        if ti_array is None:
            ti_array = fi_subset.floris.flow_field.turbulence_intensities
        if yaw_angles is None:
            yaw_angles = self._yaw_angles_baseline_subset
        if turbine_weights is None:
            turbine_weights = self._turbine_weights_subset
        if heterogeneous_speed_multipliers is not None:
            fi_subset.floris.flow_field.\
                heterogenous_inflow_config['speed_multipliers'] = heterogeneous_speed_multipliers

        # Ensure format [incompatible with _subset notation]
        yaw_angles = self._unpack_variable(yaw_angles, subset=True)

        # Calculate solutions
        # turbine_powers = np.zeros_like(yaw_angles[:, 0, :])
        fi_subset.reinitialize(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array)
        fi_subset.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = fi_subset.get_turbine_powers()

        # Multiply with turbine weighing terms
        turbine_power_weighted = np.multiply(turbine_weights, turbine_powers)
        return turbine_power_weighted

    def _calculate_baseline_turbine_powers(self):
        """
        Calculate the weighted wind farm power under the baseline turbine yaw
        angles.
        """
        if self.calc_baseline_power:
            P = self._calculate_turbine_powers(self._yaw_angles_baseline_subset)
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
        wd_array_subset = self.fi_subset.floris.flow_field.wind_directions
        ws_array_subset = self.fi_subset.floris.flow_field.wind_speeds
        ti_array_subset = self.fi_subset.floris.flow_field.turbulence_intensities
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
            if (hasattr(self.fi.floris.flow_field, 'heterogenous_inflow_config') and
                self.fi.floris.flow_field.heterogenous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fi.floris.flow_field.heterogenous_inflow_config['speed_multipliers']
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
                    self.fi_subset.floris.flow_field.n_findex,
                    self.nturbs
                )
            )

        return turbine_powers
        
    def optimize(self, current_yaw_offsets, constrain_yaw_dynamics=True, print_progress=False):
        
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """        
        self.print_progress = print_progress
        # compute baseline (no yaw) powers instead
        self._calculate_baseline_turbine_powers()

        self._yaw_lbs = copy.deepcopy(self._minimum_yaw_angle_subset)
        self._yaw_ubs = copy.deepcopy(self._maximum_yaw_angle_subset)
        # wd_tmp = np.reshape(self.fi.floris.flow_field.wind_directions, (self.n_wind_preview_samples, self.n_horizon))
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
                    current_yaw_offsets = self.fi.floris.flow_field.wind_directions - current_yaw_setpoints
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
                # _yaw_angles_opt_subset_original = np.array(self._yaw_angles_opt_subset)
                self._yaw_angles_opt_subset = np.tile(self._yaw_angles_opt_subset, (self.n_wind_preview_samples, 1))
                # assert np.sum(np.diff(evaluation_grid[0, :, 0])) == 0.0
                # evaluation_grid = evaluation_grid[:, 0:1, :]
                
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
                    
                norm_turbine_powers = turbine_powers / np.max(self.turbine_powers_baseline, axis=1)[np.newaxis, :, np.newaxis] # choose power not in wake to normalize with
                # for each value in farm_powers, get corresponding next_yaw_angles from evaluation grid
                
                # just selecting one (and only) wind speed, negative because the actual control inputs measure change in absolute yaw angle, not offset
                # control_inputs = -(evaluation_grid[:, :self.n_horizon, :] - current_yaw_offsets) * (1 / (self.yaw_rate * self.dt))
                # current_yaw_setpoints_tmp = np.reshape(current_yaw_setpoints, (self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                evaluation_grid_tmp = np.reshape(evaluation_grid, (self.Ny_passes[Nii], self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                # np.sum(np.diff(evaluation_grid_tmp[:, :, 0, :], axis=1))
                # np.sum(np.diff(current_yaw_offsets[np.newaxis, : ,:], axis=1))
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

                # print(turbine_depth, args_opt, sep='\n\n')

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
                np.sum(np.diff(yaw_angles_opt_new, axis=0))
                np.sum(np.diff(np.reshape(yaw_angles_opt_new, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=1))

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
                # yaw_angles_opt[:, np.where(ids_better)[0], turbids] = yaw_angles_opt_new[:, np.where(ids_better)[0], turbids]

                turbine_powers_opt = turbine_powers_opt_prev
                turbine_powers_opt[ids] = turbine_powers_opt_new[ids]
                # turbine_powers_opt[:, np.where(ids_better)[0], turbids] = turbine_powers_opt_new[:, np.where(ids_better)[0], turbids]

                # ids = (*np.where(ids_better), 0)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids[0]] = farm_powers_opt_new[ids[0]]
                # farm_power_opt[:, np.where(ids_better)[0]] = farm_powers_opt_new[:, np.where(ids_better)[0]]

                # Update bounds for next iteration to close proximity of optimal solution
                # evaluation_grid = np.tile(evaluation_grid, (1, self.n_wind_preview_samples, 1))
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

        # x = df_opt.iloc[0:n_horizon]["yaw_angles_opt"].to_numpy()
        # y = df_opt.iloc[n_horizon:2*n_horizon]["yaw_angles_opt"].to_numpy()

        df_opt = df_opt.iloc[0:self.n_horizon] # only want a single row for all samples
        # y = []
        # for i in range(self.n_wind_preview_samples):
        #     x = df_opt.iloc[self.n_horizon * i:self.n_horizon * (i + 1)]["yaw_angles_opt"].to_numpy()
        #     y.append(x)
        # y = np.stack(y, axis=1)
        # np.all(np.sum(np.diff(y, axis=1)) == 0.)

        df_opt["cost_states"] = cost_terms_opt[0][0]
        df_opt["cost_control_inputs"] = cost_terms_opt[0][1]
        df_opt["cost"] = cost_opt[0]
        return df_opt

class MPC(ControllerBase):

    # SLSQP, NSGA2, ParOpt, CONMIN, ALPSO
    max_iter = 25
    acc = 1e-6
    optimizers = [
        SLSQP(options={"IPRINT": -1, "MAXIT": max_iter, "ACC": acc}),
        # NSGA2(options={"xinit": 1, "PrintOut": 0, "maxGen": 50})
        # CONMIN(options={"IPRINT": 1, "ITMAX": max_iter})
        # ALPSO(options={}) #"maxOuterIter": 25})
        ]

    def __init__(self, interface, input_dict, verbose=False, **kwargs):
        
        super().__init__(interface, verbose=verbose)
        
        self.optimizer_idx = optimizer_idx
        self.dt = input_dict["controller"]["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]
        assert self.n_turbines == interface.n_turbines
        self.turbines = range(self.n_turbines)
        self.yaw_limits = input_dict["controller"]["yaw_limits"]
        self.yaw_rate = input_dict["controller"]["yaw_rate"]
        self.yaw_increment = input_dict["controller"]["yaw_increment"]
        self.alpha = input_dict["controller"]["alpha"]
        self.n_horizon = input_dict["controller"]["n_horizon"]

        wf = WindField(**WIND_FIELD_CONFIG)
        # wind_preview_generator = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
        if input_dict["controller"]["wind_preview_type"] == "stochastic":
            wind_preview_func = partial(generate_wind_preview, 
                                        wf._sample_wind_preview, 
                                        input_dict["controller"]["n_horizon"],
                                        input_dict["controller"]["n_wind_preview_samples"])
            
        elif input_dict["controller"]["wind_preview_type"] == "persistent":
            def wind_preview_func(current_freestream_measurements, time_step):
                wind_preview_data = defaultdict(list)
                for j in range(input_dict["controller"]["n_horizon"] + 1):
                    wind_preview_data[f"FreestreamWindMag_{j}"] += [kwargs["wind_mag_ts"][time_step]]
                    wind_preview_data[f"FreestreamWindDir_{j}"] += [kwargs["wind_dir_ts"][time_step]]
                return wind_preview_data
            
        elif input_dict["controller"]["wind_preview_type"] == "perfect":
            def wind_preview_func(current_freestream_measurements, time_step):
                wind_preview_data = defaultdict(list)
                for j in range(input_dict["controller"]["n_horizon"] + 1):
                    wind_preview_data[f"FreestreamWindMag_{j}"] += [kwargs["wind_mag_ts"][time_step + j]]
                    wind_preview_data[f"FreestreamWindDir_{j}"] += [kwargs["wind_dir_ts"][time_step + j]]
                return wind_preview_data
            
        self.wind_preview_func = wind_preview_func

        self.wind_preview_type = input_dict["controller"]["wind_preview_type"]
        if self.wind_preview_type == "stochastic":
            self.n_wind_preview_samples = input_dict["controller"]["n_wind_preview_samples"]
        else:
            self.n_wind_preview_samples = 1
        
        self.warm_start = input_dict["controller"]["warm_start"]

        if self.warm_start == "lut":
            fi_lut = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
                        .load_floris(config_path=input_dict["controller"]["floris_input_file"])
        
            ctrl_lut = LookupBasedWakeSteeringController(fi_lut, input_dict=input_dict, 
                                                    lut_path=os.path.join(os.path.dirname(whoc.__file__), 
                                                                        f"../examples/mpc_wake_steering_florisstandin/lut_{fi_lut.n_turbines}.csv"), 
                                                    generate_lut=False)
            def warm_start_func(measurements: dict):
                # feed interface with new disturbances
                fi_lut.step(disturbances={"wind_speeds": [measurements["wind_speeds"][0]],
                                        "wind_directions": [measurements["wind_directions"][0]]},
                                        seed=1)
                
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
        self.solver = input_dict["controller"]["solver"]
        self.n_solve_turbines = self.n_turbines if self.solver != "sequential_pyopt" else 1
        self.n_solve_states = self.n_solve_turbines * self.n_horizon
        self.n_solve_control_inputs = self.n_solve_turbines * self.n_horizon

        self.dyn_state_jac, self.state_jac = self.con_sens_rules()
        
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
        
        self.opt_sol = {"states": [], "control_inputs": []}
        
        if input_dict["controller"]["control_input_domain"].lower() in ['discrete', 'continuous']:
            self.control_input_domain = input_dict["controller"]["control_input_domain"].lower()
        else:
            raise TypeError("control_input_domain must be have value of 'discrete' or 'continuous'")
        
        self.wind_ti = 0.08

        self.fi = ControlledFlorisInterface(max_workers=interface.max_workers,
                                            yaw_limits=self.yaw_limits, dt=self.dt, yaw_rate=self.yaw_rate) \
            .load_floris(config_path=input_dict["controller"]["floris_input_file"])
        
        if self.solver == "floris":
            # self.fi_opt = FlorisInterfaceDev(input_dict["controller"]["floris_input_file"]) #.replace("floris", "floris_dev"))
            if self.warm_start == "lut":
                print("Can't warm-start FLORIS SR solver, setting self.warm_start to none")
                self.warm_start = "none"
        elif self.solver == "pyopt" or self.solver == "sequential_pyopt":
            self.setup_pyopt_solver()
        elif self.solver == "zsgd":
            pass
        
    
    def con_sens_rules(self):
        dyn_state_con_sens = {"states": [], "control_inputs": []}
        state_con_sens = {"states": [], "control_inputs": []}

        for j in range(self.n_horizon):
            for i in range(self.n_solve_turbines):
                current_idx = (self.n_solve_turbines * j) + i
                
                # scaled by yaw limit
                dyn_state_con_sens["control_inputs"].append([
                    -(self.dt * (self.yaw_rate / self.yaw_norm_const)) if idx == current_idx else 0
                    for idx in range(self.n_solve_control_inputs)
                ])
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    
                    dyn_state_con_sens["states"].append([
                        1 if idx == current_idx else 0
                        for idx in range(self.n_solve_states)
                    ])
                    
                    
                else:
                    prev_idx = (self.n_solve_turbines * (j - 1)) + i
                    
                    dyn_state_con_sens["states"].append([
                        1 if idx == current_idx else (-1 if idx == prev_idx else 0)
                        for idx in range(self.n_solve_states)
                    ])
                
                state_con_sens["states"].append([
                        -1 if idx == current_idx else 0
                        for idx in range(self.n_solve_states)
                    ])
                state_con_sens["control_inputs"].append([0] * self.n_solve_control_inputs)
         
        return dyn_state_con_sens, state_con_sens
    
    def dyn_state_rules(self, opt_var_dict):

        # define constraints
        dyn_state_cons = []
        for j in range(self.n_horizon):
            for i in range(self.n_solve_turbines):
                current_idx = (self.n_solve_turbines * j) + i
                delta_yaw = self.dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[i] + delta_yaw)]
                    
                    
                else:
                    prev_idx = (self.n_solve_turbines * (j - 1)) + i
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [
                        opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]
                    
        return dyn_state_cons
    
    def state_rules(self, opt_var_dict, disturbance_dict):
        # define constraints
        state_cons = []
        
        for j in range(self.n_horizon):
            for i in range(self.n_solve_turbines):
                current_idx = (self.n_solve_turbines * j) + i
                
                state_cons = state_cons + [(disturbance_dict["wind_direction"][j] / self.yaw_norm_const) - opt_var_dict["states"][current_idx]]
                    
        return state_cons
        
    def opt_rules(self, opt_var_dict, compute_constraints=True, compute_derivatives=True):
        
        funcs = {}
        if self.solver == "sequential_pyopt":
            control_inputs = np.array(self.opt_sol["control_inputs"])
            control_inputs[self.solve_turbine_id::self.n_turbines] = opt_var_dict["control_inputs"]

            yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                    + (self.yaw_rate * self.dt * np.sum(control_inputs[i:(self.n_turbines * j) + i:self.n_turbines])))
                                    for i in range(self.n_turbines)] for j in range(self.n_horizon)])
        else:
            yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                    + (self.yaw_rate * self.dt * np.sum(opt_var_dict["control_inputs"][i:(self.n_turbines * j) + i:self.n_turbines])))
                                    for i in range(self.n_turbines)] for j in range(self.n_horizon)])

        assert np.all(np.isclose(self.wind_preview_samples[f"FreestreamWindDir_{0}"], self.measurements_dict["wind_directions"][0]))
        
        # plot_distribution_samples(pd.DataFrame(wind_preview_samples), self.n_horizon)
        # derivative of turbine power output with respect to yaw angles
        
        # TODO add dynamically varying constraints to overleaf
        if compute_constraints:
            funcs["state_cons"] = self.state_rules(opt_var_dict, 
                                                {
                                                    "wind_direction": [np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j}"]) for j in range(self.n_horizon)]
                                                    })
        
        # self.norm_turbine_powers = np.zeros((self.n_wind_preview_samples, self.n_horizon, self.n_turbines))

        # derivative of turbine power output with respect to yaw angle changes
        # norm_turbine_powers_ctrl_inputs_drvt = np.zeros((self.n_wind_preview_samples, self.n_horizon * self.n_turbines))

        # TODO use the freestream wind dire to calculate offsets or local wind dir?
        current_yaw_offsets = np.vstack([(self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - yaw_setpoints[j, :]) for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])

        # send yaw angles 
        self.fi.env.calculate_wake(current_yaw_offsets)
        yawed_turbine_powers = self.fi.env.get_turbine_powers()
        # yawed_turbine_powers = np.reshape(yawed_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))

        # greedily yaw directly into wind for normalization constant
        self.fi.env.calculate_wake(np.zeros((self.n_wind_preview_samples * self.n_horizon, self.n_turbines)))
        greedy_yaw_turbine_powers = self.fi.env.get_turbine_powers()
        # greedy_yaw_turbine_powers = np.reshape(greedy_yaw_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
        greedy_yaw_turbine_powers = np.max(greedy_yaw_turbine_powers, axis=1)[:, np.newaxis] # choose unwaked turbine for normalization constant

        # normalize power by no yaw output
        self.norm_turbine_powers = np.divide(yawed_turbine_powers, greedy_yaw_turbine_powers,
                                                where=greedy_yaw_turbine_powers!=0,
                                                out=np.zeros_like(yawed_turbine_powers))
        self.norm_turbine_powers = np.reshape(self.norm_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines))

        # compute power based on sampling from wind preview
        funcs["cost_states"] = sum([-0.5*np.mean((self.norm_turbine_powers[:, j, i])**2) * self.Q
                            for j in range(self.n_horizon) for i in range(self.n_turbines)])
        # TODO should this be averaged too considering influence of wind directions...
        funcs["cost_control_inputs"] = sum(0.5*(opt_var_dict["control_inputs"][(self.n_solve_turbines * j) + i])**2 * self.R
                            for j in range(self.n_horizon) for i in range(self.n_solve_turbines))
            
        if compute_derivatives:
            if self.wind_preview_type == "stochastic":
                u = np.random.normal(loc=0.0, scale=1.0, size=(self.n_wind_preview_samples * self.n_horizon, self.n_solve_turbines))
                # self.norm_turbine_powers_states_drvt = np.zeros((self.n_wind_preview_samples, self.n_horizon, self.n_turbines, self.n_solve_turbines))
                # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
                
                if self.solver == "sequential_pyopt":
                    mask = np.zeros((self.n_turbines,))
                    mask[self.solve_turbine_id] = 1
                    plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * u * mask
                else:
                    plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * u

                self.fi.env.calculate_wake(plus_yaw_offsets)
                plus_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                norm_turbine_power_diff = np.divide((plus_perturbed_yawed_turbine_powers - yawed_turbine_powers), greedy_yaw_turbine_powers,
                                                    where=greedy_yaw_turbine_powers!=0,
                                                    out=np.zeros_like(yawed_turbine_powers))

                # should compute derivative of each power of each turbine wrt state (yaw angle) of each turbine
                # self.norm_turbine_powers_states_drvt = (norm_turbine_power_diff / self.nu).T @ u
                self.norm_turbine_powers_states_drvt = np.einsum("ia, ib->iab", norm_turbine_power_diff / self.nu, u)
                self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines, self.n_solve_turbines))

            
            elif self.wind_preview_type == "persistent" or self.wind_preview_type == "perfect":

                self.norm_turbine_powers_states_drvt = np.zeros((self.n_wind_preview_samples * self.n_horizon, self.n_turbines, self.n_solve_turbines))
                # perturb each state (each yaw angle) by +/= nu to estimate derivative of all turbines power output for a variation in each turbines yaw offset
                # if any yaw offset are out of the [-90, 90] range, then the power output of all turbines will be nan. clip to avoid this

                # TODO parallelize across solve turbine perturbations
                # u = np.tile(np.eye(self.n_solve_turbines), (self.n_solve_turbines * self.n_horizon, 1))

                for i in range(self.n_solve_turbines):
                    mask = np.zeros((self.n_turbines,))
                    if self.solver == "sequential_pyopt":
                        mask[self.solve_turbine_id] = 1
                    else:
                        mask[i] = 1
                    
                    # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
                    plus_yaw_offsets = current_yaw_offsets - mask * self.nu * self.yaw_norm_const
                    
                    self.fi.env.calculate_wake(plus_yaw_offsets)
                    plus_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                    # we add negative since current_yaw_offsets = wind dir - yaw setpoints
                    neg_yaw_offsets = current_yaw_offsets + mask * self.nu * self.yaw_norm_const

                    self.fi.env.calculate_wake(neg_yaw_offsets)
                    neg_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                    norm_turbine_power_diff = np.divide((plus_perturbed_yawed_turbine_powers - neg_perturbed_yawed_turbine_powers), greedy_yaw_turbine_powers,
                                                    where=greedy_yaw_turbine_powers!=0,
                                                    out=np.zeros_like(plus_perturbed_yawed_turbine_powers))

                    # should compute derivative of each power of each turbine wrt state (yaw angle) of each turbine
                    # self.norm_turbine_powers_states_drvt = (norm_turbine_power_diff / self.nu).T @ u
                    self.norm_turbine_powers_states_drvt[:, :, i] = norm_turbine_power_diff / (2 * self.nu)
                    # np.einsum("ia, ib->iab", norm_turbine_power_diff / self.nu, u)
                self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, self.n_turbines, self.n_solve_turbines))

        funcs["cost"] = funcs["cost_states"] + funcs["cost_control_inputs"]
        
        if compute_constraints:
            funcs["dyn_state_cons"] = self.dyn_state_rules(opt_var_dict)

        if self.solver == "sequential_pyopt":
            self.opt_cost_terms[0] += funcs["cost_states"]
            self.opt_cost_terms[1] += funcs["cost_control_inputs"]
        else:
            self.opt_cost_terms = [funcs["cost_states"], funcs["cost_control_inputs"]]
        
        fail = False
        
        return funcs, fail
    
    def sens_rules(self, opt_var_dict, obj_con_dict):
        sens = {"cost": {"states": [], "control_inputs": []},
                "dyn_state_cons": {"states": [], "control_inputs": []},
                "state_cons": {"states": [], "control_inputs": []}}
        
        
        
        # if self.wind_preview_type == "stochastic":
        for j in range(self.n_horizon):
            for i in range(self.n_solve_turbines):
                current_idx = (self.n_solve_turbines * j) + i
                sens["cost"]["control_inputs"].append(
                    opt_var_dict["control_inputs"][current_idx] * self.R
                )
                # compute power derivative based on sampling from wind preview
                # using derivative: power of each turbine wrt each turbine's yaw setpoint, summing over terms for each turbine
                sens["cost"]["states"].append(
                            np.mean(np.sum(-(self.norm_turbine_powers[:, j, :]) * self.Q * self.norm_turbine_powers_states_drvt[:, j, :, i], axis=1))
                        )
        
        sens["dyn_state_cons"] = self.dyn_state_jac # self.con_sens_rules()
        sens["state_cons"] = self.state_jac
        
        self.sens = sens
        return sens

    def setup_pyopt_solver(self):
        
        # initialize optimization object
        self.pyopt_prob = Optimization("Wake Steering MPC", self.opt_rules, sens=self.sens_rules)
        
        # add design variables
        self.pyopt_prob.addVarGroup("states", self.n_solve_states,
                                    varType="c",  # continuous variables
                                    lower=[0] * self.n_solve_states,
                                    upper=[1] * self.n_solve_states,
                                    value=[0] * self.n_solve_states)
                                    # scale=(1 / self.yaw_norm_const))
        
        if self.control_input_domain == 'continuous':
            self.pyopt_prob.addVarGroup("control_inputs", self.n_solve_control_inputs,
                                        varType="c",
                                        lower=[-1] * self.n_solve_control_inputs,
                                        upper=[1] * self.n_solve_control_inputs,
                                        value=[0] * self.n_solve_control_inputs)
        else:
            self.pyopt_prob.addVarGroup("control_inputs", self.n_solve_control_inputs,
                                        varType="i",
                                        lower=[-1] * self.n_solve_control_inputs,
                                        upper=[1] * self.n_solve_control_inputs,
                                        value=[0] * self.n_solve_control_inputs)
        
        # # add dynamic state equation constraints
        # jac = self.con_sens_rules()
        self.pyopt_prob.addConGroup("dyn_state_cons", self.n_solve_states, lower=0.0, upper=0.0)
                        #   linear=True, wrt=["states", "control_inputs"], # NOTE supplying fixed jac won't work because value of initial_state changes
                        #   jac=jac)
        self.pyopt_prob.addConGroup("state_cons", self.n_solve_states, lower=self.yaw_limits[0] / self.yaw_norm_const, upper=self.yaw_limits[1] / self.yaw_norm_const)
        
        # add objective function
        self.pyopt_prob.addObj("cost")
        
        # display optimization problem
        # if self.solver != "sequential_pyopt":
            # print(self.pyopt_prob)
    
    def compute_controls(self):
        """
        solve OCP to minimize objective over future horizon
        """
        
        # get current time-step
        current_time_step = int(self.measurements_dict["time"] // self.dt)
        
        if current_time_step > 0.:
            # update initial state self.mi_model.initial_state
            self.initial_state = self.measurements_dict["yaw_angles"] / self.yaw_norm_const # scaled by yaw limits
        
        if self.current_freestream_measurements is None:
            self.current_freestream_measurements = [
                    self.measurements_dict["wind_speeds"][0]
                    * np.cos((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.)),
                    self.measurements_dict["wind_speeds"][0]
                    * np.sin((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.))
                ]
            
        self.wind_preview_samples = self.wind_preview_func(self.current_freestream_measurements, current_time_step)

        # update floris model TODO test with floris
        if self.solver == "pyopt" or self.solver == "sequential_pyopt":
            self.fi.env.reinitialize(
                wind_directions=[self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)],
                wind_speeds=[self.wind_preview_samples[f"FreestreamWindMag_{j}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)]
            )
        elif self.solver == "floris":
            self.fi.env.reinitialize(
                wind_directions=[self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon + 1)],
                wind_speeds=[self.wind_preview_samples[f"FreestreamWindMag_{j}"][m] for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon + 1)]
            )            
       
        if self.solver == "pyopt":
            yaw_star = self.pyopt_solve()
        elif self.solver == "sequential_pyopt":
            yaw_star = self.SR_pyopt_solve()
        elif self.solver == "floris":
            yaw_star = self.floris_solve()
        elif self.solver == "zsgd":
            yaw_star = self.zsgd_solve()
        
        # check constraints
        # assert np.isclose(sum(self.opt_sol["states"][:self.n_turbines] - (self.initial_state + self.opt_sol["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_norm_const) * self.dt)), 0, atol=1e-2)
        print(sum(self.opt_sol["states"][:self.n_solve_turbines] - (self.initial_state + self.opt_sol["control_inputs"][:self.n_solve_turbines] * (self.yaw_rate / self.yaw_norm_const) * self.dt)))
        # assert np.isclose(sum(self.opt_sol["states"][self.n_turbines:] - (self.opt_sol["states"][:-self.n_turbines] + self.opt_sol["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_norm_const) * self.dt)), 0)
        print(sum(
            self.opt_sol["states"][self.n_solve_turbines:] - (self.opt_sol["states"][:-self.n_solve_turbines] + self.opt_sol["control_inputs"][self.n_solve_turbines:] * (self.yaw_rate / self.yaw_norm_const) * self.dt)
            ))
        
        # assert np.isclose(sum((np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j}"]) / self.yaw_norm_const) - self.opt_sol["states"][(j * self.n_turbines) + i] for j in range(self.n_horizon) for i in range(self.n_turbines)), 0)
        state_cons = [(np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j}"]) / self.yaw_norm_const) - self.opt_sol["states"][(j * self.n_solve_turbines) + i] for j in range(self.n_horizon) for i in range(self.n_solve_turbines)]
        
        print(all([(c <= (self.yaw_limits[1] / self.yaw_norm_const) + 1e-12) and (c >= (self.yaw_limits[0] / self.yaw_norm_const) - 1e-12) for c in state_cons]))
        # self.norm_turbine_powers_states_drvt
        # [c * self.yaw_norm_const for c in state_cons] # offsets
        self.controls_dict = {"yaw_angles": yaw_star}
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

    def floris_solve(self):
        
        # warm-up with previous solution
        self.warm_start_opt_vars()
        if not self.opt_sol:
            self.opt_sol = self.init_sol
        
        opt_yaw_setpoints = np.zeros((self.n_horizon, self.n_turbines))
        opt_cost = np.zeros((self.n_horizon))
        opt_cost_terms = np.zeros((self.n_horizon, 2))

        run_parallel = False
        unconstrained_solve = True

        if run_parallel: # parallel version, needs v4...
            
            # optimize yaw angles
            yaw_offset_opt = YawOptimizationSRRHC(self.fi, 
                            self.yaw_rate, self.dt, self.alpha,
                            n_wind_preview_samples=self.n_wind_preview_samples,
                            n_horizon=self.n_horizon,
                            minimum_yaw_angle=self.yaw_limits[0],
                            maximum_yaw_angle=self.yaw_limits[1],
                        #  yaw_angles_baseline=np.zeros((1, 1, self.n_turbines)),
                        yaw_angles_baseline=np.zeros((self.n_turbines,)),
                        #  normalize_control_variables=True,
                            Ny_passes=[8, 4],
                        # x0=(self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] 
                        #     - self.init_sol["states"][j * self.n_turbines:(j + 1) * self.n_turbines] * self.yaw_norm_const)[np.newaxis, np.newaxis, :],
                            verify_convergence=False)#, exploit_layout_symmetry=False)
            
            yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                    + (self.yaw_rate * self.dt * np.sum(self.opt_sol["control_inputs"][i:(self.n_turbines * j) + i:self.n_turbines])))
                                    for i in range(self.n_turbines)] for j in range(self.n_horizon)])
            current_yaw_offsets = np.vstack([(self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - yaw_setpoints[j, :]) for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])

            opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets=current_yaw_offsets, constrain_yaw_dynamics=True, print_progress=True)
            opt_yaw_setpoints = np.array([self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - opt_yaw_offsets_df["yaw_angles_opt"].iloc[(self.n_horizon * m) + j] for m in range(self.n_wind_preview_samples) for j in range(1, self.n_horizon)])
            opt_cost = opt_yaw_offsets_df["cost"]
            opt_cost_terms[:, 0] = opt_yaw_offsets_df["cost_states"]
            opt_cost_terms[:, 1] = opt_yaw_offsets_df["cost_control_inputs"]

        elif unconstrained_solve:
            
            yaw_setpoints = np.array([[((self.initial_state[i] * self.yaw_norm_const) 
                                    + (self.yaw_rate * self.dt * np.sum(self.opt_sol["control_inputs"][i:(self.n_turbines * j) + i:self.n_turbines])))
                                    for i in range(self.n_turbines)] for j in range(self.n_horizon)])
            
            # current_yaw_offsets = (np.mean(self.wind_preview_samples[f"FreestreamWindDir_{0}"]) - yaw_setpoints[0, :])
            current_yaw_offsets = np.array(self.wind_preview_samples[f"FreestreamWindDir_{0}"])[:, np.newaxis] - yaw_setpoints[0, :]
            # current_yaw_setpoints = yaw_setpoints[0, :]

            # solve for each horizon step independently
            # wd_tmp = (359.0 - 180.0) * np.random.random_sample(self.n_wind_preview_samples * (self.n_horizon)) + 180.
            # # wd_tmp[0::self.n_horizon] = wd_tmp[0]
            # ws_tmp = (12.0 - 8.0) * np.random.random_sample(self.n_wind_preview_samples * (self.n_horizon)) + 8.
            # # ws_tmp[0::self.n_horizon] = ws_tmp[0]
            # self.fi_opt.reinitialize(
            #     wind_directions=wd_tmp,
            #     wind_speeds=ws_tmp
            # )
        
            # optimize yaw angles
            yaw_offset_opt = YawOptimizationSRRHC(self.fi, 
                            self.yaw_rate, self.dt, self.alpha,
                            n_wind_preview_samples=self.n_wind_preview_samples,
                            n_horizon=self.n_horizon,
                            minimum_yaw_angle=self.yaw_limits[0],
                            maximum_yaw_angle=self.yaw_limits[1],
                            yaw_angles_baseline=np.zeros((self.n_turbines,)),
                            Ny_passes=[6, 4],
                            verify_convergence=False)

            opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets=current_yaw_offsets, 
                                                         constrain_yaw_dynamics=False, print_progress=True)
            opt_yaw_setpoints = np.vstack([np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j}"]) - opt_yaw_offsets_df["yaw_angles_opt"].iloc[j] for j in range(self.n_horizon)])
            opt_cost = opt_yaw_offsets_df["cost"].to_numpy()
            opt_cost_terms[:, 0] = opt_yaw_offsets_df["cost_states"].to_numpy()
            opt_cost_terms[:, 1] = opt_yaw_offsets_df["cost_control_inputs"].to_numpy()

            # check if solution adheres to dynamic state equaion
            # ensure that the rate of change is not greater than yaw_rate
            for j in range(self.n_horizon):
                # gamma(k+1) in [gamma(k) - gamma_dot delta_t, gamma(k) + gamma_dot delta_t]
                init_gamma = self.initial_state * self.yaw_norm_const if j == 0 else opt_yaw_setpoints[j - 1, :]
                # delta_gamma = opt_yaw_setpoints[j, :] - init_gamma
                # if np.any(np.abs(delta_gamma) > self.yaw_rate * self.dt):
                #     pass
            
                opt_yaw_setpoints[j, :] = np.clip(opt_yaw_setpoints[j, :], init_gamma - self.yaw_rate * self.dt, init_gamma + self.yaw_rate * self.dt)
                
            
            opt_cost = np.sum(opt_cost)
            opt_cost_terms = np.sum(opt_cost_terms, axis=0)
        else:
            for j in range(self.n_horizon):
                for m in range(self.n_wind_preview_samples):
                    # solve at each time-step, checking that new yaw angles are feasible given last yaw angles, then average solutions over all samples
                    # optimize yaw angles
                    yaw_offset_opt = YawOptimizationSRRHC(self.fi, 
                                    self.yaw_rate, self.dt, self.alpha,
                                    n_wind_preview_samples=self.n_wind_preview_samples,
                                    n_horizon=self.n_horizon,
                                    minimum_yaw_angle=self.yaw_limits[0],
                                    maximum_yaw_angle=self.yaw_limits[1],
                                #  yaw_angles_baseline=np.zeros((1, 1, self.n_turbines)),
                                yaw_angles_baseline=np.zeros((self.n_turbines,)),
                                #  normalize_control_variables=True,
                                    Ny_passes=[8, 4],
                                x0=(self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] 
                                    - self.init_sol["states"][j * self.n_turbines:(j + 1) * self.n_turbines] * self.yaw_norm_const)[np.newaxis, np.newaxis, :],
                                    verify_convergence=False)#, exploit_layout_symmetry=False)
                    if j == 0:
                        # self.initial_state = np.array([self.yaw_IC / self.yaw_norm_const] * self.n_turbines)
                        # current_yaw_offsets = self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - (self.initial_state * self.yaw_norm_const)
                        current_yaw_offsets = np.zeros((self.n_turbines, ))
                    else:
                        current_yaw_offsets = self.wind_preview_samples[f"FreestreamWindDir_{j-1}"][m] - opt_yaw_setpoints[m, j - 1, :]
                        
                    opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets, print_progress=True)
                    opt_yaw_setpoints[m, j, :] = self.wind_preview_samples[f"FreestreamWindDir_{j}"][m] - opt_yaw_offsets_df["yaw_angles_opt"].iloc[0]
                    opt_cost[m, j] = opt_yaw_offsets_df["cost"].iloc[0]
                    opt_cost_terms[m, j, 0] = opt_yaw_offsets_df["cost_states"].iloc[0]
                    opt_cost_terms[m, j, 1] = opt_yaw_offsets_df["cost_control_inputs"].iloc[0]

            # TODO change this     
            opt_yaw_setpoints = np.mean(opt_yaw_setpoints, axis=0)
            opt_cost = np.sum(np.mean(opt_cost, axis=0))
            opt_cost_terms = np.sum(np.mean(opt_cost_terms, axis=0), axis=0)
                    
        self.opt_sol = {
            "states": np.array([opt_yaw_setpoints[j, i] / self.yaw_norm_const for j in range(self.n_horizon) for i in range(self.n_turbines)]), 
            "control_inputs": np.array([(opt_yaw_setpoints[j, i] - (opt_yaw_setpoints[j-1, i] if j > 0 else self.initial_state[i] * self.yaw_norm_const)) * (1 / (self.yaw_rate * self.dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)])
            }
        self.opt_code = {"text": None}
        self.opt_cost = opt_cost
        # funcs, _ = self.opt_rules(self.opt_sol)
        self.opt_cost_terms = opt_cost_terms

        return np.rint(opt_yaw_setpoints[0, :] / self.yaw_increment) * self.yaw_increment
            # reinitialize the floris object with the predicted wind magnitude and direction at this time-step in the horizon

    def warm_start_opt_vars(self):
        self.init_sol = {"states": [], "control_inputs": []}
        current_yaw_setpoints = self.measurements_dict["yaw_angles"]
        
        if self.warm_start == "previous":
            current_time_step = int(self.measurements_dict["time"] // self.dt)
            if current_time_step > 0:
                self.init_sol = {
                    "states": np.clip(np.concatenate([
                     self.opt_sol["states"][self.n_turbines:], self.opt_sol["states"][-self.n_turbines:]
                     ]), -1, 1),
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
            
            self.init_sol = {
                    "states": [],
                    "control_inputs": []
            }
            for j in range(self.n_horizon):
                # compute yaw angle setpoints from warm_start_func
                
                # if self.wind_preview_type == "stochastic":
                next_yaw_setpoints = self.warm_start_func({"wind_speeds": 
                                    [np.mean(self.wind_preview_samples[f"FreestreamWindMag_{j + 1}"])], 
                                    "wind_directions": [np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"])]})
                
                current_control_inputs = (next_yaw_setpoints - current_yaw_setpoints) * (1 / (self.yaw_rate * self.dt))
                
                self.init_sol["states"] = self.init_sol["states"] + list(next_yaw_setpoints / self.yaw_norm_const)
                self.init_sol["control_inputs"] = self.init_sol["control_inputs"] + list(current_control_inputs)
                
                current_yaw_setpoints = next_yaw_setpoints

            self.init_sol["states"] = np.array(self.init_sol["states"])
            self.init_sol["control_inputs"] = np.clip(self.init_sol["control_inputs"], -1, 1)
        elif self.warm_start == "none":
            self.init_sol["states"] = np.array([self.yaw_IC / self.yaw_norm_const] * self.n_turbines)
            self.init_sol["control_inputs"] = np.zeros((self.n_turbines,))
        elif self.warm_start == "random":
            self.init_sol["states"] = np.random.uniform(low=-1.0, high=1.0, size=(self.n_turbines * self.n_horizon,))
            self.init_sol["control_inputs"] = np.zeros((self.n_turbines * self.n_horizon,))
        
        if self.basin_hop:
            def basin_hop_obj(opt_var_arr):
                funcs, _ = self.opt_rules({"states": opt_var_arr[:self.n_horizon * self.n_turbines],
                                "control_inputs": opt_var_arr[self.n_horizon * self.n_turbines:]},
                                compute_derivatives=False, compute_constraints=False)
                return funcs["cost"]
            
            basin_hop_init_sol = basinhopping(basin_hop_obj, np.concatenate([self.init_sol["states"], self.init_sol["control_inputs"]]), niter=20, stepsize=0.2, disp=True)
            self.init_sol["states"] = basin_hop_init_sol.x[:self.n_horizon * self.n_turbines]
            self.init_sol["control_inputs"] = basin_hop_init_sol.x[self.n_horizon * self.n_turbines:]
        # self.init_sol = [s.value for s in self.pyopt_prob.variables["states"]] + [c.value for c in self.pyopt_prob.variables["control_inputs"]]
        
    def SR_pyopt_solve(self):
        # set self.opt_sol to initial solution
        self.warm_start_opt_vars()
        self.opt_sol = dict(self.init_sol)
        self.opt_cost = 0
        self.opt_cost_terms = [0, 0]
        self.opt_code = []

        # rotate turbine coordinates based on most recent wind direction measurement
        # order turbines based on order of wind incidence
        layout_x = self.fi.env.layout_x
        layout_y = self.fi.env.layout_y
        # turbines_ordered_array = []
        wd = self.wind_preview_samples["FreestreamWindDir_0"][0]
        layout_x_rot = (
            np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
            - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
        )
        turbines_ordered = np.argsort(layout_x_rot)
        # turbines_ordered_array.append(turbines_ordered)
        # self.turbines_ordered_array = np.vstack(turbines_ordered_array)

        # for each turbine in sorted array
        for i in range(self.n_turbines):
            self.solve_turbine_id = turbines_ordered[i]
            
            # setup pyopt problem to consider 2 optimization variables for this turbine and set all others as fixed parameters

            for j in range(self.n_horizon):
                
                self.pyopt_prob.variables["states"][j].value \
                    = self.init_sol["states"][(j * self.n_turbines) + self.solve_turbine_id]
                
                self.pyopt_prob.variables["control_inputs"][j].value \
                    = self.init_sol["control_inputs"][(j * self.n_turbines) + self.solve_turbine_id]
            
            # solve problem based on self.opt_sol
            sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens=self.sens_rules) #, sensMode='pgc')
            # sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens="FD")

            # update self.opt_sol with most recent solutions for all turbines
            self.opt_sol["states"][self.solve_turbine_id::self.n_turbines] = np.array(sol.xStar["states"])
            self.opt_sol["control_inputs"][self.solve_turbine_id::self.n_turbines] = np.array(sol.xStar["control_inputs"])
            self.opt_code.append(sol.optInform)
            self.opt_cost += sol.fStar
            
        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment

    def pyopt_solve(self):
        
        # warm start Vars by reinitializing the solution from last time-step self.mi_model.states
        self.warm_start_opt_vars()

        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (j * self.n_turbines) + i
                # next_idx = ((j + 1) * self.n_turbines) + i
                self.pyopt_prob.variables["states"][current_idx].value \
                    = self.init_sol["states"][current_idx]
                
                self.pyopt_prob.variables["control_inputs"][current_idx].value \
                    = self.init_sol["control_inputs"][current_idx]

        sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens=self.sens_rules) #, sensMode='pgc')
        # sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens="FD")
        self.pyopt_sol_obj = sol
        self.opt_sol = dict(sol.xStar)
        self.opt_code = sol.optInform
        self.opt_cost = sol.fStar
        assert sum(self.opt_cost_terms) == self.opt_cost

        self.wind_preview_samples["FreestreamWindDir_0"][0] - (self.initial_state * self.yaw_norm_const)
        self.norm_turbine_powers_states_drvt[0, :, :] # should be p
        # solution is scaled by yaw limit
        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment
        # return (self.initial_state * self.yaw_norm_const) \
        # 	+ self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]  # solution is scaled by yaw limit
    
if __name__ == '__main__':
    # import
    from hercules.utilities import load_yaml
    
    # options
    max_workers = 16
    # input_dict = load_yaml(sys.argv[1])
    input_dict = load_yaml("../../examples/hercules_input_001.yaml")
    
    ## Simulate wind farm with interface and controller
    
    # instantiate wind field if files don't already exist
    # TODO replace wind_field_config with yaml
    regenerate_wind_field = True
    wind_field_filenames = glob(f"{DATA_SAVE_DIR}/case_*.csv")
    if not len(wind_field_filenames) or regenerate_wind_field:
        generate_multi_wind_ts(WIND_FIELD_CONFIG, N_CASES)
        wind_field_filenames = [f"case_{i}.csv" for i in range(N_CASES)]
        regenerate_wind_field = True
    
    WIND_TYPE = "stochastic"
    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(DATA_SAVE_DIR):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))

            if WIND_TYPE == "step":
                # n_rows = len(wind_field_data[-1].index)
                wind_field_data[-1].loc[:15, f"FreestreamWindMag"] = 8.0
                wind_field_data[-1].loc[15:, f"FreestreamWindMag"] = 11.0
                wind_field_data[-1].loc[:45, f"FreestreamWindDir"] = 260.0
                wind_field_data[-1].loc[45:, f"FreestreamWindDir"] = 270.0
    
    # true wind disturbance time-series
    case_idx = 0
    time_ts = wind_field_data[case_idx]["Time"].to_numpy()
    wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
    wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
    
    # instantiate controller and execute simulation
    results = simulate_controller(MPC, input_dict, wind_mag_ts, wind_dir_ts, episode_max_time=EPISODE_MAX_TIME)
    
    # yaw_angles_ts, yaw_angles_change_ts, turbine_powers_ts, convergence_time_ts, opt_codes_ts, opt_cost_terms_ts 
    results_df = results
    # filt_wind_dir_ts = ctrl._first_ord_filter(wind_dir_ts, ctrl.wd_lpf_alpha)
    # filt_wind_speed_ts = ctrl._first_ord_filter(wind_mag_ts, ctrl.ws_lpf_alpha)
    
    # save results and configurataion in a dataframe of results
    # make folder for results
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_persistent_timevarying")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_stochastic_timevarying")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_perfect_timevarying")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_persistent")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_stochastic")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_perfect")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "floris_persistent")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "floris_stochastic_unconstrained")
    results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "floris_perfect")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", "pyopt_sequential_perfect")
    # results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    
    os.makedirs(results_dir)
    with open(os.path.join(results_dir, "input_config.yaml"), 'w') as fp:
        yaml.dump(input_dict, fp)
    

    results_df.to_csv(os.path.join(results_dir, "time_series_results.csv"))

    fig_wind, ax_wind = plot_wind_field_ts(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], 
                                        wind_dir_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])],
                                        wind_mag_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])],
                                        results_dir)

    opt_var_fig, opt_var_ax = plot_opt_var_ts(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], 
                                              np.hstack([yaw_angles_ts, yaw_angles_change_ts]), 
                                              wind_dir_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1],
                                              input_dict["controller"]["yaw_limits"],
                                              results_dir)
    
    opt_cost_fig, opt_cost_ax = plot_opt_cost_ts(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], 
                                              opt_cost_terms_ts, results_dir)