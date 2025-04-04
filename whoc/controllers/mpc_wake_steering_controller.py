from time import perf_counter
import copy
import numpy as np
from pyoptsparse import Optimization, SLSQP
from mpi4py import MPI
# from pyoptsparse.pyOpt_history import History
from scipy.optimize import linprog, basinhopping
from scipy.signal import lfilter
from scipy.stats import norm
from itertools import product
import pandas as pd

# from memory_profiler import profile
# from multiprocessing import Process, RawArray

from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.wind_field.WindField import WindField, generate_wind_preview

from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

optimizer_idx = 0

def run_floris_proc(floris_env, turbine_powers_arr):
    floris_env.run()
    turbine_powers_arr = np.frombuffer(turbine_powers_arr, dtype=np.double, count=len(turbine_powers_arr))
    turbine_powers_arr[:] = floris_env.get_turbine_powers().flatten()

class YawOptimizationSRRHC(YawOptimizationSR):
    def __init__(
        self,
        fmodel,
        yaw_rate,
        controller_dt,
        alpha,
        n_wind_preview_samples,
        wind_preview_type,
        n_horizon,
        rated_turbine_power,
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
        self.controller_dt = controller_dt
        self.Q = alpha
        self.R = (1 - alpha) # need to scale control input componet to contend with power component
        self.n_wind_preview_samples = n_wind_preview_samples
        self.n_horizon = n_horizon
        self.wind_preview_type = wind_preview_type
        self.rated_turbine_power = rated_turbine_power 

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
        
    def optimize(self, current_yaw_offsets, current_offline_status, wind_preview_interval_probs, constrain_yaw_dynamics=True, print_progress=False):
        
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
                
                # norm_current_yaw_angles + self.simulation_dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                # clip for control input values between -1 and 1
                
                # if we are solving an optimization problem constrainted by the dynamic state equation, constrain the alowwable range of yaw angles accordingly
                if constrain_yaw_dynamics:
                    # current_yaw_offsets = self.fmodel.core.flow_field.wind_directions - current_yaw_setpoints
                    self._yaw_lbs = np.max([current_yaw_offsets - (self.controller_dt * self.yaw_rate), self._yaw_lbs], axis=0)
                    self._yaw_ubs = np.min([current_yaw_offsets + (self.controller_dt * self.yaw_rate), self._yaw_ubs], axis=0)

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
                norm_turbine_powers = turbine_powers / self.rated_turbine_power
                # for each value in farm_powers, get corresponding next_yaw_angles from evaluation grid
                
                # just selecting one (and only) wind speed, negative because the actual control inputs measure change in absolute yaw angle, not offset
                
                evaluation_grid_tmp = np.reshape(evaluation_grid, (self.Ny_passes[Nii], self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                
                init_yaw_setpoint_change = -(evaluation_grid_tmp[:, :, 0, :] - current_yaw_offsets[np.newaxis, : ,:])[:, :, np.newaxis, :]
                subsequent_yaw_setpoint_changes = -np.diff(evaluation_grid_tmp, axis=2)

                assert np.isclose(np.sum(np.diff(init_yaw_setpoint_change, axis=1)), 0.0), "dynamic state equation for first time-step in horizon should be satisfied in YawOptimizationSRRHC.optimize"
                assert np.isclose(np.sum(np.diff(subsequent_yaw_setpoint_changes, axis=1)), 0.0), "dynamic state equation for subsequent time-step in horizon should be satisfied in YawOptimizationSRRHC.optimize"

                control_inputs = np.concatenate([init_yaw_setpoint_change[:, 0, :, :], subsequent_yaw_setpoint_changes[:, 0, :, :]], axis=1) * (1 / (self.yaw_rate * self.controller_dt))

                norm_turbine_powers = np.reshape(norm_turbine_powers, (self.Ny_passes[Nii], self.n_wind_preview_samples, self.n_horizon, self.nturbs))
                
                if self.wind_preview_type == "stochastic_sample":
                    # cost_states = np.mean(np.sum(norm_turbine_powers**2, axis=(2, 3)), axis=1)[:, np.newaxis] * (-0.5) * self.Q 
                    cost_states = np.sum(norm_turbine_powers**2, axis=(1, 2, 3))[:, np.newaxis] * (-0.5) * self.Q * (1 / self.n_wind_preview_samples)
                elif "stochastic_interval" in self.wind_preview_type:
                    # cost_states = np.sum(np.sum(norm_turbine_powers**2, axis=3) * wind_preview_interval_probs, axis=(1, 2))[:, np.newaxis] * (-0.5) * self.Q
                    # cost_states = np.sum(norm_turbine_powers**2 * wind_preview_interval_probs[np.newaxis, :, :, np.newaxis], axis=(1,2,3))[:, np.newaxis]  * (-0.5) * self.Q  
                    cost_states = np.einsum("xsht, sh -> x", norm_turbine_powers**2, wind_preview_interval_probs)[:, np.newaxis]  * (-0.5) * self.Q 
                else:
                    cost_states = np.einsum("xsht -> x", norm_turbine_powers**2)[:, np.newaxis]  * (-0.5) * self.Q 

                cost_control_inputs = np.sum(control_inputs**2, axis=(1, 2))[:, np.newaxis] * 0.5 * self.R
                cost_terms = np.stack([cost_states, cost_control_inputs], axis=2) # axis=3
                cost = cost_states + cost_control_inputs
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
                assert np.sum(np.diff(np.reshape(yaw_angles_opt_new, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=0)) == 0.0, "optimized yaw offsets should be equal over multiple wind samples in YawOptimizationSRRHC.optimize"

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

                assert np.sum(np.diff(np.reshape(yaw_angles_opt, (self.n_wind_preview_samples, self.n_horizon, self.nturbs)), axis=0)) == 0., "optimized yaw offsets should be equal over multiple wind samples in YawOptimizationSRRHC.optimize"

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
    # max_iter = 15
    # acc = 1e-6
    # optimizers = [
    #     SLSQP(options={"IPRINT": 0, "MAXIT": max_iter, "ACC": acc}),
    #     # NSGA2(options={"xinit": 1, "PrintOut": 0, "maxGen": 50})
    #     # CONMIN(options={"IPRINT": 1, "ITMAX": max_iter})
    #     # ALPSO(options={}) #"maxOuterIter": 25})
    #     ]
    def __init__(self, interface, wind_forecast, simulation_input_dict, wind_field_config, verbose=False, **kwargs):
        
        super().__init__(interface, verbose=verbose)
        
        self.optimizer_idx = optimizer_idx
        # TODO set time-limit
        
        self.optimizer = SLSQP(options={"IPRINT": 0 if verbose else -1, 
                                        "MAXIT": simulation_input_dict["controller"]["max_iter"], 
                                        "ACC": simulation_input_dict["controller"]["acc"]})
        self.init_time = interface.init_time
        self.controller_dt = simulation_input_dict["controller"]["controller_dt"]
        self.simulation_dt = simulation_input_dict["simulation_dt"]
        self.n_turbines = interface.n_turbines
        self.turbines = range(self.n_turbines)
        self.yaw_limits = simulation_input_dict["controller"]["yaw_limits"]
        self.yaw_norm_const = 360.0

        if simulation_input_dict["controller"]["decay_type"].lower() in ["exp", "linear", "cosine", "zero", "none"]:
            self.decay_type = simulation_input_dict["controller"]["decay_type"].lower()
        else:
            raise TypeError("solver must be have value of 'exp', 'linear', or 'cosine', 'zero', or 'none'")
        
        if not isinstance(simulation_input_dict["controller"]["decay_const"], (int, float)) or simulation_input_dict["controller"]["decay_const"] < np.max(np.abs(self.yaw_limits)):
            raise TypeError("decay_const must be a integer or a float representing a yaw offset for which decay = zero, greater than yaw limits")

        self.decay_const = simulation_input_dict["controller"]["decay_const"]
        # if type(simulation_input_dict["controller"]["decay_const"]) is int or type(simulation_input_dict["controller"]["decay_const"]) is float:
        self.decay_all  = simulation_input_dict["controller"]["decay_all"]
        if self.decay_type in ["cosine", "linear", "exp"]:	
            decay_range = self.decay_const - np.max(np.abs(self.yaw_limits))
        if self.decay_type == "cosine":
            # decay_range =  #* (2 * np.pi / 180.0)
            self.decay_factor = (360.0 / (4 * decay_range)) #* (2 * np.pi / 180.0)
        elif self.decay_type == "exp":
            self.decay_factor = np.log(1e-6) / decay_range
        elif self.decay_type == "linear":
            self.decay_factor = 1 / decay_range
        
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
            x = np.arange(0, 60, 1) * 2 * np.pi / 180.0
            y = np.cos(self.decay_factor * x)
            ax.plot(x, y)

        self.yaw_rate = simulation_input_dict["controller"]["yaw_rate"]
        self.yaw_increment = simulation_input_dict["controller"]["yaw_increment"]
        self.alpha = simulation_input_dict["controller"]["alpha"]
        self.beta = simulation_input_dict["controller"]["beta"]
        self.n_horizon = simulation_input_dict["controller"]["n_horizon"]
        self.wind_field_ts = kwargs["wind_field_ts"]
        self.wf_source = kwargs["wf_source"]
        
        self.tid2idx_mapping = kwargs["tid2idx_mapping"] 
        self.target_turbine_indices = simulation_input_dict["controller"]["target_turbine_indices"]
        if self.target_turbine_indices != "all":
            self.sorted_tids = sorted(list(self.target_turbine_indices))
        else:
            self.sorted_tids = np.arange(len(self.tid2idx_mapping))
            
        self.uncertain = simulation_input_dict["controller"]["uncertain"]
        
        # if self.wind_forecast and self.uncertain:
        #     self.mean_ws_horz_cols = [f"loc_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     self.mean_ws_vert_cols = [f"loc_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     self.sd_ws_horz_cols = [f"sd_ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     self.sd_ws_vert_cols = [f"sd_ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     # self.mean_nd_cos_cols = [f"loc_nd_cos_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     # self.mean_nd_sin_cols = [f"loc_nd_sin_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        # else:
        #     self.mean_ws_horz_cols = [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     self.mean_ws_vert_cols = [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #     self.sd_ws_horz_cols = self.sd_ws_vert_cols = []
         
        # self.turbine_ids = np.arange(self.n_turbines) + 1
        # self.historic_measurements = pd.DataFrame(columns=["time"] 
        #                                           + [f"ws_horz_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #                                           + [f"ws_vert_{self.idx2tid_mapping[t_idx]}" for t_idx in np.arange(len(self.idx2tid_mapping))]
        #                                           + [f"nd_cos_{tid}" for tid in self.tid2idx_mapping]
        #                                           + [f"nd_sin_{tid}" for tid in self.tid2idx_mapping], 
        #                                           dtype=pd.Float64Dtype())

        self._last_yaw_setpoints = None
        self._last_measured_time = None
        self.current_time = 0.0
        self.run_custom_sens = "chain" in simulation_input_dict["controller"]["diff_type"] or "direct" in simulation_input_dict["controller"]["diff_type"] or simulation_input_dict["controller"]["diff_type"] == "none"
        # self.run_brute_force_sens = simulation_input_dict["controller"]["diff_type"] == "brute_force"
        self.run_brute_force_sens = False
        # self.run_brute_force_sens = True
        if not isinstance(simulation_input_dict["controller"]["clip_value"], (int, float)) or simulation_input_dict["controller"]["clip_value"] < np.max(np.abs(self.yaw_limits)):
            raise TypeError("clip_value must be a integer or a float representing a yaw offset at which perturbed values are clipped before entering FLORIS, greater or equal to yaw limits")	

        self.clip_value = simulation_input_dict["controller"]["clip_value"]


        if simulation_input_dict["controller"]["solver"].lower() in ['slsqp', 'sequential_slsqp', 'serial_refine', 'zsgd']:
            self.solver = simulation_input_dict["controller"]["solver"].lower()
        else:
            raise TypeError("solver must be have value of 'slsqp', 'sequential_slsqp', 'serial_refine', or 'zsgd")
        
        self.use_filt = simulation_input_dict["controller"]["use_filtered_wind_dir"]
        
        self.lpf_time_const = simulation_input_dict["controller"]["lpf_time_const"]
        self.lpf_start_time = self.init_time + pd.Timedelta(seconds=simulation_input_dict["controller"]["lpf_start_time"])
        self.lpf_alpha = np.exp(-(1 / simulation_input_dict["controller"]["lpf_time_const"]) * simulation_input_dict["simulation_dt"])
        self.historic_measurements = {"wind_directions": [],
                                      "wind_speeds": []}
        self.state_cons_activated = {"lower": None, "upper": None}
        # self.filtered_measurements = {"wind_directions": [],
        #                               "wind_speeds": []}
        
        self.use_state_cons = simulation_input_dict["controller"]["use_state_cons"]
        self.use_dyn_state_cons = simulation_input_dict["controller"]["use_dyn_state_cons"]

        if simulation_input_dict["controller"]["state_con_type"].lower() in ["extreme"]:
            self.state_con_type = simulation_input_dict["controller"]["state_con_type"].lower()
        else:
            raise TypeError("state_con_type must be have value of 'extreme'")

        # self.seed = kwargs["seed"] if "seed" in kwargs else None

        if simulation_input_dict["controller"]["wind_preview_type"].lower() in ['stochastic_sample', 'stochastic_interval_rectangular', 'stochastic_interval_elliptical', 'persistent', 'perfect']:
            self.wind_preview_type = simulation_input_dict["controller"]["wind_preview_type"].lower()
        else:
            raise TypeError("wind_preview_type must be have value of 'stochastic_sample', 'stochastic_interval_rectangular', 'stochastic_interval_elliptical', 'persistent', or 'perfect")
        
        if ("stochastic_interval" in self.wind_preview_type and simulation_input_dict["controller"]["diff_type"].lower() in ["chain_cd", "chain_fd", "direct_cd", "direct_fd"]) \
            or (self.wind_preview_type == "stochastic_sample" and simulation_input_dict["controller"]["diff_type"].lower() in ["chain_cd", "chain_fd", "direct_cd", "direct_fd", "chain_zscg", "direct_zscg"]) \
                or (self.wind_preview_type in ["persistent", "perfect"] and simulation_input_dict["controller"]["diff_type"].lower() in ["chain_cd", "chain_fd", "direct_cd", "direct_fd"]):
            self.diff_type = simulation_input_dict["controller"]["diff_type"].lower()
        else:
            raise TypeError(f"diff_type must be have value of 'central_diff', 'chain_cd', or 'chain_fd', 'direct_cd', 'direct_fd', 'chain_zscg', or 'direct_zscg', instead it has value {simulation_input_dict['controller']['diff_type'].lower()}. Only 'chain_fd', or 'direct_fd' are permitted for wind_preview_type == stochastic_sample")

        if self.wind_preview_type == "stochastic_sample":
            self.stochastic_sample_u_scale = simulation_input_dict["controller"]["stochastic_sample_u_scale"]

        if "stochastic_interval" in self.wind_preview_type:
            
            if (simulation_input_dict["controller"]["n_wind_preview_samples"] % 2 == 0):
                print(f"n_wind_preview_samples must be an odd number to include mean value of distribution, increasing to {simulation_input_dict['controller']['n_wind_preview_samples'] + 1}")
                simulation_input_dict["controller"]["n_wind_preview_samples"] += 1

        wind_field_config["n_preview_steps"] = simulation_input_dict["controller"]["n_horizon"] * int(simulation_input_dict["controller"]["controller_dt"] / simulation_input_dict["simulation_dt"])
        wind_field_config["n_samples_per_init_seed"] = simulation_input_dict["controller"]["n_wind_preview_samples"] 
        
        wf = WindField(**wind_field_config)
        # wind_preview_generator = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
        if self.wind_preview_type == "stochastic_sample":
            self.n_wind_preview_samples = simulation_input_dict["controller"]["n_wind_preview_samples"]
        elif self.wind_preview_type == "stochastic_interval_rectangular":
            self.n_wind_preview_samples = simulation_input_dict["controller"]["n_wind_preview_samples"]**2 # cross-product of u an v values
            self.n_wind_preview_intervals = simulation_input_dict["controller"]["n_wind_preview_samples"]
        elif self.wind_preview_type == "stochastic_interval_elliptical": # ((n_intervals**2 - 1) * (int(n_intervals // 2))) + 1
            # self.n_wind_preview_samples = (simulation_input_dict["controller"]["n_wind_preview_samples"]**2 - 1) * (int(simulation_input_dict["controller"]["n_wind_preview_samples"] // 2)) + 1 #
            self.n_wind_preview_samples = ((int((simulation_input_dict["controller"]["n_wind_preview_samples"] // 4) \
                                       + np.ceil((simulation_input_dict["controller"]["n_wind_preview_samples"] % 4) / 4)) * 4) - 1)\
                                          * (int(simulation_input_dict["controller"]["n_wind_preview_samples"] // 2)) + 1
            self.n_wind_preview_intervals = simulation_input_dict["controller"]["n_wind_preview_samples"]
        else:
            self.n_wind_preview_samples = 1
            simulation_input_dict["controller"]["n_wind_preview_samples"] = 1
        
        if "stochastic" in simulation_input_dict["controller"]["wind_preview_type"]:
            def wind_preview_func(current_freestream_measurements, time_step, seed=None, return_interval_values=False, n_intervals=simulation_input_dict["controller"]["n_wind_preview_samples"], max_std_dev=2): 
                # returns cond_mean_u, cond_mean_v, cond_cov_u, cond_cov_v
                if return_interval_values:
                    distribution_params = generate_wind_preview( 
                                    wf, current_freestream_measurements, time_step,
                                    wind_preview_generator=wf._sample_wind_preview, 
                                    return_params=True)
                    
                    # if self.wind_preview_type == "stochastic_interval_rectangular":
                    # 	n_samples = n_intervals**2
                    # elif self.wind_preview_type == "stochastic_interval_elliptical" or self.wind_preview_type == "stochastic_sample":
                    # 	n_samples = ((n_intervals**2 - 1) * (int(n_intervals // 2))) + 1
                    
                    std_u = np.sqrt(np.diag(distribution_params[2]))[np.newaxis, :]
                    std_v = np.sqrt(np.diag(distribution_params[3]))[np.newaxis, :]
                     # creates trajectories over the horizon for each standard deviation from the mean e.g. trajectory for -2 stds from mean for u, v .... to +2 stds from mean for u, v
                    if self.wind_preview_type == "stochastic_interval_rectangular" or self.wind_preview_type == "stochastic_sample":
                        if n_intervals > 1:
                            std_divisions = np.linspace(-max_std_dev, max_std_dev, n_intervals)[:, np.newaxis]
                        else:
                            std_divisions = np.array([0])[:, np.newaxis]
                        dev_u = np.matmul(std_divisions, std_u)
                        dev_v = np.matmul(std_divisions, std_v)
                        u_vals = distribution_params[0] + dev_u
                        v_vals = distribution_params[1] + dev_v
                        uv_combs = np.swapaxes(list(product(u_vals, v_vals)), 1, 2)
                    elif self.wind_preview_type == "stochastic_interval_elliptical": 
                        # Choose the probabilistic constraints for the stochastic_sample method with the stochastic_interval_elliptical method
                        if n_intervals > 1:
                            std_divisions = np.linspace(0, max_std_dev, (n_intervals // 2) + 1)[:, np.newaxis]
                        else:
                            std_divisions = np.array([0])[:, np.newaxis]
                        dev_u = np.matmul(std_divisions, std_u)
                        dev_v = np.matmul(std_divisions, std_v)
                        # multiple of 4 number of angular intervals, equally divided over each quadrature
                        theta = np.linspace(0, 2 * np.pi, int((n_intervals // 4) + np.ceil((n_intervals % 4) / 4)) * 4)[:-1, np.newaxis] 
                        u_vals = np.vstack([distribution_params[0] + dev_u[0, :], 
                                              distribution_params[0] + (
                                                  dev_u[1:, :] * np.cos(theta)[:, np.newaxis]).reshape(-1, self.n_horizon)])
                        v_vals = np.vstack([distribution_params[1] + dev_v[0, :], 
                                              distribution_params[1] + (
                                                  dev_v[1:, :] * np.sin(theta)[:, np.newaxis]).reshape(-1, self.n_horizon)])
                        uv_combs = np.dstack([u_vals, v_vals])
                    
                    n_samples = uv_combs.shape[0]
                    wind_preview_data = {
                        "FreestreamWindMag": np.zeros((n_samples, self.n_horizon + 1)), 
                        "FreestreamWindDir": np.zeros((n_samples, self.n_horizon + 1))
                    }
                    
                    mag = np.linalg.norm([current_freestream_measurements[0], current_freestream_measurements[1]])
                    wind_preview_data[f"FreestreamWindMag"][:, 0] = [mag] * n_samples
                    
                    # compute freestream wind direction angle from above, clockwise from north
                    direction = np.arctan2(current_freestream_measurements[0], current_freestream_measurements[1])
                    direction = (180.0 + np.rad2deg(direction)) % 360.0

                    wind_preview_data[f"FreestreamWindDir"][:, 0] = [direction] * n_samples

                    mag_vals = np.linalg.norm(uv_combs, axis=2)
                    # compute directions
                    dir_vals = np.arctan2(uv_combs[:, :, 0], uv_combs[:, :, 1])
                    dir_vals = (180.0 + np.rad2deg(dir_vals)) % 360.0

                    wind_preview_probs = (norm.pdf(uv_combs[:, :, 0], loc=distribution_params[0], scale=std_u) \
                         * norm.pdf(uv_combs[:, :, 1], loc=distribution_params[1], scale=std_v))

                    # add values and marginal probabilities corresponding to n_wind_preview_samples division of gaussian
                    
                    wind_preview_data[f"FreestreamWindMag"][:, 1:] = mag_vals
                    wind_preview_data[f"FreestreamWindDir"][:, 1:] = dir_vals

                    # wind_preview_probs = np.array(wind_preview_probs).T
                    wind_preview_probs = np.divide(wind_preview_probs, np.sum(wind_preview_probs, axis=0))

                    if False and self.wind_preview_type in ["stochastic_interval_rectangular", "stochastic_interval_elliptical"]:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        import pandas as pd
                        
                        df = pd.DataFrame({
                            "Downwind": uv_combs[:, :, 0].flatten(), 
                            "Crosswind": uv_combs[:, :, 1].flatten(), 
                            "Direction": dir_vals.flatten(), 
                            "Magnitude": mag_vals.flatten(), 
                             "Time Step": np.tile(np.arange(self.n_horizon) + 1, (uv_combs.shape[0], )).astype(int),
                            #  "Std. Dev": np.tile(list(product(std_divisions, std_divisions)), (uv_combs.shape[0], )), 
                             "Sample": np.repeat(np.arange(uv_combs.shape[0]), (self.n_horizon, )) 
                        })

                        fig, ax = plt.subplots(1, 2)
                        sns.scatterplot(ax=ax[0], data=df, x="Downwind", y="Crosswind", hue="Time Step")
                        sns.scatterplot(ax=ax[1], data=df, x="Direction", y="Magnitude", hue="Time Step")
                        ax[0].legend([], [], frameon=False)
                        ax[0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Downwind Wind Speed [m/s]")
                        ax[1].set(ylabel="Wind Magnitude [m/s]", xlabel="Wind Direction [$^\\circ$]")
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.suptitle(f"{self.wind_preview_type}_scatter")
                        # fig.show()
                        fig.savefig(f"/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/{self.wind_preview_type}_scatter.png")
                        
                        # plot trajectories
                        fig, ax = plt.subplots(2, 2)
                        sns.lineplot(ax=ax[0, 0], data=df, x="Time Step", y="Downwind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1, 0], data=df, x="Time Step", y="Crosswind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[0, 1], data=df, x="Time Step", y="Direction", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1, 1], data=df, x="Time Step", y="Magnitude", hue="Sample", marker="o")
                        # ax[0, 0].legend([], [], frameon=False)
                        ax[0, 0].set(ylabel="Downwind Wind Speed [m/s]", xlabel="")
                        ax[1, 0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Horizon Time Step [-]")
                        ax[0, 1].set(ylabel="Wind Direction [$^\\circ$]", xlabel="")
                        ax[1, 1].set(ylabel="Wind Magnitude [m/s]", xlabel="Horizon Time Step [-]")
                        ax[0, 0].legend([], [], frameon=False)
                        ax[1, 0].legend([], [], frameon=False)
                        ax[1, 1].legend([], [], frameon=False)
                        sns.move_legend(ax[0, 1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.suptitle(f"{self.wind_preview_type}_trajectories")
                        fig.savefig(f"/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/{self.wind_preview_type}_trajectories_time.png")
                        # df.sort_values(by=["Sample", "Time Step"])
                        fig, ax = plt.subplots(1, 2)
                        sns.lineplot(ax=ax[0], data=df[["Sample", "Time Step", "Downwind", "Crosswind"]].sort_values(by=["Sample", "Time Step"]), x="Downwind", y="Crosswind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1], data=df[["Sample", "Time Step", "Direction", "Magnitude"]].sort_values(by=["Sample", "Time Step"]), x="Direction", y="Magnitude", hue="Sample", marker="o")
                        # ax[0, 0].legend([], [], frameon=False)
                        ax[0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Downwind Wind Speed [m/s]")
                        ax[1].set(ylabel="Wind Magnitude [m/s]", xlabel="Wind Direction [$^\\circ$]")
                        ax[0].legend([], [], frameon=False)
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.suptitle(f"{self.wind_preview_type}_trajectories_2d")
                        fig.savefig(f"/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/{self.wind_preview_type}_trajectories_2d.png")

                        if self.wind_preview_type == "stochastic_interval_rectangular":
                            df = pd.DataFrame({
                                "Downwind": u_vals.flatten(), 
                                "Crosswind": v_vals.flatten(), 
                                # "Direction": dir_vals.flatten(), 
                                # "Magnitude": mag_vals.flatten(), 
                                "Time Step": np.tile(np.arange(self.n_horizon) + 1, (u_vals.shape[0],)).astype(int),
                                "# Standard Deviations": np.repeat(std_divisions, (self.n_horizon,)), 
                                #  "Sample": np.repeat(np.arange(dev_u.shape[0]), (self.n_horizon, )) 
                            })
                            fig, ax = plt.subplots(1, 2)
                            sns.scatterplot(ax=ax[0], data=df, x="# Standard Deviations", y="Downwind", hue="Time Step")
                            sns.scatterplot(ax=ax[1], data=df, x="# Standard Deviations", y="Crosswind", hue="Time Step")
                            ax[0].legend([], [], frameon=False)
                            ax[0].set(ylabel="Downwind Wind Speed [m/s]")
                            ax[1].set(ylabel="Crosswind Wind Speed [m/s]")
                            sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                            plt.tight_layout(pad=2.0)
                            # fig.savefig("/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/stochastic_interval_intervals.png")

                        # for i in range(self.n_horizon):
                        # ax[0, 0].scatter(std_divisions, u_vals[:, i])
                        # ax[0, 1].scatter(std_divisions, v_vals[:, i])
                        # ax[1, 0].scatter(std_divisions, mag_vals[:, i])
                        # ax[1, 1].scatter(std_divisions, dir_vals[:, i])
                        # sns.scatter(ax[0], uv_combs[:, i, 0], uv_combs[:, i, 1])
                        # sns.scatter(ax[1], dir_vals[:, i], mag_vals[:, i])

                        # ax[0].set(title="v_vals vs. u_vals")
                        # ax[1].set(title="mag_vals vs. dir_vals")
                        # ax[0, 0].set(title="std_divisions vs. u_vals")
                        # ax[0, 1].set(title="std_divisions vs. v_vals")
                        # ax[1, 0].set(title="std_divisions vs. mag_vals")
                        # ax[1, 1].set(title="std_divisions vs. dir_vals")
                        

                else:
                    wind_preview_data = generate_wind_preview(wf, current_freestream_measurements, time_step,
                                 wind_preview_generator=wf._sample_wind_preview, 
                                return_params=False, include_uv=False, seed=seed)
                    wind_preview_probs = None
                # NOTE run this for generate_sample_figures
                if False and self.n_wind_preview_samples == 500 and self.wind_preview_type == "stochastic_sample":
                        import matplotlib.pyplot as plt
                        import pandas as pd
                        import seaborn as sns

                        wind_preview_data_sample = generate_wind_preview(wf, current_freestream_measurements, time_step,
                                 wind_preview_generator=wf._sample_wind_preview, 
                                return_params=False, include_uv=True)
                        
                        df = pd.DataFrame({
                            "Downwind": wind_preview_data_sample["FreestreamWindSpeedU"][:, 1:].flatten(), 
                            "Crosswind": wind_preview_data_sample["FreestreamWindSpeedV"][:, 1:].flatten(), 
                            "Direction": wind_preview_data_sample["FreestreamWindDir"][:, 1:].flatten(), 
                            "Magnitude": wind_preview_data_sample["FreestreamWindMag"][:, 1:].flatten(), 
                             "Time Step": np.tile(np.arange(self.n_horizon) + 1, (wind_preview_data_sample["FreestreamWindSpeedU"].shape[0], )).astype(int),
                             "Sample": np.repeat(np.arange(wind_preview_data_sample["FreestreamWindSpeedU"].shape[0]), (self.n_horizon, )) 
                        })
                        fig, ax = plt.subplots(1, 2)
                        sns.scatterplot(ax=ax[0], data=df, x="Downwind", y="Crosswind", hue="Time Step")
                        sns.scatterplot(ax=ax[1], data=df, x="Direction", y="Magnitude", hue="Time Step")
                        ax[0].legend([], [], frameon=False)
                        ax[0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Downwind Wind Speed [m/s]")
                        ax[1].set(ylabel="Wind Magnitude [m/s]", xlabel="Wind Direction [$^\\circ$]")
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.show()
                        fig.savefig("/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/stochastic_sample_scatter.png")

                        # plot trajectories
                        fig, ax = plt.subplots(2, 2)
                        sns.lineplot(ax=ax[0, 0], data=df, x="Time Step", y="Downwind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1, 0], data=df, x="Time Step", y="Crosswind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[0, 1], data=df, x="Time Step", y="Direction", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1, 1], data=df, x="Time Step", y="Magnitude", hue="Sample", marker="o")
                        # ax[0, 0].legend([], [], frameon=False)
                        ax[0, 0].set(ylabel="Downwind Wind Speed [m/s]", xlabel="")
                        ax[1, 0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Horizon Time Step [-]")
                        ax[0, 1].set(ylabel="Wind Direction [$^\\circ$]", xlabel="")
                        ax[1, 1].set(ylabel="Wind Magnitude [m/s]", xlabel="Horizon Time Step [-]")
                        ax[0, 0].legend([], [], frameon=False)
                        ax[1, 0].legend([], [], frameon=False)
                        ax[1, 1].legend([], [], frameon=False)
                        sns.move_legend(ax[0, 1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.suptitle(f"{self.wind_preview_type}_trajectories")
                        fig.savefig("/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/stochastic_sample_trajectories_time.png")	

                        fig, ax = plt.subplots(1, 2)
                        sns.lineplot(ax=ax[0], data=df[["Sample", "Time Step", "Downwind", "Crosswind"]].sort_values(by=["Sample", "Time Step"]), x="Downwind", y="Crosswind", hue="Sample", marker="o")
                        sns.lineplot(ax=ax[1], data=df[["Sample", "Time Step", "Direction", "Magnitude"]].sort_values(by=["Sample", "Time Step"]), x="Direction", y="Magnitude", hue="Sample", marker="o")
                        # ax[0, 0].legend([], [], frameon=False)
                        ax[0].set(ylabel="Crosswind Wind Speed [m/s]", xlabel="Downwind Wind Speed [m/s]")
                        ax[1].set(ylabel="Wind Magnitude [m/s]", xlabel="Wind Direction [$^\\circ$]")
                        ax[0].legend([], [], frameon=False)
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        plt.tight_layout(pad=2.0)
                        fig.suptitle(f"{self.wind_preview_type}_trajectories")
                        fig.savefig(f"/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/{self.wind_preview_type}_trajectories_2d.png")
    
                        # ax = ax.flatten()
                        # for i in range(self.n_horizon):
                        # 	# ax[i].scatter(wind_preview_data["FreestreamWindMag"][:, i], wind_preview_data["FreestreamWindDir"][:, i])
                        # 	# ax[i].set(xlim=(np.sqrt(6**2 + 0**2), np.sqrt(8**2 + 4**2)), 
                        # 	# ylim=((270.0 - (np.arctan2(4, 6) * (180 / np.pi))) % 360.0, (270.0 - (np.arctan2(-4, 6) * (180 / np.pi))) % 360.0))
                        # 	ax[i].scatter(wind_preview_data["FreestreamWindSpeedU"][:, i], wind_preview_data["FreestreamWindSpeedV"][:, i])
                        # 	ax[i].set(xlim=(6, 8), 
                        # 	ylim=(-4, 4))
                        # ax[0].set(title="mag_vals vs. dir_vals")
                        # ax[0, 0].set(title="std_divisions vs. u_vals")
                        # ax[0, 1].set(title="std_divisions vs. v_vals")
                        # ax[1, 0].set(title="std_divisions vs. mag_vals")
                        # ax[1, 1].set(title="std_divisions vs. dir_vals")
                        # fig.show()
                
                return wind_preview_data, wind_preview_probs
        
        elif simulation_input_dict["controller"]["wind_preview_type"] == "persistent":
            def wind_preview_func(current_freestream_measurements, time_step, seed=None, return_interval_values=False, n_intervals=None, max_std_dev=None):
                wind_preview_data = {
                    "FreestreamWindMag": np.zeros((self.n_wind_preview_samples, self.n_horizon + 1)),
                    "FreestreamWindDir": np.zeros((self.n_wind_preview_samples, self.n_horizon + 1))
                    }
                # for j in range(simulation_input_dict["controller"]["n_horizon"] + 1):
                wind_preview_data[f"FreestreamWindMag"] = np.broadcast_to(kwargs["wind_field_ts"]["FreestreamWindMag"].iloc[time_step], wind_preview_data[f"FreestreamWindMag"].shape)
                wind_preview_data[f"FreestreamWindDir"] = np.broadcast_to(kwargs["wind_field_ts"]["FreestreamWindDir"].iloc[time_step], wind_preview_data[f"FreestreamWindDir"].shape)
                
                if return_interval_values:
                    return wind_preview_data, np.ones((1, self.n_horizon))
                else:
                    return wind_preview_data, None
        
        elif simulation_input_dict["controller"]["wind_preview_type"] == "perfect":
            def wind_preview_func(current_freestream_measurements, time_step, seed=None, return_interval_values=False, n_intervals=None, max_std_dev=None):
                # TODO HIGH check that this is returning true future wind mag/dir based on proper computation from horz/vert components
                wind_preview_data = {
                    "FreestreamWindMag": np.zeros((self.n_wind_preview_samples, self.n_horizon + 1)), 
                    "FreestreamWindDir": np.zeros((self.n_wind_preview_samples, self.n_horizon + 1))
                }
                delta_k = slice(time_step, 
                                time_step + (simulation_input_dict["controller"]["n_horizon"] + 1) * int(simulation_input_dict["controller"]["controller_dt"] // simulation_input_dict["simulation_dt"]), 
                                int(simulation_input_dict["controller"]["controller_dt"] // simulation_input_dict["simulation_dt"]))
                wind_preview_data[f"FreestreamWindMag"] = np.broadcast_to(kwargs["wind_field_ts"]["FreestreamWindMag"].iloc[delta_k], wind_preview_data[f"FreestreamWindMag"].shape)
                wind_preview_data[f"FreestreamWindDir"] = np.broadcast_to(kwargs["wind_field_ts"]["FreestreamWindDir"].iloc[delta_k], wind_preview_data[f"FreestreamWindDir"].shape)
                
                if return_interval_values:
                    return wind_preview_data, np.ones((1, self.n_horizon))
                else:
                    return wind_preview_data, None
        
        self.wind_preview_func = wind_preview_func

        self.max_std_dev = simulation_input_dict["controller"]["max_std_dev"]

        self.warm_start = simulation_input_dict["controller"]["warm_start"]

        if self.warm_start == "lut" or self.warm_start == "constrained_lut":
            
            self.fi_lut = ControlledFlorisModel(yaw_limits=simulation_input_dict["controller"]["yaw_limits"],
                                        simulation_dt=simulation_input_dict["simulation_dt"],
                                        yaw_rate=simulation_input_dict["controller"]["yaw_rate"],
                                        config_path=simulation_input_dict["controller"]["floris_input_file"])

            lut_input_dict = dict(simulation_input_dict)
            # lut_input_dict["controller"]["use_lut_filtered_wind_dir"] = False
            # lut_input_dict["controller"]["controller_dt"] = self.simulation_dt
            self.ctrl_lut = LookupBasedWakeSteeringController(self.fi_lut, simulation_input_dict=lut_input_dict, 
                                                    lut_path=simulation_input_dict["controller"]["lut_path"], 
                                                    generate_lut=simulation_input_dict["controller"]["generate_lut"], 
                                                    wind_field_ts=kwargs["wind_field_ts"])

        self.Q = self.alpha
        self.R = (1 - self.alpha)
        self.nu = simulation_input_dict["controller"]["nu"]
        # self.sequential_neighborhood_solve = simulation_input_dict["controller"]["sequential_neighborhood_solve"]
        
        self.basin_hop = simulation_input_dict["controller"]["basin_hop"]

        self.n_solve_turbines = self.n_turbines if self.solver != "sequential_slsqp" else 1
        self.n_solve_states = self.n_solve_turbines * self.n_horizon
        self.n_solve_control_inputs = self.n_solve_turbines * self.n_horizon

        self.dyn_state_jac, self.state_jac = self.con_sens_rules(self.n_solve_turbines)
        
        # Set initial conditions
        if isinstance(simulation_input_dict["controller"]["initial_conditions"]["yaw"], (float, list, np.ndarray)):
            self.yaw_IC = simulation_input_dict["controller"]["initial_conditions"]["yaw"]
        elif simulation_input_dict["controller"]["initial_conditions"]["yaw"] == "auto":
            self.yaw_IC = None
        else:
            raise Exception("must choose float or 'auto' for initial yaw value")

        if hasattr(self.yaw_IC, "__len__"):
            if len(self.yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": np.array(self.yaw_IC)}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": self.yaw_IC * np.ones((self.n_turbines,))}
        
        self.initial_state = (self.yaw_IC / self.yaw_norm_const) * np.ones((self.n_turbines,))
        
        self.opt_sol = {"states": np.tile(self.initial_state, self.n_horizon), 
                        "control_inputs": np.zeros((self.n_horizon * self.n_turbines,))}
                        # "control_inputs_start": np.zeros((self.n_horizon * self.n_turbines,))}
        
        if simulation_input_dict["controller"]["control_input_domain"].lower() in ['discrete', 'continuous']:
            self.control_input_domain = simulation_input_dict["controller"]["control_input_domain"].lower()
        else:
            raise TypeError("control_input_domain must be have value of 'discrete' or 'continuous'")
        
        self.wind_ti = 0.08

        self.fi = ControlledFlorisModel(t0=kwargs["wind_field_ts"]["time"].iloc[0], 
                                        yaw_limits=self.yaw_limits, 
                                        simulation_dt=self.simulation_dt, 
                                        yaw_rate=self.yaw_rate, 
                                        config_path=simulation_input_dict["controller"]["floris_input_file"],
                                        target_turbine_indices=simulation_input_dict["controller"]["target_turbine_indices"] or "all",
                                        uncertain=simulation_input_dict["controller"]["uncertain"],
                                        turbine_signature=kwargs["turbine_signature"],
                                        tid2idx_mapping=kwargs["tid2idx_mapping"])
        self.rated_turbine_power = simulation_input_dict["controller"]["rated_turbine_power"]
        # list(self.fi.env.core.farm.turbine_power_thrust_tables.values())[0]["power"].max() * 1e3, 
        # self.floris_proc = Process(target=run_floris_proc, args=(self.fi.env,))
        
        if self.solver == "serial_refine":
            # self.fi_opt = FlorisModelDev(simulation_input_dict["controller"]["floris_input_file"]) #.replace("floris", "floris_dev"))
            if self.warm_start == "lut":
                print("Can't warm-start FLORIS SR solver, setting self.warm_start to none")
                # self.warm_start = "greedy"
        elif self.solver == "slsqp":
            if self.run_custom_sens:
                self.pyopt_prob = self.setup_slsqp_solver(list(range(self.n_turbines)), [], use_sens_rules=True)
            if self.run_brute_force_sens:
                self.pyopt_prob_nosens = self.setup_slsqp_solver(list(range(self.n_turbines)), [], use_sens_rules=False)
        elif self.solver == "zsgd":
            pass

    def _first_ord_filter(self, x, alpha):
        
        b = [1 - alpha]
        a = [1, -alpha]
        return lfilter(b, a, x)

    def con_sens_rules(self, n_solve_turbines):
        n_solve_states = n_solve_turbines * self.n_horizon
        n_solve_control_inputs = n_solve_turbines * self.n_horizon

        dyn_state_con_sens = {"states": [], "control_inputs": []}#, "control_inputs_start": []}
        state_con_sens = {"states": [], "control_inputs": []}#, "control_inputs_start": []}

        dyn_state_con_sens["control_inputs"] = -(self.controller_dt * (self.yaw_rate / self.yaw_norm_const)) * np.eye(n_solve_control_inputs)
        # dyn_state_con_sens["control_inputs_start"] = 0 * np.eye(n_solve_control_inputs)
        
        dyn_state_con_sens["states"] = np.zeros((n_solve_states, n_solve_states))
        dyn_state_con_sens["states"][n_solve_turbines:, :-n_solve_turbines] = -np.eye(n_solve_states - n_solve_turbines)
        dyn_state_con_sens["states"] += np.eye(n_solve_states)

        state_con_sens["states"] = -np.eye(n_solve_control_inputs)
        state_con_sens["control_inputs"] = np.zeros((n_solve_control_inputs, n_solve_control_inputs))
        # state_con_sens["control_inputs_start"] = np.zeros((n_solve_control_inputs, n_solve_control_inputs))

        if self.state_con_type == "extreme":
            state_con_sens["states"] = np.tile(state_con_sens["states"], (2, 1))
            state_con_sens["control_inputs"] = np.tile(state_con_sens["control_inputs"], (2, 1))
            # state_con_sens["control_inputs_start"] = np.tile(state_con_sens["control_inputs_start"], (2, 1))

        return dyn_state_con_sens, state_con_sens
    
    def dyn_state_rules(self, opt_var_dict, solve_turbine_ids):
        n_solve_turbines = len(solve_turbine_ids)
        opt_var_dict["states"] = np.array(opt_var_dict["states"])
        opt_var_dict["control_inputs"] = np.array(opt_var_dict["control_inputs"])
        # opt_var_dict["control_inputs_start"] = np.array(opt_var_dict["control_inputs_start"])
        # define constraints
        n_solve_states = self.n_horizon * n_solve_turbines
        delta_yaw = self.controller_dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"]
        dyn_state_cons = np.zeros(n_solve_states)
        # TODO HIGH does it make sense to apply mod here
        dyn_state_cons[:n_solve_turbines:] = opt_var_dict["states"][:n_solve_turbines] - np.mod(self.initial_state[solve_turbine_ids] + delta_yaw[:n_solve_turbines], 1)
        dyn_state_cons[n_solve_turbines:] = opt_var_dict["states"][n_solve_turbines:] - np.mod(opt_var_dict["states"][:-n_solve_turbines] + delta_yaw[n_solve_turbines:], 1)

        return dyn_state_cons
    
    def state_rules(self, opt_var_dict, disturbance_dict, yaw_setpoints, solve_turbine_ids):
        
        # define constraints
        # rather than including every sample, could only include most 'extreme' wind directions...
        wd_range = np.vstack([np.min(disturbance_dict["wind_direction"], axis=0), np.max(disturbance_dict["wind_direction"], axis=0)])
        # in order of all turbines for first horizon step min wind dir, then all turbines for second horizon step min wind dir
        # ... all turbines for last horizon step min wind dir, then all turbines for first horizon step max wind dir...
        yaw_offset = wd_range[:, :, np.newaxis] - yaw_setpoints[np.newaxis, :, solve_turbine_ids]
        abs_yaw_offset = np.abs(yaw_offset)
        abs_yaw_offset =  np.stack([abs_yaw_offset, 360.0 - abs_yaw_offset], axis=3)
        yaw_diff_idx = np.argmin(abs_yaw_offset, axis=3)
        rows, cols, depths = np.indices(yaw_diff_idx.shape) 
        abs_yaw_offset = abs_yaw_offset[rows, cols, depths, yaw_diff_idx]
        dir_yaw_offset = np.sign(yaw_offset)
        dir_yaw_offset[yaw_diff_idx == 1] = -dir_yaw_offset[yaw_diff_idx == 1]
        
        state_cons = (abs_yaw_offset * dir_yaw_offset).flatten() / self.yaw_norm_const	
        return state_cons
    
    def setup_slsqp_solver(self, solve_turbine_ids, downstream_turbine_ids, use_sens_rules=True):
        n_solve_turbines = len(solve_turbine_ids)
        n_solve_states = n_solve_turbines * self.n_horizon
        n_solve_control_inputs = n_solve_turbines * self.n_horizon

        # initialize optimization object
        n_wind_samples = self.n_wind_preview_samples * self.n_horizon
        
        if self.diff_type == "chain_cd" or self.diff_type == "direct_cd":
            self.plus_slices = [slice((2 * i + 1) * n_wind_samples, ((2 * i) + 2) * n_wind_samples) for i in solve_turbine_ids]
            self.neg_slices = [slice((2 * i + 2) * n_wind_samples, ((2 * i) + 3) * n_wind_samples) for i in solve_turbine_ids]
        elif self.diff_type == "chain_fd" or self.diff_type == "direct_fd":
            self.plus_slices = [slice((i + 1) * n_wind_samples, (i + 2) * n_wind_samples) for i in solve_turbine_ids]	
        
        if self.diff_type in ["chain_cd", "chain_fd", "direct_cd", "direct_fd"]:
            self.plus_indices = np.concatenate([np.arange(s.start, s.stop, s.step) for s in self.plus_slices])
        
        if "cd" in self.diff_type:
            self.neg_indices = np.concatenate([np.arange(s.start, s.stop, s.step) for s in self.neg_slices])
         
        dyn_state_jac, state_jac = self.con_sens_rules(n_solve_turbines)
        sens_rules = self.generate_sens_rules(solve_turbine_ids, downstream_turbine_ids, dyn_state_jac, state_jac)
        if use_sens_rules:
            opt_rules = self.generate_opt_rules(solve_turbine_ids, downstream_turbine_ids)
            pyopt_prob = Optimization("Wake Steering MPC", opt_rules, sens=sens_rules, comm=MPI.COMM_SELF)
        else:
            opt_rules = self.generate_opt_rules(solve_turbine_ids, downstream_turbine_ids, compute_derivatives=False)
            pyopt_prob = Optimization("Wake Steering MPC", opt_rules, sens="CD", comm=MPI.COMM_SELF)
        
        # add design variables
        pyopt_prob.addVarGroup("states", n_solve_states,
                                    varType="c",  # continuous variables
                                    lower=[0.0] * n_solve_states,
                                    upper=[1.0] * n_solve_states,
                                    value=[0.0] * n_solve_states)
                                    # scale=(1 / self.yaw_norm_const))
        
        if self.control_input_domain == 'continuous':
            pyopt_prob.addVarGroup("control_inputs", n_solve_control_inputs,
                                        varType="c",
                                        lower=[-1.0] * n_solve_control_inputs,
                                        upper=[1.0] * n_solve_control_inputs,
                                        value=[0.0] * n_solve_control_inputs)
        else:
            pyopt_prob.addVarGroup("control_inputs", n_solve_control_inputs,
                                        varType="i",
                                        lower=[-1] * n_solve_control_inputs,
                                        upper=[1] * n_solve_control_inputs,
                                        value=[0] * n_solve_control_inputs)
        
        # # add dynamic state equation constraints
        # jac = self.con_sens_rules()
        if self.use_dyn_state_cons:
            pyopt_prob.addConGroup("dyn_state_cons", n_solve_states, lower=0.0, upper=0.0)
                    #   linear=True, wrt=["states", "control_inputs"], # NOTE supplying fixed jac won't work because value of initial_state changes
                        #   jac=jac)

        if self.use_state_cons:
            pyopt_prob.addConGroup("state_cons", n_solve_states * 2, lower=self.yaw_limits[0] / self.yaw_norm_const, upper=self.yaw_limits[1] / self.yaw_norm_const)
        
        # add objective function
        pyopt_prob.addObj("cost")
        return pyopt_prob
    
    
    def compute_controls(self):
        """
        solve OCP to minimize objective over future horizon
        """
        
        # TODO HIGH only run compute_controls when new amr reading comes in (ie with new timestamp), also in LUT and Greedy, keep track of curent_time independently of measurements_dict
        # current_wind_directions = np.atleast_2d(self.measurements_dict["wind_directions"])
        if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
            pass

        # if self.verbose:
        # 	print(f"self._last_measured_time == {self._last_measured_time}")
        # 	print(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

        self.current_time = self._last_measured_time = self.measurements_dict["time"]

        # current_wind_direction = self.wind_dir_ts[int(self.current_time // self.simulation_dt)]
        if self.wf_source == "floris":
            # current_wind_directions = self.measurements_dict["wind_directions"]
            current_farm_wind_direction = self.measurements_dict["amr_wind_direction"]
            # current_farm_wind_speed = self.measurements_dict["amr_wind_speed"]
            # current_ws_horz = self.measurements_dict["wind_speeds"] * np.sin(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
            # current_ws_vert = self.measurements_dict["wind_speeds"] * np.cos(np.deg2rad(self.measurements_dict["wind_directions"] + 180.0))
        else:
            current_row = self.wind_field_ts.loc[self.wind_field_ts["time"] == self.current_time, :]
            current_ws_horz = np.hstack([current_row[f"ws_horz_{tid}"].values for tid in self.tid2idx_mapping])
            current_ws_vert = np.hstack([current_row[f"ws_vert_{tid}"].values for tid in self.tid2idx_mapping])
            current_wind_directions = 180.0 + np.rad2deg(
                np.arctan2(
                    current_ws_horz, 
                    current_ws_vert
                )
            )
            current_farm_wind_direction = np.mean(current_wind_directions, axis=1)
        
        # if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
        if len(self.measurements_dict["wind_directions"]) == 0 or np.allclose(self.measurements_dict["wind_directions"], 0):
            # yaw angles will be set to initial values
            if self.verbose:
                print("Bad wind direction measurement received, reverting to previous measurement.")

        # need historic measurements for filter or for wind forecast
        if self.use_filt or self.wind_forecast:
            self.historic_measurements["wind_directions"] = np.append(self.historic_measurements["wind_directions"],
                                                            current_farm_wind_direction)[-int((self.lpf_time_const // self.simulation_dt) * 1e3):]
            
        assert np.all(self.wind_field_ts["FreestreamWindDir"][:len(self.historic_measurements["wind_directions"])] == self.historic_measurements["wind_directions"]), "collected historic wind_direction measurements should be equal to actual historci wind_direction measurements in MPC.compute_controls"

        # TODO MISHA this is a patch up for AMR wind initialization problem, also in Greedy/LUT
        current_yaw_setpoints = self.controls_dict["yaw_angles"]
        if (self.current_time - self.init_time).total_seconds() > 0.0:
            # update initial state self.mi_model.initial_state
            # TODO MISHA should be able to get this from measurements dict, also in Greedy/LUT
            self.initial_state = current_yaw_setpoints / self.yaw_norm_const # scaled by yaw limits
            
        if self.verbose:
            logging.info(f"unfiltered current farm wind direction = {current_farm_wind_direction}")
             
        if not (self.current_time < self.lpf_start_time or not self.use_filt):
            # use filtered wind direction and speed
            current_filtered_measurements = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"], self.lpf_alpha)])
            # self.filtered_measurements["wind_directions"].append(current_filtered_measurements[0, -1])
            current_farm_wind_direction = current_filtered_measurements[0, -1]
            
            if self.verbose:
                print(f"filtered farm wind direction = {current_farm_wind_direction}")
        
        # current_wind_speed = self.wind_mag_ts[int(self.current_time // self.simulation_dt)]
        current_farm_wind_speed = self.measurements_dict["amr_wind_speed"]

        if (((self.current_time - self.init_time).total_seconds() % self.controller_dt) == 0.0):
            self.current_freestream_measurements = [
                    current_farm_wind_speed * np.sin(np.deg2rad(current_farm_wind_direction + 180.)),
                    current_farm_wind_speed * np.cos(np.deg2rad(current_farm_wind_direction + 180.))
            ]
            
            # returns n_preview_samples of horizon preview realiztions in the case of stochastic preview type, 
            # else just returns single values for persistent or perfect preview type
            # 
            # returns dictionary of mean, min, max value expected from distribution, in the cahse of stochastic preview type
            time_step = self.seed = int((self.current_time - self.init_time).total_seconds() // self.simulation_dt)
            if "stochastic_interval" in self.wind_preview_type:
                self.wind_preview_intervals, self.wind_preview_interval_probs = self.wind_preview_func(self.current_freestream_measurements, 
                                                                    time_step,
                                                                    seed=self.seed,
                                                                    return_interval_values=True, n_intervals=self.n_wind_preview_intervals,
                                                                    max_std_dev=self.max_std_dev)
                self.wind_preview_samples = self.wind_preview_intervals
            else: # use for extreme constraints, lut warm up
                # TODO HIGH make sure perfect 
                self.wind_preview_intervals, self.wind_preview_interval_probs = self.wind_preview_func(self.current_freestream_measurements, 
                                                                   time_step,
                                                                    seed=self.seed,
                                                                    return_interval_values=True, n_intervals=3, 
                                                                    max_std_dev=self.max_std_dev)

                self.wind_preview_samples, _ = self.wind_preview_func(self.current_freestream_measurements, 
                                                                time_step,
                                                                seed=self.seed,
                                                                return_interval_values=False)
            
            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 1, sharex=True)
                for j in range(self.n_horizon + 1):
                    ax[0].scatter([j] * len(self.wind_preview_samples[f"FreestreamWindMag"][:, j]), self.wind_preview_samples[f"FreestreamWindMag_{j}"])
                    ax[0].scatter([j], self.wind_preview_intervals[f"FreestreamWindMag"][int(self.n_wind_preview_samples // 2), j], marker="s")
                    ax[0].scatter([j], self.wind_preview_intervals[f"FreestreamWindMag"][0, j], marker="s")
                    ax[0].scatter([j], self.wind_preview_intervals[f"FreestreamWindMag"][-1, j], marker="s")
                    ax[1].scatter([j] * len(self.wind_preview_samples[f"FreestreamWindDir"][:, j]), self.wind_preview_samples[f"FreestreamWindDir_{j}"])
                    ax[1].scatter([j], self.wind_preview_intervals[f"FreestreamWindDir"][int(self.n_wind_preview_samples // 2), j], marker="s")
                    ax[1].scatter([j], self.wind_preview_intervals[f"FreestreamWindDir"][0, j], marker="s")
                    ax[1].scatter([j], self.wind_preview_intervals[f"FreestreamWindDir"][-1, j], marker="s")
                    ax[0].set(title="FreestreamWindMag")
                    ax[1].set(title="FreestreamWindDir", xlabel="horizon step")
                ax[0].plot(np.arange(self.n_horizon + 1), self.wind_field_ts["FreestreamWindMag"].iloc[int(self.measurements_dict["time"] // self.simulation_dt):int(self.measurements_dict["time"] // self.simulation_dt) + int(self.controller_dt // self.simulation_dt) * (self.n_horizon + 1):int(self.controller_dt // self.simulation_dt)])
                ax[1].plot(np.arange(self.n_horizon + 1), self.wind_field_ts["FreestreamWindDir"].iloc[int(self.measurements_dict["time"] // self.simulation_dt):int(self.measurements_dict["time"] // self.simulation_dt) + int(self.controller_dt // self.simulation_dt) * (self.n_horizon + 1):int(self.controller_dt // self.simulation_dt)])

            if "slsqp" in self.solver and self.wind_preview_type == "stochastic_sample":
                # tile twice: once for current_yaw_offsets, once for plus_yaw_offsets
                if "zscg" in self.diff_type:
                    n_wind_preview_repeats = 2
                elif "cd" in self.diff_type:
                    n_wind_preview_repeats = 1 + 2 * self.n_turbines #self.n_solve_turbines
                elif "fd" in self.diff_type:
                    n_wind_preview_repeats = 1 + self.n_turbines
                
            elif "slsqp" in self.solver: # and self.wind_preview_type in ["perfect", "persistent"]:
                # tile 2 * self.n_solve_turbines: once for current_yaw_offsets, once for plus_yaw_offsets and once for neg_yaw_offsets for each turbine
                if "cd" in self.diff_type:
                    n_wind_preview_repeats = 1 + 2 * self.n_turbines #self.n_solve_turbines
                elif "fd" in self.diff_type:
                    n_wind_preview_repeats = 1 + self.n_turbines
                else:
                    n_wind_preview_repeats = 1
            else:
                n_wind_preview_repeats = 1

            # TODO is this a valid way to check for amr-wind
            current_powers = self.measurements_dict["turbine_powers"]
            self.offline_status = np.isclose(current_powers, 0.0)

            self.fi._load_floris()

            wd_arr = self.wind_preview_samples[f"FreestreamWindDir"][:, 1:].flatten()
            ws_arr = self.wind_preview_samples[f"FreestreamWindMag"][:, 1:].flatten()
            ti_arr = [self.fi.env.core.flow_field.turbulence_intensities[0]] * self.n_wind_preview_samples * self.n_horizon
            self.fi.env.set(
                wind_directions=np.tile(wd_arr, (n_wind_preview_repeats,)),
                wind_speeds=np.tile(ws_arr, (n_wind_preview_repeats,)),
                turbulence_intensities=np.tile(ti_arr, (n_wind_preview_repeats,))
            )
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
            init_dyn_state_cons = self.opt_sol["states"][:self.n_turbines] - (self.initial_state + self.opt_sol["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_norm_const) * self.controller_dt)
            atol = 1e-3
            # this can sometimes not be satisfied if the iteration limit is exceeded
            if not np.allclose(init_dyn_state_cons, 0.0, atol=atol): #and not np.any(["Successfully" in c["Text"] for c in np.atleast_1d(self.opt_code)]):
                if self.verbose:
                    print(f"nonzero init_dyn_state_cons = {init_dyn_state_cons}")
                else:
                    print(f"Warning: nonzero init_dyn_state_cons")
            
            subsequent_dyn_state_cons = self.opt_sol["states"][self.n_turbines:] - (self.opt_sol["states"][:-self.n_turbines] + self.opt_sol["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_norm_const) * self.controller_dt)

            if not np.allclose(subsequent_dyn_state_cons, 0.0, atol=atol): # this can sometimes not be satisfied if the iteration limit is exceeded
                if self.verbose:
                    print(f"nonzero subsequent_dyn_state_cons = {subsequent_dyn_state_cons}") # self.pyopt_sol_obj
                else:
                    print(f"Warning: nonzero subsequent_dyn_state_cons") 
            
            state_cons = np.array([(self.wind_preview_samples[f"FreestreamWindDir"][0, j + 1] / self.yaw_norm_const) - self.opt_sol["states"][(j * self.n_turbines) + i] for j in range(self.n_horizon) for i in range(self.n_turbines)])
            
            state_con_bools = np.all((state_cons <= (self.yaw_limits[1] / self.yaw_norm_const) + atol) & (state_cons >= (self.yaw_limits[0] / self.yaw_norm_const) - atol))

            if not state_con_bools: # this can sometimes not be satisfied if the iteration limit is exceeded
                if self.verbose:
                    print(f"nonzero state_con_bools = {state_con_bools}")
                else:
                    print(f"Warning: nonzero state_con_bools") 
            
            self.target_controls_dict = {"yaw_angles": list(yaw_star)}
        
            # change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
            setpoint_change = self.target_controls_dict["yaw_angles"] - current_yaw_setpoints
            abs_setpoint_change = np.vstack([np.abs(setpoint_change), 360.0 - np.abs(setpoint_change)]) 
            setpoint_change_idx = np.argmin(abs_setpoint_change, axis=0) # if == 0, need to change within 360 deg, otherwise if == 1 faster to cross 360/0 boundary
            abs_setpoint_change = abs_setpoint_change[setpoint_change_idx, np.arange(self.n_turbines)]
            self.dir_setpoint_change = np.sign(setpoint_change)
            self.dir_setpoint_change[setpoint_change_idx == 1] = -self.dir_setpoint_change[setpoint_change_idx == 1]
            self.target_controls_dict["yaw_angles"] = current_yaw_setpoints + self.dir_setpoint_change * abs_setpoint_change
            self.target_controls_dict["yaw_angles"] = np.rint(np.mod(self.target_controls_dict["yaw_angles"], 360) / self.yaw_increment) * self.yaw_increment
            
        lower_dyn_bounds = np.array(self.target_controls_dict["yaw_angles"])
        lower_dyn_bounds[self.dir_setpoint_change >= 0] = -np.inf
        upper_dyn_bounds = np.array(self.target_controls_dict["yaw_angles"])
        upper_dyn_bounds[self.dir_setpoint_change < 0] = np.inf
        
        self.controls_dict["yaw_angles"] = np.mod(np.clip(
                            self.controls_dict["yaw_angles"] + (self.yaw_rate * self.simulation_dt * self.dir_setpoint_change),
                            lower_dyn_bounds, upper_dyn_bounds), 360)
        

    def zsgd_solve(self):

        # initialize optimization object
        solve_turbine_ids = np.arange(self.n_solve_turbines)
        downstream_turbine_ids = []
        self.warm_start_opt_vars()
        opt_rules = self.generate_opt_rules(solve_turbine_ids, downstream_turbine_ids)
        dyn_state_jac, state_jac = self.con_sens_rules(self.n_solve_turbines)
        sens_rules = self.generate_sens_rules(solve_turbine_ids, downstream_turbine_ids, dyn_state_jac, state_jac)

        state_bounds = (0, 1)
        control_input_bounds = (-1, 1)
        n_solve_states = 2 * self.n_horizon * self.n_solve_turbines
        A_eq = np.zeros(((self.n_horizon * self.n_solve_turbines), n_solve_states)) # *2 for states AND control inputs
        b_eq = np.zeros(((self.n_horizon * self.n_solve_turbines), ))

        # if self.state_con_type == "check_all_samples":
        # 	n_state_cons = 2 * n_solve_states * self.n_wind_preview_samples
        if self.state_con_type == "extreme":
            n_state_cons = 2 * 2 * n_solve_states
        
        # upper and lower bounds for yaw offset for each turbine for each horizon step
        A_ub = np.zeros((n_state_cons, n_solve_states))
        b_ub = np.zeros((n_state_cons, ))
        delta_yaw_coeff = self.controller_dt * (self.yaw_rate / self.yaw_norm_const)

        # TODO vectorize
        state_con_idx = 0
        if self.state_con_type == "check_all_samples":
            wind_dirs = self.wind_preview_samples[f"FreestreamWindDir"][:, 1:].flatten()
            for m in range(self.n_wind_preview_samples):
                for j in range(self.n_horizon):
                    for i, turbine_i in enumerate(solve_turbine_ids):
                        # turbine_control_input_slice = slice((self.n_horizon * self.n_solve_turbines) + i, 
                        #                                     (self.n_horizon * self.n_solve_turbines) + (self.n_solve_turbines * (j + 1)) + i, 
                        #                                     self.n_turbines)
                        
                        # A_ub[state_con_idx, turbine_control_input_slice] = -delta_yaw_coeff
                        # A_ub[state_con_idx + 1, turbine_control_input_slice] = delta_yaw_coeff

                        turbine_control_input_slice = slice((self.n_solve_turbines * j) + i, (self.n_solve_turbines * (j + 1)) + i, self.n_solve_turbines)
                        A_ub[state_con_idx, turbine_control_input_slice] = -1
                        A_ub[state_con_idx + 1, turbine_control_input_slice] = 1

                        b_ub[state_con_idx] = ((self.yaw_limits[1] - wind_dirs[(m * self.n_horizon) + j]) / self.yaw_norm_const)
                        b_ub[state_con_idx + 1] = -((self.yaw_limits[0] - wind_dirs[(m * self.n_horizon) + j]) / self.yaw_norm_const)

                        state_con_idx += 2
        elif self.state_con_type == "extreme":
            # rather than including every sample, could only include most 'extreme' wind directions...
            wind_dirs = self.wind_preview_intervals[f"FreestreamWindDir"][:, 1:]
            
            max_wd = [wind_dirs[-1, j] for j in range(self.n_horizon)]
            min_wd = [wind_dirs[0, j] for j in range(self.n_horizon)]
            
            for wd in [max_wd, min_wd]:    
                for j in range(self.n_horizon):
                    for i, turbine_i in enumerate(solve_turbine_ids):
                        
                        # indices corresponding to control inputs for this turbine, up to this horizon step
                        # turbine_control_input_slice = slice((self.n_horizon * self.n_solve_turbines) + i, 
                        #                                     (self.n_horizon * self.n_solve_turbines) + (self.n_solve_turbines * (j + 1)) + i, 
                        #                                     self.n_turbines)
                        # A_ub[state_con_idx, turbine_control_input_slice] = -delta_yaw_coeff
                        # A_ub[state_con_idx + 1, turbine_control_input_slice] = delta_yaw_coeff

                        turbine_control_input_slice = slice((self.n_solve_turbines * j) + i, (self.n_solve_turbines * (j + 1)) + i, self.n_solve_turbines)
                        A_ub[state_con_idx, turbine_control_input_slice] = -1 # upper bound 
                        A_ub[state_con_idx + 1, turbine_control_input_slice] = 1 # lower bound

                        b_ub[state_con_idx] = ((self.yaw_limits[1] - wd[j]) / self.yaw_norm_const)
                        b_ub[state_con_idx + 1] = -((self.yaw_limits[0] - wd[j]) / self.yaw_norm_const)
                        
                        state_con_idx += 2
        
        for j in range(self.n_horizon):
            for i in range(self.n_solve_turbines):
                current_idx = (self.n_solve_turbines * j) + i
                # delta_yaw = self.simulation_dt * (self.yaw_rate / self.yaw_norm_const) * opt_var_dict["control_inputs"][current_idx]
                A_eq[current_idx, (self.n_horizon * self.n_solve_turbines) + current_idx] = -delta_yaw_coeff
                A_eq[current_idx, current_idx] = 1

                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    # scaled by yaw limit
                    # dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[i] + delta_yaw)]
                    
                    b_eq[current_idx] = self.initial_state[i]
                    
                else:
                    prev_idx = (self.n_solve_turbines * (j - 1)) + i
                    # scaled by yaw limit
                    # dyn_state_cons = dyn_state_cons + [
                    #     opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]\
                    A_eq[current_idx, prev_idx] = -1
                    b_eq[current_idx] = 0
        
        i = 0
        MPC.max_iter = 100
        step_size = 1 / (MPC.max_iter)
        # step_size = 0.9
        acc = MPC.acc
        bounds = [state_bounds for s in range(self.n_horizon * self.n_solve_turbines)] + [control_input_bounds for s in range(self.n_horizon * self.n_solve_turbines)]

        z_next = np.concatenate([self.init_sol["states"], self.init_sol["control_inputs"]])
        opt_var_dict = dict(self.init_sol)
        while i < MPC.max_iter:
            
            funcs, fail = opt_rules(opt_var_dict)
            sens = sens_rules(opt_var_dict, {})

            c = sens["cost"]["states"] + sens["cost"]["control_inputs"]
            # A_ub = np.vstack([1, -1] * np.ones((len(c),)))
            # b_ub = ([1] * (self.n_horizon * self.n_turbines) * 4)

            res = linprog(c=c, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
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

        # yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.simulation_dt * self.opt_sol["control_inputs"][:self.n_solve_turbines]))
        yaw_setpoints = self.opt_sol["states"][:self.n_solve_turbines] * self.yaw_norm_const
        
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment

    
    def sr_solve(self):
        
        # warm-up with previous solution
        self.warm_start_opt_vars()
        if not self.opt_sol:
            self.opt_sol = {k: v.copy() for k, v in self.init_sol.items()}
        
        opt_yaw_setpoints = np.zeros((self.n_horizon, self.n_turbines))
        opt_cost = np.zeros((self.n_horizon,))
        opt_cost_terms = np.zeros((self.n_horizon, 2))

        unconstrained_solve = True
        if unconstrained_solve:
            
            yaw_setpoints = (self.opt_sol["states"] * self.yaw_norm_const).reshape((self.n_horizon, self.n_turbines))
            
            current_yaw_offsets = self.wind_preview_samples[f"FreestreamWindDir"][0, 0] - yaw_setpoints[0, :][np.newaxis, :]
        
            # optimize yaw angles
            yaw_offset_opt = YawOptimizationSRRHC(self.fi.env, 
                            self.yaw_rate, self.controller_dt, self.alpha,
                            n_wind_preview_samples=self.n_wind_preview_samples,
                            wind_preview_type=self.wind_preview_type,
                            n_horizon=self.n_horizon,
                            rated_turbine_power=self.rated_turbine_power,
                            minimum_yaw_angle=self.yaw_limits[0],
                            maximum_yaw_angle=self.yaw_limits[1],
                            yaw_angles_baseline=np.zeros((self.n_turbines,)),
                            Ny_passes=[5, 4],
                            verify_convergence=False)

            opt_yaw_offsets_df = yaw_offset_opt.optimize(current_yaw_offsets=current_yaw_offsets, 
                                                         current_offline_status=self.offline_status,
                                                         wind_preview_interval_probs=self.wind_preview_interval_probs,
                                                         constrain_yaw_dynamics=False, print_progress=self.verbose)
            
            # opt_yaw_setpoints = np.vstack([np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"]) - opt_yaw_offsets_df["yaw_angles_opt"].iloc[j] for j in range(self.n_horizon)])\
            # mean value
            opt_yaw_offsets = np.vstack(opt_yaw_offsets_df["yaw_angles_opt"].values)
            opt_yaw_setpoints = self.wind_preview_samples[f"FreestreamWindDir"][int(self.n_wind_preview_samples // 2), 1:] - opt_yaw_offsets.T
            
            assert np.allclose(self.fi.env.core.flow_field.wind_directions - np.array(self.wind_preview_samples[f"FreestreamWindDir"][:, 1:].flatten()), 0.0), "FLORIS wind directions should come from self.wind_preview_intervals in sr_solve"
            opt_cost = opt_yaw_offsets_df["cost"].to_numpy()
            opt_cost_terms[:, 0] = opt_yaw_offsets_df["cost_states"].to_numpy()
            opt_cost_terms[:, 1] = opt_yaw_offsets_df["cost_control_inputs"].to_numpy()

            # check that all yaw offsets are within limits, possible issue above
            assert np.all((opt_yaw_offsets <= self.yaw_limits[1]) & (opt_yaw_offsets >= self.yaw_limits[0])), "optimized yaw offsets should satisfy upper and lower bounds in sr_solve"
            
            for j in range(self.n_horizon):
                # gamma(k+1) in [gamma(k) - gamma_dot delta_t, gamma(k) + gamma_dot delta_t]
                init_gamma = self.initial_state * self.yaw_norm_const if j == 0 else opt_yaw_setpoints[:, j - 1]
            
                opt_yaw_setpoints[:, j] = np.clip(opt_yaw_setpoints[:, j], init_gamma - self.yaw_rate * self.controller_dt, init_gamma + self.yaw_rate * self.controller_dt)

            # assert np.all([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) <= self.yaw_limits[1] + 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            # assert np.all([(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - opt_yaw_setpoints[j, :]) >= self.yaw_limits[0] - 1e-12 for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
            
            opt_cost = np.sum(opt_cost)
            opt_cost_terms = np.sum(opt_cost_terms, axis=0)
            opt_yaw_setpoints = opt_yaw_setpoints
                    
            
        self.opt_sol = {
            "states": np.array([opt_yaw_setpoints[i, j] / self.yaw_norm_const for j in range(self.n_horizon) for i in range(self.n_turbines)]), 
            "control_inputs": np.array([(opt_yaw_setpoints[i, j] - (opt_yaw_setpoints[i, j - 1] if j > 0 else self.initial_state[i] * self.yaw_norm_const)) * (1 / (self.yaw_rate * self.controller_dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)])
        }
        
        self.opt_code = {"text": None}
        self.opt_cost = opt_cost
        # funcs, _ = self.opt_rules(self.opt_sol)
        self.opt_cost_terms = opt_cost_terms

        return np.rint(opt_yaw_setpoints[:, 0] / self.yaw_increment) * self.yaw_increment
            # set the floris object with the predicted wind magnitude and direction at this time-step in the horizon

    def warm_start_opt_vars(self):
        self.init_sol = {"states": [], "control_inputs": []}
        
        if self.warm_start == "previous":
            current_time = self.measurements_dict["time"]
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
                next_yaw_setpoints = (self.yaw_IC / self.yaw_norm_const) * np.ones((self.n_horizon * self.n_turbines,))
                current_control_inputs = np.zeros((self.n_horizon * self.n_turbines,))
                self.init_sol["states"] = next_yaw_setpoints
                self.init_sol["control_inputs"] = current_control_inputs

        elif self.warm_start == "lut":
            # TODO HIGH filtered wind dir not in use here? do
            target_yaw_offsets = self.ctrl_lut.wake_steering_interpolant(self.wind_preview_intervals[f"FreestreamWindDir"][int(self.wind_preview_intervals[f"FreestreamWindDir"].shape[0] // 2), :-1], 
                                                       self.wind_preview_intervals[f"FreestreamWindMag"][int(self.wind_preview_intervals[f"FreestreamWindDir"].shape[0] // 2), :-1])
            target_yaw_setpoints = np.rint((np.atleast_2d([self.wind_preview_intervals[f"FreestreamWindDir"][int(self.wind_preview_intervals[f"FreestreamWindDir"].shape[0] // 2), :-1]]).T - target_yaw_offsets) / self.yaw_increment) * self.yaw_increment
            
            self.init_sol["states"] = target_yaw_setpoints.flatten() / self.yaw_norm_const
            self.init_sol["control_inputs"] = (self.init_sol["states"] - self.opt_sol["states"]) * (self.yaw_norm_const / (self.yaw_rate * self.controller_dt))
        
        elif self.warm_start == "greedy":
            self.init_sol["states"] = np.concatenate([(self.wind_preview_intervals[f"FreestreamWindDir"][int(self.wind_preview_intervals[f"FreestreamWindDir"].shape[0] // 2), :-1] / self.yaw_norm_const) for i in range(self.n_turbines)])
            self.init_sol["control_inputs"] = (self.init_sol["states"] - self.opt_sol["states"]) * (self.yaw_norm_const / (self.yaw_rate * self.controller_dt))

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
        wd = self.wind_preview_samples["FreestreamWindDir"][0, 0]
        layout_x_rot = (
            np.cos(np.deg2rad(wd + 180.0)) * layout_y
            + np.sin(np.deg2rad(wd + 180.0)) * layout_x
        )
        turbines_ordered = np.argsort(layout_x_rot)

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
            
            downstream_turbine_ids = []
            for ds_group_idx in range(turbine_group_idx + 1, n_solve_turbine_groups):
                downstream_turbine_ids += grouped_turbines_ordered[ds_group_idx]

            n_solve_turbines = len(solve_turbine_ids)
            solutions.append(self.solve_turbine_group(solve_turbine_ids, downstream_turbine_ids))

            # update self.opt_sol with most recent solutions for all turbines
            for opt_var_idx, solve_turbine_idx in enumerate(solve_turbine_ids):
                self.opt_sol["states"][solve_turbine_idx::self.n_turbines] = np.array(solutions[turbine_group_idx].xStar["states"])[opt_var_idx::n_solve_turbines]
                self.opt_sol["control_inputs"][solve_turbine_idx::self.n_turbines] = np.array(solutions[turbine_group_idx].xStar["control_inputs"])[opt_var_idx::n_solve_turbines]
            self.opt_code.append(solutions[turbine_group_idx].optInform)
            self.opt_cost = ((self.opt_cost * turbine_group_idx) + solutions[turbine_group_idx].fStar) / (turbine_group_idx + 1)
            
        yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.controller_dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        
        return np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment

    
    def slsqp_solve(self):
        
        # warm start Vars by reinitializing the solution from last time-step self.mi_model.states, set self.init_sol
        self.warm_start_opt_vars()

        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (j * self.n_turbines) + i
                # next_idx = ((j + 1) * self.n_turbines) + i
                if self.run_custom_sens:
                    self.pyopt_prob.variables["states"][current_idx].value \
                        = self.init_sol["states"][current_idx]
                    self.pyopt_prob.variables["control_inputs"][current_idx].value \
                    = self.init_sol["control_inputs"][current_idx]
                
                if self.run_brute_force_sens:
                    self.pyopt_prob_nosens.variables["states"][current_idx].value \
                        = self.init_sol["states"][current_idx]
                    self.pyopt_prob_nosens.variables["control_inputs"][current_idx].value \
                        = self.init_sol["control_inputs"][current_idx]
            
        if True and self.run_brute_force_sens and self.run_custom_sens:
            
            self.optimizer.optProb = self.pyopt_prob_nosens
            self.optimizer.optProb.finalize()
            self.optimizer._setInitialCacheValues()
            self.optimizer._setSens(sens=None, sensStep=self.nu, sensMode=None)
            grad_nosens = self.optimizer.sens
            # sol_nosens = self.optimizer(self.pyopt_prob_nosens) #, storeHistory=f"{os.path.dirname(whoc.__file__)}/floris_case_studies/optimizer_histories/fd_sens_{current_time}.hst")
            
            self.optimizer.optProb = self.pyopt_prob
            self.optimizer.optProb.finalize()
            self.optimizer._setInitialCacheValues()
            self.optimizer._setSens(None, None, None)
            grad_sens = self.optimizer.sens
            

            # np.random.seed(0)
            # sample_sol = {"states": np.random.uniform(0, 1, self.init_sol["states"].shape), "control_inputs": np.random.uniform(-1, 1, self.init_sol["control_inputs"].shape)}
            # sample_sol = {"states": 0.7 * np.ones_like(self.init_sol["states"]), "control_inputs": 0.5 * np.ones_like(self.init_sol["control_inputs"])}
            sample_sol = {"states": np.random.choice([0.7, 0.7, 0.7], size=self.init_sol["states"].shape), "control_inputs": np.random.uniform(-1, 1, self.init_sol["control_inputs"].shape)}
            #funcs1, fail = self.optimizer.optProb.objFun(sample_sol)
            funcs = {"state_cons": [0] * self.n_turbines * self.n_horizon * 2, "dyn_state_cons": [0] * self.n_turbines * self.n_horizon, "cost_states": 0, "cost_control_inputs": 0, "cost": 0}
            self._last_yaw_setpoints = sample_sol["states"][:self.n_turbines] * self.yaw_norm_const
            self._last_solve_turbine_ids = None
            grad_sens_res = grad_sens(sample_sol, funcs) # no computation bc update_norm_powers computed in line above
            self._last_yaw_setpoints = sample_sol["states"][:self.n_turbines] * self.yaw_norm_const
            self._last_solve_turbine_ids = None
            grad_nosens_res = grad_nosens(sample_sol, funcs)
            np.vstack([grad_nosens_res[0]["cost"]["control_inputs"], np.array(grad_sens_res["cost"]["control_inputs"])]).T	
            np.vstack([grad_nosens_res[0]["cost"]["states"], np.array(grad_sens_res["cost"]["states"])]).T
            # False for  0,  1,  2,  3,  5,  7, 10, 12, 13 with states part of cost only, no Falses for control inputs only
            # print(np.where(~np.isclose(grad_nosens_res[0]["cost"]["states"], np.array(grad_sens_res["cost"]["states"]), atol=1e-5)))
            # no Falses with states part of cost only, no Falses for control inputs only
            print(f"\nwind_preview_type - {self.wind_preview_type} - {self.n_wind_preview_samples}, alpha = {self.alpha}")
            print("state sens diff", np.where(~np.isclose(grad_nosens_res[0]["cost"]["states"], np.array(grad_sens_res["cost"]["states"]))))
            print("ctrl_input sens diff", np.where(~np.isclose(grad_nosens_res[0]["cost"]["control_inputs"], np.array(grad_sens_res["cost"]["control_inputs"]))))
        
        if self.run_custom_sens:
            sol = self.optimizer(self.pyopt_prob) #, storeHistory=f"{os.path.dirname(whoc.__file__)}/floris_case_studies/optimizer_histories/custom_sens_{current_time}.hst") # timeLimit=self.simulation_dt) #, sens=sens_rules) #, sensMode='pgc')
        else:
            sol = self.optimizer(self.pyopt_prob_nosens, sensStep=self.nu)
        
        if self.run_brute_force_sens and self.run_custom_sens:
            sol_nosens = self.optimizer(self.pyopt_prob_nosens, sensStep=self.nu)
            s_diff = np.vstack([sol.xStar["states"] - self.init_sol["states"], sol_nosens.xStar["states"] - self.init_sol["states"]]).T
            s_diff_dir = s_diff / np.abs(s_diff)
            np.vstack([sol.xStar["states"], sol_nosens.xStar["states"]]).T * self.yaw_norm_const
            (sol.xStar["states"] - sol_nosens.xStar["states"]) * self.yaw_norm_const
            
            # print(np.where(~np.isclose(grad_nosens_res[0]["cost"]["states"], np.array(grad_sens_res["cost"]["states"]), atol=1e-3)))	
            print(f"\nwind_preview_type - {self.wind_preview_type} - {self.n_wind_preview_samples}, alpha = {self.alpha}")
            print("max yaw_setpoint diff = ", np.max(np.abs(s_diff[:, 0] - s_diff[:, 1]) * self.yaw_norm_const))
            print(f"sol.fStar vs. sol_nosens.fStar = {100 * (sol.fStar - sol_nosens.fStar) / sol_nosens.fStar} %")
            # assert np.all(s_diff_dir[:, 0] == s_diff_dir[:, 1])
            # Note: same gradient, solution for perfect/persistent wind_preview_type
            # assert np.all(c_dir[:, 0] == c_dir[:, 1])
        
        # check if optimal solution is on boundary anywhere.
        # assert not any(var.value > var.upper or var.value < var.lower for var in sol.variables["states"]) or any(var.value > var.upper or var.value < var.lower for var in sol.variables["control_inputs"]), "optimization variables should satisfy upper bounds in slsqp_solve"
        # print(f"alpha = {self.alpha}, wind_preview_type = {self.wind_preview_type}")
        # print(f"state cost term = {self.opt_cost_terms[0]}, ctrl_inp cost term = {self.opt_cost_terms[1]}")
        # sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens="FD")
        self.pyopt_sol_obj = sol
        self.opt_sol = {k: v[:] for k, v in sol.xStar.items()}
        self.opt_code = sol.optInform

        # reshape to 2 (min and max wind direction) x n_horizon x n_turbines
        self.state_cons_activated["lower"] = np.where(np.isclose(sol.constraints["state_cons"].value, sol.constraints["state_cons"].lower).reshape((2, self.n_horizon, self.n_solve_turbines)))
        self.state_cons_activated["upper"] = np.where(np.isclose(sol.constraints["state_cons"].value, sol.constraints["state_cons"].upper).reshape((2, self.n_horizon, self.n_solve_turbines)))
        
        if self.opt_code["value"]:
            print(f"Warning, nonzero inform code: {self.opt_code['text']}")
        # self.opt_cost = sol.fStar
        # assert sum(self.opt_cost_terms) == self.opt_cost, "sum of self.opt_cost_terms should equal self.opt_cost in slsqp_solve"
        # solution is scaled by yaw limit
        # yaw_setpoints = ((self.initial_state * self.yaw_norm_const) + (self.yaw_rate * self.simulation_dt * self.opt_sol["control_inputs"][:self.n_turbines]))
        yaw_setpoints = self.opt_sol["states"][:self.n_turbines] * self.yaw_norm_const
        rounded_yaw_setpoints = np.mod(np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment, 360)
        return rounded_yaw_setpoints

    
    def solve_turbine_group(self, solve_turbine_ids, downstream_turbine_ids):
        # solve_turbine_ids = grouped_turbines_ordered[turbine_group_idx]
        n_solve_turbines = len(solve_turbine_ids)

        pyopt_prob = self.setup_slsqp_solver(solve_turbine_ids, downstream_turbine_ids)
        # sens_rules = self.generate_sens_rules(solve_turbine_ids)
        
        # setup pyopt problem to consider 2 optimization variables for this turbine and set all others as fixed parameters
        # opt_var_indices = [i + (j * n_solve_turbines) for j in range(self.n_horizon) for i in solve_turbine_ids]
        # pyopt_prob.variables["states"] = self.init_sol["states"][opt_var_indices]
        # pyopt_prob.variables["control_inputs"] = self.init_sol["control_inputs"][opt_var_indices]
        for j in range(self.n_horizon):
            for opt_var_idx, solve_turbine_idx in enumerate(solve_turbine_ids):
                pyopt_prob.variables["states"][(j * n_solve_turbines) + opt_var_idx].value \
                    = self.init_sol["states"][(j * self.n_turbines) + solve_turbine_idx]
                
                pyopt_prob.variables["control_inputs"][(j * n_solve_turbines) + opt_var_idx].value \
                    = self.init_sol["control_inputs"][(j * self.n_turbines) + solve_turbine_idx]
        
        # solve problem based on self.opt_sol
        sol = self.optimizer(pyopt_prob) #, timeLimit=self.simulation_dt) #, sens=sens_rules) #, sensMode='pgc')
        return sol

    def generate_opt_rules(self, solve_turbine_ids, downstream_turbine_ids, compute_constraints=True, compute_derivatives=True):
        n_solve_turbines = len(solve_turbine_ids)
        
        def opt_rules(opt_var_dict):

            funcs = {}
            if self.solver == "sequential_slsqp":
                states = np.array(self.opt_sol["states"])
                control_inputs = np.array(self.opt_sol["control_inputs"])
                
                for opt_var_id, turbine_id in enumerate(solve_turbine_ids):
                    states[turbine_id::self.n_turbines] = opt_var_dict["states"][opt_var_id::n_solve_turbines]
                    control_inputs[turbine_id::self.n_turbines] = opt_var_dict["control_inputs"][opt_var_id::n_solve_turbines]

                # the yaw setpoints for the future horizon (current/iniital state is known and not an optimization variable)
                # yaw_setpoints = np.array([[states[(self.n_turbines * j) + i] 
                # 						for i in range(self.n_turbines)] for j in range(self.n_horizon)]) * self.yaw_norm_const
                yaw_setpoints = states.reshape((self.n_horizon, self.n_turbines)) * self.yaw_norm_const
            
            else:
                # the yaw setpoints for the future horizon (current/iniital state is known and not an optimization variable)
                yaw_setpoints = opt_var_dict["states"].reshape((self.n_horizon, self.n_turbines)) * self.yaw_norm_const
            
            # plot_distribution_samples(pd.DataFrame(wind_preview_samples), self.n_horizon)
            # derivative of turbine power output with respect to yaw angles
            
            if compute_constraints:
                if self.use_state_cons:
                    # TODO HIGH store information regarding activation of constraints
                    if self.state_con_type == "extreme":
                        funcs["state_cons"] = self.state_rules(opt_var_dict, 
                                                            {
                                                                "wind_direction": self.wind_preview_intervals[f"FreestreamWindDir"][:, 1:]
                                                                }, yaw_setpoints, solve_turbine_ids)
                    else:
                        funcs["state_cons"] = self.state_rules(opt_var_dict, 
                                                            {
                                                                "wind_direction": self.wind_preview_samples[f"FreestreamWindDir"][:, 1:]
                                                                }, yaw_setpoints, solve_turbine_ids)
                
                if self.use_dyn_state_cons:
                    funcs["dyn_state_cons"] = self.dyn_state_rules(opt_var_dict, solve_turbine_ids)
            # send yaw angles 

            # compute power based on sampling from wind preview
            self.update_norm_turbine_powers(yaw_setpoints, solve_turbine_ids, downstream_turbine_ids, compute_derivatives) 
            
            # weighted mean based on probabilities of samples used
            # outer sum over all samples and horizons, inner sum over all turbines
            if self.wind_preview_type == "stochastic_sample":
                funcs["cost_states"] = np.sum(self.norm_turbine_powers**2) * (-0.5 * self.Q / self.n_wind_preview_samples)
            elif "stochastic_interval" in self.wind_preview_type:
                funcs["cost_states"] = np.sum(self.norm_turbine_powers**2 * self.wind_preview_interval_probs[:, :, np.newaxis]) * (-0.5 * self.Q)
            else:
                funcs["cost_states"] = np.sum(self.norm_turbine_powers**2) * (-0.5 * self.Q)
                # funcs["cost_states"] = np.einsum("sht, sh...", -0.5*self.norm_turbine_powers**2 * self.Q, self.wind_preview_interval_probs[:, :, np.newaxis], [0])
                # funcs["cost_states"] = np.dot(np.sum(-0.5*self.norm_turbine_powers**2 * self.Q, axis=2), self.wind_preview_interval_probs)
            
            funcs["cost_control_inputs"] = np.sum((opt_var_dict["control_inputs"])**2) * 0.5 * self.R

            funcs["cost"] = funcs["cost_states"] + funcs["cost_control_inputs"]
            
            if self.solver == "sequential_slsqp":
                self.opt_cost_terms[0] += funcs["cost_states"]
                self.opt_cost_terms[1] += funcs["cost_control_inputs"]
                # self.opt_cost = funcs["cost"]
            else:
                self.opt_cost_terms = [funcs["cost_states"], funcs["cost_control_inputs"]]
                self.opt_cost = sum(self.opt_cost_terms)
            
            fail = False
            
            return funcs, fail
        
        # opt_rules.__qualname__ = "opt_rules"
        return opt_rules

    
    def update_norm_turbine_powers(self, yaw_setpoints, solve_turbine_ids, downstream_turbine_ids, compute_derivatives=True):
        # no need to update norm_turbine_powers if yaw_setpoints have not changed

        if (self._last_yaw_setpoints is not None) and np.allclose(yaw_setpoints, self._last_yaw_setpoints) and np.all(solve_turbine_ids == self._last_solve_turbine_ids):
            return None
        
        self._last_yaw_setpoints = np.array(yaw_setpoints)
        self._last_solve_turbine_ids = np.array(solve_turbine_ids)
        n_solve_turbines = len(solve_turbine_ids)
        influenced_turbine_ids = solve_turbine_ids + downstream_turbine_ids
        n_influenced_turbines = len(influenced_turbine_ids)

        # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
        n_wind_samples = self.n_wind_preview_samples * self.n_horizon
        # np.vstack([(self.wind_preview_intervals[f"FreestreamWindDir"][m, j+1] - yaw_setpoints[j, :]) for m in range(self.n_wind_preview_samples) for j in range(self.n_horizon)])
        current_yaw_offsets = (self.wind_preview_samples[f"FreestreamWindDir"][:, 1:, np.newaxis] - yaw_setpoints).reshape((n_wind_samples, self.n_turbines))
        current_yaw_offsets = current_yaw_offsets % 360.0
        current_yaw_offsets[current_yaw_offsets > 180.0] = -(360.0 - current_yaw_offsets[current_yaw_offsets > 180.0])
        current_yaw_offsets[current_yaw_offsets < -180.0] = (360.0 + current_yaw_offsets[current_yaw_offsets < -180.0])
        
        
        if compute_derivatives:
            if self.wind_preview_type == "stochastic_sample" and "zscg" in self.diff_type:
                np.random.seed(self.seed)
                self.stochastic_sample_u = np.random.normal(loc=0.0, scale=self.stochastic_sample_u_scale, size=(n_wind_samples, n_solve_turbines))
                # u = np.random.choice([-1, 1], size=(n_wind_samples, n_solve_turbines))
                
                # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
                
                if self.solver == "sequential_slsqp":
                    masked_u = np.zeros((n_wind_samples, self.n_turbines))
                    masked_u[:, solve_turbine_ids] = self.stochastic_sample_u
                    plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * masked_u
                else:
                    plus_yaw_offsets = current_yaw_offsets - self.nu * self.yaw_norm_const * self.stochastic_sample_u

                plus_yaw_offsets = plus_yaw_offsets % 360.0
                plus_yaw_offsets[plus_yaw_offsets > 180.0] = -(360.0 - plus_yaw_offsets[plus_yaw_offsets > 180.0])
                plus_yaw_offsets[plus_yaw_offsets < -180.0] = (360.0 + plus_yaw_offsets[plus_yaw_offsets < -180.0])
                
                all_yaw_offsets = np.vstack([current_yaw_offsets, plus_yaw_offsets])
            
            # perturb each state (each yaw angle) by +/= nu to estimate derivative of all turbines power output for a variation in each turbines yaw offset
            # if any yaw offset are out of the [-90, 90] range, then the power output of all turbines will be nan. clip to av
            # we subtract plus change since current_yaw_offsets = wind dir - yaw setpoints
            # we add negative since current_yaw_offsets = wind dir - yaw setpoints
            
            elif "cd" in self.diff_type:
                change_mask = np.array([-1] * n_wind_samples + [1] * n_wind_samples)
                no_change_mask = np.zeros((2 * n_wind_samples,))
                mask = np.vstack([np.zeros((n_wind_samples, self.n_turbines))] + [np.vstack([change_mask if (i == ii and ii in solve_turbine_ids) else no_change_mask for ii in range(self.n_turbines)]).T for i in range(self.n_turbines)])
                all_yaw_offsets = np.tile(current_yaw_offsets, ((2 * self.n_turbines) + 1, 1)) + mask * self.nu * self.yaw_norm_const
            elif "fd" in self.diff_type:
                change_mask = np.array([-1] * n_wind_samples)
                no_change_mask = np.zeros((n_wind_samples,))
                mask = np.vstack([np.zeros((n_wind_samples, self.n_turbines))] + [np.vstack([change_mask if (i == ii and ii in solve_turbine_ids) else no_change_mask for ii in range(self.n_turbines)]).T for i in range(self.n_turbines)])
                all_yaw_offsets = np.tile(current_yaw_offsets, (self.n_turbines + 1, 1)) + mask * self.nu * self.yaw_norm_const
        else:
        # if (not compute_derivatives) or (self.diff_type == "central_diff"): 
            if self.run_brute_force_sens:
                all_yaw_offsets = np.tile(current_yaw_offsets, (int(self.fi.env.n_findex // current_yaw_offsets.shape[0]), 1)) # TODO only needed if computing w/ central_diff and custom in same run
            else:
                all_yaw_offsets = current_yaw_offsets

        self.fi.env.set_operation(
            yaw_angles=np.clip(all_yaw_offsets, -self.clip_value, self.clip_value), #, dtype="float16"), # must provide cushion because added_yaw brings in gauss.py it over 90.0
            disable_turbines=self.offline_status,
        )
        
        self.fi.env.run()
        
        all_yawed_turbine_powers = self.fi.env.get_turbine_powers()[:, influenced_turbine_ids]
        all_yaw_offsets = all_yaw_offsets[:, influenced_turbine_ids]

        # normalize power by no yaw output
        # yawed_turbine_powers = all_yawed_turbine_powers[:current_yaw_offsets.shape[0], :]

        if self.decay_type != "none":
            
            # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
            # all_yaw_offsets[n_wind_samples:, :].shape[0]
            # perturbed_mask = np.zeros((all_yaw_offsets.shape[0],1), dtype=bool)
            # perturbed_mask[n_wind_samples:, 0] = True
            # perturbed_mask[:, 0] = True
            pos_idx = (all_yaw_offsets > self.yaw_limits[1]) #& perturbed_mask
            multi_clip_row_idx = pos_idx.sum(axis=1) > 1
            pos_idx[multi_clip_row_idx, :] = False
            pos_idx[multi_clip_row_idx, np.argmax(all_yaw_offsets[multi_clip_row_idx, :], axis=1)] = True
            pos_decay_idx = np.broadcast_to(pos_idx.any(1)[:, np.newaxis], pos_idx.shape) if self.decay_all else pos_idx#& decay_mask

            neg_idx = (all_yaw_offsets < self.yaw_limits[0])# & perturbed_mask
            multi_clip_row_idx = neg_idx.sum(axis=1) > 1
            neg_idx[multi_clip_row_idx, :] = False
            neg_idx[multi_clip_row_idx, np.argmin(all_yaw_offsets[multi_clip_row_idx, :], axis=1)] = True
            neg_decay_idx = np.broadcast_to(neg_idx.any(1)[:, np.newaxis], neg_idx.shape) if self.decay_all else neg_idx #& decay_mask
            # TODO can I merge this into single operation for pos and neg
            # oob_yaw_range = np.vstack([all_yaw_offsets[pos_idx] - self.yaw_limits[1], self.yaw_limits[0] - all_yaw_offsets[neg_idx]])
            if self.decay_type == "cosine":
                pos_decay = np.cos(self.decay_factor * (all_yaw_offsets[pos_idx] - self.yaw_limits[1]))
                neg_decay = np.cos(self.decay_factor * (self.yaw_limits[0] - all_yaw_offsets[neg_idx]))
                # pos_neg_decay = np.cos(self.decay_factor * oob_yaw_range)
            elif self.decay_type == "exp":
                try:
                    pos_decay = np.exp(self.decay_factor * (all_yaw_offsets[pos_idx] - self.yaw_limits[1]))
                except FloatingPointError:
                    pos_decay = np.array([0])
                try:
                    neg_decay = np.exp(self.decay_factor * (self.yaw_limits[0] - all_yaw_offsets[neg_idx]))
                except FloatingPointError:
                    neg_decay = np.array([0])

                # pos_neg_decay = np.exp(-self.decay_factor * oob_yaw_range)
            elif self.decay_type == "linear":
                pos_decay = -self.decay_factor * (all_yaw_offsets[pos_idx] - self.yaw_limits[1]) + 1.0
                neg_decay = -self.decay_factor * (self.yaw_limits[0] - all_yaw_offsets[neg_idx]) + 1.0
                # pos_neg_decay = self.decay_factor * oob_yaw_range + 1.0
            elif self.decay_type == "zero":
                pos_decay = np.array([0])
                neg_decay = np.array([0])
                #pos_neg_decay = np.array([0])
            
            if self.decay_all:
                # all_yawed_turbine_powers[pos_idx.any(1) & neg_idx.any(1), :] = all_yawed_turbine_powers[pos_idx.any(1) & neg_idx.any(1), :] * np.minimum(pos_decay[:, np.newaxis], neg_decay[:, np.newaxis]) # TODO test
                all_yawed_turbine_powers[pos_idx.any(1), :] = all_yawed_turbine_powers[pos_idx.any(1), :] * pos_decay[:, np.newaxis]
                all_yawed_turbine_powers[neg_idx.any(1), :] = all_yawed_turbine_powers[neg_idx.any(1), :] * neg_decay[:, np.newaxis]
                # all_yawed_turbine_powers[pos_idx.any(1), :] = all_yawed_turbine_powers[pos_idx.any(1), :] * pos_neg_decay[:, np.newaxis]	
            else:
                all_yawed_turbine_powers[pos_decay_idx] = all_yawed_turbine_powers[pos_decay_idx] * pos_decay
                all_yawed_turbine_powers[neg_decay_idx] = all_yawed_turbine_powers[neg_decay_idx] * neg_decay

        self.norm_turbine_powers = all_yawed_turbine_powers[:n_wind_samples, :].copy() / self.rated_turbine_power
        self.norm_turbine_powers = np.reshape(self.norm_turbine_powers, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines))

        if compute_derivatives:
            if self.wind_preview_type == "stochastic_sample":
                # should compute derivative of each power of each turbine wrt state (yaw angle) of each turbine
                if self.diff_type == "chain_zscg":
                    norm_turbine_power_diff = (all_yawed_turbine_powers[n_wind_samples:, :] - all_yawed_turbine_powers[:n_wind_samples, :]) / self.rated_turbine_power	
                    self.norm_turbine_powers_states_drvt = np.reshape(np.einsum("ia, ib->iab", norm_turbine_power_diff / self.nu, self.stochastic_sample_u), (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))
                elif self.diff_type == "direct_zscg":
                    self.plus_norm_turbine_powers = all_yawed_turbine_powers[n_wind_samples:, :] / self.rated_turbine_power
                elif self.diff_type == "chain_cd":
                    self.norm_turbine_powers_states_drvt = (
                        np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) 
                            - np.dstack([all_yawed_turbine_powers[i, :] for i in self.neg_slices])) / (self.rated_turbine_power * 2 * self.nu)
                    self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))
                
                elif self.diff_type == "chain_fd":
                    self.norm_turbine_powers_states_drvt = (
                        np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) 
                            - all_yawed_turbine_powers[:n_wind_samples, :, np.newaxis]) / (self.rated_turbine_power * self.nu)
                    self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))

                elif self.diff_type == "direct_cd":
                    # adding to yaw setpoints, subtracting from yaw offsets
                    self.plus_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) / self.rated_turbine_power	
                    # subtracting from yaw setpoints, adding to yaw offsets
                    self.neg_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.neg_slices])	/ self.rated_turbine_power
                
                elif self.diff_type == "direct_fd":
                    self.plus_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) / self.rated_turbine_power
    
                
            else:
                if self.diff_type == "chain_cd":
                    self.norm_turbine_powers_states_drvt = (
                        np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) 
                            - np.dstack([all_yawed_turbine_powers[i, :] for i in self.neg_slices])) / (self.rated_turbine_power * 2 * self.nu)
                    self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))
                
                elif self.diff_type == "chain_fd":
                    self.norm_turbine_powers_states_drvt = (
                        np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) 
                            - all_yawed_turbine_powers[:n_wind_samples, :, np.newaxis]) / (self.rated_turbine_power * self.nu)
                    self.norm_turbine_powers_states_drvt = np.reshape(self.norm_turbine_powers_states_drvt, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))

                if self.diff_type == "direct_cd":
                    self.plus_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) / self.rated_turbine_power	
                    self.neg_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.neg_slices])	/ self.rated_turbine_power
                
                elif self.diff_type == "direct_fd":
                    self.plus_norm_turbine_powers = np.dstack([all_yawed_turbine_powers[i, :] for i in self.plus_slices]) / self.rated_turbine_power

                # NOTE for chain_cd diff_type, if positive yaw offsets and negative yaw offsets are out of range for a turbine and a given wind sample row, then both will be decayed, if both are decayed to zero, then the gradient will show that perturbing that turbine's yaw offset result in no power change for that turbine...

        
    def generate_sens_rules(self, solve_turbine_ids, downstream_turbine_ids, dyn_state_jac, state_jac):
        def sens_rules(opt_var_dict, obj_con_dict):
            # use last_yaw_setpoints here bc need all turbines even if only solving for subset, so can't pull from opt_var_dict
            n_solve_turbines = len(solve_turbine_ids)
            influenced_turbine_ids = solve_turbine_ids + downstream_turbine_ids
            n_influenced_turbines = len(influenced_turbine_ids)
            # yaw_setpoints = opt_var_dict["states"].reshape((self.n_horizon, )) * self.yaw_norm_const
            self.update_norm_turbine_powers(self._last_yaw_setpoints, solve_turbine_ids, downstream_turbine_ids, compute_derivatives=True)
            # self.update_norm_turbine_powers(yaw_setpoints, solve_turbine_ids, downstream_turbine_ids, compute_derivatives=True)

            sens = {"cost": {"states": [], "control_inputs": []}}
            
            if self.use_state_cons:
                sens["state_cons"] = {"states": [], "control_inputs": []}

            if self.use_dyn_state_cons:
                sens["dyn_state_cons"] = {"states": [], "control_inputs": []}
            
            # compute power derivative based on sampling from wind preview with respect to changes to the state/control input of this turbine/horizon step
            # 		 using derivative: power of each turbine wrt each turbine's yaw setpoint, summing over terms for each turbine
            # 		 states part of cost

            if self.wind_preview_type == "stochastic_sample": # np.mean(np.sum(-0.5*self.norm_turbine_powers**2 * self.Q, axis=(1, 2)))
                if self.diff_type == "chain_zscg":
                    sens["cost"]["states"] = np.einsum("sht,shti->hi", self.norm_turbine_powers, self.norm_turbine_powers_states_drvt).flatten() * (-self.Q / self.n_wind_preview_samples)
                elif self.diff_type == "direct_zscg":
                    sens["cost"]["states"] = np.einsum("sht,shi->hi", np.reshape(self.plus_norm_turbine_powers**2, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines)) - 
                                                                       self.norm_turbine_powers**2, 
                                                                       np.reshape(self.stochastic_sample_u, (self.n_wind_preview_samples, self.n_horizon, n_solve_turbines))).flatten() * (-0.5 * self.Q) / (self.nu * self.n_wind_preview_samples)
                elif "chain" in self.diff_type:
                    sens["cost"]["states"] = np.einsum("sht,shti->hi", self.norm_turbine_powers, self.norm_turbine_powers_states_drvt).flatten() * (-self.Q / self.n_wind_preview_samples)
                elif "direct" in self.diff_type:
                    if self.diff_type == "direct_cd":
                        sens["cost"]["states"] = np.einsum("shti->hi", np.reshape(self.plus_norm_turbine_powers**2 - self.neg_norm_turbine_powers**2, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))).flatten() * (-0.5 * self.Q) / (2 * self.nu * self.n_wind_preview_samples)
                    elif self.diff_type == "direct_fd":
                        sens["cost"]["states"] = np.einsum("shti->hi", np.reshape(self.plus_norm_turbine_powers**2 - 
                                                                   np.reshape(self.norm_turbine_powers, (self.n_wind_preview_samples * self.n_horizon, n_influenced_turbines))[:, :, np.newaxis]**2, 
                                                                   (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))).flatten() * (-0.5 * self.Q) / (self.nu * self.n_wind_preview_samples)
            elif "stochastic_interval" in self.wind_preview_type: # np.sum(self.norm_turbine_powers**2 * self.wind_preview_interval_probs[:, :, np.newaxis]) * (-0.5 * self.Q)
                if "chain" in self.diff_type:
                    sens["cost"]["states"] = np.einsum("sht,shti,sh->hi", self.norm_turbine_powers, self.norm_turbine_powers_states_drvt, self.wind_preview_interval_probs).flatten() * (-self.Q)
                elif "direct" in self.diff_type:
                    if self.diff_type == "direct_cd":
                        sens["cost"]["states"] = np.einsum("shti,sh->hi", np.reshape(self.plus_norm_turbine_powers**2 - self.neg_norm_turbine_powers**2, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines)), 
                                                                          self.wind_preview_interval_probs).flatten() * (-0.5 * self.Q) / (2 * self.nu)
                    elif self.diff_type == "direct_fd":
                        sens["cost"]["states"] = np.einsum("shti,sh->hi", np.reshape(self.plus_norm_turbine_powers**2 - 
                                                                   np.reshape(self.norm_turbine_powers, (self.n_wind_preview_samples * self.n_horizon, n_influenced_turbines))[:, :, np.newaxis]**2, 
                                                                   (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines)), 
                                                                   self.wind_preview_interval_probs).flatten() * (-0.5 * self.Q) / (self.nu)
            else:
                if "chain" in self.diff_type:
                    sens["cost"]["states"] = np.einsum("sht,shti->hi", self.norm_turbine_powers, self.norm_turbine_powers_states_drvt).flatten() * (-self.Q)
                elif "direct" in self.diff_type:
                    if self.diff_type == "direct_cd":
                        sens["cost"]["states"] = np.einsum("shti->hi", np.reshape(self.plus_norm_turbine_powers**2 - self.neg_norm_turbine_powers**2, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))).flatten() * (-0.5 * self.Q) / (2 * self.nu)
                    elif self.diff_type == "direct_fd":
                        sens["cost"]["states"] = np.einsum("shti->hi", np.reshape(self.plus_norm_turbine_powers**2 - np.reshape(self.norm_turbine_powers, (self.n_wind_preview_samples * self.n_horizon, n_influenced_turbines))[:, :, np.newaxis]**2, (self.n_wind_preview_samples, self.n_horizon, n_influenced_turbines, n_solve_turbines))).flatten() * (-0.5 * self.Q) / (self.nu)

            sens["cost"]["control_inputs"] = opt_var_dict["control_inputs"] * self.R

            if self.use_state_cons:
                sens["state_cons"] = state_jac

            if self.use_dyn_state_cons:
                sens["dyn_state_cons"] = dyn_state_jac

            return sens
        
        # sens_rules.__qualname__ = "sens_rules"
        return sens_rules