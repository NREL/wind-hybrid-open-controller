# import pyomo.core
# TODO add option to .yaml to assume perfect vs stochastic preview

from time import perf_counter, time
from glob import glob
from functools import partial
from multiprocessing import Pool, cpu_count
import copy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyoptsparse import Optimization, SLSQP #, NSGA2, PSQP, ParOpt, CONMIN, ALPSO
from scipy.optimize import linprog

from whoc.config import *
# import pyomo.environ as pyo
from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.wind_field.WindField import generate_multi_wind_ts, generate_multi_wind_preview_ts, WindField, generate_wind_preview
from whoc.helpers import cluster_turbines

from floris_dev.tools import FlorisInterface as FlorisInterfaceDev
from floris_dev.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

optimizer_idx = 0

class YawOptimizationSRRHC(YawOptimizationSR):
    def __init__(
        self,
        fi,
        yaw_rate,
        dt,
        alpha,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5, 4],  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=True,
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
            exploit_layout_symmetry=exploit_layout_symmetry,
            verify_convergence=verify_convergence,
        )
        self.yaw_rate = yaw_rate
        self.dt = dt
        self.Q = alpha
        self.R = 100.0 * (1 - alpha) # need to scale control input componet to contend with power component
        self._turbine_power_opt_subset = np.zeros_like(self._minimum_yaw_angle_subset)
        self._cost_opt_subset = np.zeros_like(self._farm_power_baseline_subset)

    def _calculate_turbine_powers(self, yaw_angles=None, wd_array=None, turbine_weights=None,
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
        if yaw_angles is None:
            yaw_angles = self._yaw_angles_baseline_subset
        if turbine_weights is None:
            turbine_weights = self._turbine_weights_subset
        if heterogeneous_speed_multipliers is not None:
            fi_subset.floris.flow_field.\
                heterogenous_inflow_config['speed_multipliers'] = heterogeneous_speed_multipliers

        # Ensure format [incompatible with _subset notation]
        yaw_angles = self._unpack_variable(yaw_angles, subset=True)

        # # Correct wind direction definition: 270 deg is from left, cw positive
        # wd_array = wrap_360(wd_array)

        # Calculate solutions
        turbine_powers = np.zeros_like(self._minimum_yaw_angle_subset[:, 0, :])
        fi_subset.reinitialize(wind_directions=wd_array)
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
            self.turbine_powers_baseline = self._unreduce_variable(P)
        else:
            self._turbine_powers_baseline_subset = None
            self.turbine_powers_baseline = None

    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        # farm_power_opt_subset = self._farm_power_opt_subset
        turbine_power_opt_subset = self._turbine_power_opt_subset
        wd_array_subset = self.fi_subset.floris.flow_field.wind_directions
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 4)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1, 1))
            # farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny, 1))
            turbine_power_opt_subset = np.tile(turbine_power_opt_subset, (Ny, 1, 1))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1, 1))

        # Initialize empty matrix for floris farm power outputs
        # farm_powers = np.zeros((yaw_angles_subset.shape[0], yaw_angles_subset.shape[1]))
        turbine_powers = np.zeros(yaw_angles_subset.shape)

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=2).all(axis=1)
            # farm_powers[idx, :] = farm_power_opt_subset[idx, :]
            turbine_powers[idx, :, :] = turbine_power_opt_subset[idx, :, :]
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
            # farm_powers[~idx, :] = self._calculate_farm_power(
            #     wd_array=wd_array_subset[~idx],
            #     turbine_weights=turbine_weights_subset[~idx, :, :],
            #     yaw_angles=yaw_angles_subset[~idx, :, :],
            #     heterogeneous_speed_multipliers=het_sm
            # )
            turbine_powers[~idx, :, :] = self._calculate_turbine_powers(
                wd_array=wd_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :, :],
                yaw_angles=yaw_angles_subset[~idx, :, :],
                heterogeneous_speed_multipliers=het_sm
            )
            self.time_spent_in_floris += (perf_counter() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            turbine_powers = np.reshape(
                turbine_powers,
                (
                    Ny,
                    self.fi_subset.floris.flow_field.n_wind_directions,
                    self.fi_subset.floris.flow_field.n_wind_speeds,
                    self.nturbs
                )
            )

        return turbine_powers
        
    def optimize(self, current_yaw_angles, print_progress=False):
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """
        norm_current_yaw_angles = current_yaw_angles / self.maximum_yaw_angle
        self.print_progress = print_progress
        # compute baseline (no yaw) powers instead
        self._calculate_baseline_turbine_powers()

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
                
                # norm_current_yaw_angles + self.dt * (self.yaw_rate / self.yaw_limits[1]) * opt_var_dict["control_inputs"][current_idx]
                evaluation_grid = self._generate_evaluation_grid(
                    pass_depth=Nii,
                    turbine_depth=turbine_depth
                )
                # clip for control input values between -1 and 1
                evaluation_grid = np.clip(evaluation_grid,
                                           current_yaw_angles - (self.dt * self.yaw_rate),
                                            current_yaw_angles + (self.dt * self.yaw_rate))
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
                norm_turbine_powers = turbine_powers / self.turbine_powers_baseline
                # for each value in farm_powers, get corresponding next_yaw_angles from evaluation grid
                
                control_inputs = (evaluation_grid - current_yaw_angles) * (1 / (self.yaw_rate * self.dt))
                
                cost = np.sum((-0.5 * norm_turbine_powers**2 * self.Q) + (0.5 * control_inputs**2 * self.R), axis=3)
                
                args_opt = np.expand_dims(np.nanargmin(cost, axis=0), axis=0)

                cost_opt_new = np.squeeze(
                    np.take_along_axis(cost, args_opt, axis=0),
                    axis=0,
                )

                turbine_powers_opt_new = np.squeeze(
                    np.take_along_axis(turbine_powers, 
                                       np.expand_dims(args_opt, axis=3), 
                                       axis=0),
                    axis=0,
                )
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(np.sum(turbine_powers, axis=3), args_opt, axis=0),
                    axis=0,
                )
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=3),
                        axis=0
                    ),
                    axis=0
                )

                cost_opt_prev = self._cost_opt_subset
                farm_powers_opt_prev = self._farm_power_opt_subset
                turbine_powers_opt_prev = self._turbine_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                # Now update optimal farm powers if better than previous
                ids_better = (cost_opt_new < cost_opt_prev)
                cost_opt = cost_opt_prev
                cost_opt[ids_better] = cost_opt_new[ids_better]
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids_better] = farm_powers_opt_new[ids_better]

                # Now update optimal yaw angles if better than previous
                turbs_sorted = self.turbines_ordered_array_subset
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (*np.where(ids_better), turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                turbine_powers_opt = turbine_powers_opt_prev
                turbine_powers_opt[ids] = turbine_powers_opt_new[ids]


                # Update bounds for next iteration to close proximity of optimal solution
                dx = (
                    evaluation_grid[1, :, :, :] -
                    evaluation_grid[0, :, :, :]
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
                self._cost_opt_subset = cost_opt
                self._farm_power_opt_subset = farm_power_opt
                self._turbine_power_opt_subset = turbine_powers_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

        # Finalize optimization, i.e., retrieve full solutions
        self.cost_opt = self._unreduce_variable(self._cost_opt_subset)
        df_opt = self._finalize()
        df_opt["cost"] = self.cost_opt
        return df_opt



# TODO check if dynamic state equation constraints hold
class MPC(ControllerBase):

    # SLSQP, NSGA2, ParOpt, CONMIN, ALPSO
    max_iter = 25
    acc = 1e-6
    optimizers = [
        SLSQP(options={"IPRINT": -1, "MAXIT": max_iter, "ACC": acc})
        # NSGA2(options={"xinit": 1, "PrintOut": 0, "maxGen": 50})
        # CONMIN(options={"IPRINT": 0}), # "ITMAX": 25}),
        # ALPSO(options={}) #"maxOuterIter": 25})
        ]

    def __init__(self, interface, input_dict, optimizer_idx, verbose=False):
        
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
        self.wind_preview_func = input_dict["controller"]["wind_preview_func"]
        self.n_wind_preview_samples = input_dict["controller"]["n_wind_preview_samples"]
        self.stochastic = input_dict["controller"]["stochastic"]
        self.warm_start = input_dict["controller"]["warm_start"]
        self.Q = self.alpha
        self.R = 100 * (1 - self.alpha)
        self.nu = input_dict["controller"]["nu"]
        self.sequential_neighborhood_solve = input_dict["controller"]["sequential_neighborhood_solve"]

        self.n_cluster_turbines = self.n_turbines if not self.sequential_neighborhood_solve else 4
        self.n_states = self.n_horizon * self.n_turbines if not self.sequential_neighborhood_solve else self.n_horizon * self.n_cluster_turbines
        self.n_control_inputs = self.n_horizon * self.n_turbines if not self.sequential_neighborhood_solve else self.n_horizon * self.n_cluster_turbines

        self.dyn_state_jac = self.con_sens_rules()
        self.solver = input_dict["controller"]["solver"]
        
        
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
        
        # self.norm_turbine_powers = np.ones((self.n_wind_preview_samples, self.n_horizon, self.n_turbines)) * np.nan
        self.initial_state = np.zeros((self.n_turbines,))
        # self.most_recent_solution = {"states": [], "control_inputs": []}
        self.opt_sol = {"states": [], "control_inputs": []}
        
        if input_dict["controller"]["control_input_domain"].lower() in ['discrete', 'continuous']:
            self.control_input_domain = input_dict["controller"]["control_input_domain"].lower()
        else:
            raise TypeError("control_input_domain must be have value of 'discrete' or 'continuous'")
        
        # TODO pull from true disturbances
        self._disturbance_preview = {"wind_mag": [12.] * self.n_horizon,
                                     "wind_dir": [270.] * self.n_horizon,
                                     "wind_ti": [0.08] * self.n_horizon}
        
        self.fi = ControlledFlorisInterface(max_workers=interface.max_workers,
                                            yaw_limits=self.yaw_limits, dt=self.dt, yaw_rate=self.yaw_rate) \
            .load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
        
        # self.setup_pyomo_solver()
        if self.solver == "floris":
            self.fi_opt = FlorisInterfaceDev(input_dict["controller"]["floris_input_file"])
        elif self.solver == "pyopt":
            self.setup_pyopt_solver()
        elif self.solver == "zsgd":
            pass
            # raise NotImplementedError
        
    
    def con_sens_rules(self):
        sens = {"states": [], "control_inputs": []}

        for j in range(self.n_horizon):
            for i in range(self.n_cluster_turbines):
                current_idx = (self.n_cluster_turbines * j) + i
                
                # scaled by yaw limit
                sens["control_inputs"].append([
                    -(self.dt * (self.yaw_rate / self.yaw_limits[1])) if idx == current_idx else 0
                    for idx in range(self.n_control_inputs)
                ])
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    
                    sens["states"].append([
                        1 if idx == current_idx else 0
                        for idx in range(self.n_states)
                    ])
                    
                    
                else:
                    prev_idx = (self.n_cluster_turbines * (j - 1)) + i
                    
                    # drvt_states_ji = 1, drvt_states_(j-1,i) = -1,
                    # drvt_control_inputs_ji = -(self.dt * (self.yaw_rate / self.yaw_limits[1]))
                    
                    sens["states"].append([
                        1 if idx == current_idx else (-1 if idx == prev_idx else 0)
                        for idx in range(self.n_states)
                    ])
        return sens
    
    def dyn_state_rules(self, opt_var_dict):
        # define constraints
        dyn_state_cons = []
        for j in range(self.n_horizon):
            for i in range(self.n_cluster_turbines):
                current_idx = (self.n_cluster_turbines * j) + i
                delta_yaw = self.dt * (self.yaw_rate / self.yaw_limits[1]) * opt_var_dict["control_inputs"][current_idx]
                
                if j == 0:  # corresponds to time-step k=1 for states,
                    # pass initial state as parameter
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[i] + delta_yaw)]
                    # drvt_states_ji = 1, drvt_control_inputs_ji = -(self.dt * (self.yaw_rate / self.yaw_limits[1]))
                    
                else:
                    prev_idx = (self.n_cluster_turbines * (j - 1)) + i
                    # scaled by yaw limit
                    dyn_state_cons = dyn_state_cons + [
                        opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]
                    # drvt_states_ji = 1, drvt_states_(j-1,i) = -1,
                    # drvt_control_inputs_ji = -(self.dt * (self.yaw_rate / self.yaw_limits[1]))
        return dyn_state_cons
        
    def opt_rules(self, opt_var_dict):
        # TODO parallelize this, check if reset is necessary
        
        funcs = {}
        
        # define objective function
        # yaw_angles = np.array([[opt_var_dict["states"][(self.n_turbines * j) + i] * self.yaw_limits[1]
        #                         for i in range(self.n_turbines)] for j in range(self.n_horizon)])
        # TODO fetch last solution for turbines not in the current cluster and concatenate with current 
        # solutions for turbines in current cluster in opt_var_dict
        yaw_angles = np.clip(np.array([[((self.initial_state[i] * self.yaw_limits[1]) 
                                + (self.yaw_rate * self.dt * np.sum(opt_var_dict["control_inputs"][i:(self.n_cluster_turbines * j) + i:self.n_cluster_turbines])))
                                for i in range(self.n_cluster_turbines)] for j in range(self.n_horizon)]), *self.yaw_limits)
        
        # plot_distribution_samples(pd.DataFrame(wind_preview_samples), self.n_horizon)
        # derivative of turbine power output with respect to yaw angles
        if self.stochastic:
            
            u = np.random.normal(loc=0.0, scale=1.0, size=(self.n_wind_preview_samples, self.n_horizon, self.n_cluster_turbines))
            self.norm_turbine_powers_states_drvt = np.zeros((self.n_wind_preview_samples, self.n_horizon, self.n_cluster_turbines))
            self.norm_turbine_powers = np.zeros((self.n_wind_preview_samples, self.n_horizon, self.n_cluster_turbines))
            # derivative of turbine power output with respect to yaw angle changes
            # norm_turbine_powers_ctrl_inputs_drvt = np.zeros((self.n_wind_preview_samples, self.n_horizon * self.n_turbines))
        
            for m in range(self.n_wind_preview_samples):
                for j in range(self.n_horizon):
                    
                    self.fi.env.reinitialize(
                        wind_directions=[self.wind_preview_samples[f"FreestreamWindDir_{j}"][m]],
                        wind_speeds=[self.wind_preview_samples[f"FreestreamWindMag_{j}"][m]],
                        turbulence_intensity=self.disturbance_preview["wind_ti"][j]
                    )
                
                    # send yaw angles
                    self.fi.env.calculate_wake(yaw_angles[j, :][np.newaxis, :])
                    yawed_turbine_powers = self.fi.env.get_turbine_powers()

                    self.fi.env.calculate_wake((yaw_angles[j, :] + self.nu * u[m, j, :])[np.newaxis, :])
                    perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()
                
                    self.fi.env.calculate_wake(np.zeros((1, self.n_turbines)))
                    noyaw_turbine_powers = self.fi.env.get_turbine_powers()
                
                    # normalize power by no yaw output
                    self.norm_turbine_powers[m, j, :] = np.divide(yawed_turbine_powers, noyaw_turbine_powers,
                                                            where=noyaw_turbine_powers != 0,
                                                            out=np.zeros_like(noyaw_turbine_powers))
                    
                    self.norm_turbine_powers_states_drvt[m, j, :] = np.divide(((perturbed_yawed_turbine_powers - yawed_turbine_powers) / self.nu) * u[m, j, :], noyaw_turbine_powers, 
                                                                                where=noyaw_turbine_powers != 0,
                                                                                out=np.zeros_like(noyaw_turbine_powers))
                    

            # compute power based on sampling from wind preview
            funcs["cost"] = sum([-0.5*np.mean((self.norm_turbine_powers[:, j, i])**2) * self.Q
                                for j in range(self.n_horizon) for i in range(self.n_turbines)]) \
                            + sum(0.5*(opt_var_dict["control_inputs"][(self.n_turbines * j) + i])**2 * self.R
                                for j in range(self.n_horizon) for i in range(self.n_turbines))
        
        else:
            # if not using stochastic variation, pass mean values of u, v disturbances to FLORIS, rather than multiple realizations of probability distribution
            self.norm_turbine_powers_states_drvt = np.zeros((self.n_horizon, self.n_turbines))
            self.norm_turbine_powers = np.zeros((self.n_horizon, self.n_turbines))
            # test_norm_turbine_powers = np.zeros((self.n_horizon, self.n_turbines))

            for j in range(self.n_horizon):
                # self.measurements_dict["wind_speeds"][0]
                # self.measurements_dict["wind_directions"][0]
                # assume persistance of current measurement
                # assume persistant freestream wind conditions over course of horizon
                self.fi.env.reinitialize(
                    wind_directions=[self.measurements_dict["wind_directions"][0]],
                    wind_speeds=[self.measurements_dict["wind_speeds"][0]],
                    turbulence_intensity=self.disturbance_preview["wind_ti"][0]
                )
            
                # send yaw angles
                self.fi.env.calculate_wake(yaw_angles[j, :][np.newaxis, :])
                yawed_turbine_powers = self.fi.env.get_turbine_powers()

                # self.fi.env.calculate_wake(np.array([-30, -13.565, 0.0])[np.newaxis, np.newaxis, :])
                # test_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                self.fi.env.calculate_wake(np.clip((yaw_angles[j, :] + self.nu)[np.newaxis, :], *self.yaw_limits))
                plus_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                self.fi.env.calculate_wake(np.clip((yaw_angles[j, :] - self.nu)[np.newaxis, :], *self.yaw_limits))
                neg_perturbed_yawed_turbine_powers = self.fi.env.get_turbine_powers()

                # if yaw_angles[j, 0] == -30.0:
                # 	print('here')

                self.fi.env.calculate_wake(np.zeros((1, self.n_turbines)))
                noyaw_turbine_powers = self.fi.env.get_turbine_powers()
            
                # normalize power by no yaw output
                self.norm_turbine_powers[j, :] = np.divide(yawed_turbine_powers, noyaw_turbine_powers,
                                                        where=noyaw_turbine_powers != 0,
                                                        out=np.zeros_like(noyaw_turbine_powers))
                # test_norm_turbine_powers[j, :] = np.divide(test_yawed_turbine_powers, noyaw_turbine_powers,
                # 										where=noyaw_turbine_powers != 0,
                # 										out=np.zeros_like(noyaw_turbine_powers))
                
                self.norm_turbine_powers_states_drvt[j, :] = np.divide(((plus_perturbed_yawed_turbine_powers - neg_perturbed_yawed_turbine_powers) / (2 * self.nu)), 
                                                                            noyaw_turbine_powers,
                                                                            where=noyaw_turbine_powers != 0,
                                                                            out=np.zeros_like(noyaw_turbine_powers))

                # self.norm_turbine_powers_states_drvt[j, :] = np.divide(((perturbed_yawed_turbine_powers - yawed_turbine_powers) / self.nu), 
                # 														 noyaw_turbine_powers,
                # 														 where=noyaw_turbine_powers != 0,
                # 														 out=np.zeros_like(noyaw_turbine_powers))

            # compute power based on mean values from wind preview (persistance)
            # TODO power term is dominating for approx zero control inputs
            funcs["cost"] = sum([-0.5*(self.norm_turbine_powers[j, i])**2 * self.Q
                                for j in range(self.n_horizon) for i in range(self.n_turbines)]) \
                            + sum(0.5*(opt_var_dict["control_inputs"][(self.n_turbines * j) + i])**2 * self.R
                                for j in range(self.n_horizon) for i in range(self.n_turbines))


        
        funcs["dyn_state_cons"] = self.dyn_state_rules(opt_var_dict)
        # print(dyn_state_cons)
        fail = False
        
        return funcs, fail
    
    def sens_rules(self, opt_var_dict, obj_con_dict):
        sens = {"cost": {"states": [], "control_inputs": []},
                "dyn_state_cons": {"states": [], "control_inputs": []}}
        
        # yaw_angles = np.clip(np.array([[((self.initial_state[i] * self.yaw_limits[1]) 
        # 					  + (self.yaw_rate * self.dt * np.sum(opt_var_dict["control_inputs"][i:(self.n_turbines * j) + i:self.n_turbines])))
        #                         for i in range(self.n_turbines)] for j in range(self.n_horizon)]), *self.yaw_limits)
        

        if self.stochastic:
            for j in range(self.n_horizon):
                for i in range(self.n_turbines):
                    current_idx = (self.n_turbines * j) + i
                    sens["cost"]["control_inputs"].append(
                        opt_var_dict["control_inputs"][current_idx] * self.R
                    )
                    # compute power derivative based on sampling from wind preview
                    sens["cost"]["states"].append(
                                np.mean(-(self.norm_turbine_powers[:, j, i]) * self.Q * self.norm_turbine_powers_states_drvt[:, j, i])
                            )
                    
        else:
            # if not using stochastic variation, pass mean values of u, v disturbances to FLORIS, rather than multiple realizations of probability distribution
            # norm_turbine_powers_states_drvt = np.zeros((self.n_horizon, self.n_turbines))
            # norm_turbine_powers = np.zeros((self.n_horizon, self.n_turbines))

            for j in range(self.n_horizon):
                
                for i in range(self.n_turbines):
                    current_idx = (self.n_turbines * j) + i
                    sens["cost"]["control_inputs"].append(
                        opt_var_dict["control_inputs"][current_idx] * self.R
                    )
                    sens["cost"]["states"].append(
                            -(self.norm_turbine_powers[j, i]) * self.Q * self.norm_turbine_powers_states_drvt[j, i]
                        )

                # print(np.sum(self.norm_turbine_powers[j, :] - norm_turbine_powers[j, :]))
                # print(np.sum(self.norm_turbine_powers_states_drvt[j, :] - norm_turbine_powers_states_drvt[j, :]))

        sens["dyn_state_cons"] = self.dyn_state_jac # self.con_sens_rules()
        
        return sens

    def setup_pyopt_solver(self):
        # TODO if using sequential solver need to run for ever solve

        self.pyopt_sens = self.sens_rules
        # initialize optimization object
        self.pyopt_prob = Optimization("Wake Steerng MPC", self.opt_rules) #, sens=sens_rules)
        
        

        # add design variables
        self.pyopt_prob.addVarGroup("states", self.n_states,
                                    "c",  # continuous variables
                                    # lower=[self.yaw_limits[0]] * (self.n_horizon * self.n_turbines),
                                    # upper=[self.yaw_limits[1]] * (self.n_horizon * self.n_turbines),
                                    lower=[-1] * self.n_states,
                                    upper=[1] * self.n_states,
                                    value=[0] * self.n_states)
                                    # scale=(1 / self.yaw_limits[1]))
        
        if self.control_input_domain == 'continuous':
            self.pyopt_prob.addVarGroup("control_inputs", self.n_control_inputs,
                                        varType="c",
                                        lower=[-1] * self.n_control_inputs,
                                        upper=[1] * self.n_control_inputs,
                                        value=[0] * self.n_control_inputs)
        else:
            self.pyopt_prob.addVarGroup("control_inputs", self.n_control_inputs,
                                        varType="i",
                                        lower=[-1] * self.n_control_inputs,
                                        upper=[1] * self.n_control_inputs,
                                        value=[0] * self.n_control_inputs)
        
        # # add dynamic state equation constraints
        # jac = self.con_sens_rules()
        self.pyopt_prob.addConGroup("dyn_state_cons", self.n_horizon * self.n_turbines, lower=0.0, upper=0.0)
                        #   linear=True, wrt=["states", "control_inputs"], # TODO supplying fixed jac won't work because value of initial_state changes
                        #   jac=jac)
        
        # add objective function
        self.pyopt_prob.addObj("cost")
        
        # display optimization problem
        print(self.pyopt_prob)
    
    def compute_controls(self):
        """
        solve OCP to minimize objective over future horizon
        """
        
        # get current time-step
        current_time_step = int(self.measurements_dict["time"] // self.dt)
        # self.norm_turbine_powers.fill(np.nan)
        
        if current_time_step > 0.:
            # update initial state self.mi_model.initial_state
            self.initial_state = self.measurements_dict["yaw_angles"] / self.yaw_limits[1] # scaled by yaw limits
        
        current_freestream_measurements = [
                self.measurements_dict["wind_speeds"][0]
                * np.cos((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.)),
                self.measurements_dict["wind_speeds"][0]
                * np.sin((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.))
            ]
            
        self.wind_preview_samples = self.wind_preview_func(current_freestream_measurements)
       
        if self.solver == "pyopt":
            yaw_star = self.pyopt_solve()
        elif self.solver == "floris":
            yaw_star = self.floris_solve()
        elif self.solver == "zsgd":
            yaw_star = self.zsgd_solve()
        
        # self.turbine_powers = np.mean((self.norm_turbine_powers[:, j, i]) for j in range(self.n_horizon) for i in range(self.n_turbines)) # \
        
        self.controls_dict = {"yaw_angles": yaw_star}
    
    def zsgd_solve(self):

        # current_yaw_angles = self.measurements_dict["yaw_angles"]

        # warm start with previous solution
        # current_time_step = int(self.measurements_dict["time"] // self.dt)
        # init_states = np.zeros((self.n_horizon, self.n_turbines))
        
        # warm-up with previous solution
        # if current_time_step > 0:
        #     for j in range(self.n_horizon - 1):
        #         next_idx = slice(((j + 1) * self.n_turbines), ((j + 2) * self.n_turbines))
        #         init_states[j, :] = np.clip(self.opt_sol["states"][next_idx], -1, 1)
        #     init_states[self.n_horizon - 1, :] = np.clip(self.opt_sol["states"][-self.n_turbines:], -1, 1)
        
        # # TODO is init_sol appearing in optimized solution?
        # self.init_sol = [init_states[j, i] for j in range(self.n_horizon) for i in range(self.n_turbines)] + \
        #     [(init_states[j, i] - (init_states[j - 1, i] if j > 0 else self.initial_state[i])) 
        #              * (self.yaw_limits[1] / (self.yaw_rate * self.dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)]

        self.warm_start_opt_vars()

        bounds = (-1, 1)
        A_eq = np.zeros(((self.n_horizon * self.n_turbines), self.n_horizon * self.n_turbines * 2))
        b_eq = np.zeros(((self.n_horizon * self.n_turbines), ))
        
        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (self.n_turbines * j) + i
                # delta_yaw = self.dt * (self.yaw_rate / self.yaw_limits[1]) * opt_var_dict["control_inputs"][current_idx]
                A_eq[current_idx, (self.n_horizon * self.n_turbines) + current_idx] = -self.dt * (self.yaw_rate / self.yaw_limits[1])
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

        # TODO this is not zero
        print(np.sum(opt_var_dict["states"][:self.n_turbines] - (self.initial_state + opt_var_dict["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_limits[1]) * self.dt)))

        print(np.sum(opt_var_dict["states"][self.n_turbines:] - (opt_var_dict["states"][:-self.n_turbines] + opt_var_dict["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_limits[1]) * self.dt)))

        print(self.opt_sol["states"][:self.n_turbines] - 
              ((self.initial_state) + ((self.yaw_rate / self.yaw_limits[1]) * self.dt * self.opt_sol["control_inputs"][:self.n_turbines])))
        
        return np.rint(np.clip(((self.initial_state * self.yaw_limits[1]) + (self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines])), *self.yaw_limits) / self.yaw_increment) * self.yaw_increment

    def floris_solve(self):
        current_yaw_angles = self.measurements_dict["yaw_angles"]

        # warm start with previous solution
        current_time_step = int(self.measurements_dict["time"] // self.dt)
        init_states = np.zeros((self.n_horizon, self.n_turbines))
        
        # warm-up with previous solution
        self.warm_start_opt_vars()
        # if current_time_step > 0:
        #     for j in range(self.n_horizon - 1):
        #         next_idx = slice(((j + 1) * self.n_turbines), ((j + 2) * self.n_turbines))
        #         init_states[j, :] = np.clip(self.opt_sol["states"][next_idx], -1, 1)
        #     init_states[self.n_horizon - 1, :] = np.clip(self.opt_sol["states"][-self.n_turbines:], -1, 1)
        
        # TODO is init_sol appearing in optimized solution?
        # self.init_sol = [init_states[j, i] for j in range(self.n_horizon) for i in range(self.n_turbines)] + \
        #     [(init_states[j, i] - (init_states[j - 1, i] if j > 0 else self.initial_state[i])) 
        #              * (self.yaw_limits[1] / (self.yaw_rate * self.dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)]
        # TODO Misha question: problem if Floris wind direction array does not intersect 0.0 and 180.0.?
        if self.stochastic:
            opt_yaw_angles = np.zeros((self.n_wind_preview_samples, self.n_horizon, self.n_turbines))
            opt_cost = np.zeros((self.n_wind_preview_samples, self.n_horizon))
            for j in range(self.n_horizon):
                for m in range(self.n_wind_preview_samples):
                    self.fi_opt.reinitialize(
                        wind_directions=[self.wind_preview_samples[f"FreestreamWindDir_{j}"][m]],
                        wind_speeds=[self.wind_preview_samples[f"FreestreamWindMag_{j}"][m]],
                        turbulence_intensity=self.disturbance_preview["wind_ti"][j]
                    )
                
                    # optimize yaw angles
                    # TODO MISHA is there a reason not to optimize for negative yaw angles?
                    yaw_opt = YawOptimizationSRRHC(self.fi_opt, 
                                 self.yaw_rate, self.dt, self.alpha,
                                 minimum_yaw_angle=self.yaw_limits[0],
                                 maximum_yaw_angle=self.yaw_limits[1],
                                 yaw_angles_baseline=np.zeros((1, 1, self.n_turbines)),
                                #  normalize_control_variables=True,
                                    Ny_passes=[10, 8, 6, 4],
                                 x0=init_states[j, :][np.newaxis, np.newaxis, :],
                                 verify_convergence=False)#, exploit_layout_symmetry=False)
                    if j == 0:
                        current_yaw_angles = self.initial_state * self.yaw_limits[1]
                    else:
                        current_yaw_angles = opt_yaw_angles[m, j - 1, :]
                    # TODO can we optimize for all samples in one sweep
                    tmp = yaw_opt.optimize(current_yaw_angles)
                    opt_yaw_angles[m, j, :] = tmp["yaw_angles_opt"].iloc[0]
                    opt_cost[m, j] = tmp["cost"].iloc[0]

            # TODO MISHA averaging solutions for stochastic preview??
            
            opt_yaw_angles = np.mean(opt_yaw_angles, axis=0)
            opt_cost = np.sum(np.mean(opt_cost, axis=0))
                
        else:
            opt_yaw_angles = np.zeros((self.n_horizon, self.n_turbines))
            opt_cost = 0
            # if not using stochastic variation, pass mean values of u, v disturbances to FLORIS, rather than multiple realizations of probability distribution
            for j in range(self.n_horizon):
                # assume persistance of current measurement
                self.fi_opt.reinitialize(
                    wind_directions=[self.measurements_dict["wind_directions"][0]],
                    wind_speeds=[self.measurements_dict["wind_speeds"][0]],
                    turbulence_intensity=self.disturbance_preview["wind_ti"][0]
                )
                # optimize yaw angles
                yaw_opt = YawOptimizationSRRHC(self.fi_opt, self.yaw_rate, self.dt, self.alpha,
                                                minimum_yaw_angle=self.yaw_limits[0],
                                                maximum_yaw_angle=self.yaw_limits[1],
                                                yaw_angles_baseline=np.zeros((1, 1, self.n_turbines)),
                                                # normalize_control_variables=True,
                                                x0=init_states[j, :][np.newaxis, np.newaxis, :],
                                                verify_convergence=False)#, exploit_layout_symmetry=False)
                
                if j == 0:
                    current_yaw_angles = self.initial_state * self.yaw_limits[1]
                else:
                    current_yaw_angles = opt_yaw_angles[j - 1, :]
                
                tmp = yaw_opt.optimize(current_yaw_angles)
                opt_yaw_angles[j, :] = tmp["yaw_angles_opt"].iloc[0]
                opt_cost += tmp["cost"].iloc[0]

        self.opt_sol = {"states": np.array([opt_yaw_angles[j, i] / self.yaw_limits[1] for j in range(self.n_horizon) for i in range(self.n_turbines)]), 
                        "control_inputs": np.array([(opt_yaw_angles[j, i] - (opt_yaw_angles[j-1, i] if j > 0 else self.initial_state[i] * self.yaw_limits[1])) * (1 / (self.yaw_rate * self.dt)) for j in range(self.n_horizon) for i in range(self.n_turbines)])}
        self.opt_code = {"text": None}
        self.opt_cost = opt_cost

        return opt_yaw_angles[0, :]
            # reinitialize the floris object with the predicted wind magnitude and direction at this time-step in the horizon

    def warm_start_opt_vars(self,):
        self.init_sol = {"states": [], "control_inputs": []}
        current_yaw_angles = self.measurements_dict["yaw_angles"]
        
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
                next_yaw_angles = np.array([self.yaw_IC / self.yaw_limits[1]] * (self.n_horizon * self.n_turbines))
                current_control_inputs = np.array([0] * (self.n_horizon * self.n_turbines))
                self.init_sol["states"] = next_yaw_angles
                self.init_sol["control_inputs"] = current_control_inputs

        elif self.warm_start == "lut":
            # delta_yaw = self.dt * (self.yaw_rate / self.yaw_limits[1]) * opt_var_dict["control_inputs"][prev_idx]
            
            self.init_sol = {
                    "states": [],
                    "control_inputs": []
            }
            for j in range(self.n_horizon):
                # compute yaw angle setpoints from warm_start_func
                
                if self.stochastic:
                    next_yaw_angles = self.warm_start_func({"wind_speeds": 
                                        [np.mean(self.wind_preview_samples[f"FreestreamWindMag_{j + 1}"])], 
                                        "wind_directions": [np.mean(self.wind_preview_samples[f"FreestreamWindDir_{j + 1}"])]})
                else:
                    next_yaw_angles = self.warm_start_func({
                        "wind_speeds": [self.measurements_dict["wind_speeds"][0]], # TODO should be freestream
                        "wind_directions": [self.measurements_dict["wind_directions"][0]]})
                
                current_control_inputs = (next_yaw_angles - current_yaw_angles) * (1 / (self.yaw_rate * self.dt))
                
                self.init_sol["states"] = self.init_sol["states"] + list(next_yaw_angles / self.yaw_limits[1])
                self.init_sol["control_inputs"] = self.init_sol["control_inputs"] + list(current_control_inputs)

                # for i in range(self.n_turbines):
                #     current_idx = (j * self.n_turbines) + i
                #     next_idx = ((j + 1) * self.n_turbines) + i

                #     self.pyopt_prob.variables["states"][current_idx].value \
                #         = np.clip(next_yaw_angles[i] / self.yaw_limits[1], -1, 1)
                    
                #     self.pyopt_prob.variables["control_inputs"][current_idx].value \
                #         = np.clip(current_control_inputs[i], -1, 1)
                
                current_yaw_angles = next_yaw_angles
            

            self.init_sol["states"] = np.clip(self.init_sol["states"], -1, 1)
            self.init_sol["control_inputs"] = np.clip(self.init_sol["control_inputs"], -1, 1)

        # self.init_sol = [s.value for s in self.pyopt_prob.variables["states"]] + [c.value for c in self.pyopt_prob.variables["control_inputs"]]
        

    def pyopt_solve(self):
        
        # opt_options = {}
        # opt = ParOpt(options=opt_options) # requires install paropt
        # opt = ALPSO(options=opt_options)
        
        # warm start Vars by reinitializing the solution from last time-step self.mi_model.states
        self.warm_start_opt_vars()

        # TODO only set values corresponding to cluster of turbines being optimized,
        # set yaw angles. yaw angle changes of other as parameters for clustered approach
        for j in range(self.n_horizon):
            for i in range(self.n_turbines):
                current_idx = (j * self.n_turbines) + i
                # next_idx = ((j + 1) * self.n_turbines) + i
                self.pyopt_prob.variables["states"][current_idx].value \
                    = np.clip(self.init_sol["states"][current_idx], -1, 1)
                
                self.pyopt_prob.variables["control_inputs"][current_idx].value \
                    = np.clip(self.init_sol["control_inputs"][current_idx], -1, 1)
        
        sol = MPC.optimizers[self.optimizer_idx](self.pyopt_prob, sens=self.pyopt_sens) #, sensMode='pgc')
        # TODO why do control inputs deviate from initial values of zero and problem still "terminates successfully"
        # TODO first dynamic state constraint violated when jac passed to equality constraints
        # print(f"469 = {self.initial_state}")
        print(np.sum(sol.xStar["states"][:self.n_turbines] - (self.initial_state + sol.xStar["control_inputs"][:self.n_turbines] * (self.yaw_rate / self.yaw_limits[1]) * self.dt)))
        print(np.sum(sol.xStar["states"][self.n_turbines:] - (sol.xStar["states"][:-self.n_turbines] + sol.xStar["control_inputs"][self.n_turbines:] * (self.yaw_rate / self.yaw_limits[1]) * self.dt)))
        # TODO try to update dynamic state constraints at each iteration, ie redefine self.dyn_state_rules
        # self.dyn_state_rules_partial = partial(self.dyn_state_rules, self.initial_state)
        # print(sol)
        # sol.xStar["states"][:self.n_turbines] * self.yaw_limits[1]
        # sol.constraints["dyn_state_cons"].value
        # TODO opt_sol must be populate with most recent solution for all turbines and horizon steps for clustered approach
        self.opt_sol = dict(sol.xStar)
        self.opt_code = sol.optInform
        self.opt_cost = sol.fStar
        # opt_elapsed_time = [res[1] for res in results]
        # solution is scaled by yaw limit
        # assert np.all(np.isclose(self.initial_state + (self.yaw_rate / self.yaw_limits[1])* self.dt * self.opt_sol["control_inputs"][:self.n_turbines], self.opt_sol["states"][:self.n_turbines]))
        return np.rint(np.clip(((self.initial_state * self.yaw_limits[1]) + (self.yaw_rate * self.dt * sol.xStar["control_inputs"][:self.n_turbines])), *self.yaw_limits) / self.yaw_increment) * self.yaw_increment
        # return (self.initial_state * self.yaw_limits[1]) \
        # 	+ self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]  # solution is scaled by yaw limit
    
    
    @property
    def disturbance_preview(self):
        return self._disturbance_preview
    
    @disturbance_preview.setter
    def disturbance_preview(self, disturbance_preview):
        self._disturbance_preview = disturbance_preview
    
def mpc_runner(input_dict, wind_mag_ts, wind_dir_ts, wind_ti_ts, wind_mag_preview_ts, wind_dir_preview_ts, wind_ti_preview_ts, optimizer_idx):
    # Load a FLORIS object for AEP calculations
    fi_mpc = ControlledFlorisInterface(yaw_limits=input_dict["controller"]["yaw_limits"],
                                        dt=input_dict["dt"],
                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
        .load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
    
    ctrl_mpc = MPC(fi_mpc, input_dict=input_dict, optimizer_idx=optimizer_idx)
    # TODO use coroutines or threading for hercules interfaces
    # optionally warm-start with LUT solution
    fi_lut = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
                                    dt=DT,
                                    yaw_rate=input_dict["controller"]["yaw_rate"]) \
                    .load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
    input_dict["controller"]["floris_input_file"] = WIND_FIELD_CONFIG["floris_input_file"]
    ctrl_lut = LookupBasedWakeSteeringController(fi_lut, input_dict=input_dict, lut_path="./lut.csv")
    if input_dict["controller"]["warm_start"] == "lut":
        def warm_start_func(measurements: dict):
            # feed interface with new disturbances
            fi_lut.step(disturbances={"wind_speeds": [measurements["wind_speeds"][0]],
                                      "wind_directions": [measurements["wind_directions"][0]],
                                      "turbulence_intensity": 0.08}) # TODO include this in measurments? 
            
            # receive measurements from interface, compute control actions, and send to interface
            ctrl_lut.step()
            return ctrl_lut.controls_dict["yaw_angles"]

        # TODO define descriptor class to ensure that this function takes current_measurements
        ctrl_mpc.warm_start_func = warm_start_func
    
    yaw_angles_ts = []
    turbine_powers_ts = []
    convergence_time_ts = []
    opt_codes_ts = []
    opt_costs_ts = []
    fi_mpc.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
                            "wind_directions": [wind_dir_ts[0]],
                            "turbulence_intensity": wind_ti_ts[0]})
    
    for k, t in enumerate(np.arange(0, EPISODE_MAX_TIME - input_dict["dt"], input_dict["dt"])):

        # feed interface with new disturbances
        fi_mpc.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
                                "wind_directions": [wind_dir_ts[k]],
                                "turbulence_intensity": wind_ti_ts[k]})
        
        # update disturbance preview
        ctrl_mpc.disturbance_preview = {"wind_mag": wind_mag_preview_ts[k],
                                        "wind_dir": wind_dir_preview_ts[k],
                                        "wind_ti": wind_ti_preview_ts[k]}
        
        # receive measurements from interface, compute control actions, and send to interface
        start_time = time()
        ctrl_mpc.step()
        end_time = time()
        convergence_time_ts.append(end_time - start_time)
        opt_codes_ts.append(ctrl_mpc.opt_code)
        opt_costs_ts.append(ctrl_mpc.opt_cost)
        
        # print(f"Time = {ctrl_mpc.measurements_dict['time']} for Optimizer {MPC.optimizers[optimizer_idx].__class__}")
        print(f"\nTime = {ctrl_mpc.measurements_dict['time']}",
            f"Measured Freestream Wind Direction = {wind_dir_ts[k]}",
            f"Measured Freestream Wind Magnitude = {wind_mag_ts[k]}",
            f"Measured Turbine Wind Directions = {ctrl_mpc.measurements_dict['wind_directions']}",
            f"Measured Turbine Wind Magnitudes = {ctrl_mpc.measurements_dict['wind_speeds']}",
            f"Measured Yaw Angles = {ctrl_mpc.measurements_dict['yaw_angles']}",
            f"Measured Turbine Powers = {ctrl_mpc.measurements_dict['powers']}",
            f"Initial Yaw Angle Solution = {np.array(ctrl_mpc.init_sol['states']) * ctrl_mpc.yaw_limits[1]}",
            f"Initial Yaw Angle Change Solution = {ctrl_mpc.init_sol['control_inputs']}",
            f"Optimizer Output = {ctrl_mpc.opt_code['text']}",
            f"Optimized Yaw Angle Solution = {ctrl_mpc.opt_sol['states'] * ctrl_mpc.yaw_limits[1]}",
            f"Optimized Yaw Angle Change Solution = {ctrl_mpc.opt_sol['control_inputs']}",
            f"Optimized Yaw Angles = {ctrl_mpc.controls_dict['yaw_angles']}",
            f"Convergence Time = {convergence_time_ts[-1]}",
            sep='\n')
        
        yaw_angles_ts.append(ctrl_mpc.measurements_dict['yaw_angles'])
        turbine_powers_ts.append(ctrl_mpc.measurements_dict['powers'])

        if k > 0:
            print(f"Change in Optimized Farm Powers relative to previous time-step = {100 * (sum(turbine_powers_ts[-1]) - sum(turbine_powers_ts[-2])) /  sum(turbine_powers_ts[-2])} %")
            print(f"Change in Optimized Cost relative to previous time-step = {100 * (opt_costs_ts[-1] - opt_costs_ts[-2]) /  opt_costs_ts[-2]} %")
            print(f"Change in Optimized Cost relative to first time-step = {100 * (opt_costs_ts[-1] - opt_costs_ts[0]) /  opt_costs_ts[0]} %")
            # print(f"Change in Optimized Cost relative to second time-step = {100 * (opt_costs_ts[-1] - opt_costs_ts[1]) /  opt_costs_ts[1]} %")

            ctrl_mpc.fi.env.reinitialize(
                    wind_directions=[wind_dir_ts[k]],
                    wind_speeds=[wind_mag_ts[k]],
                    turbulence_intensity=wind_ti_ts[k]
                )
                    
            # send yaw angles to compute lut solution
            fi_lut.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
                                "wind_directions": [wind_dir_ts[k]],
                                "turbulence_intensity": wind_ti_ts[k]})
            
            # receive measurements from interface, compute control actions, and send to interface
            ctrl_lut.step()
            ctrl_mpc.fi.env.calculate_wake(ctrl_lut.controls_dict["yaw_angles"][np.newaxis, :])
            lut_yawed_turbine_powers = np.squeeze(ctrl_mpc.fi.env.get_turbine_powers())
        
            ctrl_mpc.fi.env.calculate_wake(np.zeros((1, ctrl_mpc.n_turbines)))
            noyaw_turbine_powers = np.squeeze(ctrl_mpc.fi.env.get_turbine_powers())

            print(f"Change in Optimized Farm Powers relative to No Yaw = {100 * (sum(turbine_powers_ts[-1]) - sum(noyaw_turbine_powers)) /  sum(noyaw_turbine_powers)} %")
            print(f"Change in Optimized Farm Powers relative to LUT = {100 * (sum(turbine_powers_ts[-1]) - sum(lut_yawed_turbine_powers)) /  sum(lut_yawed_turbine_powers)} %")

    yaw_angles_ts = np.vstack(yaw_angles_ts)
    turbine_powers_ts = np.vstack(turbine_powers_ts)
    convergence_time_ts = np.vstack(convergence_time_ts)

    return yaw_angles_ts, turbine_powers_ts, convergence_time_ts, opt_codes_ts, opt_costs_ts

if __name__ == '__main__':
    # import
    from hercules.utilities import load_yaml
    
    # options
    max_workers = 16
    # input_dict = load_yaml(sys.argv[1])
    input_dict = load_yaml("../../examples/hercules_input_001.yaml")

    input_dict["controller"]["n_wind_preview_samples"] = 20

    # instantiate wind preview distribution
    wf = WindField(**WIND_FIELD_CONFIG)
    # wind_preview_generator = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
    wind_preview_func = partial(generate_wind_preview, 
                                wf._sample_wind_preview, 
                                input_dict["controller"]["n_horizon"],
                                input_dict["controller"]["n_wind_preview_samples"])
    
    # instantiate controller
    input_dict["controller"]["wind_preview_func"] = wind_preview_func
    # ctrl_mpc = MPC(fi_mpc, input_dict=input_dict)
    
    
    ## Simulate wind farm with interface and controller
    
    # instantiate wind field if files don't already exist
    regenerate_wind_preview = False
    wind_field_filenames = glob(f"{DATA_SAVE_DIR}/case_*.csv")
    if not len(wind_field_filenames):
        generate_multi_wind_ts(WIND_FIELD_CONFIG, N_CASES)
        wind_field_filenames = [f"case_{i}.csv" for i in range(N_CASES)]
        regenerate_wind_preview = True
    
    # if wind field data exists, get it
    wind_field_data = []
    if os.path.exists(DATA_SAVE_DIR):
        for fn in wind_field_filenames:
            wind_field_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))

            if True:
                wind_field_data[-1][f"FreestreamWindMag"] = 9.0
                wind_field_data[-1][f"FreestreamWindDir"] = 260.0
    
    # instantiate wind field preview if files don't already exist
    wind_field_preview_filenames = glob(f"{DATA_SAVE_DIR}/preview_case_*.csv")
    if not len(wind_field_preview_filenames) or regenerate_wind_preview:
        generate_multi_wind_preview_ts(WIND_FIELD_CONFIG, N_CASES, wind_field_data)
        wind_field_preview_filenames = [f"preview_case_{i}.csv" for i in range(N_CASES)]
    
    # if wind field preview data exists, get it
    wind_field_preview_data = []
    if os.path.exists(DATA_SAVE_DIR):
        for fn in wind_field_preview_filenames:
            wind_field_preview_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))

            if True:
                for j in range(wf.n_preview_steps):
                    wind_field_preview_data[-1][f"FreestreamWindMag_{j}"] = 9.0
                    wind_field_preview_data[-1][f"FreestreamWindDir_{j}"] = 260.0
    
    # true wind disturbance time-series
    case_idx = 0
    time_ts = wind_field_data[case_idx]["Time"].to_numpy()
    wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
    wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
    wind_ti_ts = [0.08] * int(EPISODE_MAX_TIME // input_dict["dt"])
    
    # n_preview_steps = int(WIND_FIELD_CONFIG["wind_speed_preview_time"]
    #                       // WIND_FIELD_CONFIG["wind_speed_sampling_time_step"])
    
    # estimated wind disturbance time-series based on mean values from true wind disturbance time-series above
    n_preview_steps = input_dict["controller"]["n_horizon"]
    wind_mag_preview_ts = wind_field_preview_data[case_idx][
        [f"FreestreamWindMag_{i}" for i in range(n_preview_steps)]].to_numpy()
    wind_dir_preview_ts = wind_field_preview_data[case_idx][
        [f"FreestreamWindDir_{i}" for i in range(n_preview_steps)]].to_numpy()
    wind_ti_preview_ts = 0.08 * np.ones((int(EPISODE_MAX_TIME // input_dict["dt"]), n_preview_steps))
    
    
    # partial_mpc_runner = partial(mpc_runner, input_dict, wind_mag_ts, wind_dir_ts, wind_ti_ts, wind_mag_preview_ts, wind_dir_preview_ts, wind_ti_preview_ts)
    # with Pool(mp.cpu_count()) as p:
    # results = p.map(partial_mpc_runner, range(len(MPC.optimizers)))
    
    results = mpc_runner(input_dict, wind_mag_ts, wind_dir_ts, wind_ti_ts, wind_mag_preview_ts, wind_dir_preview_ts, wind_ti_preview_ts, optimizer_idx=optimizer_idx)
    
    yaw_angles_ts, turbine_powers_ts, convergence_time_ts, opt_codes_ts, opt_costs_ts = results
    # filt_wind_dir_ts = ctrl_mpc._first_ord_filter(wind_dir_ts, ctrl_mpc.wd_lpf_alpha)
    # filt_wind_speed_ts = ctrl_mpc._first_ord_filter(wind_mag_ts, ctrl_mpc.ws_lpf_alpha)
    # TODO save results and configurataion in a dataframe of results

    fig_wind, ax_wind = plt.subplots(2, 1, figsize=FIGSIZE)
    ax_wind[0].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_dir_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
    # ax[0].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_dir_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
    #            label='filtered')
    ax_wind[0].set(title='Wind Direction [deg]', xlabel='Time')
    ax_wind[1].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_mag_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
    # ax[1].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_speed_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
    #            label='filtered')
    ax_wind[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
    # ax.set_xlim((time_ts[1], time_ts[-1]))
    ax_wind[0].legend()
    fig_wind.savefig(os.path.join(FIG_DIR, f'mpc_{MPC.optimizers[optimizer_idx].__class__}_wind_results.png'))
    fig_wind.show()

    fig_outputs, ax_outputs = plt.subplots(3, 1, figsize=FIGSIZE)
    ax_outputs[0].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], yaw_angles_ts)
    ax_outputs[0].set(title='Yaw Angles [deg]', xlabel='Time [s]')
    ax_outputs[1].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], turbine_powers_ts)
    ax_outputs[1].set(title="Turbine Powers [MW]")
    ax_outputs[2].scatter(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], convergence_time_ts)
    ax_outputs[2].set(title="Convergence Time [s]")
    fig_outputs.savefig(os.path.join(FIG_DIR, f'mpc_{MPC.optimizers[optimizer_idx].__class__}_outputs_results.png'))
    fig_outputs.show()