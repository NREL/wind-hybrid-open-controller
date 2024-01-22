import pyomo.core

from controller_base import ControllerBase
import numpy as np
import pandas as pd
from whoc.config import *
import pyomo.environ as pyo
from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.wind_field.WindField import generate_multi_wind_ts, generate_multi_wind_preview_ts, generate_preview_noise, \
	WindField
from glob import glob
import matplotlib.pyplot as plt
from pyomo.core.expr import ExternalFunctionExpression
from pyomo.core.base.external import ExternalFunction, \
	PythonCallbackFunction  # https://github.com/Pyomo/pyomo/blob/main/pyomo/core/tests/unit/test_external.py
from pyoptsparse import Optimization, SLSQP, NSGA2, ParOpt, CONMIN, ALPSO
from functools import partial


class Objective:
	def __init__(self, is_quadratic, is_stochastic, n_horizon, dyn_model, **kwargs):
		self.is_quadratic = is_quadratic
		self.is_stochastic = is_stochastic
		self.dyn_model = dyn_model
		
		if is_quadratic:
			# transform into block matrices
			# self.Qhat = np.diag(np.concatenate([np.repeat(np.diag(kwargs['Q']), (1, n_horizon)). np.diag(kwargs['P'])]))
			# self.Rhat = np.repeat(np.diag(kwargs['R']), (1, n_horizon))
			self.Qhat = np.diag([np.kron(np.eye(n_horizon), kwargs['Q']), kwargs['P']])
			self.Rhat = np.kron(np.eye(n_horizon), kwargs['R'])
			
			self.Hhat = self.Rhat + dyn_model.Bhat.T @ self.Qhat @ dyn_model.Bhat
			self.Hhat = (self.Hhat + self.Hhat.T) / 2  # ensure symmetrical
			self.fhat = dyn_model.Bhat.T @ self.Qhat @ dyn_model.Ahat  # @ x0
		
		if is_stochastic:
			self.stochastic_varnames = kwargs['stochastic_varnames']
	
	# get expected value
	
	def compute(self, x0, X, U):
		"""
		X = [x1 x2 ... xN]
		U = [u0 u1 ... uN-1]
		"""
		if self.is_quadratic:
			fhat = self.fhat @ x0
			return 0.5 * U.T @ self.Hhat @ U + U.T @ fhat


# return X.T @ self.Qhat @ X + U.T @ self.Rhat @ U


class DynamicModel:
	def __init__(self, n_outputs, n_states, n_ctrl_inputs, n_disturbances, n_horizon, **kwargs):
		self._state = np.zeros((n_states,))
		self._output = np.zeros((n_outputs,))
		self._ctrl_inpt = np.zeros((n_ctrl_inputs,))
		self._dist = np.zeros((n_disturbances,))
		self._stochastic = False
		
		if 'A' in kwargs and 'B' in kwargs:
			# dense formulation, for integrating dynamic state equation into cost function
			# self.Ahat = np.vstack([np.linalg.matrix_power(kwargs['A'], i) for i in range(1, n_horizon)])
			# self.Bhat = np.vstack(
			# 	[[np.linalg.matrix_power(kwargs['A'], i) @ kwargs['B']
			# 	  for j in range(n_horizon)]
			# 	 for i in range(n_horizon)]
			# )
			# TODO formulate for output in cost function
			self.Ahat = np.kron(np.zeros((n_horizon, 1)), kwargs['A'])
			self.Bhat = np.kron(np.zeros(n_horizon), kwargs['B'])
			for j in range(n_horizon):
				# zero vector with 1 at location of current power of A^k
				ej = np.arange(n_horizon) == j
				# off-diagonal identity
				Ij = np.diag(np.ones(n_horizon - j), -j)
				
				self.Ahat = self.Ahat + np.kron(ej, np.linalg.matrix_power(kwargs['A'], j + 1))
				self.Bhat = self.Bhat + np.kron(Ij, np.linalg.matrix_power(kwargs['A'], j) @ kwargs['B'])
		
		if 'C' in kwargs and 'D' in kwargs:
			self.C = kwargs['C']
			self.D = kwargs['D']
		elif 'output_func' in kwargs:
			self.output_func = kwargs['output_func']
	
	@property
	def state(self):
		return self._state
	
	@property
	def ctrl_inpt(self):
		return self._ctrl_inpt
	
	@property
	def output(self):
		return self._output
	
	@property
	def disturbance(self):
		return self._dist
	
	def next(self):
		pass


class MPC(ControllerBase):
	def __init__(self, interface, input_dict, verbose=False):
		
		super().__init__(interface, verbose=verbose)
		
		self.dt = input_dict["controller"]["dt"]
		self.n_turbines = input_dict["controller"]["num_turbines"]
		self.turbines = range(self.n_turbines)
		self.yaw_limits = input_dict["controller"]["yaw_limits"]
		self.yaw_rate = input_dict["controller"]["yaw_rate"]
		self.alpha = input_dict["controller"]["alpha"]
		self.n_horizon = input_dict["controller"]["n_horizon"]
		self.wind_preview_func = input_dict["controller"]["wind_preview_func"]
		self.n_wind_preview_samples = input_dict["controller"]["n_wind_preview_samples"]
		
		# Set initial conditions
		yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
		if hasattr(yaw_IC, "__len__"):
			if len(yaw_IC) == self.n_turbines:
				self.controls_dict = {"yaw_angles": yaw_IC}
			else:
				raise TypeError(
					"yaw initial condition should be a float or "
					+ "a list of floats of length num_turbines."
				)
		else:
			self.controls_dict = {"yaw_angles": [yaw_IC] * self.n_turbines}
		
		self.norm_turbine_powers = np.ones((self.n_horizon, self.n_turbines, self.n_wind_preview_samples)) * np.nan
		self.initial_state = np.zeros((self.n_turbines,))
		self.opt_sol = None
		
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
		self.setup_pyopt_solver()
	
	# def _generate_feasible_int_ctrl_inputs(self):
	# 	"""
	# 	loop through prediction horizon time-steps
	# 	given initial yaw angles
	# 	at time-step j=0, generate all turbine control inputs (-1, 0, or 1) which will lead to feasible yaw angles in the next time-step
	# 	and compute yaw angles at j=1
	# 	repeat until j=n_horizon-1
	# 	"""
	
	def setup_pyopt_solver(self):
		Q = self.alpha
		R = 1 - self.alpha
		
		def opt_rules(opt_var_dict):
			# TODO parallelize this, check if reset is necessary
			# TODO stochastic variation: sample wind_dir/wind_mag
			# if np.all(np.isnan(self.norm_turbine_powers[j, :])):
			# for j in self.mi_model.j:
			
			funcs = {}
			sens = {"cost": {"states": [], "control_inputs": []},
			        "dyn_state_cons": {"states": [], "control_inputs": []}}
			
			
			# define objective function
			yaw_angles = np.array([[opt_var_dict["states"][(self.n_turbines * j) + i]
			                        for i in range(self.n_turbines)] for j in range(self.n_horizon)])
			
			current_freestream_measurements = [
				self.measurements_dict["wind_speeds"][0]
				* np.cos((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.)),
				self.measurements_dict["wind_speeds"][0]
				* np.sin((270. - self.measurements_dict["wind_directions"][0]) * (np.pi / 180.))
			]
			wind_preview_samples = self.wind_preview_func(current_freestream_measurements, self.n_wind_preview_samples)
			
			for j in range(self.n_horizon):
				for m in range(self.n_wind_preview_samples):
					
					self.fi.env.reinitialize(
						wind_directions=[wind_preview_samples[m]["wind_dir"][j]],
						wind_speeds=[wind_preview_samples[m]["wind_mag"][j]],
						turbulence_intensity=self.disturbance_preview["wind_ti"][j]
					)
				
					# send yaw angles
					self.fi.env.calculate_wake(yaw_angles[j, :][np.newaxis, np.newaxis, :])
					yawed_turbine_powers = self.fi.env.get_turbine_powers()
				
					self.fi.env.calculate_wake(np.zeros((1, 1, self.n_turbines)))
					noyaw_turbine_powers = self.fi.env.get_turbine_powers()
				
					# normalize power by no yaw output
					self.norm_turbine_powers[j, :, m] = np.divide(yawed_turbine_powers, noyaw_turbine_powers,
					                                           where=noyaw_turbine_powers != 0,
					                                           out=np.zeros_like(noyaw_turbine_powers))
			# TODO square in overleaf
			# TODO compute power based on sampling from wind preview
			funcs["cost"] = sum([-(0.5)*np.mean((self.norm_turbine_powers[j, i, :])**2) * Q
			                    for j in range(self.n_horizon) for i in range(self.n_turbines)]) \
			                + sum((0.5)*(opt_var_dict["control_inputs"][(self.n_turbines * j) + i])**2 * R
			                      for j in range(self.n_horizon) for i in range(self.n_turbines))
			
			for j in range(self.n_horizon):
				for i in range(self.n_turbines):
					current_idx = (self.n_turbines * j) + i
					sens["cost"]["control_inputs"].append(
						opt_var_dict["control_inputs"][(self.n_turbines * j) + i] * R
					)
					# TODO compute power derivative based on sampling from wind preview
					sens["cost"]["states"].append(
						-(self.norm_turbine_powers[j, i]) * Q
					)
			
			# define constraints
			dyn_state_cons = []
			for j in range(self.n_horizon):
				for i in range(self.n_turbines):
					current_idx = (self.n_turbines * j) + i
					
					# scaled by yaw limit
					delta_yaw = self.dt * (self.yaw_rate / self.yaw_limits[1]) * opt_var_dict["control_inputs"][current_idx]
					sens["dyn_state_cons"]["control_inputs"].append([
						-(self.dt * (self.yaw_rate / self.yaw_limits[1])) if idx == current_idx else 0
						for idx in range(self.n_horizon * self.n_turbines)
					])
					
					if j == 0:  # corresponds to time-step k=1 for states,
						# pass initial state as parameter
						dyn_state_cons = dyn_state_cons + [opt_var_dict["states"][current_idx] - (self.initial_state[i] + delta_yaw)]
						# drvt_states_ji = 1, drvt_control_inputs_ji = -(self.dt * (self.yaw_rate / self.yaw_limits[1]))
						
						sens["dyn_state_cons"]["states"].append([
							1 if idx == current_idx else 0
							for idx in range(self.n_horizon * self.n_turbines)
						])
						
						
					else:
						prev_idx = (self.n_turbines * (j - 1)) + i
						dyn_state_cons = dyn_state_cons + [
							opt_var_dict["states"][current_idx] - (opt_var_dict["states"][prev_idx] + delta_yaw)]
						# drvt_states_ji = 1, drvt_states_(j-1,i) = -1,
						# drvt_control_inputs_ji = -(self.dt * (self.yaw_rate / self.yaw_limits[1]))
						
						sens["dyn_state_cons"]["states"].append([
							1 if idx == current_idx else (-1 if idx == prev_idx else 0)
							for idx in range(self.n_horizon * self.n_turbines)
						])
			
			funcs["dyn_state_cons"] = dyn_state_cons
			
			fail = False
			
			return funcs, fail
		
		# initialize optimization object
		self.pyopt_prob = Optimization("Wake Steerng MPC", opt_rules)
		
		# add design variables
		self.pyopt_prob.addVarGroup("states", self.n_horizon * self.n_turbines,
		                            "c",  # continuous variables
		                            # lower=[self.yaw_limits[0]] * (self.n_horizon * self.n_turbines),
		                            # upper=[self.yaw_limits[1]] * (self.n_horizon * self.n_turbines),
		                            lower=[-1] * (self.n_horizon * self.n_turbines),
		                            upper=[1] * (self.n_horizon * self.n_turbines),
		                            value=[0] * (self.n_horizon * self.n_turbines))
		                            # scale=(1 / self.yaw_limits[1]))
		
		if self.control_input_domain == 'continuous':
			self.pyopt_prob.addVarGroup("control_inputs", self.n_horizon * self.n_turbines,
			                            varType="c",
			                            lower=[-1] * (self.n_horizon * self.n_turbines),
			                            upper=[1] * (self.n_horizon * self.n_turbines),
			                            value=[0] * (self.n_horizon * self.n_turbines))
		else:
			self.pyopt_prob.addVarGroup("control_inputs", self.n_horizon * self.n_turbines,
			                            varType="i",
			                            lower=[-1] * (self.n_horizon * self.n_turbines),
			                            upper=[1] * (self.n_horizon * self.n_turbines),
			                            value=[0] * (self.n_horizon * self.n_turbines))
		
		# add dynamic state equation constraints
		self.pyopt_prob.addConGroup("dyn_state_cons", self.n_horizon * self.n_turbines, lower=0.0, upper=0.0)
		
		# add objective function
		self.pyopt_prob.addObj("cost")
		
		# display optimization problem
		print(self.pyopt_prob)
	
	def pyopt_solve(self):
		# SLSQP, NSGA2, ParOpt, CONMIN, ALPSO
		opt = SLSQP(options={"IPRINT": -1, "MAXIT": 25})
		# opt_options = {}
		# opt = NSGA2(options={"xinit": 1, "PrintOut": 0})
		# opt = ParOpt(options=opt_options) # requires install paropt
		# opt = CONMIN(options={"IPRINT": 2})
		# opt = ALPSO(options=opt_options)
		
		# warm start Vars by reinitializing the solution from last time-step self.mi_model.states
		current_time_step = int(self.measurements_dict["time"] // self.dt)
		if current_time_step > 0:
			for j in range(self.n_horizon - 1):
				for i in range(self.n_turbines):
					self.pyopt_prob.variables["states"][(j * self.n_turbines) + i].value \
						= np.clip(self.opt_sol["states"][((j + 1) * self.n_turbines) + i], -1, 1)
					
					self.pyopt_prob.variables["control_inputs"][(j * self.n_turbines) + i].value \
						= np.clip(self.opt_sol["control_inputs"][((j + 1) * self.n_turbines) + i], -1, 1)
		
		# TODO pass function handle for derivatives
		sol = opt(self.pyopt_prob, sens="FD")
		
		# print(sol)
		self.opt_sol = sol.xStar
		return self.opt_sol["states"][:self.n_turbines] * self.yaw_limits[1] # solution is scaled by yaw limit
		# return (self.initial_state * self.yaw_limits[1]) \
		# 	+ self.yaw_rate * self.dt * self.opt_sol["control_inputs"][:self.n_turbines]  # solution is scaled by yaw limit
	
	def setup_pyomo_solver(self):
		# instantiate mixed-integer solver
		self.mi_model = pyo.ConcreteModel()
		
		# define turbine and time-index sets
		self.mi_model.j = pyo.Set(initialize=list(range(self.n_horizon)), doc="time-index set")
		self.mi_model.i = pyo.Set(initialize=list(range(self.n_turbines)), doc="turbine-index set")
		self.mi_model.ji = pyo.Set(within=self.mi_model.j * self.mi_model.i)
		
		# Define Parameters
		self.mi_model.Q = pyo.Param(initialize=self.alpha, domain=pyo.Reals,
		                            doc="penalty an i-th turbine negative power at j-th time-step")
		self.mi_model.P = pyo.Param(initialize=self.alpha, domain=pyo.Reals,
		                            doc="penalty an i-th turbine negative power at N-th time-step")
		self.mi_model.R = pyo.Param(initialize=1 - self.alpha, domain=pyo.Reals,
		                            doc="penalty an i-th turbine yaw rate at j-th time-step")
		self.mi_model.max_yaw_angle = pyo.Param(initialize=self.yaw_limits[1], domain=pyo.Reals,
		                                        doc="maximimum yaw angle")
		self.mi_model.min_yaw_angle = pyo.Param(initialize=self.yaw_limits[0], domain=pyo.Reals,
		                                        doc="minimum yaw angle")
		self.mi_model.yaw_rate = pyo.Param(initialize=self.yaw_rate, domain=pyo.NonNegativeReals,
		                                   doc="yaw rate of change")
		self.mi_model.sampling_time = pyo.Param(initialize=self.dt, domain=pyo.NonNegativeReals,
		                                        doc="controller sampling time")
		
		self.mi_model.initial_state = pyo.Param(self.mi_model.i,
		                                        initialize=lambda model, i: 0,
		                                        domain=pyo.Reals, mutable=True, doc="yaw angles at time-step j=0")
		
		self.mi_model.wind_mag_disturbances = pyo.Param(self.mi_model.j, domain=pyo.NonNegativeReals, mutable=True,
		                                                initialize=lambda model, j: 0,
		                                                doc="Preview of freestream wind magnintude disturbances "
		                                                    "at the j-th time-step")
		self.mi_model.wind_dir_disturbances = pyo.Param(self.mi_model.j, domain=pyo.Reals, mutable=True,
		                                                initialize=lambda model, j: 0,
		                                                doc="Preview of freestream wind direction disturbances "
		                                                    "at the j-th time-step")
		self.mi_model.wind_ti_disturbances = pyo.Param(self.mi_model.j, domain=pyo.Reals, mutable=True,
		                                               initialize=lambda model, j: 0,
		                                               doc="Preview of freestream wind turbulence intensity disturbances "
		                                                   "at the j-th time-step")
		# self.mi_model.is_power_computed = pyo.Param(self.mi_model.j, domain=pyo.Boolean, mutable=True,
		#                                             initialize=lambda model,j: False,
		#                                             doc="Boolean indicating whether reinitialize and calculate_wake "
		#                                                 "have been called for the j-th time-step")
		
		# Define optimization variables: yaw angles of all turbines at all time-steps (continuous),
		# yaw rates of change of all turbines at all time-steps excluding terminal one (integer-values)
		
		self.mi_model.control_inputs = pyo.Var(self.mi_model.j, self.mi_model.i,
		                                       domain=pyo.Integers if self.control_input_domain == 'discrete' else pyo.Reals,
		                                       bounds=(-1, 1),
		                                       initialize=lambda model, j, i: 0.0,
		                                       doc="Yaw rate-of-change integers of the i-th turbine at the j-th time-step")
		
		self.mi_model.states = pyo.Var(self.mi_model.j, self.mi_model.i, domain=pyo.Reals, bounds=self.yaw_limits,
		                               initialize=lambda model, j, i: 0.0,
		                               doc="Yaw angle of the i-th turbine at the j-th time-step")
		
		# Define outputs
		# def horizon_cost_rule(states, control_inputs, Q, R):
		def horizon_cost_rule(states):
			# TODO parallelize this, check if reset is necessary
			# TODO stochastic variation: sample wind_dir/wind_mag
			# if np.all(np.isnan(self.norm_turbine_powers[j, :])):
			# for j in self.mi_model.j:
			
			yaw_angles = np.array([[pyo.value(self.mi_model.states[j, i]) * self.yaw_limits[1] # scaled by yaw limits
			                        for i in range(self.n_turbines)] for j in range(self.n_horizon)])
			
			for j in range(self.n_horizon):
				self.fi.env.reinitialize(
					wind_directions=[self.disturbance_preview["wind_dir"][j]],
					wind_speeds=[self.disturbance_preview["wind_mag"][j]],
					turbulence_intensity=self.disturbance_preview["wind_ti"][j]
				)
				
				# send yaw angles
				self.fi.env.calculate_wake(yaw_angles[j, :][np.newaxis, np.newaxis, :])
				yawed_turbine_powers = self.fi.env.get_turbine_powers()
				
				self.fi.env.calculate_wake(np.zeros((1, 1, self.n_turbines)))
				noyaw_turbine_powers = self.fi.env.get_turbine_powers()
				
				# normalize power by no yaw output
				self.norm_turbine_powers[j, :] = np.divide(yawed_turbine_powers, noyaw_turbine_powers,
				                                           where=noyaw_turbine_powers != 0,
				                                           out=np.zeros_like(noyaw_turbine_powers))
			
			# if np.any(np.isnan(self.norm_turbine_powers[j, :])):
			# 	print(self.disturbance_preview["wind_dir"][j])
			# 	print(self.disturbance_preview["wind_mag"][j])
			# 	print(self.disturbance_preview["wind_ti"][j])
			# print(yawed_turbine_powers)
			# print(noyaw_turbine_powers)
			# print(yaw_angles[j, :])
			
			# return -self.norm_turbine_powers
			# print(self.norm_turbine_powers)
			# print(pyo.value(self.mi_model.control_inputs[0, 0]))
			
			cost = sum(-self.norm_turbine_powers[j, i] * pyo.value(self.mi_model.Q)
			           for j in range(self.n_horizon) for i in range(self.n_turbines)) \
			       + sum(pyo.value(self.mi_model.control_inputs[j, i]) * pyo.value(self.mi_model.R)
			             for j in range(self.n_horizon) for i in range(self.n_turbines))
			
			# print(np.sum(self.norm_turbine_powers), cost)
			# print(yaw_angles[0, :])
			
			# return sum(pyo.value(self.mi_model.states[0, i]) for i in range(self.n_turbines))
			return cost
		
		self.mi_model.horizon_cost_func = ExternalFunction(function=horizon_cost_rule)
		
		# self.mi_model.outputs = pyo.Expression(expr=self.mi_model.output_func(self.mi_model.states))
		
		# self.mi_model.outputs = self.mi_model.output_func(self.mi_model, self.mi_model.j, self.mi_model.i)
		# ExternalFunctionExpression(self.mi_model.j, self.mi_model.i,
		#                                    fcn='output_func',
		#                                    doc="Normalized negative power output of the i-th turbine at the j-th time-step")
		
		# Define constraints
		def dyn_yaw_angle_rule(model, j, i):
			# return pyo.Constraint.Skip
			# print(227)
			delta_yaw = self.mi_model.sampling_time * self.mi_model.yaw_rate * self.mi_model.control_inputs[j, i]
			if j == 0:  # corresponds to time-step k=1 for states,
				# pass initial state as parameter
				return model.states[j, i] == model.initial_state[i] + delta_yaw
			else:
				return model.states[j, i] == model.states[j - 1, i] + delta_yaw
		
		self.mi_model.dyn_yaw_angle = pyo.Constraint(self.mi_model.j, self.mi_model.i, rule=dyn_yaw_angle_rule)
		
		# Define objective
		# def horizon_cost_rule(model):
		# 	# if j < self.n_horizon - 1) \
		# 	# print(240)
		# 	cost = sum(model.outputs(model.states) * model.Q for j in model.j for i in model.i) \
		# 		   + sum(model.control_inputs[j, i] * model.R for j in model.j for i in model.i)
		# 	return cost
		
		# + sum(model.outputs[self.n_horizon-1, i] * model.P for i in model.i)
		# states, control_inputs, Q, R, J, I
		self.mi_model.horizon_cost = pyo.Objective(expr=self.mi_model.horizon_cost_func(
			self.mi_model.states
			# self.mi_model.control_inputs,
			# self.mi_model.Q, self.mi_model.R
		), sense=pyo.minimize)
	
	@property
	def disturbance_preview(self):
		return self._disturbance_preview
	
	@disturbance_preview.setter
	def disturbance_preview(self, disturbance_preview):
		self._disturbance_preview = disturbance_preview
	
	def _pyomo_postprocess(self, options=None, instance=None, results=None):
		self.mi_model.states.display()
		self.mi_model.control_inputs.display()
	
	def compute_controls(self):
		"""
		solve OCP to minimize objective over future horizon
		"""
		
		# get current time-step
		current_time_step = int(self.measurements_dict["time"] // self.dt)
		self.norm_turbine_powers.fill(np.nan)
		
		if current_time_step > 0.:
			# update initial state self.mi_model.initial_state
			self.initial_state = self.measurements_dict["yaw_angles"] / self.yaw_limits[1] # scaled by yaw limits
		
		yaw_star = self.pyopt_solve()
		
		self.controls_dict = {"yaw_angles": yaw_star}
	
	def pyomo_solve(self):
		# warm start Vars by reinitializing the solution from last time-step self.mi_model.states
		for j in self.mi_model.j:
			if j == self.n_horizon - 1:
				continue
			for i in self.mi_model.i:
				self.mi_model.states[j, i] = np.clip(pyo.value(self.mi_model.states[j + 1, i]), *self.yaw_limits)
				self.mi_model.control_inputs[j, i] = np.clip(pyo.value(self.mi_model.control_inputs[j + 1, i]), -1, 1)
		
		# discrete control space - mathematical program with equilibrium constraints (mpec) - mixed integer nonlinear program (minlp)
		if self.control_input_domain == 'discrete':
			opt = pyo.SolverFactory('mpec_minlp')
			results = opt.solve(self.mi_model)
			results.write()
			self.pyomo_postprocess(results=results)
			
			# for j in self.mi_model.j:
			# 	for i in self.mi_model.i:
			# 		print(f"x({j}, {i}) = {pyo.value(self.mi_model.states[j, i])}")
			# 		print(f"u({j}, {i}) = {pyo.value(self.mi_model.control_inputs[j, i])}")
			
			self.controls_dict = {"yaw_angles":
				                      np.array([pyo.value(self.mi_model.states[0, i]) for i in self.mi_model.i])}
		
		else:
			# continuous control space
			opt = pyo.SolverFactory('mpec_nlp')
			results = opt.solve(self.mi_model)
			
			self.controls_dict = {"yaw_angles":
				                      np.array([pyo.value(self.mi_model.states[0, i]) for i in self.mi_model.i])}


# opt.solve(self.mi_model, warmstart=True)

# pyomo help --solvers


if __name__ == '__main__':
	# import
	from hercules.utilities import load_yaml
	
	# options
	max_workers = 16
	# input_dict = load_yaml(sys.argv[1])
	input_dict = load_yaml("../../examples/hercules_input_001.yaml")
	
	# Load a FLORIS object for AEP calculations
	fi_mpc = ControlledFlorisInterface(yaw_limits=input_dict["controller"]["yaw_limits"],
                                         dt=input_dict["dt"],
                                         yaw_rate=input_dict["controller"]["yaw_rate"]) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	
	# instantiate wind preview distribution
	wf = WindField(**WIND_FIELD_CONFIG)
	wind_preview_generator = wf._generate_preview_noise(noise_func=np.random.multivariate_normal, noise_args=None)
	wind_preview_func = partial(generate_preview_noise, wind_preview_generator, wf.n_preview_steps)
	
	# instantiate controller
	input_dict["wind_preview_func"] = wind_preview_func
	input_dict["n_wind_preview_samples"] = 20
	ctrl_mpc = MPC(fi_mpc, input_dict=input_dict)
	
	## Simulate wind farm with interface and controller
	
	# instantiate wind field if files don't already exist
	wind_field_filenames = glob(f"{DATA_SAVE_DIR}/case_*.csv")
	if not len(wind_field_filenames):
		generate_multi_wind_ts(WIND_FIELD_CONFIG, N_CASES)
		wind_field_filenames = [f"case_{i}.csv" for i in range(N_CASES)]
	
	# if wind field data exists, get it
	wind_field_data = []
	if os.path.exists(DATA_SAVE_DIR):
		for fn in wind_field_filenames:
			wind_field_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))
	
	# instantiate wind field preview if files don't already exist
	wind_field_preview_filenames = glob(f"{DATA_SAVE_DIR}/preview_case_*.csv")
	if not len(wind_field_preview_filenames):
		generate_multi_wind_preview_ts(WIND_FIELD_CONFIG, N_CASES, wind_field_data)
		wind_field_preview_filenames = [f"preview_case_{i}.csv" for i in range(N_CASES)]
	
	# if wind field preview data exists, get it
	wind_field_preview_data = []
	if os.path.exists(DATA_SAVE_DIR):
		for fn in wind_field_preview_filenames:
			wind_field_preview_data.append(pd.read_csv(os.path.join(DATA_SAVE_DIR, fn)))
	
	# select wind field case
	case_idx = 0
	time_ts = wind_field_data[case_idx]["Time"].to_numpy()
	wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
	wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
	wind_ti_ts = [0.08] * int(EPISODE_MAX_TIME // input_dict["dt"])
	
	n_preview_steps = int(WIND_FIELD_CONFIG["wind_speed_preview_time"]
	                      // WIND_FIELD_CONFIG["wind_speed_sampling_time_step"])
	wind_mag_preview_ts = wind_field_preview_data[case_idx][
		[f"FreestreamWindMag_{i}" for i in range(n_preview_steps)]].to_numpy()
	wind_dir_preview_ts = wind_field_preview_data[case_idx][
		[f"FreestreamWindDir_{i}" for i in range(n_preview_steps)]].to_numpy()
	wind_ti_preview_ts = 0.08 * np.ones((int(EPISODE_MAX_TIME // input_dict["dt"]), n_preview_steps))
	
	yaw_angles_ts = []
	turbine_powers_ts = []
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
		ctrl_mpc.step()
		
		print(f"Time = {ctrl_mpc.measurements_dict['time']}",
		      f"Freestream Wind Direction = {wind_dir_ts[k]}",
		      f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
		      f"Turbine Wind Directions = {ctrl_mpc.measurements_dict['wind_directions']}",
		      f"Turbine Wind Magnitudes = {ctrl_mpc.measurements_dict['wind_speeds']}",
		      f"Turbine Powers = {ctrl_mpc.measurements_dict['powers']}",
		      f"Yaw Angles = {ctrl_mpc.measurements_dict['yaw_angles']}",
		      sep='\n')
		yaw_angles_ts.append(ctrl_mpc.measurements_dict['yaw_angles'])
		turbine_powers_ts.append(ctrl_mpc.measurements_dict['powers'])
	
	yaw_angles_ts = np.vstack(yaw_angles_ts)
	turbine_powers_ts = np.vstack(turbine_powers_ts)
	#
	# filt_wind_dir_ts = ctrl_mpc._first_ord_filter(wind_dir_ts, ctrl_mpc.wd_lpf_alpha)
	# filt_wind_speed_ts = ctrl_mpc._first_ord_filter(wind_mag_ts, ctrl_mpc.ws_lpf_alpha)
	fig, ax = plt.subplots(3, 1)
	ax[0].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_dir_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
	# ax[0].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_dir_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
	#            label='filtered')
	ax[0].set(title='Wind Direction [deg]', xlabel='Time')
	ax[1].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], wind_mag_ts[:int(EPISODE_MAX_TIME // input_dict["dt"])], label='raw')
	# ax[1].plot(time_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], filt_wind_speed_ts[50:int(EPISODE_MAX_TIME // input_dict["dt"])], '--',
	#            label='filtered')
	ax[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
	# ax.set_xlim((time_ts[1], time_ts[-1]))
	ax[0].legend()
	ax[2].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], yaw_angles_ts)
	fig.show()
