import pyomo.core

from controller_base import ControllerBase
import numpy as np
import pandas as pd
from whoc.config import *
import pyomo.environ as pyo
from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.wind_field.WindField import generate_multi_wind_ts, generate_multi_wind_preview_ts
from glob import glob
import matplotlib.pyplot as plt
from pyomo.core.expr import ExternalFunctionExpression
from pyomo.core.base.external import ExternalFunction, PythonCallbackFunction # https://github.com/Pyomo/pyomo/blob/main/pyomo/core/tests/unit/test_external.py
from pyoptsparse import Optimization, SLSQP

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
	def __init__(self, interface, n_horizon: int, dt: float, alpha: float, control_input_domain: str = "continuous"):
		
		super().__init__(interface)
		
		self.n_turbines = interface.n_turbines
		self.yaw_limits = interface.yaw_limits
		self.yaw_rate = interface.yaw_rate
		self.dt = dt
		self.alpha = alpha
		self.n_horizon = n_horizon
		self.norm_turbine_powers = np.ones((self.n_horizon, self.n_turbines)) * np.nan
		
		if control_input_domain.lower() in ['discrete', 'continuous']:
			self.control_input_domain = control_input_domain.lower()
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
	
	def setup_pyopt_solver(self):
		def horizon_cost_rule(states):
			# TODO parallelize this, check if reset is necessary
			# TODO stochastic variation: sample wind_dir/wind_mag
			# if np.all(np.isnan(self.norm_turbine_powers[j, :])):
			# for j in self.mi_model.j:
			
			yaw_angles = np.array([[pyo.value(self.mi_model.states[j, i])
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
			
			cost = sum(-self.norm_turbine_powers[j, i] * pyo.value(self.mi_model.Q)
			           for j in range(self.n_horizon) for i in range(self.n_turbines)) \
			       + sum(pyo.value(self.mi_model.control_inputs[j, i]) * pyo.value(self.mi_model.R)
			             for j in range(self.n_horizon) for i in range(self.n_turbines))
			
			return cost
	
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
			
			yaw_angles = np.array([[pyo.value(self.mi_model.states[j, i])
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
			if j == 0: # corresponds to time-step k=1 for states,
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
		
		# update disturbances in model
		for j in self.mi_model.j:
			self.mi_model.wind_mag_disturbances[j] = self.disturbance_preview["wind_mag"][j]
			self.mi_model.wind_dir_disturbances[j] = self.disturbance_preview["wind_dir"][j]
			self.mi_model.wind_ti_disturbances[j] = self.disturbance_preview["wind_ti"][j]
	
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
		
		if current_time_step > 0:
			# update initial state self.mi_model.initial_state
			for i in self.mi_model.i:
				self.mi_model.initial_state[i] = self.measurements_dict["yaw_angles"][i]
			
			# warm start Vars by reinitializing the solution from last time-step self.mi_model.states
			for j in self.mi_model.j:
				if j == self.n_horizon - 1:
					continue
				for i in self.mi_model.i:
					self.mi_model.states[j, i] = np.clip(pyo.value(self.mi_model.states[j + 1, i]), *self.yaw_limits)
					self.mi_model.control_inputs[j, i] = np.clip(pyo.value(self.mi_model.control_inputs[j + 1, i]), -1, 1)
		
		#     appsi_cbc                     Automated persistent interface to
		#                                   Cbc
		#     appsi_cplex                   Automated persistent interface to
		#                                   Cplex
		#     appsi_gurobi                  Automated persistent interface to
		#                                   Gurobi
		#     appsi_highs                   Automated persistent interface to
		#                                   Highs
		#    +appsi_ipopt         3.14.13   Automated persistent interface to
		#                                   Ipopt
		#    *asl                           Interface for solvers using the AMPL
		#                                   Solver Library
		#     baron                         The BARON MINLP solver
		#     cbc                           The CBC LP/MIP solver
		#     conopt                        The CONOPT NLP solver
		#     contrib.gjh                   Interface to the AMPL GJH "solver"
		#     cp_optimizer                  Direct interface to CPLEX CP
		#                                   Optimizer
		#     cplex                         The CPLEX LP/MIP solver
		#     cplex_direct                  Direct python interface to CPLEX
		#     cplex_persistent              Persistent python interface to CPLEX
		#     cyipopt                       Cyipopt: direct python bindings to
		#                                   the Ipopt NLP solver
		#     gams                          The GAMS modeling language
		#    +gdpopt              22.5.13   The GDPopt decomposition-based
		#                                   Generalized Disjunctive Programming
		#                                   (GDP) solver
		#    +gdpopt.gloa         22.5.13   The GLOA (global logic-based outer
		#                                   approximation) Generalized
		#                                   Disjunctive Programming (GDP) solver
		#    +gdpopt.lbb          22.5.13   The LBB (logic-based branch and
		#                                   bound) Generalized Disjunctive
		#                                   Programming (GDP) solver
		#    +gdpopt.loa          22.5.13   The LOA (logic-based outer
		#                                   approximation) Generalized
		#                                   Disjunctive Programming (GDP) solver
		#    +gdpopt.ric          22.5.13   The RIC (relaxation with integer
		#                                   cuts) Generalized Disjunctive
		#                                   Programming (GDP) solver
		#    +glpk                5.0       The GLPK LP/MIP solver
		#     gurobi                        The GUROBI LP/MIP solver
		#     gurobi_direct                 Direct python interface to Gurobi
		#     gurobi_persistent             Persistent python interface to
		#                                   Gurobi
		#    +ipopt               3.14.13   The Ipopt NLP solver
		#    +mindtpy             0.1       MindtPy: Mixed-Integer Nonlinear
		#                                   Decomposition Toolbox in Pyomo
		#    +mindtpy.ecp         0.1       MindtPy: Mixed-Integer Nonlinear
		#                                   Decomposition Toolbox in Pyomo
		#    +mindtpy.fp          0.1       MindtPy: Mixed-Integer Nonlinear
		#                                   Decomposition Toolbox in Pyomo
		#    +mindtpy.goa         0.1       MindtPy: Mixed-Integer Nonlinear
		#                                   Decomposition Toolbox in Pyomo
		#    +mindtpy.oa          0.1       MindtPy: Mixed-Integer Nonlinear
		#                                   Decomposition Toolbox in Pyomo
		#     mosek                         The MOSEK LP/QP/SOCP/MIP solver
		#     mosek_direct                  Direct python interface to MOSEK
		#     mosek_persistent              Persistent python interface to
		#                                   MOSEK.
		#    +mpec_minlp                    MPEC solver transforms to a MINLP
		#    +mpec_nlp                      MPEC solver that optimizes a
		#                                   nonlinear transformation
		#    +multistart                    MultiStart solver for NLPs
		#     path                          Nonlinear MCP solver
		#    *py                            Direct python solver interfaces
		#     scip                          The SCIP LP/MIP solver
		#    +scipy.fsolve        1.11.4    fsolve: A SciPy wrapper around
		#                                   MINPACK's hybrd and hybrj algorithms
		#    +scipy.newton        1.11.4    newton: Find a zero of a scalar-
		#                                   valued function
		#    +scipy.root          1.11.4    root: Find the root of a vector
		#                                   function
		#    +scipy.secant-newton 1.11.4    secant-newton: Take a few secant
		#                                   iterations to try to converge a
		#                                   potentially linear equation quickly,
		#                                   then switch to Newton's method
		#    +trustregion         0...2...0 Trust region algorithm "solver" for
		#                                   black box/glass box optimization
		#     xpress                        The XPRESS LP/MIP solver
		#     xpress_direct                 Direct python interface to XPRESS
		#     xpress_persistent             Persistent python interface to
		#                                   Xpress
		
		# continuous control space - Interior Point Optimizer (ipopt) - large scale nonlinear optimization of continuous systems, no equality constraints
		# opt = pyo.SolverFactory('ipopt')
		# opt.solve(self.mi_model)
		# for j in self.mi_model.j:
		# 	for i in self.mi_model.i:
		# 		# print(f"x({j}, {i}) = {pyo.value(self.mi_model.states[j, i])}")
		# 		print(f"u({j}, {i}) = {pyo.value(self.mi_model.control_inputs[j, i])}")
		
		# discrete control space - Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)
		# opt = pyo.SolverFactory('mindtpy')
		# opt.solve(self.mi_model)
		# for j in self.mi_model.j:
		# 	for i in self.mi_model.i:
		# 		print(f"x({j}, {i}) = {pyo.value(self.mi_model.states[j, i])}")
		# 		print(f"u({j}, {i}) = {pyo.value(self.mi_model.control_inputs[j, i])}")
		
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
			
			# for j in self.mi_model.j:
			# 	for i in self.mi_model.i:
			# 		print(f"x({j}, {i}) = {pyo.value(self.mi_model.states[j, i])}")
			# 		print(f"u({j}, {i}) = {pyo.value(self.mi_model.control_inputs[j, i])}")
			
			# TODO will this work with continuous control inputs...
			[pyo.value(self.mi_model.control_inputs[0, i]) for i in self.mi_model.i]
			self.controls_dict = {"yaw_angles":
				                      np.array([pyo.value(self.mi_model.states[0, i]) for i in self.mi_model.i])}
			
	# opt.solve(self.mi_model, warmstart=True)
	
	# pyomo help --solvers


if __name__ == '__main__':
	# Load a FLORIS object for AEP calculations
	fi_mpc = ControlledFlorisInterface(yaw_limits=YAW_ANGLE_RANGE, dt=DT, yaw_rate=YAW_RATE) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	
	# instantiate controller
	n_horizon = 10
	dt = 60
	alpha = 0.5
	control_input_domain = "continuous"
	ctrl_mpc = MPC(fi_mpc, dt=dt, n_horizon=n_horizon, alpha=alpha, control_input_domain=control_input_domain)
	
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
	wind_ti_ts = [0.08] * int(EPISODE_MAX_TIME // DT)
	
	n_preview_steps = int(WIND_FIELD_CONFIG["wind_speed_preview_time"]
	                      // WIND_FIELD_CONFIG["wind_speed_sampling_time_step"])
	wind_mag_preview_ts = wind_field_preview_data[case_idx][
		[f"FreestreamWindMag_{i}" for i in range(n_preview_steps)]].to_numpy()
	wind_dir_preview_ts = wind_field_preview_data[case_idx][
		[f"FreestreamWindDir_{i}" for i in range(n_preview_steps)]].to_numpy()
	wind_ti_preview_ts = 0.08 * np.ones((int(EPISODE_MAX_TIME // DT), n_preview_steps))
	
	yaw_angles_ts = []
	fi_mpc.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
	                           "wind_directions": [wind_dir_ts[0]],
	                           "turbulence_intensity": wind_ti_ts[0]})
	
	for k, t in enumerate(np.arange(0, EPISODE_MAX_TIME - DT, DT)):
		print(f'Time = {t}')
		
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
		
		# print(f"Time = {ctrl_mpc.measurements_dict['time']}",
		#       f"Freestream Wind Direction = {wind_dir_ts[k]}",
		#       f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
		#       f"Turbine Wind Directions = {ctrl_mpc.measurements_dict['wind_directions']}",
		#       f"Turbine Wind Magnitudes = {ctrl_mpc.measurements_dict['wind_speeds']}",
		#       f"Turbine Powers = {ctrl_mpc.measurements_dict['powers']}",
		#       f"Yaw Angles = {ctrl_mpc.measurements_dict['yaw_angles']}",
		#       sep='\n')
		yaw_angles_ts.append(ctrl_mpc.measurements_dict['yaw_angles'])
	
	yaw_angles_ts = np.vstack(yaw_angles_ts)
	#
	# filt_wind_dir_ts = ctrl_mpc._first_ord_filter(wind_dir_ts, ctrl_mpc.wd_lpf_alpha)
	# filt_wind_speed_ts = ctrl_mpc._first_ord_filter(wind_mag_ts, ctrl_mpc.ws_lpf_alpha)
	fig, ax = plt.subplots(3, 1)
	ax[0].plot(time_ts[:int(EPISODE_MAX_TIME // DT)], wind_dir_ts[:int(EPISODE_MAX_TIME // DT)], label='raw')
	# ax[0].plot(time_ts[50:int(EPISODE_MAX_TIME // DT)], filt_wind_dir_ts[50:int(EPISODE_MAX_TIME // DT)], '--',
	#            label='filtered')
	ax[0].set(title='Wind Direction [deg]', xlabel='Time')
	ax[1].plot(time_ts[:int(EPISODE_MAX_TIME // DT)], wind_mag_ts[:int(EPISODE_MAX_TIME // DT)], label='raw')
	# ax[1].plot(time_ts[50:int(EPISODE_MAX_TIME // DT)], filt_wind_speed_ts[50:int(EPISODE_MAX_TIME // DT)], '--',
	#            label='filtered')
	ax[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
	# ax.set_xlim((time_ts[1], time_ts[-1]))
	ax[0].legend()
	ax[2].plot(time_ts[:int(EPISODE_MAX_TIME // DT) - 1], yaw_angles_ts)
	fig.show()

# Q = np.eye(ctrl_mpc.n_turbines) * alpha
# P = np.eye(ctrl_mpc.n_turbines) * alpha
# R = np.eye(ctrl_mpc.n_turbines) * (1 - alpha)
# Q = np.ones((ctrl_mpc.n_turbines,)) * alpha
# P = np.ones((ctrl_mpc.n_turbines,)) * alpha
# R = np.ones((ctrl_mpc.n_turbines, )) * (1 - alpha)
# Q(j, i) = penalty an i-th turbine negative power at j-th time-step
# Q = {(j, i): alpha for j in range(n_horizon) for i in range(fi_mpc.n_turbines)}
# mi_model.Q = pyo.Param(mi_model.j, mi_model.i, initialize=Q,
#                        doc="penalty an i-th turbine negative power at j-th time-step")
# P = {i: alpha for i in range(fi_mpc.n_turbines)}
# mi_model.P = pyo.Param(mi_model.i, initialize=P,
#                        doc="penalty an i-th turbine negative power at N-th time-step")
# R = {(j, i): (1 - alpha) for j in range(n_horizon) for i in range(fi_mpc.n_turbines)}
# mi_model.R = pyo.Param(mi_model.j, mi_model.i, initialize=R,
#                        doc="penalty an i-th turbine yaw rate at j-th time-step")
#

# mi_model.initial_state = pyo.Param(initialize=np.zeros((ctrl_mpc.n_turbines)), doc="controller sampling time")

#
# TODO convert config.py to yaml
# TODO define/import parameters Q, R, P, N, Ts, n_turbines, yaw rate of change, yaw limits

# def obj_expression()

# TODO define quadratic objective function in terms of Q, R, P, given single vector [x nu] with most recent disturbance preview measurement

# TODO define linear inequality constraints on y and nu: G, h

# TODO define dynamic state and output equality constraint: gamma(j+1) = gamma(j) + nu(j)Ts: A, b

# TODO instantiate FLORIS interface for output/power computation and define output equality constraint

# TODO instatiate wind preview object

# TODO instantiate cvx nonlinear solver object
