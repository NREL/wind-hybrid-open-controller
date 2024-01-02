from controller_base import ControllerBase
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds
from cvxopt.solvers import qp


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
			self.Hhat = (self.Hhat + self.Hhat.T) / 2 # ensure symmetrical
			self.fhat = dyn_model.Bhat.T @ self.Qhat @ dyn_model.Ahat # @ x0
		
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
				
				self.Ahat = self.Ahat + np.kron(ej, np.linalg.matrix_power(kwargs['A'], j+1))
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
	def __init__(self, objective, bounds: Bounds, dyn_model: DynamicModel, n_horizon: int, dt: float, interface):
		
		super().__init__(interface)
		self.bounds = bounds
		self._state_horizon = None # optimized states over current prediction horizon
		self._output_horizon = None # modeled outputs over current prediction horizon
		self._ctrl_inpt_horizon = None # optimized control inputs over current prediction horizon
		self._dist_horizon = None # preview or prediction of disturbance
		
		self.objective = objective
		self.dyn_model = dyn_model
	
	def solve(self):
		"""
		solve OCP to minimize objective over future horizon
		"""
		pass
	

