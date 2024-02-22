from whoc.controllers.controller_base import ControllerBase
from whoc.config import *

class NoYawController(ControllerBase):
	def __init__(self, interface, input_dict, verbose=False):
		super().__init__(interface, verbose=verbose)
		
		self.dt = input_dict["dt"]
		self.n_turbines = input_dict["controller"]["num_turbines"]
		self.turbines = range(self.n_turbines)
		self.yaw_limits = input_dict["controller"]["yaw_limits"]
		
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
		
	def yaw_offsets_interpolant(self, wind_directions, wind_speeds):
		# return np.ones((*wind_directions.shape, 1)) * self.controls_dict["yaw_angles"]
		yaw_offsets = wind_directions[:, :, np.newaxis] - self.controls_dict["yaw_angles"]
		yaw_offsets[yaw_offsets < -180.0] = 360.0 + yaw_offsets[yaw_offsets < -180.0]
		return yaw_offsets
	
	def compute_controls(self):
		
		# maintain initial yaw angles
		# self.controls_dict = {"yaw_angles": np.zeros((self.n_turbines,))}
		
		return None