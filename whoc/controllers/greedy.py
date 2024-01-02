"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""

from controller_base import ControllerBase
import numpy as np

from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface

class GreedyController(ControllerBase):
	def __init__(self, interface):
		super().__init__(interface)
		self.n_turbines = interface.n_turbines
	
	def compute_controls(self):
		current_time = self.measurements_dict["time"]
		if current_time <= 10.0:
			yaw_setpoint = [0.0] * self.n_turbines
		else:
			yaw_setpoint = 270.0 - self.measurements_dict["wind_directions"]
			# [270.0 - wd for wd in self.measurements_dict["turbine_wind_directions"]]
		
		self.controls_dict = {"yaw_angles": yaw_setpoint}
		
		return None
		
		
if __name__ == '__main__':
	# Parallel options
	max_workers = 16
	
	# Yaw options TODO put this in config file
	yaw_limits = (-25, 25)
	dt = 1
	
	# results wind field options
	wind_directions_init = [250]
	wind_speeds_init = [16]
	turbulence_intensity_init = 0.08
	
	# Load a dataframe containing the wind rose information
	df_windrose, windrose_interpolant \
		= ControlledFlorisInterface.load_windrose(
		windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
	
	# Load a FLORIS object for AEP calculations
	fi_greedy = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=yaw_limits, dt=dt) \
		.load_floris(config_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/emgauss.yaml',
	                 wind_directions=wind_directions_init, wind_speeds=wind_speeds_init,
	                 turbulence_intensity=turbulence_intensity_init)
	
	# instantiate controller
	ctrl_greedy = GreedyController(fi_greedy)
	
	for t in range(0, 600, 1):
		# calculate wake
		ctrl_greedy.step(obs_args={})
		print(ctrl_greedy.controls_dict)
		print(ctrl_greedy.measurements_dict)
	
	
	# plot horizontal plane
	# fi_greedy.env.calculate_horizontal_plane()