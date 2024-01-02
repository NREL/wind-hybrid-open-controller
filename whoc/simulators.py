import numpy as np
from floris.tools import FlorisInterface, ParallelComputingInterface
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

class SimulatorBase:
	def __init__(self):
		self._state_trajectory = None  # actual measurements of states over implementation
		self._output_trajectory = None  # actual measurements of outputs over implementation
		self._ctrl_inpt_trajectory = None # optimized control inputs over implementation, selected from first time step of each horizon
		self._dist_trajectory = None  # actual measurements of disturbances over implementation
	
	def reset(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Resets the environment to an initial state, required before calling step.
		Returns the first agent observation for an episode and information, i.e. metrics, debug info.
		Returns initial state and output.
		"""
		raise NotImplementedError
	def step(self, ctrl_inpt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
		Updates an environment with actions returning the next agent observation, the reward for taking that actions,
		if the environment has terminated or truncated due to the latest action and information from the environment
		about the step, i.e. metrics, debug info. Returns state and output.
		"""
		raise NotImplementedError
	
	def render(self):
		"""
		Renders the environments to help visualise what the agent see,
		examples modes are “human”, “rgb_array”, “ansi” for text.
		"""
		raise NotImplementedError
	
	def close(self):
		"""
		Closes the environment, important when external software is used, i.e. pygame for rendering, databases
		"""
		raise NotImplementedError
	
class FlorisSimulator(SimulatorBase):
	def __init__(self, max_workers, yaw_limits):
		super().__init__()
		self.max_workers = max_workers
		self.yaw_limits = yaw_limits
	
	def load_floris(self, config_path):
		# Load the default example floris object
		self.env = FlorisInterface(config_path)  # GCH model matched to the default "legacy_gauss" of V2
		# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model
		
		# Specify wind farm layout and update in the floris object
		N = 4  # number of turbines per row and per column
		X, Y = np.meshgrid(
			5.0 * self.env.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
			5.0 * self.env.floris.farm.rotor_diameters_sorted[0][0][0] * np.arange(0, N, 1),
		)
		self.env.reinitialize(layout_x=X.flatten(), layout_y=Y.flatten())
		
		return self.env
	
	@classmethod
	def load_windrose(cls, windrose_path):
		# Grab a linear interpolant from this wind rose
		df = pd.read_csv(windrose_path)
		interp = LinearNDInterpolator(points=df[["wd", "ws"]], values=df["freq_val"], fill_value=0.0)
		return df, interp
	
	def parallelize(self):
		# Pour this into a parallel computing interface
		parallel_interface = "concurrent"
		self.env = ParallelComputingInterface(
			fi=self.env,
			max_workers=self.max_workers,
			n_wind_direction_splits=self.max_workers,
			n_wind_speed_splits=1,
			interface=parallel_interface,
			print_timings=True,
		)
	
	def compute_aep(self, wind_directions, wind_speeds, windrose_interpolant):
		# Calculate frequency of occurrence for each bin and normalize sum to 1.0
		wd_grid, ws_grid = np.meshgrid(wind_directions, wind_speeds, indexing="ij")
		freq_grid = windrose_interpolant(wd_grid, ws_grid)
		freq_grid = freq_grid / np.sum(freq_grid)  # Normalize to 1.0
		
		# Calculate farm power greedy control
		farm_power_greedy = self.env.get_farm_power()
		aep = np.sum(24 * 365 * np.multiply(farm_power_greedy, freq_grid))
		return aep