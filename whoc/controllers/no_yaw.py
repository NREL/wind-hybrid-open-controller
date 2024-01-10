import matplotlib.pyplot as plt

from controller_base import ControllerBase
import numpy as np

from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.config import *
from whoc.wind_field.WindField import generate_multi_wind_ts

from floris.tools.visualization import visualize_quiver2

from scipy.signal import lfilter

class NoYawController(ControllerBase):
	def __init__(self, interface):
		super().__init__(interface)
		self.n_turbines = interface.n_turbines
		self.yaw_limits = interface.yaw_limits
		
	def yaw_angles_interpolant(self, wind_directions, wind_speeds):
		return np.zeros((*wind_directions.shape, self.n_turbines))
	
	def compute_controls(self):
		
		self.controls_dict = {"yaw_angles": np.zeros((self.n_turbines,))}
		
		return None
