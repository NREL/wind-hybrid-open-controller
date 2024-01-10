"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""
import matplotlib.pyplot as plt

from controller_base import ControllerBase
import numpy as np

from whoc.interfaces.controlled_floris_interface import ControlledFlorisInterface
from whoc.config import *
from whoc.wind_field.WindField import generate_multi_wind_ts
from whoc.plotting import plot_power_vs_speed, plot_yaw_vs_dir, plot_power_vs_dir
from whoc.controllers.no_yaw import NoYawController

from floris.tools.visualization import visualize_quiver2

from scipy.signal import lfilter

import pandas as pd
from glob import glob
# np.seterr(all="ignore")

class GreedyController(ControllerBase):
	def __init__(self, interface, lpf_omega_c=0.005, lpf_T=60, deadband_thr=8, yaw_rate=0.3):
		super().__init__(interface)
		self.n_turbines = interface.n_turbines
		self.yaw_limits = interface.yaw_limits
		self.historic_measurements = {"wind_directions": np.zeros((0, self.n_turbines))}
		
		self.lpf_alpha = np.exp(-lpf_omega_c * lpf_T)
		self.deadband_thr = deadband_thr
		self.yaw_rate = yaw_rate
		# self.filtered_measurements["wind_direction"] = []
	
	def _first_ord_filter(self, x):
		b = [1 - self.lpf_alpha]
		a = [1, -self.lpf_alpha]
		return lfilter(b, a, x)
	
	
	def yaw_angles_interpolant(self, wind_directions, wind_speeds):
		# return np.zeros((*wind_directions.shape, self.n_turbines))
		return np.array([[[270.0 - wd for i in range(self.n_turbines)] for ws in wind_speeds] for wd in wind_directions])
	
	def compute_controls(self):
		current_time = self.measurements_dict["time"]
		self.historic_measurements["wind_directions"] = np.vstack([self.historic_measurements["wind_directions"],
		                                                          self.measurements_dict["wind_directions"]])
		
		if current_time <= 60.0:
			yaw_setpoints = [0.0] * self.n_turbines
		else:
			filtered_wind_dir = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"][:, i])
			                              for i in range(self.n_turbines)]).T
			
			
			yaw_setpoints = 270.0 - filtered_wind_dir[-1, :]
		
		# yaw_positions = 270 - self.measurements_dict["yaw_angles"]  # clockwise angle from north
		# offsets = yaw_positions - filtered_wind_dir[:, -1]
			# yaw_setpoints = []
			# for i in range(self.n_turbines):
			# 	offset = filtered_wind_dir[-1, i] - yaw_positions[i]
			# 	if abs(offset) > self.deadband_thr:
			# 		if offset > 0:
			# 			# TODO shouldn't the controller signal be the yaw direction rather than a setpoint? Misha: how was this implemented before
			# 			yaw_setpoints.append(self.yaw_limits[1]) # yaw clockwise
			#
			# 		else:
			# 			yaw_setpoints.append(self.yaw_limits[0]) # yaw anticlockwise
			# 	else:
			# 		yaw_setpoints.append(0.0)
		
		# if current_time <= 60.0:
		# 	yaw_setpoint = [0.0] * self.n_turbines
		# else:
		# 	yaw_setpoint = 270.0 - self.measurements_dict["wind_directions"]
		# 	[270.0 - wd for wd in self.measurements_dict["turbine_wind_directions"]]
		
		self.controls_dict = {"yaw_angles": np.clip(yaw_setpoints, *self.yaw_limits)}
		
		return None
		
		
if __name__ == '__main__':
	# Parallel options
	max_workers = 16
	
	# results wind field options
	wind_directions_tgt = np.arange(0.0, 360.0, 1.0)
	wind_speeds_tgt = np.arange(1.0, 25.0, 1.0)
	
	# Load a dataframe containing the wind rose information
	df_windrose, windrose_interpolant \
		= ControlledFlorisInterface.load_windrose(
		windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
	
	## First, get baseline AEP, without wake steering
	
	# Load a FLORIS object for AEP calculations 
	fi_noyaw = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=YAW_ANGLE_RANGE, dt=DT) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	fi_noyaw.env.reinitialize(
		wind_directions=wind_directions_tgt,
		wind_speeds=wind_speeds_tgt,
		turbulence_intensity=0.08  # Assume 8% turbulence intensity
	)
	
	# Pour this into a parallel computing interface
	fi_noyaw.parallelize()
	
	# instantiate controller
	ctrl_noyaw = NoYawController(fi_noyaw)
	
	farm_power_noyaw, farm_aep_noyaw, farm_energy_noyaw = ControlledFlorisInterface.compute_aep(fi_noyaw, ctrl_noyaw,
	                                                                                            windrose_interpolant,
	                                                                                            wind_directions_tgt,
	                                                                                            wind_speeds_tgt)
	

	# Load a FLORIS object for AEP calculations
	fi_greedy = ControlledFlorisInterface(max_workers=max_workers, yaw_limits=YAW_ANGLE_RANGE, dt=DT) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	
	# instantiate controller
	ctrl_greedy = GreedyController(fi_greedy)
	
	farm_power_lut, farm_aep_lut, farm_energy_lut = ControlledFlorisInterface.compute_aep(fi_greedy, ctrl_greedy,
	                                                                                      windrose_interpolant,
	                                                                                      wind_directions_tgt,
	                                                                                      wind_speeds_tgt)
	aep_uplift = 100.0 * (farm_aep_lut / farm_aep_noyaw - 1)
	
	print(" ")
	print("===========================================================")
	print("Calculating optimized annual energy production (AEP)...")
	print(f"Optimized AEP: {farm_aep_lut / 1.0e9:.3f} GWh.")
	print(f"Relative AEP uplift by wake steering: {aep_uplift:.3f} %.")
	print("===========================================================")
	print(" ")
	
	# Now calculate helpful variables and then plot wind rose information
	wd_grid, ws_grid = np.meshgrid(wind_directions_tgt, wind_speeds_tgt, indexing="ij")
	freq_grid = windrose_interpolant(wd_grid, ws_grid)
	freq_grid = freq_grid / np.sum(freq_grid)
	df = pd.DataFrame({
		"wd": wd_grid.flatten(),
		"ws": ws_grid.flatten(),
		"freq_val": freq_grid.flatten(),
		"farm_power_baseline": farm_power_noyaw.flatten(),
		"farm_power_opt": farm_power_lut.flatten(),
		"farm_power_relative": farm_power_lut.flatten() / farm_power_noyaw.flatten(),
		"farm_energy_baseline": farm_energy_noyaw.flatten(),
		"farm_energy_opt": farm_energy_lut.flatten(),
		"energy_uplift": (farm_energy_lut - farm_energy_noyaw).flatten(),
		"rel_energy_uplift": farm_energy_lut.flatten() / np.sum(farm_energy_noyaw)
	})
	
	plot_power_vs_speed(df)
	plot_yaw_vs_dir(ctrl_greedy.yaw_angles_interpolant, ctrl_greedy.n_turbines)
	plot_power_vs_dir(df, fi_greedy.env.floris.flow_field.wind_directions)
	
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
	
	# select wind field case
	case_idx = 0
	time_ts = wind_field_data[case_idx]["Time"].to_numpy()
	wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
	wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
	turbulence_intensity_ts = [0.08] * int(EPISODE_MAX_TIME // DT)
	
	# Simulate wind farm with interface and controller
	fi_greedy.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
		                             "wind_directions": [wind_dir_ts[0]],
		                             "turbulence_intensity": turbulence_intensity_ts[0]})
	yaw_angles_ts = []
	for k, t in enumerate(np.arange(0, EPISODE_MAX_TIME - DT, DT)):
		print(f'Time = {t}')
		
		# feed interface with new disturbances
		fi_greedy.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
		                             "wind_directions": [wind_dir_ts[k]],
		                             "turbulence_intensity": turbulence_intensity_ts[k]})
		
		if False and k == 5:
			# TODO why does fi_greedy.env.floris.flow_field.u/v not change from 270 with changing freestream wind direction?
			# Using the FlorisInterface functions, get 2D slices.
			horizontal_plane = fi_greedy.env.calculate_horizontal_plane(
				height=90.0,
				x_resolution=20,
				y_resolution=10,
				yaw_angles=np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0.]]]),
			)
			visualize_quiver2(horizontal_plane)
			plt.show()
			
		# receive measurements from interface, compute control actions, and send to interface
		ctrl_greedy.step()
		
		print(f"Time = {ctrl_greedy.measurements_dict['time']}",
		      f"Freestream Wind Direction = {wind_dir_ts[k]}",
		      f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
		      f"Turbine Wind Directions = {ctrl_greedy.measurements_dict['wind_directions']}",
		      f"Turbine Wind Magnitudes = {ctrl_greedy.measurements_dict['wind_speeds']}",
		      f"Turbine Powers = {ctrl_greedy.measurements_dict['powers']}",
		      f"Yaw Angles = {ctrl_greedy.measurements_dict['yaw_angles']}",
		      sep='\n')
		yaw_angles_ts.append(ctrl_greedy.measurements_dict['yaw_angles'])
		# print(ctrl_greedy.controls_dict)
		# print(ctrl_greedy.measurements_dict)
	
	yaw_angles_ts = np.vstack(yaw_angles_ts)
	
	# test first order filter and plot evolution of wind direction and yaw angles
	filt_wind_dir_ts = ctrl_greedy._first_ord_filter(wind_dir_ts)
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(time_ts[:int(EPISODE_MAX_TIME // DT)], wind_dir_ts[:int(EPISODE_MAX_TIME // DT)], label='raw')
	ax[0].plot(time_ts[50:int(EPISODE_MAX_TIME // DT)], filt_wind_dir_ts[50:int(EPISODE_MAX_TIME // DT)], '--', label='filtered')
	ax[0].set(title='Wind Direction [deg]', xlabel='Time [s]')
	# ax.set_xlim((time_ts[1], time_ts[-1]))
	ax[0].legend()
	ax[1].plot(time_ts[:int(EPISODE_MAX_TIME // DT)-1], yaw_angles_ts)
	fig.show()
	
	# TODO plot video of horizontal plane
	# fi_greedy.env.calculate_horizontal_plane()