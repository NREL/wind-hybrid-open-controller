"""
LUT controller class. Given wind speed components u and v for some future prediction horizon at each wind turbine,
Output yaw angles equal to those optimized for that wind magnitude and direction offline.
Perform optimization upon LUT controller class initialization
"""

from controller_base import ControllerBase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import LinearNDInterpolator
from scipy.signal import lfilter

from controller_base import ControllerBase

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.config import *
from whoc.wind_field.WindField import generate_multi_wind_ts
from whoc.plotting import plot_power_vs_speed, plot_yaw_vs_dir, plot_power_vs_dir
from whoc.controllers.no_yaw_wake_steering_controller import NoYawController

from floris.tools.visualization import visualize_quiver2

from glob import glob
import re

"""
This example demonstrates how to perform a yaw optimization using parallel computing.
...
"""


class LUTController(ControllerBase):
	def __init__(self, interface, ws_lpf_omega_c=0.005, wd_lpf_omega_c=0.005, lpf_T=60, yaw_rate=0.3,
	             lut_path=None):
		super().__init__(interface)
		self.n_turbines = interface.n_turbines
		self.yaw_limits = interface.yaw_limits
		self.historic_measurements = {"wind_directions": np.zeros((0, self.n_turbines)),
		                              "wind_speeds": np.zeros((0, self.n_turbines))}
		
		self.ws_lpf_alpha = np.exp(-ws_lpf_omega_c * lpf_T)
		self.wd_lpf_alpha = np.exp(-wd_lpf_omega_c * lpf_T)
		self.yaw_rate = yaw_rate
		
		# optimize, unless passed existing lookup table
		# os.path.abspath(lut_path)
		self._optimize_lookup_table(lut_path=lut_path, load_lut=lut_path is not None)
	
	def _first_ord_filter(self, x, alpha):
		
		b = [1 - alpha]
		a = [1, -alpha]
		return lfilter(b, a, x)
	
	def _optimize_lookup_table(self, lut_path=None, load_lut=False):
		if load_lut is True and lut_path is not None and os.path.exists(lut_path):
			df_lut = pd.read_csv(lut_path, index_col=0)
			df_lut["yaw_angles_opt"] = [[float(f) for f in re.findall(r"-*\d+\.\d*", s)] for i, s in df_lut["yaw_angles_opt"].items()]
		else:
			# if csv to load from is not given, optimize
			# LUT optimizer wind field options
			wind_directions_lut = np.arange(0.0, 360.0, 3.0)
			wind_speeds_lut = np.arange(6.0, 14.0, 2.0)
			
			## Get optimized AEP, with wake steering
			
			# Load a FLORIS object for yaw optimization
			fi_lut = ControlledFlorisModel(max_workers=max_workers, yaw_limits=YAW_ANGLE_RANGE, dt=DT, yaw_rate=YAW_RATE) \
				.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
			
			fi_lut.env.reinitialize(
				wind_directions=wind_directions_lut,
				wind_speeds=wind_speeds_lut,
				turbulence_intensity=0.08  # Assume 8% turbulence intensity
			)
			
			# Pour this into a parallel computing interface
			fi_lut.parallelize()
			
			# Now optimize the yaw angles using the Serial Refine method
			df_lut = fi_lut.par_env.optimize_yaw_angles(
				minimum_yaw_angle=self.yaw_limits[0],
				maximum_yaw_angle=self.yaw_limits[1],
				Ny_passes=[5, 4],
				exclude_downstream_turbines=True,
				exploit_layout_symmetry=False,
			)
			
			# Assume linear ramp up at 5-6 m/s and ramp down at 13-14 m/s,
			# add to table for linear interpolant
			df_copy_lb = df_lut[df_lut["wind_speed"] == 6.0].copy()
			df_copy_ub = df_lut[df_lut["wind_speed"] == 13.0].copy()
			df_copy_lb["wind_speed"] = 5.0
			df_copy_ub["wind_speed"] = 14.0
			df_copy_lb["yaw_angles_opt"] *= 0.0
			df_copy_ub["yaw_angles_opt"] *= 0.0
			df_lut = pd.concat([df_copy_lb, df_lut, df_copy_ub], axis=0).reset_index(drop=True)
			
			# Deal with 360 deg wrapping: solutions at 0 deg are also solutions at 360 deg
			df_copy_360deg = df_lut[df_lut["wind_direction"] == 0.0].copy()
			df_copy_360deg["wind_direction"] = 360.0
			df_lut = pd.concat([df_lut, df_copy_360deg], axis=0).reset_index(drop=True)
			
			if lut_path is not None:
				df_lut.to_csv(lut_path)
		
		# Derive linear interpolant from solution space
		self.yaw_angles_interpolant = LinearNDInterpolator(
			points=df_lut[["wind_direction", "wind_speed"]],
			values=np.vstack(df_lut["yaw_angles_opt"]),
			fill_value=0.0,
		)
	
	def compute_controls(self):
		self.historic_measurements["wind_directions"] = np.vstack([self.historic_measurements["wind_directions"],
		                                                          self.measurements_dict["wind_directions"]])
		self.historic_measurements["wind_speeds"] = np.vstack([self.historic_measurements["wind_speeds"],
		                                                          self.measurements_dict["wind_speeds"]])
		
		filtered_wind_dir = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"][:, i],
		                                                     self.wd_lpf_alpha)
		                              for i in range(self.n_turbines)]).T
		filtered_wind_speed = np.array([self._first_ord_filter(self.historic_measurements["wind_speeds"][:, i],
		                                                       self.ws_lpf_alpha)
		                              for i in range(self.n_turbines)]).T
		
		# TODO shouldn't freestream wind speed/dir also be availalbe in measurements_dict
		# TODO filter wind speed and dir?
		yaw_setpoints = self.yaw_angles_interpolant(filtered_wind_dir[-1, 0], filtered_wind_speed[-1, 0])
		self.controls_dict = {"yaw_angles": np.clip(yaw_setpoints, *self.yaw_limits)}
		
		return None


if __name__ == "__main__":
	# Parallel options
	max_workers = 16
	
	# results wind field options
	wind_directions_tgt = np.arange(0.0, 360.0, 1.0)
	wind_speeds_tgt = np.arange(1.0, 25.0, 1.0)
	
	
	# Load a dataframe containing the wind rose information
	df_windrose, windrose_interpolant \
		= ControlledFlorisModel.load_windrose(
		windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
	
	## First, get baseline AEP, without wake steering
	
	# Load a FLORIS object for AEP calculations
	fi_noyaw = ControlledFlorisModel(max_workers=max_workers, yaw_limits=YAW_ANGLE_RANGE, dt=DT, yaw_rate=YAW_RATE) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	fi_noyaw.env.reinitialize(
		wind_directions=wind_directions_tgt,
		wind_speeds=wind_speeds_tgt,
		turbulence_intensity=0.08  # Assume 8% turbulence intensity
	)
	
	# Pour this into a parallel computing interface
	fi_noyaw.parallelize()
	
	# instantiate controller
	input_dict = {"controller": {
		"num_turbines": fi_noyaw.n_turbines, "yaw_limits": fi_noyaw.yaw_limits, "yaw_rate": fi_noyaw.yaw_rate,
		"ws_lpf_omega_c": 0.005, "wd_lpf_omega_c": 0.005, "lpf_T": 60,
		"lut_path": os.path.join(os.path.dirname(__file__), "../../examples/lut.csv"),
		"initial_conditions": {
			"yaw": 0
		}}}
	ctrl_noyaw = NoYawController(fi_noyaw, input_dict=input_dict)
	
	farm_power_noyaw, farm_aep_noyaw, farm_energy_noyaw = ControlledFlorisModel.compute_aep(fi_noyaw, ctrl_noyaw, windrose_interpolant,
	                                                                     wind_directions_tgt, wind_speeds_tgt)
	
	# instantiate interface
	fi_lut = ControlledFlorisModel(max_workers=max_workers, yaw_limits=YAW_ANGLE_RANGE, dt=DT, yaw_rate=YAW_RATE) \
		.load_floris(config_path=WIND_FIELD_CONFIG["floris_input_file"])
	
	# instantiate controller, and load lut from csv if it exists
	
	ctrl_lut = LUTController(fi_lut, input_dict=input_dict)
	
	farm_power_lut, farm_aep_lut, farm_energy_lut = ControlledFlorisModel.compute_aep(fi_lut, ctrl_lut, windrose_interpolant,
	                                                       wind_directions_tgt, wind_speeds_tgt)
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
	plot_yaw_vs_dir(ctrl_lut.yaw_angles_interpolant, ctrl_lut.n_turbines)
	plot_power_vs_dir(df, fi_lut.env.floris.flow_field.wind_directions)
	
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
	
	# select wind field case
	case_idx = 0
	time_ts = wind_field_data[case_idx]["Time"].to_numpy()
	wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
	wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
	turbulence_intensity_ts = [0.08] * int(simulation_max_time // DT)
	yaw_angles_ts = []
	fi_lut.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
		                             "wind_directions": [wind_dir_ts[0]],
		                             "turbulence_intensity": turbulence_intensity_ts[0]})
	for k, t in enumerate(np.arange(0, simulation_max_time - DT, DT)):
		print(f'Time = {t}')
		
		# feed interface with new disturbances
		fi_lut.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
		                             "wind_directions": [wind_dir_ts[k]],
		                             "turbulence_intensity": turbulence_intensity_ts[k]})
		
		# receive measurements from interface, compute control actions, and send to interface
		ctrl_lut.step()
		
		print(f"Time = {ctrl_lut.measurements_dict['time']}",
		      f"Freestream Wind Direction = {wind_dir_ts[k]}",
		      f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
		      f"Turbine Wind Directions = {ctrl_lut.measurements_dict['wind_directions']}",
		      f"Turbine Wind Magnitudes = {ctrl_lut.measurements_dict['wind_speeds']}",
		      f"Turbine Powers = {ctrl_lut.measurements_dict['powers']}",
		      f"Yaw Angles = {ctrl_lut.measurements_dict['yaw_angles']}",
		      sep='\n')
		yaw_angles_ts.append(ctrl_lut.measurements_dict['yaw_angles'])
	
	yaw_angles_ts = np.vstack(yaw_angles_ts)

	filt_wind_dir_ts = ctrl_lut._first_ord_filter(wind_dir_ts, ctrl_lut.wd_lpf_alpha)
	filt_wind_speed_ts = ctrl_lut._first_ord_filter(wind_mag_ts, ctrl_lut.ws_lpf_alpha)
	fig, ax = plt.subplots(3, 1)
	ax[0].plot(time_ts[:int(simulation_max_time // DT)], wind_dir_ts[:int(simulation_max_time // DT)], label='raw')
	ax[0].plot(time_ts[50:int(simulation_max_time // DT)], filt_wind_dir_ts[50:int(simulation_max_time // DT)], '--', label='filtered')
	ax[0].set(title='Wind Direction [deg]', xlabel='Time')
	ax[1].plot(time_ts[:int(simulation_max_time // DT)], wind_mag_ts[:int(simulation_max_time // DT)], label='raw')
	ax[1].plot(time_ts[50:int(simulation_max_time // DT)], filt_wind_speed_ts[50:int(simulation_max_time // DT)], '--', label='filtered')
	ax[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
	# ax.set_xlim((time_ts[1], time_ts[-1]))
	ax[0].legend()
	ax[2].plot(time_ts[:int(simulation_max_time // DT)-1], yaw_angles_ts)
	fig.show()