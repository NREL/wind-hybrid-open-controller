# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import numpy as np
import pandas as pd
import os
import re

from memory_profiler import profile

from whoc.controllers.controller_base import ControllerBase
from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel

from scipy.interpolate import LinearNDInterpolator
from scipy.signal import lfilter

from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

class LookupBasedWakeSteeringController(ControllerBase):
	def __init__(self, interface, input_dict, verbose=False, **kwargs):
		super().__init__(interface, verbose=verbose)
		self.simulation_dt = input_dict["dt"]
		self.dt = input_dict["controller"]["dt"]  # Won't be needed here, but generally good to have
		self.n_turbines = interface.n_turbines #input_dict["controller"]["num_turbines"]
		self.turbines = range(self.n_turbines)
		self.historic_measurements = {"wind_directions": [],
									  "wind_speeds": []}
		self.filtered_measurements = {"wind_directions": [],
									  "wind_speeds": []}
		# self.ws_lpf_alpha = np.exp(-input_dict["controller"]["ws_lpf_omega_c"] * input_dict["controller"]["lpf_T"])
		self.use_filt = input_dict["controller"]["use_lut_filtered_wind_dir"]
		self.lpf_time_const = input_dict["controller"]["lpf_time_const"]
		self.lpf_start_time = input_dict["controller"]["lpf_start_time"]
		self.lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["dt"])
		self.deadband_thr = input_dict["controller"]["deadband_thr"]
		self.floris_input_file = input_dict["controller"]["floris_input_file"]
		self.yaw_limits = input_dict["controller"]["yaw_limits"]
		self.yaw_rate = input_dict["controller"]["yaw_rate"]
		self.yaw_increment = input_dict["controller"]["yaw_increment"]
		self.max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 16

		self.wind_dir_ts = kwargs["wind_dir_ts"]
		self.wind_mag_ts = kwargs["wind_mag_ts"]

		self._last_measured_time = None
		self.is_yawing = np.array([False for _ in range(self.n_turbines)])

		# Handle yaw optimizer object
		if "df_yaw" in kwargs:
			self.wake_steering_interpolant = get_yaw_angles_interpolant(kwargs["df_yaw"])
		else:
			# optimize, unless passed existing lookup table
			# os.path.abspath(lut_path)
			self._optimize_lookup_table(lut_path=input_dict["controller"]["lut_path"], generate_lut=input_dict["controller"]["generate_lut"])
		# Set initial conditions
		self.yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
		if hasattr(self.yaw_IC, "__len__"):
			if len(self.yaw_IC) == self.n_turbines:
				self.controls_dict = {"yaw_angles": np.array(self.yaw_IC)}
			else:
				raise TypeError(
					"yaw initial condition should be a float or "
					+ "a list of floats of length num_turbines."
				)
		else:
			self.controls_dict = {"yaw_angles": np.array([self.yaw_IC] * self.n_turbines)}

		# For startup
		self.previous_target_yaw_setpoints = self.controls_dict["yaw_angles"]
	
	def _first_ord_filter(self, x, alpha):
		
		b = [1 - alpha]
		a = [1, -alpha]
		return lfilter(b, a, x)
	
	# @profile
	def _optimize_lookup_table(self, lut_path=None, generate_lut=True):
		if not generate_lut and lut_path is not None and os.path.exists(lut_path):
			df_lut = pd.read_csv(lut_path, index_col=0)
			df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
		else:
			# if csv to load from is not given, optimize
			# LUT optimizer wind field options
			wind_directions_lut = np.arange(0.0, 360.0, 3.0)
			wind_speeds_lut = np.arange(6.0, 22.0, 2.0)
			
			## Get optimized AEP, with wake steering
			
			# Load a FLORIS object for yaw optimization
			
			fi_lut = ControlledFlorisModel(yaw_limits=self.yaw_limits, dt=self.dt,
											   yaw_rate=self.yaw_rate, floris_version='v4', config_path=self.floris_input_file)
			
			wd_grid, ws_grid = np.meshgrid(wind_directions_lut, wind_speeds_lut, indexing="ij")
			fi_lut.env.set(
				wind_directions=wd_grid.flatten(),
				wind_speeds=ws_grid.flatten(),
				turbulence_intensities=[fi_lut.env.core.flow_field.turbulence_intensities[0]] * len(ws_grid.flatten())
			)
			
			# Pour this into a parallel computing interface
			# fi_lut.parallelize()
			
			yaw_opt = YawOptimizationSR(fi_lut.env, 
										minimum_yaw_angle=self.yaw_limits[0],
										maximum_yaw_angle=self.yaw_limits[1],
										# yaw_angles_baseline=np.zeros((len(wind_directions_lut), len(wind_speeds_lut), self.n_turbines)),
										Ny_passes=[12, 10, 8, 4])
			df_lut = yaw_opt.optimize()
			
			# Assume linear ramp up at 5-6 m/s and ramp down at 13-14 m/s,
			# add to table for linear interpolant
			df_copy_lb = df_lut[df_lut["wind_speed"] == 6.0].copy()
			df_copy_ub = df_lut[df_lut["wind_speed"] == 13.0].copy()
			df_copy_lb["wind_speed"] = 5.0
			df_copy_lb["wind_speed"] = 14.0
			df_copy_lb["yaw_angles_opt"] *= 0.0
			df_copy_ub["yaw_angles_opt"] *= 0.0
			df_lut = pd.concat([df_copy_lb, df_lut, df_copy_ub], axis=0).reset_index(drop=True)
			
			# Deal with 360 deg wrapping: solutions at 0 deg are also solutions at 360 deg
			df_copy_360deg = df_lut[df_lut["wind_direction"] == 0.0].copy()
			df_copy_360deg["wind_direction"] = 360.0
			df_lut = pd.concat([df_lut, df_copy_360deg], axis=0).reset_index(drop=True)
			# ['wind_direction', 'wind_speed', 'turbulence_intensity',
			#        'yaw_angles_opt', 'farm_power_opt', 'farm_power_baseline']
			
			if lut_path is not None:
				df_lut.to_csv(lut_path)

		# pd.unique(df_lut.iloc[np.where(np.any(np.vstack(df_lut["yaw_angles_opt"].array) != 0, axis=1))[0]]["wind_direction"])
		# Derive linear interpolant from solution space
		self.wake_steering_interpolant = LinearNDInterpolator(
			points=df_lut[["wind_direction", "wind_speed"]].values,
			values=np.vstack(df_lut["yaw_angles_opt"].values),
			fill_value=0.0,
		)
	
	def compute_controls(self):
		# TODO update LUT for turbine breakdown
		if (self._last_measured_time is not None) and self._last_measured_time == self.measurements_dict["time"]:
			return

		if self.verbose:
			print(f"self._last_measured_time == {self._last_measured_time}")
			print(f"self.measurements_dict['time'] == {self.measurements_dict['time']}")

		self._last_measured_time = self.measurements_dict["time"]

		self.current_time = self.measurements_dict["time"]

		if self.verbose:
			print(f"self.current_time == {self.current_time}")

		current_wind_direction = self.measurements_dict["amr_wind_direction"]
		current_wind_speed = self.measurements_dict["amr_wind_speed"]
		# current_wind_direction = self.wind_dir_ts[int(self.current_time // self.simulation_dt)]
		# current_wind_speed = self.wind_mag_ts[int(self.current_time // self.simulation_dt)]
		if self.use_filt:
			self.historic_measurements["wind_directions"].append(current_wind_direction)
			self.historic_measurements["wind_directions"] = self.historic_measurements["wind_directions"][-int((self.lpf_time_const // self.simulation_dt) * 1e3):]

		# if current_time < 2 * self.simulation_dt:
		if len(self.measurements_dict["wind_directions"]) == 0 or np.all(np.isclose(self.measurements_dict["wind_directions"], 0)):
			# yaw angles will be set to initial values
			if self.verbose:
				print("Bad wind direction measurement received, reverting to previous measurement.")
		
		elif (abs(self.current_time % self.simulation_dt) == 0.0) or (np.all(self.controls_dict["yaw_angles"] == self.yaw_IC) and self.current_time == self.simulation_dt * 2):
			# if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
			if self.verbose:
				print(f"unfiltered wind direction = {current_wind_direction}")
			if self.current_time < self.lpf_start_time or not self.use_filt:
				wind_dir = current_wind_direction
				wind_speed = current_wind_speed
			else:
				# use filtered wind direction and speed, NOTE historic_measurments includes controller_dt steps into the future such that we can run simulation in time batches
				wind_dir = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"],
																	self.lpf_alpha)])
				self.filtered_measurements["wind_directions"].append(wind_dir[0, -1])
				wind_dir = wind_dir[0, -1]
				wind_speed = current_wind_speed
				# wind_speeds = 8.0
			
			if self.verbose:
				print(f"{'filtered' if self.use_filt else 'unfiltered'} wind direction = {wind_dir}")
			
			current_yaw_setpoints = self.controls_dict["yaw_angles"]
			
			# flip the boolean value of those turbines which were actively yawing towards a previous setpoint, but now have reached that setpoint
			if self.verbose and any(self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)):
				print(f"LUT Controller turbines {np.where(self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints))[0]} have reached their target setpoint at time {self.current_time}")
			
			self.is_yawing[self.is_yawing & (current_yaw_setpoints == self.previous_target_yaw_setpoints)] = False

			new_yaw_setpoints = np.array(current_yaw_setpoints)

			target_yaw_offsets = self.wake_steering_interpolant(wind_dir, wind_speed)
			target_yaw_setpoints = np.rint((wind_dir - target_yaw_offsets) / self.yaw_increment) * self.yaw_increment

			# change the turbine yaw setpoints that have surpassed the threshold difference AND are not already yawing towards a previous setpoint
			is_target_changing = (np.abs(target_yaw_setpoints - current_yaw_setpoints) > self.deadband_thr) & ~self.is_yawing

			if self.verbose and any(is_target_changing):
				print(f"LUT Controller starting to yaw turbines {np.where(is_target_changing)[0]} from {current_yaw_setpoints[is_target_changing]} to {target_yaw_setpoints[is_target_changing]} at time {self.current_time}")
			
			if self.verbose and any(self.is_yawing):
				print(f"LUT Controller continuing to yaw turbines {np.where(self.is_yawing)[0]} from {current_yaw_setpoints[self.is_yawing]} to {self.previous_target_yaw_setpoints[self.is_yawing]} at time {self.current_time}")
			
			# else:
			# 	print(f"LUT Controller current_setpoints = {current_yaw_setpoints}, \n previous_target_yaw_setpoints = {self.previous_target_yaw_setpoints}, \n target_setpoints={target_yaw_setpoints}")
			
			new_yaw_setpoints[is_target_changing] = target_yaw_setpoints[is_target_changing]
			new_yaw_setpoints[self.is_yawing] = self.previous_target_yaw_setpoints[self.is_yawing].copy()

			# stores target setpoints from prevoius compute_controls calls, update only those elements which are not already yawing towards a previous setpoint
			self.previous_target_yaw_setpoints = np.rint(new_yaw_setpoints / self.yaw_increment) * self.yaw_increment

			self.is_yawing[is_target_changing] = True

			constrained_yaw_setpoints = np.clip(new_yaw_setpoints, current_yaw_setpoints - self.simulation_dt * self.yaw_rate, current_yaw_setpoints + self.simulation_dt * self.yaw_rate)
			
			# if np.all(np.diff(constrained_yaw_setpoints) == 0) and not np.all(np.diff(new_yaw_setpoints) == 0):
			# 	print(f"Note: all yaw angles have been constrained by the yaw rate equally at time {self.current_time}")
			
			# if not len(is_target_changing):
			# 	print(f"Note: no yaw angle setpoints surpass the deadband threshold at time {self.current_time}")

			constrained_yaw_setpoints = np.rint(constrained_yaw_setpoints / self.yaw_increment) * self.yaw_increment

			self.controls_dict = {"yaw_angles": list(constrained_yaw_setpoints)}
		
		return None

def get_yaw_angles_interpolant(df_opt, ramp_up_ws=[4, 5], ramp_down_ws=[10, 12], minimum_yaw_angle=None, maximum_yaw_angle=None):
	"""Create an interpolant for the optimal yaw angles from a dataframe
	'df_opt', which contains the rows 'wind_direction', 'wind_speed',
	'turbulence_intensity', and 'yaw_angles_opt'. This dataframe is typically
	produced automatically from a FLORIS yaw optimization using Serial Refine
	or SciPy. One can additionally apply a ramp-up and ramp-down region
	to transition between non-wake-steering and wake-steering operation.

	Args:
		df_opt (pd.DataFrame): Dataframe containing the rows 'wind_direction',
		'wind_speed', 'turbulence_intensity', and 'yaw_angles_opt'.
		ramp_up_ws (list, optional): List with length 2 depicting the wind
		speeds at which the ramp starts and ends, respectively, on the lower
		end. This variable defaults to [4, 5], meaning that the yaw offsets are
		zero at and below 4 m/s, then linearly transition to their full offsets
		at 5 m/s, and continue to be their full offsets past 5 m/s. Defaults to
		[4, 5].
		ramp_down_ws (list, optional): List with length 2 depicting the wind
		speeds at which the ramp starts and ends, respectively, on the higher
		end. This variable defaults to [10, 12], meaning that the yaw offsets are
		full at and below 10 m/s, then linearly transition to zero offsets
		at 12 m/s, and continue to be zero past 12 m/s. Defaults to [10, 12].

	Returns:
		LinearNDInterpolator: An interpolant function which takes the inputs
		(wind_directions, wind_speeds, turbulence_intensities), all of equal
		dimensions, and returns the yaw angles for all turbines. This function
		incorporates the ramp-up and ramp-down regions.
	"""

	# Load data and set up a linear interpolant
	points = df_opt[["wind_direction", "wind_speed", "turbulence_intensity"]]
	values = np.vstack(df_opt["yaw_angles_opt"])

	# Derive maximum and minimum yaw angle (deg)
	if minimum_yaw_angle is None:
		minimum_yaw_angle = np.min(values)
	if maximum_yaw_angle is None:
		maximum_yaw_angle = np.max(values)

	# Expand wind direction range to cover 0 deg to 360 deg
	points_copied = points[points["wind_direction"] == 0.0].copy()
	points_copied.loc[points_copied.index, "wind_direction"] = 360.0
	values_copied = values[points["wind_direction"] == 0.0, :]
	points = np.vstack([points, points_copied])
	values = np.vstack([values, values_copied])

	# Copy lowest wind speed / TI solutions to -1.0 to create lower bound
	for col in [1, 2]:
		ids_to_copy_lb = points[:, col] == np.min(points[:, col])
		points_copied = np.array(points[ids_to_copy_lb, :], copy=True)
		values_copied = np.array(values[ids_to_copy_lb, :], copy=True)
		points_copied[:, col] = -1.0  # Lower bound
		points = np.vstack([points, points_copied])
		values = np.vstack([values, values_copied])

		# Copy highest wind speed / TI solutions to 999.0
		ids_to_copy_ub = points[:, col] == np.max(points[:, col])
		points_copied = np.array(points[ids_to_copy_ub, :], copy=True)
		values_copied = np.array(values[ids_to_copy_ub, :], copy=True)
		points_copied[:, col] = 999.0  # Upper bound
		points = np.vstack([points, points_copied])
		values = np.vstack([values, values_copied])

	# Now create a linear interpolant for the yaw angles
	interpolant = LinearNDInterpolator(
		points=points,
		values=values,
		fill_value=np.nan
	)

	# Now create a wrapper function with ramp-up and ramp-down
	def interpolant_with_ramps(wd_array, ws_array, ti_array=None):
		# Deal with missing ti_array
		if ti_array is None:
			ti_ref = float(np.median(interpolant.points[:, 2]))
			ti_array = np.ones(np.shape(wd_array), dtype=float) * ti_ref

		# Format inputs
		wd_array = np.array(wd_array, dtype=float)
		ws_array = np.array(ws_array, dtype=float)
		ti_array = np.array(ti_array, dtype=float)
		yaw_angles = interpolant(wd_array, ws_array, ti_array)
		yaw_angles = np.array(yaw_angles, dtype=float)

		# Define ramp down factor
		rampdown_factor = np.interp(
			x=ws_array,
			xp=[0.0, *ramp_up_ws, *ramp_down_ws, 999.0],
			fp=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
		)

		# Saturate yaw offsets to threshold
		axis = len(np.shape(yaw_angles)) - 1
		nturbs = np.shape(yaw_angles)[-1]
		yaw_lb = np.expand_dims(
			minimum_yaw_angle * rampdown_factor, axis=axis
		).repeat(nturbs, axis=axis)
		yaw_ub = np.expand_dims(
			maximum_yaw_angle * rampdown_factor, axis=axis
		).repeat(nturbs, axis=axis)

		return np.clip(yaw_angles, yaw_lb, yaw_ub)

	return interpolant_with_ramps


# if __name__ == "__main__":
	
#     # # Clear old log files for clarity
#     # rm loghercules logfloris
#     # # Set up the helics broker
#     # helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT &
#     # python3 hercules_runscript.py hercules_input_000.yaml >> loghercules 2>&1 & # Start the controller center and pass in input file
#     # python3 floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris 2>&1
	
#     with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml")) as fp:
#         wind_field_config = yaml.safe_load(fp)

#     # Parallel options
#     max_workers = 16

#     # input_dict = load_yaml(sys.argv[1])
#     input_dict = load_yaml("../../examples/hercules_input_001.yaml")
#     interface = HerculesADInterface(input_dict)
#     controller = LookupBasedWakeSteeringController(interface, input_dict, max_workers=max_workers)
#     py_sims = PySims(input_dict)
	
#     emulator = Emulator(controller, py_sims, input_dict)
#     emulator.run_helics_setup()
#     emulator.enter_execution(function_targets=[], function_arguments=[[]])
	
#     amr_input_file = "amr_input.inp"
#     amr_standin_data_file = "amr_standin_data.csv"
#     launch_floris(amr_input_file, amr_standin_data_file)

	
#     # results wind field options
#     wind_directions_tgt = np.arange(0.0, 360.0, 1.0)
#     wind_speeds_tgt = np.arange(1.0, 25.0, 1.0)
	
#     # Load a dataframe containing the wind rose information
#     df_windrose, windrose_interpolant \
#         = ControlledFlorisModel.load_windrose(
#         windrose_path='/Users/aoifework/Documents/toolboxes/floris/examples/inputs/wind_rose.csv')
	
#     ## First, get baseline AEP, without wake steering
	
#     # Load a FLORIS object for AEP calculations
#     fi_noyaw = ControlledFlorisModel(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
#                                          dt=input_dict["dt"],
#                                          yaw_rate=input_dict["controller"]["yaw_rate"]) \
#         .load_floris(config_path=input_dict["controller"]["floris_input_file"])
#     fi_noyaw.env.reinitialize(
#         wind_directions=wind_directions_tgt,
#         wind_speeds=wind_speeds_tgt        # turbulence_intensity=0.08  # Assume 8% turbulence intensity
#     )
	
#     # Pour this into a parallel computing interface
#     fi_noyaw.parallelize()
	
#     ctrl_noyaw = NoYawController(fi_noyaw, input_dict=input_dict)
	
#     farm_power_noyaw, farm_aep_noyaw, farm_energy_noyaw = ControlledFlorisModel.compute_aep(fi_noyaw, ctrl_noyaw,
#                                                                                                 windrose_interpolant,
#                                                                                                 wind_directions_tgt,
#                                                                                                 wind_speeds_tgt)
	
#     # instantiate interface
#     fi_lut = ControlledFlorisModel(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
#                                        dt=input_dict["dt"],
#                                        yaw_rate=input_dict["controller"]["yaw_rate"]) \
#         .load_floris(config_path=input_dict["controller"]["floris_input_file"])
	
#     # instantiate controller, and load lut from csv if it exists
#     input_dict["controller"]["floris_input_file"] = input_dict["controller"]["floris_input_file"]
#     ctrl_lut = LookupBasedWakeSteeringController(fi_lut, input_dict=input_dict)
	
#     farm_power_lut, farm_aep_lut, farm_energy_lut = ControlledFlorisModel.compute_aep(fi_lut, ctrl_lut,
#                                                                                           windrose_interpolant,
#                                                                                           wind_directions_tgt,
#                                                                                           wind_speeds_tgt)
#     aep_uplift = 100.0 * (farm_aep_lut / farm_aep_noyaw - 1)
	
#     print(" ")
#     print("===========================================================")
#     print("Calculating optimized annual energy production (AEP)...")
#     print(f"Optimized AEP: {farm_aep_lut / 1.0e9:.3f} GWh.")
#     print(f"Relative AEP uplift by wake steering: {aep_uplift:.3f} %.")
#     print("===========================================================")
#     print(" ")
	
#     # Now calculate helpful variables and then plot wind rose information
#     wd_grid, ws_grid = np.meshgrid(wind_directions_tgt, wind_speeds_tgt, indexing="ij")
#     freq_grid = windrose_interpolant(wd_grid, ws_grid)
#     freq_grid = freq_grid / np.sum(freq_grid)
#     df = pd.DataFrame({
#         "wd": wd_grid.flatten(),
#         "ws": ws_grid.flatten(),
#         "freq_val": freq_grid.flatten(),
#         "farm_power_baseline": farm_power_noyaw.flatten(),
#         "farm_power_opt": farm_power_lut.flatten(),
#         "farm_power_relative": farm_power_lut.flatten() / farm_power_noyaw.flatten(),
#         "farm_energy_baseline": farm_energy_noyaw.flatten(),
#         "farm_energy_opt": farm_energy_lut.flatten(),
#         "energy_uplift": (farm_energy_lut - farm_energy_noyaw).flatten(),
#         "rel_energy_uplift": farm_energy_lut.flatten() / np.sum(farm_energy_noyaw)
#     })
	
#     plot_power_vs_speed(df)
#     plot_yaw_vs_dir(ctrl_lut.wake_steering_interpolant, ctrl_lut.n_turbines)
#     plot_power_vs_dir(df, fi_lut.env.floris.flow_field.wind_directions)
	
#     ## Simulate wind farm with interface and controller
#     # instantiate wind field if files don't already exist
#     wind_field_filenames = glob(f"{wind_field_config['data_save_dir']}/case_*.csv")
#     if not len(wind_field_filenames):
#         generate_multi_wind_ts(wind_field_config)
#         wind_field_filenames = [f"case_{i}.csv" for i in range(wind_field_config["n_wind_field_cases"])]
	
#     # if wind field data exists, get it
#     wind_field_data = []
#     if os.path.exists(wind_field_config["data_save_dir"]):
#         for fn in wind_field_filenames:
#             wind_field_data.append(pd.read_csv(os.path.join(wind_field_config["data_save_dir"], fn)))
	
#     # select wind field case
#     case_idx = 0
#     time_ts = wind_field_data[case_idx]["Time"].to_numpy()
#     wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
#     wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
#     turbulence_intensity_ts = [0.08] * int(wind_field_config["simulation_max_time"] // input_dict["dt"])
#     yaw_angles_ts = []
#     fi_lut.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
#                                "wind_directions": [wind_dir_ts[0]],
#                                "turbulence_intensity": turbulence_intensity_ts[0]})
#     for k, t in enumerate(np.arange(0, wind_field_config["simulation_max_time"] - input_dict["dt"], input_dict["dt"])):
#         print(f'Time = {t}')
		
#         # feed interface with new disturbances
#         fi_lut.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
#                                   "wind_directions": [wind_dir_ts[k]],
#                                   "turbulence_intensity": turbulence_intensity_ts[k]})
		
#         # receive measurements from interface, compute control actions, and send to interface
#         ctrl_lut.step()
		
#         print(f"Time = {ctrl_lut.measurements_dict['time']}",
#               f"Freestream Wind Direction = {wind_dir_ts[k]}",
#               f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
#               f"Turbine Wind Directions = {ctrl_lut.measurements_dict['wind_directions']}",
#               f"Turbine Wind Magnitudes = {ctrl_lut.measurements_dict['wind_speeds']}",
#               f"Turbine Powers = {ctrl_lut.measurements_dict['powers']}",
#               f"Yaw Angles = {ctrl_lut.measurements_dict['yaw_angles']}",
#               sep='\n')
#         yaw_angles_ts.append(ctrl_lut.measurements_dict['yaw_angles'])
	
#     yaw_angles_ts = np.vstack(yaw_angles_ts)
	
#     filt_wind_dir_ts = ctrl_lut._first_ord_filter(wind_dir_ts, ctrl_lut.wd_lpf_alpha)
#     filt_wind_speed_ts = ctrl_lut._first_ord_filter(wind_mag_ts, ctrl_lut.ws_lpf_alpha)
#     fig, ax = plt.subplots(3, 1)
#     ax[0].plot(time_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], wind_dir_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], label='raw')
#     ax[0].plot(time_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], filt_wind_dir_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], '--',
#                label='filtered')
#     ax[0].set(title='Wind Direction [deg]', xlabel='Time')
#     ax[1].plot(time_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], wind_mag_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], label='raw')
#     ax[1].plot(time_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], filt_wind_speed_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], '--',
#                label='filtered')
#     ax[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
#     # ax.set_xlim((time_ts[1], time_ts[-1]))
#     ax[0].legend()
#     ax[2].plot(time_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"]) - 1], yaw_angles_ts)
#     fig.show()
