"""
Greedy controller class. Given wind speed components ux and uy for some future prediction horizon at each wind turbine,
Output yaw angles equal to those wind directions
"""
# TODO should sampling frequency be every second for greedy controller? MISHA
import numpy as np

from scipy.signal import lfilter

from whoc.controllers.controller_base import ControllerBase

# from floris.tools.visualization import visualize_quiver2

# np.seterr(all="ignore")

class GreedyController(ControllerBase):
	def __init__(self, interface, input_dict, verbose=False, **kwargs):
		super().__init__(interface)
		self.n_turbines = input_dict["controller"]["num_turbines"]
		self.yaw_limits = input_dict["controller"]["yaw_limits"]
		self.yaw_rate = input_dict["controller"]["yaw_rate"]
		self.yaw_increment = input_dict["controller"]["yaw_increment"]
		self.simulation_dt = input_dict["dt"]
		self.dt = input_dict["controller"]["dt"]
		self.turbines = range(self.n_turbines)
		self.historic_measurements = {"wind_directions": np.zeros((0, self.n_turbines))}
		
		self.lpf_time_const = input_dict["controller"]["lpf_time_const"]
		self.lpf_alpha = np.exp(-(1 / input_dict["controller"]["lpf_time_const"]) * input_dict["dt"])
		self.deadband_thr = input_dict["controller"]["deadband_thr"]
		self.use_filt = input_dict["controller"]["use_filtered_wind_dir"]

		# Set initial conditions
		self.yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
		if hasattr(self.yaw_IC, "__len__"):
			if len(self.yaw_IC) == self.n_turbines:
				self.controls_dict = {"yaw_angles": self.yaw_IC}
			else:
				raise TypeError(
					"yaw initial condition should be a float or "
					+ "a list of floats of length num_turbines."
				)
		else:
			self.controls_dict = {"yaw_angles": [self.yaw_IC] * self.n_turbines}
	
	# self.filtered_measurements["wind_direction"] = []
	
	def _first_ord_filter(self, x):
		b = [1 - self.lpf_alpha]
		a = [1, -self.lpf_alpha]
		return lfilter(b, a, x)
	
	def yaw_offsets_interpolant(self, wind_directions, wind_speeds):
		# return np.zeros((*wind_directions.shape, self.n_turbines))
		return np.array([[[wind_directions[i, j] for t in range(self.n_turbines)] for j in range(wind_directions.shape[1])] for i in range(wind_directions.shape[0])])
	
	def compute_controls(self):
		# TODO MISHA should we check this at every simulation step rather than every 60, for threshold changes?
		current_time = np.atleast_1d(self.measurements_dict["time"])[0]
		# if current_time < 2 * self.simulation_dt:
		
		if np.all(self.measurements_dict["wind_directions"] == 0):
			pass # will be set to initial values
		# TODO MISHA this is a patch up for AMR wind initialization problem
		elif (abs(current_time % self.simulation_dt) == 0.0) or (current_time == self.simulation_dt * 2):
			current_wind_directions = np.atleast_2d(self.measurements_dict["wind_directions"])
			if self.use_filt:
				self.historic_measurements["wind_directions"] = np.vstack([
					self.historic_measurements["wind_directions"],
					current_wind_directions])[-int((self.lpf_time_const // self.simulation_dt) * 1e3):, :]
				
			# if not enough wind data has been collected to filter with, or we are not using filtered data, just get the most recent wind measurements
			if current_time < 60. or not self.use_filt:
				if np.size(current_wind_directions) == 0:
					if self.verbose:
						print("Bad wind direction measurement received, reverting to previous measurement.")
					wind_dirs = self.wd_store
				else:
					wind_dirs = current_wind_directions[0, :]
					self.wd_store = list(wind_dirs)
					
			else:
				# use filtered wind direction and speed
				wind_dirs = np.array([self._first_ord_filter(self.historic_measurements["wind_directions"][:, i])
												for i in range(self.n_turbines)]).T[-1, :]
				
			print(f"{'filtered' if self.use_filt else 'unfiltered'} wind directions = {wind_dirs}")
				
			# TODO MISHA can't rely on receiving yaw_angles from measurements?
			yaw_setpoints = []
			current_yaw_setpoints = np.atleast_2d(self.controls_dict["yaw_angles"])[0, :]
			
			for i in range(self.n_turbines):
				
				# current_yaw_angles = np.atleast_2d(self.measurements_dict["yaw_angles"])[0, :]
				if np.abs([current_yaw_setpoints[i] - wind_dirs[i]]) > self.deadband_thr:
					# move towards zero offset if the deadband has been superceded
					yaw_setpoints.append(wind_dirs[i] - 0.0)
				else:
					# otherwise stay put
					yaw_setpoints.append(current_yaw_setpoints[i])

			yaw_setpoints = np.array(yaw_setpoints)
			yaw_setpoints = np.clip(yaw_setpoints, current_yaw_setpoints - self.simulation_dt * self.yaw_rate, current_yaw_setpoints + self.simulation_dt * self.yaw_rate)
			yaw_setpoints = np.rint(yaw_setpoints / self.yaw_increment) * self.yaw_increment
			
			self.controls_dict = {"yaw_angles": list(yaw_setpoints)}

		return None


# if __name__ == '__main__':
	
# 	with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml")) as fp:
# 		wind_field_config = yaml.safe_load(fp)
	
# 	# options
# 	max_workers = 16
# 	# input_dict = load_yaml(sys.argv[1])
# 	input_dict = load_yaml(os.path.join(os.path.dirname(__file__), "../../examples/hercules_input_001.yaml"))
	
# 	# results wind field options
# 	wind_directions_tgt = np.arange(0.0, 360.0, 1.0)
# 	wind_speeds_tgt = np.arange(1.0, 25.0, 1.0)
	
# 	# Load a dataframe containing the wind rose information
# 	# import floris
# 	# df_windrose, windrose_interpolant \
# 	# 	= ControlledFlorisModel.load_windrose(
# 	# 	windrose_path=os.path.join(os.path.dirname(floris.__file__), "../examples/inputs/wind_rose.csv"))
	
# 	# Read in the wind rose using the class
	
# 	windrose_path = os.path.join(os.path.dirname(floris.__file__), "../examples/inputs/wind_rose.csv")
# 	df = pd.read_csv(windrose_path)
# 	interp = LinearNDInterpolator(points=df[["wd", "ws"]], values=df["freq_val"], fill_value=0.0)
# 	wd_grid, ws_grid = np.meshgrid(wind_directions_tgt, wind_speeds_tgt, indexing="ij")
# 	freq_table = interp(wd_grid, ws_grid)
# 	wind_rose = WindRose(wind_directions=wind_directions_tgt, wind_speeds=wind_speeds_tgt, freq_table=freq_table)
# 	# wind_rose.read_wind_rose_csv("inputs/wind_rose.csv")

	
# 	## First, get baseline AEP, without wake steering
	
# 	# Load a FLORIS object for AEP calculations 
# 	fi_noyaw = ControlledFlorisModel(max_workers=max_workers,
# 										 yaw_limits=input_dict["controller"]["yaw_limits"],
# 										 dt=input_dict["dt"],
# 										 yaw_rate=input_dict["controller"]["yaw_rate"]) \
# 		.load_floris(config_path=input_dict["controller"]["floris_input_file"])
# 	fi_noyaw.env.reinitialize(
# 		wind_directions=wind_rose.wd_flat,
# 		wind_speeds=wind_rose.ws_flat,
# 		turbulence_intensities=[0.08]  # Assume 8% turbulence intensity
# 	)
	
# 	# Pour this into a parallel computing interface
# 	# fi_noyaw.parallelize()
	
# 	# instantiate controller
# 	ctrl_noyaw = NoYawController(fi_noyaw, input_dict)
	
# 	farm_power_noyaw, farm_aep_noyaw, farm_energy_noyaw = ControlledFlorisModel.compute_aep(fi_noyaw, ctrl_noyaw, wind_rose)
	
# 	# Load a FLORIS object for AEP calculations
# 	fi_greedy = ControlledFlorisModel(max_workers=max_workers, yaw_limits=input_dict["controller"]["yaw_limits"],
# 										  dt=input_dict["dt"],
# 										  yaw_rate=input_dict["controller"]["yaw_rate"]) \
# 		.load_floris(config_path=input_dict["controller"]["floris_input_file"])
	
# 	# instantiate controller
# 	ctrl_greedy = GreedyController(fi_greedy, input_dict)
# 	farm_power_lut, farm_aep_lut, farm_energy_lut = ControlledFlorisModel.compute_aep(fi_greedy, ctrl_greedy, wind_rose)
# 	aep_uplift = 100.0 * (farm_aep_lut / farm_aep_noyaw - 1)
	
# 	print(" ")
# 	print("===========================================================")
# 	print("Calculating optimized annual energy production (AEP)...")
# 	print(f"Optimized AEP: {farm_aep_lut / 1.0e9:.3f} GWh.")
# 	print(f"Relative AEP uplift by wake steering: {aep_uplift:.3f} %.")
# 	print("===========================================================")
# 	print(" ")
	
# 	# Now calculate helpful variables and then plot wind rose information
	
# 	# freq_grid = windrose_interpolant(wd_grid, ws_grid)
# 	# freq_grid = freq_grid / np.sum(freq_grid)
	
# 	df = pd.DataFrame({
# 		"wd": wind_rose.wd_flat,
# 		"ws": wind_rose.ws_flat,
# 		"freq_val": wind_rose.freq_table_flat,
# 		"farm_power_baseline": farm_power_noyaw,
# 		"farm_power_opt": farm_power_lut,
# 		"farm_power_relative": farm_power_lut / farm_power_noyaw,
# 		"farm_energy_baseline": farm_energy_noyaw,
# 		"farm_energy_opt": farm_energy_lut,
# 		"energy_uplift": (farm_energy_lut - farm_energy_noyaw),
# 		"rel_energy_uplift": farm_energy_lut / np.sum(farm_energy_noyaw)
# 	})
	
# 	power_speed_fig = plot_power_vs_speed(df)
# 	yaw_figs = plot_yaw_vs_dir(ctrl_greedy.yaw_offsets_interpolant, ctrl_greedy.n_turbines)
# 	power_dir_fig = plot_power_vs_dir(df, fi_greedy.env.floris.flow_field.wind_directions)
	
# 	results_dir = os.path.join(whoc.__file__, "../examples/greedy_wake_steering_florisstandin/sweep_results")
# 	if not os.path.exists(results_dir):
# 		os.makedirs(results_dir)

# 	df.to_csv(os.path.join(results_dir, "results.csv"))

# 	power_speed_fig.savefig(os.path.join(results_dir, "power_speed"))
# 	power_dir_fig.savefig(os.path.join(results_dir, "spower_dir"))
# 	for i in range(len(yaw_figs)):
# 		yaw_figs[i].savefig(os.path.join(results_dir, f"yaw{i}"))
	
# 	exit()

# 	# instantiate wind field if files don't already exist
# 	wind_field_filenames = glob(f"{wind_field_config['data_save_dir']}/case_*.csv")
# 	if not len(wind_field_filenames):
# 		generate_multi_wind_ts(wind_field_config)
# 		wind_field_filenames = [f"case_{i}.csv" for i in range(wind_field_config["n_wind_field_cases"])]
	
# 	# if wind field data exists, get it
# 	wind_field_data = []
# 	if os.path.exists(wind_field_config['data_save_dir']):
# 		for fn in wind_field_filenames:
# 			wind_field_data.append(pd.read_csv(os.path.join(wind_field_config["data_save_dir"], fn)))
	
# 	# select wind field case
# 	case_idx = 0
# 	time_ts = wind_field_data[case_idx]["Time"].to_numpy()
# 	wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
# 	wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
# 	turbulence_intensity_ts = [0.08] * int(wind_field_config["simulation_max_time"] // input_dict["dt"])
	
# 	# Simulate wind farm with interface and controller
# 	fi_greedy.reset(disturbances={"wind_speeds": [wind_mag_ts[0]],
# 								  "wind_directions": [wind_dir_ts[0]],
# 								  "turbulence_intensity": turbulence_intensity_ts[0]})
# 	yaw_angles_ts = []
# 	for k, t in enumerate(np.arange(0, wind_field_config["simulation_max_time"] - input_dict["dt"], input_dict["dt"])):
# 		print(f'Time = {t}')
		
# 		# feed interface with new disturbances
# 		fi_greedy.step(disturbances={"wind_speeds": [wind_mag_ts[k]],
# 									 "wind_directions": [wind_dir_ts[k]],
# 									 "turbulence_intensity": turbulence_intensity_ts[k]})
		
# 		if False and k == 5:
# 			# TODO why does fi_greedy.env.floris.flow_field.u/v not change from 270 with changing freestream wind direction?
# 			# Using the FlorisInterface functions, get 2D slices.
# 			horizontal_plane = fi_greedy.env.calculate_horizontal_plane(
# 				height=90.0,
# 				x_resolution=20,
# 				y_resolution=10,
# 				yaw_angles=np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0.]]]),
# 			)
# 			visualize_quiver2(horizontal_plane)
# 			plt.show()
		
# 		# receive measurements from interface, compute control actions, and send to interface
# 		ctrl_greedy.step()
		
# 		print(f"Time = {ctrl_greedy.measurements_dict['time']}",
# 			  f"Freestream Wind Direction = {wind_dir_ts[k]}",
# 			  f"Freestream Wind Magnitude = {wind_mag_ts[k]}",
# 			  f"Turbine Wind Directions = {ctrl_greedy.measurements_dict['wind_directions']}",
# 			  f"Turbine Wind Magnitudes = {ctrl_greedy.measurements_dict['wind_speeds']}",
# 			  f"Turbine Powers = {ctrl_greedy.measurements_dict['powers']}",
# 			  f"Yaw Angles = {ctrl_greedy.measurements_dict['yaw_angles']}",
# 			  sep='\n')
# 		yaw_angles_ts.append(ctrl_greedy.measurements_dict['yaw_angles'])
# 	# print(ctrl_greedy.controls_dict)
# 	# print(ctrl_greedy.measurements_dict)
	
# 	yaw_angles_ts = np.vstack(yaw_angles_ts)
	
# 	# test first order filter and plot evolution of wind direction and yaw angles
# 	filt_wind_dir_ts = ctrl_greedy._first_ord_filter(wind_dir_ts)
# 	fig, ax = plt.subplots(2, 1)
# 	ax[0].plot(time_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], wind_dir_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], label='raw')
# 	ax[0].plot(time_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], filt_wind_dir_ts[50:int(wind_field_config["simulation_max_time"] // input_dict["dt"])], '--',
# 			   label='filtered')
# 	ax[0].set(title='Wind Direction [deg]', xlabel='Time [s]')
# 	# ax.set_xlim((time_ts[1], time_ts[-1]))
# 	ax[0].legend()
# 	ax[1].plot(time_ts[:int(wind_field_config["simulation_max_time"] // input_dict["dt"]) - 1], yaw_angles_ts)
# 	fig.show()

# # TODO plot video of horizontal plane
# # fi_greedy.env.calculate_horizontal_plane()
