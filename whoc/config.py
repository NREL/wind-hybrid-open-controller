import os
import sys
import matplotlib as mpl
import numpy as np

FARM_LAYOUT = '9turb'

if sys.platform == 'darwin':
	PROJECT_DIR = '/Users/ahenry/Documents/Research/nn_wake_modeling/'
	FLORIS_DIR = '/Users/ahenry/Documents/toolboxes/floris'
	EPISODE_MAX_TIME = 60 * 60 * 1  # 60 minutes
	N_CASES = 1
	STORAGE_DIR = '/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/wind_field_data'
elif sys.platform == 'linux':
	STORAGE_DIR = f'/scratch/alpine/aohe7145/nn_wake_modeling/'
	PROJECT_DIR = '/projects/aohe7145/projects/nn_wake_modeling'
	FLORIS_DIR = '/projects/aohe7145/toolboxes/floris'
	EPISODE_MAX_TIME = 60 * 60 * 24  # 1 day
	N_CASES = 500

DATA_SAVE_DIR = os.path.join(STORAGE_DIR, 'raw_data')
# TS_SAVE_DIR = os.path.join(STORAGE_DIR, f'{FARM_LAYOUT}_wake_field_tsdata')
# SIM_SAVE_DIR = os.path.join(STORAGE_DIR, f'{FARM_LAYOUT}_wake_field_simulations')
FIG_DIR = os.path.join(STORAGE_DIR, 'figs')

SIM_MODEL_FLORIS_DIR = os.path.join(FLORIS_DIR, f'examples/inputs/gch.yaml')
# BASE_MODEL_FLORIS_DIR = os.path.join(FLORIS_DIR, f'examples/inputs/{FARM_LAYOUT}_base_model_floris_input.json')


for dir in [DATA_SAVE_DIR, FIG_DIR]:
	if not os.path.exists(dir):
		os.makedirs(dir)

YAW_CHANGES = [-1, 0, 1]
YAW_ANGLE_RANGE = [-30.0, 30.0]
AI_FACTOR_RANGE = [0, 1 / 3]

DT = 1.0  # discrete-time step for wind farm control
SAMPLING_TIME_STEP = {'freestream_wind_speed': int(60 // DT),
				 'freestream_wind_dir': int(60 // DT),
				 'yaw_angle': int(60 // DT),
                 'ai_factor': int(1 // DT)}  # interval of DT seconds at which each agent takes a step

EPISODE_MAX_TIME_STEPS = int(EPISODE_MAX_TIME // DT) # number of discrete time-stpes
N_PREVIEW_STEPS = 600

WIND_SPEED_U_RANGE = (8, 12)
WIND_SPEED_V_RANGE = (4, 6)
# WIND_SPEED_RANGE = (1, 24)
WIND_DIR_RANGE = (250, 290)

EPS = 0.0  # substitute for zero axial-ind factor
YAW_RATE = 0.5  # degrees per second
DELTA_YAW = DT * YAW_RATE
PTFM_RNG = 200  # 200 of platform relocation range

YAW_ACTUATION = True
AI_FACTOR_ACTUATION = True
PTFM_ACTUATION = False

MAX_YAW_TRAVEL_THR = 100  # 100 degrees
MAX_YAW_TRAVEL_TIME = 600  # 10 minutes

TIME_VARYING = {'power_ref': False, 'wind_speed_mean': False, 'wind_dir_mean': False, 'online': False,
                'wind_speed_turbulence': False, 'wind_dir_turbulence': False}

WORKING_DIR = os.getcwd()

# TODO this should be a small value for power in the case of yaw, where we just want to coarsely follow the power reference, and large value for ax ind factor, where we want to follow it closesly
TURBINE_ALPHA = {'power': 10, 'rotor_thrust': 1, 'yaw_travel': 1}
TURBINE_WEIGHTING = {'power': 2, 'rotor_thrust': 1, 'yaw_travel': 1}

ACTION_RANGE = {'yaw_angle': YAW_ANGLE_RANGE, 'ai_factor': AI_FACTOR_RANGE}
INIT_VALS = {'yaw_angle': 0, 'ai_factor': 1 / 3, 'freestream_wind_speed': 10, 'freestream_wind_direction': 270}

WIND_SPEED_CHANGE_PROBABILITY = 0.25
WIND_SPEED_VAR = 0.25
WIND_SPEED_TURB_STD = 1
WIND_DIR_CHANGE_PROBABILITY = 0.25
WIND_DIR_VAR = 2.5
WIND_DIR_TURB_STD = 2.5
YAW_ANGLE_CHANGE_PROBABILITY = 0.0
YAW_ANGLE_CHANGE_VAR = 1
YAW_ANGLE_VAR = DELTA_YAW * SAMPLING_TIME_STEP['yaw_angle']
YAW_ANGLE_TURB_STD = 0
AI_FACTOR_CHANGE_PROBABILITY = 0.0
AI_FACTOR_VAR = 0.01
AI_FACTOR_TURB_STD = 0
ONLINE_CHANGE_PROBABILITY = 0.0
WIND_CHANGE_FREQ = int(3600 // DT)  # let the mean wind speed/direction change once an hour

FIGSIZE = (42, 21)
COLOR_1 = 'darkgreen'
COLOR_2 = 'indigo'
BIG_FONT_SIZE = 70
SMALL_FONT_SIZE = 66
mpl.rcParams.update({'font.size': SMALL_FONT_SIZE,
					 'axes.titlesize': BIG_FONT_SIZE,
					 'figure.figsize': FIGSIZE,
					 'legend.fontsize': SMALL_FONT_SIZE,
					 'xtick.labelsize': SMALL_FONT_SIZE,
					 'ytick.labelsize': SMALL_FONT_SIZE,
                     'lines.linewidth': 4,
					 'figure.autolayout': True,
                     'lines.markersize': 10,
                     'yaxis.labellocation': 'top'
                     })

mpl.rcParams.update({
					 'figure.figsize': FIGSIZE,
                     'lines.linewidth': 4,
					 'figure.autolayout': True,
                     'lines.markersize':10})

WIND_FIELD_CONFIG = {
        "floris_input_file": SIM_MODEL_FLORIS_DIR,
        "turbine_layout_std": 1,
        "offline_probability": ONLINE_CHANGE_PROBABILITY,
        # probability of any given turbine going offline at each time-step
        "wind_speed_change_probability": WIND_SPEED_CHANGE_PROBABILITY,
        # probability of wind speed/direction changing (1/2 for increase, 1/2 for decrease)
        "wind_dir_change_probability": WIND_DIR_CHANGE_PROBABILITY,
        # probability of wind speed/direction changing (1/2 for increase, 1/2 for decrease)
		"yaw_angle_change_probability": YAW_ANGLE_CHANGE_PROBABILITY,
		"ai_factor_change_probability": AI_FACTOR_CHANGE_PROBABILITY,
        "wind_speed_var": WIND_SPEED_VAR,  # step change in m/s of wind speed
        "wind_dir_var": WIND_DIR_VAR,  # step change in degrees of wind direction,
		"yaw_angle_change_var": YAW_ANGLE_CHANGE_VAR,  # step change in m/s of wind speed
		"yaw_angle_var": YAW_ANGLE_VAR,  # step change in m/s of wind speed
        "ai_factor_var": AI_FACTOR_VAR,  # step change in degrees of wind direction,
        # 5, # standard deviation of normal turbulence  of wind direction, set to 0 for no turbulence
		"yaw_angle_turb_std": YAW_ANGLE_TURB_STD,
		"ai_factor_turb_std": AI_FACTOR_TURB_STD,
        "episode_max_time_steps": EPISODE_MAX_TIME_STEPS + N_PREVIEW_STEPS,
        # ensure there is enough power reference preview steps left before the full 24 hour mark
        "simulation_sampling_time": DT,
		"wind_speed_sampling_time_step": SAMPLING_TIME_STEP['freestream_wind_speed'],
		"wind_dir_sampling_time_step": SAMPLING_TIME_STEP['freestream_wind_dir'],
		"yaw_angle_sampling_time_step": SAMPLING_TIME_STEP['yaw_angle'],
		"ai_factor_sampling_time_step": SAMPLING_TIME_STEP['ai_factor'],
		"yaw_angle_roc": YAW_RATE,
		# 0.5, # standard deviation of normal turbulence of wind speed, set to 0 for no turbulence
		"wind_speed_noise_func": np.random.normal, # the noise added within the time-series generate function, not suitable for preview joint distribution noise, np.random.poisson,
		"wind_speed_u_noise_args": {"scale": WIND_SPEED_TURB_STD},
		"wind_speed_v_noise_args": {"scale": WIND_SPEED_TURB_STD},
		"wind_speed_u_range": WIND_SPEED_U_RANGE,
		"wind_speed_v_range": WIND_SPEED_V_RANGE,
		"wind_speed_preview_time": 600,
		"yaw_limits": YAW_ANGLE_RANGE
    }