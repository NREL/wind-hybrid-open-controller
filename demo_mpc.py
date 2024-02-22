from whoc.controllers.mpc_wake_steering_controller import MPC, Objective, DynamicModel
from scipy.optimize import Bounds
import numpy as np

if __name__ == '__main__':
	# define ocp parameters
	Np = 10 * 60 # 10 minutes
	Nt = 9
	dt = 1 * 60 # 1 min
	# rho = 1.293
	# D = 256
	
	# define output function for yaw angles
	# TODO do we need absolute value of yaw rate of change if it will be squared in cost function
	def turbine_powers_norm_approx(ux, uy, gamma):
		# TODO include Cp ratio?
		wind_dir = np.atan(ux/uy) + np.pi
		return np.sin(wind_dir)**2 - gamma * np.sin(2 * wind_dir)
	
	# def turbine_yaw_time_norm(gamma_dot):
	# 	return np.abs(gamma_dot)
	
	C = np.diag(np.concatenate([np.ones((Nt)), np.zeros((Nt))]))
	D = np.zeros(2 * Nt)
	
	# define dynamic state equation
	A = np.diag([np.ones(Nt), np.zeros(Nt)])
	B = np.diag([np.ones(Nt) * dt, ])
	dyn_model = DynamicModel(n_outputs=Nt*2, # power output and yawing time for each turbine
	                             n_states=Nt, # yaw angle for each turbine
	                             n_ctrl_inputs=Nt * 2, # yaw angle rate of change direction (-1 or +1), and magnitude (0 or 1) for each turbine
	                             n_disturbances=Nt*2, # disturbance on power in the form of effective wind speed at each turbine
	                             n_horizon=Np, A=A, B=B, C=C, D=D)
	
	# define objective function
	alpha = 0.5
	Q = np.diag(np.concatenate([alpha * np.ones((Nt)),
	                            (1 - alpha) * np.ones((Nt))]))
	P = Q
	R = np.zeros((Nt,))
	obj_func = Objective(is_quadratic=True, is_stochastic=False, n_horizon=Np, dyn_model=dyn_model, Q=Q, P=P, R=R)
	
	x0 = np.random.randint(0, 1, (2 * Nt,))
	X = np.random.randint(0, 1, (Np * 2 * Nt,))
	U = np.random.randint(0, 1, (Np * Nt,))
	print(obj_func.compute(x0, X, U))
	
	# define constraints
	
	