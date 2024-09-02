import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from scipy.optimize import linprog, minimize
from pyoptsparse import Optimization, SLSQP

# TODO compare to central difference

def custom_diff(fi, yaw_setpoints, wind_preview_samples, n_wind_preview_samples, n_horizon, n_turbines, offline_status, yaw_limits, yaw_norm_const):
    yaw_offsets = np.zeros((n_wind_preview_samples * n_horizon, n_turbines))
    fi.env.set_operation(
        yaw_angles=yaw_offsets,
        disable_turbines=offline_status,
    )
    fi.env.run()
    all_greedy_yaw_turbine_powers = fi.env.get_turbine_powers()
    # greedy_yaw_turbine_powers = np.reshape(greedy_yaw_turbine_powers, (n_wind_preview_samples, n_horizon, n_turbines))
    greedy_yaw_turbine_powers = np.max(all_greedy_yaw_turbine_powers, axis=1)[:, np.newaxis] # choose unwaked turbine for normalization constant
    
    # if effective yaw is greater than90, set negative powers, sim to interior point method, gradual penalty above 30deg offsets TEST
    
    current_yaw_offsets = np.vstack([(wind_preview_samples[f"FreestreamWindDir_{j + 1}"][m] - yaw_setpoints[j, :]) for m in range(n_wind_preview_samples) for j in range(n_horizon)])
    # current_yaw_offsets = np.vstack([(wind_preview_samples[f"FreestreamWindDir_{j + 1}"][0] - yaw_setpoints[j, :]) for j in range(n_horizon)])
    
    yaw_offsets = np.clip(current_yaw_offsets, *yaw_limits)
    fi.env.set_operation(
        yaw_angles=yaw_offsets,
        disable_turbines=offline_status,
    )
    
    # fi.env.set(yaw_angles=np.clip(current_yaw_offsets, *yaw_limits), disable_turbines=offline_status)
    fi.env.run()
    yawed_turbine_powers = fi.env.get_turbine_powers()

    decay_factor = -np.log(1e-6) / ((90 - np.max(np.abs(yaw_limits))) / yaw_norm_const)
    neg_decay = np.exp(-decay_factor * (yaw_limits[0] - current_yaw_offsets[current_yaw_offsets < yaw_limits[0]]) / yaw_norm_const)
    pos_decay = np.exp(-decay_factor * (current_yaw_offsets[current_yaw_offsets > yaw_limits[1]] - yaw_limits[1]) / yaw_norm_const)
    yawed_turbine_powers[(current_yaw_offsets < yaw_limits[0])] = yawed_turbine_powers[current_yaw_offsets < yaw_limits[0]] * neg_decay
    yawed_turbine_powers[(current_yaw_offsets > yaw_limits[1])] = yawed_turbine_powers[current_yaw_offsets > yaw_limits[1]] * pos_decay
	
    norm_turbine_powers = np.divide(yawed_turbine_powers, greedy_yaw_turbine_powers,
                                        where=greedy_yaw_turbine_powers!=0,
                                        out=np.zeros_like(yawed_turbine_powers))
    norm_turbine_powers = np.reshape(norm_turbine_powers, (n_wind_preview_samples, n_horizon, n_turbines))

if __name__ == "__main__":

    # PLOT_DRVT = False
    # PLOT_U = False
    # x = np.linspace(-np.pi, np.pi, 100)
    # y = np.linspace(-np.pi, np.pi, 100)
    # xy = np.array(list(product(x, y)))

    z1_func = lambda xy: np.sin(np.atleast_2d(xy)[:, 0]) + np.cos(2*np.atleast_2d(xy)[:, 1])
    z2_func = lambda xy: -np.cos(np.atleast_2d(xy)[:, 0]) + np.sin(-4*np.atleast_2d(xy)[:, 1])

    # # z1_func = lambda xy: np.atleast_2d(xy)[:, 0]**2
    # # z2_func = lambda xy: (np.atleast_2d(xy)[:, 1] - 0.5)**2

    z_func = lambda xy: np.array([z1_func(xy), z2_func(xy)]) #+ np.random.normal(0, 0.001)
    # z = z_func(xy)

    dz1dxy_func = lambda xy: np.array([np.cos(np.atleast_2d(xy)[:, 0]), -2*np.sin(2*np.atleast_2d(xy)[:, 1])])
    dz2dxy_func = lambda xy: np.array([np.sin(np.atleast_2d(xy)[:, 0]), -4*np.cos(-4*np.atleast_2d(xy)[:, 1])])

    # # dz1dxy_func = lambda xy: np.array([2 * (np.atleast_2d(xy)[:, 0]), np.zeros_like(np.atleast_2d(xy)[:, 0])])
    # # dz2dxy_func = lambda xy: np.array([np.zeros_like(np.atleast_2d(xy)[:, 0]), 2 * (np.atleast_2d(xy)[:, 1] - 0.5)])

    # dzdxy_func = lambda xy: np.array([dz1dxy_func(xy), dz2dxy_func(xy)]).T
    
    if 1:
        nu = 0.01
        alpha = 0.01
        n_samples = 1000
        max_iter = 1000

        xy_init = np.array([-0.5, 0.4])
        xy = np.array(xy_init)
        xy_true = np.array(xy_init)
        drvt_err = []
        xy_vals = []
        true_xy_vals = []
        def approx_dzdxy_func(xy):
            z = z_func(xy).T

            u = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
            plus_x = xy[0] + nu * u[:, 0]
            plus_y = xy[1] + nu * u[:, 1]
            plus_z = z_func(np.stack([plus_x, plus_y], axis=1)).T

            diff_z = (plus_z - z)
            # approx_dzdxy = (diff_z[:, np.newaxis] / nu) * u
            approx_dzdxy = np.einsum("ia, ib->iab", diff_z / nu, u)
            approx_dzdxy = np.mean(approx_dzdxy, axis=0)
            return approx_dzdxy #.sum(axis=0)

        i = 0
        while i < max_iter:
            approx_dzdxy = approx_dzdxy_func(xy)

            true_dz1dxy = dz1dxy_func(xy).T
            true_dz2dxy = dz2dxy_func(xy).T
            true_dzdxy = np.vstack([true_dz1dxy, true_dz2dxy])
            drvt_err.append(np.linalg.norm(approx_dzdxy - true_dzdxy))
            xy_vals.append(xy)
            true_xy_vals.append(xy_true)
            # xy_z1_vals.append(xy_z1)
            # xy_z2_vals.append(xy_z2)

            if 0:
                c = [approx_dzdxy[0], + approx_dzdxy[1]]
                res = linprog(c=c, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
                xy = (1 - alpha) * xy + alpha * res.x
            else:
                # use computed gradient in standard gradient descent update
                xy = xy - alpha * np.sum(approx_dzdxy, axis=0) # TODO check that this is what SLSQP is doing
                xy_true = xy_true - alpha * np.sum(true_dzdxy, axis=0) # summed because the different functions are summed to make objective function
            i += 1

        xy_vals = np.vstack(xy_vals)
        true_xy_vals = np.vstack(true_xy_vals)
        # # xy_z1_vals = np.vstack(xy_z1_vals)
        # # xy_z2_vals = np.vstack(xy_z2_vals)

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(np.arange(max_iter), drvt_err, label="drvt_err")
        ax[1].plot(np.arange(max_iter), xy_vals[:, 0], label="x_z1")
        ax[1].plot(np.arange(max_iter), xy_vals[:, 1], label="y_z1")
        ax[2].plot(np.arange(max_iter), true_xy_vals[:, 0], label="true_x_z1")
        ax[2].plot(np.arange(max_iter), true_xy_vals[:, 1], label="true_y_z1")
        # ax[1].plot(np.arange(max_iter), xy_z2_vals[:, 0], label="x_z2")
        # ax[1].plot(np.arange(max_iter), xy_z2_vals[:, 1], label="y_z2")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        # # xy_range = np.linspace(-np.pi, np.pi, 1000)
        res_min = minimize(lambda xy: z1_func(xy) + z2_func(xy), xy_init, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
        # res_min_zsgd = minimize(lambda xy: z1_func(xy) + z2_func(xy), xy_init, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], method='SLSQP', jac=approx_dzdxy_func)
        # print("Minimizer computed using minimize:", (res_min.x, res_min.fun))
        # print("Minimizer computed using SLSQP + ZSGD derivative:", (res_min_zsgd.x, res_min_zsgd.fun))
        # print("Minimizer computed using ZSGD", (xy_vals[-1, :], np.sum(z_func(xy_vals[-1, :]))))
        # print("Minimizer computed using ZCGD", [xy_z1_vals[-1, :], xy_z2_vals[-1, :]])
        
        # if PLOT_U:
        #     u_ax.scatter(u)
                
        # plus_x = xy[:, 0] + nu * u[:, 0:1]
        # plus_y = xy[:, 1] + nu * u[:, 1:2]
        # plus_z = z_func(plus_x, plus_y)

        # diff_z = (plus_z - z)
        # approx_dzdxy = np.einsum("ia, ib->iab", diff_z / nu, u)
        # approx_dzdxy = np.mean(approx_dzdxy, axis=0)

        # np.linalg.norm(approx_dzdxy - true_dzdxy)
        
    #     if PLOT_DRVT:
    #         drvt_ax.plot(x, approx_dydx, linestyle=":", label=f"n_samples = {n_samples}")
    # else:
    #     i = -1
    #     for n_samples in [20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]:
    #         for nu in [0.001, 0.01, 0.1]:
    #         # for nu in [1]:
    #             i += 1
                
    #             u = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    #             if PLOT_U:
    #                 u_ax.scatter([i] * len(u), u)
                
    #             plus_x = xy[:, 0] + nu * u[:, 0:1]
    #             plus_y = xy[:, 1] + nu * u[:, 1:2]
    #             plus_z = z_func(plus_x, plus_y)

    #             diff_z = (plus_z - z)
    #             approx_dydx = np.einsum("ia, ib->iab", diff_z / nu, u)
    #             approx_dydx = np.mean(approx_dydx, axis=0)

    #             np.linalg.norm(approx_dydx - true_dydx)
                
    #             if PLOT_DRVT:
    #                 drvt_ax.plot(x, approx_dydx, linestyle=":", label=f"n_samples = {n_samples}, nu = {nu}")
                
    #             results["n_samples"].append(n_samples)
    #             results["nu"].append(nu)
    #             results["err"].append(np.linalg.norm(approx_dydx - true_dydx))

    #     results = pd.DataFrame(results)
    #     results.sort_values(by="err", ascending=True, inplace=True)

    # if PLOT_DRVT:
    #     drvt_ax.plot(x, true_dydx, label="true")
    #     drvt_ax.legend()
    #     drvt_fig.show()