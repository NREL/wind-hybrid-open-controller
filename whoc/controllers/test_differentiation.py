import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from scipy.optimize import linprog, minimize

# TODO compare to central difference

if __name__ == "__main__":
    PLOT_DRVT = False
    PLOT_U = False
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    xy = np.array(list(product(x, y)))

    # x_grid, y_grid = np.meshgrid(x, y)
    # x_init, y_init = [np.pi, np.pi / 2]

    z1_func = lambda xy: np.sin(np.atleast_2d(xy)[:, 0]) + np.cos(2*np.atleast_2d(xy)[:, 1])
    z2_func = lambda xy: -np.cos(np.atleast_2d(xy)[:, 0]) + np.sin(-4*np.atleast_2d(xy)[:, 1])

    z1_func = lambda xy: np.atleast_2d(xy)[:, 0]**2
    z2_func = lambda xy: (np.atleast_2d(xy)[:, 1] - 0.5)**2
    z_func = lambda xy: np.array([z1_func(xy), z2_func(xy)]) #+ np.random.normal(0, 0.001)
    z = z_func(xy)
    # z_grid = z_func(x_grid, y_grid)

    # dz1dxy_func = lambda xy: np.array([np.cos(np.atleast_2d(xy)[:, 0]), -2*np.sin(2*np.atleast_2d(xy)[:, 1])])
    # dz2dxy_func = lambda xy: np.array([np.sin(np.atleast_2d(xy)[:, 0]), -4*np.cos(-4*np.atleast_2d(xy)[:, 1])])

    dz1dxy_func = lambda xy: np.array([2 * (np.atleast_2d(xy)[:, 0]), np.zeros_like(np.atleast_2d(xy)[:, 0])])
    dz2dxy_func = lambda xy: np.array([np.zeros_like(np.atleast_2d(xy)[:, 0]), 2 * (np.atleast_2d(xy)[:, 1] - 0.5)])
    dzdxy_func = lambda xy: np.array([dz1dxy_func(xy), dz2dxy_func(xy)]).T
    
    if 1:
        nu = 0.01
        alpha = 0.01
        n_samples = 1000
        max_iter = 1000

        xy_init = np.array([-0.5, 0.4])
        xy = np.array(xy_init)
        
        drvt_err = []
        xy_vals = []

        i = 0
        while i < max_iter:
            z = z_func(xy).T

            u = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
            plus_x = xy[0] + nu * u[:, 0]
            plus_y = xy[1] + nu * u[:, 1]
            plus_z = z_func(np.stack([plus_x, plus_y], axis=1)).T

            diff_z = (plus_z - z)
            # approx_dzdxy = (diff_z[:, np.newaxis] / nu) * u
            approx_dzdxy = np.einsum("ia, ib->iab", diff_z / nu, u)
            approx_dzdxy = np.mean(approx_dzdxy, axis=0)

            true_dz1dxy = dz1dxy_func(xy).T
            true_dz2dxy = dz2dxy_func(xy).T
            drvt_err.append(np.linalg.norm(approx_dzdxy - np.vstack([true_dz1dxy, true_dz2dxy])))
            xy_vals.append(xy)
            # xy_z1_vals.append(xy_z1)
            # xy_z2_vals.append(xy_z2)

            if 0:
                c = [approx_dzdxy[0], + approx_dzdxy[1]]
                res = linprog(c=c, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
                xy = (1 - alpha) * xy + alpha * res.x
            else:
                # use computed gradient in standard gradient descent update
                xy = xy - alpha * np.sum(approx_dzdxy, axis=0) # TODO check that this is what SLSQP is doing

            i += 1

        xy_vals = np.vstack(xy_vals)
        # xy_z1_vals = np.vstack(xy_z1_vals)
        # xy_z2_vals = np.vstack(xy_z2_vals)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.arange(max_iter), drvt_err, label="drvt_err")
        ax[1].plot(np.arange(max_iter), xy_vals[:, 0], label="x_z1")
        ax[1].plot(np.arange(max_iter), xy_vals[:, 1], label="y_z1")
        # ax[1].plot(np.arange(max_iter), xy_z2_vals[:, 0], label="x_z2")
        # ax[1].plot(np.arange(max_iter), xy_z2_vals[:, 1], label="y_z2")
        ax[0].legend()
        ax[1].legend()

        # xy_range = np.linspace(-np.pi, np.pi, 1000)
        res = minimize(lambda xy: z1_func(xy) + z2_func(xy), xy_init, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
        print("Minimizer computed using minimize:", (res.x, res.fun))
        print("Minimizer computed using ZCGD", (xy_vals[-1, :], np.sum(z_func(xy_vals[-1, :]))))
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