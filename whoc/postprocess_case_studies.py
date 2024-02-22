import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# TODO better to pass df with all data to plotting functions

def plot_wind_field_ts(time_ts, wind_dir_ts, wind_mag_ts, save_dir):
    fig_wind, ax_wind = plt.subplots(2, 1)
    ax_wind[0].plot(time_ts, wind_dir_ts, label='raw')
    ax_wind[0].set(title='Wind Direction [deg]', xlabel='Time')
    ax_wind[1].plot(time_ts, wind_mag_ts, label='raw')
    ax_wind[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
    ax_wind[0].legend()
    fig_wind.savefig(os.path.join(save_dir, f'wind_ts.png'))
    fig_wind.show()

    return fig_wind, ax_wind

def plot_opt_var_ts(time_ts, opt_vars_ts, wind_dir_ts, yaw_offset_bounds, save_dir):
    
    fig_opt_vars, ax_opt_vars = plt.subplots(2, 1)
    ax_opt_vars[0].plot(time_ts, opt_vars_ts[:, 0])
    ax_opt_vars[0].set(title='Yaw Angles [deg]', xlabel='Time [s]')
    ax_opt_vars[0].plot(time_ts, wind_dir_ts - yaw_offset_bounds[0], 'k--', label="Upper Bound")
    ax_opt_vars[0].plot(time_ts, wind_dir_ts - yaw_offset_bounds[1], 'k--', label="Lower Bound")
    ax_opt_vars[1].plot(time_ts, opt_vars_ts[:, 1])
    ax_opt_vars[1].set(title='Yaw Angles Change [-]', xlabel='Time [s]')
    # ax_outputs[1, 0].plot(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], turbine_powers_ts)
    # ax_outputs[1, 0].set(title="Turbine Powers [MW]")
    
    fig_opt_vars.savefig(os.path.join(save_dir, f'opt_vars_ts.png'))
    fig_opt_vars.show()

    return fig_opt_vars, ax_opt_vars

def plot_opt_cost_ts(time_ts, opt_cost_terms_ts, save_dir):
    fig_opt_cost, ax_opt_cost = plt.subplots(2, 1)
    
    ax_opt_cost[0].scatter(time_ts, opt_cost_terms_ts[:, 0])
    ax_opt_cost[0].set(title="Optimization Yaw Angle Cost [-]")
    ax_opt_cost[1].scatter(time_ts, opt_cost_terms_ts[:, 1])
    ax_opt_cost[1].set(title="Optimization Yaw Angle Change Cost [-]")
    # ax_outputs[2].scatter(time_ts[:int(EPISODE_MAX_TIME // input_dict["dt"]) - 1], convergence_time_ts)
    # ax_outputs[2].set(title="Convergence Time [s]")
    fig_opt_cost.savefig(os.path.join(save_dir, f'opt_costs_ts.png'))
    fig_opt_cost.show()

    return fig_opt_cost, ax_opt_cost

def plot_power_ts(time_ts, wind_directions_ts, yaw_offsets_ts, turbine_powers_ts, save_dir):
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 5)
    n_turbines = turbine_powers_ts.shape[1]

    # Direction
    for t in range(n_turbines):
        line, = ax[0].plot(time_ts, wind_directions_ts[:, t], label="T{0:03d} wind dir.".format(t))
        ax[0].plot(time_ts, yaw_offsets_ts[:, t], color=line.get_color(), label="T{0:03d} yaw pos.".format(t),
            linestyle=":")
        if t == 0:
            ax[1].fill_between(time_ts, turbine_powers_ts[:, t] / 1e3, color=line.get_color(),
                label="T{0:03d} power".format(t))
        else:
            ax[1].fill_between(time_ts, turbine_powers_ts[:,:t + 1].sum(axis=1) / 1e3, 
                               turbine_powers_ts[:, :t].sum(axis=1)/1e3,
                color=line.get_color(), label="T{0:03d} power".format(t))
    ax[1].plot(time_ts, turbine_powers_ts.sum(axis=1)/1e3, color="black", label="Farm power")

    fig.savefig(os.path.join(save_dir, f'yaw_power_ts.png'))
    fig.show
    return fig, ax

def barplot_opt_cost(case_lists, case_name_lists, opt_cost_terms_list, save_dir):
    width = 0.25
    multiplier = 0
    case_groups = [case_name_list[0].split(" ")[0] for case_name_list in case_name_lists]
    
    x_vals = []

    fig, ax = plt.subplots(layout="constrained")

    for case_group_idx in range(len(case_name_lists)):

        x = np.arange(len(case_name_lists[case_group_idx])) + width * case_group_idx \
        + (sum(len(case_name_lists[i]) for i in range(case_group_idx - 1)) if case_group_idx > 0 else 0)
        x_vals.append(x)

        y = opt_cost_terms_list[case_group_idx]

        rects = ax.bar(x_vals[-1], y, width, label=case_name_lists[case_group_idx])

    ax.set_ylabel("Mean Optimized Cost Terms")
    ax.set_xticks(np.concatenate(x_vals))
    ax.set_xticklabels(np.concatenate(case_name_lists))

    fig.savefig(os.path.join(save_dir, f'opt_cost_comparison.png'))
    fig.show()


if __name__ == '__main__':
    pass