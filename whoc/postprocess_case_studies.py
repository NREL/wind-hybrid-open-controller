import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import defaultdict

import seaborn as sns
sns.set_theme(style="darkgrid")

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)

# TODO better to pass df with all data to plotting functions

def compare_simulations(results_dfs):
    result_summary_dict = defaultdict(list)

    for case_name, results_df in results_dfs.items():

        # res = ResultsSummary(YawAngleChangeAbsSum=results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum(),
        #                      FarmPowerSum=results_df["FarmPower"].sum(),
        #                      TotalOptimizationCostSum=results_df["TotalOptimizationCost"].sum(),
        #                      ConvergenceTimeSum=results_df["ConvergenceTime"].sum())
        
        yaw_angles_change_ts = results_df[[c for c in results_df.columns if "TurbineYawAngleChange_" in c]]
        turbine_offline_status_ts = results_df[[c for c in results_df.columns if "TurbineOfflineStatus_" in c]]
        
        result_summary_dict["SolverType"].append(case_name)
        # result_summary_dict["YawAngleChangeAbsSum"].append(results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum())
        result_summary_dict["YawAngleChangeAbsMean"].append(yaw_angles_change_ts.abs().sum().to_numpy().mean())
        result_summary_dict["RelativeYawAngleChangeAbsMean"].append(((yaw_angles_change_ts.abs() * ~turbine_offline_status_ts).sum()) / ((~turbine_offline_status_ts).sum()).to_numpy().mean())
        # result_summary_dict["FarmPowerSum"].append(results_df["FarmPower"].sum())
        result_summary_dict["FarmPowerMean"].append(results_df["FarmPower"].mean())
        result_summary_dict["RelativeFarmPowerMean"].append(results_df["RelativeFarmPower"].mean())
        # result_summary_dict["TotalRunningOptimizationCostSum"].append(results_df["TotalRunningOptimizationCost"].sum())
        result_summary_dict["TotalRunningOptimizationCostMean"].append(results_df["TotalRunningOptimizationCost"].mean())
        result_summary_dict["OptimizationConvergenceTimeMean"].append(results_df["OptimizationConvergenceTime"].mean())
        # result_summary_dict["OptimizationConvergenceTimeSum"].append(results_df["OptimizationConvergenceTime"].sum())
    
    result_summary_df = pd.DataFrame(result_summary_dict)
    return result_summary_df

def plot_wind_field_ts(data_df, save_path):
    fig_wind, ax_wind = plt.subplots(2, 1, sharex=True)
    # fig_wind.set_size_inches(10, 5)

    ax_wind[0].plot(data_df["Time"], data_df["FreestreamWindDir"], label='raw')
    ax_wind[0].set(title='Wind Direction [deg]', xlabel='Time')
    ax_wind[1].plot(data_df["Time"], data_df["FreestreamWindMag"], label='raw')
    ax_wind[1].set(title='Wind Speed [m/s]', xlabel='Time [s]')
    ax_wind[0].legend()
    # fig_wind.tight_layout()
    fig_wind.savefig(save_path)
    # fig_wind.show()

    return fig_wind, ax_wind

def plot_opt_var_ts(data_df, yaw_offset_bounds, save_path):
    
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" in col])
    yaw_angle_change_cols = sorted([col for col in data_df.columns if "TurbineYawAngleChange_" in col])

    fig_opt_vars, ax_opt_vars = plt.subplots(2, 1, sharex=True)
    # fig_opt_vars.set_size_inches(10, 5)

    ax_opt_vars[0].plot(data_df["Time"], data_df[yaw_angle_cols])
    ax_opt_vars[0].set(title='Yaw Angles [deg]', xlabel='Time [s]')
    ax_opt_vars[0].plot(data_df["Time"], data_df["FreestreamWindDir"] - yaw_offset_bounds[0], 'k--', label="Upper Bound")
    ax_opt_vars[0].plot(data_df["Time"], data_df["FreestreamWindDir"] - yaw_offset_bounds[1], 'k--', label="Lower Bound")
    ax_opt_vars[1].plot(data_df["Time"], data_df[yaw_angle_change_cols])
    ax_opt_vars[1].set(title='Yaw Angles Change [-]', xlabel='Time [s]')
    # ax_outputs[1, 0].plot(time_ts[:int(simulation_max_time // input_dict["dt"]) - 1], turbine_powers_ts)
    # ax_outputs[1, 0].set(title="Turbine Powers [MW]")
    
    fig_opt_vars.savefig(save_path)
    # fig_opt_vars.show()

    return fig_opt_vars, ax_opt_vars

def plot_opt_cost_ts(data_df, save_path):
    fig_opt_cost, ax_opt_cost = plt.subplots(2, 1, sharex=True)
    # fig_opt_cost.set_size_inches(10, 5)
    
    ax_opt_cost[0].step(data_df["Time"], data_df["RunningOptimizationCostTerm_0"])
    ax_opt_cost[0].set(title="Optimization Yaw Angle Cost [-]")
    ax_opt_cost[1].step(data_df["Time"], data_df["RunningOptimizationCostTerm_1"])
    ax_opt_cost[1].set(title="Optimization Yaw Angle Change Cost [-]", xlabel='Time [s]')
    # ax_outputs[2].scatter(time_ts[:int(simulation_max_time // input_dict["dt"]) - 1], convergence_time_ts)
    # ax_outputs[2].set(title="Convergence Time [s]")
    fig_opt_cost.savefig(save_path)
    # fig_opt_cost.show()

    return fig_opt_cost, ax_opt_cost

def plot_power_ts(data_df, save_path):
    fig, ax = plt.subplots(2, 1, sharex=True)
    # fig.set_size_inches(10, 5)
    
    turbine_wind_direction_cols = sorted([col for col in data_df.columns if "TurbineWindDir_" in col])
    turbine_power_cols = sorted([col for col in data_df.columns if "TurbinePower_" in col])
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" in col])

    # Direction
    for t, (wind_dir_col, power_col, yaw_col) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols)):
        line, = ax[0].plot(data_df["Time"], data_df[wind_dir_col], label="T{0:03d} wind dir.".format(t))
        ax[0].plot(data_df["Time"], data_df[yaw_col], color=line.get_color(), label="T{0:03d} yaw pos.".format(t), linestyle=":")
        if t == 0:
            ax[1].fill_between(data_df["Time"], data_df[power_col] / 1e3, color=line.get_color(), label="T{0:03d} power".format(t))
        else:
            ax[1].fill_between(data_df["Time"], data_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e3, 
                               data_df[turbine_power_cols[:t]].sum(axis=1) / 1e3,
                color=line.get_color(), label="T{0:03d} power".format(t))
    ax[1].plot(data_df["Time"], data_df[turbine_power_cols].sum(axis=1) / 1e3, color="black", label="Farm power")
    
    ax[0].set(title="Wind Direction / Yaw Angle [deg]")
    ax[0].legend()
    ax[1].set(xlabel="Time [s]", title="Turbine Powers [kW]")
    ax[1].legend()

    fig.savefig(save_path)
    # fig.show()
    return fig, ax

def barplot_opt_cost(data_summary_df, save_dir, relative=False):
    width = 0.25
    multiplier = 0
    case_groups = sorted(set("_".join(case.split("_")[:-1]) for case in data_summary_df["SolverType"].to_list()))
    all_cases = sorted(data_summary_df["SolverType"].to_list())
    grouped_cases = [[case for case in all_cases if case_group in case] for case_group in case_groups]

    sequential_colormaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu'][:len(case_groups)]
    
    x_vals = []

    fig, ax = plt.subplots(layout="constrained")

    if relative:
        base_case = data_summary_df.loc[data_summary_df["SolverType"] == "baseline_controllers_1", "TotalRunningOptimizationCostSum"].iloc[0] # greedy control

    for case_group_idx in range(len(case_groups)):

        x = (np.arange(len(grouped_cases[case_group_idx]))  \
            + (sum(len(grouped_cases[i]) for i in range(case_group_idx)) + 1 if case_group_idx > 0 else 0)) * width
        x_vals.append(x)

        y = data_summary_df.loc[data_summary_df["SolverType"].str.contains(case_groups[case_group_idx]), "TotalRunningOptimizationCostSum"]

        if relative:
            y = 100.0 * y / base_case
            # y = 100.0 * (y - base_case) / base_case

        cmap = matplotlib.colormaps[sequential_colormaps[case_group_idx]]
        colors = [cmap(f) for f in np.linspace(0.5, 1, len(grouped_cases[case_group_idx]))]
        rects = ax.bar(x_vals[-1], y, width, label=case_groups[case_group_idx])
        for r, rect in enumerate(rects):
            rect.set_color(colors[r])

    if relative:
        ax.set(title="Total Running Optimization Cost Relative to Greedy Controller [%]")
    else:
        ax.set(title="Total Running Optimization Cost [-]")
    ax.set_xticks(np.concatenate(x_vals))
    ax.set_xticklabels(all_cases)

    fig.savefig(os.path.join(save_dir, f'opt_cost_comparison.png'))
    # fig.show()

def plot_cost_function_pareto_curve(data_summary_df, save_dir):
   
    """
    plot mean farm level power vs mean sum of absolute yaw changes for different values of alpha
    """
    fig, ax = plt.subplots(1)
    sub_df = data_summary_df.loc["alpha_" in data_summary_df["SolverType"].str, :]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    sns.scatterplot(data=sub_df, x="YawAngleChangeAbsMean", y="FarmPowerMean", size="SolverType", ax=ax)
    fig.savefig(os.path.join(save_dir, "cost_function_pareto_curve.png"))

def plot_breakdown_robustness(data_summary_df, save_dir):
    # TODO could also make countplot and plot all time-step data points for different values of probability
    """
    plot mean relative farm level power vs mean relative sum of absolute yaw changes for different values of breakdown probability
    """
    fig, ax = plt.subplots(1)
    sub_df = data_summary_df.loc["Breakdown" in data_summary_df["SolverType"].str, :]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    sns.scatterplot(data=sub_df, x="YawAngleChangeAbsMean", y="FarmPowerMean", size="SolverType", ax=ax)
    fig.savefig(os.path.join(save_dir, "breakdown_robustness.png"))

if __name__ == '__main__':
    pass