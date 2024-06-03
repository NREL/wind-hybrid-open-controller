import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import defaultdict
from whoc import __file__ as whoc_file
from hercules.utilities import load_yaml
from itertools import cycle

import seaborn as sns
sns.set_theme(style="darkgrid", rc={'figure.figsize':(4,4)})
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

def read_amr_outputs(results_paths, hercules_dict):
    dfs = []
    for controller_class, seed, results_path in results_paths:
        df = pd.read_csv(results_path)
        df["ControllerClass"] = controller_class #os.path.basename(results_dir).split("_")[0]
        df["WindSeed"] = seed
        df["Time"] = np.cumsum(df["dt"]) - df["dt"].iloc[0]
        yaw_angle_cols = [col for col in df.columns if f"turbine_yaw_angles" in col]
        df[[f"TurbineYawAngleChange_{int(col.split('.')[-1])}" for col in yaw_angle_cols]] \
            = np.vstack([df.iloc[0][yaw_angle_cols] - hercules_dict["controller"]["initial_conditions"]["yaw"], df[yaw_angle_cols].diff().iloc[1:]])
        df[[f"AbsoluteTurbineYawAngleChange_{int(col.split('.')[-1])}" for col in yaw_angle_cols]] = df[[f"TurbineYawAngleChange_{int(col.split('.')[-1])}" for col in yaw_angle_cols]].abs()
        dfs.append(df)

    df = pd.concat(dfs)

    # select important column names, rename columns
    # hercules_comms.amr_wind.wind_farm_0.turbine_yaw_angles.003
    cols = []
    new_cols = []
    for c in range(len(df.columns)):
        col = df.columns[c]
        if not ("turbine_powers" in col or "turbine_yaw_angles" in col):
            continue
        cols.append(col)
        if "." in col:
            parts = col.split(".")
            new_col = f"{parts[-2]}_{int(parts[-1])}"
        else:
            new_col = col
        new_cols.append(new_col)

    cols += ["Time", "ControllerClass", "WindSeed"] + [col for col in df.columns if "YawAngleChange" in col]
    new_cols += ["Time", "ControllerClass", "WindSeed"] + [col for col in df.columns if "YawAngleChange" in col]
    df = df[cols]
    df.rename(columns={col: new_col for col, new_col in zip(cols, new_cols)}, inplace=True)

    # remove rows corresponding to all zero turbine powers
    df = df.loc[~(df[[col for col in df.columns if f"turbine_powers" in col]] == 0).all(axis="columns"), :]
    df.rename(columns={col: f"TurbinePower_{col.split('_')[-1]}" for col in df.columns if "turbine_powers" in col}, inplace=True)
    df.rename(columns={col: f"TurbineYawAngle_{col.split('_')[-1]}" for col in df.columns if "turbine_yaw_angles" in col}, inplace=True)
    df.loc[:, "Time"] = df["Time"] - df.iloc[0]["Time"]

    df["ControllerClass"] = pd.Categorical(df["ControllerClass"], ["Greedy", "LUT", "MPC"])
    df.sort_values(by=["ControllerClass", "Time"], inplace=True)

    df["FarmAbsoluteYawAngleChange"] = df[[col for col in df.columns if "TurbineYawAngleChange_" in col]].abs().sum(axis=1)
    df["FarmPower"] = df[[col for col in df.columns if "TurbinePower_" in col]].sum(axis=1)

    return df

def plot_yaw_power_ts(data_df, turbine_indices, save_path, seed=0):
    """
    For each controller class (different lineplots), and for a select few turbine_indices (different subplots), plot their angle changes and powers vs time with a combo plot for each turbine.
    """
    n_rows = int(np.floor(np.sqrt(len(turbine_indices))))
    if np.sqrt(len(turbine_indices)) % 1.0 == 0:
        fig1, ax1 = plt.subplots(n_rows, n_rows, sharex=True, sharey=True)
    else:
        fig1, ax1 = plt.subplots(n_rows, n_rows + 1, sharex=True, sharey=True)
    ax1 = ax1.flatten()
    
    # data_df = data_df.melt()

    for i in range(len(turbine_indices)):
        ax1[i] = sns.lineplot(x="Time", y=f"TurbineYawAngleChange_{turbine_indices[i]}", hue="ControllerClass", data=data_df.loc[data_df["WindSeed"] == seed], 
                              color=sns.color_palette()[0],
                              ax=ax1[i], sort=False, legend=i==0)
        ax1[i].xaxis.label.set_text(f"Time [s]")
        ax1[i].title.set_text(f"Turbine {turbine_indices[i]}Absolute Yaw Angle Change [$^\circ$]")
        # ax1[i].yaxis.label.set_color(ax1[i].get_lines()[0].get_color())
        # ax1[i].tick_params(axis="y", color=ax1[i].get_lines()[0].get_color())
    ax1[0].legend(loc="upper right")
    # ax2 = []
    # for i in range(len(turbine_indices)):
    #     ax2.append(ax1[i].twinx())

    if np.sqrt(len(turbine_indices)) % 1.0 == 0:
        fig2, ax2 = plt.subplots(n_rows, n_rows, sharex=True, sharey=True)
    else:
        fig2, ax2 = plt.subplots(n_rows, n_rows + 1, sharex=True, sharey=True)
    ax2 = ax2.flatten()

    for i in range(len(turbine_indices)):
        ax2[i] = sns.lineplot(x="Time", y=f"TurbinePower_{turbine_indices[i]}", hue="ControllerClass", data=data_df.loc[data_df["WindSeed"] == seed], 
                              color=sns.color_palette()[1],
                              ax=ax2[i], sort=False, legend=i==0)
        ax2[i].xaxis.label.set_text(f"Time [s]")
        ax2[i].title.set_text(f"Turbine {turbine_indices[i]} Power [MW]")
        # ax2[i].yaxis.label.set_color(ax2[i].get_lines()[0].get_color())
        # ax2[i].tick_params(axis="y", color=ax2[i].get_lines()[0].get_color())

    ax2[0].legend(loc="upper right")

    fig1.set_size_inches((11.2, 4.8))
    fig1.show()
    fig1.savefig(save_path.replace(".png", "_abs_yaw_change.png"))

    fig2.set_size_inches((11.2, 4.8))
    fig2.show()
    fig2.savefig(save_path.replace(".png", "_power.png"))


def plot_yaw_power_distribution(data_df, save_path):
    """
    For each controller class (categorical, along x-axis), plot the distribution of total farm powers and total absolute yaw angle changes over all time-steps and seeds (different subplots), plot their angle changes and powers vs time with a combo plot for each turbine.
    """
    plt.figure(1)
    ax1 = sns.catplot(x="ControllerClass", y="FarmAbsoluteYawAngleChange", data=data_df, kind="boxen")
    ax1.ax.xaxis.label.set_text("Controller")
    ax1.ax.title.set_text("Farm Absolute Yaw Angle Change [$^\circ$]")
    ax1.ax.yaxis.label.set_text("")
    plt.show()
    plt.savefig(save_path.replace(".png", "_abs_yaw_change.png"))

    plt.figure(2)
    ax2 = sns.catplot(x="ControllerClass", y="FarmPower", data=data_df, kind="boxen")
    ax2.ax.xaxis.label.set_text("Controller")
    ax2.ax.title.set_text("Farm Power [MW]")
    ax2.ax.yaxis.label.set_text("")
    ax2.ax.set_yticklabels(ax2.ax.get_yticks() / 1e3)
    plt.show()
    plt.savefig(save_path.replace(".png", "_power.png"))

    # fig1.set_size_inches((11.2, 4.8))
    # fig2.set_size_inches((11.2, 4.8))

def compare_simulations(results_dfs, save_dir):
    result_summary_dict = defaultdict(list)

    for df_name, results_df in results_dfs.items():
        # res = ResultsSummary(YawAngleChangeAbsSum=results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum(),
        #                      FarmPowerSum=results_df["FarmPower"].sum(),
        #                      TotalOptimizationCostSum=results_df["TotalOptimizationCost"].sum(),
        #                      ConvergenceTimeSum=results_df["ConvergenceTime"].sum())
        
        for seed in pd.unique(results_df["WindSeed"]):
            seed_df = results_df.loc[results_df["WindSeed"] == seed, :]
            
            yaw_angles_change_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbineYawAngleChange_" in c]))]
            turbine_offline_status_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbineOfflineStatus_" in c]))]
            turbine_power_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbinePower_" in c]))]
            # TODO doesn't work for some case families
            result_summary_dict["CaseFamily"].append(df_name.replace(f"_{seed_df['CaseName'].iloc[0]}", ""))
            result_summary_dict["CaseName"].append(seed_df["CaseName"].iloc[0])
            result_summary_dict["WindSeed"].append(seed)
            # result_summary_dict["YawAngleChangeAbsSum"].append(results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum())
            result_summary_dict["YawAngleChangeAbsMean"].append(yaw_angles_change_ts.abs().sum(axis=1).mean())
            result_summary_dict["RelativeYawAngleChangeAbsMean"].append(((yaw_angles_change_ts.abs().to_numpy() * np.logical_not(turbine_offline_status_ts)).sum(axis=1) / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean())
            # result_summary_dict["FarmPowerSum"].append(results_df["FarmPower"].sum())
            result_summary_dict["FarmPowerMean"].append(turbine_power_ts.sum(axis=1).mean())
            result_summary_dict["RelativeFarmPowerMean"].append(((turbine_power_ts.to_numpy() * np.logical_not(turbine_offline_status_ts)).sum(axis=1) / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean())
            # result_summary_dict["TotalRunningOptimizationCostSum"].append(results_df["TotalRunningOptimizationCost"].sum())
            result_summary_dict["TotalRunningOptimizationCostMean"].append(seed_df["TotalRunningOptimizationCost"].mean())
            result_summary_dict["RelativeTotalRunningOptimizationCostMean"].append((seed_df["TotalRunningOptimizationCost"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean())

            result_summary_dict["RelativeRunningOptimizationCostTerm_0"].append((seed_df["RunningOptimizationCostTerm_0"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean())
            result_summary_dict["RelativeRunningOptimizationCostTerm_1"].append((seed_df["RunningOptimizationCostTerm_1"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean())

            result_summary_dict["OptimizationConvergenceTimeMean"].append(seed_df["OptimizationConvergenceTime"].mean())
        # result_summary_dict["OptimizationConvergenceTimeSum"].append(results_df["OptimizationConvergenceTime"].sum())
    
    result_summary_df = pd.DataFrame(result_summary_dict)
    result_summary_df = result_summary_df.groupby(by=["CaseFamily", "CaseName"])[[col for col in result_summary_df.columns if col not in ["CaseFamily", "CaseName", "WindSeed"]]].agg(["min", "max", "mean"])
    
    result_summary_df.to_csv(os.path.join(save_dir, f"comparison_time_series_results.csv"))

    return result_summary_df

def plot_wind_field_ts(data_df, save_path, filter_func=None):
    fig_wind, ax_wind = plt.subplots(2, 1, sharex=True, figsize=(15.12, 7.98))
    # fig_wind.set_size_inches(10, 5)

    for seed in sorted(pd.unique(data_df["WindSeed"])):
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        ax_wind[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"], label=f"Seed {seed}")
        if filter_func is not None:
            ax_wind[0].plot(seed_df["Time"], filter_func(x=seed_df["FreestreamWindDir"]), label=f"Seed {seed}")
        ax_wind[0].set(title='Wind Direction [deg]')
        ax_wind[1].plot(seed_df["Time"], seed_df["FreestreamWindMag"], label=f"Seed {seed}")
        ax_wind[1].set(title='Wind Speed [m/s]', xlabel='Time [s]', xlim=(0, seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]))
        ax_wind[0].legend()
    # fig_wind.tight_layout()
    fig_wind.savefig(save_path)
    # fig_wind.show()

    return fig_wind, ax_wind

def plot_opt_var_ts(data_df, yaw_offset_bounds, save_path):
    colors = sns.color_palette(palette='Paired')
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" in col])
    yaw_angle_change_cols = sorted([col for col in data_df.columns if "TurbineYawAngleChange_" in col])

    fig_opt_vars, ax_opt_vars = plt.subplots(2, 1, sharex=True, figsize=(15.12, 7.98))
    # fig_opt_vars.set_size_inches(10, 5)
    plot_seed = 0
    time_frac = 1;
    plot_turbine = int(len(yaw_angle_cols) // 2)
    for seed in sorted(pd.unique(data_df["WindSeed"])):
        if seed != plot_seed:
            continue
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        ax_opt_vars[0].plot(seed_df["Time"], seed_df[yaw_angle_cols[plot_turbine]])
        ax_opt_vars[0].set(title='Yaw Angles [deg]')
        ax_opt_vars[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"] - yaw_offset_bounds[0], color=colors[seed], linestyle='dotted')
        ax_opt_vars[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"] - yaw_offset_bounds[1], color=colors[seed], linestyle='dotted', label="Lower/Upper Bounds")
        ax_opt_vars[1].plot(seed_df["Time"], seed_df[yaw_angle_change_cols[plot_turbine]], color=colors[seed], linestyle='-')
        ax_opt_vars[1].set(title='Yaw Angles Change [deg]', xlabel='Time [s]', xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) * time_frac)), ylim=(-2, 2))
    # ax_outputs[1, 0].plot(time_ts[:int(simulation_max_time // input_dict["dt"]) - 1], turbine_powers_ts)
    # ax_outputs[1, 0].set(title="Turbine Powers [MW]")
    ax_opt_vars[0].legend()
    fig_opt_vars.savefig(save_path)
    # fig_opt_vars.show()

    return fig_opt_vars, ax_opt_vars

def plot_opt_cost_ts(data_df, save_path):
    fig_opt_cost, ax_opt_cost = plt.subplots(2, 1, sharex=True, figsize=(15.12, 7.98))
    # fig_opt_cost.set_size_inches(10, 5)
    time_frac = 1;
    plot_seed = 0
    # plot_turbine = 4
    for seed in sorted(pd.unique(data_df["WindSeed"])):
        if seed != plot_seed:
            continue
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        ax_opt_cost[0].step(seed_df["Time"], seed_df["RunningOptimizationCostTerm_0"])
        ax_opt_cost[0].set(title="Optimization Farm Power Cost [-]")
        ax_opt_cost[1].step(seed_df["Time"], seed_df["RunningOptimizationCostTerm_1"])

        ax_opt_cost[1].set(title="Optimization Yaw Angle Change Cost [-]", xlabel='Time [s]', xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) * time_frac)), ylim=(0, 0.05))
    # ax_outputs[2].scatter(time_ts[:int(simulation_max_time // input_dict["dt"]) - 1], convergence_time_ts)
    # ax_outputs[2].set(title="Convergence Time [s]")
    fig_opt_cost.savefig(save_path)
    # fig_opt_cost.show()

    return fig_opt_cost, ax_opt_cost

def plot_power_ts(data_df, save_path):
    colors = sns.color_palette(palette='Paired')
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15.12, 7.98))
    # fig.set_size_inches(10, 5)
    
    turbine_wind_direction_cols = sorted([col for col in data_df.columns if "TurbineWindDir_" in col])
    turbine_power_cols = sorted([col for col in data_df.columns if "TurbinePower_" in col])
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" in col])

    plot_seed = 0
    # plot_turbine = 4
    for seed in sorted(pd.unique(data_df["WindSeed"])):
        if seed != plot_seed:
            continue
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        
        ax[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"], label="Freestream wind dir.", color="black")
        ax[0].plot(seed_df["Time"], seed_df["FilteredFreestreamWindDir"], label="Filtered freestream wind dir.", color="black", linestyle="--")
            
        # Direction
        for t, (wind_dir_col, power_col, yaw_col, color) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols, cycle(colors))):
            ax[0].plot(seed_df["Time"], seed_df[yaw_col], color=color, label="T{0:01d} yaw setpoint".format(t), linestyle=":")
            if t == 0:
                ax[1].fill_between(seed_df["Time"], seed_df[power_col] / 1e3, color=color, label="T{0:01d} power".format(t))
            else:
                ax[1].fill_between(seed_df["Time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e3, 
                                  seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e3,
                    color=color, label="T{0:01d} power".format(t))
        ax[1].plot(seed_df["Time"], seed_df[turbine_power_cols].sum(axis=1) / 1e3, color="black", label="Farm power")
    
    ax[0].set(title="Wind Direction / Yaw Angle [deg]", xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) // 1)), ylim=(245, 295))
    ax[0].legend(ncols=2, loc="upper left")
    ax[1].set(xlabel="Time [s]", title="Turbine Powers [MW]")
    ax[1].legend(ncols=2)

    results_dir = os.path.dirname(save_path)
    fig.suptitle("_".join([os.path.basename(results_dir), data_df["CaseName"].iloc[0].replace('/', '_'), "yaw_power_ts"]))
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

def plot_cost_function_pareto_curve(data_summary_df, case_studies, save_dir):
   
    """
    plot mean farm level power vs mean sum of absolute yaw changes for different values of alpha
    """
    # TODO update based on new data_summary_df format

    fig, ax = plt.subplots(1, figsize=(10.29,  5.5))
    sub_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily") == "cost_func_tuning_alpha", :]
    sub_df.reset_index(level="CaseName", inplace=True)
    sub_df.loc[:, "CaseName"] = [float(x[-1]) for x in sub_df["CaseName"].str.split("_")]
    sub_df[("RelativeFarmPowerMean", "mean")] = sub_df[("RelativeFarmPowerMean", "mean")] / 1e6
    sub_df[("RelativeFarmPowerMean", "min")] = sub_df[("RelativeFarmPowerMean", "min")] / 1e6
    sub_df[("RelativeFarmPowerMean", "max")] = sub_df[("RelativeFarmPowerMean", "max")] / 1e6

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    ax = sns.scatterplot(data=sub_df, x=("RelativeYawAngleChangeAbsMean", "mean"), y=("RelativeFarmPowerMean", "mean"),
                    size="CaseName", #size_order=reversed(sub_df["CaseName"].to_numpy()),
                    ax=ax)
    ax.set(xlabel="Mean Absolute Relative Yaw Angle Change [deg]", ylabel="Mean Relative Farm Power [MW]")
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    ax.legend([], [], frameon=False)
    fig.savefig(os.path.join(save_dir, "cost_function_pareto_curve.png"))

def plot_breakdown_robustness(data_summary_df, case_studies, save_dir):
    # TODO could also make countplot and plot all time-step data points for different values of probability
    # TODO update based on new data_summary_df format
    """
    plot mean relative farm level power vs mean relative sum of absolute yaw changes for different values of breakdown probability
    """
    
    sub_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily") == "breakdown_robustness", :]
    sub_df.reset_index(level="CaseName", inplace=True)
    sub_df[("RelativeFarmPowerMean", "mean")] = sub_df[("RelativeFarmPowerMean", "mean")] / 1e6
    sub_df[("RelativeFarmPowerMean", "min")] = sub_df[("RelativeFarmPowerMean", "min")] / 1e6
    sub_df[("RelativeFarmPowerMean", "max")] = sub_df[("RelativeFarmPowerMean", "max")] / 1e6
    # sub_df["CaseName"] = [case_studies["breakdown_robustness"]["case_names"]["vals"][int(solver_type.split("_")[-1])] for solver_type in sub_df["SolverType"]]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    fig, ax = plt.subplots(1, figsize=(10.29,  5.5))
    sns.scatterplot(data=sub_df, x=("RelativeYawAngleChangeAbsMean", "mean"), y=("RelativeFarmPowerMean", "mean"), size="CaseName", 
                    size_order=reversed(sub_df["CaseName"]), ax=ax)
    ax.set(xlabel="Mean Absolute Relative Yaw Angle Change [deg]", ylabel="Mean Relative Farm Power [MW]")
    # ax.legend()
    ax.legend_.set_title("Chance of Breakdown")
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    
    ax.legend_.texts[0].set_text("50%")
    ax.legend_.texts[1].set_text("20%")
    ax.legend_.texts[2].set_text("5%")
    ax.legend_.texts[3].set_text("2.5%")
    ax.legend_.texts[4].set_text("0%")

    fig.savefig(os.path.join(save_dir, "breakdown_robustness.png"))

if __name__ == '__main__':
    results_path = os.path.join(os.path.dirname(whoc_file), "..", "examples")
    # TODO how to find particular seed
    results_dirs = [(controller_class, 0, os.path.join(results_path, controller_dir, "outputs", "hercules_output.csv"))
                    # for seed in range(6)
                    for controller_class, controller_dir in [("Greedy", "greedy_wake_steering_florisstandin"), 
                                                             ("LUT", "lookup-based_wake_steering_florisstandin"), 
                                                             ("MPC", "mpc_wake_steering_florisstandin")]]
    input_dict = load_yaml(os.path.join(os.path.dirname(whoc_file), "../examples/hercules_input_001.yaml"))
    data_df = read_amr_outputs(results_dirs, input_dict)
    
    turbine_indices = [0, 1, 5, 6]
    wind_field_fig_dir = os.path.join(os.path.dirname(whoc_file), '../examples/wind_field_data/figs') 
    plot_yaw_power_ts(data_df, turbine_indices, os.path.join(wind_field_fig_dir, "amr_yaw_power_ts.png"))
    plot_yaw_power_distribution(data_df,  os.path.join(wind_field_fig_dir, "amr_yaw_power_dist.png"))