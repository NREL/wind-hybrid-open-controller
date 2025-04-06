import os
import re
import yaml
from itertools import cycle
import warnings
import pickle

import numpy as np
import pandas as pd
from matplotlib import colormaps
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", rc={'figure.figsize':(4,4)})

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d

factor = 1.5
# factor = 3.0 # single column
plt.rc('font', size=12*factor)          # controls default text sizes
plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
plt.rc('legend', fontsize=12*factor)    # legend fontsize
plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize

def plot_wind_farm(floris_input_files, lut_paths, save_dir):

    # fig, axarr = plt.subplots(int(len(floris_input_files)**0.5), int(len(floris_input_files)**0.5), figsize=(16, 10))
    # axarr = axarr.flatten()

    for floris_input_file, lut_path in zip(floris_input_files, lut_paths):

        df_lut = pd.read_csv(lut_path, index_col=0)
        df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
        wake_steering_interpolant = LinearNDInterpolator(
                points=df_lut[["wind_direction", "wind_speed"]].values,
                values=np.vstack(df_lut["yaw_angles_opt"].values),
                fill_value=0.0,
            )

        MIN_WS = 1.0
        MAX_WS = 8.0

        fmodel = FlorisModel(floris_input_file)
        fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[0.08])
        lut_angles = wake_steering_interpolant([270.0], [8.0])
        fmodel.set_operation(yaw_angles=lut_angles)
        # Plot 2: Show a particular flow case
        # turbine_names = [f"T{i}" for i in range(fmodel.n_turbines)]
        # layoutviz.plot_turbine_points(fmodel, ax=ax)
        # layoutviz.plot_turbine_labels(
        #     fmodel, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={"facecolor": "r"}
        # )

        # Plot 2: Show turbine rotors on flow
        horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
        ax = visualize_cut_plane(horizontal_plane, min_speed=MIN_WS, max_speed=MAX_WS, color_bar=True)
        layoutviz.plot_turbine_rotors(fmodel, ax=ax, yaw_angles=lut_angles)
        layoutviz.plot_turbine_labels(
            fmodel, ax=ax, turbine_names=[f"T{tid+1}" for tid in range(fmodel.n_turbines)],
            label_offset=fmodel.core.farm.turbine_definitions[0]["rotor_diameter"] * 0.5)

        # if a > 1:
        ax.set(xlabel="Downwind Distance [m]")
        
        # if a == 0 or a == 2:
        ax.set(ylabel="Crosswind Distance [m]")
        # figManager = plt.get_current_fig_manager()
        # figManager.full_screen_toggle()
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"wind_farm_plot_{fmodel.n_turbines}.png"))
        # plt.close(fig)

def read_case_family_agg_data(case_family, save_dir):
    all_agg_df_path = os.path.join(save_dir, case_family, "agg_results_all.csv")
    print(f"Reading case family {case_family} aggregate dataframe.")
    return pd.read_csv(all_agg_df_path, header=[0,1], index_col=[0, 1], skipinitialspace=True)

def write_case_family_agg_data(case_family, new_agg_df, save_dir):
    all_agg_df_path = os.path.join(save_dir, case_family, "agg_results_all.csv")
    print(f"Writing case family {case_family} aggregate dataframe.")
    new_agg_df.loc[new_agg_df.index.get_level_values("CaseFamily") == case_family, :].to_csv(all_agg_df_path)   

def read_case_family_time_series_data(case_family, save_dir):
    # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
    all_ts_df_path = os.path.join(save_dir, case_family, "time_series_results_all.csv") 
    print(f"Reading combined case family {case_family} time-series dataframe.")
    return pd.read_csv(all_ts_df_path, index_col=[0, 1])

def write_case_family_time_series_data(case_family, new_time_series_df, save_dir):
    all_ts_df_path = os.path.join(save_dir, case_family, "time_series_results_all.csv") # if reaggregate_simulations, or if the aggregated time series data doesn't exist for this case family, read the csv files for that case family
    print(f"Writing combined case family {case_family} time-series dataframe.")
    print(f"Directory of time_series_results_all.csv: {os.path.join(save_dir, case_family)}")

    new_time_series_df.iloc[new_time_series_df.index.get_level_values("CaseFamily") == case_family].to_csv(all_ts_df_path)

def read_time_series_data(results_path):
    # TODO fix scalability Greedy/LUT offline status at end for 25 turbines
    warnings.simplefilter('error', pd.errors.DtypeWarning)
    try:
        df = pd.read_csv(results_path, index_col=0)
        print(f"Read {results_path}")
        df = df.set_index(["CaseFamily", "CaseName"])
        return df
    except pd.errors.DtypeWarning as w:
        print(f"DtypeWarning with combined time series file {results_path}: {w}")
        warnings.simplefilter('ignore', pd.errors.DtypeWarning)
        bad_df = pd.read_csv(results_path, index_col=0)
        bad_cols = [bad_df.columns[int(s) - len(bad_df.index.names)] for s in re.findall(r"(?<=Columns \()(.*)(?=\))", w.args[0])[0].split(",")]
        bad_df.loc[bad_df[bad_cols].isna().any(axis=1)][["Time", "CaseFamily", "CaseName"]].values
        bad_df["Time"].max()
    except pd.errors.EmptyDataError as e:
        print(f"Dataframe {results_path} not read correctly due to error {e}")

def generate_outputs(agg_results_df, save_dir):

    # agg_results_df.sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True)[("RelativeTotalRunningOptimizationCostMean", "mean")]
    # agg_results_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[("YawAngleChangeAbsMean", "mean")]
    # agg_results_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[("FarmPowerMean", "mean")]

    # agg_results_df[("FarmPowerMean", "mean")]


    # agg_results_df.sort_values(by=("TotalRunningOptimizationCostMean", "mean"), ascending=True).groupby(level=0)[("TotalRunningOptimizationCostMean", "mean")]
    # agg_results_df[("TotalRunningOptimizationCostMean", "mean")].sort_values(ascending=True)

    # (-(agg_results_df[("FarmPowerMean", "mean")] * 1e-8) + (agg_results_df[("YawAngleChangeAbsMean", "mean")])).sort_values(ascending=True)
    # (agg_results_df[("FarmPowerMean", "mean")].sort_values(ascending=False)).to_csv("./mpc_configs_maxpower.csv")
    # (agg_results_df[("YawAngleChangeAbsMean", "mean")].sort_values(ascending=True)).to_csv("./mpc_configs_minyaw.csv")
    # ((agg_results_df[("FarmPowerMean", "mean")] * 1e-7) - agg_results_df[("YawAngleChangeAbsMean", "mean")]).sort_values(ascending=False).to_csv("./mpc_configs_mincost")
    # ((agg_results_df[("FarmPowerMean", "mean")] * 1e-7) / agg_results_df[("YawAngleChangeAbsMean", "mean")]).sort_values(ascending=False).to_csv("./mpc_configs_max_power_yaw_ratio.csv")

    # # agg_results_df.groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).head(3))[("RelativeTotalRunningOptimizationCostMean", "mean")]
    # x = agg_results_df.loc[agg_results_df[("RelativeYawAngleChangeAbsMean", "mean")] > 0, :].groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeYawAngleChangeAbsMean", "mean"), ascending=True).head(10))[("RelativeYawAngleChangeAbsMean", "mean")]
    # y = agg_results_df.groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeFarmPowerMean", "mean"), ascending=False).head(10))[("RelativeFarmPowerMean", "mean")]
    agg_results_df.groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeFarmPowerMean", "mean"), ascending=False).head(10))[("RelativeFarmPowerMean", "mean")] 


    # generate results table in tex
    # solver_type_df = agg_results_df.loc[agg_results_df.index.get_level_values("CaseFamily") == "solver_type", :].reset_index("CaseName")
    # solver_type_df.loc[solver_type_df.CaseName == 'SLSQP', ("RelativeYawAngleChangeAbsMean", "mean")]

    # x = agg_results_df.loc[(agg_results_df.index.get_level_values("CaseFamily") != "scalability") & (agg_results_df.index.get_level_values("CaseFamily") != "breakdown_robustness"), :]
    # x = x.loc[:, x.columns.get_level_values(1) == "mean"]
    # x = x.loc[:, ("RelativeTotalRunningOptimizationCostMean", "mean")]
    # x = x.groupby("CaseFamily", group_keys=False).nsmallest(3)
    # Set alpha to 0.1, n_horizon to 12, solver to SLSQP, warm-start to LUT

    get_result = lambda case_family, case_name, parameter: agg_results_df.loc[(agg_results_df.index.get_level_values("CaseFamily") == case_family) & (agg_results_df.index.get_level_values("CaseName") == case_name), (parameter, "mean")].iloc[0]
    # get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean')
    # get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean')
    # get_result('solver_type', 'SLSQP', 'TotalRunningOptimizationCostMean')
    # get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTime')

    # if all(col in agg_results_df.index.get_level_values("CaseFamily") for col in 
    #        ["baseline_controllers", "solver_type",
    #          "wind_preview_type", "warm_start", 
    #           "horizon_length", "scalability"]):
    values = {"Baseline": {"labels": ["Greedy", "LUT"], 
                           "farm_power": [get_result('baseline_controllers', 'Greedy', 'FarmPowerMean') / 1e6, get_result('baseline_controllers', 'LUT', 'FarmPowerMean') / 1e6],
                           "yaw_change": [get_result('baseline_controllers', 'Greedy', 'YawAngleChangeAbsMean'), get_result('baseline_controllers', 'LUT', 'YawAngleChangeAbsMean')],
                           "conv_time": [get_result('baseline_controllers', 'Greedy', 'OptimizationConvergenceTime'), get_result('baseline_controllers', 'LUT', 'OptimizationConvergenceTime')]
                           },
                "Solver": {"labels": ["SLSQP", "Sequential SLSQP", "Serial Refine"], 
                           "farm_power": [get_result('solver_type', 'SLSQP', 'FarmPowerMean') / 1e6, get_result('solver_type', 'Sequential SLSQP', 'FarmPowerMean') / 1e6, get_result('solver_type', 'Sequential Refine', 'FarmPowerMean') / 1e6],
                           "yaw_change": [get_result('solver_type', 'SLSQP', 'YawAngleChangeAbsMean'), get_result('solver_type', 'Sequential SLSQP', 'YawAngleChangeAbsMean'), get_result('solver_type', 'Sequential Refine', 'YawAngleChangeAbsMean')],
                           "conv_time": [get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTime'), get_result('solver_type', 'Sequential SLSQP', 'OptimizationConvergenceTime'), get_result('solver_type', 'Sequential Refine', 'OptimizationConvergenceTime')]
                           },
                "Wind Preview Type": {"labels": ["Perfect", "Persistent", 
                                                  "$3$ Elliptical Interval Samples", "$5$ Elliptical Interval Samples", "$11$ Elliptical Interval Samples", 
                                                  "$3$ Rectangular Interval Samples", "$5$ Rectangular Interval Samples", "$11$ Rectangular Interval Samples", 
                                                  "$25$ Stochastic Samples", "$50$ Stochastic Samples", "$100$ Stochastic Samples"], 
                           "farm_power": ([get_result('wind_preview_type', 'Perfect', 'FarmPowerMean') / 1e6, get_result('wind_preview_type', 'Persistent', 'FarmPowerMean') / 1e6] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Elliptical {x}", 'FarmPowerMean') / 1e6 for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Rectangular {x}", 'FarmPowerMean') / 1e6 for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Sample {x}", 'FarmPowerMean') / 1e6 for x in [25, 50, 100]]),
                           "yaw_change": ([get_result('wind_preview_type', 'Perfect', 'YawAngleChangeAbsMean'), get_result('wind_preview_type', 'Persistent', 'YawAngleChangeAbsMean')] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Elliptical {x}", 'YawAngleChangeAbsMean') for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Rectangular {x}", 'YawAngleChangeAbsMean') for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Sample {x}", 'YawAngleChangeAbsMean') for x in [25, 50, 100]]),
                           "conv_time": ([get_result('wind_preview_type', 'Perfect', 'OptimizationConvergenceTime'), get_result('wind_preview_type', 'Persistent', 'OptimizationConvergenceTime')] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Elliptical {x}", 'OptimizationConvergenceTime') for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Interval Rectangular {x}", 'OptimizationConvergenceTime') for x in [3, 5, 11]] 
                                          + [get_result('wind_preview_type', f"Stochastic Sample {x}", 'OptimizationConvergenceTime') for x in [25, 50, 100]]),
                           },
                "Warm-Starting Method": {"labels": ["Greedy", "LUT", "Previous Solution"], 
                           "farm_power": [get_result('warm_start', 'Greedy', 'FarmPowerMean') / 1e6, get_result('warm_start', 'LUT', 'FarmPowerMean') / 1e6, get_result('warm_start', 'Previous', 'FarmPowerMean') / 1e6],
                           "yaw_change": [get_result('warm_start', 'Greedy', 'YawAngleChangeAbsMean'), get_result('warm_start', 'LUT', 'YawAngleChangeAbsMean'), get_result('warm_start', 'Previous', 'YawAngleChangeAbsMean')],
                           "conv_time": [get_result('warm_start', 'Greedy', 'OptimizationConvergenceTime'), get_result('warm_start', 'LUT', 'OptimizationConvergenceTime'), get_result('warm_start', 'Previous', 'OptimizationConvergenceTime')]
                           }}
    
    all_farm_powers = np.concatenate([vals["farm_power"] for vals in values.values()])
    all_yaw_changes = np.concatenate([vals["yaw_change"] for vals in values.values()])
    all_conv_times = np.concatenate([vals["conv_time"] for vals in values.values()])
    # val_type_ranges = {"farm_power": (all_farm_powers.min(), all_farm_powers.max()),
    #                    "yaw_change": (all_yaw_changes.min(), all_yaw_changes.max()),
    #                    "conv_time": (all_conv_times.min(), all_conv_times.max())}
    val_type_ranges = {"farm_power": (sorted(all_farm_powers), np.round(np.linspace(1.0, 0.4, len(all_farm_powers)), 3)),
                       "yaw_change": (sorted(all_yaw_changes), np.round(np.linspace(1.0, 0.4, len(all_yaw_changes)), 3)),
                       "conv_time": (sorted(all_conv_times), np.round(np.linspace(1.0, 0.4, len(all_conv_times)), 3))}

    for case_family, vals in values.items():
        for val_type in ["farm_power", "yaw_change", "conv_time"]:
            # grey_shades = np.round(np.interp(values[case_family][val_type], xp=val_type_ranges[val_type], fp=[1.0, 0.4]), 3)
            # grey_shades = np.round(np.linspace(1.0, 0.4, len(values[case_family][val_type])), 3)[np.argsort(values[case_family][val_type])]
            grey_shades = [val_type_ranges[val_type][1][val_type_ranges[val_type][0].index(val)] for val in values[case_family][val_type]]
            values[case_family][val_type] = [(val, grey_shades[v]) for v, val in enumerate(values[case_family][val_type])]

    compare_results_latex = f"\\begin{{tabular}}{{l|lllll}}\n"
    compare_results_latex += f"\\textbf{{Case Family}} & \\textbf{{Case Name}} & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Farm Power [MW]}}}} & \\thead{{\\textbf{{Mean Absolute}} \\\\ \\textbf{{Yaw Angle Change [$^\\circ$]}}}} & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Convergence Time [s]}}}} \\\\ \\hline \n" 
    
    for case_family, vals in values.items():
        compare_results_latex += f"\\multirow{{3}}{{*}}{{\\textbf{{{case_family}}}}} & {vals['labels'][0]} & ${vals['farm_power'][0][0]:.3f}$ \\cellcolor[gray]{{{vals['farm_power'][0][1]}}} & ${vals['yaw_change'][0][0]:.3f}$ \\cellcolor[gray]{{{vals['yaw_change'][0][1]}}} & ${vals['conv_time'][0][0]:.2f}$ \\cellcolor[gray]{{{vals['conv_time'][0][1]}}} \\\\ \n"
        for i in range(1, len(vals['labels']) - 1):
            compare_results_latex += f" & {vals['labels'][i]} & ${vals['farm_power'][i][0]:.3f}$ \\cellcolor[gray]{{{vals['farm_power'][i][1]}}} & ${vals['yaw_change'][i][0]:.3f}$ \\cellcolor[gray]{{{vals['yaw_change'][i][1]}}} & ${vals['conv_time'][i][0]:.2f}$ \\cellcolor[gray]{{{vals['conv_time'][i][1]}}}  \\\\ \n"
        compare_results_latex += f" & {vals['labels'][-1]} & ${vals['farm_power'][-1][0]:.3f}$  \\cellcolor[gray]{{{vals['farm_power'][-1][1]}}} & ${vals['yaw_change'][-1][0]:.3f}$ \\cellcolor[gray]{{{vals['yaw_change'][-1][1]}}} & ${vals['conv_time'][-1][0]:.2f}$ \\cellcolor[gray]{{{vals['conv_time'][-1][1]}}}  \\\\ \\hline \n"

    compare_results_latex += f"\\end{{tabular}}"

    # case_family = 'Solver'
    # compare_results_latex += f"\\multirow{{3}}{{*}}{{\\textbf{{Solver}}}} & {values[case_family]['labels'][0]} & ${values[case_family]['farm_power'][0]:.3f}$  & ${values[case_family]['yaw_change'][0]:.3f}$ & ${values[case_family]['conv_time'][0]:.2f}$ \\\\ \n"
    # for i in range(1, len(values[case_family]['labels']) - 1):
    #     compare_results_latex += f" & {values[case_family]['labels'][i]} & ${values[case_family]['farm_power'][i]:.3f}$  & ${values[case_family]['yaw_change'][i]:.3f}$ & ${values[case_family]['conv_time'][i]:.2f}$ \\\\ \n"
    # compare_results_latex += f" & {values[case_family]['labels'][-1]} & ${values[case_family]['farm_power'][-1]:.3f}$  & ${values[case_family]['yaw_change'][-1]:.3f}$ & ${values[case_family]['conv_time'][-1]:.2f}$ \\\\ \hline \n"

    # case_family = 'Wind Preview Model'
    # compare_results_latex += f"\\multirow{{3}}{{*}}{{\\textbf{{Wind Preview Model}}}} & {values[case_family]['labels'][0]} & ${values[case_family]['farm_power'][0]:.3f}$  & ${values[case_family]['yaw_change'][0]:.3f}$ & ${values[case_family]['conv_time'][0]:.2f}$ \\\\ \n"
    # for i in range(1, len(values[case_family]['labels']) - 1):
    #     compare_results_latex += f" & {values[case_family]['labels'][i]} & ${values[case_family]['farm_power'][i]:.3f}$  & ${values[case_family]['yaw_change'][i]:.3f}$ & ${values[case_family]['conv_time'][i]:.2f}$ \\\\ \n"
    # compare_results_latex += f" & {values[case_family]['labels'][-1]} & ${values[case_family]['farm_power'][-1]:.3f}$  & ${values[case_family]['yaw_change'][-1]:.3f}$ & ${values[case_family]['conv_time'][-1]:.2f}$ \\\\ \hline \n"

    # case_family = 'Warm-Starting Method'
    # compare_results_latex += f"\\multirow{{3}}{{*}}{{\\textbf{{Warm-Starting Method}}}} & {values[case_family]['labels'][0]} & ${values[case_family]['farm_power'][0]:.3f}$  & ${values[case_family]['yaw_change'][0]:.3f}$ & ${values[case_family]['conv_time'][0]:.2f}$ \\\\ \n"
    # for i in range(1, len(values[case_family]['labels']) - 1):
    #     compare_results_latex += f" & {values[case_family]['labels'][i]} & ${values[case_family]['farm_power'][i]:.3f}$  & ${values[case_family]['yaw_change'][i]:.3f}$ & ${values[case_family]['conv_time'][i]:.2f}$ \\\\ \n"
    # compare_results_latex += f" & {values[case_family]['labels'][-1]} & ${values[case_family]['farm_power'][-1]:.3f}$  & ${values[case_family]['yaw_change'][-1]:.3f}$ & ${values[case_family]['conv_time'][-1]:.2f}$ \\\\ \hline \n"

    # compare_results_latex2 = (
    #     f"\\begin{{tabular}}{{l|lllll}}\n"
    #     f"\\textbf{{Case Family}} & \\textbf{{Case Name}} & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Farm Power [MW]}}}}                                                                    & \\thead{{\\textbf{{Mean Absolute}} \\\\ \\textbf{{Yaw Angle Change [$^\\circ$]}}}}                           & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Convergence Time [s]}}}} \\\\ \\hline \n"
    #     f"\\multirow{{3}}{{*}}{{\\textbf{{Baseline}}}} & Greedy                       & ${get_result('baseline_controllers', 'Greedy', 'FarmPowerMean') / 1e6:.3f}$                           & ${get_result('baseline_controllers', 'Greedy', 'YawAngleChangeAbsMean'):.3f}$                                & ${get_result('baseline_controllers', 'Greedy', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                           LUT                           & ${get_result('baseline_controllers', 'LUT', 'FarmPowerMean') / 1e6:.3f}$                              & ${get_result('baseline_controllers', 'LUT', 'YawAngleChangeAbsMean'):.3f}$                                   & ${get_result('baseline_controllers', 'LUT', 'OptimizationConvergenceTime'):.2f}$ \\\\ \\hline \n"
    #     f"\\multirow{{3}}{{*}}{{\\textbf{{Solver}}}} & \\textbf{{SLSQP}}            & ${get_result('solver_type', 'SLSQP', 'FarmPowerMean') / 1e6:.3f}$                                     & ${get_result('solver_type', 'SLSQP', 'YawAngleChangeAbsMean'):.3f}$                                          & ${get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                          Sequential SLSQP               & ${get_result('solver_type', 'Sequential SLSQP', 'FarmPowerMean') / 1e6:.3f}$                          & ${get_result('solver_type', 'Sequential SLSQP', 'YawAngleChangeAbsMean'):.3f}$                               & ${get_result('solver_type', 'Sequential SLSQP', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                          Serial Refine                  & ${get_result('solver_type', 'Sequential Refine', 'FarmPowerMean') / 1e6:.3f}$                         & ${get_result('solver_type', 'Sequential Refine', 'YawAngleChangeAbsMean'):.3f}$                              & ${get_result('solver_type', 'Sequential Refine', 'OptimizationConvergenceTime'):.2f}$  \\\\ \\hline \n"
    #     f"\\multirow{{3}}{{*}}{{\\textbf{{Wind Preview Model}}}} & Perfect          & ${get_result('wind_preview_type', 'Perfect', 'FarmPowerMean') / 1e6:.3f}$                             & ${get_result('wind_preview_type', 'Perfect', 'YawAngleChangeAbsMean'):.3f}$                                  & ${get_result('wind_preview_type', 'Perfect', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      Persistent         & ${get_result('wind_preview_type', 'Persistent', 'FarmPowerMean') / 1e6:.3f}$                          & ${get_result('wind_preview_type', 'Persistent', 'YawAngleChangeAbsMean'):.3f}$                               & ${get_result('wind_preview_type', 'Persistent', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $3$ Elliptical Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 3', 'FarmPowerMean') / 1e6:.3f}$    & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 3', 'YawAngleChangeAbsMean'):.3f}$         & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 3', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $5$ Elliptical Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 5', 'FarmPowerMean') / 1e6:.3f}$    & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 5', 'YawAngleChangeAbsMean'):.3f}$         & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 5', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $7$ Elliptical Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 7', 'FarmPowerMean') / 1e6:.3f}$    & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 7', 'YawAngleChangeAbsMean'):.3f}$         & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 7', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $9$ Elliptical Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 9', 'FarmPowerMean') / 1e6:.3f}$    & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 9', 'YawAngleChangeAbsMean'):.3f}$         & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 9', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $11$ Elliptical Interval Samples    & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 11', 'FarmPowerMean') / 1e6:.3f}$   & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 11', 'YawAngleChangeAbsMean'):.3f}$        & ${get_result('wind_preview_type', 'Stochastic Interval Elliptical 11', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $3$ Rectangular Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 3', 'FarmPowerMean') / 1e6:.3f}$   & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 3', 'YawAngleChangeAbsMean'):.3f}$        & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 3', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $5$ Rectangular Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 5', 'FarmPowerMean') / 1e6:.3f}$   & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 5', 'YawAngleChangeAbsMean'):.3f}$        & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 5', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $7$ Rectangular Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 7', 'FarmPowerMean') / 1e6:.3f}$   & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 7', 'YawAngleChangeAbsMean'):.3f}$        & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 7', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $9$ Rectangular Interval Samples     & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 9', 'FarmPowerMean') / 1e6:.3f}$   & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 9', 'YawAngleChangeAbsMean'):.3f}$        & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 9', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $11$ Rectangular Interval Samples    & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 11', 'FarmPowerMean') / 1e6:.3f}$  & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 11', 'YawAngleChangeAbsMean'):.3f}$       & ${get_result('wind_preview_type', 'Stochastic Interval Rectangular 11', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $25$ Stochastic Samples     & ${get_result('wind_preview_type', 'Stochastic Sample 25', 'FarmPowerMean') / 1e6:.3f}$                & ${get_result('wind_preview_type', 'Stochastic Sample 25', 'YawAngleChangeAbsMean'):.3f}$                     & ${get_result('wind_preview_type', 'Stochastic Sample 25', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $50$ Stochastic Samples     & ${get_result('wind_preview_type', 'Stochastic Sample 50', 'FarmPowerMean') / 1e6:.3f}$                & ${get_result('wind_preview_type', 'Stochastic Sample 50', 'YawAngleChangeAbsMean'):.3f}$                     & ${get_result('wind_preview_type', 'Stochastic Sample 50', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $100$ Stochastic Samples    & ${get_result('wind_preview_type', 'Stochastic Sample 100', 'FarmPowerMean') / 1e6:.3f}$               & ${get_result('wind_preview_type', 'Stochastic Sample 100', 'YawAngleChangeAbsMean'):.3f}$                    & ${get_result('wind_preview_type', 'Stochastic Sample 100', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $250$ Stochastic Samples    & ${get_result('wind_preview_type', 'Stochastic Sample 250', 'FarmPowerMean') / 1e6:.3f}$               & ${get_result('wind_preview_type', 'Stochastic Sample 250', 'YawAngleChangeAbsMean'):.3f}$                    & ${get_result('wind_preview_type', 'Stochastic Sample 250', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                      $500$ Stochastic Samples    & ${get_result('wind_preview_type', 'Stochastic Sample 500', 'FarmPowerMean') / 1e6:.3f}$               & ${get_result('wind_preview_type', 'Stochastic Sample 500', 'YawAngleChangeAbsMean'):.3f}$                    & ${get_result('wind_preview_type', 'Stochastic Sample 500', 'OptimizationConvergenceTime'):.2f}$ \\\\ \\hline \n"
    #     f"\\multirow{{3}}{{*}}{{\\textbf{{Warm-Starting Method}}}} & Greedy         & ${get_result('warm_start', 'Greedy', 'FarmPowerMean') / 1e6:.3f}$                                     & ${get_result('warm_start', 'Greedy', 'YawAngleChangeAbsMean'):.3f}$                                          & ${get_result('warm_start', 'Greedy', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                       \\textbf{{LUT}}   & ${get_result('warm_start', 'LUT', 'FarmPowerMean') / 1e6:.3f}$                                        & ${get_result('warm_start', 'LUT', 'YawAngleChangeAbsMean'):.3f}$                                             & ${get_result('warm_start', 'LUT', 'OptimizationConvergenceTime'):.2f}$ \\\\ \n"
    #     f"&                                                       Previous Solution & ${get_result('warm_start', 'Previous', 'FarmPowerMean') / 1e6:.3f}$                                   & ${get_result('warm_start', 'Previous', 'YawAngleChangeAbsMean'):.3f}$                                        & ${get_result('warm_start', 'Previous', 'OptimizationConvergenceTime'):.2f}$ \\\\ \\hline \n"
    #     f"\\end{{tabular}}"
    #     )
    with open(os.path.join(save_dir, "comparison_time_series_results_table.tex"), "w") as fp:
            fp.write(compare_results_latex)

def plot_simulations(time_series_df, plotting_cases, save_dir, include_power=True, legend_loc="best", single_plot=False):
    # TODO delete all extra files in directories before rerunning simulations
    if single_plot:
        yaw_power_ts_fig, yaw_power_ts_ax = plt.subplots(int(1 + include_power), 1, sharex=True) # 1 subplot of yaw, another for power
    
    for case_family in pd.unique(time_series_df.index.get_level_values("CaseFamily")):
        case_family_df = time_series_df.loc[(time_series_df.index.get_level_values("CaseFamily") == case_family), :]
        for case_name in pd.unique(case_family_df.index.get_level_values("CaseName")):
            if (case_family, case_name) not in plotting_cases:
                continue
            case_name_df = case_family_df.loc[case_family_df.index.get_level_values("CaseName") == case_name, :]
            input_fn = [fn for fn in os.listdir(os.path.join(save_dir, case_family)) if "input_config" in fn and case_name in fn][0]
            
            with open(os.path.join(save_dir, case_family, input_fn), 'rb') as fp:
                input_config =  pickle.load(fp)
            if single_plot:
                fig, _ = plot_yaw_power_ts(case_name_df, os.path.join(save_dir, case_family, f"yaw_power_ts_{case_name}.png"), include_power=include_power, legend_loc=legend_loc,
                                        controller_dt=None, include_filtered_wind_dir=(case_family=="baseline_controllers"), single_plot=single_plot, fig=yaw_power_ts_fig, ax=yaw_power_ts_ax, case_label=case_name)
            else:
                fig, _ = plot_yaw_power_ts(case_name_df, os.path.join(save_dir, case_family, f"yaw_power_ts_{case_name}.png"), include_power=include_power, legend_loc=legend_loc,
                                        controller_dt=None, include_filtered_wind_dir=(case_family=="baseline_controllers_3"), single_plot=single_plot)
                                    #    controller_dt=input_config["controller"]["dt"])

    if False:
        # TODO why is the target setpoint always 252? If the equality across turbines is due to the dynamic constraint, why are the differences across time-steps not equal?
        x = results_dfs["baseline_controllers_LUT"][[col for col in results_dfs["baseline_controllers_LUT"] if "TurbineYawAngle_" in col]]
        (x.nunique(axis=1) == 1).all()
        print(np.where(~(x.nunique(axis=1) == 1))) # all equal at all indices

        yaw_cols = [col for col in results_dfs["wind_preview_type_Persistent"] if "TurbineYawAngle_" in col]
        persistent_df = results_dfs["wind_preview_type_Persistent"][["Time", "FreestreamWindDir", "FreestreamWindMag"] + yaw_cols]
        (persistent_df[yaw_cols].nunique(axis=1) == 1).all()
        print(np.where(~(persistent_df.nunique(axis=1) == 1))) # all equal until 73, due to thresholding?

        perfect_df = results_dfs["wind_preview_type_Perfect"][[col for col in results_dfs["wind_preview_type_Perfect"] if "TurbineYawAngle_" in col]]
        (perfect_df.nunique(axis=1) == 1).all()
        print(np.where(~(perfect_df.nunique(axis=1) == 1))) # all equal until 58, due to thresholding?

        x = results_dfs["wind_preview_type_Stochastic"][[col for col in results_dfs["wind_preview_type_Stochastic"] if "TurbineYawAngle_" in col]]
        (x.nunique(axis=1) == 1).all()
        print(np.where(~(x.nunique(axis=1) == 1))) # all equal until 56, due to thresholding?

        a = results_dfs["baseline_controllers_MPC_with_Filter"][[col for col in results_dfs["baseline_controllers_MPC_with_Filter"] if "TurbineYawAngle_" in col]]
        (a.nunique(axis=1) == 1).all()
        print(np.where(~(a.nunique(axis=1) == 1))) # all equal until 10, due to thresholding?

        b = results_dfs["baseline_controllers_MPC_without_Filter"][[col for col in results_dfs["baseline_controllers_MPC_without_Filter"] if "TurbineYawAngle_" in col]]
        (b.nunique(axis=1) == 1).all()
        print(np.where(~(b.nunique(axis=1) == 1))) # all equal until 10, due to thresholding?

        (a == b).all()

    # summary_df = pd.read_csv(os.path.join(save_dir, f"comparison_time_series_results.csv"), index_col=0)
    # barplot_opt_cost(summary_df, save_dir, relative=True)

# TODO this should be in another file
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
    df = df.rename(columns={col: new_col for col, new_col in zip(cols, new_cols)})

    # remove rows corresponding to all zero turbine powers
    df = df.loc[~(df[[col for col in df.columns if f"turbine_powers" in col]] == 0).all(axis="columns"), :]
    df = df.rename(columns={col: f"TurbinePower_{col.split('_')[-1]}" for col in df.columns if "turbine_powers" in col})
    df = df.rename(columns={col: f"TurbineYawAngle_{col.split('_')[-1]}" for col in df.columns if "turbine_yaw_angles" in col})
    df.loc[:, "Time"] = df["Time"] - df.iloc[0]["Time"]

    df["ControllerClass"] = pd.Categorical(df["ControllerClass"], ["Greedy", "LUT", "MPC"])
    df = df.sort_values(by=["ControllerClass", "Time"])

    df["FarmAbsoluteYawAngleChange"] = df[[col for col in df.columns if "TurbineYawAngleChange_" in col]].abs().sum(axis=1)
    df["FarmPower"] = df[[col for col in df.columns if "TurbinePower_" in col]].sum(axis=1)

    return df

# def plot_yaw_power_ts(data_df, turbine_indices, save_path, seed=0):
#     """
#     For each controller class (different lineplots), and for a select few turbine_indices (different subplots), plot their angle changes and powers vs time with a combo plot for each turbine.
#     """
#     n_rows = int(np.floor(np.sqrt(len(turbine_indices))))
#     if np.sqrt(len(turbine_indices)) % 1.0 == 0:
#         fig1, ax1 = plt.subplots(n_rows, n_rows, sharex=True, sharey=True)
#     else:
#         fig1, ax1 = plt.subplots(n_rows, n_rows + 1, sharex=True, sharey=True)
#     ax1 = ax1.flatten()
    
#     # data_df = data_df.melt()

#     for i in range(len(turbine_indices)):
#         ax1[i] = sns.lineplot(x="Time", y=f"TurbineYawAngleChange_{turbine_indices[i]}", hue="ControllerClass", data=data_df.loc[data_df["WindSeed"] == seed], 
#                               color=sns.color_palette()[0],
#                               ax=ax1[i], sort=False, legend=i==0)
#         ax1[i].xaxis.label.set_text(f"Time [s]")
#         ax1[i].title.set_text(f"Turbine {turbine_indices[i]}Absolute Yaw Angle Change [$^\\circ$]")
#         # ax1[i].yaxis.label.set_color(ax1[i].get_lines()[0].get_color())
#         # ax1[i].tick_params(axis="y", color=ax1[i].get_lines()[0].get_color())
#     ax1[0].legend(loc="upper right")
#     # ax2 = []
#     # for i in range(len(turbine_indices)):
#     #     ax2.append(ax1[i].twinx())

#     if np.sqrt(len(turbine_indices)) % 1.0 == 0:
#         fig2, ax2 = plt.subplots(n_rows, n_rows, sharex=True, sharey=True)
#     else:
#         fig2, ax2 = plt.subplots(n_rows, n_rows + 1, sharex=True, sharey=True)
#     ax2 = ax2.flatten()

#     for i in range(len(turbine_indices)):
#         ax2[i] = sns.lineplot(x="Time", y=f"TurbinePower_{turbine_indices[i]}", hue="ControllerClass", data=data_df.loc[data_df["WindSeed"] == seed], 
#                               color=sns.color_palette()[1],
#                               ax=ax2[i], sort=False, legend=i==0)
#         ax2[i].xaxis.label.set_text(f"Time [s]")
#         ax2[i].title.set_text(f"Turbine {turbine_indices[i]} Power [MW]")
#         # ax2[i].yaxis.label.set_color(ax2[i].get_lines()[0].get_color())
#         # ax2[i].tick_params(axis="y", color=ax2[i].get_lines()[0].get_color())

#     ax2[0].legend(loc="upper right")

#     fig1.set_size_inches((11.2, 4.8))
#     fig1.show()
#     fig1.savefig(save_path.replace(".png", "_abs_yaw_change.png"))

#     fig2.set_size_inches((11.2, 4.8))
#     fig2.show()
#     fig2.savefig(save_path.replace(".png", "_power.png"))


def plot_yaw_power_distribution(data_df, save_path):

    """
    For each controller class (categorical, along x-axis), plot the distribution of total farm powers and total absolute yaw angle changes over all time-steps and seeds (different subplots), plot their angle changes and powers vs time with a combo plot for each turbine.

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
    """
    plt.figure(1)
    ax1 = sns.catplot(x="ControllerClass", y="FarmAbsoluteYawAngleChange", data=data_df, kind="boxen")
    ax1.ax.xaxis.label.set_text("Controller")
    ax1.ax.title.set_text("Farm Absolute Yaw Angle Change [$^\\circ$]")
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

# def process_all_time_series(results_dfs, save_dir):
#     result_summary = []
#     for df_name, results_df in results_dfs.items():
#         result_summary.append(process_case(df_name, results_df, save_dir))

#     result_summary_df = pd.DataFrame(result_summary, 
                                    #  columns=["CaseFamily", "CaseName", "WindSeed",
                                            #   "YawAngleChangeAbsMean", "RelativeYawAngleChangeAbsMean",
                                            #   "FarmPowerMean", "RelativeFarmPowerMean", 
                                            #   "TotalRunningOptimizationCostMean", "RelativeTotalRunningOptimizationCostMean",
                                            #   "RelativeRunningOptimizationCostTerm_0", "RelativeRunningOptimizationCostTerm_1"])
#     result_summary_df = result_summary_df.groupby(by=["CaseFamily", "CaseName"])[[col for col in result_summary_df.columns if col not in ["CaseFamily", "CaseName", "WindSeed"]]].agg(["min", "max", "mean"])
    
#     result_summary_df.to_csv(os.path.join(save_dir, f"comparison_time_series_results.csv"))

#     return result_summary_df

def aggregate_time_series_data(time_series_df, input_dict_path, n_seeds):
    """_summary_
    Process csv data (all wind seeds) for single case name and single case family, from single diretory in floris_case_studies
    Args:
        time_series_df (_type_): _description_
        yaml_path (_type_): _description_
        n_seeds (_type_): _description_

    Returns:
        _type_: _description_
    """
    # x = time_series_df.reset_index(level=["CaseFamily", "CaseName"], drop=True)
    time_series_df = time_series_df.drop(columns=[col for col in time_series_df.columns if "Predicted" in col or "Stddev" in col])
    case_seeds = pd.unique(time_series_df["WindSeed"])
    case_family = time_series_df.index.get_level_values("CaseFamily")[0]
    # case_family = df_name.replace(f"_{results_df['CaseName'].iloc[0]}", "")
    case_name = time_series_df.index.get_level_values("CaseName")[0]
    if len(case_seeds) < n_seeds:
       print(f"NOT aggregating data for {case_family}={case_name} due to insufficient seed simulations.")
       return None

    with open(input_dict_path, 'rb') as fp:
        input_config = pickle.load(fp)

    stoptime = (np.ceil(input_config["hercules_comms"]["helics"]["config"]["stoptime"] / input_config["simulation_dt"]) * input_config["simulation_dt"]).astype(int)
    time_series_df = time_series_df.loc[time_series_df["Time"] < stoptime, :]
    time = pd.unique(time_series_df["Time"])
    
    if len(pd.unique(time)) != int(stoptime // input_config["simulation_dt"]):
       print(f"NOT aggregating data for {case_family}={case_name} due to insufficient time steps.")
       return None
   
    result_summary = []
    # input_fn = f"input_config_case_{case_name}.yaml"
    print(f"Aggregating data for {case_family}={case_name}")
    
    if "lpf_start_time" in input_config["controller"]:
        lpf_start_time = input_config["controller"]["lpf_start_time"]
    else:
        lpf_start_time = 180.0
    
    for seed in case_seeds:

        if time_series_df["Time"].max() > lpf_start_time:
            seed_df = time_series_df.loc[(time_series_df["WindSeed"] == seed) & (time_series_df["Time"] >= lpf_start_time), :]
        else:
            seed_df = time_series_df.loc[(time_series_df["WindSeed"] == seed), :]
        
        yaw_angles_change_ts = seed_df[sorted([c for c in time_series_df.columns if "TurbineYawAngleChange_" in c], key=lambda s: int(s.split("_")[-1]))]
        turbine_offline_status_ts = seed_df[sorted([c for c in time_series_df.columns if "TurbineOfflineStatus_" in c], key=lambda s: int(s.split("_")[-1]))]
        turbine_power_ts = seed_df[sorted([c for c in time_series_df.columns if "TurbinePower_" in c], key=lambda s: int(s.split("_")[-1]))]
        
        try:
            result_summary.append((seed_df.index.get_level_values("CaseFamily")[0], 
                                   seed_df.index.get_level_values("CaseName")[0], 
                                   seed, 
                                yaw_angles_change_ts.abs().sum(axis=1).mean(), 
                                ((yaw_angles_change_ts.abs().to_numpy() * np.logical_not(turbine_offline_status_ts)).sum(axis=1) / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean(),
                                turbine_power_ts.sum(axis=1).mean(), 
                                ((turbine_power_ts.to_numpy() * np.logical_not(turbine_offline_status_ts)).sum(axis=1) / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean(),
                                seed_df["TotalRunningOptimizationCost"].mean(), 
                                (seed_df["TotalRunningOptimizationCost"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean(),
                                (seed_df["RunningOptimizationCostTerm_0"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean(),
                                (seed_df["RunningOptimizationCostTerm_1"] / ((np.logical_not(turbine_offline_status_ts)).sum(axis=1))).mean(),
                                seed_df["OptimizationConvergenceTime"].mean()))
        except ZeroDivisionError:
            print("oh")
        
    # print(f"Aggregated data for {case_family}={case_name}")
    agg_df = pd.DataFrame(result_summary, columns=["CaseFamily", "CaseName", "WindSeed",
                                              "YawAngleChangeAbsMean", "RelativeYawAngleChangeAbsMean",
                                              "FarmPowerMean", "RelativeFarmPowerMean", 
                                              "TotalRunningOptimizationCostMean", "RelativeTotalRunningOptimizationCostMean",
                                              "RelativeRunningOptimizationCostTerm_0", "RelativeRunningOptimizationCostTerm_1",
                                              "OptimizationConvergenceTime"])
    
    agg_df = agg_df.groupby(by=["CaseFamily", "CaseName"])[[col for col in agg_df.columns if col not in ["CaseFamily", "CaseName", "WindSeed"]]].agg(["min", "max", "mean"])
    # agg_df.to_csv(results_path)
    return agg_df

def plot_wind_field_ts(data_df, save_path, filter_func=None):
    fig_wind, ax_wind = plt.subplots(2, 1, sharex=True)
    # fig_wind.set_size_inches(10, 5)

    for seed in sorted(pd.unique(data_df["WindSeed"])):
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        ax_wind[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"], label=f"Seed {seed}")
        if filter_func is not None:
            ax_wind[0].plot(seed_df["Time"], filter_func(x=seed_df["FreestreamWindDir"]), label=f"Seed {seed}")
        ax_wind[0].set(title='Wind Direction [$^\\circ$]')
        ax_wind[1].plot(seed_df["Time"], seed_df["FreestreamWindMag"], label=f"Seed {seed}")
        ax_wind[1].set(title='Wind Speed [m/s]', xlabel='Time [s]', xlim=(0, seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]))
        ax_wind[0].legend()
    # fig_wind.tight_layout()
    fig_wind.savefig(os.path.join(save_path, "wind_mag_dir_ts.png"))
    # fig_wind.show()

    return fig_wind, ax_wind

def plot_opt_var_ts(data_df, yaw_offset_bounds, save_path):
    colors = sns.color_palette(palette='Paired')
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" == col[:len("TurbineYawAngle_") and not pd.isna(data_df[col]).any()]], key=lambda s: int(s.split("_")[-1]))
    yaw_angle_change_cols = sorted([col for col in data_df.columns if "TurbineYawAngleChange_" in col], key=lambda s: int(s.split("_")[-1]))

    fig_opt_vars, ax_opt_vars = plt.subplots(2, 1, sharex=True)
    # fig_opt_vars.set_size_inches(10, 5)
    plot_seed = 0
    time_frac = 1;
    plot_turbine = int(len(yaw_angle_cols) // 2)
    for seed in sorted(pd.unique(data_df["WindSeed"])):
        if seed != plot_seed:
            continue
        seed_df = data_df.loc[data_df["WindSeed"] == seed].sort_values(by="Time")
        ax_opt_vars[0].plot(seed_df["Time"], seed_df[yaw_angle_cols[plot_turbine]])
        ax_opt_vars[0].set(title='Yaw Angles [$^\\circ$]')
        ax_opt_vars[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"] - yaw_offset_bounds[0], color=colors[seed], linestyle='dotted')
        ax_opt_vars[0].plot(seed_df["Time"], seed_df["FreestreamWindDir"] - yaw_offset_bounds[1], color=colors[seed], linestyle='dotted', label="Lower/Upper Bounds")
        ax_opt_vars[1].plot(seed_df["Time"], seed_df[yaw_angle_change_cols[plot_turbine]], color=colors[seed], linestyle='-')
        ax_opt_vars[1].set(title='Yaw Angles Change [$^\\circ$]', xlabel='Time [s]', xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) * time_frac)), ylim=(-2, 2))
    # ax_outputs[1, 0].plot(time_ts[:int(simulation_max_time // input_dict["dt"]) - 1], turbine_powers_ts)
    # ax_outputs[1, 0].set(title="Turbine Powers [MW]")
    ax_opt_vars[0].legend()
    fig_opt_vars.savefig(save_path)
    # fig_opt_vars.show()

    return fig_opt_vars, ax_opt_vars

def plot_opt_cost_ts(data_df, save_path):
    fig_opt_cost, ax_opt_cost = plt.subplots(2, 1, sharex=True)
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

def plot_yaw_offset_wind_direction(data_dfs, case_names, case_labels, lut_path, save_path, plot_turbine_ids, include_yaw=True, include_power=True, interpolate=True, scatter=True):
    """
    Plot yaw offset vs wind-direction based on the lookup-table (line), 
    and scatter plots of MPC stochastic_interval with n_wind_preview_samples=1 (assuming mean value),
    MPC stochastic_interval with n_wind_preview_samples=3 (considering variation),
    and LUT simulation for each turbine
    """
    sns.set(font_scale=2)
    colors = sns.color_palette("Set2")

    fig = plt.figure()
    ax = []
    
    if include_yaw:
        for col_idx, turbine_idx in enumerate(plot_turbine_ids):
            subplot_idx = col_idx
            if col_idx == 0:
                ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1))
            else:
                ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1, sharex=ax[0], sharey=ax[0]))

            for case_name, case_label, color in zip(case_names, case_labels, cycle(colors)):
            # for case_name, case_label in zip(case_names, case_labels):
                case_df = data_dfs.loc[(data_dfs.index.get_level_values("CaseFamily") == "yaw_offset_study") & (data_dfs.index.get_level_values("CaseName") == case_name), :]
                # turbine_wind_direction_cols = sorted([col for col in case_df.columns if "TurbineWindDir_" in col])
                yaw_angle_cols = sorted([col for col in case_df.columns if "TurbineYawAngle_" == col[:len("TurbineYawAngle_")] and not pd.isna(case_df[col]).any()], key=lambda s: int(s.split("_")[-1]))

                # turbine_wind_dirs = case_df[turbine_wind_direction_cols[turbine_idx]].sort_values(by="Time")
                freestream_wind_dirs = case_df["FreestreamWindDir"]
                yaw_offsets = freestream_wind_dirs - case_df[yaw_angle_cols[turbine_idx]]

                sort_idx = np.argsort(freestream_wind_dirs)
                freestream_wind_dirs = freestream_wind_dirs.iloc[sort_idx].reset_index(drop=True) 
                yaw_offsets = yaw_offsets.iloc[sort_idx].reset_index(drop=True).rename("YawOffset")

                df = pd.concat([freestream_wind_dirs, yaw_offsets], axis=1)

                if interpolate:
                    df = df.groupby("FreestreamWindDir")["YawOffset"].mean().reset_index()
                    # interp = UnivariateSpline(freestream_wind_dirs, yaw_offsets)
                    interp = interp1d(df["FreestreamWindDir"], df["YawOffset"])
                    freestream_wind_dirs = np.arange(np.ceil(df["FreestreamWindDir"].min()), np.floor(df["FreestreamWindDir"].max()), 0.1)
                    df = pd.DataFrame(data={"FreestreamWindDir": freestream_wind_dirs,
                                           "YawOffset": interp(freestream_wind_dirs)})

                if "LUT" in case_name:
                    # ax[subplot_idx].scatter(freestream_wind_dirs, yaw_offsets, label=f"{case_name} Simulation", color=colors[len(case_names)], marker=".")
                    # sns.scatterplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="YawOffset", label=f"Simulated {case_label}", color="darkorange", marker=".")
                    if scatter:
                        sns.scatterplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="YawOffset", label=f"Simulated {case_label}", marker=".")
                    else:
                        df["FreestreamWindDir"] = df["FreestreamWindDir"].round(0)
                        df = df.groupby("FreestreamWindDir").agg(lambda rows: rows.values.mean())
                        sns.lineplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="YawOffset", label=f"Simulated {case_label}")
                else:
                    # ax[subplot_idx].scatter(freestream_wind_dirs, yaw_offsets, label=f"{case_name} Simulation", color=color, marker=".")
                    if scatter:
                        sns.scatterplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="YawOffset", label=f"Simulated {case_label}", color=color, marker=".")
                    else:
                        df["FreestreamWindDir"] = df["FreestreamWindDir"].round(0)
                        df = df.groupby("FreestreamWindDir").agg(lambda rows: rows.values.mean())
                        sns.lineplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="YawOffset", label=f"Simulated {case_label}", color=color)
                
                # ax[subplot_idx].legend([], [], frameon=False)

        if "LUT" in case_labels: 
            df_lut = pd.read_csv(lut_path, index_col=0)
            df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
            df_lut = df_lut.loc[np.vstack(df_lut["yaw_angles_opt"].values).sum(axis=1) != 0, :]
            lut_yawoffsets = np.vstack(df_lut["yaw_angles_opt"].values)
            lut_winddirs = df_lut["wind_direction"].values
            df_lut = pd.DataFrame(data={"FreestreamWindDir": lut_winddirs, 
                                    **{f"YawOffset_{i}": lut_yawoffsets[:, i] for i in plot_turbine_ids}})

        for col_idx, turbine_idx in enumerate(plot_turbine_ids):
            # ax[col_idx].scatter(lut_winddirs, lut_yawoffsets[:, turbine_idx], label="LUT", color=colors[len(case_names)], marker=">")
            if "LUT" in case_labels:
                sns.scatterplot(data=df_lut, ax=ax[col_idx], x="FreestreamWindDir", y=f"YawOffset_{turbine_idx}", label=f"LUT", marker="s", color="darkorange")
            ax[col_idx].set(xlim=(245., 295.))
            
            if col_idx != len(plot_turbine_ids) - 1:
                ax[col_idx].legend([], [], frameon=False)
            else:
                sns.move_legend(ax[col_idx], "upper left", bbox_to_anchor=(1, 1), ncols=1)
                if scatter:
                    for lh in ax[col_idx].legend_.legendHandles:
                        lh.set_sizes([200])
            
            if col_idx != 0:
                ax[col_idx].set(ylabel="")
                
            ax[col_idx].set(xlabel="")
            
            # if include_power:
            #     ax[col_idx].set_xticks([])

        ax[0].set(ylabel="Yaw Offset [$^\\circ$]")
        # ax[0].legend()
        if not include_power:
            ax[int(len(plot_turbine_ids) // 2)].set(xlabel="Freestream Wind Direction [$^\\circ$]")
            
    if include_power:
        for col_idx, turbine_idx in enumerate(plot_turbine_ids):
            subplot_idx = (col_idx + len(plot_turbine_ids)) if include_yaw else col_idx
            if col_idx == 0:
                if include_yaw:
                    ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1, sharex=ax[0]))
                else:
                    ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1))
            else:
                if include_yaw:
                    ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1, sharex=ax[0], sharey=ax[len(plot_turbine_ids)]))
                else:
                    ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1, sharex=ax[0], sharey=ax[0]))

            for case_name, case_label, color in zip(case_names, case_labels, cycle(colors)):
            # for case_name, case_label in zip(case_names, case_labels):
                case_df = data_dfs.loc[(data_dfs.index.get_level_values("CaseFamily") == "yaw_offset_study") & (data_dfs.index.get_level_values("CaseName") == case_name), :]
                # turbine_wind_direction_cols = sorted([col for col in case_df.columns if "TurbineWindDir_" in col])
                turbine_power_cols = sorted([col for col in case_df.columns if "TurbinePower_" in col], key=lambda s: int(s.split("_")[-1]))

                # turbine_wind_dirs = case_df[turbine_wind_direction_cols[turbine_idx]].sort_values(by="Time")
                freestream_wind_dirs = case_df["FreestreamWindDir"]
                turbine_powers = case_df[turbine_power_cols[turbine_idx]] / 1e6

                sort_idx = np.argsort(freestream_wind_dirs)
                freestream_wind_dirs = freestream_wind_dirs.iloc[sort_idx].reset_index(drop=True)
                turbine_powers = turbine_powers.iloc[sort_idx].reset_index(drop=True).rename("TurbinePower")

                df = pd.concat([freestream_wind_dirs, turbine_powers], axis=1)
                
                if interpolate:
                    # interp = UnivariateSpline(freestream_wind_dirs, turbine_powers)
                    df = df.groupby("FreestreamWindDir")["TurbinePower"].mean().reset_index() 
                    interp = interp1d(df["FreestreamWindDir"], df["TurbinePower"])
                    freestream_wind_dirs = np.arange(np.ceil(df["FreestreamWindDir"].min()), np.floor(df["FreestreamWindDir"].max()), 0.1)
                    df = pd.DataFrame(data={"FreestreamWindDir": freestream_wind_dirs,
                                           "TurbinePower": interp(freestream_wind_dirs)})
                if "LUT" in case_name:
                    # ax[subplot_idx].scatter(freestream_wind_dirs, turbine_powers, label=f"{case_label} Simulation", color=colors[len(case_names)], marker=".")
                    if scatter:
                        sns.scatterplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="TurbinePower", label=f"Simulated {case_label}", marker=".")
                    else:
                        df["FreestreamWindDir"] = df["FreestreamWindDir"].round(0)
                        df = df.groupby("FreestreamWindDir").agg(lambda rows: rows.values.mean())
                        sns.lineplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="TurbinePower", label=f"Simulated {case_label}")
                else:
                    # ax[subplot_idx].scatter(freestream_wind_dirs, turbine_powers, label=f"{case_label} Simulation", color=color, marker=".")
                    if scatter:
                        sns.scatterplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="TurbinePower", label=f"Simulated {case_label}", color=color, marker=".")
                    else:
                        df["FreestreamWindDir"] = df["FreestreamWindDir"].round(0)
                        df = df.groupby("FreestreamWindDir").agg(lambda rows: rows.values.mean())
                        sns.lineplot(data=df, ax=ax[subplot_idx], x="FreestreamWindDir", y="TurbinePower", label=f"Simulated {case_label}", color=color)
        
                if not include_yaw and col_idx != len(plot_turbine_ids) - 1:
                    ax[subplot_idx].legend([], [], frameon=False)
                elif include_yaw:
                    ax[subplot_idx].legend([], [], frameon=False) 
                else:
                    sns.move_legend(ax[subplot_idx], "upper left", bbox_to_anchor=(1, 1), ncols=1)
                    if scatter:
                        for lh in ax[subplot_idx].legend_.legendHandles:
                            lh.set_sizes([200])
                            
                ax[subplot_idx].set(xlabel="")
                if subplot_idx != 0:
                    ax[subplot_idx].set(ylabel="")
            
        ax[len(plot_turbine_ids) if include_yaw else 0].set(ylabel="Turbine Power [MW]")    
        ax[(int(len(plot_turbine_ids) // 2) + len(plot_turbine_ids)) if include_yaw else int(len(plot_turbine_ids) // 2)].set(xlabel="Freestream Wind Direction [$^\\circ$]")

    results_dir = os.path.dirname(save_path)
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    fig.suptitle("_".join([os.path.basename(results_dir), "yawoffset_winddir_ts"]))
    
    if include_power:
        for i in range(len(plot_turbine_ids)):
            plt.setp(ax[i].get_xticklabels(), visible=False)
    plt.tight_layout()
    fig.savefig(save_path)
    # plt.close(fig)
    # fig.show()
    return fig, ax

def plot_yaw_power_ts(data_df, save_path, include_yaw=True, include_power=True, include_filtered_wind_dir=True, controller_dt=None, legend_loc="best", single_plot=False, fig=None, ax=None, case_label=None):
    #TODO only plot some turbines, not ones with overlapping yaw offsets, eg single column on farm
    colors = sns.color_palette("Paired")
    colors = [colors[1], colors[3], colors[5]]

    if not single_plot:
        fig, ax = plt.subplots(int(include_yaw + include_power), 1, sharex=True)
    
    ax = np.atleast_1d(ax)
    
    turbine_wind_direction_cols = sorted([col for col in data_df.columns if "TurbineWindDir_" in col and not pd.isna(data_df[col]).any()], key=lambda s: int(s.split("_")[-1]))
    turbine_power_cols = sorted([col for col in data_df.columns if "TurbinePower_" in col and not pd.isna(data_df[col]).any()], key=lambda s: int(s.split("_")[-1]))
    yaw_angle_cols = sorted([col for col in data_df.columns if "TurbineYawAngle_" == col[:len("TurbineYawAngle_")] and not pd.isna(data_df[col]).any()], key=lambda s: int(s.split("_")[-1]))

    plot_seed = 0
    
    for seed in sorted(pd.unique(data_df["WindSeed"])):
        if seed != plot_seed:
            continue
        seed_df = data_df.loc[data_df["WindSeed"] == seed, :].sort_values(by="Time")
        
        if include_yaw:
            ax_idx = 0
            sns.lineplot(data=seed_df, x="Time", y="FreestreamWindDir", label="Wind dir.", color="black", ax=ax[ax_idx])
            if include_filtered_wind_dir:
                sns.lineplot(data=seed_df, x="Time", y="FilteredFreestreamWindDir", label="Filtered wind dir.", color="black", linestyle="--", ax=ax[ax_idx])
            
        # Direction
        # for t, (wind_dir_col, power_col, yaw_col) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols)):
        for t, (wind_dir_col, power_col, yaw_col, color) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols, cycle(colors))):
            
            if include_yaw:
                ax_idx = 0
                if single_plot:
                    sns.lineplot(data=seed_df, x="Time", y=yaw_col, label="T{0:01d} yaw setpoint, {1}".format(t + 1, case_label), linestyle=":", ax=ax[ax_idx])
                else:
                    sns.lineplot(data=seed_df, x="Time", y=yaw_col, color=color, label="T{0:01d} yaw setpoint".format(t + 1), linestyle=":", ax=ax[ax_idx])
                ax[ax_idx].set(ylabel="")
                
                if controller_dt is not None:
                    [ax[ax_idx].axvline(x=_x, linestyle=(0, (1, 10)), linewidth=0.5) for _x in np.arange(0, seed_df["Time"].iloc[-1], controller_dt)]

            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                if t == 0:
                    if single_plot:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[power_col] / 1e6, label="T{0:01d} power, {1}".format(t + 1, case_label))
                    else:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[power_col] / 1e6, color=color, label="T{0:01d} power".format(t + 1))
                else:
                    if single_plot:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e6, 
                                        seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e6,
                                        label="T{0:01d} power, {1}".format(t + 1, case_label))
                    else:
                        ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e6, 
                                        seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e6,
                            color=color, label="T{0:01d} power".format(t + 1))
        
        if include_power:
            next_ax_idx = (1 if include_yaw else 0)
            seed_df["FarmPower"] = seed_df[turbine_power_cols].sum(axis=1) / 1e6
            if single_plot:
                sns.lineplot(data=seed_df, x="Time", y="FarmPower", label=f"Farm power, {case_label}", ax=ax[next_ax_idx])
            else:
                sns.lineplot(data=seed_df, x="Time", y="FarmPower", color="black", label="Farm power", ax=ax[next_ax_idx])
            ax[next_ax_idx].set(ylabel="")
    
    # n_cols = 1 if single_plot else 2
    n_cols = 1
    if include_yaw:
        ax_idx = 0
        ax[ax_idx].set(title="Wind Direction / Yaw Angle [$^\\circ$]", xlim=(0, int((data_df["Time"].max() + data_df["Time"].diff().iloc[1]) // 1)), ylim=(220, 320))
        ax[ax_idx].legend() 
        if legend_loc != "outer":
            ax[ax_idx].legend(ncols=n_cols, loc=legend_loc)
        else:
            sns.move_legend(ax[ax_idx], "upper left", bbox_to_anchor=(1, 1), ncols=n_cols)
        # ax[ax_idx].legend([], [], frameon=False)
        if not include_power:
            ax[ax_idx].set(xlabel="Time [s]")
    
    if include_power:
        next_ax_idx = (1 if include_yaw else 0)
        ax[next_ax_idx].set(xlabel="Time [s]", title="Turbine Powers [MW]")
        ax[next_ax_idx].legend(ncols=n_cols) 
        if legend_loc != "outer":
            ax[next_ax_idx].legend(ncols=n_cols, loc=legend_loc)
        else:
            sns.move_legend(ax[next_ax_idx], "upper left", bbox_to_anchor=(1, 1), ncols=n_cols)
        # ax[next_ax_idx].legend([], [], frameon=False)

    results_dir = os.path.dirname(save_path)
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    fig.suptitle("_".join([os.path.basename(results_dir), data_df.index.get_level_values("CaseName")[0].replace('/', '_'), "yaw_power_ts"]))
    # plt.get_current_fig_manager().full_screen_toggle()
    plt.tight_layout()
    fig.savefig(save_path)
    # 
    # fig.show()
    return fig, ax
    
def plot_parameter_sweep(agg_dfs, mpc_type, save_dir, plot_columns, merge_wind_preview_types, estimator="max"):
    mpc_df = agg_dfs.iloc[agg_dfs.index.get_level_values("CaseFamily")  == mpc_type, :]
    lut_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily").str.contains("baseline_controllers")) & (agg_dfs.index.get_level_values("CaseName") == "LUT")] 
    greedy_df = agg_dfs.iloc[(agg_dfs.index.get_level_values("CaseFamily").str.contains("baseline_controllers")) & (agg_dfs.index.get_level_values("CaseName") == "Greedy")]
                
    mpc_df = mpc_df.sort_values(by="FarmPowerMean", ascending=False).reset_index(level="CaseFamily", drop=True).reset_index(level="CaseName", drop=False)
    mpc_df["FarmPowerMean"] = mpc_df["FarmPowerMean"] / 1e6

    # plot of farm power vs n_wind_preview_samples, bar for each sampling type
    if all(c in plot_columns for c in ["n_wind_preview_samples", "wind_preview_type", "FarmPowerMean"]) and len(pd.unique(mpc_df["n_wind_preview_samples"])) > len(pd.unique(mpc_df["wind_preview_type"])):
        mpc_df = mpc_df.loc[(mpc_df["CaseName"] != "Persistent") & (mpc_df["CaseName"] != "Perfect"), :]
        unique_sir_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_rectangular", "n_wind_preview_samples"])).astype(int)
        unique_sie_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_interval_elliptical", "n_wind_preview_samples"])).astype(int)
        unique_ss_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["wind_preview_type"] == "stochastic_sample", "n_wind_preview_samples"])).astype(int)
        ax = sns.catplot(data=mpc_df, kind="bar", x="n_wind_preview_samples_index", y="FarmPowerMean", estimator=estimator, hue="wind_preview_type", errorbar=None, legend_out=False)
        ax.ax.set(ylabel="", xlabel="# Wind Preview Samples", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.04))
        ax.ax.set_xticklabels([f"{sie_val}    {sir_val}    {ss_val}" for sir_val, sie_val, ss_val in zip(unique_sir_vals, unique_sie_vals, unique_ss_vals)]) 
        n_xticks = len(pd.unique(mpc_df["n_wind_preview_samples_index"]))
        ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, color="forestgreen", label="Greedy")
        ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, color="darkorange", label="LUT")
        plt.legend(loc="lower right")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
        sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1) 
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "n_wind_preview_samples", "param_sweep_n_wind_preview_samples.png"))

        ax.ax.set_ylim((3.02, 3.04))
        handles, labels = ax.ax.get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
        labels = [l for l in labels if l not in ["Greedy", "LUT"]]
        ax.ax.legend(handles, labels)
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
        sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
        plt.tight_layout() 
        plt.savefig(os.path.join(save_dir, "n_wind_preview_samples", "param_sweep_n_wind_preview_samples_zoom.png"))

    # unique_sir_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["preview_type"] == "stochastic_interval_rectangular", "diff_type"]))
    # unique_sie_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["preview_type"] == "stochastic_interval_elliptical", "diff_type"]))
    # unique_ss_vals = np.sort(pd.unique(mpc_df.loc[mpc_df["preview_type"] == "stochastic_sample", "diff_type"]))
    # 
    if merge_wind_preview_types:
        mpc_df = mpc_df.groupby([c for c in ["diff_type", "diff_direction", "diff_steps", "nu", "decay_type", "max_std_dev", "n_wind_preview_samples", "n_wind_preview_samples_index"] if c in mpc_df.columns])["FarmPowerMean"].mean().sort_values(ascending=False).reset_index([c for c in ["diff_type", "nu", "decay_type", "max_std_dev", "n_wind_preview_samples", "n_wind_preview_samples_index"] if c in mpc_df.columns], drop=False).reset_index([c for c in ["diff_direction", "diff_steps"] if c in mpc_df.columns], drop=True)

    if all(c in plot_columns for c in ["diff_type", "wind_preview_type", "FarmPowerMean"]):
        
        if merge_wind_preview_types:
            ax = sns.catplot(data=mpc_df, kind="bar", x="diff_type", y="FarmPowerMean", estimator=estimator, errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["diff_type"]))
            ax.ax.scatter(x=np.arange(n_xticks), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", color="forestgreen", s=250, label="Greedy")
            ax.ax.scatter(x=np.arange(n_xticks), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", color="darkorange", s=250, label="LUT")
            fn = "param_sweep_diff_type_merge"
        else:
            ax = sns.catplot(data=mpc_df, kind="bar", x="diff_type", y="FarmPowerMean", estimator=estimator, hue="wind_preview_type", errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["diff_type"]))
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            ax.ax.legend(loc="lower right")
            
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
            fn = "param_sweep_diff_type"

        ax.ax.set(ylabel="", xlabel="Differentiation Method", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.05))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}.png"))
        
        ax.ax.set_ylim((3.02, 3.04))

        if not merge_wind_preview_types:
            handles, labels = ax.ax.get_legend_handles_labels()
            handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
            labels = [l for l in labels if l not in ["Greedy", "LUT"]]
            ax.ax.legend(handles, labels)
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
        
        plt.tight_layout() 
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}_zoom.png"))
    
    if all(c in plot_columns for c in ["nu", "wind_preview_type", "FarmPowerMean"]):
        if merge_wind_preview_types:
            ax = sns.catplot(data=mpc_df, kind="bar", x="nu", y="FarmPowerMean", estimator=estimator, errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["nu"]))
            ax.ax.scatter(x=np.arange(n_xticks), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")

            fn = f"param_sweep_{mpc_type}_nu_merge"
        else: 
            ax = sns.catplot(data=mpc_df, kind="bar", x="nu", y="FarmPowerMean", estimator=estimator, hue="wind_preview_type", errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["nu"]))
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            ax.ax.legend(loc="lower right")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
            fn = f"param_sweep_{mpc_type}_nu"
        
        ax.ax.set(ylabel="", xlabel="Step Size", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.05))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}.png"))

        ax.ax.set_ylim((3.02, 3.04))

        if not merge_wind_preview_types:
            handles, labels = ax.ax.get_legend_handles_labels()
            handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
            labels = [l for l in labels if l not in ["Greedy", "LUT"]]
            ax.ax.legend(handles, labels) 
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
        
        plt.tight_layout() 
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}_zoom.png"))

    if all(c in plot_columns for c in ["decay_type", "wind_preview_type", "FarmPowerMean"]):

        if merge_wind_preview_types:
            ax = sns.catplot(data=mpc_df, kind="bar", x="decay_type", y="FarmPowerMean", estimator=estimator, errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["decay_type"]))
            ax.ax.scatter(x=np.arange(n_xticks), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            fn = f"param_sweep_{mpc_type}_decay_type_merge"
        else:  
            ax = sns.catplot(data=mpc_df, kind="bar", x="decay_type", y="FarmPowerMean", estimator=estimator, hue="wind_preview_type", errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["decay_type"]))
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            ax.ax.legend(loc="lower right")   
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
            fn = f"param_sweep_{mpc_type}_decay_type"
        
        ax.ax.set(ylabel="", xlabel="Decay Type", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.05))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}.png"))
        
        ax.ax.set_ylim((3.02, 3.04))

        if not merge_wind_preview_types:
            handles, labels = ax.ax.get_legend_handles_labels()
            handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
            labels = [l for l in labels if l not in ["Greedy", "LUT"]]
            ax.ax.legend(handles, labels)     
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
        
        plt.tight_layout()  
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}_zoom.png"))

    # if "max_std_dev" in plot_columns:
    if all(c in plot_columns for c in ["max_std_dev", "wind_preview_type", "FarmPowerMean"]):

        if merge_wind_preview_types:
            ax = sns.catplot(data=mpc_df, kind="bar", x="max_std_dev", y="FarmPowerMean", estimator=estimator, errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["max_std_dev"]))
            ax.ax.scatter(x=np.arange(n_xticks), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            fn = f"param_sweep_{mpc_type}_max_std_dev_merge"
        else:  
            ax = sns.catplot(data=mpc_df, kind="bar", x="max_std_dev", y="FarmPowerMean", estimator=estimator, hue="wind_preview_type", errorbar=None, legend_out=False)
            n_xticks = len(pd.unique(mpc_df["max_std_dev"]))
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
            ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
            ax.ax.legend(loc="lower right")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
            fn = f"param_sweep_{mpc_type}_max_std_dev"
        
        ax.ax.set(ylabel="", xlabel="Maximum Standard Deviation", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.05))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}.png"))
            
        ax.ax.set_ylim((3.02, 3.04))

        if not merge_wind_preview_types:
            handles, labels = ax.ax.get_legend_handles_labels()
            handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
            labels = [l for l in labels if l not in ["Greedy", "LUT"]]
            ax.ax.legend(handles, labels)     
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
            ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
            sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)

        plt.tight_layout() 
        plt.savefig(os.path.join(save_dir, "gradient_type", f"{fn}_zoom.png"))

    if all(c in plot_columns for c in ["diff_steps", "diff_direction", "wind_preview_type", "FarmPowerMean", "diff_type", "nu"]):
        # plot of direct vs. chain fd/cd, with size of scatter = farm power, hue = nu
        ax = sns.catplot(data=mpc_df.loc[mpc_df["wind_preview_type"] != "stochastic_sample"].sort_values(by=["diff_steps", "diff_direction"]), 
                        kind="bar", x="diff_type", y="FarmPowerMean", hue="nu", estimator=estimator, legend_out=False, errorbar=None)
        ax.ax.set(ylabel="", xlabel="Derivative Type", title="Farm Power [MW]")
        ax.ax.set_ylim((2.65, 3.05))
        ax.ax.set_xticklabels(["Chain \nCentral Diff.", "Chain \nForward Diff.", "Direct \nCentral Diff.", "Direct \nForward Diff."])
        n_xticks = len(pd.unique(mpc_df.loc[mpc_df["preview_type"] != "stochastic_sample"]["diff_type"]))
        ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(greedy_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="^", s=250, label="Greedy", color="forestgreen")
        ax.ax.scatter(x=np.arange(n_xticks) - ax.ax.patches[0].get_width(), y=[(lut_df["FarmPowerMean"] / 1e6).iloc[0]] * n_xticks, marker="s", s=250, label="LUT", color="darkorange")
        ax.ax.legend(loc="lower right")
        ax.ax.get_legend().get_texts()[0].set_text("0.001 step size")
        ax.ax.get_legend().get_texts()[1].set_text("0.01 step size")
        sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)  
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gradient_type", f"param_sweep_{mpc_type}_drvt.png"))
            
        ax.ax.set_ylim((3.02, 3.04))
        handles, labels = ax.ax.get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if l not in ["Greedy", "LUT"]]
        labels = [l for l in labels if l not in ["Greedy", "LUT"]]
        ax.ax.legend(handles, labels)
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_rectangular")].set_text("Stochastic Rectangular Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_interval_elliptical")].set_text("Stochastic Elliptical Interval")
        ax.ax.get_legend().get_texts()[[s._text for s in ax.ax.get_legend().get_texts()].index("stochastic_sample")].set_text("Stochastic Sample")
        sns.move_legend(ax.ax, "upper left", bbox_to_anchor=(1, 1), ncols=1)
        plt.tight_layout()  
        plt.savefig(os.path.join(save_dir, "gradient_type", f"param_sweep_{mpc_type}_drvt_zoom.png"))


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

        cmap = colormaps[sequential_colormaps[case_group_idx]]
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
    sns.set(font_scale=2)

    fig, ax = plt.subplots(1)
    baseline_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily").str.contains("baseline_controllers"), :].copy().reset_index(level="CaseName")
    baseline_df[("FarmPowerMean", "mean")] = baseline_df[("FarmPowerMean", "mean")] / 1e6
    # baseline_df[("FarmPowerMean", "min")] = baseline_df[("FarmPowerMean", "min")] / 1e6
    # baseline_df[("FarmPowerMean", "max")] = baseline_df[("FarmPowerMean", "max")] / 1e6

    sub_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily") == "cost_func_tuning", :].copy()
    sub_df = sub_df.reset_index(level="CaseName")
    sub_df.loc[:, "CaseName"] = [float(x[-1]) for x in sub_df["CaseName"].str.split("_")]
    sub_df[("FarmPowerMean", "mean")] = sub_df[("FarmPowerMean", "mean")] / 1e6
    # sub_df[("FarmPowerMean", "min")] = sub_df[("FarmPowerMean", "min")] / 1e6
    # sub_df[("FarmPowerMean", "max")] = sub_df[("FarmPowerMean", "max")] / 1e6

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    ax = sns.scatterplot(data=sub_df, x=("YawAngleChangeAbsMean", "mean"), y=("FarmPowerMean", "mean"),
                    size="CaseName", #size_order=reversed(sub_df["CaseName"].to_numpy()),
                    ax=ax)
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    ax.legend([], [], frameon=False)
    ax.set(xlabel="Mean Absolute Yaw Angle Change [$^\\circ$]", ylabel="Mean Farm Power [MW]")

    for (idx, row), m, c in zip(baseline_df.iterrows(), ["^", "s"], ["forestgreen", "darkorange"]):
        ax.scatter(x=[row[("YawAngleChangeAbsMean", "mean")]], 
                   y=[row[("FarmPowerMean", "mean")]], 
                   label=row["CaseName"].iloc[0], marker=m, color=c,
                   s=360)
                #    s=np.max(ax.collections[0].get_sizes()))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[-2:], l[-2:])
    ax.set(xlim=(-0.01, ax.get_xlim()[-1]))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cost_function_pareto_curve.png"))
    plt.close(fig)

def plot_horizon_length(data_summary_df, save_dir):
    """
    plot mean farm level power vs mean relative sum of absolute yaw changes for different values of breakdown probability
    where marker size=convergence time, marker color=horizon length, marker type=dt
    """
    sns.set(font_scale=3.)
    greedy_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "baseline_controllers") & 
                                      ((data_summary_df.index.get_level_values("CaseName").str.contains("Greedy"))), :].copy()
    lut_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "baseline_controllers") &
                                       (data_summary_df.index.get_level_values("CaseName").str.contains("LUT")), :].copy()
    # baseline_df.reset_index(level="CaseName", inplace=True)
    greedy_df[("FarmPowerMean", "mean")] = greedy_df[("FarmPowerMean", "mean")] / 1e6
    lut_df[("FarmPowerMean", "mean")] = lut_df[("FarmPowerMean", "mean")] / 1e6

    greedy_df = greedy_df.sort_values(by="CaseName")
    lut_df = lut_df.sort_values(by="CaseName")

    sub_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "horizon_length"), :].copy()
    sub_df["n_horizon"] = sub_df["n_horizon"].astype(int)
    sub_df["dt"] = sub_df["dt"].astype(int)
    sub_df = sub_df.reset_index(level="CaseName")
    sub_df.loc[:, "CaseName"] = [float(x[-1]) for x in sub_df["CaseName"].str.split("_")]
    sub_df[("FarmPowerMean", "mean")] = sub_df[("FarmPowerMean", "mean")] / 1e6
    sub_df = sub_df.sort_values(by="CaseName")
    sub_df = sub_df.droplevel(1, axis=1)
    greedy_df = greedy_df.droplevel(1, axis=1)
    lut_df = lut_df.droplevel(1, axis=1)
    # sub_df["CaseName"] = [case_studies["breakdown_robustness"]["case_names"]["vals"][int(solver_type.split("_")[-1])] for solver_type in sub_df["SolverType"]]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    fig, ax = plt.subplots(1)
    # sns.scatterplot(data=greedy_df, x="YawAngleChangeAbsMean", y="FarmPowerMean", ax=ax, marker="^")
    # sns.scatterplot(data=lut_df, x="YawAngleChangeAbsMean", y="FarmPowerMean", ax=ax, marker="s")

    # ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    # ax.collections[1].set_sizes(ax.collections[1].get_sizes() * 5)

    for (idx, row), m, c in zip(lut_df.iterrows(), ["s"], ["darkorange"]):
        ax.scatter(x=[row["YawAngleChangeAbsMean"]], 
                   y=[row["FarmPowerMean"]], 
                   label=row.name[1], marker=m, color=c)

    sns.scatterplot(data=sub_df, x="YawAngleChangeAbsMean", y="FarmPowerMean", 
                     hue="n_horizon", style="dt", ax=ax)
                    # size_order=reversed(sub_df["CaseName"]), ax=ax)
    # ax.collections[1].set_sizes(ax.collections[1].get_sizes() * 9)
    # marker_scale = 360 / ax.collections[1].get_sizes()[0]
    ax.collections[1].set_sizes([360])

    ax.set(xlabel="Mean Absolute Yaw Angle Change [$^\\circ$]", ylabel="Mean Farm Power [MW]")
    
    # ax.legend([], [], frameon=False)
    # h, l = ax.get_legend_handles_labels()
    # ax.legend(h[-2:], l[-2:])
    # ax.set(xlim=(-0.01, ax.get_xlim()[-1]))
    

                #    s=np.mean(ax.collections[0].get_sizes()), marker=m)
    h, l = ax.get_legend_handles_labels()
    # lut_idx = [i for i in range(len(l)) if "LookupBasedWakeSteeringController" in l[i]][-1]
    # ax.collections[1].set_sizes(ax.collections[1].get_sizes() * 5)
    # ax.legend([h[lut_idx]], ["LUT"])
     
    first_legend = ax.legend(handles=h[1:], markerscale=3.0, ncols=2, loc="upper left")
    # ax.legend_.texts[1].set_text("$N_p$")
    # ax.legend_.texts[2 + len(pd.unique(sub_df["dt"]))].set_text("$\\Delta t_c^{\\text{MPC}}$")
    first_legend.texts[0].set_text("$N_p$")
    first_legend.texts[1 + len(pd.unique(sub_df["dt"]))].set_text("$\\Delta t_c^{\\text{MPC}}$")
    ax.add_artist(first_legend)
    # second_legend = mlines.Line2D([], [], color="darkorange", marker="s", linestyle=None, label="LUT")
    second_legend = ax.legend(handles=[h[0]], markerscale=3.0, loc="lower right")
    ax.add_artist(second_legend)
    # ax.legend_.texts[1].set_text("$N_p$")
    # ax.legend_.texts[2 + len(pd.unique(sub_df["dt"]))].set_text("$\\Delta t_c^{\\text{MPC}}$")
    # ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 9)
    ax.collections[0].set_sizes([360])
    # ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    
    # ax.legend_.texts[0].set_text("50%")
    # ax.legend_.texts[1].set_text("20%")
    # ax.legend_.texts[2].set_text("5%")
    # ax.legend_.texts[3].set_text("2.5%")
    # ax.legend_.texts[4].set_text("0%")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "horizon_length.png"))
    plt.close(fig)

def plot_breakdown_robustness(data_summary_df, save_dir):
    # TODO could also make countplot and plot all time-step data points for different values of probability
    
    """
    plot mean farm level power vs mean relative sum of absolute yaw changes for different values of breakdown probability
    """
    
    # baseline_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily") == "baseline_controllers", :].copy().reset_index(level="CaseName", inplace=False)
    # baseline_df[("RelativeFarmPowerMean", "mean")] = baseline_df[("RelativeFarmPowerMean", "mean")] / 1e6
    # baseline_df[("RelativeFarmPowerMean", "min")] = baseline_df[("RelativeFarmPowerMean", "min")] / 1e6
    # baseline_df[("RelativeFarmPowerMean", "max")] = baseline_df[("RelativeFarmPowerMean", "max")] / 1e6

    greedy_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "breakdown_robustness") & 
                                      ((data_summary_df.index.get_level_values("CaseName").str.contains("GreedyController"))), :].copy()
    lut_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "breakdown_robustness") &
                                       (data_summary_df.index.get_level_values("CaseName").str.contains("LookupBasedWakeSteeringController")), :].copy()
    # baseline_df.reset_index(level="CaseName", inplace=True)
    greedy_df[("RelativeFarmPowerMean", "mean")] = greedy_df[("RelativeFarmPowerMean", "mean")] / 1e6
    # greedy_df[("RelativeFarmPowerMean", "min")] = greedy_df[("RelativeFarmPowerMean", "min")] / 1e6
    # greedy_df[("RelativeFarmPowerMean", "max")] = greedy_df[("RelativeFarmPowerMean", "max")] / 1e6
    lut_df[("RelativeFarmPowerMean", "mean")] = lut_df[("RelativeFarmPowerMean", "mean")] / 1e6
    # lut_df[("RelativeFarmPowerMean", "min")] = lut_df[("RelativeFarmPowerMean", "min")] / 1e6
    # lut_df[("RelativeFarmPowerMean", "max")] = lut_df[("RelativeFarmPowerMean", "max")] / 1e6

    greedy_df = greedy_df.sort_values(by="CaseName")
    lut_df = lut_df.sort_values(by="CaseName")

    sub_df = data_summary_df.loc[(data_summary_df.index.get_level_values("CaseFamily") == "breakdown_robustness") & 
                                 (data_summary_df.index.get_level_values("CaseName").str.contains("MPC")), :].copy()
    sub_df = sub_df.reset_index(level="CaseName")
    sub_df.loc[:, "CaseName"] = [float(x[-1]) for x in sub_df["CaseName"].str.split("_")]
    sub_df[("RelativeFarmPowerMean", "mean")] = sub_df[("RelativeFarmPowerMean", "mean")] / 1e6
    # sub_df[("RelativeFarmPowerMean", "min")] = sub_df[("RelativeFarmPowerMean", "min")] / 1e6
    # sub_df[("RelativeFarmPowerMean", "max")] = sub_df[("RelativeFarmPowerMean", "max")] / 1e6
    sub_df = sub_df.sort_values(by="CaseName")
    # sub_df["CaseName"] = [case_studies["breakdown_robustness"]["case_names"]["vals"][int(solver_type.split("_")[-1])] for solver_type in sub_df["SolverType"]]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    fig, ax = plt.subplots(1)
    sns.scatterplot(data=sub_df, x=("RelativeYawAngleChangeAbsMean", "mean"), y=("RelativeFarmPowerMean", "mean"), size="CaseName", ax=ax)
                    # size_order=reversed(sub_df["CaseName"]), ax=ax)
    
    ax.set(xlabel="Mean Absolute Yaw Angle Change / No. Active Turbines [$^\\circ$]", ylabel="Mean Farm Power / No. Active Turbines [MW]")

    sns.scatterplot(data=greedy_df, x=("RelativeYawAngleChangeAbsMean", "mean"), y=("RelativeFarmPowerMean", "mean"), size="CaseName", ax=ax, marker="^", color="forestgreen")
    sns.scatterplot(data=lut_df, x=("RelativeYawAngleChangeAbsMean", "mean"), y=("RelativeFarmPowerMean", "mean"), size="CaseName", ax=ax, marker="s", color="darkorange")

    # for (idx, row), m in zip(baseline_df.iterrows(), ["^", "s"]):
    #     ax.scatter(x=[row[("RelativeYawAngleChangeAbsMean", "mean")]], 
    #                y=[row[("RelativeFarmPowerMean", "mean")]], 
    #                label=row["CaseName"].iloc[0], s=np.mean(ax.collections[0].get_sizes()), marker=m)
    h, l = ax.get_legend_handles_labels()
    greedy_idx = [i for i in range(len(l)) if "GreedyController" in l[i]][-1]
    lut_idx = [i for i in range(len(l)) if "LookupBasedWakeSteeringController" in l[i]][-1]
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    ax.legend([h[greedy_idx], h[lut_idx]], ["Greedy", "LUT"]) 
    
    # ax.legend()
    # ax.legend_.set_title("Chance of Breakdown") d
    # ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    
    # ax.legend_.texts[0].set_text("50%")
    # ax.legend_.texts[1].set_text("20%")
    # ax.legend_.texts[2].set_text("5%")
    # ax.legend_.texts[3].set_text("2.5%")
    # ax.legend_.texts[4].set_text("0%")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "breakdown_robustness.png"))
    plt.close(fig)

def plot_power_increase_vs_prediction_time(plot_df, save_dir):
    """
    Plots percentage power increase compared to persistence and perfect forecasts
    against prediction time for different forecasters.
    """
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=("prediction_timedelta", ""), y=("power_ratio", ""), data=plot_df, marker="o", ax=ax)
    
    ax.set(title="Percentage Power Increase vs. Prediction Time for Different Forecasters",
           xlabel="Prediction Time (s)", ylabel="% Power Increase")
    ax.legend(title="Forecaster")
    ax.grid(True)
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "power_increase_vs_prediction_time.png"))

    plt.show()  

def plot_true_vs_predicted_wind_speed(data_df, save_dir):
    """
    Plots true vs predicted wind speed with predicted standard deviation for Kalman Filter and ML Forecasters.
    """
    
    # Select relevant columns (only turbines 4 and 6)
    true_speed_cols = ["TrueTurbineWindSpeedVert_4", "TrueTurbineWindSpeedVert_6"]
    predicted_speed_cols = ["PredictedTurbineWindSpeedVert_4", "PredictedTurbineWindSpeedVert_6"]
    stddev_cols = ["StddevTurbineWindSpeedVert_4", "StddevTurbineWindSpeedVert_6"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for true_col, pred_col, std_col in zip(true_speed_cols, predicted_speed_cols, stddev_cols):
        sns.lineplot(x=data_df.index, y=data_df[true_col], label=f"True {true_col}", ax=ax)
        sns.lineplot(x=data_df.index, y=data_df[pred_col], label=f"Predicted {pred_col}", ax=ax)
        ax.fill_between(data_df.index, 
                        data_df[pred_col] - data_df[std_col], 
                        data_df[pred_col] + data_df[std_col], 
                        alpha=0.2)
    
    ax.set(title="True vs Predicted Wind Speed for Kalman Filter & ML Forecasters (Turbines 4 & 6)",
           xlabel="Time Step", ylabel="Wind Speed (m/s)")
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "true_vs_predicted_wind_speed.png"))

    plt.show()
    print(f'Breakpoint test!')

def plot_yaw_angles_and_power(data_df, save_dir):
    """
    Plots yaw angles and power for Persistence, Perfect, and other forecasters at best prediction times.
    """
    yaw_angle_cols = ["FarmYawAngleChangeAbsSum", "RelativeFarmYawAngleChangeAbsSum"]
    power_cols = ["FarmPower", "RelativeFarmPower"]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot yaw angles
    for col in yaw_angle_cols:
        sns.lineplot(x=data_df["PredictedTime"], y=data_df[col], label=col, ax=ax1)
    
    ax1.set_xlabel("Prediction Time (s)")
    ax1.set_ylabel("Yaw Angle Change (°)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot power
    ax2 = ax1.twinx()
    for col in power_cols:
        sns.lineplot(x=data_df["PredictedTime"], y=data_df[col], label=col, ax=ax2, linestyle="dashed")
    
    ax2.set_ylabel("Power (MW)")
    ax2.legend()
    ax1.set_title("Yaw Angles and Power for Persistence, Perfect vs. Other Forecasters")
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "yaw_angles_and_power.png"))

    plt.show()