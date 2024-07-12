import pandas as pd
import os
import yaml
import re
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import defaultdict
from itertools import cycle
import yaml
import seaborn as sns
sns.set_theme(style="darkgrid", rc={'figure.figsize':(4,4)})

import floris.layout_visualization as layoutviz
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane

from scipy.interpolate import LinearNDInterpolator

from whoc import __file__ as whoc_file

def plot_wind_farm(floris_input_files, lut_paths, save_dir):

    # fig, axarr = plt.subplots(int(len(floris_input_files)**0.5), int(len(floris_input_files)**0.5), figsize=(16, 10))
    # axarr = axarr.flatten()

    for floris_input_file, lut_path in zip(floris_input_files, lut_paths):
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

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
        turbine_names = [f"T{i}" for i in range(fmodel.n_turbines)]
        # layoutviz.plot_turbine_points(fmodel, ax=ax)
        # layoutviz.plot_turbine_labels(
        #     fmodel, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={"facecolor": "r"}
        # )

        # Plot 2: Show turbine rotors on flow
        horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
        visualize_cut_plane(horizontal_plane, ax=ax, min_speed=MIN_WS, max_speed=MAX_WS)
        layoutviz.plot_turbine_rotors(fmodel, ax=ax, yaw_angles=lut_angles)

        # if a > 1:
        ax.set(xlabel="$x$ [m]")
        
        # if a == 0 or a == 2:
        ax.set(ylabel="$y$ [m]")
    
        fig.savefig(os.path.join(save_dir, f"wind_farm_plot_{fmodel.n_turbines}.png"))

def get_results_data(results_dirs):
    # from whoc.wind_field.WindField import first_ord_filter
    results_dfs = {}
    for results_dir in results_dirs:
        case_family = os.path.split(os.path.basename(results_dir))[-1]
        for f, fn in enumerate([fn for fn in os.listdir(results_dir) if ".csv" in fn]):
            seed = int(re.findall(r"(?<=seed\_)(\d*)", fn)[0])
            case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0]

            df = pd.read_csv(os.path.join(results_dir, fn), index_col=0)

            # df["FilteredFreestreamWindDir"] = first_ord_filter(df["FreestreamWindDir"])
            # df.to_csv(os.path.join(results_dir, f"time_series_results_case_{r}.csv"))
            case_tag = f"{case_family}_{case_name}"
            if case_tag not in results_dfs:
                results_dfs[case_tag] = df
            else:
                results_dfs[case_tag] = pd.concat([results_dfs[case_tag], df])

        # results_dfs[case_tag].to_csv(os.path.join(results_dir, fn))
    return results_dfs

def process_simulations(results_dirs, case_studies, save_dir):
    results_dfs = get_results_data(results_dirs) # TODO change save name of compare_results_df
    compare_results_df = compare_simulations(results_dfs, save_dir)
    compare_results_df.sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True)[("RelativeTotalRunningOptimizationCostMean", "mean")]
    compare_results_df.sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[("YawAngleChangeAbsMean", "mean")]
    compare_results_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[("FarmPowerMean", "mean")]

    compare_results_df[("FarmPowerMean", "mean")]
    mpc_df = compare_results_df.iloc[compare_results_df.index.get_level_values("CaseFamily") == "slsqp_solver_sweep"]  
    lut_df = compare_results_df.iloc[compare_results_df.index.get_level_values("CaseName") == "LUT"] 
    greedy_df = compare_results_df.iloc[compare_results_df.index.get_level_values("CaseName") == "Greedy"]

    better_than_lut_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > lut_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < lut_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
    # print(mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]) & (mpc_df[("YawAngleChangeAbsMean", "mean")] < greedy_df[("YawAngleChangeAbsMean", "mean")].iloc[0]), ("RelativeTotalRunningOptimizationCostMean", "mean")].sort_values(ascending=True))
    better_than_greedy_df = mpc_df.loc[(mpc_df[("FarmPowerMean", "mean")] > greedy_df[("FarmPowerMean", "mean")].iloc[0]), [("RelativeTotalRunningOptimizationCostMean", "mean"), ("YawAngleChangeAbsMean", "mean"), ("FarmPowerMean", "mean")]].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True).reset_index(level="CaseFamily", drop=True)
    better_than_greedy_df = better_than_greedy_df.loc[better_than_greedy_df.index.isin(better_than_lut_df.index)]
    best_idx_sum = np.inf
    best_case = None
    for best_lut_idx, best_lut_case_name in enumerate(better_than_lut_df.index):
        best_greedy_idx = np.where(better_than_greedy_df.index == best_lut_case_name)[0][0]
        if best_lut_idx + best_greedy_idx < best_idx_sum:
            best_case = best_lut_case_name
    better_than_lut_df.iloc[0]._name
    # a = compare_results_df.loc[compare_results_df[("YawAngleChangeAbsMean", "mean")] > 0, :].sort_values(by=("FarmPowerMean", "mean"), ascending=False)[("FarmPowerMean", "mean")] * 1e-7
    # b = compare_results_df.loc[compare_results_df[("YawAngleChangeAbsMean", "mean")] > 0, :].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)[("YawAngleChangeAbsMean", "mean")]
    
    # mask, idx = compare_results_df.index.get_loc_level("slsqp_solver_sweep", level="CaseFamily")
    if 0:
        a = compare_results_df.loc[pd.IndexSlice["slsqp_solver_sweep_small", :], :].sort_values(by=("FarmPowerMean", "mean"), ascending=False)
        b = compare_results_df.loc[pd.IndexSlice["slsqp_solver_sweep_small", :], :].sort_values(by=("YawAngleChangeAbsMean", "mean"), ascending=True)
        a = a.loc[a[("YawAngleChangeAbsMean", "mean")] > 0, :][("FarmPowerMean", "mean")] * 1e-7
        b = b.loc[b[("YawAngleChangeAbsMean", "mean")] > 0, :][("YawAngleChangeAbsMean", "mean")]
        
        # best case prioritizing high farm_power
        farm_power_case_names = [ind[1] for ind in a.index]
        yaw_change_case_names = [ind[1] for ind in b.index]
        min_index = len(yaw_change_case_names)
        best_case = None
        for farm_power_idx, (case_family, case_name) in enumerate(a.index.values):
            print(f"farm_power_idx = {farm_power_idx}")
            yaw_change_idx = yaw_change_case_names.index(case_name)
            print(f"yaw_change_idx = {yaw_change_idx}")
            if farm_power_idx + yaw_change_idx < min_index:
                print(f"min_index, farm_power_idx, yaw_change_idx = {min_index, farm_power_idx, yaw_change_idx}")
                min_index = farm_power_idx + yaw_change_idx
                best_case = case_name # 'alpha_0.995_controller_class_MPC_n_wind_preview_samples_50_nu_1.0_solver_slsqp'
        
        farm_power_case_names.index(best_case)
        yaw_change_case_names.index(best_case)

    # compare_results_df.sort_values(by=("TotalRunningOptimizationCostMean", "mean"), ascending=True).groupby(level=0)[("TotalRunningOptimizationCostMean", "mean")]
    compare_results_df[("TotalRunningOptimizationCostMean", "mean")].sort_values(ascending=True)

    (-(compare_results_df[("FarmPowerMean", "mean")] * 1e-8) + (compare_results_df[("YawAngleChangeAbsMean", "mean")])).sort_values(ascending=True)
    (compare_results_df[("FarmPowerMean", "mean")].sort_values(ascending=False)).to_csv("./mpc_configs_maxpower.csv")
    (compare_results_df[("YawAngleChangeAbsMean", "mean")].sort_values(ascending=True)).to_csv("./mpc_configs_minyaw.csv")
    ((compare_results_df[("FarmPowerMean", "mean")] * 1e-7) - compare_results_df[("YawAngleChangeAbsMean", "mean")]).sort_values(ascending=False).to_csv("./mpc_configs_mincost")
    ((compare_results_df[("FarmPowerMean", "mean")] * 1e-7) / compare_results_df[("YawAngleChangeAbsMean", "mean")]).sort_values(ascending=False).to_csv("./mpc_configs_max_power_yaw_ratio.csv")

    # compare_results_df.groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeTotalRunningOptimizationCostMean", "mean"), ascending=True).head(3))[("RelativeTotalRunningOptimizationCostMean", "mean")]
    x = compare_results_df.loc[compare_results_df[("RelativeYawAngleChangeAbsMean", "mean")] > 0, :].groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeYawAngleChangeAbsMean", "mean"), ascending=True).head(10))[("RelativeYawAngleChangeAbsMean", "mean")]
    y = compare_results_df.groupby("CaseFamily", group_keys=False).apply(lambda x: x.sort_values(by=("RelativeFarmPowerMean", "mean"), ascending=False).head(10))[("RelativeFarmPowerMean", "mean")]
    
    if "breakdown_robustness" in compare_results_df.index.get_level_values("CaseFamily"):
        plot_breakdown_robustness(compare_results_df, case_studies, save_dir)

    if "scalability" in compare_results_df.index.get_level_values("CaseFamily"):
        plot_cost_function_pareto_curve(compare_results_df, case_studies, save_dir)

    # generate results table in tex
    # solver_type_df = compare_results_df.loc[compare_results_df.index.get_level_values("CaseFamily") == "solver_type", :].reset_index("CaseName")
    # solver_type_df.loc[solver_type_df.CaseName == 'SLSQP', ("RelativeYawAngleChangeAbsMean", "mean")]

    x = compare_results_df.loc[(compare_results_df.index.get_level_values("CaseFamily") != "scalability") & (compare_results_df.index.get_level_values("CaseFamily") != "breakdown_robustness"), :]
    # x = x.loc[:, x.columns.get_level_values(1) == "mean"]
    x = x.loc[:, ("RelativeTotalRunningOptimizationCostMean", "mean")]
    x = x.groupby("CaseFamily", group_keys=False).nsmallest(3)
    # Set alpha to 0.1, n_horizon to 12, solver to SLSQP, warm-start to LUT

    get_result = lambda case_family, case_name, parameter: compare_results_df.loc[(compare_results_df.index.get_level_values("CaseFamily") == case_family) & (compare_results_df.index.get_level_values("CaseName") == case_name), (parameter, "mean")].iloc[0]
    # get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean')
    # get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean')
    # get_result('solver_type', 'SLSQP', 'TotalRunningOptimizationCostMean')
    # get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTimeMean')

    if all(col in compare_results_df.index.get_level_values("CaseFamily") for col in ["baseline_controllers", "solver_type",
                    "wind_preview_type", "warm_start", 
                    "horizon_length", "breakdown_robustness",
                    "scalability", "cost_func_tuning"]):
        compare_results_latex = (
        f"\\begin{{tabular}}{{l|lllll}}\n"
        f"\\textbf{{Case Family}} & \\textbf{{Case Name}} & \\thead{{\\textbf{{Relative Mean}} \\\\ \\textbf{{Farm Power [MW]}}}}                                                                    & \\thead{{\\textbf{{Relative Mean Absolute}} \\\\ \\textbf{{Yaw Angle Change [$^\\circ$]}}}}                    & \\thead{{\\textbf{{Relative}} \\\\ \\textbf{{Mean Cost [-]}}}}                                                        & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Convergence Time [s]}}}} \\\\ \\hline \n"
        f"\\multirow{{3}}{{*}}{{\\textbf{{Solver}}}} & \\textbf{{SLSQP}}                     & ${get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean') / 1e6:.3f}$                              & ${get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$                              & ${get_result('solver_type', 'SLSQP', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                 & ${int(get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                          Sequential SLSQP                       & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('solver_type', 'Sequential SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                          Serial Refine                          & ${get_result('solver_type', 'Sequential Refine', 'RelativeFarmPowerMean') / 1e6:.3f}$                  & ${get_result('solver_type', 'Sequential Refine', 'RelativeYawAngleChangeAbsMean'):.3f}$                  & ${get_result('solver_type', 'Sequential Refine', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                     & ${int(get_result('solver_type', 'Sequential Refine', 'OptimizationConvergenceTimeMean')):d}$  \\\\ \\hline \n"
        f"\\multirow{{3}}{{*}}{{\\textbf{{Wind Preview Model}}}} & Perfect                   & ${get_result('wind_preview_type', 'Perfect', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('wind_preview_type', 'Perfect', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('wind_preview_type', 'Perfect', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('wind_preview_type', 'Perfect', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                      Persistent                 & ${get_result('wind_preview_type', 'Persistent', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('wind_preview_type', 'Persistent', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('wind_preview_type', 'Persistent', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('wind_preview_type', 'Persistent', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                      \\textbf{{Stochastic}}     & ${get_result('wind_preview_type', 'Stochastic', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('wind_preview_type', 'Stochastic', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('wind_preview_type', 'Stochastic', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('wind_preview_type', 'Stochastic', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \\hline \n"
        f"\\multirow{{3}}{{*}}{{\\textbf{{Warm-Starting Method}}}} & Greedy                  & ${get_result('warm_start', 'Greedy', 'RelativeFarmPowerMean') / 1e6:.3f}$                              & ${get_result('warm_start', 'Greedy', 'RelativeYawAngleChangeAbsMean'):.3f}$                              & ${get_result('warm_start', 'Greedy', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                 & ${int(get_result('warm_start', 'Greedy', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        LUT                      & ${get_result('warm_start', 'LUT', 'RelativeFarmPowerMean') / 1e6:.3f}$                                 & ${get_result('warm_start', 'LUT', 'RelativeYawAngleChangeAbsMean'):.3f}$                                 & ${get_result('warm_start', 'LUT', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                    & ${int(get_result('warm_start', 'LUT', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        Previous Solution        & ${get_result('warm_start', 'Previous', 'RelativeFarmPowerMean') / 1e6:.3f}$                            & ${get_result('warm_start', 'Previous', 'RelativeYawAngleChangeAbsMean'):.3f}$                            & ${get_result('warm_start', 'Previous', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                               & ${int(get_result('warm_start', 'Previous', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \\hline \n"
        f"\\multirow{{4}}{{*}}{{\\textbf{{Wind Farm Size}}}}       & $3 \\times 1$           & ${get_result('scalability', '3 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                         & ${get_result('scalability', '3 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                         & ${get_result('scalability', '3 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                            & ${int(get_result('scalability', '3 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $\\bm{{3 \\times 3}}$    & ${get_result('scalability', '9 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                         & ${get_result('scalability', '9 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                         & ${get_result('scalability', '9 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                            & ${int(get_result('scalability', '9 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $5 \\times 5$            & ${get_result('scalability', '25 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                        & ${get_result('scalability', '25 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                        & ${get_result('scalability', '25 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                           & ${int(get_result('scalability', '25 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $10 \\times 10$          & ${get_result('scalability', '100 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('scalability', '100 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('scalability', '100 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('scalability', '100 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \\hline \n"
        f"\\multirow{{5}}{{*}}{{\\textbf{{Horizon Length}}}}       & $6$                     & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('horizon_length_N', 'N_p = 6', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $8$                      & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('horizon_length_N', 'N_p = 8', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $\\bm{{10}}$             & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 10', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $12$                     & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 12', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        f"&                                                        $14$                     & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 14', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \\hline \n"
        # f"\multirow{{5}}{{*}}{{\\textbf{{Probability of Turbine Failure}}}} & $\\bm{{0\%}}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        # f"&                                                                  $1\%$          & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        # f"&                                                                  $5\%$          & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        # f"&                                                                  $10\%$         & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
        # f"&                                                                  $20\%$         & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \n"
        f"\\end{{tabular}}"
        )
        with open(os.path.join(save_dir, "comparison_time_series_results_table.tex"), "w") as fp:
            fp.write(compare_results_latex)
    

def plot_simulations(results_dirs, save_dir):
    # TODO delete all extra files in directories before rerunning simulations
    results_dfs = get_results_data(results_dirs)
    
    # plot yaw vs wind dir TODO only for three turbine cases
    case_names = ["yaw_offset_study_LUT_3turb", "yaw_offset_study_StochasticInterval_1_3turb", "yaw_offset_study_StochasticInterval_3turb"]
    case_labels = ["LUT", "Deterministic", "Stochastic"]
    plot_yaw_offset_wind_direction(results_dfs, case_names, case_labels,
                                   os.path.join(os.path.dirname(whoc_file), f"../examples/mpc_wake_steering_florisstandin/lut_{3}.csv"), 
                                   os.path.join(save_dir, f"yawoffset_winddir_ts.png"), plot_turbine_ids=[0, 1, 2], include_yaw=True, include_power=True)
    
    for r, results_dir in enumerate(results_dirs):
        input_filenames = [fn for fn in os.listdir(results_dir) if "input_config" in fn]
        # input_case_names = [re.findall(r"(?<=case_)(.*)(?=.yaml)", input_fn)[0] for input_fn in input_filenames]
        # data_filenames = sorted([fn for fn in os.listdir(results_dir) if ("time_series_results" in fn 
        #                         and re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0] in input_case_names)], 
        #                         key=lambda data_fn: input_case_names.index(re.findall(r"(?<=case_)(.*)(?=_seed)", data_fn)[0]))
        # for f, (input_fn, data_fn) in enumerate(zip(input_filenames, data_filenames)):
        # for f, data_fn in enumerate(data_filenames):
        for f, input_fn in enumerate(input_filenames):
            case_family = os.path.basename(results_dir)
            # data_case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", data_fn)[0]
            # input_fn = f"input_config_case_{data_case_name}.yaml"
            case_name = re.findall(r"(?<=input_config_case_)(.*)(?=.yaml)", input_fn)[0]
            # case_family = os.path.basename(results_dir)
            # # input_case_name = re.findall(r"(?<=case_)(.*)(?=.yaml)", input_fn)[0]
            # data_case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", data_fn)[0]
            # input_fn = f"input_config_case_{data_case_name}.yaml"
            # assert input_case_name == data_case_name
            
            if not (
                # ((case_family == "baseline_controllers") and ("time_series_results_case_Greedy_seed_0.csv" in data_fn)) or
                ((case_family == "baseline_controllers") and ("time_series_results_case_LUT_seed_0.csv" in data_fn)) or
                ((case_family == "slsqp_solver_sweep_small") and ("time_series_results_case_alpha_0.995_controller_class_MPC_n_wind_preview_samples_7_nu_0.1_solver_slsqp_wind_preview_type_stochastic_interval_seed_0.csv" in data_fn))):
                continue
            
            with open(os.path.join(results_dir, input_fn), 'r') as fp:
                input_config = yaml.safe_load(fp)

            case_name = f"{case_family}_{case_name}"
            df = results_dfs[case_name]
            # df.loc[df.CaseName == "alpha_0.995_controller_class_MPC_n_wind_preview_samples_7_nu_0.1_solver_slsqp_wind_preview_type_stochastic_interval", "FarmPower"]
            # if "Time" not in df.columns:
            #     df["Time"] = np.arange(0, 3600.0 - 60.0, 60.0)

            # fig, _ = plot_wind_field_ts(df, os.path.join(results_dir, "wind_ts.png"))
            # fig.suptitle("_".join([os.path.basename(results_dir), "wind_ts"]))

            # fig, _ = plot_opt_var_ts(df, input_config["controller"]["yaw_limits"], os.path.join(results_dir, f"opt_var_ts_{input_config['controller']['case_names'].replace('/', '_')}.png"))
            # fig.suptitle("_".join([os.path.basename(results_dir), input_config['controller']['case_names'].replace('/', '_'), "opt_var_ts"]))
            # plot_opt_var_ts(df, (-30.0, 30.0), os.path.join(results_dir, f"opt_vars_ts_{f}.png"))
            
            # fig, _ = plot_opt_cost_ts(df, os.path.join(results_dir, f"opt_costs_ts_{input_config['controller']['case_names'].replace('/', '_')}.png"))
            # fig.suptitle("_".join([os.path.basename(results_dir), input_config['controller']['case_names'].replace('/', '_'), "opt_costs_ts"]))
        
            fig, _ = plot_yaw_power_ts(df, os.path.join(results_dir, f"yaw_power_ts_{case_name}.png"), include_power=True, controller_dt=input_config["controller"]["dt"])
    
    
    # lut_df = results_dfs[f"{'baseline_controllers'}_{'LUT'}"]
    # lut_df.loc[lut_df.Time > 180.0]["FarmPower"].mean()

    # mpc_df = results_dfs[f"{'slsqp_solver_sweep_small'}_{'alpha_0.995_controller_class_MPC_n_wind_preview_samples_7_nu_0.1_solver_slsqp_wind_preview_type_stochastic_interval'}"]
    # mpc_df.loc[mpc_df.Time > 180.0]["FarmPower"].mean()
    
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

    summary_df = pd.read_csv(os.path.join(save_dir, f"comparison_time_series_results.csv"), index_col=0)
    barplot_opt_cost(summary_df, save_dir, relative=True)

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

def compare_simulations(results_dfs, save_dir):
    result_summary_dict = defaultdict(list)

    for df_name, results_df in results_dfs.items():
        # res = ResultsSummary(YawAngleChangeAbsSum=results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum(),
        #                      FarmPowerSum=results_df["FarmPower"].sum(),
        #                      TotalOptimizationCostSum=results_df["TotalOptimizationCost"].sum(),
        #                      ConvergenceTimeSum=results_df["ConvergenceTime"].sum())
        
        case_family = df_name.replace(f"_{results_df['CaseName'].iloc[0]}", "")
        case_name = results_df['CaseName'].iloc[0]
        input_fn = f"input_config_case_{case_name}.yaml"
        
        with open(os.path.join(save_dir, case_family, input_fn), 'r') as fp:
            input_config = yaml.safe_load(fp)

        if "lpf_start_time" in input_config["controller"]:
            lpf_start_time = input_config["controller"]["lpf_start_time"]
        else:
            lpf_start_time = 180.0

        for seed in pd.unique(results_df["WindSeed"]):

            seed_df = results_df.loc[(results_df["WindSeed"] == seed) & (results_df.Time >= lpf_start_time), :]
            
            yaw_angles_change_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbineYawAngleChange_" in c]))]
            turbine_offline_status_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbineOfflineStatus_" in c]))]
            turbine_power_ts = seed_df[sorted(list([c for c in results_df.columns if "TurbinePower_" in c]))]
            # TODO doesn't work for some case families
            result_summary_dict["CaseFamily"].append(case_family)
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

def plot_yaw_offset_wind_direction(data_dfs, case_names, case_labels, lut_path, save_path, plot_turbine_ids, include_yaw=True, include_power=True):
    """
    Plot yaw offset vs wind-direction based on the lookup-table (line), 
    and scatter plots of MPC stochastic_interval with n_wind_preview_samples=1 (assuming mean value),
    MPC stochastic_interval with n_wind_preview_samples=3 (considering variation),
    and LUT simulation for each turbine
    """
    colors = sns.color_palette("Paired")

    fig = plt.figure(figsize=(15.12, 7.98))
    ax = []
    
    if include_yaw:
        for col_idx, turbine_idx in enumerate(plot_turbine_ids):
            subplot_idx = col_idx
            if col_idx == 0:
                ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1))
            else:
                ax.append(plt.subplot(int(include_yaw + include_power), len(plot_turbine_ids), subplot_idx + 1, sharex=ax[0], sharey=ax[0]))

            for case_name, case_label, color in zip(case_names, case_labels, cycle(colors)):
                case_df = data_dfs[case_name]
                # turbine_wind_direction_cols = sorted([col for col in case_df.columns if "TurbineWindDir_" in col])
                yaw_angle_cols = sorted([col for col in case_df.columns if "TurbineYawAngle_" in col])

                # turbine_wind_dirs = case_df[turbine_wind_direction_cols[turbine_idx]].sort_values(by="Time")
                freestream_wind_dirs = case_df["FreestreamWindDir"]

                yaw_offsets = freestream_wind_dirs - case_df[yaw_angle_cols[turbine_idx]]
                if "LUT" in case_name:
                    ax[subplot_idx].scatter(freestream_wind_dirs, yaw_offsets, label=f"{case_name} Simulation", color=colors[len(case_names)], marker=".", s=5)
                else:
                    ax[subplot_idx].scatter(freestream_wind_dirs, yaw_offsets, label=f"{case_name} Simulation", color=color, marker=".", s=5)
        
        df_lut = pd.read_csv(lut_path, index_col=0)
        df_lut["yaw_angles_opt"] = df_lut["yaw_angles_opt"].apply(lambda s: np.array(re.findall(r"-*\d+\.\d*", s), dtype=float))
        lut_yawoffsets = np.vstack(df_lut["yaw_angles_opt"].values)
        lut_winddirs = df_lut["wind_direction"].values
        for col_idx, turbine_idx in enumerate(plot_turbine_ids):
            ax[col_idx].scatter(lut_winddirs, lut_yawoffsets[:, turbine_idx], label="LUT", color=colors[len(case_names)], marker=">", s=20)
            ax[col_idx].set(xlim=(250., 290.))
            if not include_power:
                ax[col_idx].set(xlabel="Freestream Wind Direction [$^\\circ$]")
        
        ax[0].set(ylabel="Yaw Offset [$^\\circ$]")
        ax[0].legend()

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
                case_df = data_dfs[case_name]
                # turbine_wind_direction_cols = sorted([col for col in case_df.columns if "TurbineWindDir_" in col])
                turbine_power_cols = sorted([col for col in case_df.columns if "TurbinePower_" in col])

                # turbine_wind_dirs = case_df[turbine_wind_direction_cols[turbine_idx]].sort_values(by="Time")
                freestream_wind_dirs = case_df["FreestreamWindDir"]

                turbine_powers = case_df[turbine_power_cols[turbine_idx]] / 1e6
                if "LUT" in case_name:
                    ax[subplot_idx].scatter(freestream_wind_dirs, turbine_powers, label=f"{case_label} Simulation", color=colors[len(case_names)], marker=".", s=5)
                else:
                    ax[subplot_idx].scatter(freestream_wind_dirs, turbine_powers, label=f"{case_label} Simulation", color=color, marker=".", s=5)
        
        ax[-len(plot_turbine_ids)].set(ylabel="Turbine Power [MW]")

    results_dir = os.path.dirname(save_path)
    fig.suptitle("_".join([os.path.basename(results_dir), "yawoffset_winddir_ts"]))
    fig.savefig(save_path)
    # fig.show()
    return fig, ax

def plot_yaw_power_ts(data_df, save_path, include_yaw=True, include_power=True, controller_dt=None):
    colors = sns.color_palette(palette='Paired')

    fig, ax = plt.subplots(int(include_yaw + include_power), 1, sharex=True, figsize=(15.12, 7.98))
    ax = np.atleast_1d(ax)
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
        
        if include_yaw:
            ax_idx = 0
            ax[ax_idx].plot(seed_df["Time"], seed_df["FreestreamWindDir"], label="Freestream wind dir.", color="black")
            ax[ax_idx].plot(seed_df["Time"], seed_df["FilteredFreestreamWindDir"], label="Filtered freestream wind dir.", color="black", linestyle="--")
            
        # Direction
        for t, (wind_dir_col, power_col, yaw_col, color) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols, cycle(colors))):
            
            if include_yaw:
                ax_idx = 0
                ax[ax_idx].plot(seed_df["Time"], seed_df[yaw_col], color=color, label="T{0:01d} yaw setpoint".format(t), linestyle=":")
                
                if controller_dt is not None:
                    [ax[ax_idx].axvline(x=_x, linestyle=(0, (1, 10)), linewidth=0.5) for _x in np.arange(0, seed_df["Time"].iloc[-1], controller_dt)]

            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                if t == 0:
                    ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[power_col] / 1e3, color=color, label="T{0:01d} power".format(t))
                else:
                    ax[next_ax_idx].fill_between(seed_df["Time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e3, 
                                    seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e3,
                        color=color, label="T{0:01d} power".format(t))
        
        if include_power:
            next_ax_idx = (1 if include_yaw else 0)
            ax[next_ax_idx].plot(seed_df["Time"], seed_df[turbine_power_cols].sum(axis=1) / 1e3, color="black", label="Farm power")
    
    if include_yaw:
        ax_idx = 0
        ax[ax_idx].set(title="Wind Direction / Yaw Angle [$^\\circ$]", xlim=(0, int((seed_df["Time"].max() + seed_df["Time"].diff().iloc[1]) // 1)), ylim=(245, 295))
        ax[ax_idx].legend(ncols=2, loc="lower right")
        if not include_power:
            ax[ax_idx].set(xlabel="Time [s]", title="Turbine Powers [MW]")
    
    if include_power:
        next_ax_idx = (1 if include_yaw else 0)
        ax[next_ax_idx].set(xlabel="Time [s]", title="Turbine Powers [MW]")
        ax[next_ax_idx].legend(ncols=2, loc="lower right")

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
    sub_df[("FarmPowerMean", "mean")] = sub_df[("FarmPowerMean", "mean")] / 1e6
    sub_df[("FarmPowerMean", "min")] = sub_df[("FarmPowerMean", "min")] / 1e6
    sub_df[("FarmPowerMean", "max")] = sub_df[("FarmPowerMean", "max")] / 1e6

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    ax = sns.scatterplot(data=sub_df, x=("YawAngleChangeAbsMean", "mean"), y=("FarmPowerMean", "mean"),
                    size="CaseName", #size_order=reversed(sub_df["CaseName"].to_numpy()),
                    ax=ax)
    ax.set(xlabel="Mean Absolute Yaw Angle Change [$^\\circ$]", ylabel="Mean Farm Power [MW]")
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    ax.legend([], [], frameon=False)
    fig.savefig(os.path.join(save_dir, "cost_function_pareto_curve.png"))

def plot_breakdown_robustness(data_summary_df, case_studies, save_dir):
    # TODO could also make countplot and plot all time-step data points for different values of probability
    # TODO update based on new data_summary_df format
    """
    plot mean farm level power vs mean relative sum of absolute yaw changes for different values of breakdown probability
    """
    
    sub_df = data_summary_df.loc[data_summary_df.index.get_level_values("CaseFamily") == "breakdown_robustness", :]
    sub_df.reset_index(level="CaseName", inplace=True)
    sub_df[("FarmPowerMean", "mean")] = sub_df[("FarmPowerMean", "mean")] / 1e6
    sub_df[("FarmPowerMean", "min")] = sub_df[("FarmPowerMean", "min")] / 1e6
    sub_df[("FarmPowerMean", "max")] = sub_df[("FarmPowerMean", "max")] / 1e6
    # sub_df["CaseName"] = [case_studies["breakdown_robustness"]["case_names"]["vals"][int(solver_type.split("_")[-1])] for solver_type in sub_df["SolverType"]]

    # Plot "RelativeFarmPowerMean" vs. "RelativeYawAngleChangeAbsMean" for all "SolverType" == "cost_func_tuning"
    fig, ax = plt.subplots(1, figsize=(10.29,  5.5))
    sns.scatterplot(data=sub_df, x=("YawAngleChangeAbsMean", "mean"), y=("FarmPowerMean", "mean"), size="CaseName", 
                    size_order=reversed(sub_df["CaseName"]), ax=ax)
    ax.set(xlabel="Mean Absolute Yaw Angle Change [$^\\circ$]", ylabel="Mean Farm Power [MW]")
    # ax.legend()
    ax.legend_.set_title("Chance of Breakdown")
    ax.collections[0].set_sizes(ax.collections[0].get_sizes() * 5)
    
    ax.legend_.texts[0].set_text("50%")
    ax.legend_.texts[1].set_text("20%")
    ax.legend_.texts[2].set_text("5%")
    ax.legend_.texts[3].set_text("2.5%")
    ax.legend_.texts[4].set_text("0%")

    fig.savefig(os.path.join(save_dir, "breakdown_robustness.png"))