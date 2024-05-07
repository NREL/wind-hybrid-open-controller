import pandas as pd
import os
import yaml
import re
import sys

from whoc import __file__ as whoc_file
from whoc.postprocess_case_studies import plot_wind_field_ts, plot_opt_var_ts, plot_opt_cost_ts, plot_power_ts, barplot_opt_cost, compare_simulations, plot_cost_function_pareto_curve, plot_breakdown_robustness
from whoc.case_studies.initialize_case_studies import case_studies, STORAGE_DIR, case_families

def get_results_data(results_dirs):
    results_dfs = {}
    for results_dir in results_dirs:
        for f, fn in enumerate([fn for fn in os.listdir(results_dir) if ".csv" in fn]):
            seed = int(re.findall(r"(?<=seed\_)(\d*)", fn)[0])
            case_name = re.findall(r"(?<=case_)(.*)(?=_seed)", fn)[0]

            df = pd.read_csv(os.path.join(results_dir, fn), index_col=0)

            # df["WindSeed"] = [0] * len(df.index)
            # df.to_csv(os.path.join(results_dir, f"time_series_results_case_{r}.csv"))
            case_tag = f"{os.path.split(os.path.basename(results_dir))[-1]}_{case_name}"
            if case_tag not in results_dfs:
                results_dfs[case_tag] = df
            else:
                results_dfs[case_tag] = pd.concat([results_dfs[case_tag], df])
    return results_dfs

def process_simulations(results_dirs):
    results_dfs = get_results_data(results_dirs) # TODO change save name of compare_results_df
    compare_results_df = compare_simulations(results_dfs)
    compare_results_df.sort_values(by=("FarmPowerMean", "mean"), ascending=False)[("FarmPowerMean", "mean")]
    compare_results_df.sort_values(by=("TotalRunningOptimizationCostMean", "mean"), ascending=True)[("TotalRunningOptimizationCostMean", "mean")]
    
    plot_breakdown_robustness(compare_results_df, case_studies, STORAGE_DIR)
    plot_cost_function_pareto_curve(compare_results_df, case_studies, STORAGE_DIR)

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
    compare_results_latex = (
    f"\\begin{{tabular}}{{l|lllll}}\n"
    f"\\textbf{{Case Family}} & \\textbf{{Case Name}} & \\thead{{\\textbf{{Relative Mean}} \\\\ \\textbf{{Farm Power [MW]}}}}                                                                    & \\thead{{\\textbf{{Relative Mean Absolute}} \\\\ \\textbf{{Yaw Angle Change [deg]}}}}                    & \\thead{{\\textbf{{Relative}} \\\\ \\textbf{{Mean Cost [-]}}}}                                                        & \\thead{{\\textbf{{Mean}} \\\\ \\textbf{{Convergence Time [s]}}}} \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Solver}}}} & \\textbf{{SLSQP}}                     & ${get_result('solver_type', 'SLSQP', 'RelativeFarmPowerMean') / 1e6:.3f}$                              & ${get_result('solver_type', 'SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$                              & ${get_result('solver_type', 'SLSQP', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                 & ${int(get_result('solver_type', 'SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                          Sequential SLSQP                       & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('solver_type', 'Sequential SLSQP', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('solver_type', 'Sequential SLSQP', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                          Serial Refine                          & ${get_result('solver_type', 'Sequential Refine', 'RelativeFarmPowerMean') / 1e6:.3f}$                  & ${get_result('solver_type', 'Sequential Refine', 'RelativeYawAngleChangeAbsMean'):.3f}$                  & ${get_result('solver_type', 'Sequential Refine', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                     & ${int(get_result('solver_type', 'Sequential Refine', 'OptimizationConvergenceTimeMean')):d}$  \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Wind Preview Model}}}} & Perfect                   & ${get_result('wind_preview_type', 'Perfect', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('wind_preview_type', 'Perfect', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('wind_preview_type', 'Perfect', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('wind_preview_type', 'Perfect', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                      Persistent                 & ${get_result('wind_preview_type', 'Persistent', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('wind_preview_type', 'Persistent', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('wind_preview_type', 'Persistent', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('wind_preview_type', 'Persistent', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                      \\textbf{{Stochastic}}     & ${get_result('wind_preview_type', 'Stochastic', 'RelativeFarmPowerMean') / 1e6:.3f}$                   & ${get_result('wind_preview_type', 'Stochastic', 'RelativeYawAngleChangeAbsMean'):.3f}$                   & ${get_result('wind_preview_type', 'Stochastic', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                      & ${int(get_result('wind_preview_type', 'Stochastic', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{3}}{{*}}{{\\textbf{{Warm-Starting Method}}}} & Greedy                  & ${get_result('warm_start', 'Greedy', 'RelativeFarmPowerMean') / 1e6:.3f}$                              & ${get_result('warm_start', 'Greedy', 'RelativeYawAngleChangeAbsMean'):.3f}$                              & ${get_result('warm_start', 'Greedy', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                 & ${int(get_result('warm_start', 'Greedy', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        LUT                      & ${get_result('warm_start', 'LUT', 'RelativeFarmPowerMean') / 1e6:.3f}$                                 & ${get_result('warm_start', 'LUT', 'RelativeYawAngleChangeAbsMean'):.3f}$                                 & ${get_result('warm_start', 'LUT', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                                    & ${int(get_result('warm_start', 'LUT', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        Previous Solution        & ${get_result('warm_start', 'Previous', 'RelativeFarmPowerMean') / 1e6:.3f}$                            & ${get_result('warm_start', 'Previous', 'RelativeYawAngleChangeAbsMean'):.3f}$                            & ${get_result('warm_start', 'Previous', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                               & ${int(get_result('warm_start', 'Previous', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{4}}{{*}}{{\\textbf{{Wind Farm Size}}}}       & $3 \\times 1$           & ${get_result('scalability', '3 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                         & ${get_result('scalability', '3 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                         & ${get_result('scalability', '3 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                            & ${int(get_result('scalability', '3 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $\\bm{{3 \\times 3}}$    & ${get_result('scalability', '9 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                         & ${get_result('scalability', '9 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                         & ${get_result('scalability', '9 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                            & ${int(get_result('scalability', '9 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $5 \\times 5$            & ${get_result('scalability', '25 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                        & ${get_result('scalability', '25 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                        & ${get_result('scalability', '25 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                           & ${int(get_result('scalability', '25 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $10 \\times 10$          & ${get_result('scalability', '100 Turbines', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('scalability', '100 Turbines', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('scalability', '100 Turbines', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('scalability', '100 Turbines', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    f"\multirow{{5}}{{*}}{{\\textbf{{Horizon Length}}}}       & $6$                     & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('horizon_length_N', 'N_p = 6', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('horizon_length_N', 'N_p = 6', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $8$                      & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeFarmPowerMean') / 1e6:.3f}$                       & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeYawAngleChangeAbsMean'):.3f}$                       & ${get_result('horizon_length_N', 'N_p = 8', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                          & ${int(get_result('horizon_length_N', 'N_p = 8', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $\\bm{{10}}$             & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 10', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 10', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $12$                     & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 12', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 12', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    f"&                                                        $14$                     & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeFarmPowerMean') / 1e6:.3f}$                      & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeYawAngleChangeAbsMean'):.3f}$                      & ${get_result('horizon_length_N', 'N_p = 14', 'RelativeTotalRunningOptimizationCostMean'):.3f}$                         & ${int(get_result('horizon_length_N', 'N_p = 14', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \hline \n"
    # f"\multirow{{5}}{{*}}{{\\textbf{{Probability of Turbine Failure}}}} & $\\bm{{0\%}}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '00.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    # f"&                                                                  $1\%$          & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '02.5% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    # f"&                                                                  $5\%$          & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '05.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    # f"&                                                                  $10\%$         & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '20.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \\\\ \n"
    # f"&                                                                  $20\%$         & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeFarmPowerMean') / 1e6:.3f}$ & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeYawAngleChangeAbsMean'):.3f}$ & ${get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'RelativeTotalRunningOptimizationCostMean'):.3f}$    & ${int(get_result('breakdown_robustness', '50.0% Chance of Breakdown', 'OptimizationConvergenceTimeMean')):d}$ \n"
    f"\end{{tabular}}"
    )
    with open(os.path.join(STORAGE_DIR, "comparison_time_series_results_table.tex"), "w") as fp:
        fp.write(compare_results_latex)
    

def plot_simulations(results_dirs):
    results_dfs = get_results_data(results_dirs)
    for r, results_dir in enumerate(results_dirs):
        
        for f, (input_fn, data_fn) in enumerate(zip(sorted([fn for fn in os.listdir(results_dir) if "input_config" in fn]), 
                                   sorted([fn for fn in os.listdir(results_dir) if "time_series_results" in fn]))):
            with open(os.path.join(results_dir, input_fn), 'r') as fp:
                input_config = yaml.safe_load(fp)

            df = results_dfs[f"{os.path.basename(results_dir)}_{input_config['controller']['case_names']}"]

            # if "Time" not in df.columns:
            #     df["Time"] = np.arange(0, 3600.0 - 60.0, 60.0)

            plot_wind_field_ts(df, os.path.join(results_dir, "wind_ts.png"))

            plot_opt_var_ts(df, input_config["controller"]["yaw_limits"], results_dir)
            # plot_opt_var_ts(df, (-30.0, 30.0), os.path.join(results_dir, f"opt_vars_ts_{f}.png"))
            
            plot_opt_cost_ts(df, os.path.join(results_dir, f"opt_costs_ts_{f}.png"))
        
            plot_power_ts(df, os.path.join(results_dir, f"yaw_power_ts_{f}.png"))
    
    summary_df = pd.read_csv(os.path.join(STORAGE_DIR, f"comparison_time_series_results.csv"), index_col=0)
    barplot_opt_cost(summary_df, STORAGE_DIR, relative=True)

if __name__ == "__main__":
    DEBUG = sys.argv[1].lower() == "debug"

    if len(sys.argv) > 4:
        CASE_FAMILY_IDX = [int(i) for i in sys.argv[4:]]
    else:
        CASE_FAMILY_IDX = list(range(len(case_families)))

    if DEBUG:
        N_SEEDS = 1
    else:
        N_SEEDS = 6

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": 2, "vals": [i for i in range(N_SEEDS)]}

    # run_simulations(["perfect_preview_type"], REGENERATE_WIND_FIELD)
    print([case_families[i] for i in CASE_FAMILY_IDX])

    results_dirs = [os.path.join(STORAGE_DIR, case_families[i]) for i in CASE_FAMILY_IDX]

    # compute stats over all seeds
    process_simulations(results_dirs)

    plot_simulations(results_dirs[0:2])