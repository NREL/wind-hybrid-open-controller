import os
import whoc
import pandas as pd
from collections import defaultdict
# from collections import namedtuple

def compare_simulations(results_dfs):
    result_summary_dict = defaultdict(list)

    for case_name, results_df in results_dfs.items():

        # res = ResultsSummary(YawAngleChangeAbsSum=results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum(),
        #                      FarmPowerSum=results_df["FarmPower"].sum(),
        #                      TotalOptimizationCostSum=results_df["TotalOptimizationCost"].sum(),
        #                      ConvergenceTimeSum=results_df["ConvergenceTime"].sum())
        result_summary_dict["SolverType"].append(case_name)
        result_summary_dict["YawAngleChangeAbsSum"].append(results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum())
        result_summary_dict["YawAngleChangeAbsMean"].append(results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().mean())
        result_summary_dict["FarmPowerSum"].append(results_df["FarmPower"].sum())
        result_summary_dict["FarmPowerMean"].append(results_df["FarmPower"].mean())
        result_summary_dict["TotalRunningOptimizationCostSum"].append(results_df["TotalRunningOptimizationCost"].sum())
        result_summary_dict["TotalRunningOptimizationCostMean"].append(results_df["TotalRunningOptimizationCost"].mean())
        result_summary_dict["OptimizationConvergenceTimeMean"].append(results_df["OptimizationConvergenceTime"].mean())
        result_summary_dict["OptimizationConvergenceTimeSum"].append(results_df["OptimizationConvergenceTime"].sum())
    
    result_summary_df = pd.DataFrame(result_summary_dict)
    return result_summary_df


if __name__ == "__main__":
    dirnames = ["pyopt_sequential_perfect", "pyopt_perfect", "floris_perfect"]
    
    compare_simulations(dirnames)