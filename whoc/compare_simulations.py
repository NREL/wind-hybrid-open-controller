import os
import whoc
import pandas as pd
# from collections import namedtuple

def compare_simulations(results_dirnames):
    results_dfs = {}
    result_summary_dict = {"SolverType": [], "YawAngleChangeAbsSum": [], "FarmPowerSum": [], "TotalOptimizationCostSum": [], "OptimizationConvergenceTimeSum": []}
    # ResultsSummary = namedtuple("ResultsSummary", ["YawAngleChangeAbsSum", "FarmPowerSum", "TotalOptimizationCostSum", "ConvergenceTimeSum"])

    for dn in results_dirnames:
        results_dir = os.path.join(os.path.dirname(whoc.__file__), "case_studies", dn)

        results_df = pd.read_csv(os.path.join(results_dir, "time_series_results.csv"))
        results_dfs[dn] = results_df
        # res = ResultsSummary(YawAngleChangeAbsSum=results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum(),
        #                      FarmPowerSum=results_df["FarmPower"].sum(),
        #                      TotalOptimizationCostSum=results_df["TotalOptimizationCost"].sum(),
        #                      ConvergenceTimeSum=results_df["ConvergenceTime"].sum())
        result_summary_dict["SolverType"].append(dn)
        result_summary_dict["YawAngleChangeAbsSum"].append(results_df[[c for c in results_df.columns if "YawAngleChange" in c]].abs().sum().to_numpy().sum())
        result_summary_dict["FarmPowerSum"].append(results_df["FarmPower"].sum())
        result_summary_dict["TotalOptimizationCostSum"].append(results_df["TotalOptimizationCost"].sum())
        result_summary_dict["OptimizationConvergenceTimeSum"].append(results_df["OptimizationConvergenceTime"].sum())
    
    result_summary_df = pd.DataFrame(result_summary_dict)
    return result_summary_df


if __name__ == "__main__":
    dirnames = ["pyopt_sequential_perfect", "pyopt_perfect", "floris_perfect"]
    
    compare_simulations(dirnames)