from whoc.wind_preview.WindPreview import SVRPreview, KalmanFilterPreview
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import datetime
import argparse
import yaml

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-m", "--model", type=str, choices=["svr", "kf", "informer", "autoformer", "spacetimeformer"], required=True)
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)
    
    prediction_timedelta = config["dataset"]["prediction_length"] * pd.Timedelta(config["dataset"]["resample_freq"]).to_pytimedelta()
    context_timedelta = config["dataset"]["context_length"] * pd.Timedelta(config["dataset"]["resample_freq"]).to_pytimedelta()
    historic_measurements_limit = datetime.timedelta(minutes=30)
    
    true_wf = pl.scan_parquet(config["dataset"]["data_path"])
    true_wf_norm_consts = pd.read_csv(config["dataset"]["normalization_consts_path"], index_col=None)
    norm_min_cols = [col for col in true_wf_norm_consts if "_min" in col]
    norm_max_cols = [col for col in true_wf_norm_consts if "_max" in col]
    data_min = true_wf_norm_consts[norm_min_cols].values.flatten()
    data_max = true_wf_norm_consts[norm_max_cols].values.flatten()
    norm_min_cols = [col.replace("_min", "") for col in norm_min_cols]
    norm_max_cols = [col.replace("_max", "") for col in norm_max_cols]
    feature_range = (-1, 1)
    norm_scale = ((feature_range[1] - feature_range[0]) / (data_max - data_min))
    norm_min = feature_range[0] - (data_min * norm_scale)
    true_wf = true_wf.with_columns([(cs.starts_with(col) - norm_min[c]) 
                                                / norm_scale[c] 
                                                for c, col in enumerate(norm_min_cols)])\
                                .collect()
                                
    wind_dt = true_wf.select(pl.col("time").diff().shift(-1).first()).item()
    true_wf = true_wf.with_columns(data_type=pl.lit("True"))
    # true_wf = true_wf.with_columns(pl.col("time").cast(pl.Datetime(time_unit=pred_slice.unit)))
    # true_wf_plot = pd.melt(true_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    longest_cg = true_wf.select(pl.col("continuity_group")).to_series().value_counts().sort("count", descending=True).select(pl.col("continuity_group").first()).item()
    true_wf = true_wf.filter(pl.col("continuity_group") == longest_cg)
    historic_measurements = true_wf.slice(0, true_wf.select(pl.len()).item() - int(prediction_timedelta / wind_dt))
    
    if args.model == "svr": 
        model = SVRPreview(freq=wind_dt,
                                    prediction_timedelta=prediction_timedelta,
                                    context_timedelta=context_timedelta,
                                    svr_kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200))
    elif args.model == "kf":
        kf_preview = KalmanFilterPreview(freq=wind_dt,
                                        prediction_timedelta=prediction_timedelta,
                                        context_timedelta=context_timedelta,
                                        kf_kwargs=dict(H=np.array([1])))
        
    model.tune_hyperparameters_multi(historic_measurements)