from whoc.wind_forecast.WindForecast import SVRForecast
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import argparse
import yaml
import os
import logging 
from floris import FlorisModel
import gc
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    
    logging.info("Parsing arguments and configuration yaml.")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-mcnf", "--model_config", type=str)
    parser.add_argument("-dcnf", "--data_config", type=str)
    parser.add_argument("-sn", "--study_name", type=str)
    parser.add_argument("-m", "--multiprocessor", choices=["mpi", "cf"], default="cf")
    parser.add_argument("-i", "--initialize", action="store_true")
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("-md", "--model", type=str, choices=["svr", "kf", "preview", "informer", "autoformer", "spacetimeformer"], required=True)
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()
    
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
        
    assert model_config["optuna"]["backend"] in ["sqlite", "mysql", "journal"]
    
    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
        
    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
     
    fmodel = FlorisModel(data_config["farm_input_path"])
    
    prediction_timedelta = model_config["dataset"]["prediction_length"] * pd.Timedelta(model_config["dataset"]["resample_freq"]).to_pytimedelta()
    context_timedelta = model_config["dataset"]["context_length"] * pd.Timedelta(model_config["dataset"]["resample_freq"]).to_pytimedelta()
    
    # %% SETUP SEED
    logging.info(f"Setting random seed to {args.seed}")
    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% READING WIND FIELD TRAINING DATA
    logging.info("Reading input wind field.") 
    true_wf = pl.scan_parquet(model_config["dataset"]["data_path"])
    true_wf_norm_consts = pd.read_csv(model_config["dataset"]["normalization_consts_path"], index_col=None)
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
    
    # longest_cg = true_wf.select(pl.col("continuity_group")).to_series().value_counts().sort("count", descending=True).select(pl.col("continuity_group").first()).item()
    # true_wf = true_wf.filter(pl.col("continuity_group") == longest_cg)
    true_wf = true_wf.partition_by("continuity_group")
    historic_measurements = [wf.slice(0, wf.select(pl.len()).item() - int(prediction_timedelta / wind_dt)) for wf in true_wf]
    
    # %% PREPARING DIRECTORIES
    os.makedirs(model_config["optuna"]["storage_dir"], exist_ok=True)
    os.makedirs(data_config["temp_storage_dir"], exist_ok=True)
    
    # %% INSTANTIATING MODEL
    logging.info("Instantiating model.")  
    if args.model == "svr": 
        model = SVRForecast(measurements_timedelta=wind_dt,
                            controller_timedelta=None,
                            prediction_timedelta=prediction_timedelta,
                            context_timedelta=context_timedelta,
                            fmodel=fmodel,
                            true_wind_field=true_wf,
                            model_config=model_config,
                            kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200,
                                        n_neighboring_turbines=3, max_n_samples=None),
                            tid2idx_mapping=tid2idx_mapping,
                            turbine_signature=turbine_signature,
                            use_tuned_params=False,
                            temp_save_dir=data_config["temp_storage_dir"],
                            multiprocessor=args.multiprocessor)
    
    
    # %% PREPARING DATA FOR TUNING
    if args.initialize:
        logging.info("Preparing data for tuning")
        model.prepare_training_data(historic_measurements=historic_measurements)
        
        logging.info("Reinitializing storage") 
        if args.restart_tuning:
            storage = model.get_storage(
                backend=model_config["optuna"]["backend"], 
                    study_name=args.study_name, 
                    storage_dir=model_config["optuna"]["storage_dir"])
            for s in storage.get_all_studies():
                storage.delete_study(s._study_id)
    else: 
        # %% TUNING MODEL
        logging.info("Running tune_hyperparameters_multi")
        pruning_kwargs = model_config["optuna"]["pruning"] 
        #{"type": "hyperband", "min_resource": 2, "max_resource": 5, "reduction_factor": 3, "percentile": 25}
        model.tune_hyperparameters_single(study_name=args.study_name,
                                        backend=model_config["optuna"]["backend"],
                                        n_trials=model_config["optuna"]["n_trials"], 
                                        storage_dir=model_config["optuna"]["storage_dir"],
                                        seed=args.seed)
                                        #  trial_protection_callback=handle_trial_with_oom_protection)
    
        # %% TESTING LOADING HYPERPARAMETERS
        # Test setting parameters
        # model.set_tuned_params(backend=model_config["optuna"]["backend"], study_name_root=args.study_name, 
        #                        storage_dir=model_config["optuna"]["storage_dir"]) 
    
        # %% After training completes
        # torch.cuda.empty_cache()
        gc.collect()
        logging.info("Optuna hyperparameter tuning completed.")