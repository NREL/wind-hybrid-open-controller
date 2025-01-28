"""Module for wind speed component forecasting and preview functionality."""

from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import os
import datetime
import yaml
from functools import partial
from mysql.connector import connect as sql_connect

# import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.distributions import DistributionOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler

from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
import pytorch_lightning as L

from wind_forecasting.preprocessing.data_inspector import DataInspector

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from optuna import create_study
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

from filterpy.kalman import KalmanFilter

from wind_forecasting.preprocessing.data_module import DataModule

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class WindPreview:
    """Wind speed component forecasting module that provides various prediction methods."""
    context_timedelta: datetime.timedelta
    prediction_timedelta: datetime.timedelta
    freq: datetime.timedelta
    
    # def read_measurements(self):
    #     """_summary_
    #     Read in new measurements, add to internal container.
    #     """
    #     raise NotImplementedError()

    def __post_init__(self):
        self.n_context = int(self.context_timedelta / self.freq)
        self.n_prediction = int(self.prediction_timedelta / self.freq)
        
    def _tuning_objective(self, trial, X_train, y_train):
        """
        Objective function to be minimized in Optuna
        """
        # define hyperparameter search space
        params = self.get_params(trial)
         
        # train svr model
        model = self.__class__.create_model(**params)
        
        # evaluate with cross-validation
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring="neg_mean_squared_error").mean()
    
    
    def tune_hyperparameters_single(self, historic_measurements, scaler, study_name, storage, n_trials=1):
        X_train, y_train = self._get_training_data(historic_measurements, scaler)
        
        logging.info(f"Creating Optuna study {study_name}.") 
        study = create_study(study_name=study_name,
                             storage=storage,
                             load_if_exists=True)
        
        logging.info(f"Optimizing Optuna study {study_name}.") 
        study.optimize(partial(self._tuning_objective, X_train=X_train, y_train=y_train), n_trials=n_trials, show_progress_bar=True)
        return study.best_params
    
    def tune_hyperparameters_multi(self, historic_measurements, study_name, n_trials=1, use_rdb=False, journal_storage_dir=None):
        
        outputs = historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        best_params = {}
        
        if True:
            for output in outputs:
                
                if use_rdb:
                    logging.info(f"Connecting to RDB database {study_name}_{output}")
                    try:
                        db = sql_connect(host="localhost", user="root",
                                        database=f"{study_name}_{output}")       
                    except Exception: 
                        db = sql_connect(host="localhost", user="root")
                        cursor = db.cursor()
                        cursor.execute(f"CREATE DATABASE {study_name}_{output}") 
                    finally:
                        storage = RDBStorage(url=f"mysql://{db.user}@{db.server_host}:{db.server_port}/{db.database}")
                else:
                    logging.info(f"Connecting to Journal database {study_name}_{output}")
                    storage = JournalStorage(JournalFileBackend(os.path.join(journal_storage_dir, f"{study_name}_{output}.log")))
                
                best_params[output] = self.tune_hyperparameters_single(
                                                historic_measurements=historic_measurements.select(pl.col(output)),
                                                scaler=self.scaler[output],
                                                study_name=study_name,
                                                storage=storage, n_trials=n_trials)
            
                # self.model[output].set_params(best_params[output]) 
        
        # np.save("./svr_best_params.npz", best_params, allow_pickle=True)
        else:
            executor = ProcessPoolExecutor()
            with executor as tune_exec:
                futures = {output: tune_exec.submit(self.tune_hyperparameters_single,
                                                    historic_measurements=historic_measurements.select(pl.col(output)),
                                                    scaler=self.scaler[output],
                                                    storage=storage,
                                                    n_trials=n_trials)
                        for output in outputs}
                best_params = {output: futures[output].result() for output in outputs}
        
   
    def predict_sample(self, historic_measurements: pd.DataFrame, n_samples: int):
        """_summary_
        Predict a given number of samples for each time step in the horizon
        Args:
            n_samples (int): _description_
        """
        raise NotImplementedError()

    def predict_point(self, historic_measurements: pd.DataFrame):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        raise NotImplementedError()

    def predict_distr(self, historic_measurements: pd.DataFrame):
        """_summary_
        Generate the parameters of the forecasted distribution
        """
        raise NotImplementedError()

    def get_pred_interval(self, current_time):
        return pl.datetime_range(start=current_time, end=current_time + self.prediction_timedelta, interval=self.freq, eager=True, closed="right")

    @staticmethod
    def plot_preview(preview_wf, true_wf):
        fig, axs = plt.subplots(1, 2, sharex=True)
        
        for f, feat in enumerate(["ws_horz", "ws_vert"]):
            ax = sns.lineplot(data=true_wf.filter(pl.col("feature") == feat), x="time", y="value", hue="turbine_id", style="data_type", dashes=[[1, 0]], ax=axs[f])
            ax = sns.lineplot(data=preview_wf.filter(pl.col("feature") == feat), x="time", y="value", hue="turbine_id", style="data_type", dashes=[[4, 4]], marker="o", ax=axs[f])
            axs[f].set(xlabel="Time", ylabel="Wind Speed [m/s]", 
                       xlim=(true_wf.filter((pl.col("time") < pl.lit(preview_wf.select(pl.col("time").min()).item()))).select(pl.col("time").last()).item(), 
                             preview_wf.select(pl.col("time").max()).item()), title=feat)
        
        # axs[-].set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(preview_wf.select(pl.col("time").min()).item()], preview_wf.select(pl.col("time").max()).item()))
        axs[0].legend([], [], frameon=False)
        h, l = ax.get_legend_handles_labels()
        labels_1 = ["data_type", "True", "Preview"]
        labels_2 = ["turbine_id"] + sorted(list(preview_wf.select(pl.col("turbine_id").unique()).to_numpy().flatten()))
        handles_1 = [h[l.index(label)] for label in labels_1]
        handles_2 = [h[l.index(label)] for label in labels_2]
        leg1 = plt.legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(0.98, 1), frameon=False)
        leg2 = plt.legend(handles_2, labels_2, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        
        axs[-1].add_artist(leg1)
    
    @staticmethod
    def plot_turbine_data(long_df, fig_dir):
        fig_ts, ax_ts = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
        # fig_ts.set_size_inches(12, 6)
        if hasattr(ax_ts, '__len__'):
            ax_ts = ax_ts.flatten()
        else:
            ax_ts = [ax_ts]
        
        for cg in long_df.select(pl.col("continuity_group")).unique().to_numpy().flatten(): 
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="ws_horz", ax=ax_ts[0])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="ws_vert", ax=ax_ts[1])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="nd_cos", ax=ax_ts[2])
            sns.lineplot(data=long_df.filter(pl.col("continuity_group") == cg), hue="turbine_id", x="time", y="nd_sin", ax=ax_ts[3])

        ax_ts[0].set(title='Downwind Freestream Wind Speed, U [m/s]', ylabel="")
        ax_ts[1].set(title='Crosswind Freestream Wind Speed, V [m/s]', ylabel="")
        ax_ts[2].set(title='Nacelle Direction Cosine [-]', ylabel="")
        ax_ts[3].set(title='Nacelle Direction Sine [-]', ylabel="")

        # handles, labels, kwargs = mlegend._parse_legend_args([ax_ts[0]], ncol=2, title="Wind Seed")
        # ax_ts[0].legend_ = mlegend.Legend(ax_ts[0], handles, labels, **kwargs)
        # ax_ts[0].legend_.set_ncols(2)
        for i in range(0, len(ax_ts)):
            ax_ts[i].legend([], [], frameon=False)
        
        # time = long_df.select("time").to_numpy().flatten()
        # for i in range(len(ax_ts) - 2, len(ax_ts)):
        #     ax_ts[i].set(xticks=time[0:-1:int(60 * 12 // (time[1] - time[0]))], xlabel='Time [s]')
            # xlim=(time.iloc[0], 3600.0)) 
        
        plt.tight_layout()
        fig_ts.savefig(os.path.join(fig_dir, f'wind_field_ts.png'))
    

@dataclass
class PerfectPreview(WindPreview):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    true_wind_field: pd.DataFrame
    col_mapping: Optional[dict] = None
    
    # def read_measurements(self):
    #     pass
    
    # def __post_init__(self):
    #     if all(col in self.true_wind_field.columns for col in self.col_mapping.values()):
    #         self.map_columns = False
    #     else:
    #         self.map_columns = True
    
    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        sub_df = self.true_wind_field.rename(columns=self.col_mapping) if self.col_mapping else self.true_wind_field
        sub_df = sub_df.filter(pl.col("time").is_between(current_time, current_time + self.prediction_timedelta, closed="right"))
        assert sub_df.select(pl.len()).item() == int(self.prediction_timedelta / self.freq)
        return sub_df

@dataclass
class PersistentPreview(WindPreview):
    """Wind speed component forecasting using persistence model that assumes future values equal current value."""
    
    # def read_measurements(self):
    #     pass
    
    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        pred_slice = self.get_pred_interval(current_time)
        assert (historic_measurements["time"] == current_time).any()
        last_measurement = historic_measurements.filter(pl.col("time") == current_time)
        return pl.concat([pl.DataFrame({"time": pred_slice}), 
                          last_measurement.select(pl.exclude("time").repeat_by(len(pred_slice)).explode())], 
                         how="horizontal")

@dataclass
class GaussianPreview(WindPreview):
    """Wind speed component forecasting using Gaussian parameterization."""
    
    # def read_measurements(self):
    #     pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

@dataclass
class SVRPreview(WindPreview):
    """Wind speed component forecasting using Support Vector Regression."""
    svr_kwargs: dict
    
    def __post_init__(self):
        # TODO try to compute for each turbine? No just individual turbines, but can use consensus for historic measurments...
        super().__post_init__()
        self.scaler = defaultdict(SVRPreview.create_scaler)
        self.model = defaultdict(SVRPreview.create_model)
        # self.model = defaultdict(lambda: make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVR(**self.svr_kwargs)))
        
    # def read_measurements(self):
    #     pass
    
    @staticmethod
    def create_scaler():
        return MinMaxScaler(feature_range=(-1, 1))
    
    @staticmethod
    def create_model(**kwargs):
        return SVR(**kwargs)
    
    def _get_training_data(self, historic_measurements, scaler):
        if scaler:
            historic_measurements = scaler.fit_transform(historic_measurements.to_numpy())
            
        X_train = np.ascontiguousarray(np.vstack([
            historic_measurements[i:i+self.n_context, 0]
            for i in range(historic_measurements.shape[0] - self.n_context)
        ]))
        
        # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_timedelta][output])
        y_train = np.ascontiguousarray(historic_measurements[self.n_context:, 0])
        
        return X_train, y_train
    
    def get_params(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-6, 1e6, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-6, 1e-1, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
        }
    

    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
         # TODO include yaw angles in inputs? 
        pred_slice = self.get_pred_interval(current_time)
        # pred = defaultdict(list)
        context_expr = pl.col("time").is_between((current_time - self.context_timedelta), current_time, closed="right")
        # context_df = historic_measurements.filter(context_expr)
        
        assert historic_measurements.select(context_expr.sum()).item() == self.n_context
        executor = ProcessPoolExecutor()
        outputs = historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        
        with executor as train_exec:
            # for output in historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns:
            futures = {output: train_exec.submit(self._train_model, 
                                                 historic_measurements.select(pl.col(output)), 
                                                 self.scaler[output],
                                                 self.model[output])
                       for output in outputs}
            res = {output: futures[output].result() for output in outputs}
            self.scaler = {output: res[output][0] for output in outputs}
            self.model = {output: res[output][1] for output in outputs}
            
        # rescale back
        pred = {output: self.scaler[output].inverse_transform(res[output][2][np.newaxis, :]).flatten() for output in outputs}
        
        return pl.DataFrame({"time": pred_slice}).with_columns(**pred)

    def _train_model(self, historic_measurements, scaler, model):
        historic_measurements = scaler.fit_transform(historic_measurements.to_numpy())
        X_train = np.ascontiguousarray(np.vstack([
            historic_measurements[i:i+self.n_context, 0]
            for i in range(historic_measurements.shape[0] - self.n_context)
        ]))
        
        # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_timedelta][output])
        y_train = np.ascontiguousarray(historic_measurements[self.n_context:, 0])
        model.fit(X_train, y_train)
        
        X_pred = np.ascontiguousarray(historic_measurements[-self.n_context:, :].T)
        # X_pred = np.ascontiguousarray(historic_measurements.iloc[-1][output])[:, np.newaxis]
        
        for _ in range(self.n_prediction):
            y_pred = model.predict(X_pred)
            X_pred = np.hstack([X_pred[:, 1:], y_pred[:, np.newaxis]])
            # X_pred = np.ascontiguousarray(pred[output][-1])[:, np.newaxis]
        
        pred = X_pred.flatten()[-self.n_prediction:] 
        assert len(pred) == self.n_prediction
        return scaler, model, pred
    
    def predict_distr(self):
        pass

class KalmanFilterWrapper(KalmanFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y):
        pred = []
        self.x = X[0]
        for i in range(X.shape[0]):
            z = X[i]
            self.predict()
            self.update(z)
            pred.append(self.x[0])
    
    def set_params(self, **kwargs):
        if "R" in kwargs:
            self.R = kwargs["R"]
        if "H" in kwargs:
            self.H = kwargs["H"]
        if "Q" in kwargs:
            self.Q = kwargs["Q"]

@dataclass
class KalmanFilterPreview(WindPreview):
    """Wind speed component forecasting using Kalman filtering."""
    kf_kwargs: dict
    # def read_measurements(self):
    #     pass
    
    def __post_init__(self):
        super().__post_init__()
        self.model = defaultdict(KalmanFilterPreview.create_model)
        self.scaler = defaultdict(KalmanFilterPreview.create_scaler)
        
    @staticmethod
    def create_scaler():
        return None
    
    @staticmethod
    def create_model(**kwargs):
        model = KalmanFilterWrapper(dim_x=1, dim_z=1)
        if "R" in kwargs:
            model.R = kwargs["R"]
        if "H" in kwargs:
            model.H = kwargs["H"]
        if "Q" in kwargs:
            model.Q = kwargs["Q"]
        return model
    
    def _get_training_data(self, historic_measurements, scaler):
        if scaler:
            historic_measurements = scaler.fit_transform(historic_measurements.to_numpy())
        X_train = historic_measurements[0:-self.n_context, 0]
        
        # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_timedelta][output])
        y_train = historic_measurements[self.n_context:, 0]
        
        return X_train, y_train
    
    def get_params(self, trial):
        return {
            "R": np.array([trial.suggest_float("R", 1e-3, 1e2, log=True)]),
            "H": np.array([trial.suggest_float("H", 1e-3, 1e2, log=True)]),
            "Q": np.array([trial.suggest_float("Q", 1e-3, 1e2, log=True)])
        }
       
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        pred_slice = self.get_pred_interval(current_time)
        pred = defaultdict(list)
        outputs = historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        
        # futures = {output: self._train_model(historic_measurements.select(pl.col(output)), 
        #                                          self.model[output]) for output in outputs}
         
        executor = ProcessPoolExecutor() 
        with executor as train_exec:
            # for output in historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns:
            futures = {output: train_exec.submit(self._train_model, 
                                                 historic_measurements.select(pl.col(output)), 
                                                 self.model[output])
                       for output in outputs}
            res = {output: futures[output].result() for output in outputs}
            self.model = {output: res[output][0] for output in outputs}
            pred = {output: res[output][1] for output in outputs}
            
        return pl.DataFrame({"time": pred_slice}).with_columns(**pred)

    def _train_model(self, historic_measurements, model):
        pred = []
        model.x = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(historic_measurements.select(pl.len()).item()):
            z = historic_measurements.slice(i, 1).to_numpy().flatten()
            model.predict()
            model.update(z)
        
        z = historic_measurements.slice(-1, 1).to_numpy().flatten()
        for i in range(self.n_prediction):
            model.predict()
            model.update(z)
            z = model.x
            pred.append(z[0])
            
        return model, np.array(pred)
    
    def predict_distr(self):
        pass

@dataclass
class MLPreview(WindPreview):
    """Wind speed component forecasting using machine learning models."""
    model_checkpoint_path: Path
    model_config_path: Path
    model_config_key: str # must be key in given model_config_path
    estimator_class: PyTorchLightningEstimator
    distr_output: DistributionOutput 
    lightning_module: L.LightningModule
    
    def __post_init__(self):
        super().__post_init__()
        with open(self.model_config_path, 'r') as file:
            config  = yaml.safe_load(file)
            
        self.data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                per_turbine_target=config["dataset"]["per_turbine_target"], dtype=None)
        self.data_module.get_dataset_info() 
        # logging.info("Declaring estimator")
        estimator = self.estimator_class(
            freq=self.data_module.freq, 
            prediction_length=self.data_module.prediction_length,
            context_length=self.data_module.context_length,
            num_feat_dynamic_real=self.data_module.num_feat_dynamic_real, 
            num_feat_static_cat=self.data_module.num_feat_static_cat,
            cardinality=self.data_module.cardinality,
            num_feat_static_real=self.data_module.num_feat_static_real,
            input_size=self.data_module.num_target_vars,
            scaling=False,
            batch_size=config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=config["trainer"].setdefault("limit_train_batches", 50), # TODO set this to be arbitrarily high st limit train_batches dominates
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=self.data_module.context_length, min_future=self.data_module.prediction_length), # TODO should be context_len + max(seq_len) to avoid padding..
            validation_sampler=ValidationSplitSampler(min_past=self.data_module.context_length, min_future=self.data_module.prediction_length),
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=self.distr_output(dim=self.data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=config["trainer"],
            **config["model"][self.model_config_key]
        )
        self.data_module.freq = pd.Timedelta(self.data_module.freq).to_pytimedelta()
        self.normalization_consts = pd.read_csv(config["dataset"]["normalization_consts_path"], index_col=None)
        model = self.lightning_module.load_from_checkpoint(self.model_checkpoint_path)
        transformation = estimator.create_transformation(use_lazyframe=False)
        self.predictor = estimator.create_predictor(transformation, model, 
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))
        
        # normalize data to -1, 1 using saved normalization consts
        norm_min_cols = [col for col in self.normalization_consts if "_min" in col]
        norm_max_cols = [col for col in self.normalization_consts if "_max" in col]
        data_min = self.normalization_consts[norm_min_cols].values.flatten()
        data_max = self.normalization_consts[norm_max_cols].values.flatten()
        self.norm_min_cols = [col.replace("_min", "") for col in norm_min_cols]
        self.norm_max_cols = [col.replace("_max", "") for col in norm_max_cols]
        feature_range = (-1, 1)
        self.norm_scale = ((feature_range[1] - feature_range[0]) / (data_max - data_min))
        self.norm_min = feature_range[0] - (data_min * self.norm_scale)
    
    # def read_measurements(self):
    #     pass
    
    def _generate_test_data(self, historic_measurements: pd.DataFrame):
        # resample data to frequency model was trained on
        if self.data_module.freq != self.freq:
            if self.freq > self.data_module.freq:
                historic_measurements = historic_measurements.with_columns(time=pl.col("time").dt.round(self.data_module.freq))\
                                                              .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                historic_measurements = historic_measurements.upsample(time_column="time", every=self.data_module.freq).fill_null(strategy="forward")
        
        historic_measurements = historic_measurements.with_columns(cs.numeric().cast(pl.Float32))
        # test_data must be iterable where each item generated is a dict with keys start, target, item_id, and feat_dynamic_real
        # this should include measurements at all turbines
        if self.data_module.per_turbine_target:
            test_data = (
                {
                    "item_id": f"TURBINE{turbine_id}",
                    "start": pd.Period(historic_measurements.select(pl.col("time").first()).item(), freq=self.data_module.freq), 
                    "target": historic_measurements.select(self.data_module.target_cols).to_numpy().T, 
                    "feat_dynamic_real": pl.concat([
                        historic_measurements.select(self.data_module.feat_dynamic_real_cols),
                        historic_measurements.select([pl.col(col).last().repeat_by(int(self.prediction_timedelta / self.data_module.freq)).explode() 
                                                      for col in self.data_module.feat_dynamic_real_cols])], how="vertical")
                } for turbine_id in self.data_module.target_suffixes)
        else:
            test_data = [{
                "start": pd.Period(historic_measurements.select(pl.col("time").first()).item(), freq=self.data_module.freq), 
                    "target": historic_measurements.select(self.data_module.target_cols).to_numpy().T, 
                    "feat_dynamic_real": pl.concat([
                        historic_measurements.select(self.data_module.feat_dynamic_real_cols),
                        historic_measurements.select([pl.col(col).last().repeat_by(int(self.prediction_timedelta / self.data_module.freq)).explode() 
                                                      for col in self.data_module.feat_dynamic_real_cols])], how="vertical").to_numpy().T
            }]
        return test_data
    
    def predict_sample(self, historic_measurements: pd.DataFrame, current_time, n_samples: int):
        # TODO should I be able to feed future time features and yaw angles into the test data? check what happens in forward method
        historic_measurements = historic_measurements.with_columns([(cs.starts_with(col) * self.norm_scale[c]) 
                                                    + self.norm_min[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        test_data = self._generate_test_data(historic_measurements)
            
        pred = self.predictor.predict(test_data, num_samples=n_samples, output_distr_params=False)
        pred = list(pred)
        
        # resample historic measurements to model frequency and return as pandas dataframe
        pred = pred.select([(cs.starts_with(col) - self.norm_min[c]) 
                                                    / self.norm_min[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        
        if self.model_freq != self.freq: # TODO check that we are comparing same types here
            pred = pred.resample(self.freq).mean()
        
        # TODO rename pred
        return pred   

    def predict_point(self, historic_measurements: pd.DataFrame, current_time): 
        pass

    def predict_distr(self, historic_measurements: pd.DataFrame, current_time):
        test_data = self._generate_test_data(historic_measurements)
        pred = self.predictor.predict(test_data, num_samples=1, output_distr_params=True)
        if self.model_freq != self.freq: # TODO check that we are comparing same types here
            pred = pred.resample(self.freq).mean()
        return pred



if __name__ == "__main__":
    import yaml
    import os
    import argparse
    from glob import glob
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(prog="ModelTuning")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-m", "--model", type=str, choices=["svr", "kf", "informer", "autoformer", "spacetimeformer"], required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)
    
    prediction_timedelta = config["dataset"]["prediction_length"] * pd.Timedelta(config["dataset"]["resample_freq"]).to_pytimedelta()
    context_timedelta = config["dataset"]["context_length"] * pd.Timedelta(config["dataset"]["resample_freq"]).to_pytimedelta()
    historic_measurements_limit = datetime.timedelta(minutes=30)
    
    # TODO in GreedyController replace historic_measurements with future_predictions (if wind_preview is not None) to compute wind_dirs (before passing to lpf)
    #      and compute optimal yaw angles and power outputs with ControlledFlorisInterface
    #      then compare optimal yaw angles and power outputs to those computed without wind_preview
    
    ## GET TRUE WIND FIELD
    # pull ws_horz, ws_vert, nacelle_direction, normalization_consts from awaken data and run for ML, SVR
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
    data_config_path = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml"
    with open(data_config_path, 'r') as file:
        data_config  = yaml.safe_load(file)
    # if wind field data exists, get it
    # wind_field_dir = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "floris_case_studies", "wind_field_data", "raw_data")        
    # wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    # true_wfs = []
    # if os.path.exists(wind_field_dir) and len(wind_field_filenames):
    #     for fn in wind_field_filenames:
    #         true_wfs.append(pd.read_csv(os.path.join(wind_field_dir, fn), index_col=0))
    # else:
    #     raise Exception(f"No wind field files found in {wind_field_dir}.")
    
    if True:
        true_wf_long = DataInspector.unpivot_dataframe(true_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                turbine_signature=data_config["turbine_signature"])
        WindPreview.plot_turbine_data(long_df=true_wf_long, fig_dir="./")
        del true_wf_long
    # plt.savefig(os.path.join(wind_field_config["fig_dir"], "wind_field_ts.png"))

    wind_dt = true_wf.select(pl.col("time").diff().shift(-1).first()).item()
    true_wf = true_wf.with_columns(data_type=pl.lit("True"))
    # true_wf = true_wf.with_columns(pl.col("time").cast(pl.Datetime(time_unit=pred_slice.unit)))
    # true_wf_plot = pd.melt(true_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    
    longest_cg = true_wf.select(pl.col("continuity_group")).to_series().value_counts().sort("count", descending=True).select(pl.col("continuity_group").first()).item()
    true_wf = true_wf.filter(pl.col("continuity_group") == longest_cg)
    historic_measurements = true_wf.slice(0, true_wf.select(pl.len()).item() - int(prediction_timedelta / wind_dt))
    future_measurements = true_wf.slice(true_wf.select(pl.len()).item() - int(prediction_timedelta / wind_dt), int(prediction_timedelta / wind_dt))
    current_time = historic_measurements.select(pl.col("time").last()).item()
    assert int(context_timedelta / wind_dt) <= historic_measurements.select(pl.len()).item()
    assert int(prediction_timedelta / wind_dt) <= future_measurements.select(pl.len()).item()
    
    id_vars = true_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns  
    true_wf_long = DataInspector.unpivot_dataframe(true_wf,
                                                value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                turbine_signature=data_config["turbine_signature"])\
                                 .unpivot(index=["turbine_id"] + id_vars, 
                                            on=["ws_horz", "ws_vert", "nd_cos", "nd_sin"], 
                                            variable_name="feature", value_name="value")
     
    ## GENERATE PERFECT PREVIEW \
    if False:
        perfect_preview = PerfectPreview(
            freq=wind_dt,
            prediction_timedelta=prediction_timedelta,
            context_timedelta=context_timedelta,
            true_wind_field=true_wf,
        )
        perfect_preview_wf = perfect_preview.predict_point(historic_measurements, current_time)
        perfect_preview_wf = perfect_preview_wf.with_columns(data_type=pl.lit("Preview"))
        id_vars = perfect_preview_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns 
        perfect_preview_wf = DataInspector.unpivot_dataframe(perfect_preview_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], 
                                                    variable_name="feature", value_name="value")
    
        WindPreview.plot_preview(perfect_preview_wf, true_wf_long)
     
    ## GENERATE PERSISTENT PREVIEW
    if False:
        persistent_preview = PersistentPreview(freq=wind_dt,
                                            prediction_timedelta=prediction_timedelta,
                                            context_timedelta=context_timedelta)
        
        persistent_preview_wf = persistent_preview.predict_point(historic_measurements, current_time)
        persistent_preview_wf = persistent_preview_wf.with_columns(data_type=pl.lit("Preview"))
        id_vars = persistent_preview_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        persistent_preview_wf = DataInspector.unpivot_dataframe(persistent_preview_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], 
                                                    variable_name="feature", value_name="value")
        
        WindPreview.plot_preview(persistent_preview_wf, true_wf_long) 
    
    ## GENERATE SVR PREVIEW
    if True:
        svr_preview = SVRPreview(freq=wind_dt,
                                prediction_timedelta=prediction_timedelta,
                                context_timedelta=context_timedelta,
                                svr_kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200))
        
        svr_preview.tune_hyperparameters_multi(historic_measurements)
        
        svr_preview_wf = svr_preview.predict_point(historic_measurements, current_time)
        svr_preview_wf = svr_preview_wf.with_columns(data_type=pl.lit("Preview"))
        id_vars = svr_preview_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        svr_preview_wf = DataInspector.unpivot_dataframe(svr_preview_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], variable_name="feature", value_name="value")
        
        
        WindPreview.plot_preview(svr_preview_wf, true_wf_long)
    
    ## GENERATE KF PREVIEW 
    if False:
        kf_preview = KalmanFilterPreview(freq=wind_dt,
                                        prediction_timedelta=prediction_timedelta,
                                        context_timedelta=context_timedelta,
                                        kf_kwargs=dict(H=np.array([1])))
        
        kf_preview_wf = kf_preview.predict_point(historic_measurements, current_time)
        kf_preview_wf = kf_preview_wf.with_columns(data_type=pl.lit("Preview"))
        id_vars = kf_preview_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        kf_preview_wf = DataInspector.unpivot_dataframe(kf_preview_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], variable_name="feature", value_name="value")
    
        WindPreview.plot_preview(kf_preview_wf, true_wf_long)
    
    ## GENERATE ML PREVIEW
    if True:
        ml_preview = MLPreview(freq=wind_dt,
                                prediction_timedelta=prediction_timedelta,
                                context_timedelta=context_timedelta,
                                estimator_class=InformerEstimator,
                                lightning_module=InformerLightningModule,
                                distr_output=LowRankMultivariateNormalOutput,
                                model_config_key="informer",
                                model_checkpoint_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/checkpoints/lightning_logs/version_172/checkpoints/epoch=2-step=300.ckpt",
                                model_config_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml")
        
        ml_preview_wf =  ml_preview.predict_sample(historic_measurements, current_time, n_samples=50)
        ml_preview_wf = ml_preview_wf.with_columns(data_type=pl.lit("Preview"))
        id_vars = kf_preview_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        ml_preview_wf = DataInspector.unpivot_dataframe(
            ml_preview_wf, 
            id_vars=id_vars,
            value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
            turbine_signature=data_config["turbine_signature"])\
    .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], variable_name="feature", value_name="value")
        
        WindPreview.plot_preview(ml_preview_wf, true_wf_long)
    
    ## TODO other baseline eg. LSTM or DeepAR from gluonts
    print("here")
    
    # # TODO pass the checkpoint filepath and the model
    # ml_preview = MLPreview(prediction_timedelta=prediction_timedelta,
    #                                  context_timedelta=context_timedelta,
    #                                  probabilistic=True,
    #                                  distr_output=LowRankMultivariateNormalOutput)