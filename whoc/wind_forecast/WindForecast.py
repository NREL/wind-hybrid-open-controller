"""Module for wind speed component forecasting and preview functionality."""

from typing import Optional, Union
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
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
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
class WindForecast:
    """Wind speed component forecasting module that provides various prediction methods."""
    context_timedelta: datetime.timedelta
    prediction_timedelta: datetime.timedelta
    measurements_dt: datetime.timedelta
    true_wind_field: Optional[Union[pd.DataFrame, pl.DataFrame]]
    
    # def read_measurements(self):
    #     """_summary_
    #     Read in new measurements, add to internal container.
    #     """
    #     raise NotImplementedError()

    def __post_init__(self):
        self.n_context = int(self.context_timedelta / self.measurements_dt)
        self.n_prediction = int(self.prediction_timedelta / self.measurements_dt)
    
    def _get_ws_cols(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame]):
        if isinstance(historic_measurements, pl.DataFrame):
            return historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        elif isinstance(historic_measurements, pd.DataFrame):
            return [col for col in historic_measurements.columns if (col.startswith("ws_horz") or col.startswith("ws_vert"))]
     
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
                                direction="maximize",
                                load_if_exists=True) # maximize negative mse ie minimize mse
        
        logging.info(f"Optimizing Optuna study {study_name}.") 
        study.optimize(partial(self._tuning_objective, X_train=X_train, y_train=y_train), n_trials=n_trials, show_progress_bar=True)
        return study.best_params
    
    def get_storage(self, use_rdb, study_name, journal_storage_dir=None):
        if use_rdb:
            logging.info(f"Connecting to RDB database {study_name}")
            try:
                db = sql_connect(host="localhost", user="root",
                                database=study_name)       
            except Exception: 
                db = sql_connect(host="localhost", user="root")
                cursor = db.cursor()
                cursor.execute(f"CREATE DATABASE {study_name}") 
            finally:
                storage = RDBStorage(url=f"mysql://{db.user}@{db.server_host}:{db.server_port}/{study_name}")
        else:
            logging.info(f"Connecting to Journal database {study_name}")
            storage = JournalStorage(JournalFileBackend(os.path.join(journal_storage_dir, f"{study_name}.log")))
        return storage
    
    def tune_hyperparameters_multi(self, historic_measurements, study_name_root, n_trials=1, use_rdb=False, journal_storage_dir=None, restart_study=False):
        
        self.outputs = historic_measurements.select(cs.starts_with("ws_horz") | cs.starts_with("ws_vert")).columns
        best_params = {}
        
        for output in self.outputs:
            storage = self.get_storage(use_rdb=use_rdb, study_name=f"{study_name_root}_{output}", journal_storage_dir=journal_storage_dir)
            if restart_study:
                for s in storage.get_all_studies():
                    storage.delete_study(s._study_id)
                    
            best_params[output] = self.tune_hyperparameters_single(
                                            historic_measurements=historic_measurements.select(pl.col(output)),
                                            scaler=self.scaler[output],
                                            study_name=f"{study_name_root}_{output}",
                                            storage=storage, n_trials=n_trials)
        
        
    def set_tuned_params(self, use_rdb, study_name_root):
        for output in self.outputs:
            storage = self.get_storage(use_rdb=use_rdb, study_name=f"{study_name_root}_{output}")
            try:
                study_id = storage.get_study_id_from_name(f"{study_name_root}_{output}")
            except Exception:
                raise Exception(f"Optuna study {study_name_root}_{output} not found. Please run tune_hyperparameters_multi for all outputs first.")
            # self.model[output].set_params(**storage.get_best_trial(study_id).params)
            # storage.get_all_studies()[0]._study_id
            self.model[output] = self.create_model(**storage.get_best_trial(study_id).params)
   
    def predict_sample(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], n_samples: int):
        """_summary_
        Predict a given number of samples for each time step in the horizon
        Args:
            n_samples (int): _description_
        """
        raise NotImplementedError()

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame]):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        raise NotImplementedError()

    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame]):
        """_summary_
        Generate the parameters of the forecasted distribution
        """
        raise NotImplementedError()

    def get_pred_interval(self, current_time):
        return pl.datetime_range(start=current_time, end=current_time + self.prediction_timedelta, interval=self.measurements_dt, eager=True, closed="right")

    @staticmethod
    def plot_forecast(preview_wf, true_wf, prediction_type="point", per_turbine_target=False):
        fig, axs = plt.subplots(1, 2, sharex=True)
        
        for f, feat in enumerate(["ws_horz", "ws_vert"]):
            sns.lineplot(data=true_wf.filter(
                            (pl.col("feature") == feat) & (pl.col("time").is_between(preview_wf.select(pl.col("time").min()).item(), preview_wf.select(pl.col("time").max()).item(), closed="both"))), 
                                 x="time", y="value", ax=axs[f], style="data_type", hue="turbine_id")
            
            # df = pl.concat([true_wf.filter(
            #                 (pl.col("feature") == feat) & (pl.col("time").is_between(preview_wf.select(pl.col("time").min()).item(), preview_wf.select(pl.col("time").max()).item(), closed="both"))), 
            #                 preview_wf.filter((pl.col("feature") == f"loc_{feat}"))], 
            #       how="diagonal_relaxed")
            
            # ax = sns.lineplot(data=true_wf.filter((pl.col("feature") == feat) 
            #                                       & (pl.col("time").is_between(preview_wf.select(pl.col("time").min()).item(), preview_wf.select(pl.col("time").max()).item(), closed="both"))), 
            #                   x="time", y="value", hue="turbine_id", linestyle="solid", ax=axs[f])
            if prediction_type == "distribution":
                if per_turbine_target:
                    # TODO test
                    sns.lineplot(data=preview_wf.filter((pl.col("feature") == f"loc_{feat}")), 
                                 x="time", y="value", ax=axs[f], style="data_type", dashes=[[4, 4]], marker="o")
                    
                    axs[f].fill_between(
                        preview_wf.select("time"), 
                        preview_wf.filter((pl.col("feature") == f"loc_{feat}")) - preview_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                        preview_wf.filter((pl.col("feature") == f"loc_{feat}")) + preview_wf.filter((pl.col("feature") == f"sd_{feat}")), 
                        alpha=0.2, 
                    )
                else:
                    sns.lineplot(data=preview_wf.filter(pl.col("feature") == f"loc_{feat}"), 
                                 x="time", y="value", hue="turbine_id", style="data_type", ax=axs[f], dashes=[[4, 4]], marker="o")
                    
                    for t, tid in enumerate(preview_wf["turbine_id"].unique(maintain_order=True)):
                        # color = loc_ax.get_lines()[t].get_color()
                        tid_df = preview_wf.filter((pl.col("feature").str.contains(feat)) & (pl.col("turbine_id") == tid))
                        color = sns.color_palette()[t]
                        axs[f].fill_between(
                            tid_df.filter(pl.col("feature") == f"loc_{feat}").select("time").to_numpy().flatten(), 
                            (tid_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                             - tid_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                            (tid_df.filter(pl.col("feature") == f"loc_{feat}").select(pl.col("value")) 
                             + tid_df.filter(pl.col("feature") == f"sd_{feat}").select(pl.col("value"))).to_numpy().flatten(), 
                        alpha=0.2, 
                    )
            else:
                
                sns.lineplot(data=preview_wf.filter(pl.col("feature") == feat), x="time", y="value", hue="turbine_id", style="data_type", dashes=[[4, 4]], marker="o", ax=axs[f])
                
            axs[f].set(xlabel="Time", ylabel="Wind Speed [m/s]", 
                       xlim=(true_wf.filter((pl.col("time") < pl.lit(preview_wf.select(pl.col("time").min()).item()))).select(pl.col("time").last()).item(), 
                             preview_wf.select(pl.col("time").max()).item()), title=feat)
        
        axs[0].legend([], [], frameon=False)
        h, l = axs[-1].get_legend_handles_labels()
        labels_1 = ["data_type", "True", "Forecast"]
        labels_2 = ["turbine_id"] + sorted(list(preview_wf.select(pl.col("turbine_id").unique()).to_numpy().flatten()))
        handles_1 = [h[l.index(label)] for label in labels_1]
        handles_2 = [h[l.index(label)] for label in labels_2]
        leg1 = plt.legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(0.98, 1), frameon=False)
        leg2 = plt.legend(handles_2, labels_2, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        axs[-1].add_artist(leg1)
        # axs[-].set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(preview_wf.select(pl.col("time").min()).item()], preview_wf.select(pl.col("time").max()).item()))

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
class PerfectForecast(WindForecast):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    
    col_mapping: Optional[dict] = None
    
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        if isinstance(self.true_wind_field, pl.DataFrame):
            sub_df = self.true_wind_field.rename(self.col_mapping) if self.col_mapping else self.true_wind_field
            sub_df = sub_df.filter(pl.col("time").is_between(current_time, current_time + self.prediction_timedelta, closed="right"))
            assert sub_df.select(pl.len()).item() == int(self.prediction_timedelta / self.measurements_dt)
        elif isinstance(self.true_wind_field, pd.DataFrame):
            sub_df = self.true_wind_field.rename(columns=self.col_mapping) if self.col_mapping else self.true_wind_field
            sub_df = sub_df.loc[(sub_df["time"] > current_time) & (sub_df["time"] <= (current_time + self.prediction_timedelta)), :].reset_index(drop=True)
            assert len(sub_df.index) == int(self.prediction_timedelta / self.measurements_dt)
        return sub_df

@dataclass
class PersistentForecast(WindForecast):
    """ Wind speed component forecasting using persistence model that assumes future values equal current value. """
    
    # def read_measurements(self):
    #     pass
    
    def predict_point(self, historic_measurements: Union[pl.DataFrame, pd.DataFrame], current_time):
        pred_slice = self.get_pred_interval(current_time)
        
        if isinstance(historic_measurements, pd.DataFrame):
            assert (historic_measurements["time"] == current_time).any()
            last_measurement = historic_measurements.loc[historic_measurements["time"] == current_time, :]
            return pd.concat([
                pd.DataFrame(data={"time": pred_slice}), 
                last_measurement.loc[last_measurement.index.repeat(len(pred_slice)), [col for col in last_measurement.columns if col != "time"]].reset_index(drop=True)
            ], axis=1)
            
        elif isinstance(historic_measurements, pl.DataFrame):
            assert historic_measurements.select((pl.col("time") == current_time).any()).item()
            last_measurement = historic_measurements.filter(pl.col("time") == current_time)
            return pl.concat([pl.DataFrame({"time": pred_slice}), 
                          last_measurement.select(pl.exclude("time").repeat_by(len(pred_slice)).explode())], 
                         how="horizontal")
             
        

@dataclass
class PreviewForecast(WindForecast):
    """
    Reads wind speed components from upstream turbines, and computes time of arrival based on taylor's frozen wake hypothesis.
    """
    def create_preview_mapping(self, historic_measurments):
        # TODO create a mapping from each turbine id to the turbine id used for preview measurements
        
        # rotate turbine coordinates based on most recent wind direction measurement
		# order turbines based on order of wind incidence
        layout_x = self.fi.env.layout_x
        layout_y = self.fi.env.layout_y
        wd = historic_measurements["FreestreamWindDir"][0, 0]
        # wd = 250.0
        layout_x_rot = (
            np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
            - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
        )
        turbines_ordered = np.argsort(layout_x_rot)
    
    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        pred_slice = self.get_pred_interval(current_time)
        if isinstance(historic_measurements, pd.DataFrame):
            assert (historic_measurements["time"] == current_time).any()
        elif isinstance(historic_measurements, pl.DataFrame):
            assert historic_measurements.select((pl.col("time") == current_time).any()).item()
        # TODO
        # last_measurement = historic_measurements.filter(pl.col("time") == current_time)
        # return pl.concat([pl.DataFrame({"time": pred_slice}), 
        #                   last_measurement.select(pl.exclude("time").repeat_by(len(pred_slice)).explode())], 
        #                  how="horizontal") 

@dataclass
class GaussianForecast(WindForecast):
    """ Wind speed component forecasting using Gaussian parameterization. """
    
    # def read_measurements(self):
    #     pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

@dataclass
class SVRForecast(WindForecast):
    """Wind speed component forecasting using Support Vector Regression."""
    svr_kwargs: dict
    
    def __post_init__(self):
        # TODO try to compute for each turbine? No just individual turbines, but can use consensus for historic measurments...
        super().__post_init__()
        self.scaler = defaultdict(SVRForecast.create_scaler)
        self.model = defaultdict(SVRForecast.create_model)
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

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
         # TODO include yaw angles in inputs? 
        pred_slice = self.get_pred_interval(current_time)
        # pred = defaultdict(list)
        
        # context_df = historic_measurements.filter(context_expr)
        outputs = self._get_ws_cols(historic_measurements)
        if isinstance(historic_measurements, pl.DataFrame):
            context_expr = pl.col("time").is_between((current_time - self.context_timedelta), current_time, closed="right")
            assert historic_measurements.select(context_expr.sum()).item() == self.n_context
        elif isinstance(historic_measurements, pd.DataFrame):
            context_expr = (historic_measurements["time"] > (current_time - self.context_timedelta)) & (historic_measurements["time"] <= current_time) 
            assert len(historic_measurements.loc[context_expr, :].index) == self.n_context
            historic_measurements = pl.DataFrame(historic_measurements)
        
        executor = ProcessPoolExecutor()
        
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
    def __init__(self, dim_x=1, dim_z=1, **kwargs): # TODO need this st cross validation will work in Optuna objective function, ideally clone could capture dim_x, dim_z automatically
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.set_params(**kwargs)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X=None):
        if X is None:
            super().predict()
        else:
            pred = []
            self.x = X[:1]
            # for s in range(X.shape[0]): # loop over samples
                # pred.append([])
            for i in range(X.shape[0]): # loop over history and update matrices
                z = X[i:i+1]
                super().predict()
                self.update(z)
                pred.append(self.x[0])
            return np.array(pred)
    
    def set_params(self, **kwargs):
        if "R" in kwargs:
            self.R = kwargs["R"]
        if "H" in kwargs:
            self.H = kwargs["H"]
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
            
    def get_params(self, deep: bool):
        return {
            "R": self.R,
            "H": self.H,
            "Q": self.Q
        }

@dataclass
class KalmanFilterForecast(WindForecast):
    """Wind speed component forecasting using Kalman filtering."""
    kf_kwargs: dict
    # def read_measurements(self):
    #     pass
    
    def __post_init__(self):
        super().__post_init__()
        self.model = defaultdict(KalmanFilterForecast.create_model)
        self.scaler = defaultdict(KalmanFilterForecast.create_scaler)
        
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
        y_train = historic_measurements[self.n_context:, 0]
        
        # X_train = np.vstack([
        #     historic_measurements[i:i+self.n_context, 0]
        #     for i in range(historic_measurements.shape[0] - self.n_context)
        # ])
        
        # y_train = historic_measurements[self.n_context:, 0]
        
        return X_train, y_train
    
    def get_params(self, trial):
        return {
            "R": np.array([[trial.suggest_float("R", 1e-3, 1e2, log=True)]]),
            "H": np.array([[trial.suggest_float("H", 1e-3, 1e2, log=True)]]),
            "Q": np.array([[trial.suggest_float("Q", 1e-3, 1e2, log=True)]])
        }
       
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        pred_slice = self.get_pred_interval(current_time)
        pred = defaultdict(list)
        outputs = self._get_ws_cols(historic_measurements)
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(pd.DataFrame)
        
        executor = ProcessPoolExecutor() 
        with executor as train_exec:
            
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
class MLForecast(WindForecast):
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
        self.distr_predictor = estimator.create_predictor(transformation, model, 
                                                          forecast_generator=DistributionForecastGenerator(estimator.distr_output))
        self.sample_predictor = estimator.create_predictor(transformation, model, 
                                                           forecast_generator=SampleForecastGenerator())
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
    
    def _generate_test_data(self, historic_measurements: pl.DataFrame):
        # resample data to frequency model was trained on
            
        if self.data_module.freq != self.measurements_dt:
            if self.measurements_dt > self.data_module.freq:
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
    
    def predict_sample(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time, n_samples: int):
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(pd.DataFrame)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        historic_measurements = historic_measurements.with_columns([(cs.starts_with(col) * self.norm_scale[c]) 
                                                    + self.norm_min[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        test_data = self._generate_test_data(historic_measurements)
            
        pred = self.sample_predictor.predict(test_data, num_samples=n_samples, output_distr_params=False)
        
        if self.data_module.per_turbine_target:
            # TODO test
            pred_df = pl.concat([pl.DataFrame(
                data={
                    **{"time": np.tile(pred.index.to_timestamp(), (n_samples,)),
                        "sample": np.repeat(np.arange(n_samples), (turbine_pred.prediction_length,))},
                    **{col: turbine_pred.samples[:, :, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).rename(columns={output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                              for output_type in self.data_module.target_cols}).sort_values(["sample", "time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
        else:
            pred = next(pred)
            # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
            pred_df = pl.DataFrame(
                data={
                    # "turbine_id": pred_turbine_id,
                    **{"time": np.tile(pred.index.to_timestamp(), (n_samples,)),
                       "sample": np.repeat(np.arange(n_samples), (pred.prediction_length,))},
                    **{col: pred.samples[:, :, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).sort(by=["sample", "time"])
        
        # denormalize data 
        pred_df = pred_df.with_columns([(cs.starts_with(col) - self.norm_min[c]) 
                                                    / self.norm_scale[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        
        # check if the data that trained the model differs from the frequency of historic_measurments
        if self.data_module.freq != self.measurements_dt:
            # resample historic measurements to historic_measurements frequency and return as pandas dataframe
            if self.measurements_dt < self.data_module.freq:
                pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_dt))\
                                                              .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                pred_df = pred_df.upsample(time_column="time", every=self.measurements_dt).fill_null(strategy="forward")
        
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()

    def predict_point(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time): 
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(pd.DataFrame)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        historic_measurements = historic_measurements.with_columns([(cs.starts_with(col) * self.norm_scale[c]) 
                                                    + self.norm_min[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        test_data = self._generate_test_data(historic_measurements)
            
        pred = self.distr_predictor.predict(test_data, num_samples=1, output_distr_params=True)
        
        if self.data_module.per_turbine_target:
            # TODO test
            pred_df = pl.concat([pl.DataFrame(
                data={
                    **{"time": pred.index.to_timestamp()},
                    **{col: turbine_pred.distribution.loc[:, c].flatten() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).rename(columns={output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                              for output_type in self.data_module.target_cols}).sort_values(["time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
        else:
            pred = next(pred)
            # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
            pred_df = pl.DataFrame(
                data={
                    **{"time": pred.index.to_timestamp()},
                    **{col: pred.distribution.loc[:, c].cpu().numpy() for c, col in enumerate(self.data_module.target_cols)}
                }
            ).sort(by=["time"])
        
        # denormalize data 
        pred_df = pred_df.with_columns([(cs.contains(col) - self.norm_min[c]) 
                                                    / self.norm_scale[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        
        # check if the data that trained the model differs from the frequency of historic_measurments
        if self.data_module.freq != self.measurements_dt:
            # resample historic measurements to historic_measurements frequency and return as pandas dataframe
            if self.measurements_dt < self.data_module.freq:
                pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_dt))\
                                                              .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                pred_df = pred_df.upsample(time_column="time", every=self.measurements_dt).fill_null(strategy="forward")
        
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()


    def predict_distr(self, historic_measurements: Union[pd.DataFrame, pl.DataFrame], current_time):
        
        if isinstance(historic_measurements, pd.DataFrame):
            historic_measurements = pl.DataFrame(pd.DataFrame)
            return_pl = False
        else:
            return_pl = True
            
        # normalize historic measurements
        historic_measurements = historic_measurements.with_columns([(cs.starts_with(col) * self.norm_scale[c]) 
                                                    + self.norm_min[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        test_data = self._generate_test_data(historic_measurements)
            
        pred = self.distr_predictor.predict(test_data, num_samples=1, output_distr_params=True)
        
        if self.data_module.per_turbine_target:
            # TODO test
            pred_df = pl.concat([pl.DataFrame(
                data={
                    **{"time": pred.index.to_timestamp()},
                    **{f"loc_{col}": turbine_pred.distribution.loc[:, c].flatten() for c, col in enumerate(self.data_module.target_cols)},
                    **{f"sd_{col}": np.sqrt(turbine_pred.distribution.cov_diag[:, c].flatten()) for c, col in enumerate(self.data_module.target_cols)}
                }
            ).rename(columns={output_type: f"{output_type}_{self.data_module.static_features.iloc[t]['turbine_id']}" 
                              for output_type in self.data_module.target_cols}).sort_values(["time"]) for t, turbine_pred in enumerate(pred)], how="horizontal")
        else:
            pred = next(pred)
            # pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(pred.prediction_length)])
            pred_df = pl.DataFrame(
                data={
                    **{"time": pred.index.to_timestamp()},
                    **{f"loc_{col}": pred.distribution.loc[:, c].cpu().numpy() for c, col in enumerate(self.data_module.target_cols)},
                    **{f"sd_{col}": np.sqrt(pred.distribution.cov_diag[:, c].cpu().numpy()) for c, col in enumerate(self.data_module.target_cols)}
                }
            ).sort(by=["time"])
        
        # denormalize data 
        pred_df = pred_df.with_columns([(cs.contains(col) - self.norm_min[c]) 
                                                    / self.norm_scale[c] 
                                                    for c, col in enumerate(self.norm_min_cols)])
        
        # check if the data that trained the model differs from the frequency of historic_measurments
        if self.data_module.freq != self.measurements_dt:
            # resample historic measurements to historic_measurements frequency and return as pandas dataframe
            if self.measurements_dt < self.data_module.freq:
                pred_df = pred_df.with_columns(time=pl.col("time").dt.round(self.measurements_dt))\
                                                              .group_by("time").agg(cs.numeric().mean()).sort("time")
            else:
                pred_df = pred_df.upsample(time_column="time", every=self.measurements_dt).fill_null(strategy="forward")
        
        if return_pl: 
            return pred_df
        else:
            return pred_df.to_pandas()


if __name__ == "__main__":
    import yaml
    import os
    import argparse
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(prog="ModelTuning")
    parser.add_argument("-mcnf", "--model_config", type=str)
    parser.add_argument("-dcnf", "--data_config", type=str)
    parser.add_argument("-m", "--model", type=str, choices=["perfect", "persistent", "svr", "kf", "informer", "autoformer", "spacetimeformer"], required=True)
    parser.add_argument("-pt", "--prediction_type", type=str, choices=["point", "sample", "distribution"], default="point")
    args = parser.parse_args()
    
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
    
    prediction_timedelta = model_config["dataset"]["prediction_length"] * pd.Timedelta(model_config["dataset"]["resample_freq"]).to_pytimedelta()
    context_timedelta = model_config["dataset"]["context_length"] * pd.Timedelta(model_config["dataset"]["resample_freq"]).to_pytimedelta()
    historic_measurements_limit = datetime.timedelta(minutes=30)
    
    # TODO in GreedyController replace historic_measurements with future_predictions (if wind_forecast is not None) to compute wind_dirs (before passing to lpf)
    #      and compute optimal yaw angles and power outputs with ControlledFlorisInterface
    #      then compare optimal yaw angles and power outputs to those computed without wind_forecast
    
    ## GET TRUE WIND FIELD
    # pull ws_horz, ws_vert, nacelle_direction, normalization_consts from awaken data and run for ML, SVR
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
    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
    
    if True:
        true_wf_long = DataInspector.unpivot_dataframe(true_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                turbine_signature=data_config["turbine_signature"])
        WindForecast.plot_turbine_data(long_df=true_wf_long, fig_dir="./")
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
    if args.model == "perfect":
        perfect_forecast = PerfectForecast(
            measurements_dt=wind_dt,
            prediction_timedelta=prediction_timedelta,
            context_timedelta=context_timedelta,
            true_wind_field=true_wf,
        )
        perfect_forecast_wf = perfect_forecast.predict_point(historic_measurements, current_time)
        perfect_forecast_wf = perfect_forecast_wf.with_columns(data_type=pl.lit("Forecast"))
        id_vars = perfect_forecast_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns 
        perfect_forecast_wf = DataInspector.unpivot_dataframe(perfect_forecast_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], 
                                                    variable_name="feature", value_name="value")
    
        WindForecast.plot_forecast(perfect_forecast_wf, true_wf_long)
     
    ## GENERATE PERSISTENT PREVIEW
    elif args.model == "persistent":
        persistent_forecast = PersistentForecast(measurements_dt=wind_dt,
                                            prediction_timedelta=prediction_timedelta,
                                            context_timedelta=context_timedelta)
        
        persistent_forecast_wf = persistent_forecast.predict_point(historic_measurements, current_time)
        persistent_forecast_wf = persistent_forecast_wf.with_columns(data_type=pl.lit("Forecast"))
        id_vars = persistent_forecast_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        persistent_forecast_wf = DataInspector.unpivot_dataframe(persistent_forecast_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], 
                                                    variable_name="feature", value_name="value")
        
        WindForecast.plot_forecast(persistent_forecast_wf, true_wf_long) 
    
    ## GENERATE SVR PREVIEW
    elif args.model == "svr":
        svr_forecast = SVRForecast(measurements_dt=wind_dt,
                                prediction_timedelta=prediction_timedelta,
                                context_timedelta=context_timedelta,
                                svr_kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200))
        # TODO set parameters from optuna storage
        
        svr_forecast_wf = svr_forecast.predict_point(historic_measurements, current_time)
        svr_forecast_wf = svr_forecast_wf.with_columns(data_type=pl.lit("Forecast"))
        id_vars = svr_forecast_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        svr_forecast_wf = DataInspector.unpivot_dataframe(svr_forecast_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], variable_name="feature", value_name="value")
        
        
        WindForecast.plot_forecast(svr_forecast_wf, true_wf_long)
    
    ## GENERATE KF PREVIEW 
    elif args.model == "kf":
        kf_forecast = KalmanFilterForecast(measurements_dt=wind_dt,
                                        prediction_timedelta=prediction_timedelta,
                                        context_timedelta=context_timedelta,
                                        kf_kwargs=dict(H=np.array([1])))
        
        kf_forecast_wf = kf_forecast.predict_point(historic_measurements, current_time)
        kf_forecast_wf = kf_forecast_wf.with_columns(data_type=pl.lit("Forecast"))
        id_vars = kf_forecast_wf.select(~(cs.starts_with("ws_horz") | cs.starts_with("ws_vert") | cs.starts_with("nd_cos") | cs.starts_with("nd_sin"))).columns
        kf_forecast_wf = DataInspector.unpivot_dataframe(kf_forecast_wf, 
                                                    value_vars=["nd_cos", "nd_sin", "ws_horz", "ws_vert"], 
                                                    turbine_signature=data_config["turbine_signature"])\
                                            .unpivot(index=["turbine_id"] + id_vars, on=["ws_horz", "ws_vert"], variable_name="feature", value_name="value")
    
        WindForecast.plot_forecast(kf_forecast_wf, true_wf_long)
    
    ## GENERATE ML PREVIEW
    elif args.model in ["informer", "autoformer", "spacetimeformer", "tactis"]:
        ml_forecast = MLForecast(measurements_dt=wind_dt,
                                prediction_timedelta=prediction_timedelta,
                                context_timedelta=context_timedelta,
                                estimator_class=globals()[f"{args.model.capitalize()}Estimator"],
                                lightning_module=globals()[f"{args.model.capitalize()}LightningModule"],
                                distr_output=globals()[model_config["model"]["distr_output"]["class"]],
                                model_config_key=args.model,
                                model_checkpoint_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/checkpoints/lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt",
                                model_config_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml")
        PREDICT_TYPE = "point"
        
        if PREDICT_TYPE == "distribution":
            ml_forecast_wf =  ml_forecast.predict_distr(historic_measurements, current_time)
        elif PREDICT_TYPE == "sample":
            ml_forecast_wf =  ml_forecast.predict_sample(historic_measurements, current_time, n_samples=50)
        elif PREDICT_TYPE == "point":
            ml_forecast_wf =  ml_forecast.predict_point(historic_measurements, current_time)     
        
        ml_forecast_wf = ml_forecast_wf.with_columns(data_type=pl.lit("Forecast"))
        id_vars = ml_forecast_wf.select(~(cs.contains("ws_horz") | cs.contains("ws_vert") | cs.contains("nd_cos") | cs.contains("nd_sin"))).columns
        
        ml_forecast_wf = DataInspector.unpivot_dataframe(
            ml_forecast_wf, 
            value_vars=[f"{pf}_{sf}" for pf in ["loc", "sd"] for sf in ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]], 
            turbine_signature=data_config["turbine_signature"])\
    .unpivot(index=["turbine_id"] + id_vars, on=["loc_ws_horz", "loc_ws_vert", "sd_ws_horz", "sd_ws_vert"], variable_name="feature", value_name="value")
        
        WindForecast.plot_forecast(ml_forecast_wf, true_wf_long, prediction_type=PREDICT_TYPE, per_turbine_target=ml_forecast.data_module.per_turbine_target)
    
    print("here")