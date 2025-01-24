"""Module for wind speed component forecasting and preview functionality."""

from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import yaml

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
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from wind_forecasting.preprocessing.data_module import DataModule

@dataclass
class WindPreview:
    """Wind speed component forecasting module that provides various prediction methods."""
    context_length: int
    prediction_length: int # TODO ensure that these are given in terms of wind_dt, and transformed as necessary for models trained on different values of dt
    freq: pd.Timedelta
    
    # def read_measurements(self):
    #     """_summary_
    #     Read in new measurements, add to internal container.
    #     """
    #     raise NotImplementedError()

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
        return pd.date_range(start=current_time, periods=self.prediction_length, freq=self.freq)

    @staticmethod
    def plot_preview(preview_wf, true_wf):
        fig = plt.figure()
        ax = sns.lineplot(data=true_wf, x="time", y="wind_speed", hue="wind_component", style="data_type", dashes=[[1, 0]])
        ax = sns.lineplot(data=preview_wf, x="time", y="wind_speed", hue="wind_component", style="data_type", dashes=[[4, 4]], marker="o")
        ax.set(xlabel="Time [s]", ylabel="Wind Speed [m/s]", xlim=(preview_wf["time"].iloc[0], preview_wf["time"].iloc[-1]))
        h, l = ax.get_legend_handles_labels()
        labels = ["ws_horz", "ws_vert", "True", "Preview"]
        handles = [h[l.index(label)] for label in labels]
        ax.legend(handles, labels, ncol=2)

@dataclass
class PerfectPreview(WindPreview):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    true_wind_field: pd.DataFrame
    col_mapping: dict
    
    # def read_measurements(self):
    #     pass
    
    def __post_init__(self):
        if all(col in self.true_wind_field.columns for col in self.col_mapping.values()):
            self.map_columns = False
        else:
            self.map_columns = True
    
    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        pred_slice = self.get_pred_interval(current_time)
        sub_df = self.true_wind_field.rename(columns=self.col_mapping) if self.map_columns else self.true_wind_field 
        sub_df = sub_df.loc[sub_df["time"].isin(pred_slice)]
        assert len(sub_df.index) == self.prediction_length
        return pd.DataFrame({k: sub_df[k] for k in self.col_mapping.values()})

@dataclass
class PersistentPreview(WindPreview):
    """Wind speed component forecasting using persistence model that assumes future values equal current value."""
    
    # def read_measurements(self):
    #     pass
    
    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        pred_slice = self.get_pred_interval(current_time)
        assert (historic_measurements["time"] == current_time).any()
        return pd.DataFrame({"time": pred_slice})\
                 .assign(u=historic_measurements.iloc[-1]["ws_horz"], 
                         v=historic_measurements.iloc[-1]["ws_vert"])

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
        self.model = {output: make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVR(**self.svr_kwargs)) for output in ["ws_horz", "ws_vert"]} 
    
    # def read_measurements(self):
    #     pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
         # TODO include yaw angles in inputs? 
        # TODO reshape x_train to include context_len 
        pred_slice = self.get_pred_interval(current_time)
        pred = defaultdict(list)
        for output in ["ws_horz", "ws_vert"]:
            X_train = np.ascontiguousarray(np.vstack([
               historic_measurements.iloc[i:i+self.context_length][output] 
               for i in range(len(historic_measurements) - self.context_length)
            ]))
            
            # X_train = np.ascontiguousarray(historic_measurements.iloc[:-self.context_length][output])
            y_train = np.ascontiguousarray(historic_measurements.iloc[self.context_length:][output])
            self.model[output].fit(X_train, y_train)
            
            X_pred = np.ascontiguousarray(historic_measurements.iloc[-self.context_length:][output])[np.newaxis, :]
            # X_pred = np.ascontiguousarray(historic_measurements.iloc[-1][output])[:, np.newaxis]
            
            for _ in range(self.prediction_length):
                y_pred = self.model[output].predict(X_pred)
                X_pred = np.hstack([X_pred[:, 1:], y_pred[:, np.newaxis]])
                # X_pred = np.ascontiguousarray(pred[output][-1])[:, np.newaxis]
            
            pred[output] = X_pred.flatten()[-self.prediction_length:] 
            assert len(pred[output]) == self.prediction_length
            
        return pd.DataFrame({"time": pred_slice}).assign(**pred)

    def predict_distr(self):
        pass

@dataclass
class KalmanFilterPreview(WindPreview):
    """Wind speed component forecasting using Kalman filtering."""
    kf_kwargs: dict
    # def read_measurements(self):
    #     pass
    
    def __post_init__(self):
        self.model = {}
        for output in ["ws_horz", "ws_vert"]:
            self.model[output] = KalmanFilter(dim_x=1, dim_z=1)
            self.model[output].R = np.array([[0.1]])
            self.model[output].H = np.array([[1]])
            self.model[output].Q = np.array([[0.13]])
        
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        pred_slice = self.get_pred_interval(current_time)
        pred = defaultdict(list)
        for output in ["ws_horz", "ws_vert"]:
            self.model[output].x = historic_measurements.iloc[-1:][output].to_numpy().flatten()
            for i in range(len(historic_measurements) - (self.context_length + self.prediction_length)):
                z = historic_measurements.iloc[i:i+1][output].to_numpy()
                self.model[output].predict()
                self.model[output].update(z)
            
            z = historic_measurements.iloc[-1:][output].to_numpy()
            for i in range(self.prediction_length):
                self.model[output].predict()
                self.model[output].update(z)
                z = self.model[output].x
                pred[output].append(z[0])
        
        return pd.DataFrame({"time": pred_slice}).assign(**pred) 

    def predict_distr(self):
        pass

@dataclass
class MLPreview(WindPreview):
    """Wind speed component forecasting using machine learning models."""
    model_checkpoint_path: Path
    model_config_path: Path
    model_config_key: str # must be key in given model_config_path
    estimator_class: PyTorchLightningEstimator
    lightning_module: pl.LightningModule
    
    def __post_init__(self):
        with open(self.model_config_path, 'r') as file:
            config  = yaml.safe_load(file)
            
        self.data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                per_turbine_target=config["dataset"]["per_turbine_target"], dtype=None)
        
        # logging.info("Declaring estimator")
        estimator = self.estimator_class(
            freq=data_module.freq, 
            prediction_length=data_module.prediction_length,
            context_length=data_module.context_length,
            num_feat_dynamic_real=data_module.num_feat_dynamic_real, 
            num_feat_static_cat=data_module.num_feat_static_cat,
            cardinality=data_module.cardinality,
            num_feat_static_real=data_module.num_feat_static_real,
            input_size=data_module.num_target_vars,
            scaling=False,
            batch_size=config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=config["trainer"].setdefault("limit_train_batches", 50), # TODO set this to be arbitrarily high st limit train_batches dominates
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=data_module.context_length, min_future=data_module.prediction_length), # TODO should be context_len + max(seq_len) to avoid padding..
            validation_sampler=ValidationSplitSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
            activation="relu",
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=self.distr_output(dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=config["trainer"],
            **config["model"][self.model_config_key]
        )
        
        self.normalization_consts = pd.read_csv(config["dataset"]["normalization_consts_path"], index_col=None)
        model = self.lightning_module.load_from_checkpoint(self.model_checkpoint)
        transformation = estimator.create_transformation(use_lazyframe=False)
        self.predictor = estimator.create_predictor(transformation, model, 
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))
    
    # def read_measurements(self):
    #     pass
    
    def predict_sample(self, historic_measurements: pd.DataFrame, current_time, n_samples: int):
        # resample data to frequency model was trained on
        if self.model_freq != self.freq: # TODO check that we are comparing same types here
            resampled_measurements = historic_measurements.resample(self.model_freq).mean()
        
        # TODO normalize data to -1, 1 using saved normalization consts
        scaler = MinMaxScaler()
        scaler.n_features_in_ = self.data_module.num_target_cols
        # TODO make sure ordered as resampled_measurements before normalizing, and make sure resampled measurements ordered by data_module.target_cols and data_module.dynamic_real_features 
        scaler.data_min_ = self.normalization_consts[[col for col in self.normalization_consts.columns if "min" in col]]
        scaler.data_max_ = self.normalization_consts[[col for col in self.normalization_consts.columns if "max" in col]]
        # test_data must be iterable where each item generated is a dict with keys start, target, item_id, and feat_dynamic_real
        # this should include measurements at all turbines
        if self.data_module.per_turbine_target:
            test_data = ({}"item_id": f"TURBINE{turbine_id}",
                "start": pd.Period(resampled_measurements.iloc[0]["time"], freq=self.data_module.freq), 
                "target": resampled_measurements[self.data_module.target_cols].to_numpy().T, 
                "feat_dynamic_real": resampled_measurements[self.data_module.dynamic_real_cols].to_numpy().T
            } for turbine_id in self.data_module.target_suffixes)
        else:
            test_data = {"start": pd.Period(resampled_measurements.iloc[0]["time"], freq=self.data_module.freq), 
                        "target": resampled_measurements[self.data_module.target_cols].to_numpy().T, 
                        "feat_dynamic_real": resampled_measurements[self.data_module.dynamic_real_cols].to_numpy().T
                        }
            
        pred = self.predictor.predict(test_data, num_samples=n_samples, output_distr_params=False)
        # TODO resample historic measurements to model frequency and return as pandas dataframe

    def predict_point(self, historic_measurements: pd.DataFrame, current_time):
        
        pass

    def predict_distr(self, historic_measurements: pd.DataFrame, current_time):
        self.predictor.predict(test_data.input, num_samples=1, output_distr_params=True)

if __name__ == "__main__":
    import yaml
    import os
    import whoc
    from glob import glob
    from whoc.wind_field.plotting import plot_ts
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    prediction_length = 360
    context_length = 1200
    
    ## GET TRUE WIND FIELD
    # if wind field data exists, get it
    wind_field_dir = os.path.join(os.path.dirname(whoc.__file__), "..", "examples", "floris_case_studies", "wind_field_data", "raw_data")        
    wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
    true_wfs = []
    if os.path.exists(wind_field_dir) and len(wind_field_filenames):
        for fn in wind_field_filenames:
            true_wfs.append(pd.read_csv(os.path.join(wind_field_dir, fn), index_col=0))
    else:
        raise Exception(f"No wind field files found in {wind_field_dir}.")
    
    plot_ts(pd.concat(true_wfs), wind_field_dir)
    # plt.savefig(os.path.join(wind_field_config["fig_dir"], "wind_field_ts.png"))
    
    # true wind disturbance time-series
    case_idx = 0
    true_wf = true_wfs[case_idx]
    
    wind_dt = pd.Timedelta(value=true_wf["Time"].diff().iloc[-1], unit='s')
    true_wf["Time"] = pd.to_datetime(true_wf["Time"], unit="s")
    true_wf = true_wf.rename(columns={"Time": "time", "FreestreamWindSpeedU": "ws_horz", "FreestreamWindSpeedV": "ws_vert"})
    true_wf = true_wf.assign(data_type="True")
    true_wf_plot = pd.melt(true_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    historic_measurements = true_wf.iloc[:int(60*60*0.5 // wind_dt.total_seconds())]
    current_time = historic_measurements.iloc[-1]["time"] 
    assert context_length + prediction_length <= len(true_wf)
    
    ## GENERATE PERFECT PREVIEW 
    perfect_preview = PerfectPreview(
        freq=wind_dt,
        prediction_length=prediction_length,
        context_length=context_length,
        true_wind_field=true_wf,
        col_mapping={"Time": "time", "FreestreamWindSpeedU": "ws_horz", "FreestreamWindSpeedV": "ws_vert"}
    )
    perfect_preview_wf = perfect_preview.predict_point(historic_measurements, current_time)
    perfect_preview_wf = perfect_preview_wf.assign(data_type="Preview")
    perfect_preview_wf = pd.melt(perfect_preview_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    WindPreview.plot_preview(perfect_preview_wf, true_wf_plot)
     
    ## GENERATE PERSISTENT PREVIEW
    persistent_preview = PersistentPreview(freq=wind_dt,
                                           prediction_length=prediction_length,
                                           context_length=context_length)
    
    persistent_preview_wf = persistent_preview.predict_point(historic_measurements, current_time)
    persistent_preview_wf = persistent_preview_wf.assign(data_type="Preview")
    persistent_preview_wf = pd.melt(persistent_preview_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    WindPreview.plot_preview(persistent_preview_wf, true_wf_plot) 
    
    ## GENERATE SVR PREVIEW
    # TODO tune SVR parameters
    # TODO add UQ with five-fold cross-validation 
    svr_preview = SVRPreview(freq=wind_dt,
                             prediction_length=prediction_length,
                             context_length=context_length,
                             svr_kwargs=dict(kernel="rbf", C=1.0, degree=3, gamma="auto", epsilon=0.1, cache_size=200))
    
    svr_preview_wf = svr_preview.predict_point(historic_measurements, current_time)
    svr_preview_wf = svr_preview_wf.assign(data_type="Preview")
    svr_preview_wf = pd.melt(svr_preview_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    WindPreview.plot_preview(svr_preview_wf, true_wf_plot)
    
    ## GENERATE KF PREVIEW 
    # kf_preview = KalmanFilterPreview(freq=wind_dt,
    #                                  prediction_length=prediction_length,
    #                                  context_length=context_length,
    #                                  probabilistic=True,
    #                                  distr_output=None,
    #                                  kf_kwargs=dict())
    
    # kf_preview_wf = kf_preview.predict_point(historic_measurements, current_time)
    # kf_preview_wf = kf_preview_wf.assign(data_type="Preview")
    # kf_preview_wf = pd.melt(kf_preview_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    # WindPreview.plot_preview(kf_preview_wf, true_wf_plot)
    
    ## GENERATE ML PREVIEW
    ml_preview = MLPreview(freq=wind_dt,
                             prediction_length=prediction_length,
                             context_length=context_length,
                             estimator_class=InformerEstimator,
                             lightning_module=InformerLightningModule,
                             model_config_key="informer",
                             model_checkpoint_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/checkpoints/lightning_logs/version_172/checkpoints/epoch=2-step=300.ckpt",
                             model_config_path="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/training_inputs_aoifemac_awaken.yaml")
    
    ml_preview_wf =  ml_preview.predict_point(historic_measurements, current_time)
    ml_preview_wf = ml_preview_wf.assign(data_type="Preview")
    ml_preview_wf = pd.melt(ml_preview_wf, id_vars=["time", "data_type"], value_vars=["ws_horz", "ws_vert"], var_name="wind_component", value_name="wind_speed")
    
    WindPreview.plot_preview(ml_preview_wf, true_wf_plot)
    
    ## TODO other baseline eg. LSTM or DeepAR from gluonts
    print("here")
    
    # # TODO pass the checkpoint filepath and the model
    # ml_preview = MLPreview(prediction_length=prediction_length,
    #                                  context_length=context_length,
    #                                  probabilistic=True,
    #                                  distr_output=LowRankMultivariateNormalOutput)