"""Module for wind speed component forecasting and preview functionality."""

class WindPreview():
    """Wind speed component forecasting module that provides various prediction methods."""
    context_length: int
    prediction_length: int
    probabilistic: bool
    output_distr: None
    
    def read_measurements(self):
        """_summary_
        Read in new measurements, add to internal container.
        """
        raise NotImplementedError()

    def predict_sample(self, n_samples: int):
        """_summary_
        Predict a given number of samples for each time step in the horizon
        Args:
            n_samples (int): _description_
        """
        raise NotImplementedError()

    def predict_point(self):
        """_summary_
        Make a point prediction (e.g. the mean prediction) for each time step in the horizon
        """
        raise NotImplementedError()

    def predict_distr(self):
        """_summary_
        Generate the parameters of the forecasted distribution
        """
        raise NotImplementedError()


class PerfectPreview(WindPreview):
    """Perfect wind speed component forecasting that assumes exact knowledge of future wind speeds."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass


class PersistentPreview(WindPreview):
    """Wind speed component forecasting using persistence model that assumes future values equal current value."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

class GaussianPreview(WindPreview):
    """Wind speed component forecasting using Gaussian parameterization."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

class SVRPreview(WindPreview):
    """Wind speed component forecasting using Support Vector Regression."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

class KalmanFilterPreview(WindPreview):
    """Wind speed component forecasting using Kalman filtering."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass

class MLPreview(WindPreview):
    """Wind speed component forecasting using machine learning models."""
    
    def read_measurements(self):
        pass
    
    def predict_sample(self, n_samples: int):
        pass

    def predict_point(self):
        pass

    def predict_distr(self):
        pass