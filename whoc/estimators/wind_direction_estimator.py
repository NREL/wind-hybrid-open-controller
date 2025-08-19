from whoc.estimators.estimator_base import EstimatorBase
from whoc.interfaces.interface_base import InterfaceBase


class WindDirectionPassthroughEstimator(EstimatorBase):
    """
    WindDirectionPassthroughEstimator is a simple estimator that passes through the wind
    direction measurements without modification.
    """
    def __init__(self, interface: InterfaceBase, verbose: bool = False):
        super().__init__(interface, verbose=verbose)

    def compute_estimates(self, measurements_dict):
        # Simply pass through the wind directions as estimates
        return {"wind_directions": measurements_dict["wind_directions"]}
