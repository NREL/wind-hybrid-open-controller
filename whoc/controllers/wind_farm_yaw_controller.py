from __future__ import annotations

from whoc.controllers import YawSetpointPassthroughController
from whoc.controllers.controller_base import ControllerBase
from whoc.estimators import WindDirectionPassthroughEstimator
from whoc.estimators.estimator_base import EstimatorBase
from whoc.interfaces.interface_base import InterfaceBase


class WindFarmYawController(ControllerBase):
    """
    WindFarmYawController is a top-level controller that manages a combined wind estimator
    and yaw setpoint controller for a wind farm.
    """
    def __init__(
            self,
            interface: InterfaceBase,
            yaw_setpoint_controller: ControllerBase | None = None,
            wind_estimator: EstimatorBase | None = None,
            verbose: bool = False
        ):
        """
        Constructor for WindFarmYawController. 

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            input_dict (dict): Optional dictionary of input parameters.
            controller_parameters (dict): Optional dictionary of controller parameters.
            yaw_setpoint_controller (ControllerBase): Optional yaw controller to set control
                setpoints of individual wind turbines.
            wind_estimator (EstimatorBase): Optional wind estimator to provide wind direction
                estimates for individual turbines.
        """
        super().__init__(interface, verbose=verbose)

        # Establish defaults for yaw setpoint controller and wind estimator and store on self
        if yaw_setpoint_controller is None:
            yaw_setpoint_controller = YawSetpointPassthroughController(interface, verbose=verbose)
        if wind_estimator is None:
            wind_estimator = WindDirectionPassthroughEstimator(interface, verbose=verbose)
        self.yaw_setpoint_controller = yaw_setpoint_controller
        self.wind_estimator = wind_estimator

    def compute_controls(self, measurements_dict):
        estimates_dict = self.wind_estimator.compute_estimates(measurements_dict)
        controls_dict = self.yaw_setpoint_controller.compute_controls(estimates_dict)

        return controls_dict
