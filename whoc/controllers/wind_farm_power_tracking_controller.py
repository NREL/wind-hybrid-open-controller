import numpy as np

from whoc.controllers.controller_base import ControllerBase

# Default power setpoint in kW (meant to ensure power maximization)
POWER_SETPOINT_DEFAULT = 1e9 

class WindFarmPowerDistributingController(ControllerBase):
    """
    Evenly distributes wind farm power reference between turbines without 
    feedback on current power generation.
    """
    def __init__(self, interface, input_dict, verbose=False):
        super().__init__(interface, verbose=verbose)

        try:
            self.n_turbines = self.plant_parameters["n_turbines"]
        except AttributeError:
            self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

    def compute_controls(self, measurements_dict):
        if "wind_power_reference" in measurements_dict:
            farm_power_reference = measurements_dict["wind_power_reference"]
        else:
            farm_power_reference = POWER_SETPOINT_DEFAULT
        
        return self.turbine_power_references(
            farm_power_reference=farm_power_reference,
            turbine_powers=measurements_dict["wind_turbine_powers"]
        )

    def turbine_power_references(
            self,
            farm_power_reference=POWER_SETPOINT_DEFAULT,
            turbine_powers=None
        ):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """

        # Split farm power reference among turbines.
        controls_dict = {
            "wind_power_setpoints": [farm_power_reference/self.n_turbines]*self.n_turbines,
        }

        return controls_dict

class WindFarmPowerTrackingController(WindFarmPowerDistributingController):
    """
    Based on controller developed under A2e2g project. Proportional control only---
    all integral action is disabled.

    Inherits from WindFarmPowerDistributingController.
    """

    def __init__(
            self,
            interface, 
            input_dict,
            proportional_gain=1,
            ramp_rate_limit=None,
            verbose=False
        ):
        """
        Constructor for WindFarmPowerTrackingController.

        Args:
            interface: WHOC Interface object for communication with the simulation environment.
            input_dict: Dictionary containing input parameters for the controller.
            proportional_gain: Proportional gain for the controller.
            ramp_rate_limit: Ramp rate limit for the controller (kW/s). Defaults to None.
            verbose: Boolean flag for verbosity.
        """
        super().__init__(interface, input_dict, verbose=verbose)

        # Proportional gain
        self.K_p = proportional_gain * 1/self.n_turbines

        # Ramp rate limit
        self.ramp_rate_limit = ramp_rate_limit

    def turbine_power_references(
            self,
            farm_power_reference=POWER_SETPOINT_DEFAULT,
            turbine_powers=None
        ):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """
        
        farm_current_power = np.sum(turbine_powers)
        farm_current_error = farm_power_reference - farm_current_power

        # Apply ramp rate limit
        if self.ramp_rate_limit is not None:
            farm_current_error = np.clip(
                farm_current_error,
                farm_current_power - self.ramp_rate_limit * self.dt,
                farm_current_power + self.ramp_rate_limit * self.dt
            )

        self.n_saturated = 0 # TODO: determine whether to use gain scheduling
        if self.n_saturated < self.n_turbines:
            # with self.n_saturated = 0, gain_adjustment = 1
            gain_adjustment = self.n_turbines/(self.n_turbines-self.n_saturated)
        else:
            gain_adjustment = self.n_turbines
        K_p_gs = gain_adjustment*self.K_p
        #K_i_gs = gain_adjustment*self.K_i

        # Discretize and apply difference equation (trapezoid rule)
        u_p = K_p_gs*farm_current_error
        #u_i = self.dt/2*K_i_gs * (farm_current_error + self.e_prev) + self.u_i_prev

        # Apply integral anti-windup
        #eps = 0.0001 # Threshold for anti-windup
        #if (np.array(self.ai_prev) > 1/3-eps).all() or \
        #   (np.array(self.ai_prev) < 0+eps).all():
        #   u_i = 0
        
        u = u_p #+ u_i
        delta_P_ref = u

        turbine_power_setpoints = np.array(turbine_powers) + delta_P_ref

        controls_dict = {
            "wind_power_setpoints": list(turbine_power_setpoints),
        }

        # Store error, control (only needed for integral action, which is disabled)
        # self.e_prev = farm_current_error
        # self.u_prev = u
        # self.u_i_prev = u_i

        return controls_dict
