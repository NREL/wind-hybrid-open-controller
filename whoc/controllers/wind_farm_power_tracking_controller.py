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

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Set initial conditions
        self.controls_dict = {"power_setpoints": [POWER_SETPOINT_DEFAULT] * self.n_turbines}

        # For startup


    def compute_controls(self):
        if "wind_power_reference" in self.measurements_dict:
            farm_power_reference = self.measurements_dict["wind_power_reference"]
        else:
            farm_power_reference = POWER_SETPOINT_DEFAULT
        
        self.turbine_power_references(farm_power_reference=farm_power_reference)

    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """

        # Split farm power reference among turbines and set "no value" for yaw angles (Floris not
        # compatible with both power_setpoints and yaw_angles).
        self.controls_dict = {
            "power_setpoints": [farm_power_reference/self.n_turbines]*self.n_turbines,
            "yaw_angles": [-1000]*self.n_turbines
        }

        return None


class WindFarmPowerTrackingController(WindFarmPowerDistributingController):
    """
    Based on controller developed under A2e2g project. Proportional control only---
    all integral action is disabled.

    Inherits from WindFarmPowerDistributingController.
    """

    def __init__(self, interface, input_dict, proportional_gain=1, verbose=False):
        super().__init__(interface, input_dict, verbose=verbose)

        # No integral action for now. beta and omega_n not used.
        # beta=0.7
        # omega_n=0.01
        # integral_gain=0 

        self.K_p = proportional_gain * 1/self.n_turbines
        # self.K_i = integral_gain *(4*beta*omega_n)

        # Initialize controller (only used for integral action)
        # self.e_prev = 0
        # self.u_prev = 0
        # self.u_i_prev = 0
        # self.ai_prev = [0.33]*self.n_turbines # TODO: different method for anti-windup?
        # self.n_saturated = 0 

    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """
        
        turbine_current_powers = self.measurements_dict["turbine_powers"]
        farm_current_power = np.sum(turbine_current_powers)
        farm_current_error = farm_power_reference - farm_current_power

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

        turbine_power_setpoints = np.array(turbine_current_powers) + delta_P_ref
        
        # set "no value" for yaw angles (Floris not compatible with both 
        # power_setpoints and yaw_angles)
        self.controls_dict = {
            "power_setpoints": list(turbine_power_setpoints),
            "yaw_angles": [-1000]*self.n_turbines
        }

        # Store error, control (only needed for integral action, which is disabled)
        # self.e_prev = farm_current_error
        # self.u_prev = u
        # self.u_i_prev = u_i

        return None
