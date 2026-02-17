import numpy as np

from hycon.controllers.controller_base import ControllerBase

# Default power setpoint in kW (meant to ensure power maximization)
POWER_SETPOINT_DEFAULT = 1e9


class WindFarmPowerDistributingController(ControllerBase):
    """
    Evenly distributes wind farm power reference between turbines without
    feedback on current power generation.
    """

    def __init__(self, interface, input_dict, ramp_rate_limit=None, verbose=False):
        super().__init__(interface, verbose=verbose)

        # Pull plant parameters for ease of use
        self.cname = "wind_farm"

        if self.cname in self.plant_parameters:
            self.n_turbines = self.plant_parameters[self.cname]["n_turbines"]
        else:
            self.n_turbines = self.plant_parameters["n_turbines"]
        self.turbines = range(self.n_turbines)

        # Ramp rate limit
        if ramp_rate_limit is None:
            ramp_rate_limit = np.inf
        self.turbine_ramp_rate_limit = ramp_rate_limit / self.n_turbines

        # Used for initialization purposes
        self._first_call = True

    def compute_controls(self, measurements_dict):
        ref_in_lower_dict = (
            "power_reference" in measurements_dict[self.cname]
            and measurements_dict[self.cname]["power_reference"] is not None
        )
        ref_in_upper_dict = (
            "power_reference" in measurements_dict
            and measurements_dict["power_reference"] is not None
        )
        if ref_in_lower_dict and ref_in_upper_dict:
            raise KeyError(
                "Found 'power_reference' in both measurements_dict['"
                + self.cname
                + "'] and measurements_dict."
            )
        elif ref_in_lower_dict:
            farm_power_reference = measurements_dict[self.cname]["power_reference"]
        elif ref_in_upper_dict:
            farm_power_reference = measurements_dict["power_reference"]
        else:
            farm_power_reference = POWER_SETPOINT_DEFAULT

        turbine_power_setpoints = self.turbine_power_references(
            farm_power_reference=farm_power_reference,
            turbine_powers=measurements_dict[self.cname]["turbine_powers"],
        )

        self._first_call = False

        return turbine_power_setpoints

    def turbine_power_references(
        self, farm_power_reference=POWER_SETPOINT_DEFAULT, turbine_powers=None
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
        turbine_power_setpoints = np.array(
            [farm_power_reference / self.n_turbines] * self.n_turbines
        )

        # Apply ramp rate limit
        turbine_power_setpoints = self.apply_ramp_rate_limit(turbine_power_setpoints)

        controls_dict = {
            "power_setpoints": turbine_power_setpoints.tolist(),
        }

        return controls_dict

    def apply_ramp_rate_limit(self, unclipped_setpoints):
        if self._first_call:
            # On first call, ignore ramp rate limit to allow controller to initialize
            turbine_power_setpoints = unclipped_setpoints
        else:
            turbine_power_setpoints = np.clip(
                unclipped_setpoints,
                self._unclipped_prev - self.turbine_ramp_rate_limit * self.dt,
                self._unclipped_prev + self.turbine_ramp_rate_limit * self.dt,
            )

        self._unclipped_prev = turbine_power_setpoints

        return turbine_power_setpoints


class WindFarmPowerTrackingController(WindFarmPowerDistributingController):
    """
    Based on controller developed under A2e2g project. Proportional control only---
    all integral action is disabled.

    Inherits from WindFarmPowerDistributingController.
    """

    def __init__(
        self, interface, input_dict, proportional_gain=1, ramp_rate_limit=None, verbose=False
    ):
        """
        Constructor for WindFarmPowerTrackingController.

        Args:
            interface: Hycon Interface object for communication with the simulation environment.
            input_dict: Dictionary containing input parameters for the controller.
            proportional_gain: Proportional gain for the controller.
            ramp_rate_limit: Ramp rate limit for the controller (kW/s). Defaults to None.
            verbose: Boolean flag for verbosity.
        """
        super().__init__(interface, input_dict, ramp_rate_limit=ramp_rate_limit, verbose=verbose)

        # Proportional gain
        self.K_p = proportional_gain * 1 / self.n_turbines

    def turbine_power_references(
        self, farm_power_reference=POWER_SETPOINT_DEFAULT, turbine_powers=None
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

        self.n_saturated = 0  # TODO: determine whether to use gain scheduling
        if self.n_saturated < self.n_turbines:
            # with self.n_saturated = 0, gain_adjustment = 1
            gain_adjustment = self.n_turbines / (self.n_turbines - self.n_saturated)
        else:
            gain_adjustment = self.n_turbines
        K_p_gs = gain_adjustment * self.K_p

        # Discretize and apply difference equation (trapezoid rule)
        u = K_p_gs * farm_current_error

        delta_P_ref = u

        unclipped_setpoints = np.array(turbine_powers) + delta_P_ref

        # Apply ramp rate limit
        turbine_power_setpoints = self.apply_ramp_rate_limit(unclipped_setpoints)

        controls_dict = {
            "power_setpoints": list(turbine_power_setpoints),
        }

        return controls_dict
