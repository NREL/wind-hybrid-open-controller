import numpy as np

from hycon.controllers.controller_base import ControllerBase

# Default power setpoint in kW (meant to ensure power maximization)
POWER_SETPOINT_DEFAULT = 1e9


class WindFarmPowerDistributingController(ControllerBase):
    """
    Evenly distributes wind farm power reference between turbines without
    feedback on current power generation.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=False):
        """
        Constructor for WindFarmPowerDistributingController.

        Args:
            interface: Hycon Interface object for communication with the simulation environment.
            cname: Name of the controller, used for indexing into measurements and controls
                dictionaries. Should match the component name in the plant model.
            controller_parameters: Dictionary of controller parameters. See
                set_controller_parameters for details on expected controller parameters.
            verbose: Boolean flag for verbosity.
        """

        super().__init__(interface, cname, verbose=verbose)

        if self.cname in self.plant_parameters:
            self.n_turbines = self.plant_parameters[self.cname]["n_turbines"]
        else:
            self.n_turbines = self.plant_parameters["n_turbines"]
        self.turbines = range(self.n_turbines)

        # Ramp rate limit
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

        # Used for initialization purposes
        self._first_call = True

    def set_controller_parameters(self, ramp_rate_limit=None):
        """
        Set controller parameters for WindFarmPowerDistributingController.

        Args:
            ramp_rate_limit: Ramp rate limit for the controller (kW/s). Defaults to None, which
                corresponds to no ramp rate limit.
        """
        if ramp_rate_limit is None:
            ramp_rate_limit = np.inf
        elif ramp_rate_limit < 0:
            raise ValueError("ramp_rate_limit must be non-negative.")
        self.turbine_ramp_rate_limit = ramp_rate_limit / self.n_turbines

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

        return {self.cname: {"power_setpoints": turbine_power_setpoints.tolist()}}

    def apply_ramp_rate_limit(self, unclipped_setpoints):
        if self._first_call:
            # On first call, ignore ramp rate limit to allow controller to initialize
            turbine_power_setpoints = unclipped_setpoints
        else:
            turbine_power_setpoints = np.clip(
                unclipped_setpoints,
                self._setpoints_prev - self.turbine_ramp_rate_limit * self.dt,
                self._setpoints_prev + self.turbine_ramp_rate_limit * self.dt,
            )

        self._setpoints_prev = turbine_power_setpoints

        return turbine_power_setpoints


class WindFarmPowerTrackingController(WindFarmPowerDistributingController):
    """
    Based on controller developed under A2e2g project. Proportional control only---
    all integral action is disabled.

    Inherits from WindFarmPowerDistributingController.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=False):
        """
        Constructor for WindFarmPowerTrackingController.

        Args:
            interface: Hycon Interface object for communication with the simulation environment.
            cname: Name of the controller, used for indexing into measurements and controls
                dictionaries. Should match the component name in the plant model.
            controller_parameters: Dictionary of controller parameters. See
                set_controller_parameters for details on expected controller parameters.
            verbose: Boolean flag for verbosity.
        """
        super().__init__(interface, cname, verbose=verbose)

        # Using bad inheritance here, so will have to recheck ramp rate limit parameters
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self, proportional_gain=1.0, ramp_rate_limit=None):
        """
        Set controller parameters for WindFarmPowerTrackingController.

        Args:
            proportional_gain: Proportional gain for the controller. Defaults to 1.0.
            ramp_rate_limit: Ramp rate limit for the controller (kW/s). Defaults to None, which
                corresponds to no ramp rate limit.
        """
        if ramp_rate_limit is None:
            ramp_rate_limit = np.inf
        elif ramp_rate_limit < 0:
            raise ValueError("ramp_rate_limit must be non-negative.")
        self.turbine_ramp_rate_limit = ramp_rate_limit / self.n_turbines

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

        return {self.cname: {"power_setpoints": list(turbine_power_setpoints)}}
