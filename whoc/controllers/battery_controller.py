import numpy as np

from whoc.controllers.controller_base import ControllerBase


class BatteryController(ControllerBase):
    """
    Modifies power reference to consider battery degradation for single battery.

    In particular, ensures smoothness in battery reference signal to avoid rapid
    changes in power reference, which can lead to degradation.
    """
    def __init__(self, interface, input_dict, controller_parameters={}, verbose=True):
        super().__init__(interface, verbose)

        # Extract global parameters
        self.dt = input_dict["dt"]

        # Check that parameters are not specified both in input file
        # and in controller_parameters
        for cp in controller_parameters.keys():
            if cp in input_dict["controller"]:
                raise KeyError(
                    "Found key \""+cp+"\" in both input_dict[\"controller\"] and"
                    " in controller_parameters."
                )
        controller_parameters = {**controller_parameters, **input_dict["controller"]}
        self.set_controller_parameters(**controller_parameters)
        
        # Assumes one battery!
        battery_name = [k for k in input_dict["py_sims"] if "battery" in k][0]
        self.rated_power_charging = input_dict["py_sims"][battery_name]["charge_rate"] * 1e3
        self.rated_power_discharging = input_dict["py_sims"][battery_name]["discharge_rate"] * 1e3

        # Initialize controller internal state
        self.x = 0

    def set_controller_parameters(
        self,
        k_batt=0.1,
        clipping_thresholds=[0, 0, 1, 1],
        **_ # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set gains and threshold limits for BatteryController.

        Args:
            k_p_max: Maximum proportional gain. Defaults to 1.
            k_p_min: Minimum proportional gain. Defaults to None, which is
               a code for matching k_p_max.
        """        
        zeta = 2
        omega = 2 * np.pi * k_batt

        # Discrete-time, first-order state-space model of controller
        p = np.exp(-2 * zeta * omega * self.dt)
        self.a = p
        self.b = 1
        self.c = omega / (2 * zeta) * (1-p)/2 * (p + 1)
        self.d = omega / (2 * zeta) * (1-p)/2

        self.clipping_thresholds = clipping_thresholds

    def soc_clipping(self, soc, reference_power):
        clip_fraction = np.interp(
            soc,
            self.clipping_thresholds,
            [0, 1, 1, 0],
            left=0,
            right=0
        )

        r_charge = clip_fraction * self.rated_power_charging
        r_discharge = clip_fraction * self.rated_power_discharging

        return np.clip(reference_power, -r_discharge, r_charge)

    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        current_power = self.measurements_dict["battery_power"]
        soc = self.measurements_dict["battery_soc"]

        # Apply reference clipping
        reference_power = self.soc_clipping(soc, reference_power)

        e = reference_power - current_power

        # Compute control
        u = self.c * self.x + self.d * e

        # Update controller internal state
        self.x = self.a * self.x + self.b * e

        self.controls_dict["power_setpoint"] = current_power + u

class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        self.controls_dict["power_setpoint"] = reference_power
