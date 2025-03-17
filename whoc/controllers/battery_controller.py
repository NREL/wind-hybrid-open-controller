from whoc.controllers.controller_base import ControllerBase


class BatteryController(ControllerBase):
    """
    Modifies power reference to consider battery degradation for single battery.

    In particular, ensures smoothness in battery reference signal to avoid rapid
    changes in power reference, which can lead to degradation.
    """
    def __init__(self, interface, input_dict, controller_parameters={}, verbose=True):
        super().__init__(interface, verbose)

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

        # Extract other needed parameters from input_dict
        self.dt = input_dict["dt"]
        
        # Assumes one battery!
        battery_name = [k for k in input_dict["py_sims"] if "battery" in k][0]
        self.rated_power_charging = input_dict["py_sims"][battery_name]["charge_rate"] * 1e3
        self.rated_power_discharging = input_dict["py_sims"][battery_name]["discharge_rate"] * 1e3

    def set_controller_parameters(
        self,
        k_p_max=1,
        k_p_min=None,
        **_ # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set gains and threshold limits for BatteryController.

        Args:
            k_p_max: Maximum proportional gain. Defaults to 1.
            k_p_min: Minimum proportional gain. Defaults to None, which is
               a code for matching k_p_max.
        """
        if k_p_min is None:
            k_p_min = k_p_max
        self.k_p_max = k_p_max
        self.k_p_min = k_p_min

    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        # Note sign change to match battery convention
        # (positive current_power is discharging / negative battery_power is discharging)
        current_power = -self.measurements_dict["battery_power"]
        soc = self.measurements_dict["battery_soc"]

        e = reference_power - current_power

        # Evaluate gain schedule for proportional gain at the CURRENT soc (?)
        k_p = self.quadratic_gain_schedule(self.k_p_max, self.k_p_min, soc)

        u = k_p * e

        self.controls_dict["power_setpoint"] = current_power + u

    @staticmethod
    def quadratic_gain_schedule(k_p_max, k_p_min, soc):
        return -4*(k_p_max - k_p_min) * (soc - 0.5)**2 + k_p_max


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        self.controls_dict["power_setpoint"] = reference_power
