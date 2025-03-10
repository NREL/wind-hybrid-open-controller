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

        # Initialize state variables
        self._e_prev = 0
        self._battery_power_prev = 0
        self._partial_cycle_count = 0
        self._accumulated_cycles = 0

    def set_controller_parameters(
        self,
        k_p=1,
        k_d=0,
        partial_cycle_count_limit=np.inf,
        accumulated_cycle_count_limit=np.inf,
        cycle_count_reset_time=24*60*60,
        soc_throttling_upper_limit=1.0,
        soc_throttling_lower_limit=0.0,
        soc_throttling_value=0.5,
        **_kwargs
    ):
        """
        Set gains and threshold limits for BatteryController.

        Args:
            k_p: Proportional gain. Defaults to 1.
            k_d: Derivative gain. Defaults to 0.
            partial_cycle_count_limit: Maximum number of full or partial cycles allowed
                before controller prevents battery use. Defaults to infinity.
            accumulated_cycle_count_limit: Maximum number of accumulated full cycles allowed
                before controller prevents battery use. Defaults to infinity.
            cycle_count_reset_time: Time in seconds for the partial and accumulated cycle
                counts to be reset. Defaults to 24*60*60 seconds (24 hours).
            soc_throttling_upper_limit: Upper allowable SOC before throttling. Defaults to 1.0.
            soc_throttling_lower_limit: Lower allowable SOC before throttling. Defaults to 0.0.
            soc_throttling_value: Value to multiply control signal by when throttling.
                Defaults to 0.5.
        """
        self.k_p = k_p
        self.k_d = k_d
        self.partial_cycle_count_limit = partial_cycle_count_limit
        self.accumulated_cycle_count_limit = accumulated_cycle_count_limit
        self.cycle_count_reset_time = cycle_count_reset_time
        self.soc_throttling_upper_limit = soc_throttling_upper_limit
        self.soc_throttling_lower_limit = soc_throttling_lower_limit
        self.soc_throttling_value = soc_throttling_value


    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        # TODO: is the actual power output important? Should I just be smoothing
        # the power reference? Not clear.
        current_power = self.measurements_dict["battery_power"]
        soc = self.measurements_dict["battery_soc"]

        time = self.measurements_dict["time"]
        if time % self.cycle_count_reset_time == 0: # Reset counters
            self._partial_cycle_count = 0
            self._accumulated_cycles = 0

        # Check if there is a sign change in battery_power
        if self._battery_power_prev * current_power < 0: # TODO: How will this work with 0?
            self._partial_cycle_count += 0.5
        self._accumulated_cycles += (
            np.abs(current_power)
            / 0.5*(self.rated_power_charging + self.rated_power_discharging)
        )

        e = reference_power - current_power
        e_dot = (e - self._e_prev)/self.dt # Or do I want the second derivative?

        u = self.k_p * e - (self.k_d) * e_dot

        if self._partial_cycle_count >= self.partial_cycle_count_limit:
            u = 0
        if self._accumulated_cycles >= self.accumulated_cycle_count_limit:
            u = 0

        # Could also have SOC-dependent throttling logic here.
        # TODO: Should throttling apply even when "exiting" the danger region?
        if soc > self.soc_throttling_upper_limit and u > 0:
          u = self.soc_throttling_value * u
        elif soc < self.soc_throttling_lower_limit and u < 0:
          u = self.soc_throttling_value * u
        else:
          pass 

        # If I _add_ something that is proportional to the miss, that's essentially
        # adding integral action. Not sure that's what I want? Only if I add it 
        # to the existing reference?

        self.controls_dict["power_setpoint"] = current_power + u


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self):
        reference_power = self.measurements_dict["battery_power_reference"]
        self.controls_dict["power_setpoint"] = reference_power
