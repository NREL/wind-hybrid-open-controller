from whoc.controllers.controller_base import ControllerBase


class BatteryController(ControllerBase):
    """
    Modifies power reference to consider battery degradation for single battery.

    In particular, ensures smoothness in battery reference signal to avoid rapid
    changes in power reference, which can lead to degradation.
    """
    def __init__(self, interface, input_dict, k_p=None, k_d=None, verbose=True):
        super().__init__(interface, verbose)

        # Handle controller gain specification
        if k_p is None:
            if "battery_proportional_gain" in input_dict["controller"]:
                k_p = input_dict["controller"]["battery_proportional_gain"]
            else:
                k_p = 1 # default value
        elif "battery_proportional_gain" in input_dict["controller"]:
            print(
                "Found proportional gain in both input dict and controller "
                "instantiation. Using k_p = {0}".format(k_p)
            )
        else:
            pass # Use specified k_p
        if k_d is None:
            if "battery_derivative_gain" in input_dict["controller"]:
                k_d = input_dict["controller"]["battery_derivative_gain"]
            else:
                k_d = 0 # default value
        elif "battery_derivative_gain" in input_dict["controller"]:
            print(
                "Found derivative gain in both input dict and controller "
                "instantiation. Using k_d = {0}".format(k_d)
            )
        else:
            pass # Use specified k_d

        self.dt = input_dict["dt"]

        self.k_p = k_p
        self.k_d = k_d

        self._e_prev = 0
        self._battery_power_prev = 0
        self._partial_cycle_count = 0
        self._accumulated_cycles = 0

        # self._e_prev_1 = 0

    def compute_controls(self):
        reference_power = self.measurements_dict["power_reference"]
        # TODO: is the actual power output important? Should I just be smoothing
        # the power reference? Not clear.
        current_power = self.measurements_dict["battery_power"]

        partial_cycle_count_limit = 1 # TODO: allow this to be user specified
        accumulated_cycle_count_limit = 1 # TODO: allow this to be user specified
        cycle_count_unit_time = 24*60*60 # 24 hours in seconds (allow user spec)
        cycle_count_midnight_offset = 0 # 0 seconds after midnight (allow user spec)
        soc_throttling_upper_limit = 0.9 # 90% SOC
        soc_throttling_lower_limit = 0.1 # 10% SOC
        soc_throttling_value = 0.5 # 50% of control signal

        time = self.measurements_dict["time"]
        if (time-cycle_count_midnight_offset) % cycle_count_unit_time == 0:
            self._partial_cycle_count = 0
            self._accumulated_cycles = 0

        # Check if there is a sign change in battery_power
        if self._battery_power_prev * current_power < 0: # How will this work with 0?
            self._partial_cycle_count += 0.5
        self._accumulated_cycles += np.abs(current_power)

        e = reference_power - current_power
        e_dot = (e - self._e_prev)/self.dt # Or do I want the second derivative?

        u = self.k_p * e - (self.k_d) * e_dot

        if self._partial_cycle_count >= partial_cycle_count_limit:
            u = 0
        if self._accumulated_cycles >= accumulated_cycle_count_limit:
            u = 0

        # Could also have SOC-dependent throttling logic here.
        # TODO: Should throttling apply even when "exiting" the danger region?
        if SOC > soc_throttling_upper_limit and u > 0:
          u = soc_throttling_value * u
        elif SOC < soc_throttling_lower_limit and u < 0:
          u = soc_throttling_value * u
        else:
          pass 

        # If I _add_ something that is proportional to the miss, that's essentially
        # adding integral action. Not sure that's what I want? Only if I add it 
        # to the existing reference?

        # Add a cycle count based on day
        self.controls_dict["battery_power_setpoint"] = current_power + u


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self):
        reference_power = self.measurements_dict["battery_power_reference"]
        self.controls_dict["battery_power_setpoint"] = reference_power
