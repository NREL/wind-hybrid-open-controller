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
        # self._e_prev_1 = 0

    def compute_controls(self):
        reference_power = self.measurements_dict["battery_power_reference"]
        # TODO: is the actual power output important? Should I just be smoothing
        # the power reference? Not clear.
        current_power = self.measurements_dict["battery_power"]

        e = reference_power - current_power
        e_dot = (e - self._e_prev)/self.dt # Or do I want the second derivative?

        u = self.k_p * e - (self.k_d) * e_dot

        # If I _add_ something that is proportional to the miss, that's essentially
        # adding integral action. Not sure that's what I want? Only if I add it 
        # to the existing reference?
        self.controls_dict["power_setpoint"] = current_power + u
