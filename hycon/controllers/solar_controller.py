import numpy as np

from hycon.controllers.controller_base import ControllerBase


class SolarController(ControllerBase):
    """
    Modifies power reference to  ensures smoothness in solar reference signal to avoid rapid
    changes in power reference, similar to the `BatteryController` unit.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Instantiate SolarController.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the corresponding
                plant component.
            controller_parameters (dict): Dictionary of controller parameters k_solar and
                clipping_thresholds. See set_controller_parameters for more details. If
                controller parameters are provided both in input_dict and controller_parameters,
                the latter will take precedence.
            verbose (bool): If True, print debug information.
        """
        super().__init__(interface, cname, verbose)

        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

        # Initialize controller internal state
        self.x = 0

    def set_controller_parameters(
        self,
        k_solar=0.1,
    ):
        """
        Set gains and threshold limits for SolarController.

        k_solar is the controller gain. The controller will be stable and slow to react for small
        values of k_solar (e.g. k_solar=0.01), and will be fast to react (and eventually unstable)
        for large values of k_solar (e.g. k_solar=1).

        clipping_thresholds is a list of four values: [soc_min, soc_min_clip, soc_max_clip,
        soc_max]. soc_min is the minimum allowable SOC value, below which the controller output
        reference power will be zero. soc_min_clip is the SOC value below which the controller
        applies clipping to the reference power (the reference power is clipped linearly between
        soc_min and soc_min_clip). Similarly, soc_max_clip is the SOC value above which linear
        clipping is applied, until soc_max, after which the output is zero. Between soc_min_clip
        and soc_max_clip, the full reference power is used.

        Args:
            k_solar (float): Gain for controller.
            clipping_thresholds (list): SOC thresholds for clipping reference power. Should be a
                list of four values: [soc_min, soc_min_clip, soc_max_clip, soc_max].
        """
        zeta = 2
        omega = 2 * np.pi * k_solar

        # Discrete-time, first-order state-space model of controller
        p = np.exp(-2 * zeta * omega * self.dt)
        self.a = p
        self.b = 1
        self.c = omega / (2 * zeta) * (1 - p) / 2 * (p + 1)
        self.d = omega / (2 * zeta) * (1 - p) / 2

    def compute_controls(self, measurements_dict):
        """
        Main compute_controls method for SolarController.
        """
        reference_power = measurements_dict[self.cname]["power_reference"]
        current_power = measurements_dict[self.cname]["power"]
        power_limit_lower = measurements_dict[self.cname].get("power_limit_lower", 0)
        power_limit_upper = measurements_dict[self.cname].get("power_limit_upper", np.inf)

        # Clip according to upper and lower limits
        reference_power = np.clip(reference_power, power_limit_lower, power_limit_upper)

        e = reference_power - current_power

        # Compute control
        u = self.c * self.x + self.d * e

        # Update controller internal state
        self.x = self.a * self.x + self.b * e

        controls_dict = {self.cname: {"power_setpoint": current_power + u}}

        return controls_dict
