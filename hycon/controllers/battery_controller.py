import numpy as np

from hycon.controllers.controller_base import ControllerBase


class BatteryController(ControllerBase):
    """
    Modifies power reference to consider battery degradation for single battery.

    In particular, ensures smoothness in battery reference signal to avoid rapid
    changes in power reference, which can lead to degradation.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Instantiate BatteryController.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the corresponding
                plant component.
            controller_parameters (dict): Dictionary of controller parameters k_batt and
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
        k_batt=0.1,
        clipping_thresholds=[0, 0, 1, 1],
    ):
        """
        Set gains and threshold limits for BatteryController.

        k_batt is the controller gain. The controller will be stable and slow to react for small
        values of k_batt (e.g. k_batt=0.01), and will be fast to react (and eventually unstable)
        for large values of k_batt (e.g. k_batt=1).

        clipping_thresholds is a list of four values: [soc_min, soc_min_clip, soc_max_clip,
        soc_max]. soc_min is the minimum allowable SOC value, below which the controller output
        reference power will be zero. soc_min_clip is the SOC value below which the controller
        applies clipping to the reference power (the reference power is clipped linearly between
        soc_min and soc_min_clip). Similarly, soc_max_clip is the SOC value above which linear
        clipping is applied, until soc_max, after which the output is zero. Between soc_min_clip
        and soc_max_clip, the full reference power is used.

        Args:
            k_batt (float): Gain for controller.
            clipping_thresholds (list): SOC thresholds for clipping reference power. Should be a
                list of four values: [soc_min, soc_min_clip, soc_max_clip, soc_max].
        """
        zeta = 2
        omega = 2 * np.pi * k_batt

        # Discrete-time, first-order state-space model of controller
        p = np.exp(-2 * zeta * omega * self.dt)
        self.a = p
        self.b = 1
        self.c = omega / (2 * zeta) * (1 - p) / 2 * (p + 1)
        self.d = omega / (2 * zeta) * (1 - p) / 2

        self.clipping_thresholds = clipping_thresholds

    def soc_clipping(self, soc, reference_power):
        """
        Clip the input reference based on the state of charge and clipping_thresholds.

        Args:
            soc (float): Current state of charge.
            reference_power (float): Reference power to be clipped.

        Returns:
            float: Clipped reference power.
        """
        clip_fraction = np.interp(soc, self.clipping_thresholds, [0, 1, 1, 0], left=0, right=0)

        r_charge = clip_fraction * self.plant_parameters[self.cname]["charge_rate"]
        r_discharge = clip_fraction * self.plant_parameters[self.cname]["discharge_rate"]

        return np.clip(reference_power, -r_discharge, r_charge)

    def compute_controls(self, measurements_dict):
        """
        Main compute_controls method for BatteryController.
        """
        reference_power = measurements_dict[self.cname]["power_reference"]
        current_power = measurements_dict[self.cname]["power"]
        soc = measurements_dict[self.cname]["state_of_charge"]

        # Apply reference clipping
        reference_power = self.soc_clipping(soc, reference_power)

        e = reference_power - current_power

        # Compute control
        u = self.c * self.x + self.d * e

        # Update controller internal state
        self.x = self.a * self.x + self.b * e

        controls_dict = {self.cname: {"power_setpoint": current_power + u}}

        # TODO: Implement upper limit (interconnect limitation) and lower limit
        # (if no grid charging)

        return controls_dict


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Instantiate BatteryPassthroughController.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the corresponding
                plant component.
            controller_parameters (dict): Dictionary of controller parameters. Not used for
                BatteryPassthroughController, but included for consistency with ControllerBase.
            verbose (bool): If True, print debug information.
        """
        super().__init__(interface, cname, verbose)

        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self):
        """
        No parameters for BatteryPassthroughController, but method is needed to be consistent with
        ControllerBase.
        """
        return None

    def compute_controls(self, measurements_dict):
        """
        Main compute_controls method for BatteryPassthroughController.
        """
        # TODO: Implement upper limit (interconnect limitation) and lower limit
        # (if no grid charging)
        return {self.cname: {"power_setpoint": measurements_dict[self.cname]["power_reference"]}}


class BatteryPriceSOCController(ControllerBase):
    """
    Controller considers price and SOC to determine power setpoint.

    This controller implements a price-arbitrage strategy that uses day-ahead (DA)
    locational marginal prices (LMPs) and real-time (RT) LMPs to decide when to
    charge or discharge the battery. The algorithm identifies the top and bottom
    price hours of the day based on battery duration (e.g., for a 4-hour battery,
    it targets the "top_d" = 4 highest and "bottom_d" = 4 lowest priced hours).

    The decision logic is as follows:
        1. If RT price exceeds the highest DA price: discharge at full rate
           (unconditionally).
        2. Else if RT price is in the top-d highest DA prices AND SOC > low_soc:
           discharge at full rate.
        3. Else if RT price is below the lowest DA price: charge at full rate
           (unconditionally).
        4. Else if RT price is in the bottom-d lowest DA prices AND SOC < high_soc:
           charge at full rate.
        5. Otherwise: hold (power setpoint = 0).

    The SOC thresholds (high_soc, low_soc) prevent over-charging or over-discharging
    during moderate price signals, while still allowing full charge/discharge when
    prices move outside the expected DA range.

    Note:
        Charging power is represented as negative values, matching the convention
        used at the Hercules/hybrid_plant level.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Instantiate BatteryPriceSOCController.

        Args:
            interface (object): Interface object for communicating with simulator.
            cname (str): Name of controller, which should match the name of the corresponding
                plant component.
            controller_parameters (dict): Dictionary of controller parameters high_soc and low_soc.
                See set_controller_parameters method for more details.
            verbose (bool): If True, print debug information.
        """
        super().__init__(interface, cname, verbose)

        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

        self.rated_power_charging = self.plant_parameters[self.cname]["charge_rate"]
        self.rated_power_discharging = self.plant_parameters[self.cname]["discharge_rate"]

        # Save the duration rounded to nearest hour
        self.duration = round(
            self.plant_parameters[self.cname]["energy_capacity"]
            / self.plant_parameters[self.cname]["power_capacity"]
        )

        # Raise if duration makes this controller implausible
        if self.duration >= 12:
            raise ValueError(
                f"Battery duration is {self.duration} hours, which is not "
                "supported by BatteryPriceSOCController."
                " This controller is only intended for durations shorter than 12 hours."
            )

        if self.duration < 1:
            raise ValueError(
                f"Battery duration is {self.duration} hours, which is not "
                "supported by BatteryPriceSOCController."
                " This controller is only intended for durations of at least 1 hour."
            )

    def set_controller_parameters(
        self,
        high_soc=1.0,
        low_soc=0.0,
    ):
        """
        Set parameters for BatteryPriceSOCController.

        high_soc is the SOC threshold above which the battery will only charge if the price is below
        the lowest (hourly) DA price of the day.  Defaults to 1.0.

        low_soc is the SOC threshold below which the battery will only discharge if the price is
        above the highest (hourly) DA price of the day.  Defaults to 0.2.

        high_soc defaults to 1.0 (effectively disabled) as experience suggests waiting for
        very low prices is not worthwhile. low_soc defaults to 0.2 as experience suggests waiting
        for very high prices is worthwhile.

        Args:
            high_soc (float): High SOC threshold (0 to 1).  Defaults to 1.0.
            low_soc (float): Low SOC threshold (0 to 1).  Defaults to 0.2.
        """
        self.high_soc = high_soc
        self.low_soc = low_soc

    def compute_controls(self, measurements_dict):
        day_ahead_lmps = np.array(measurements_dict["DA_LMP_24hours"])
        sorted_day_ahead_lmps = np.sort(day_ahead_lmps)
        real_time_lmp = measurements_dict["RT_LMP"]

        # Extract limits
        bottom_d = sorted_day_ahead_lmps[self.duration - 1]
        top_d = sorted_day_ahead_lmps[-self.duration]
        bottom_1 = sorted_day_ahead_lmps[0]
        top_1 = sorted_day_ahead_lmps[-1]

        # Access the state of charge and LMP in real-time
        soc = measurements_dict[self.cname]["state_of_charge"]

        # Note that the convention is followed where charging is negative power
        # This matches what is in place in the hercules/hybrid_plant level and
        # will be inverted before passing into the battery modules
        if real_time_lmp > top_1:
            power_setpoint = self.rated_power_discharging
        elif (real_time_lmp > top_d) & (soc > self.low_soc):
            power_setpoint = self.rated_power_discharging
        elif real_time_lmp < bottom_1:
            power_setpoint = -self.rated_power_charging
        elif (real_time_lmp < bottom_d) & (soc < self.high_soc):
            power_setpoint = -self.rated_power_charging
        else:
            power_setpoint = 0.0

        # Limit the power_setpoint by the SOC
        if power_setpoint > 0:  # Trying to discharge
            if soc <= self.plant_parameters[self.cname]["state_of_charge_min"]:  # Fully depleted
                power_setpoint = 0.0

        # Other way
        if power_setpoint < 0:  # Trying to charge
            if soc >= self.plant_parameters[self.cname]["state_of_charge_max"]:  # Fully charged
                power_setpoint = 0.0

        # Apply limitations based on super controller
        power_limit_lower = measurements_dict[self.cname].get("power_limit_lower", -np.inf)
        power_limit_upper = measurements_dict[self.cname].get("power_limit_upper", np.inf)
        power_setpoint = np.clip(power_setpoint, power_limit_lower, power_limit_upper)

        return {self.cname: {"power_setpoint": power_setpoint}}
