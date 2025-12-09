import numpy as np

from hycon.controllers.controller_base import ControllerBase


class BatteryController(ControllerBase):
    """
    Modifies power reference to consider battery degradation for single battery.

    In particular, ensures smoothness in battery reference signal to avoid rapid
    changes in power reference, which can lead to degradation.
    """

    def __init__(self, interface, input_dict, controller_parameters={}, verbose=True):
        """
        Instantiate BatteryController.

        Args:
            interface (object): Interface object for communicating with simulator.
            input_dict (dict): Dictionary of input parameters (e.g. from Hercules).
            controller_parameters (dict): Dictionary of controller parameters k_batt and
                clipping_thresholds. See set_controller_parameters for more details. If
                controller parameters are provided both in input_dict and controller_parameters,
                the latter will take precedence.
            verbose (bool): If True, print debug information.
        """
        super().__init__(interface, verbose)

        # Extract global parameters
        self.dt = input_dict["dt"]

        # Check that parameters are not specified both in input file
        # and in controller_parameters
        if "controller" in input_dict:
            for cp in controller_parameters.keys():
                if cp in input_dict["controller"]:
                    raise KeyError(
                        'Found key "' + cp + '" in both input_dict["controller"] and'
                        " in controller_parameters."
                    )
            controller_parameters = {**controller_parameters, **input_dict["controller"]}
        self.set_controller_parameters(**controller_parameters)

        # Initialize controller internal state
        self.x = 0

    def set_controller_parameters(
        self,
        k_batt=0.1,
        clipping_thresholds=[0, 0, 1, 1],
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
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

        r_charge = clip_fraction * self.plant_parameters["battery"]["charge_rate"]
        r_discharge = clip_fraction * self.plant_parameters["battery"]["discharge_rate"]

        return np.clip(reference_power, -r_discharge, r_charge)

    def compute_controls(self, measurements_dict):
        """
        Main compute_controls method for BatteryController.
        """
        reference_power = measurements_dict["battery"]["power_reference"]
        current_power = measurements_dict["battery"]["power"]
        soc = measurements_dict["battery"]["state_of_charge"]

        # Apply reference clipping
        reference_power = self.soc_clipping(soc, reference_power)

        e = reference_power - current_power

        # Compute control
        u = self.c * self.x + self.d * e

        # Update controller internal state
        self.x = self.a * self.x + self.b * e

        controls_dict = {"power_setpoint": current_power + u}

        return controls_dict


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """

    def __init__(self, interface, input_dict, verbose=True):
        """ "
        Instantiate BatteryPassthroughController."
        """
        super().__init__(interface, verbose)

    def compute_controls(self, measurements_dict):
        """ "
        Main compute_controls method for BatteryPassthroughController.
        """
        return {"power_setpoint": measurements_dict["battery"]["power_reference"]}


class BatteryPriceSOCController(ControllerBase):
    """
    Controller considers price and SOC to determine power setpoint.
    """

    def __init__(self, interface, input_dict, controller_parameters={}, verbose=True):
        super().__init__(interface, verbose)

        # Check that parameters are not specified both in input file
        # and in controller_parameters
        if "controller" in input_dict:
            for cp in controller_parameters.keys():
                if cp in input_dict["controller"]:
                    raise KeyError(
                        'Found key "' + cp + '" in both input_dict["controller"] and'
                        " in controller_parameters."
                    )
            controller_parameters = {**controller_parameters, **input_dict["controller"]}
        self.set_controller_parameters(**controller_parameters)

        self.rated_power_charging = input_dict["battery"]["charge_rate"]
        self.rated_power_discharging = input_dict["battery"]["discharge_rate"]

    def set_controller_parameters(
        self,
        high_soc=0.8,
        low_soc=0.2,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set parameters for BatteryPriceSOCController.

        high_soc is the SOC threshold above which the battery will only charge if the price is above
        the highest (hourly) DA price of the day.

        low_soc is the SOC threshold below which the battery will only discharge if the price is
        below the lowest (hourly) DA price of the day.

        Args:
            high_soc (float): High SOC threshold (0 to 1).
            low_soc (float): Low SOC threshold (0 to 1).
        """
        self.high_soc = high_soc
        self.low_soc = low_soc

    def compute_controls(self, measurements_dict):
        day_ahead_lmps = np.array(measurements_dict["DA_LMP_24hours"])
        sorted_day_ahead_lmps = np.sort(day_ahead_lmps)
        real_time_lmp = measurements_dict["RT_LMP"]

        # Extract limits
        bottom_4 = sorted_day_ahead_lmps[3]
        top_4 = sorted_day_ahead_lmps[-4]
        bottom_1 = sorted_day_ahead_lmps[0]
        top_1 = sorted_day_ahead_lmps[-1]

        # Access the state of charge and LMP in real-time
        soc = measurements_dict["battery"]["state_of_charge"]

        # Note that the convention is followed where charging is negative power
        # This matches what is in place in the hercules/hybrid_plant level and
        # will be inverted before passing into the battery modules
        if real_time_lmp > top_1:
            power_setpoint = self.rated_power_discharging
        elif (real_time_lmp > top_4) & (soc < self.high_soc):
            power_setpoint = self.rated_power_discharging
        elif real_time_lmp < bottom_1:
            power_setpoint = -self.rated_power_charging
        elif (real_time_lmp < bottom_4) & (soc > self.low_soc):
            power_setpoint = -self.rated_power_charging
        else:
            power_setpoint = 0.0

        return {"power_setpoint": power_setpoint}
