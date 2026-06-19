import numpy as np
from scipy.interpolate import interp1d

from hycon.controllers.controller_base import ControllerBase

# coal controller notes:
# Takes in bid curve, day ahead prices, on or off status
# Look at battery controller for day ahead behavior
# Only execute this logic if plant is on


class CoalPlantController(ControllerBase):
    """
    Controller considers price, plant status, and external power reference commands to determine power setpoint.

    This controller implements a price-arbitrage strategy that uses day-ahead (DA)
    locational marginal prices (LMPs) to decide when to dispatch the coal plant.

    The controller compares the DA LMP against the plant's bid curve to determine
    the appropriate power output. The coal plant will generate power if the plant is on
    and the DA price is higher than the price of coal (bid according to input bid curve).

    Additionally, the controller enforces ramping constraints and respects an external power reference command,
    which could be used for grid services or other system-level objectives.
    The controller also ensures that the power setpoint respects the plant's minimum stable load and maximum capacity.

    Controller parameters:
        bid_curve (list): List of tuples (price, power) representing the coal plant's bid curve.
        ramp_rate (float): Maximum change in power output per time step (in MW/min).
        max_control_output (float): Maximum power output for the coal plant (in kW).

    """

    def __init__(self, interface, cname, input_dict, controller_parameters={}, verbose=True):
        super().__init__(interface, cname, verbose)

        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(
        self,
        bid_curve,
        ramp_rate,
        max_control_output=None,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set parameters for CoalPlantController.

        Args:
            bid_curve (list): List of tuples (price, power) representing the coal plant's bid curve.
            ramp_rate (float): Maximum change in power output per time step (in MW/min).
            max_control_output (float): Maximum power output for the coal plant (in kW). Defaults to None.
        """

        self.bid_curve = bid_curve
        prices, powers = zip(*bid_curve)
        self.bid_interpolator = interp1d(prices, powers, kind="quadratic", fill_value="extrapolate")
        if max_control_output is not None:
            self.max_control_output = max_control_output
        self.ramp_rate_MW_per_dt = ramp_rate * (1 / (60 / self.dt)) # convert ramp rate to MW per time step, assuming dt is in seconds

    def compute_controls(self, measurements_dict):
        # NOTE: Current power calculation is in MW!!
        day_ahead_lmp = measurements_dict["DA_LMP"]
        power_bids = self.bid_interpolator(day_ahead_lmp)
        plant_status = measurements_dict[self.cname]["status_reference"]

        external_power_reference = measurements_dict[self.cname]["power_reference"] / 1e3

        # Bid curve is in MW, so convert min stable load to MW from kW for comparison
        min_power_value = self.plant_parameters[self.cname]["min_stable_load"] / 1e3
        max_power_value = min(self.plant_parameters[self.cname]["capacity"], \
                              getattr(self, "max_control_output", float("inf")))/ 1e3

        if plant_status == 1:  # Plant is on
            # Assuming we're looking at the first hour's price for simplicity
            power_setpoint = power_bids
            power_setpoint = min(power_setpoint, external_power_reference)
            # Ensure power setpoint is within bounds
            power_setpoint = np.clip(power_setpoint, min_power_value, max_power_value)

            # Apply ramping constraints
            if hasattr(self, "ramp_rate_MW_per_dt"):
                previous_power_setpoint = measurements_dict[self.cname].get("power", 0.0) / 1e3
                power_change = power_setpoint - previous_power_setpoint
                max_power_change = self.ramp_rate_MW_per_dt  # Already in MW

                if abs(power_change) > max_power_change:
                    power_setpoint = previous_power_setpoint + np.sign(power_change) * max_power_change
        else: # Plant is off, so set power setpoint to 0
            power_setpoint = 0.0

        # Convert back to kW for control output
        return {self.cname: {"power_setpoint": float(power_setpoint*1e3), "uncurtailable": True, "ramp_rate": self.ramp_rate_MW_per_dt*1e3}}
