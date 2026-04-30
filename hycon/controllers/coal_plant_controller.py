import numpy as np
from hycon.controllers.controller_base import ControllerBase
from scipy.interpolate import interp1d


# coal controller notes:
# Takes in bid curve, day ahead prices, on or off status
# Look at battery controller for day ahead behavior
# Only execute this logic if plant is on


class CoalPlantController(ControllerBase):
    """
    Controller considers price and plant status to determine power setpoint.

    This controller implements a price-arbitrage strategy that uses day-ahead (DA)
    locational marginal prices (LMPs) to decide when to dispatch teh coal plant.
    The algorithm identifies the top and bottom
    The coal plant will discharge if the plant is on and the DA price is higher than
    the price of coal (bid according to input bid curve).
    Note:
        Add logic to override DA bids if there are other signals.

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

    def set_controller_parameters(
        self,
        bid_curve,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set parameters for CoalPlantController.

        Args:
            bid_curve (list): List of tuples (price, power) representing the coal plant's bid curve.
            low_soc (float): Low SOC threshold (0 to 1).  Defaults to 0.2.
        """
        # self.high_soc = high_soc
        # self.low_soc = low_soc
        self.bid_curve = bid_curve
        prices, powers = zip(*bid_curve)
        self.bid_interpolator = interp1d(prices, powers, kind="quadratic")

    def compute_controls(self, measurements_dict):
        day_ahead_lmps = np.array(measurements_dict["DA_LMP_24hours"])
        power_bids = self.bid_interpolator(day_ahead_lmps)
        plant_status = measurements_dict["plant"]["status"]


        if plant_status == 1:  # Plant is on
            # Assuming we're looking at the first hour's price for simplicity
            power_setpoint = power_bids[0]
        else: # Plant is off, so set power setpoint to 0
            power_setpoint = 0.0

        return {"power_setpoint": power_setpoint}
