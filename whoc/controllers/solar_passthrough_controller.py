from whoc.controllers.controller_base import ControllerBase


class SolarPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (scalar) solar simulator.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self, measurements_dict):
        return {"power_setpoint": measurements_dict["solar_power_reference"]}
