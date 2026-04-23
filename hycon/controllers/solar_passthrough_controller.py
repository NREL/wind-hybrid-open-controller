from hycon.controllers.controller_base import ControllerBase


class SolarPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (scalar) solar simulator.
    """

    def __init__(self, interface, input_dict, cname, controller_parameters={}, verbose=True):
        super().__init__(interface, cname, verbose)

    def compute_controls(self, measurements_dict):
        return {self.cname: {"power_setpoint": measurements_dict["solar_farm"]["power_reference"]}}
