from hycon.controllers.controller_base import ControllerBase


class SolarPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (scalar) solar simulator.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        super().__init__(interface, cname, verbose)
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self):
        pass

    def compute_controls(self, measurements_dict):
        return {self.cname: {"power_setpoint": measurements_dict["solar_farm"]["power_reference"]}}
