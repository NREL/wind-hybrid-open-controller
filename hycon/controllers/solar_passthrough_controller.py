from hycon.controllers.controller_base import ControllerBase


class SolarPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (scalar) solar simulator.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Constructor for SolarPassthroughController.

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            cname (str): Name of the controller, used for indexing into measurements and controls
                dictionaries. Should match the component name in the plant model.
            controller_parameters (dict): Dictionary of controller parameters. Empty for this
                passthrough controller.
        """
        super().__init__(interface, cname, verbose)
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self):
        pass

    def compute_controls(self, measurements_dict):
        return {self.cname: {"power_setpoint": measurements_dict[self.cname]["power_reference"]}}
