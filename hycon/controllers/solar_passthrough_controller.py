from hycon.controllers.controller_base import ControllerBase


class SolarPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (scalar) solar simulator.
    """

    def __init__(self, interface, cname, controller_parameters=None, verbose=True):
        """
        Constructor for SolarPassthroughController.

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            cname (str): Name of the controller, used for indexing into measurements and controls
                dictionaries. Should match the component name in the plant model.
            controller_parameters (dict): Dictionary of controller parameters. Empty for this
                passthrough controller.
        """
        if controller_parameters is None:
            controller_parameters = {}
        super().__init__(interface, cname, verbose)
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self, solar_plant_capacity=None, **_):
        self.max_control_output = solar_plant_capacity if solar_plant_capacity is not None \
                                                            else float("inf")

    def compute_controls(self, measurements_dict):
        return {self.cname: {"power_setpoint": min(measurements_dict[self.cname]["power_reference"],
                                                   self.max_control_output)}}
