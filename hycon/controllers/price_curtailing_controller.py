from hycon.controllers.controller_base import ControllerBase


class PriceCurtailingController(ControllerBase):
    """
    Curtails the component if the real-time price drops below a user-defined threshold.
    Otherwise, simply passes through the power reference.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=True):
        """
        Constructor for PriceCurtailingController.

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            cname (str): Name of the controller, used for indexing into measurements and controls
                dictionaries. Should match the component name in the plant model.
            controller_parameters (dict): Dictionary of controller parameters. See
                set_controller_parameters for details on expected controller parameters.
        """
        super().__init__(interface, cname, verbose)
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    def set_controller_parameters(self, curtailment_price=0.0):
        """
        Set controller parameters for PriceCurtailingController.

        Args:
            curtailment_price: Real-time price threshold for curtailment. If the real-time price
                drops below this threshold, the controller will curtail the component (i.e., set
                power reference to 0). Defaults to 0.0.
        """
        if not isinstance(curtailment_price, (int, float)):
            raise ValueError("curtailment_price must be a single numeric value.")
        self.curtailment_price = curtailment_price

    def compute_controls(self, measurements_dict):
        if "RT_LMP" not in measurements_dict:
            raise KeyError(
                "measurements_dict must contain key 'RT_LMP' to use "
                + self.__class__.__name__
                + "."
            )
        elif "power_reference" not in measurements_dict[self.cname]:
            raise KeyError(
                "measurements_dict['"
                + self.cname
                + "'] must contain key 'power_reference' to use "
                + self.__class__.__name__
                + "."
            )

        # Threshold based on curtailment price
        if measurements_dict["RT_LMP"] <= self.curtailment_price:
            power_setpoint = 0.0
        else:
            power_setpoint = measurements_dict[self.cname]["power_reference"]

        return {self.cname: {"power_setpoint": power_setpoint}}
