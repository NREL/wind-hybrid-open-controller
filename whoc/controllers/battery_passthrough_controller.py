from whoc.controllers.controller_base import ControllerBase


class BatteryPassthroughController(ControllerBase):
    """
    Simply passes power reference down to (single) battery.
    """
    def __init__(self, interface, input_dict, verbose=True):
        super().__init__(interface, verbose)

    def compute_controls(self):
        reference_power = self.measurements_dict["battery_power_reference"]
        self.controls_dict["power_setpoint"] = reference_power