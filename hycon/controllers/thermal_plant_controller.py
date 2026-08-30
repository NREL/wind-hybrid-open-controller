from hycon.controllers.controller_base import ControllerBase

# Default power setpoint in kW (meant to ensure power maximization)
POWER_SETPOINT_DEFAULT = 1e9


class ThermalPlantController(ControllerBase):
    """
    Sets thermal plant power reference between turbines without
    feedback on current power generation.
    """

    def __init__(self, interface, cname, controller_parameters={}, verbose=False):
        super().__init__(interface, cname, verbose)
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

    # def compute_controls(self, measurements_dict):

    #     ref_in_lower_dict = (
    #         "power_reference" in measurements_dict[self.cname]
    #         and measurements_dict[self.cname]["power_reference"] is not None
    #     )
    #     ref_in_upper_dict = (
    #         "power_reference" in measurements_dict
    #         and measurements_dict["power_reference"] is not None
    #     )
    #     if ref_in_lower_dict and ref_in_upper_dict:
    #         raise KeyError(
    #             "Found 'power_reference' in both measurements_dict['"
    #             + self.cname
    #             + "'] and measurements_dict."
    #         )
    #     elif ref_in_lower_dict:
    #         farm_power_reference = measurements_dict[self.cname]["power_reference"]
    #     elif ref_in_upper_dict:
    #         farm_power_reference = measurements_dict["power_reference"]
    #     else:
    #         farm_power_reference = POWER_SETPOINT_DEFAULT

    #     return {"power_setpoint": farm_power_reference}

    def set_controller_parameters(self):
        pass

    def compute_controls(self, measurements_dict):
        return {self.cname: {"power_setpoint": measurements_dict[self.cname]["power_reference"]}}
