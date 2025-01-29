from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesADInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]
        self.n_turbines = hercules_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Grab name of wind farm (assumes there is only one!)
        self.wf_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]

        pass

    def get_measurements(self, hercules_dict):
        wind_directions = hercules_dict["hercules_comms"]["amr_wind"][self.wf_name][
            "turbine_wind_directions"
        ]
        # wind_speeds = input_dict["hercules_comms"]\
        #                         ["amr_wind"]\
        #                         [self.wf_name]\
        #                         ["turbine_wind_speeds"]
        turbine_powers = hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_powers"]
        time = hercules_dict["time"]

        if ("external_signals" in hercules_dict
            and "wind_power_reference" in hercules_dict["external_signals"]):
            wind_power_reference = hercules_dict["external_signals"]["wind_power_reference"]
        else:
            wind_power_reference = POWER_SETPOINT_DEFAULT

        measurements = {
            "time": time,
            "wind_directions": wind_directions,
            # "wind_speeds":wind_speeds,
            "turbine_powers": turbine_powers,
            "wind_power_reference": wind_power_reference,
        }

        return measurements

    def check_controls(self, controls_dict):
        available_controls = ["yaw_angles", "power_setpoints"]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if len(controls_dict[k]) != self.n_turbines:
                raise ValueError(
                    "Length of setpoint " + k + " does not match the number of turbines."
                )

    def send_controls(self, hercules_dict, yaw_angles=None, power_setpoints=None):
        if yaw_angles is None:
            yaw_angles = [-1000] * self.n_turbines
        if power_setpoints is None:
            power_setpoints = [POWER_SETPOINT_DEFAULT] * self.n_turbines

        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_yaw_angles"] = yaw_angles
        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name][
            "turbine_power_setpoints"
        ] = power_setpoints

        return hercules_dict
