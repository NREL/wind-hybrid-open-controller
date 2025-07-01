from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesLongRunInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        # Simulation parameters
        self.dt = hercules_dict["dt"]

        # Wind farm parameters
        if "wind_farm" not in hercules_dict:
            raise ValueError(
                "hercules_dict must contain 'wind_farm' key to use this interface."
            )
        self.nameplate_capacity = hercules_dict["wind_farm"]["capacity"]
        self.n_turbines = hercules_dict["wind_farm"]["num_turbines"]
        self.turbines = range(self.n_turbines)

    def get_measurements(self, hercules_dict):
        wind_directions = [hercules_dict["wind_farm"]["wind_direction"]]*self.n_turbines
        # wind_speeds = input_dict["hercules_comms"]\
        #                         ["amr_wind"]\
        #                         [self.wf_name]\
        #                         ["turbine_wind_speeds"]
        turbine_powers = hercules_dict["wind_farm"]["turbine_powers"]
        time = hercules_dict["time"]

        # Defaults for external signals
        wind_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        # Handle external signals and overwrite defaults
        if "external_signals" in hercules_dict:
            if "wind_power_reference" in hercules_dict["external_signals"]:
                wind_power_reference = hercules_dict["external_signals"]["wind_power_reference"]

            for k in hercules_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = hercules_dict["external_signals"][k]

        measurements = {
            "time": time,
            "wind_directions": wind_directions,
            # "wind_speeds":wind_speeds,
            "wind_turbine_powers": turbine_powers,
            "wind_power_reference": wind_power_reference,
            "forecast": forecast,
        }

        return measurements

    def check_controls(self, controls_dict):
        # TODO: Implement yaw angles for this interface
        available_controls = ["power_setpoints"]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if len(controls_dict[k]) != self.n_turbines:
                raise ValueError(
                    "Length of setpoint " + k + " does not match the number of turbines."
                )

    def send_controls(self, hercules_dict, yaw_angles=None, power_setpoints=None):
        if yaw_angles is not None:
            raise NotImplementedError("TO DO: Implement yaw angles for this interface")
        # if yaw_angles is None:
        #     yaw_angles = [-1000] * self.n_turbines
        if power_setpoints is None:
            power_setpoints = [POWER_SETPOINT_DEFAULT] * self.n_turbines

        for t_idx in range(self.n_turbines):
            hercules_dict["wind_farm"][f"derating_{t_idx:03d}"] = power_setpoints[t_idx]

        return hercules_dict

class HerculesHybridLongRunInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        # Simulation parameters
        self.dt = hercules_dict["dt"]

        # Determine which components are present in the simulation
        self.contains_wind = "wind_farm" in hercules_dict
        self.contains_solar = "solar_farm" in hercules_dict
        self.contains_battery = "battery" in hercules_dict

        # Wind farm parameters
        if self.contains_wind:
            self.wind_capacity = hercules_dict["wind_farm"]["capacity"]
            self.n_turbines = hercules_dict["wind_farm"]["num_turbines"]
            self.turbines = range(self.n_turbines)

        # Solar farm parameters
        if self.contains_solar:
            self.solar_capacity = hercules_dict["solar_farm"]["capacity"]

        # Battery parameters
        if self.contains_battery:
            self.battery_power_capacity = hercules_dict["battery"]["size"] * 1e3
            self.battery_energy_capacity = hercules_dict["battery"]["energy_capacity"] * 1e3

    def check_controls(self):
        return None

    def get_measurements(self):
        return None

    def send_controls(self):
        return None
