from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesInterfaceBase(InterfaceBase):
    """
    Base class for Hercules v2 interfaces.
    This class is not intended to be instantiated directly.
    It provides a common interface for all Hercules interfaces.
    """
    def __init__(self, hercules_dict):
        super().__init__()
        self.dt = hercules_dict["dt"]

        # Controller parameters
        if "controller" in hercules_dict:
            self.controller_parameters = hercules_dict["controller"]
        else:
            self.controller_parameters = {}

        # Plant parameters
        if "plant" in hercules_dict:
            self.plant_parameters = hercules_dict["plant"]
        else:
            self.plant_parameters = {}


class HerculesWindLongRunInterface(HerculesInterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__(hercules_dict)

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

class HerculesHybridLongRunInterface(HerculesInterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__(hercules_dict)

        # Determine which components are present in the simulation
        self._has_wind_component = "wind_farm" in hercules_dict
        self._has_solar_component = "solar_farm" in hercules_dict
        self._has_battery_component = "battery" in hercules_dict
        self._has_hydrogen_component = "electrolyzer" in hercules_dict

        # Wind farm parameters
        if self._has_wind_component:
            self.wind_capacity = hercules_dict["wind_farm"]["capacity"]
            self.n_turbines = hercules_dict["wind_farm"]["num_turbines"]
            self.turbines = range(self.n_turbines)
        else:
            self.n_turbines = 0

        # Solar farm parameters
        if self._has_solar_component:
            self.solar_capacity = hercules_dict["solar_farm"]["capacity"]

        # Battery parameters
        if self._has_battery_component:
            self.battery_power_capacity = hercules_dict["battery"]["size"] * 1e3
            self.battery_energy_capacity = hercules_dict["battery"]["energy_capacity"] * 1e3

        # Electrolyzer parameters
        if self._has_hydrogen_component:
            pass # Placeholder for future electrolyzer parameters

    def check_controls(self, controls_dict):
        available_controls = [
            "wind_power_setpoints",
            "solar_power_setpoint",
            "battery_power_setpoint"
        ]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if k == "wind_power_setpoints":
                if len(controls_dict[k]) != self.n_turbines:
                    raise ValueError(
                        "Number of wind power setpoints must match number of turbines."
                    )

    def get_measurements(self, hercules_dict):
        time = hercules_dict["time"]

        # Defaults for external signals
        measurements = {
            "time": time,
        } 
        plant_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        # Handle external signals
        if "external_signals" in hercules_dict:
            if "plant_power_reference" in hercules_dict["external_signals"]:
                plant_power_reference = hercules_dict["external_signals"]["plant_power_reference"]
            else:
                plant_power_reference = POWER_SETPOINT_DEFAULT
            measurements["plant_power_reference"] = plant_power_reference

            if "wind_power_reference" in hercules_dict["external_signals"]:
                measurements["wind_power_reference"] = \
                    hercules_dict["external_signals"]["wind_power_reference"]

            if "solar_power_reference" in hercules_dict["external_signals"]:
                measurements["solar_power_reference"] = \
                    hercules_dict["external_signals"]["solar_power_reference"]

            for k in hercules_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = hercules_dict["external_signals"][k]
            measurements["forecast"] = forecast

        total_power = 0.0



        if self._has_wind_component:
            turbine_powers = hercules_dict["wind_farm"]["turbine_powers"]
            measurements["wind_turbine_powers"] =  turbine_powers
            measurements["wind_directions"] = \
                [hercules_dict["wind_farm"]["wind_direction"]]*self.n_turbines
            total_power += sum(turbine_powers)
        if self._has_solar_component:
            measurements["solar_power"] = hercules_dict["solar_farm"]["power_mw"] * 1e3
            measurements["solar_dni"] = hercules_dict["solar_farm"]["dni"]
            measurements["solar_aoi"] = hercules_dict["solar_farm"]["aoi"]
            total_power += measurements["solar_power"]
        if self._has_battery_component:
            measurements["battery_power"] = -hercules_dict["battery"]["power"] * 1e3
            measurements["battery_soc"] = hercules_dict["battery"]["soc"]
            total_power += measurements["battery_power"]
        if self._has_hydrogen_component:
            measurements["hydrogen_production_rate"] = hercules_dict["electrolyzer"]["H2_mfr"]
            if "external_signals" in hercules_dict and \
               "hydrogen_reference" in hercules_dict["external_signals"]:
                measurements["hydrogen_reference"] = \
                    hercules_dict["external_signals"]["hydrogen_reference"]
            else:
                measurements["hydrogen_reference"] = POWER_SETPOINT_DEFAULT
        measurements["total_power"] = total_power

        return measurements

    def send_controls(
            self,
            hercules_dict,
            wind_power_setpoints=None,
            solar_power_setpoint=None,
            battery_power_setpoint=None
        ):
        if wind_power_setpoints is None:
            wind_power_setpoints = [POWER_SETPOINT_DEFAULT] * self.n_turbines
        if solar_power_setpoint is None:
            solar_power_setpoint = POWER_SETPOINT_DEFAULT
        if battery_power_setpoint is None:
            battery_power_setpoint = 0.0

        if self._has_wind_component:
            for t_idx in range(self.n_turbines):
                # Set wind power setpoints for each turbine
                hercules_dict["wind_farm"][f"derating_{t_idx:03d}"] = wind_power_setpoints[t_idx]

        if self._has_solar_component:
            # Set solar power setpoint
            hercules_dict["solar_farm"]["power_setpoint_mw"] = solar_power_setpoint / 1e3

        if self._has_battery_component:
            # Set battery power setpoint (negative for discharge)
            hercules_dict["battery"]["power_setpoint"] = -battery_power_setpoint / 1e3

        return hercules_dict
