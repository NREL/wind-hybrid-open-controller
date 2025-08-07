from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesInterfaceBase(InterfaceBase):
    """
    Base class for Hercules v2 interfaces.
    This class is not intended to be instantiated directly.
    It provides a common interface for all Hercules interfaces.
    """
    def __init__(self, h_dict):
        super().__init__()
        self.dt = h_dict["dt"]

        # Controller parameters
        if "controller" in h_dict and h_dict["controller"] is not None:
            self.controller_parameters = h_dict["controller"]
        else:
            self.controller_parameters = {}

        # Plant parameters
        if "plant" in h_dict and h_dict["plant"] is not None:
            self.plant_parameters = h_dict["plant"]
        else:
            self.plant_parameters = {}

class HerculesWindLongRunInterface(HerculesInterfaceBase):
    def __init__(self, h_dict):
        super().__init__(h_dict)

        # Wind farm parameters
        if "wind_farm" not in h_dict:
            raise ValueError(
                "h_dict must contain 'wind_farm' key to use this interface."
            )
        self.plant_parameters["nameplate_capacity"] = h_dict["wind_farm"]["capacity"]
        self.plant_parameters["n_turbines"] = h_dict["wind_farm"]["n_turbines"]
        self.plant_parameters["turbines"] = range(self.plant_parameters["n_turbines"])

        # Also store n_turbines locally for convenience
        self._n_turbines = self.plant_parameters["n_turbines"]

    def get_measurements(self, h_dict):
        wind_directions = [h_dict["wind_farm"]["wind_direction"]]*self._n_turbines
        # wind_speeds = input_dict["hercules_comms"]\
        #                         ["amr_wind"]\
        #                         [self.wf_name]\
        #                         ["turbine_wind_speeds"]
        turbine_powers = h_dict["wind_farm"]["turbine_powers"]
        time = h_dict["time"]

        # Defaults for external signals
        wind_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        # Handle external signals and overwrite defaults
        if "external_signals" in h_dict:
            if "wind_power_reference" in h_dict["external_signals"]:
                wind_power_reference = h_dict["external_signals"]["wind_power_reference"]
            elif "plant_power_reference" in h_dict["external_signals"]:
                wind_power_reference = h_dict["external_signals"]["plant_power_reference"]

            for k in h_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = h_dict["external_signals"][k]

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
        available_controls = ["wind_power_setpoints"]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if len(controls_dict[k]) != self._n_turbines:
                raise ValueError(
                    "Length of setpoint " + k + " does not match the number of turbines."
                )

    def send_controls(self, h_dict, wind_power_setpoints=None):
        if wind_power_setpoints is None:
            wind_power_setpoints = [POWER_SETPOINT_DEFAULT] * self._n_turbines

        h_dict["wind_farm"]["turbine_power_setpoints"] = wind_power_setpoints

        return h_dict

class HerculesHybridLongRunInterface(HerculesInterfaceBase):
    def __init__(self, h_dict):
        super().__init__(h_dict)

        # Determine which components are present in the simulation
        self._has_wind_component = "wind_farm" in h_dict
        self._has_solar_component = "solar_farm" in h_dict
        self._has_battery_component = "battery" in h_dict
        self._has_hydrogen_component = "electrolyzer" in h_dict

        # Wind farm parameters
        if self._has_wind_component:
            self.plant_parameters["wind_capacity"] = h_dict["wind_farm"]["capacity"]
            self.plant_parameters["n_turbines"] = h_dict["wind_farm"]["n_turbines"]
            self._n_turbines = self.plant_parameters["n_turbines"]
            self.plant_parameters["turbines"] = range(self.plant_parameters["n_turbines"])
        else:
            self._n_turbines = 0

        # Solar farm parameters
        if self._has_solar_component:
            self.plant_parameters["solar_capacity"] = h_dict["solar_farm"]["capacity"]

        # Battery parameters
        if self._has_battery_component:
            self.plant_parameters["battery_power_capacity"] = h_dict["battery"]["size"] * 1e3
            self.plant_parameters["battery_energy_capacity"] = (
                h_dict["battery"]["energy_capacity"] * 1e3
            )
            self.plant_parameters["battery_charge_rate"] = (
                h_dict["battery"]["charge_rate"] * 1e3
            )

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
                if len(controls_dict[k]) != self._n_turbines:
                    raise ValueError(
                        "Number of wind power setpoints ({0})".format(len(controls_dict[k])) +
                        " must match number of turbines ({0}).".format(self._n_turbines)
                    )

    def get_measurements(self, h_dict):
        time = h_dict["time"]

        # Defaults for external signals
        measurements = {
            "time": time,
        } 
        plant_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        # Handle external signals
        if "external_signals" in h_dict:
            if "plant_power_reference" in h_dict["external_signals"]:
                plant_power_reference = h_dict["external_signals"]["plant_power_reference"]
            else:
                plant_power_reference = POWER_SETPOINT_DEFAULT
            measurements["plant_power_reference"] = plant_power_reference

            if "wind_power_reference" in h_dict["external_signals"]:
                measurements["wind_power_reference"] = \
                    h_dict["external_signals"]["wind_power_reference"]

            if "solar_power_reference" in h_dict["external_signals"]:
                measurements["solar_power_reference"] = \
                    h_dict["external_signals"]["solar_power_reference"]

            for k in h_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = h_dict["external_signals"][k]
            measurements["forecast"] = forecast

        total_power = 0.0



        if self._has_wind_component:
            turbine_powers = h_dict["wind_farm"]["turbine_powers"]
            measurements["wind_turbine_powers"] =  turbine_powers
            measurements["wind_directions"] = \
                [h_dict["wind_farm"]["wind_direction"]]*self._n_turbines
            total_power += sum(turbine_powers)
        if self._has_solar_component:
            measurements["solar_power"] = h_dict["solar_farm"]["power"]
            measurements["solar_dni"] = h_dict["solar_farm"]["dni"]
            measurements["solar_aoi"] = h_dict["solar_farm"]["aoi"]
            total_power += measurements["solar_power"]
        if self._has_battery_component:
            measurements["battery_power"] = -h_dict["battery"]["power"] * 1e3
            measurements["battery_soc"] = h_dict["battery"]["soc"]
            total_power += measurements["battery_power"]
        if self._has_hydrogen_component:
            measurements["hydrogen_production_rate"] = h_dict["electrolyzer"]["H2_mfr"]
            if "external_signals" in h_dict and \
               "hydrogen_reference" in h_dict["external_signals"]:
                measurements["hydrogen_reference"] = \
                    h_dict["external_signals"]["hydrogen_reference"]
            else:
                measurements["hydrogen_reference"] = POWER_SETPOINT_DEFAULT
        measurements["total_power"] = total_power

        return measurements

    def send_controls(
            self,
            h_dict,
            wind_power_setpoints=None,
            solar_power_setpoint=None,
            battery_power_setpoint=None
        ):
        if wind_power_setpoints is None:
            wind_power_setpoints = [POWER_SETPOINT_DEFAULT] * self._n_turbines
        if solar_power_setpoint is None:
            solar_power_setpoint = POWER_SETPOINT_DEFAULT
        if battery_power_setpoint is None:
            battery_power_setpoint = 0.0

        if self._has_wind_component:
            # Set wind power setpoints
            h_dict["wind_farm"]["turbine_power_setpoints"] = wind_power_setpoints

        if self._has_solar_component:
            # Set solar power setpoint
            h_dict["solar_farm"]["power_setpoint"] = solar_power_setpoint

        if self._has_battery_component:
            # Set battery power setpoint (negative for discharge)
            h_dict["battery"]["power_setpoint"] = -battery_power_setpoint / 1e3

        return h_dict
