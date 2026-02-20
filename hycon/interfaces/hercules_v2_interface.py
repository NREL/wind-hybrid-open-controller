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
            raise ValueError("h_dict must contain 'wind_farm' key to use this interface.")
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
            self.plant_parameters["wind_farm"] = {
                "capacity": h_dict["wind_farm"]["capacity"],
                "n_turbines": h_dict["wind_farm"]["n_turbines"],
                "turbines": range(h_dict["wind_farm"]["n_turbines"]),
            }
            self._n_turbines = self.plant_parameters["wind_farm"]["n_turbines"]
        else:
            self._n_turbines = 0

        # Solar farm parameters
        if self._has_solar_component:
            self.plant_parameters["solar_farm"] = {
                "capacity": h_dict["solar_farm"]["capacity"]
            }

        # Battery parameters
        if self._has_battery_component:
            self.plant_parameters["battery"] = {
                "power_capacity": h_dict["battery"]["size"],
                "energy_capacity": h_dict["battery"]["energy_capacity"],
                "charge_rate": h_dict["battery"]["charge_rate"],
                "discharge_rate": h_dict["battery"]["discharge_rate"],
            }

        # Electrolyzer parameters (placeholder for future electrolyzer parameters)
        if self._has_hydrogen_component:
            self.plant_parameters["hydrogen"] = {}

    def check_controls(self, controls_dict):
        available_controls = [
            "wind_power_setpoints",
            "solar_power_setpoint",
            "battery_power_setpoint",
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

        # Set up placeholder dictionary
        measurements = {
            "time": time,
            "forecast": {},
            "wind_farm": {},
            "solar_farm": {},
            "battery": {},
            "hydrogen": {},
        }

        # Handle external signals (parse and pass to individual components)
        if "external_signals" in h_dict:

            # TODO: wind_reference, hydrogen_reference, etc. should likely 
            # by copied onto their specific subdicts
            if "plant_power_reference" in h_dict["external_signals"]:
                measurements["plant_power_reference"] = (
                    h_dict["external_signals"]["plant_power_reference"]
                )

            if "wind_power_reference" in h_dict["external_signals"]:
                wind_power_reference = h_dict["external_signals"]["wind_power_reference"]
            else:
                wind_power_reference = None

            if "solar_power_reference" in h_dict["external_signals"]:
                solar_power_reference = h_dict["external_signals"]["solar_power_reference"]
            else:
                solar_power_reference = None

            if "battery_power_reference" in h_dict["external_signals"]:
                battery_power_reference = h_dict["external_signals"]["battery_power_reference"]
            else:
                battery_power_reference = None

            # TODO: other battery components in external_signals?

            if "hydrogen_reference" in h_dict["external_signals"]:
                hydrogen_reference = h_dict["external_signals"]["hydrogen_reference"]
            else:
                hydrogen_reference = None

            # Special handling for forecast elements
            for k in h_dict["external_signals"].keys():
                if "forecast" in k:
                    measurements["forecast"][k] = h_dict["external_signals"][k]

        total_power = 0.0

        if self._has_wind_component:
            measurements["wind_farm"] = {
                "turbine_powers": h_dict["wind_farm"]["turbine_powers"],
                "wind_directions": [h_dict["wind_farm"]["wind_direction"]]*self._n_turbines,
                # TODO: wind_speeds?
                "power_reference": wind_power_reference,
            }
            total_power += sum(measurements["wind_farm"]["turbine_powers"])
        if self._has_solar_component:
            measurements["solar_farm"] = {
                "power": h_dict["solar_farm"]["power"],
                "direct_normal_irradiance": h_dict["solar_farm"]["dni"],
                "angle_of_incidence": h_dict["solar_farm"]["aoi"],
                "power_reference": solar_power_reference,
            }
            total_power += measurements["solar_farm"]["power"]
        if self._has_battery_component:
            measurements["battery"] = {
                "power": h_dict["battery"]["power"],
                "state_of_charge": h_dict["battery"]["soc"],
                "power_reference": battery_power_reference,
            }
            total_power += measurements["battery"]["power"]
        if self._has_hydrogen_component:
            measurements["hydrogen"] = {
                "production_rate": h_dict["electrolyzer"]["H2_mfr"],
                "power_reference": hydrogen_reference,
            }

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
            # Set battery power setpoint (positive for discharge)
            h_dict["battery"]["power_setpoint"] = battery_power_setpoint

        return h_dict
