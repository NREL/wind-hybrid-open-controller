from hycon.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from hycon.interfaces.interface_base import InterfaceBase


class HerculesV1ADInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]
        self.n_turbines = hercules_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Grab name of wind farm (assumes there is only one!)
        self.wf_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]

        # Assign plant parameters for controller use
        self.plant_parameters = {"n_turbines": self.n_turbines}

        pass

    def get_measurements(self, hercules_dict):
        wind_directions = hercules_dict["hercules_comms"]["amr_wind"][self.wf_name][
            "turbine_wind_directions"
        ]
        turbine_powers = hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_powers"]
        time = hercules_dict["time"]

        # Defaults for external signals
        wind_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        # Handle external signals. wind_power_reference takes precedence over plant_power_reference.
        if "external_signals" in hercules_dict:
            if "wind_power_reference" in hercules_dict["external_signals"]:
                wind_power_reference = hercules_dict["external_signals"]["wind_power_reference"]
            elif "plant_power_reference" in hercules_dict["external_signals"]:
                wind_power_reference = hercules_dict["external_signals"]["plant_power_reference"]

            for k in hercules_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = hercules_dict["external_signals"][k]

        measurements = {
            "time": time,
            "total_power": sum(turbine_powers),
            "forecast": forecast,
            "wind_farm": {
                "wind_directions": wind_directions,
                "turbine_powers": turbine_powers,
                "power_reference": wind_power_reference,
            },
        }

        return measurements

    def check_controls(self, controls_dict):
        available_controls = ["yaw_angles", "power_setpoints"]

        for c in controls_dict.keys():
            for k in controls_dict[c].keys():
                if k not in available_controls:
                    raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def send_controls(self, hercules_dict, controls_dict):
        yaw_angles = controls_dict["wind_farm"].get("yaw_angles", [-1000] * self.n_turbines)
        power_setpoints = controls_dict["wind_farm"].get(
            "power_setpoints", [POWER_SETPOINT_DEFAULT] * self.n_turbines
        )

        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_yaw_angles"] = yaw_angles
        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_power_setpoints"] = (
            power_setpoints
        )

        return hercules_dict


class HerculesV1HybridADInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]
        py_sims = list(hercules_dict["py_sims"].keys())
        hercules_comms = list(hercules_dict["hercules_comms"].keys())
        tech_keys = ["solar", "battery", "wind", "hydrogen"]

        self._has_solar_component = False
        self._has_wind_component = False
        self._has_battery_component = False
        self._has_hydrogen_component = False
        # Grab name of wind, solar, and battery
        self.plant_parameters = {}
        for i in py_sims:
            if tech_keys[0] in i.split("_"):
                self.solar_name = [ps for ps in py_sims if "solar" in ps][0]
                self._has_solar_component = True
            if tech_keys[1] in i.split("_"):
                self.battery_name = [ps for ps in py_sims if "battery" in ps][0]
                self._has_battery_component = True
                self.plant_parameters["battery"] = {
                    "charge_rate": hercules_dict["py_sims"][self.battery_name]["charge_rate"]
                    * 1000,
                    "discharge_rate": hercules_dict["py_sims"][self.battery_name]["discharge_rate"]
                    * 1000,
                }  # Convert to kW
            if tech_keys[3] in i.split("_"):
                self.hydrogen_name = [ps for ps in py_sims if "hydrogen" in ps][0]
                self._has_hydrogen_component = True

        for i in hercules_comms:
            if tech_keys[2] in i.split("_"):
                self.wind_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]
                self.n_turbines = hercules_dict["controller"]["num_turbines"]
                self.turbines = range(self.n_turbines)
                self._has_wind_component = True
                self.plant_parameters["wind_farm"] = {"n_turbines": self.n_turbines}

    def get_measurements(self, hercules_dict):
        time = hercules_dict["time"]

        # Defaults for external signals
        plant_power_reference = POWER_SETPOINT_DEFAULT
        forecast = {}

        wind_power_reference = None
        solar_power_reference = None
        battery_power_reference = None
        hydrogen_power_reference = None

        # Handle external signals
        if "external_signals" in hercules_dict:
            if "plant_power_reference" in hercules_dict["external_signals"]:
                plant_power_reference = hercules_dict["external_signals"]["plant_power_reference"]

            for k in hercules_dict["external_signals"].keys():
                if "forecast" in k != "wind_power_reference":
                    forecast[k] = hercules_dict["external_signals"][k]

            if "wind_power_reference" in hercules_dict["external_signals"]:
                wind_power_reference = hercules_dict["external_signals"]["wind_power_reference"]
            if "solar_power_reference" in hercules_dict["external_signals"]:
                solar_power_reference = hercules_dict["external_signals"]["solar_power_reference"]
            if "battery_power_reference" in hercules_dict["external_signals"]:
                battery_power_reference = hercules_dict["external_signals"][
                    "battery_power_reference"
                ]
            if "hydrogen_reference" in hercules_dict["external_signals"]:
                hydrogen_power_reference = hercules_dict["external_signals"]["hydrogen_reference"]

        total_power = 0.0

        measurements = {
            "time": time,
            "plant_power_reference": plant_power_reference,
            "forecast": forecast,
        }

        if self._has_wind_component:
            turbine_powers = hercules_dict["hercules_comms"]["amr_wind"][self.wind_name][
                "turbine_powers"
            ]
            measurements["wind_farm"] = {
                "turbine_powers": turbine_powers,
                "wind_speed": hercules_dict["hercules_comms"]["amr_wind"][self.wind_name][
                    "wind_speed"
                ],
                "power_reference": wind_power_reference,
            }
            total_power += sum(turbine_powers)
        if self._has_solar_component:
            measurements["solar_farm"] = {
                "power": hercules_dict["py_sims"][self.solar_name]["outputs"]["power_mw"] * 1000,
                "direct_normal_irradiance": hercules_dict["py_sims"][self.solar_name]["outputs"][
                    "dni"
                ],
                "angle_of_incidence": hercules_dict["py_sims"][self.solar_name]["outputs"]["aoi"],
                "power_reference": solar_power_reference,
            }
            total_power += measurements["solar_farm"]["power"]
        if self._has_battery_component:
            measurements["battery"] = {
                "power": -hercules_dict["py_sims"][self.battery_name]["outputs"]["power"],
                "state_of_charge": hercules_dict["py_sims"][self.battery_name]["outputs"]["soc"],
                "power_reference": battery_power_reference,
            }
            total_power += measurements["battery"]["power"]
        if self._has_hydrogen_component:
            # hydrogen production rate in kg/s
            measurements["hydrogen"] = {
                "production_rate": hercules_dict["py_sims"][self.hydrogen_name]["outputs"][
                    "H2_mfr"
                ],
                "power_reference": hydrogen_power_reference,
            }
        measurements["total_power"] = total_power

        return measurements

    def check_controls(self, controls_dict):
        available_controls = [
            "power_setpoint",
            "power_setpoints",
            "yaw_angles",
        ]

        for c in controls_dict.keys():
            for k in controls_dict[c].keys():
                if k not in available_controls:
                    raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def send_controls(
        self,
        hercules_dict,
        controls_dict,
    ):
        if self._has_wind_component:
            wind_power_setpoints = controls_dict["wind_farm"].get(
                "power_setpoints", [POWER_SETPOINT_DEFAULT] * self.n_turbines
            )
            hercules_dict["hercules_comms"]["amr_wind"][self.wind_name][
                "turbine_power_setpoints"
            ] = wind_power_setpoints

        if self._has_solar_component:
            solar_power_setpoint = controls_dict["solar_farm"].get(
                "power_setpoint", POWER_SETPOINT_DEFAULT
            )
            hercules_dict["py_sims"]["inputs"].update(
                {"solar_setpoint_mw": solar_power_setpoint / 1000}
            )  # Convert to MW

        if self._has_battery_component:
            battery_power_setpoint = controls_dict["battery"].get("battery_power_setpoint", 0.0)
            hercules_dict["py_sims"]["inputs"].update(
                {"battery_signal": -battery_power_setpoint}
            )  # Negative because of convention in battery sim

        return hercules_dict


class HerculesV1BatteryInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]

        # Grab name of battery (assumes there is only one! Takes the first)
        batteries_in_simulation = [k for k in hercules_dict["py_sims"] if "battery" in k]
        if len(batteries_in_simulation) == 0:
            raise ValueError("No battery found in simulation.")
        elif len(batteries_in_simulation) > 1:
            raise ValueError("Multiple batteries found in simulation. Only one is allowed.")
        else:
            self.battery_name = batteries_in_simulation[0]

        self.plant_parameters = {
            "battery": {
                "charge_rate": hercules_dict["py_sims"][self.battery_name]["charge_rate"] * 1000,
                "discharge_rate": hercules_dict["py_sims"][self.battery_name]["discharge_rate"]
                * 1000,
            }
        }

    def get_measurements(self, hercules_dict):
        # Extract externally-provided power signal
        if (
            "external_signals" in hercules_dict
            and "plant_power_reference" in hercules_dict["external_signals"]
        ):
            plant_power_reference = hercules_dict["external_signals"]["plant_power_reference"]
        else:
            plant_power_reference = 0

        measurements = {
            "time": hercules_dict["time"],
            "battery": {
                "power_reference": plant_power_reference,
                "power": -hercules_dict["py_sims"][self.battery_name]["outputs"]["power"],
                "state_of_charge": hercules_dict["py_sims"][self.battery_name]["outputs"]["soc"],
            },
        }

        return measurements

    def check_controls(self, controls_dict):
        available_controls = ["power_setpoint"]

        for c in controls_dict.keys():
            for k in controls_dict[c].keys():
                if k not in available_controls:
                    raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def send_controls(self, hercules_dict, controls_dict):
        hercules_dict["py_sims"]["inputs"].update(
            {"battery_signal": -controls_dict["battery"].get("power_setpoint", 0.0)}
        )

        return hercules_dict


# Aliases for backward compatibility
HerculesBatteryInterface = HerculesV1BatteryInterface
HerculesADInterface = HerculesV1ADInterface
HerculesHybridADInterface = HerculesV1HybridADInterface
