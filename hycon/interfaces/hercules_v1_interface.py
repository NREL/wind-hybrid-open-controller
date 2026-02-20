<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesADInterface(InterfaceBase):
=======
from hycon.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from hycon.interfaces.interface_base import InterfaceBase


class HerculesV1ADInterface(InterfaceBase):
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
        # wind_speeds = input_dict["hercules_comms"]\
        #                         ["amr_wind"]\
        #                         [self.wf_name]\
        #                         ["turbine_wind_speeds"]
=======
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
                # "wind_speeds":wind_speeds,
=======
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
                "turbine_powers": turbine_powers,
                "power_reference": wind_power_reference,
            },
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name][
            "turbine_power_setpoints"
        ] = power_setpoints
=======
        hercules_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_power_setpoints"] = (
            power_setpoints
        )
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py

        return hercules_dict


<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
class HerculesHybridADInterface(InterfaceBase):
=======
class HerculesV1HybridADInterface(InterfaceBase):
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
        # Grab name of wind, solar, and battery 
        self.plant_parameters = {}
        for i in py_sims:
            if tech_keys[0] in i.split('_'):
                self.solar_name = [ps for ps in py_sims if "solar" in ps][0]
                self._has_solar_component = True
            if tech_keys[1] in i.split('_'):
                self.battery_name = [ps for ps in py_sims if "battery" in ps][0]
                self._has_battery_component = True
                self.plant_parameters["battery"] = {
                    "charge_rate":hercules_dict["py_sims"][self.battery_name]["charge_rate"]*1000
                 } # Convert to kW
=======
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
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
            if tech_keys[3] in i.split("_"):
                self.hydrogen_name = [ps for ps in py_sims if "hydrogen" in ps][0]
                self._has_hydrogen_component = True

        for i in hercules_comms:
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
            if tech_keys[2] in i.split('_'):
=======
            if tech_keys[2] in i.split("_"):
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
                self.wind_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]
                self.n_turbines = hercules_dict["controller"]["num_turbines"]
                self.turbines = range(self.n_turbines)
                self._has_wind_component = True
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
                self.plant_parameters["wind_farm"] = {
                    "n_turbines": self.n_turbines
                }

    def get_measurements(self, hercules_dict):

=======
                self.plant_parameters["wind_farm"] = {"n_turbines": self.n_turbines}

    def get_measurements(self, hercules_dict):
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
                battery_power_reference = (
                    hercules_dict["external_signals"]["battery_power_reference"]
                )
=======
                battery_power_reference = hercules_dict["external_signals"][
                    "battery_power_reference"
                ]
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
            if "hydrogen_reference" in hercules_dict["external_signals"]:
                hydrogen_power_reference = hercules_dict["external_signals"]["hydrogen_reference"]

        total_power = 0.0

        measurements = {
            "time": time,
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
            "power_reference": plant_power_reference,
            "forecast": forecast,
        } 

        if self._has_wind_component:
            turbine_powers = (
                hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]["turbine_powers"]
            )
            measurements["wind_farm"] = {
                "turbine_powers": turbine_powers,
                "wind_speed": hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]\
                    ["wind_speed"],
=======
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
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
                "power_reference": wind_power_reference,
            }
            total_power += sum(turbine_powers)
        if self._has_solar_component:
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
            # solar_power converted to kW here
            # solar_dni is the direct normal irradiance
            # solar_aoi is the 
            measurements["solar_farm"] = {
                "power": hercules_dict["py_sims"][self.solar_name]["outputs"]["power_mw"] * 1000,
                "direct_normal_irradiance": hercules_dict["py_sims"][self.solar_name]["outputs"]\
                    ["dni"],
=======
            measurements["solar_farm"] = {
                "power": hercules_dict["py_sims"][self.solar_name]["outputs"]["power_mw"] * 1000,
                "direct_normal_irradiance": hercules_dict["py_sims"][self.solar_name]["outputs"][
                    "dni"
                ],
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
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
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
                "production_rate": hercules_dict["py_sims"][self.hydrogen_name]["outputs"]\
                    ["H2_mfr"],
=======
                "production_rate": hercules_dict["py_sims"][self.hydrogen_name]["outputs"][
                    "H2_mfr"
                ],
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
                "power_reference": hydrogen_power_reference,
            }
        measurements["total_power"] = total_power

        return measurements

    def check_controls(self, controls_dict):
        available_controls = [
            "wind_power_setpoints",
            "solar_power_setpoint",
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
            "battery_power_setpoint"
=======
            "battery_power_setpoint",
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
        ]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if k == "wind_power_setpoints":
                if len(controls_dict[k]) != self.n_turbines:
                    raise ValueError(
                        "Number of wind power setpoints must match number of turbines."
                    )

    def send_controls(
<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
            self,
            hercules_dict,
            wind_power_setpoints=None,
            solar_power_setpoint=None,
            battery_power_setpoint=None
        ):
=======
        self,
        hercules_dict,
        wind_power_setpoints=None,
        solar_power_setpoint=None,
        battery_power_setpoint=None,
    ):
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
        if wind_power_setpoints is None:
            wind_power_setpoints = [POWER_SETPOINT_DEFAULT] * self.n_turbines
        if solar_power_setpoint is None:
            solar_power_setpoint = POWER_SETPOINT_DEFAULT
        if battery_power_setpoint is None:
            battery_power_setpoint = 0.0

<<<<<<< HEAD:whoc/interfaces/hercules_v1_interface.py
        hercules_dict["hercules_comms"]["amr_wind"][self.wind_name][
            "turbine_power_setpoints"
        ] = wind_power_setpoints
        hercules_dict["py_sims"]["inputs"].update(
            {"battery_signal": -battery_power_setpoint,
             "solar_setpoint_mw": solar_power_setpoint / 1000} # Convert to MW
        )

        return hercules_dict
=======
        hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]["turbine_power_setpoints"] = (
            wind_power_setpoints
        )
        hercules_dict["py_sims"]["inputs"].update(
            {
                "battery_signal": -battery_power_setpoint,
                "solar_setpoint_mw": solar_power_setpoint / 1000,
            }  # Convert to MW
        )

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

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def send_controls(self, hercules_dict, power_setpoint=0):
        hercules_dict["py_sims"]["inputs"].update({"battery_signal": -power_setpoint})

        return hercules_dict


# Aliases for backward compatibility
HerculesBatteryInterface = HerculesV1BatteryInterface
HerculesADInterface = HerculesV1ADInterface
HerculesHybridADInterface = HerculesV1HybridADInterface
>>>>>>> upstream/main:hycon/interfaces/hercules_v1_interface.py
