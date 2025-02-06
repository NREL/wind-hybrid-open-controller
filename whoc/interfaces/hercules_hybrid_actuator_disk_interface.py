from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesHybridADInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]
        py_sims = list(hercules_dict["py_sims"].keys())
        hercules_comms = list(hercules_dict["hercules_comms"].keys())
        tech_keys = ["solar", "battery", "wind"]

        self._has_solar_component = False
        self._has_wind_component = False
        self._has_battery_component = False
        # Grab name of wind, solar, and battery 
        for i in py_sims:
            if tech_keys[0] in i.split('_'):
                self.solar_name = [ps for ps in py_sims if "solar" in ps][0]
                self._has_solar_component = True
            if tech_keys[1] in i.split('_'):
                self.battery_name = [ps for ps in py_sims if "battery" in ps][0]
                self._has_battery_component = True   

        for i in hercules_comms:
            if tech_keys[2] in i.split('_'):
                self.wind_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]
                self.n_turbines = hercules_dict["controller"]["num_turbines"]
                self.turbines = range(self.n_turbines)
                self._has_wind_component = True

    def get_measurements(self, hercules_dict):

        time = hercules_dict["time"]

        if ("external_signals" in hercules_dict
            and "plant_power_reference" in hercules_dict["external_signals"]):
            plant_power_reference = hercules_dict["external_signals"]["plant_power_reference"]
        else:
            plant_power_reference = POWER_SETPOINT_DEFAULT
        measurements = {
            "time": time,
            "plant_power_reference": plant_power_reference,
        } 

        if self._has_wind_component:
            turbine_powers = (
                hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]["turbine_powers"]
            )
            measurements["wind_turbine_powers"] =  turbine_powers
            measurements["wind_speed"] =  \
                hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]["wind_speed"]
        if self._has_solar_component:
            measurements["solar_power"]= \
                hercules_dict["py_sims"][self.solar_name]["outputs"]["power_mw"] * 1000
            measurements["solar_dni"]= \
                hercules_dict["py_sims"][self.solar_name]["outputs"]["dni"]
            measurements["solar_aoi"]= \
                hercules_dict["py_sims"][self.solar_name]["outputs"]["aoi"]
        if self._has_battery_component:
            measurements["battery_power"]= \
                hercules_dict["py_sims"][self.battery_name]["outputs"]["power"]
            measurements["battery_soc"]= \
                hercules_dict["py_sims"][self.battery_name]["outputs"]["soc"]

        # Notes: solar_power converted to kW here
        # solar_dni is the direct normal irradiance
        # solar_aoi is the angle of incidence

        return measurements

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

        hercules_dict["hercules_comms"]["amr_wind"][self.wind_name][
            "turbine_power_setpoints"
        ] = wind_power_setpoints
        hercules_dict["py_sims"]["inputs"].update(
            {"battery_signal": battery_power_setpoint,
             "solar_setpoint_mw": solar_power_setpoint / 1000} # Convert to MW
        )

        return hercules_dict
