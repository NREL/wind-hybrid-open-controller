# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

# How will we handle other things here? May need to have a wind farm
# version, an electrolyzer version, etc...
from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces.interface_base import InterfaceBase


class HerculesHybridADInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        self.dt = hercules_dict["dt"]
        self.n_turbines = hercules_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Grab name of wind, solar, and battery (assumes there is only one!)
        self.wind_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]
        py_sims = list(hercules_dict["py_sims"].keys())
        self.solar_name = [ps for ps in py_sims if "solar" in ps][0]
        self.battery_name = [ps for ps in py_sims if "battery" in ps][0]

    def get_measurements(self, hercules_dict):
        turbine_powers = (
            hercules_dict["hercules_comms"]["amr_wind"][self.wind_name]["turbine_powers"]
        )
        time = hercules_dict["time"]

        if ("external_signals" in hercules_dict
            and "plant_power_reference" in hercules_dict["external_signals"]):
            plant_power_reference = hercules_dict["external_signals"]["wind_power_reference"]
        else:
            plant_power_reference = POWER_SETPOINT_DEFAULT

        measurements = {
            "time": time,
            "wind_turbine_powers": turbine_powers,
            "plant_power_reference": plant_power_reference,
            "battery_power": hercules_dict["py_sims"][self.battery_name]["outputs"]["power"],
            "solar_power": hercules_dict["py_sims"][self.solar_name]["outputs"]["power_mw"] * 1000,
        } # Note: solar_power converted to kW here

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
