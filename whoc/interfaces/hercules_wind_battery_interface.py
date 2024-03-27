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

from whoc.interfaces.interface_base import InterfaceBase


class HerculesWindBatteryInterface(InterfaceBase):
    def __init__(self, hercules_dict):
        super().__init__()

        # Grab name of wind farm (assumes there is only one!)
        self.wf_name = list(hercules_dict["hercules_comms"]["amr_wind"].keys())[0]

        # Get the name of the battery (assumes the battery is the only pysim!)
        self.battery_name = list(hercules_dict["py_sims"].keys())[0]

    def get_measurements(self, hercules_dict):
        measurements = {
            "py_sims": {"battery": hercules_dict["py_sims"][self.battery_name]["outputs"]},
            "wind_farm": {
                "turbine_powers": hercules_dict["hercules_comms"]["amr_wind"][self.wf_name][
                    "turbine_powers"
                ],
                "turbine_wind_directions": hercules_dict["hercules_comms"]["amr_wind"][
                    self.wf_name
                ]["turbine_wind_directions"],
            },
        }

        return measurements

    def check_controls(self, controls_dict):
        controls = {}
        return controls

    def send_controls(self, hercules_dict, controls_dict=None):
        hercules_dict["py_sims"]["inputs"].update(
            {"battery_signal": controls_dict["battery"]["signal"]}
        )
        return hercules_dict
