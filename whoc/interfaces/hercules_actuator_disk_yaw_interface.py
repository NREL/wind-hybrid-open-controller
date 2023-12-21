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
from whoc.interfaces.interface_base import InterfaceBase


class HerculesADYawInterface(InterfaceBase):
    def __init__(self, input_dict):
        super().__init__()

        self.dt = input_dict["dt"]
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Grab name of wind farm (assumes there is only one!)
        self.wf_name = list(input_dict["hercules_comms"]["amr_wind"].keys())[0]

        pass

    def get_measurements(self, input_dict):
        wind_directions = input_dict["hercules_comms"]["amr_wind"][self.wf_name][
            "turbine_wind_directions"
        ]
        # wind_speeds = input_dict["hercules_comms"]\
        #                         ["amr_wind"]\
        #                         [self.wf_name]\
        #                         ["turbine_wind_speeds"]
        powers = input_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_powers"]
        time = input_dict["time"]

        measurements = {
            "time": time,
            "wind_directions": wind_directions,
            # "wind_speeds":wind_speeds,
            "turbine_powers": powers,
        }

        return measurements

    def check_controls(self, setpoints_dict):
        available_setpoints = ["yaw_angles"]

        for k in setpoints_dict.keys():
            if k not in available_setpoints:
                raise ValueError("Setpoint " + k + " is not available in this configuration")

    def send_controls(self, input_dict, yaw_angles=None):
        if yaw_angles is None:
            yaw_angles = [0.0] * self.n_turbines

        input_dict["hercules_comms"]["amr_wind"][self.wf_name]["turbine_yaw_angles"] = yaw_angles

        return input_dict
