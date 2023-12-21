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

from whoc.controllers.controller_base import ControllerBase


class WakeSteeringROSCOStandin(ControllerBase):
    def __init__(self, interface):
        super.__init__(interface, timeout=100.0, verbose=True)

    def compute_controls(self):
        self.generate_turbine_references()

    def generate_turbine_references(self):
        # Something very minimal here, based on ROSCO example 17.
        # west_offset = convert_absolute_nacelle_heading_to_offset(270,
        #    self.measurements_dict["NacelleHeading"])

        current_time = self.measurements_dict["Time"]
        if current_time <= 10.0:
            yaw_setpoint = 0.0
        else:
            yaw_setpoint = 20.0

        self.controls_dict = {
            "turbine_ID": 0,  # TODO: hardcoded! Replace.
            "genTorque": 0.0,
            "nacelleHeading": yaw_setpoint,
            "bladePitch": [0.0, 0.0, 0.0],
        }

        return None

    # def run(self):

    #     connect_zmq = True
    #     while connect_zmq:
    #         self.receive_turbine_outputs()
    #         self.generate_turbine_references()
    #         self.send_turbine_references()

    #         if self.measurements_dict['iStatus'] == -1:
    #             connect_zmq = False
    #             self.s._disconnect()
