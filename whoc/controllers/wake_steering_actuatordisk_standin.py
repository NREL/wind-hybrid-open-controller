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

from whoc.controller_base import ControllerBase


class WakeSteeringADStandin(ControllerBase):
    def __init__(self, interface, input_dict):
        super().__init__(interface)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Set initial conditions
        yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        if hasattr(yaw_IC, "__len__"):
            if len(yaw_IC) == self.n_turbines:
                self.setpoints_dict = {"yaw_angles": yaw_IC}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.setpoints_dict = {"yaw_angles": [yaw_IC] * self.n_turbines}

        # Grab name of wind farm (assumes there is only one!)

    def compute_controls(self):
        self.generate_turbine_references()

    def generate_turbine_references(self):
        # Based on an early implementation for Hercules

        current_time = self.measurements_dict["time"]
        if current_time <= 10.0:
            yaw_setpoint = [270.0] * self.n_turbines
        else:
            yaw_setpoint = self.measurements_dict["wind_directions"]

        self.setpoints_dict = {"yaw_angles": yaw_setpoint}

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
