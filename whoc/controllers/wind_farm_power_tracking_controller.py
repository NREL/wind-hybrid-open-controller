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

# import numpy as np

from whoc.controllers.controller_base import ControllerBase

POWER_SETPOINT_DEFAULT = 1e9

class WindFarmPowerTrackingController(ControllerBase):
    """
    Based on controller developed under A2e2g project.
    """
    def __init__(self, interface, input_dict, verbose=False):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Set initial conditions
        self.controls_dict = {"power_setpoints": [POWER_SETPOINT_DEFAULT] * self.n_turbines}

        # For startup


    def compute_controls(self):
        self.turbine_power_references()

    def turbine_power_references(self):
        
        # Handle possible bad data
        turbine_current_powers = self.measurements_dict["turbine_powers"]
        print(turbine_current_powers)
        
        self.controls_dict = {"power_setpoints": [2000]*self.n_turbines}

        return None
