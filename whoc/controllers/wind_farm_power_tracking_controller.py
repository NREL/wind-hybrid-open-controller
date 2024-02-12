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

POWER_SETPOINT_DEFAULT = 1e9

class WindFarmPowerDistributingController(ControllerBase):
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
        if "wind_power_reference" in self.measurements_dict:
            farm_power_reference = self.measurements_dict["wind_power_reference"]
        else:
            farm_power_reference = POWER_SETPOINT_DEFAULT
        
        self.turbine_power_references(farm_power_reference=farm_power_reference)

    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """
        
        # Handle possible bad data
        turbine_current_powers = self.measurements_dict["turbine_powers"]
        print(turbine_current_powers)
        
        # set "no value" for yaw angles (Floris not compatible with both 
        # power_setpoints and yaw_angles)
        self.controls_dict = {
            "power_setpoints": [farm_power_reference/self.n_turbines]*self.n_turbines,
            "yaw_angles": [-1000]*self.n_turbines
        }

        return None
