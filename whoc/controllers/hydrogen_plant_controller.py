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

import copy

from whoc.controllers.controller_base import ControllerBase


class HydrogenPlantController(ControllerBase):
    def __init__(
            self,
            interface,
            input_dict,
            generator_controller=None,
            verbose=False
        ):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have

        # Assign the individual asset controllers
        self.generator_controller = generator_controller

        # # Set constants
        # py_sims = list(input_dict["py_sims"].keys())

        # Set K from plant inputs
        nominal_plant_power_kW = input_dict['controller']['nominal_plant_power_kW']
        nominal_hydrogen_rate_kgps = input_dict['controller']['nominal_hydrogen_rate_kgps']
        hydrogen_controller_gain = input_dict['controller']['hydrogen_controller_gain']

        self.K = nominal_plant_power_kW / nominal_hydrogen_rate_kgps * hydrogen_controller_gain   
        # Initialize filter
        self.filtered_power_prev = 0

    def compute_controls(self, measurements_dict):
        # Run supervisory control logic
        power_reference = self.supervisory_control(measurements_dict)

        # Package the controls for the individual controllers, step, and return
        if self.generator_controller:
            # Create exhaustive generator measurements dict to handle variety
            # of possible lower-level controllers
            generator_measurements_dict = copy.deepcopy(measurements_dict)
            generator_measurements_dict["power_reference"] = power_reference
            generator_controls_dict = self.generator_controller.compute_controls(
                generator_measurements_dict
            )
            if "yaw_angles" in generator_controls_dict:
                del generator_controls_dict["yaw_angles"]

        return generator_controls_dict

    def supervisory_control(self, measurements_dict):
        # Extract measurements sent
        time = measurements_dict["time"] # noqa: F841 
        current_power = measurements_dict["total_power"]
        hydrogen_output = measurements_dict["hydrogen_production_rate"]
        wind_speed = measurements_dict["wind_speed"] # noqa: F841
        hydrogen_reference = measurements_dict["hydrogen_reference"]

        # Input filtering
        a = 0.05
        filtered_power = (1-a/self.dt)*self.filtered_power_prev + a/self.dt*current_power

        # Calculate difference between hydrogen reference and hydrogen actual
        hydrogen_error = hydrogen_reference - hydrogen_output

        # Apply gain to generator power output
        power_reference = filtered_power + self.K * hydrogen_error

        if power_reference < 0:
            power_reference = 0
            
        self.filtered_power_prev = filtered_power

        return power_reference
