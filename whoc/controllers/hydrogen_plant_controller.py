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

import numpy as np

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

        # Initialize Power references
        self.wind_reference = 0 # TODO: Unused, remove?

        # Initialize filter
        self.filtered_power_prev = 0

    def compute_controls(self, measurements_dict):
        # Run supervisory control logic
        power_reference = self.supervisory_control(measurements_dict)

        # Package the controls for the individual controllers, step, and return
        controls_dict = {}
        if self.generator_controller:
            generator_measurements_dict = {
                "power_reference": power_reference,
                "wind_turbine_powers": measurements_dict["wind_turbine_powers"],
                "total_power": measurements_dict["total_power"],
            }
            wind_controls_dict = self.generator_controller.compute_controls(
                generator_measurements_dict
            )
            controls_dict["wind_power_setpoints"] = wind_controls_dict["wind_power_setpoints"]
            # TODO: Do I need to unpack other setpoints here?
        print('Wind ref, final', controls_dict["wind_power_setpoints"])

        return controls_dict

    def supervisory_control(self, measurements_dict):
        # Extract measurements sent
        time = measurements_dict["time"] # noqa: F841 
        current_power = measurements_dict["total_power"]
        hydrogen_output = measurements_dict["hydrogen_output"]
        wind_speed = measurements_dict["wind_speed"] # noqa: F841
        hydrogen_reference = measurements_dict["hydrogen_reference"]

        # Input filtering
        a = 0.05
        filtered_power = (1-a/self.dt)*self.filtered_power_prev + a/self.dt*current_power

        # TODO: Temporary print statements (note that negative battery indicates discharging)
        print("Power generated (filtered):", filtered_power)
        print("Current hydrogen:", hydrogen_output)
        print("Reference hydrogen:", hydrogen_reference)

        # Calculate difference between hydrogen reference and hydrogen actual
        hydrogen_error = hydrogen_reference - hydrogen_output
        if filtered_power > 0:
            power_scaling = filtered_power
        else:
            power_scaling = 100 # MS TODO: check for appropriate default?
        if hydrogen_output > 0:
            h2_scaling = hydrogen_output
        else:
            h2_scaling = hydrogen_reference

        # Scale gain by hydrogen output
        K = power_scaling/h2_scaling

        # Apply gain to generator power output
        power_reference = filtered_power + K * hydrogen_error

        print("Power reference value", power_reference) # TODO: remove when happy

        self.filtered_power_prev = filtered_power
        self.wind_reference = power_reference # TODO: Unused, remove?

        # # Placeholder for supervisory control logic
        # wind_reference = 20000 # kW
        # solar_reference = 5000 # kW, not currently working
        # battery_reference = -30 # kW, Negative requests discharging, positive requests charging

        print('wind reference', power_reference)
        return power_reference