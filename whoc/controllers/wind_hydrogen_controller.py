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


class WindHydrogenController(ControllerBase):
    def __init__(
            self,
            interface,
            input_dict,
            wind_controller=None,
            verbose=False
        ):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have

        # Assign the individual asset controllers
        self.wind_controller = wind_controller

        # # Set constants
        # py_sims = list(input_dict["py_sims"].keys())

        # Initialize Power references
        self.wind_reference = 0

        self.prev_wind_power = 0

    def compute_controls(self):
        # Run supervisory control logic
        wind_reference = self.supervisory_control()

        # Package the controls for the individual controllers, step, and return
        self.controls_dict = {}
        if self.wind_controller:
            self.wind_controller.measurements_dict["wind_power_reference"] = wind_reference
            self.wind_controller.measurements_dict["turbine_powers"] = (
                self.measurements_dict["wind_turbine_powers"]
            )
            self.wind_controller.compute_controls()
            self.controls_dict["wind_power_setpoints"] = (
                self.wind_controller.controls_dict["power_setpoints"]
            )

        return None

    def supervisory_control(self):
        # Extract measurements sent
        time = self.measurements_dict["time"] # noqa: F841 
        wind_power = np.array(self.measurements_dict["wind_turbine_powers"]).sum()
        hydrogen_output = self.measurements_dict["hydrogen_output"]
        wind_speed = self.measurements_dict["wind_speed"] # noqa: F841
        reference_hydrogen = self.measurements_dict["hydrogen_reference"]

        a = 0.1
        wind_power = (1-a)*self.prev_wind_power + a*wind_power

        # Temporary print statements (note that negative battery indicates discharging)
        print("Measured powers (wind):", wind_power)
        print("Current hydrogen:", hydrogen_output)
        print("Reference hydrogen:", reference_hydrogen)

        # Calculate difference between hydrogen reference and hydrogen actual
        hydrogen_difference = reference_hydrogen - hydrogen_output

        # Scale gain by hydrogen output
        K = wind_power/hydrogen_output

        # Apply gain to wind power output
        wind_reference = wind_power + K * hydrogen_difference

        print(
            "Power reference value (wind)",
            wind_reference
        )

        self.prev_wind_power = wind_power
        self.wind_reference = wind_reference

        # # Placeholder for supervisory control logic
        # wind_reference = 20000 # kW
        # solar_reference = 5000 # kW, not currently working
        # battery_reference = -30 # kW, Negative requests discharging, positive requests charging

        return wind_reference