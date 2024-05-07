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
from whoc.controllers.wind_farm_power_tracking_controller import WindFarmPowerTrackingController

class HybridSupervisoryControllerSkeleton(ControllerBase):
    def __init__(
            self,
            interface,
            input_dict,
            wind_controller=None,
            solar_controller=None,
            battery_controller=None,
            verbose=False
        ):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have

        # Assign the individual asset controllers
        self.wind_controller = wind_controller
        self.solar_controller = solar_controller
        self.battery_controller = battery_controller

    def compute_controls(self):
        # Run supervisory control logic
        wind_reference, solar_reference, battery_reference = self.supervisory_control()

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
        if self.solar_controller:
            self.solar_controller.measurements_dict["solar_power_reference"] = solar_reference
            self.solar_controller.compute_controls()
            self.controls_dict["solar_power_setpoint"] = (
                self.solar_controller.controls_dict["power_setpoint"]
            )
        if self.battery_controller:
            self.battery_controller.measurements_dict["battery_power_reference"] = battery_reference
            self.battery_controller.compute_controls()
            self.controls_dict["battery_power_setpoint"] = (
                self.battery_controller.controls_dict["power_setpoint"]
            )

        return None

    def supervisory_control(self):
        # Extract current power production
        wind_power = np.array(self.measurements_dict["wind_turbine_powers"]).sum()
        solar_power = self.measurements_dict["solar_power"]
        battery_power = self.measurements_dict["battery_power"]

        # Temporary print statements
        print("Measured powers (wind, solar, battery):", wind_power, solar_power, battery_power)

        # Placeholder for supervisory control logic
        wind_reference = 2.5 #1e9 # Unit is kW
        solar_reference = 0.5e9 # Check if this is working
        battery_reference = 3.0 # 0.5e9

        return wind_reference, solar_reference, battery_reference
