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
import pandas as pd
from whoc.controllers import (
    LookupBasedWakeSteeringController,
    WindBatteryController,
)
from whoc.interfaces import HerculesADYawInterface, HerculesWindBatteryInterface
from whoc.interfaces.interface_base import InterfaceBase


class StandinInterface(InterfaceBase):
    """
    Empty class to test controllers.
    """

    def __init__(self):
        super().__init__()

    def get_measurements(self):
        pass

    def check_controls(self):
        pass

    def send_controls(self):
        pass


test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "controller": {"num_turbines": 2, "initial_conditions": {"yaw": [270.0, 270.0]}},
    "hercules_comms": {
        "amr_wind": {
            "test_farm": {
                "turbine_wind_directions": [271.0, 272.5],
                "turbine_powers": [4000.0, 4001.0],
            }
        }
    },
    "py_sims": {"test_battery": {"outputs": 10.0}},
}


def test_controller_instantiation():
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    test_interface = StandinInterface()

    _ = LookupBasedWakeSteeringController(interface=test_interface, input_dict=test_hercules_dict)
    _ = WindBatteryController(interface=test_interface, input_dict=test_hercules_dict)


def test_LookupBasedWakeSteeringController():
    test_interface = HerculesADYawInterface(test_hercules_dict)
    
    # No lookup table passed; simply passes through wind direction to yaw angles
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface,
        input_dict=test_hercules_dict
    )

    # Check that the controller can be stepped
    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_angles = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions)

    # Lookup table that specified 20 degree offset for T000, 10 degree offset for T001 for all wind
    # directions
    test_offsets = np.array([20.0, 10.0])
    df_opt_test = pd.DataFrame(data={
        "wind_direction":[220.0, 320.0, 220.0, 320.0],
        "wind_speed":[0.0, 0.0, 20.0, 20.0],
        "yaw_angles_opt":[test_offsets]*4, 
        "turbulence_intensity":[0.06]*4
    })
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface,
        input_dict=test_hercules_dict,
        df_yaw=df_opt_test
    )

    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_angles = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions - test_offsets)

def test_WindBatteryController():
    # TODO: possibly clean up HerculesWindBatteryController class

    test_interface = HerculesWindBatteryInterface(test_hercules_dict)
    test_controller = WindBatteryController(test_interface, test_hercules_dict)
    

    # Check the low level methods behave as expected
    test_controller._receive_measurements(test_hercules_dict)

    wind_setpoints = test_controller.calc_wind_setpoints()
    assert not wind_setpoints # wind setpoints should be empty

    battery_setpoints = test_controller.calc_battery_setpoints()
    assert battery_setpoints["signal"] == -500 # battery setpoints should not be empty

    test_controller.compute_controls()
    assert test_controller.setpoints_dict == {"wind": wind_setpoints, "battery": battery_setpoints}

    # Test step
    # We will need to change these cases when the wind_battery_controller has more general behavior
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [450, 450]
    hercules_dict_out = test_controller.step(test_hercules_dict)
    assert hercules_dict_out["setpoints"]["battery"]["signal"] == 900

    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [550, 550]
    hercules_dict_out = test_controller.step(test_hercules_dict)
    assert hercules_dict_out["setpoints"]["battery"]["signal"] == -500


    

