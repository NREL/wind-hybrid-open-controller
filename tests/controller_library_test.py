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

from whoc.controllers import (
    HerculesWindBatteryController,
    WakeSteeringADStandin,
)
from whoc.interfaces import HerculesADYawInterface
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

    _ = WakeSteeringADStandin(interface=test_interface, input_dict=test_hercules_dict)
    _ = HerculesWindBatteryController(interface=test_interface, input_dict=test_hercules_dict)


def test_WakeSteeringADStandin():
    test_interface = HerculesADYawInterface(test_hercules_dict)
    test_controller = WakeSteeringADStandin(interface=test_interface, input_dict=test_hercules_dict)

    # Check that the controller can be stepped
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    assert test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"][
        "turbine_yaw_angles"
    ] == [270.0, 270.0]

    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    assert (
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )


def test_HerculesWindBatteryController():
    # TODO: write this test
    pass
