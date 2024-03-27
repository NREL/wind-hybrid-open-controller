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

import pytest
from whoc.interfaces import (
    HerculesADInterface,
    HerculesWindBatteryInterface,
)

test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "controller": {"num_turbines": 2},
    "hercules_comms": {
        "amr_wind": {
            "test_farm": {
                "turbine_wind_directions": [271.0, 272.5],
                "turbine_powers": [4000.0, 4001.0],
            }
        }
    },
    "py_sims": {"test_battery": {"outputs": 10.0}, "inputs": {}},
    "external_signals": {"wind_power_reference": 1000.0},
}


def test_interface_instantiation():
    """
    Tests whether all interfaces can be imported correctly and that they
    each implement the required methods specified by InterfaceBase.
    """

    _ = HerculesADInterface(hercules_dict=test_hercules_dict)
    _ = HerculesWindBatteryInterface(hercules_dict=test_hercules_dict)
    # _ = ROSCO_ZMQInterface()


def test_HerculesADInterface():
    interface = HerculesADInterface(hercules_dict=test_hercules_dict)

    # Test get_measurements()
    measurements = interface.get_measurements(hercules_dict=test_hercules_dict)

    assert measurements["time"] == test_hercules_dict["time"]
    assert (
        measurements["wind_directions"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert (
        measurements["turbine_powers"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    )

    # Test check_controls()
    controls_dict = {"yaw_angles": [270.0, 278.9]}
    controls_dict2 = {
        "yaw_angles": [270.0, 268.9],
        "power_setpoints": [3000.0, 3000.0],
    }
    interface.check_controls(controls_dict)
    interface.check_controls(controls_dict2)

    bad_controls_dict1 = {"yaw_angels": [270.0, 268.9]}  # Misspelling
    bad_controls_dict2 = {
        "yaw_angles": [270.0, 268.9],
        "power_setpoints": [3000.0, 3000.0],
        "unavailable_control": [0.0, 0.0],
    }
    bad_controls_dict3 = {"yaw_angles": [270.0, 268.9, 270.0]}  # Mismatched number of turbines

    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict1)
    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict2)
    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict3)

    # test send_controls()
    test_hercules_dict_out = interface.send_controls(
        hercules_dict=test_hercules_dict, **controls_dict
    )
    assert (
        controls_dict["yaw_angles"]
        == test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )

    with pytest.raises(TypeError):  # Bad kwarg
        interface.send_controls(test_hercules_dict, **bad_controls_dict1)
    with pytest.raises(TypeError):  # Bad kwarg
        interface.send_controls(test_hercules_dict, **bad_controls_dict2)
    # bad_controls_dict3 would pass, but faile the check_controls step.


def test_HerculesWindBatteryInterface():
    interface = HerculesWindBatteryInterface(hercules_dict=test_hercules_dict)

    # Test get_measurements()
    measurements = interface.get_measurements(hercules_dict=test_hercules_dict)

    assert (
        measurements["py_sims"]["battery"]
        == test_hercules_dict["py_sims"]["test_battery"]["outputs"]
    )
    assert (
        measurements["wind_farm"]["turbine_powers"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    )
    assert (
        measurements["wind_farm"]["turbine_wind_directions"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )

    # Test check_controls()
    # check_controls is pass-through

    # Test send_controls()
    controls_dict = {"battery": {"signal": 0}}
    test_hercules_dict_out = interface.send_controls(
        hercules_dict=test_hercules_dict, controls_dict=controls_dict
    )

    assert (
        test_hercules_dict_out["py_sims"]["inputs"]["battery_signal"]
        == controls_dict["battery"]["signal"]
    )
