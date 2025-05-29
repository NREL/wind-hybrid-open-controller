import pytest
from whoc.interfaces import (
    HerculesADInterface,
    HerculesBatteryInterface,
    HerculesHybridADInterface,
)

test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "controller": {"num_turbines": 2, "wind_capacity_MW": 10},
    "hercules_comms": {
        "amr_wind": {
            "test_farm": {
                "turbine_wind_directions": [271.0, 272.5],
                "turbine_powers": [4000.0, 4001.0],
                "wind_speed": 10.0,
            }
        }
    },
    "py_sims": {
        "test_battery": {"outputs": {"power": 10.0, "soc": 0.3}, "charge_rate":20},
        "test_solar": {"outputs": {"power_mw": 1.0, "dni": 1000.0, "aoi": 30.0}},
        "test_hydrogen": {"outputs": {"H2_output": 0.03} },
        "inputs": {},
    },
    "external_signals": {
        "wind_power_reference": 1000.0,
        "plant_power_reference": 1000.0,
        "forecast_ws_mean_0": 8.0,
        "forecast_ws_mean_1": 8.1,
        "ws_median_0": 8.1,
        "hydrogen_reference": 0.02,
    },
}


def test_interface_instantiation():
    """
    Tests whether all interfaces can be imported correctly and that they
    each implement the required methods specified by InterfaceBase.
    """

    _ = HerculesADInterface(hercules_dict=test_hercules_dict)
    _ = HerculesHybridADInterface(hercules_dict=test_hercules_dict)
    _ = HerculesBatteryInterface(hercules_dict=test_hercules_dict)
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
    test_forecast = {
        k: v for k, v in test_hercules_dict["external_signals"].items() if "forecast" in k
    }
    assert measurements["forecast"] == test_forecast

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

def test_HerculesHybridADInterface():
    interface = HerculesHybridADInterface(hercules_dict=test_hercules_dict)

    # Test get_measurements()
    measurements = interface.get_measurements(hercules_dict=test_hercules_dict)

    assert measurements["time"] == test_hercules_dict["time"]
    assert (
        measurements["wind_turbine_powers"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    )
    assert (
        measurements["wind_speed"]
        == test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["wind_speed"]
    )
    assert (
        measurements["plant_power_reference"]
        == test_hercules_dict["external_signals"]["wind_power_reference"]
    )
    assert (
        measurements["battery_power"]
        == -test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"]
    )
    assert (
        measurements["solar_power"]
        == test_hercules_dict["py_sims"]["test_solar"]["outputs"]["power_mw"] * 1000
    )
    assert (
        measurements["solar_dni"] == test_hercules_dict["py_sims"]["test_solar"]["outputs"]["dni"]
    )
    assert (
        measurements["solar_aoi"] == test_hercules_dict["py_sims"]["test_solar"]["outputs"]["aoi"]
    )
    test_forecast = {
        k: v for k, v in test_hercules_dict["external_signals"].items() if "forecast" in k
    }
    assert measurements["forecast"] == test_forecast

    # Test check_controls()
    controls_dict = {
        "wind_power_setpoints": [1000.0, 1000.0],
        "solar_power_setpoint": 1000.0,
        "battery_power_setpoint": 0.0,
    }
    bad_controls_dict = {
        "wind_power_setpoints": [1000.0, 1000.0],
        "solar_power_setpoint": 1000.0,
        "battery_power_setpoint": 0.0,
        "unavailable_control": 0.0,
    }

    interface.check_controls(controls_dict)

    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict)

    # Test send_controls()
    test_hercules_dict_out = interface.send_controls(
        hercules_dict=test_hercules_dict, **controls_dict
    )

    assert (
        test_hercules_dict_out["py_sims"]["inputs"]["battery_signal"]
        == -controls_dict["battery_power_setpoint"]
    )
    assert (
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
        == controls_dict["wind_power_setpoints"]
    )
    assert (
        test_hercules_dict_out["py_sims"]["inputs"]["solar_setpoint_mw"]
        == controls_dict["solar_power_setpoint"] / 1000
    )
    assert (
        measurements["hydrogen_reference"]
        == test_hercules_dict["external_signals"]["hydrogen_reference"]
    )
    assert (
        measurements["hydrogen_output"]
        == test_hercules_dict["py_sims"]["test_hydrogen"]["outputs"]["H2_output"]
    )

    with pytest.raises(TypeError):  # Bad kwarg
        interface.send_controls(test_hercules_dict, **bad_controls_dict)

def test_HerculesBatteryInterface():

    interface = HerculesBatteryInterface(hercules_dict=test_hercules_dict)

    # Check instantiation with no battery raises and error
    temp = test_hercules_dict["py_sims"].pop("test_battery")
    with pytest.raises(ValueError):
        _ = HerculesBatteryInterface(hercules_dict=test_hercules_dict)
    # Reinstate and add second battery; test that 2 batteries causes error
    test_hercules_dict["py_sims"]["test_battery"] = temp
    test_hercules_dict["py_sims"]["test_battery_2"] = temp
    with pytest.raises(ValueError):
        _ = HerculesBatteryInterface(hercules_dict=test_hercules_dict)
    test_hercules_dict["py_sims"].pop("test_battery_2")

    # Test get_measurements()
    measurements = interface.get_measurements(hercules_dict=test_hercules_dict)
    assert (
        measurements["battery_power"]
        == -test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"]
    )
    assert (
        measurements["battery_soc"]
        == test_hercules_dict["py_sims"]["test_battery"]["outputs"]["soc"]
    )
    assert (
        measurements["power_reference"]
        == test_hercules_dict["external_signals"]["plant_power_reference"]
    )

    # Test check_controls()
    controls_dict = {
        "power_setpoint": 20.0,
    }
    bad_controls_dict = {
        "power_setpoint": 2.0,
        "unavailable_control": 0.0,
    }
    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict)
    interface.check_controls(controls_dict)

    # Test send_controls()
    test_hercules_dict_out = interface.send_controls(
        hercules_dict=test_hercules_dict, **controls_dict
    )
    assert (
        test_hercules_dict_out["py_sims"]["inputs"]["battery_signal"]
        == -controls_dict["power_setpoint"]
    )
    # defaults to zero
    test_hercules_dict_out = interface.send_controls(hercules_dict=test_hercules_dict)
    assert test_hercules_dict_out["py_sims"]["inputs"]["battery_signal"] == 0
