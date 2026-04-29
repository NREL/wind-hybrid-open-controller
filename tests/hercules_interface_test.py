import pytest
from hycon.interfaces import HerculesInterface


def test_interface_instantiation(test_hercules_dict):
    """
    Tests whether all interfaces can be imported correctly and that they
    each implement the required methods specified by InterfaceBase.
    """

    _ = HerculesInterface(h_dict=test_hercules_dict)


def test_HerculesInterface_windonly(test_hercules_dict):
    # Test instantiation
    interface = HerculesInterface(h_dict=test_hercules_dict)
    assert interface.dt == test_hercules_dict["dt"]
    assert (
        interface.plant_parameters["wind_farm"]["capacity"]
        == (test_hercules_dict["wind_farm"]["capacity"])
    )
    assert (
        interface.plant_parameters["wind_farm"]["n_turbines"]
        == (test_hercules_dict["wind_farm"]["n_turbines"])
    )

    # Test get_measurements()
    measurements = interface.get_measurements(h_dict=test_hercules_dict)
    assert measurements["time"] == test_hercules_dict["time"]
    assert (
        measurements["wind_farm"]["wind_directions"]
        == [test_hercules_dict["wind_farm"]["wind_direction_mean"]] * 2
    )
    assert (
        measurements["wind_farm"]["turbine_powers"]
        == test_hercules_dict["wind_farm"]["turbine_powers"]
    )
    test_forecast = {
        k: v for k, v in test_hercules_dict["external_signals"].items() if "forecast" in k
    }
    assert measurements["forecast"] == test_forecast

    # Test check_controls()
    controls_dict = {"wind_farm": {"power_setpoint": [2000.0, 3000.0]}}
    # Invalid key
    bad_controls_dict1 = {
        "wind_farm": {
            "wind_power_setpoint": [2000.0, 3000.0],
            "unavailable_control": [0.0, 0.0],
        }
    }

    interface.check_controls(controls_dict)

    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict1)

    # test send_controls()
    test_hercules_dict_out = interface.send_controls(test_hercules_dict, controls_dict)
    assert (
        controls_dict["wind_farm"]["power_setpoint"]
        == test_hercules_dict_out["wind_farm"]["turbine_power_setpoints"]
    )

    with pytest.raises(TypeError):  # Bad kwarg
        interface.send_controls(test_hercules_dict, **bad_controls_dict1)


def test_HerculesInterface_hybrid(test_hercules_dict):
    # Test instantiation
    interface = HerculesInterface(h_dict=test_hercules_dict)
    assert interface.dt == test_hercules_dict["dt"]
    assert (
        interface.plant_parameters["wind_farm"]["capacity"]
        == (test_hercules_dict["wind_farm"]["capacity"])
    )
    assert (
        interface.plant_parameters["solar_farm"]["capacity"]
        == (test_hercules_dict["solar_farm"]["capacity"])
    )
    assert (
        interface.plant_parameters["battery"]["power_capacity"]
        == (test_hercules_dict["battery"]["size"])
    )
    assert (
        interface.plant_parameters["battery"]["energy_capacity"]
        == (test_hercules_dict["battery"]["energy_capacity"])
    )
    assert (
        interface.plant_parameters["wind_farm"]["n_turbines"]
        == (test_hercules_dict["wind_farm"]["n_turbines"])
    )

    # Test get_measurements()
    measurements = interface.get_measurements(h_dict=test_hercules_dict)
    assert measurements["time"] == test_hercules_dict["time"]
    assert (
        measurements["wind_farm"]["wind_directions"]
        == [test_hercules_dict["wind_farm"]["wind_direction_mean"]] * 2
    )
    assert (
        measurements["wind_farm"]["turbine_powers"]
        == test_hercules_dict["wind_farm"]["turbine_powers"]
    )
    assert measurements["solar_farm"]["power"] == test_hercules_dict["solar_farm"]["power"]
    assert measurements["battery"]["power"] == test_hercules_dict["battery"]["power"]
    assert measurements["battery"]["state_of_charge"] == test_hercules_dict["battery"]["soc"]

    # Test check_controls()
    controls_dict = {
        "wind_farm": {"power_setpoint": [2000.0, 3000.0]},
        "solar_farm": {"power_setpoint": 500.0},
        "battery": {"power_setpoint": -1000.0},
        # "hydrogen_power_setpoint": 0.02,
    }
    bad_controls_dict1 = {
        "wind_farm": {"power_setpoint": [2000.0, 3000.0]},
        "solar_farm": {"power_setpoint": 500.0},
        "battery": {"unavailable_control": [0.0, 0.0]},
    }

    # Should run through without error
    interface.check_controls(controls_dict)
    # Should raise error
    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict1)

    # Test send_controls()
    test_hercules_dict_out = interface.send_controls(test_hercules_dict, controls_dict)
    assert (
        controls_dict["wind_farm"]["power_setpoint"]
        == test_hercules_dict_out["wind_farm"]["turbine_power_setpoints"]
    )
    assert (
        controls_dict["solar_farm"]["power_setpoint"]
        == test_hercules_dict_out["solar_farm"]["power_setpoint"]
    )
    assert (
        controls_dict["battery"]["power_setpoint"]
        == test_hercules_dict_out["battery"]["power_setpoint"]
    )

    # Check that plant parameters are set correctly
    assert (
        interface.plant_parameters["interconnect_limit"]
        == test_hercules_dict["plant"]["interconnect_limit"]
    )
