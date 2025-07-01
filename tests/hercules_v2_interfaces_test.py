import pytest
from whoc.interfaces import (
    HerculesLongRunInterface,
    HerculesHybridLongRunInterface,
)

test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "controller": {
    },
    "wind_farm": {
        "num_turbines": 2,
        "capacity": 10000.0,
        "wind_direction": 271.0,
        "turbine_powers": [4000.0, 4001.0],
        "wind_speed": 10.0,
    },
    "solar_farm": {
        "power_mw": 1.0,
        "dni": 1000.0,
        "aoi": 30.0,
    },
    "battery": {
        "power": 10.0,
        "soc": 0.3,
        "charge_rate":20
    },
    "electrolyzer": {
        "H2_mfr": 0.03,
    },
    "external_signals": { # Is this OK like this?
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

    _ = HerculesLongRunInterface(hercules_dict=test_hercules_dict)
    #_ = HerculesHybridLongRunInterface(hercules_dict=test_hercules_dict)

def test_HerculesLongRunInterface():
    interface = HerculesLongRunInterface(hercules_dict=test_hercules_dict)

    # Test get_measurements()
    measurements = interface.get_measurements(hercules_dict=test_hercules_dict)
    assert measurements["time"] == test_hercules_dict["time"]
    assert (
        measurements["wind_directions"] == [test_hercules_dict["wind_farm"]["wind_direction"]] * 2
    )
    assert (
        measurements["wind_turbine_powers"] == test_hercules_dict["wind_farm"]["turbine_powers"]
    )
    test_forecast = {
        k: v for k, v in test_hercules_dict["external_signals"].items() if "forecast" in k
    }
    assert measurements["forecast"] == test_forecast

    # Test check_controls()
    controls_dict = {"power_setpoints": [2000.0, 3000.0]}
    bad_controls_dict1 = {
        "power_setpoints": [2000.0, 3000.0],
        "unavailable_control": [0.0, 0.0],
    }
    bad_controls_dict2 = {"power_setpoints": [2000.0, 3000.0, 0.0]}  # Mismatched number of turbines

    interface.check_controls(controls_dict)

    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict1)
    with pytest.raises(ValueError):
        interface.check_controls(bad_controls_dict2)

    # test send_controls()
    test_hercules_dict_out = interface.send_controls(
        hercules_dict=test_hercules_dict, **controls_dict
    )
    output_setpoints = [
        test_hercules_dict_out["wind_farm"]["derating_{0:03d}".format(i)] for i in range(2)
    ]
    assert (
        controls_dict["power_setpoints"] == output_setpoints
    )

    with pytest.raises(TypeError):  # Bad kwarg
        interface.send_controls(test_hercules_dict, **bad_controls_dict1)
