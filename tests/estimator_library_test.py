import numpy as np
import pytest

from whoc.estimators import WindDirectionPassthroughEstimator
from whoc.interfaces import HerculesADInterface
from whoc.interfaces.interface_base import InterfaceBase


@pytest.fixture
def test_hercules_dict():
    return {
        "dt": 1,
        "time": 0,
        "controller": {
            "num_turbines": 2,
            "initial_conditions": {"yaw": [270.0, 270.0]},
            "nominal_plant_power_kW": 10000,
            "nominal_hydrogen_rate_kgps": 0.1,
            "hydrogen_controller_gain": 1.0,
        },
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
            "test_battery": {
                "outputs": {"power": 10.0, "soc": 0.3},
                "charge_rate":20,
                "discharge_rate":20
            },
            "test_solar": {"outputs": {"power_mw": 1.0, "dni": 1000.0, "aoi": 30.0}},
            "test_hydrogen": {"outputs": {"H2_mfr": 0.03}},
            "inputs": {},
        },
        "external_signals": {"wind_power_reference": 1000.0, "plant_power_reference": 1000.0,
                            "hydrogen_reference": 0.02},
    }

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

def test_estimator_instantiation():
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    test_interface = StandinInterface()

    _ = WindDirectionPassthroughEstimator(interface=test_interface)

def test_YawSetpointPassthroughController(test_hercules_dict):
    """
    Tests that the YawSetpointPassthroughController simply passes through the yaw setpoints
    from the interface.
    """
    test_interface = HerculesADInterface(test_hercules_dict)
    test_estimator = WindDirectionPassthroughEstimator(test_interface, test_hercules_dict)

    # Check that the controller can be stepped (simply returns inputs)
    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_estimator.step(input_dict=test_hercules_dict)

    assert np.allclose(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]
            ["turbine_wind_directions"],
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )

    # Test that estimates are also computed (for passthrough, these are simply a match)
    estimates_dict = test_estimator.compute_estimates(test_estimator._measurements_dict)

    assert np.allclose(
        estimates_dict["wind_directions"],
        test_estimator._measurements_dict["wind_directions"]
    )
