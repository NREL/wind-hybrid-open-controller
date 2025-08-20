import numpy as np
from whoc.estimators import WindDirectionPassthroughEstimator


def test_estimator_instantiation(test_interface_standin):
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    _ = WindDirectionPassthroughEstimator(interface=test_interface_standin)

def test_YawSetpointPassthroughController(test_interface_hercules_ad, test_hercules_dict):
    """
    Tests that the YawSetpointPassthroughController simply passes through the yaw setpoints
    from the interface.
    """
    test_estimator = WindDirectionPassthroughEstimator(
        test_interface_hercules_ad,
        test_hercules_dict
    )

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
