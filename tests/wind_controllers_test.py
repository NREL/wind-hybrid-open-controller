import numpy as np
import pandas as pd
from hycon.controllers import (
    LookupBasedWakeSteeringController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)
from hycon.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT


def test_LookupBasedWakeSteeringController(test_hercules_v1_dict, test_interface_hercules_ad):
    # No lookup table passed; simply passes through wind direction to yaw angles
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface_hercules_ad, cname="wind_farm"
    )

    # Check that the controller can be stepped
    test_hercules_v1_dict["time"] = 20
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_angles = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions)

    # Lookup table that specified 20 degree offset for T000, 10 degree offset for T001 for all
    # wind directions
    test_offsets = np.array([20.0, 10.0])
    df_opt_test = pd.DataFrame(
        data={
            "wind_direction": [220.0, 220.0, 320.0, 320.0],
            "wind_speed": [0.0, 20.0, 0.0, 20.0],
            "yaw_angles_opt": [test_offsets] * 4,
            "turbulence_intensity": [0.06] * 4,
        }
    )
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface_hercules_ad,
        cname="wind_farm",
        controller_parameters={"df_yaw": df_opt_test},
    )

    test_hercules_v1_dict["time"] = 20
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_angles = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions - test_offsets)


def test_WindFarmPowerDistributingController(test_hercules_v1_dict, test_interface_hercules_ad):
    test_controller = WindFarmPowerDistributingController(
        interface=test_interface_hercules_ad, cname="wind_farm"
    )

    # Default behavior when no power reference is given
    test_hercules_v1_dict["time"] = 20
    test_hercules_v1_dict["external_signals"] = {}
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(
        test_power_setpoints,
        POWER_SETPOINT_DEFAULT / 2,
    )

    # Test with power reference
    test_hercules_v1_dict["external_signals"]["wind_power_reference"] = 1000
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)

    # Test that ramp rate limits are applied
    test_controller = WindFarmPowerDistributingController(
        interface=test_interface_hercules_ad,
        cname="wind_farm",
        controller_parameters={"ramp_rate_limit": 200},
    )
    test_hercules_v1_dict["external_signals"]["wind_power_reference"] = 1000
    test_controller.step(input_dict=test_hercules_v1_dict)  # To initialize previous power setpoints
    test_hercules_v1_dict["external_signals"]["wind_power_reference"] = 500
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, (1000 - 200) / 2)

    test_hercules_v1_dict["external_signals"]["wind_power_reference"] = 2000
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 1000 / 2)


def test_WindFarmPowerTrackingController(test_hercules_v1_dict, test_interface_hercules_ad):
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface_hercules_ad, cname="wind_farm"
    )

    # Test no change to power setpoints if producing desired power
    test_hercules_v1_dict["external_signals"]["wind_power_reference"] = 1000
    test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [500, 500]
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)

    # Test if power exceeds farm reference, power setpoints are reduced
    test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        <= test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test if power is less than farm reference, power setpoints are increased
    test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [550, 400]
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        >= test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test that more aggressive control leads to faster response
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface_hercules_ad,
        cname="wind_farm",
        controller_parameters={"proportional_gain": 2},
    )
    test_hercules_v1_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_dict_out = test_controller.step(input_dict=test_hercules_v1_dict)
    test_power_setpoints_a = np.array(
        test_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (test_power_setpoints_a < test_power_setpoints).all()
