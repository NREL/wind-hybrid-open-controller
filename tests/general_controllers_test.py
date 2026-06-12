import numpy as np
from hycon.controllers import (
    PriceCurtailingController,
    SolarPassthroughController,
    WindFarmPowerDistributingController,
)


def test_PriceCurtailingController(test_hercules_dict, test_interface_hercules):
    """
    Tests that the PriceCurtailingController outputs a reasonable signal
    """
    # Consider a solar farm only
    test_interface_hercules.component_names = ["solar_farm"]
    test_controller = PriceCurtailingController(
        interface=test_interface_hercules,
        cname="solar_farm",
        controller_parameters={
            "curtailment_price": 50,
            "power_tracking_controller": SolarPassthroughController(
                test_interface_hercules, "solar_farm"
            ),
        },
    )

    # Test with price above curtailment threshold
    power_setpoint_ref = 1000
    test_hercules_dict["external_signals"]["lmp_rt"] = 100
    test_hercules_dict["solar_farm"]["power_reference"] = power_setpoint_ref
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = np.array(out_dict["solar_farm"]["power_setpoint"])
    assert np.isclose(power_setpoint_test, power_setpoint_ref)

    # Test with price below curtailment threshold
    test_hercules_dict["external_signals"]["lmp_rt"] = 25
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = np.array(out_dict["solar_farm"]["power_setpoint"])
    assert np.isclose(power_setpoint_test, 0)

    # Test again with negative threshold
    test_controller.set_controller_parameters(
        curtailment_price=-10,
        power_tracking_controller=SolarPassthroughController(test_interface_hercules, "solar_farm"),
    )
    test_hercules_dict["external_signals"]["lmp_rt"] = -5
    test_hercules_dict["solar_farm"]["power_reference"] = power_setpoint_ref
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = np.array(out_dict["solar_farm"]["power_setpoint"])
    assert np.isclose(power_setpoint_test, power_setpoint_ref)

    test_hercules_dict["external_signals"]["lmp_rt"] = -15
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = np.array(out_dict["solar_farm"]["power_setpoint"])
    assert np.isclose(power_setpoint_test, 0)

    # Test with wind farm
    test_interface_hercules.component_names = ["wind_farm"]
    test_controller = PriceCurtailingController(
        interface=test_interface_hercules,
        cname="wind_farm",
        controller_parameters={
            "curtailment_price": 50,
            "power_tracking_controller": WindFarmPowerDistributingController(
                test_interface_hercules, "wind_farm"
            ),
        },
    )
    test_hercules_dict["external_signals"]["lmp_rt"] = 100
    test_hercules_dict["wind_farm"]["power_reference"] = power_setpoint_ref
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = sum(out_dict["wind_farm"]["turbine_power_setpoints"])
    assert np.isclose(power_setpoint_test, power_setpoint_ref)

    test_hercules_dict["external_signals"]["lmp_rt"] = 25
    out_dict = test_controller.step(test_hercules_dict)
    power_setpoint_test = sum(out_dict["wind_farm"]["turbine_power_setpoints"])
    assert np.isclose(power_setpoint_test, 0)
