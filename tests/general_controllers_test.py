import numpy as np
from hycon.controllers import (
    LowPassFilter,
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


def test_LowPassFilter(test_hercules_dict, test_interface_hercules):
    """
    Tests that the LowPassFilter outputs a reasonable signal
    """
    test_interface_hercules.component_names = ["solar_farm"]

    # First-order filter
    lpf = LowPassFilter(
        interface=test_interface_hercules,
        cname="test",
        controller_parameters={
            "a": [1, 0.25],
            "b": [0.25],
        },
    )

    # Test with step input
    power_setpoint_ref = 1000
    test_hercules_dict["solar_farm"]["power_reference"] = power_setpoint_ref
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf.compute_controls({"test": {"power_reference": power_setpoint_ref}})["test"][
            "power_setpoint"
        ]
    assert np.all(np.diff(out) > 0)  # Check output is increasing
    assert np.all(out < power_setpoint_ref)  # Check output is below reference

    # Confirm steady-state behavior
    for _ in range(100):
        y = lpf.compute_controls({"test": {"power_reference": power_setpoint_ref}})["test"][
            "power_setpoint"
        ]
    assert np.isclose(y, power_setpoint_ref, atol=1e-2)

    # Second-order overdamped filter
    omega_n = 0.5
    zeta = 1.5
    lpf = LowPassFilter(
        interface=test_interface_hercules,
        cname="test",
        controller_parameters={
            "a": [1, 2 * zeta * omega_n, omega_n**2],
            "b": [omega_n**2],
        },
    )

    # Test with step input
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf.compute_controls({"test": {"power_reference": power_setpoint_ref}})["test"][
            "power_setpoint"
        ]
    assert np.all(np.diff(out) > 0)  # Check output is increasing
    assert np.all(out < power_setpoint_ref)  # Check output is below reference

    # Second-order underdamped filter
    zeta = 0.5
    lpf = LowPassFilter(
        interface=test_interface_hercules,
        cname="test",
        controller_parameters={
            "a": [1, 2 * zeta * omega_n, omega_n**2],
            "b": [omega_n**2],
        },
    )

    # Test with step input
    out = np.zeros(10)
    for i in range(10):
        out[i] = lpf.compute_controls({"test": {"power_reference": power_setpoint_ref}})["test"][
            "power_setpoint"
        ]
    # Check underdamped (oscillates, overshoots)
    assert np.any(np.diff(out) < 0)
    assert np.any(out > power_setpoint_ref)

    # Steady-state behavior should still hold
    for _ in range(100):
        y = lpf.compute_controls({"test": {"power_reference": power_setpoint_ref}})["test"][
            "power_setpoint"
        ]
    assert np.isclose(y, power_setpoint_ref, atol=1e-2)
