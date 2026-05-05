import numpy as np
from hycon.controllers import (
    BatteryController,
    BatteryPriceSOCController,
)
from hycon.interfaces import (
    HerculesInterface,
)


def test_BatteryController(test_hercules_dict):
    test_hercules_dict["component_names"] = ["battery"]

    test_interface = HerculesInterface(test_hercules_dict)
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.1})

    # Test when starting with 0 power output
    power_ref = 1000
    test_hercules_dict["battery"]["power"] = 0
    test_hercules_dict["battery"]["soc"] = 0.3
    test_hercules_dict["battery"]["power_reference"] = power_ref
    test_controller.step(test_hercules_dict)
    out_0 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert 0 < out_0 < power_ref

    # Test that increasing the gain increases the control response
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.5})
    test_controller.step(test_hercules_dict)
    out_1 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert out_0 < out_1 < power_ref

    # Decreasing the gain slows the response
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.01})
    test_controller.step(test_hercules_dict)
    out_2 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert 0 < out_2 < out_0

    # More complex test for smoothing capabilities (mid-low gain)
    power_refs_in = np.tile(np.array([1000.0, -1000.0]), 5)
    power_refs_out = np.zeros_like(power_refs_in)
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.1})

    battery_power = 0
    for i, pr_in in enumerate(power_refs_in):
        test_hercules_dict["external_signals"]["plant_power_reference"] = pr_in
        test_hercules_dict["battery"]["power"] = -battery_power
        test_hercules_dict["time"] += 1
        out = test_controller.step(test_hercules_dict)
        battery_power = out["battery"]["power_setpoint"]
        power_refs_out[i] = battery_power

    assert (power_refs_out > -1000.0).all()
    assert (power_refs_out < 1000.0).all()

    # Test SOC-based clipping
    clipping_threshold_0 = [0.0, 0.0, 1.0, 1.0]  # No clipping
    clipping_threshold_1 = [0.1, 0.2, 0.8, 0.9]  # Clipping at 10%--20% and 80%--90%
    clipping_threshold_2 = [0.0, 0.5, 0.5, 1.0]  # Clipping throughout

    # at 30% SOC, all should match if power reference is small
    test_hercules_dict["battery"]["power"] = 0.0
    test_hercules_dict["battery"]["soc"] = 0.3
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller_0 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_0},
    )
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]

    test_controller_1 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_1},
    )
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]

    test_controller_2 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_2},
    )
    test_controller_2.step(test_hercules_dict)
    out_2 = test_controller_2._controls_dict["battery"]["power_setpoint"]

    assert out_0 == out_1
    assert out_0 == out_0

    # Clipping comes into play in 2 when the reference is large
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_2.x = 0
    test_hercules_dict["battery"]["power_reference"] = 20000
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]
    test_controller_2.step(test_hercules_dict)
    out_2 = test_controller_2._controls_dict["battery"]["power_setpoint"]

    assert out_0 == out_1
    assert out_0 > out_2

    # at 85% SOC and large reference, 1 should be clipped
    test_hercules_dict["battery"]["power"] = 0.0
    test_hercules_dict["battery"]["soc"] = 0.85
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]

    assert out_0 > out_1


def test_BatteryPriceSOCController_init(test_hercules_dict):
    test_interface = HerculesInterface(test_hercules_dict)

    # Initialize controller
    test_controller = BatteryPriceSOCController(test_interface, "battery")

    # Check that the controller is initialized correctly
    assert test_controller.rated_power_charging == test_hercules_dict["battery"]["charge_rate"]
    assert (
        test_controller.rated_power_discharging == test_hercules_dict["battery"]["discharge_rate"]
    )


def test_BatteryPriceSOCController_compute_controls(test_hercules_dict):
    # This test originally written assuming 4-hour battery

    test_interface = HerculesInterface(test_hercules_dict)

    # Initialize controller
    test_controller = BatteryPriceSOCController(test_interface, "battery")

    # For testing, overwrite the high_soc and low_soc
    test_controller.high_soc = 0.8
    test_controller.low_soc = 0.2

    DA_LMP_test = [i for i in range(24)]  # Price is from 0 to 23

    # Test the high soc condition when RT_LMP is below the charge price
    # but above the low_soc_price.  SOC is too high to justify charging.
    measurement_dict = {
        "battery": {"state_of_charge": 0.9},
        "RT_LMP": 2.5,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == 0.0

    # Now, change RT_LMP to be below the 1 hour low price
    measurement_dict["RT_LMP"] = -0.5
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == -test_controller.rated_power_charging

    # Test the high price / low soc condition
    measurement_dict = {
        "battery": {"state_of_charge": 0.1},
        "RT_LMP": 22,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == 0.0

    measurement_dict["RT_LMP"] = 25
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == test_controller.rated_power_discharging

    # Middle SOC tests
    measurement_dict = {
        "battery": {"state_of_charge": 0.5},
        "RT_LMP": 2,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == -test_controller.rated_power_charging

    measurement_dict["RT_LMP"] = 22
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == test_controller.rated_power_discharging

    measurement_dict["RT_LMP"] = 10
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == 0.0


def test_BatteryPriceSOCController_compute_controls_2_hour_duration(test_hercules_dict):
    # Set the duration to 2 hours
    test_hercules_dict["battery"]["energy_capacity"] = 20.0e3
    test_interface = HerculesInterface(test_hercules_dict)

    # Initialize controller
    test_controller = BatteryPriceSOCController(test_interface, "battery")

    # For testing, overwrite the high_soc and low_soc
    test_controller.high_soc = 0.8
    test_controller.low_soc = 0.2

    DA_LMP_test = [i for i in range(24)]  # Price is from 0 to 23

    # Test the in-between bottom 1 and bottom d prices
    measurement_dict = {
        "battery": {"state_of_charge": 0.5},
        "RT_LMP": 0.5,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == -test_controller.rated_power_charging

    # Now raise the state of charge to 0.85
    measurement_dict["battery"]["state_of_charge"] = 0.85
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == 0.0

    # Now drop the RT_LMP to -.5 (Going below bottom 1 price)
    measurement_dict["RT_LMP"] = -0.5
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["battery"]["power_setpoint"] == -test_controller.rated_power_charging
