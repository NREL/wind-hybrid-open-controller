from hycon.controllers.battery_controller import (
    BatteryPriceSOCController,
)
from hycon.interfaces import HerculesInterface


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
