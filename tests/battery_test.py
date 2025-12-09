from hycon.controllers.battery_controller import (
    BatteryPriceSOCController,
)
from hycon.interfaces import HerculesInterface

test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "plant": {"interconnect_limit": 10},
    "battery": {
        "size": 100.0,
        "energy_capacity": 400.0,
        "power": 100.0,
        "soc": 0.5,
        "charge_rate": 50.0 * 1e3,
        "discharge_rate": 100.0 * 1e3,
    },
    "external_signals": {
        "RT_LMP": 10.0,
    },
}


def test_BatteryPriceSOCController_init():
    test_interface = HerculesInterface(test_hercules_dict)

    # Initialize controller
    test_controller = BatteryPriceSOCController(test_interface, test_hercules_dict)

    # Check that the controller is initialized correctly
    assert test_controller.rated_power_charging == test_hercules_dict["battery"]["charge_rate"]
    assert (
        test_controller.rated_power_discharging == test_hercules_dict["battery"]["discharge_rate"]
    )


def test_BatteryPriceSOCController_compute_controls():
    test_interface = HerculesInterface(test_hercules_dict)

    # Initialize controller
    test_controller = BatteryPriceSOCController(test_interface, test_hercules_dict)

    # For testing, overwrite the high_soc and low_soc
    test_controller.high_soc = 0.8
    test_controller.low_soc = 0.2

    DA_LMP_test = [i for i in range(24)]  # Price is from 0 to 23

    # Test the high soc condition when RT_LMP is below the charge price
    # but above the low_soc_price
    measurement_dict = {
        "battery": {"state_of_charge": 0.9},
        "RT_LMP": 3,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == 0.0

    # Now, change RT_LMP to be below the 1 hour low price
    measurement_dict["RT_LMP"] = -0.5
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == -test_controller.rated_power_charging

    # Test the high SOC conditions
    measurement_dict = {
        "battery": {"state_of_charge": 0.9},
        "RT_LMP": 22,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == 0.0

    measurement_dict["RT_LMP"] = 25
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == test_controller.rated_power_discharging

    # Middle SOC tests
    measurement_dict = {
        "battery": {"state_of_charge": 0.5},
        "RT_LMP": 2,
        "DA_LMP_24hours": DA_LMP_test,
    }
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == -test_controller.rated_power_charging

    measurement_dict["RT_LMP"] = 22
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == test_controller.rated_power_discharging

    measurement_dict["RT_LMP"] = 10
    controls_dict = test_controller.compute_controls(measurement_dict)
    assert controls_dict["power_setpoint"] == 0.0
