import copy

import numpy as np
import pytest
from hycon.controllers import (
    BatteryPassthroughController,
    HybridSupervisoryControllerGeneric,
    HydrogenPlantController,
    SolarPassthroughController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)


def test_HybridSupervisoryControllerGeneric_reference_tracking(
    test_hercules_dict, test_interface_hercules
):
    """
    Tests for the HybridSupervisoryControllerGeneric when following a power reference.
    """
    # Establish lower controllers
    wind_controller = WindFarmPowerDistributingController(test_interface_hercules, "wind_farm")
    solar_controller = SolarPassthroughController(test_interface_hercules, "solar_farm")
    battery_controller = BatteryPassthroughController(test_interface_hercules, "battery")

    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface_hercules,
        controller_parameters={
            "component_controllers": [wind_controller, solar_controller, battery_controller],
        },
    )

    solar_current = 800
    wind_current = 900
    power_ref = 1000

    battery_charge_rate = test_hercules_dict["battery"]["charge_rate"]

    # Simply test the supervisory_control method, for the time being
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_hercules_dict["wind_farm"]["power"] = wind_current
    test_hercules_dict["solar_farm"]["power"] = solar_current

    # Step controller
    out_dict = test_controller.step(test_hercules_dict)
    wind_setpoint_test = sum(out_dict["wind_farm"]["turbine_power_setpoints"])
    solar_setpoint_test = out_dict["solar_farm"]["power_setpoint"]
    battery_setpoint_test = out_dict["battery"]["power_setpoint"]

    # Expected outputs
    wind_solar_current = wind_current + solar_current
    wind_setpoint_ref = battery_charge_rate + power_ref
    solar_setpoint_ref = wind_setpoint_ref - wind_current
    battery_setpoint_ref = power_ref - wind_solar_current

    assert np.allclose(
        [wind_setpoint_test, solar_setpoint_test, battery_setpoint_test],
        [wind_setpoint_ref, solar_setpoint_ref, battery_setpoint_ref],
    )


def test_HybridSupervisoryControllerGeneric_subsets(test_hercules_dict, test_interface_hercules):
    """
    Tests that the HybridSupervisoryControllerGeneric can be run with only
    some of the wind, solar, and battery controllers.
    """
    test_interface = test_interface_hercules

    # Alter dict for test
    solar_current = 800.0
    wind_current = 900.0
    power_ref = 1000.0

    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_hercules_dict["wind_farm"]["power"] = wind_current
    test_hercules_dict["solar_farm"]["power"] = solar_current

    battery_charge_rate = test_hercules_dict["battery"]["charge_rate"]

    # Establish lower controllers
    wind_controller = WindFarmPowerTrackingController(test_interface, "wind_farm")
    solar_controller = SolarPassthroughController(test_interface, "solar_farm")
    battery_controller = BatteryPassthroughController(test_interface, "battery")

    ## First, try with wind and solar only
    test_interface.component_names = ["wind_farm", "solar_farm"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [wind_controller, solar_controller]},
    )

    # Step controller
    out_dict = test_controller.step(test_hercules_dict)
    wind_setpoint_test = sum(out_dict["wind_farm"]["turbine_power_setpoints"])
    solar_setpoint_test = out_dict["solar_farm"]["power_setpoint"]

    wind_setpoint_ref = power_ref
    solar_setpoint_ref = wind_setpoint_ref - wind_current

    assert np.allclose(
        [wind_setpoint_test, solar_setpoint_test], [wind_setpoint_ref, solar_setpoint_ref]
    )

    ## Next, wind and battery only
    test_interface.component_names = ["wind_farm", "battery"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [wind_controller, battery_controller]},
    )

    # Step controller
    out_dict = test_controller.step(test_hercules_dict)
    wind_setpoint_test = sum(out_dict["wind_farm"]["turbine_power_setpoints"])
    battery_setpoint_test = out_dict["battery"]["power_setpoint"]

    wind_setpoint_ref = battery_charge_rate + power_ref
    battery_setpoint_ref = power_ref - wind_current

    assert np.allclose(
        [wind_setpoint_test, battery_setpoint_test], [wind_setpoint_ref, battery_setpoint_ref]
    )

    ## Finally, solar and battery only
    test_interface.component_names = ["solar_farm", "battery"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [solar_controller, battery_controller]},
    )

    # Step controller
    out_dict = test_controller.step(test_hercules_dict)
    solar_setpoint_test = out_dict["solar_farm"]["power_setpoint"]
    battery_setpoint_test = out_dict["battery"]["power_setpoint"]

    solar_setpoint_ref = power_ref + battery_charge_rate
    battery_setpoint_ref = power_ref - solar_current

    assert np.allclose(
        [solar_setpoint_test, battery_setpoint_test], [solar_setpoint_ref, battery_setpoint_ref]
    )

    ## Test also the case where the battery is not allowed to charge from the grid
    # Start with allowing grid charging
    test_hercules_dict["external_signals"]["plant_power_reference"] = -100.0  # Must charge
    out_dict = test_controller.step(test_hercules_dict)
    battery_setpoint_test = out_dict["battery"]["power_setpoint"]
    assert np.isclose(battery_setpoint_test, -100.0 - solar_current)

    # Switch to not allowing grid charging, capped at solar output
    test_controller.component_controllers[1].plant_parameters["battery"]["allow_grid_charging"] = (
        False
    )
    out_dict = test_controller.step(test_hercules_dict)
    battery_setpoint_test = out_dict["battery"]["power_setpoint"]
    assert np.isclose(battery_setpoint_test, -solar_current)

    ## Only wind controller
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_interface.component_names = ["wind_farm"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [wind_controller]},
    )

    out_dict = test_controller.step(test_hercules_dict)

    assert np.isclose(sum(out_dict["wind_farm"]["turbine_power_setpoints"]), power_ref)

    ## Only solar controller
    test_interface.component_names = ["solar_farm"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [solar_controller]},
    )
    out_dict = test_controller.step(test_hercules_dict)
    assert np.isclose(out_dict["solar_farm"]["power_setpoint"], power_ref)

    ## Only battery controller
    test_interface.component_names = ["battery"]
    test_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface,
        controller_parameters={"component_controllers": [battery_controller]},
    )
    out_dict = test_controller.step(test_hercules_dict)
    assert np.isclose(out_dict["battery"]["power_setpoint"], power_ref)


def test_HydrogenPlantController(test_hercules_dict, test_interface_hercules):
    """
    Tests that the HydrogenPlantController outputs a reasonable signal
    """
    ## Test with only wind providing generation
    wind_controller = WindFarmPowerTrackingController(test_interface_hercules, "wind_farm")

    # Remove components not used for first test
    test_herc_dict_windonly = copy.deepcopy(test_hercules_dict)
    del test_herc_dict_windonly["battery"]
    del test_herc_dict_windonly["solar_farm"]
    test_herc_dict_windonly["component_names"] = ["wind_farm", "electrolyzer"]
    test_interface_hercules.component_names = ["wind_farm", "electrolyzer"]

    test_controller_parameters = {
        "nominal_plant_power_kW": 10000,
        "nominal_hydrogen_rate_kgps": 0.1,
        "hydrogen_controller_gain": 1.0,
    }

    test_controller_parameters["generator_controller"] = wind_controller
    test_controller = HydrogenPlantController(
        interface=test_interface_hercules,
        cname="electrolyzer",
        controller_parameters=test_controller_parameters,
    )

    wind_current = [600, 300]
    hydrogen_ref = 0.028
    hydrogen_output = test_herc_dict_windonly["electrolyzer"]["H2_mfr"]
    hydrogen_error = hydrogen_ref - hydrogen_output

    # Simply test the supervisory_control method, for the time being
    test_herc_dict_windonly["external_signals"]["hydrogen_reference"] = hydrogen_ref
    test_herc_dict_windonly["wind_farm"]["power"] = sum(wind_current)
    test_controller.filtered_power_prev = sum(wind_current)  # To override filtering

    # Without removing wind power reference, wind controller can't reconcile its setpoint
    out_dict = test_controller.step(test_herc_dict_windonly)
    controller_gain = 10000 / 0.1 * 1.0  # Based on parameters passed to controller
    assert controller_gain == test_controller.K

    wind_cmd_ref = sum(wind_current) + controller_gain * hydrogen_error

    assert np.isclose(sum(out_dict["wind_farm"]["turbine_power_setpoints"]), wind_cmd_ref)

    # Test with a full wind/solar/battery plant
    test_interface_hercules.component_names = ["wind_farm", "solar_farm", "battery"]

    hybrid_controller = HybridSupervisoryControllerGeneric(
        interface=test_interface_hercules,
        controller_parameters={
            "component_controllers": [
                wind_controller,
                SolarPassthroughController(test_interface_hercules, "solar_farm"),
                BatteryPassthroughController(test_interface_hercules, "battery"),
            ],
        },
    )

    test_controller_parameters["generator_controller"] = hybrid_controller
    test_controller = HydrogenPlantController(
        interface=test_interface_hercules,
        cname="electrolyzer",
        controller_parameters=test_controller_parameters,
    )

    # Set up the dictionary
    solar_current = 1000
    battery_current = 500
    total_current_power = sum(wind_current) + solar_current + battery_current
    test_hercules_dict["wind_farm"]["power"] = sum(wind_current)
    test_hercules_dict["solar_farm"]["power"] = solar_current
    test_hercules_dict["battery"]["power"] = battery_current
    test_hercules_dict["external_signals"]["hydrogen_reference"] = hydrogen_ref

    test_controller.filtered_power_prev = total_current_power  # To override filtering

    meas = test_controller._s.get_measurements(test_hercules_dict)
    power_cmd_test = test_controller.supervisory_control(meas)

    power_cmd_ref = total_current_power + controller_gain * hydrogen_error

    assert np.isclose(power_cmd_test, power_cmd_ref)

    # Test instantiation using separate controller parameters
    external_controller_parameters = {
        "nominal_plant_power_kW": 10000,
        "nominal_hydrogen_rate_kgps": 0.1,
        "hydrogen_controller_gain": 1.0,
    }

    # Test an error is raised if controller_parameters is passed without generator_controller
    with pytest.raises(KeyError):
        HydrogenPlantController(
            interface=test_interface_hercules,
            controller_parameters=external_controller_parameters,
        )

    # Check instantiation fails if bad argument passed on controller_parameters
    external_controller_parameters["invalid_parameter"] = 123
    with pytest.raises(KeyError):
        HydrogenPlantController(
            interface=test_interface_hercules,
            controller_parameters=external_controller_parameters,
        )
