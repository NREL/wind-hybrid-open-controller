import copy

import numpy as np
import pandas as pd
import pytest

# import pandas as pd
from hycon.controllers import (
    BatteryController,
    BatteryPassthroughController,
    HybridSupervisoryControllerGeneric,
    HydrogenPlantController,
    LookupBasedWakeSteeringController,
    PriceCurtailingController,
    SolarPassthroughController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)
from hycon.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from hycon.interfaces import (
    HerculesBatteryInterface,
)


def test_controller_instantiation(test_interface_standin):
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    _ = LookupBasedWakeSteeringController(interface=test_interface_standin, cname="wind_farm")
    _ = WindFarmPowerDistributingController(interface=test_interface_standin, cname="wind_farm")
    _ = WindFarmPowerTrackingController(interface=test_interface_standin, cname="wind_farm")
    _ = SolarPassthroughController(interface=test_interface_standin, cname="solar_farm")
    _ = BatteryPassthroughController(interface=test_interface_standin, cname="battery")
    _ = BatteryController(interface=test_interface_standin, cname="battery")


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
    solar_current = 800
    wind_current = 900
    power_ref = 1000

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

    ## Only wind controller
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


def test_BatteryController(test_hercules_v1_dict):
    test_interface = HerculesBatteryInterface(test_hercules_v1_dict)
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.1})

    # Test when starting with 0 power output
    power_ref = 1000
    test_hercules_v1_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.3}
    test_hercules_v1_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller.step(test_hercules_v1_dict)
    out_0 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert 0 < out_0 < power_ref

    # Test that increasing the gain increases the control response
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.5})
    test_controller.step(test_hercules_v1_dict)
    out_1 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert out_0 < out_1 < power_ref

    # Decreasing the gain slows the response
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.01})
    test_controller.step(test_hercules_v1_dict)
    out_2 = test_controller._controls_dict["battery"]["power_setpoint"]
    assert 0 < out_2 < out_0

    # More complex test for smoothing capabilities (mid-low gain)
    power_refs_in = np.tile(np.array([1000.0, -1000.0]), 5)
    power_refs_out = np.zeros_like(power_refs_in)
    test_controller = BatteryController(test_interface, "battery", {"k_batt": 0.1})

    battery_power = 0
    for i, pr_in in enumerate(power_refs_in):
        test_hercules_v1_dict["external_signals"]["plant_power_reference"] = pr_in
        test_hercules_v1_dict["py_sims"]["test_battery"]["outputs"]["power"] = -battery_power
        test_hercules_v1_dict["time"] += 1
        out = test_controller.step(test_hercules_v1_dict)
        battery_power = out["py_sims"]["inputs"]["battery_signal"]
        power_refs_out[i] = battery_power

    assert (power_refs_out > -1000.0).all()
    assert (power_refs_out < 1000.0).all()

    # Test SOC-based clipping
    clipping_threshold_0 = [0.0, 0.0, 1.0, 1.0]  # No clipping
    clipping_threshold_1 = [0.1, 0.2, 0.8, 0.9]  # Clipping at 10%--20% and 80%--90%
    clipping_threshold_2 = [0.0, 0.5, 0.5, 1.0]  # Clipping throughout

    # at 30% SOC, all should match if power reference is small
    test_hercules_v1_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.3}
    test_hercules_v1_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller_0 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_0},
    )
    test_controller_0.step(test_hercules_v1_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]

    test_controller_1 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_1},
    )
    test_controller_1.step(test_hercules_v1_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]

    test_controller_2 = BatteryController(
        test_interface,
        "battery",
        {"clipping_thresholds": clipping_threshold_2},
    )
    test_controller_2.step(test_hercules_v1_dict)
    out_2 = test_controller_2._controls_dict["battery"]["power_setpoint"]

    assert out_0 == out_1
    assert out_0 == out_0

    # Clipping comes into play in 2 when the reference is large
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_2.x = 0
    test_hercules_v1_dict["external_signals"]["plant_power_reference"] = 20000
    test_controller_0.step(test_hercules_v1_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]
    test_controller_1.step(test_hercules_v1_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]
    test_controller_2.step(test_hercules_v1_dict)
    out_2 = test_controller_2._controls_dict["battery"]["power_setpoint"]

    assert out_0 == out_1
    assert out_0 > out_2

    # at 85% SOC and large reference, 1 should be clipped
    test_hercules_v1_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.85}
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_0.step(test_hercules_v1_dict)
    out_0 = test_controller_0._controls_dict["battery"]["power_setpoint"]
    test_controller_1.step(test_hercules_v1_dict)
    out_1 = test_controller_1._controls_dict["battery"]["power_setpoint"]

    assert out_0 > out_1


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
