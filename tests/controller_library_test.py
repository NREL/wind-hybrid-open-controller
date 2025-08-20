import numpy as np
import pandas as pd
import pytest

# import pandas as pd
from whoc.controllers import (
    BatteryController,
    BatteryPassthroughController,
    HybridSupervisoryControllerBaseline,
    HydrogenPlantController,
    LookupBasedWakeSteeringController,
    SolarPassthroughController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
    YawSetpointPassthroughController,
)
from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces import (
    HerculesBatteryInterface,
)


def test_controller_instantiation(test_interface_standin, test_hercules_dict):
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    _ = LookupBasedWakeSteeringController(
        interface=test_interface_standin, input_dict=test_hercules_dict
    )
    _ = WindFarmPowerDistributingController(
        interface=test_interface_standin, input_dict=test_hercules_dict
    )
    _ = WindFarmPowerTrackingController(
        interface=test_interface_standin, input_dict=test_hercules_dict
    )
    _ = HybridSupervisoryControllerBaseline(
        interface=test_interface_standin,
        input_dict=test_hercules_dict,
        wind_controller=1, # Override error raised for empty controllers
    )
    _ = SolarPassthroughController(interface=test_interface_standin, input_dict=test_hercules_dict)
    _ = BatteryPassthroughController(
        interface=test_interface_standin, input_dict=test_hercules_dict
    )
    _ = BatteryController(interface=test_interface_standin, input_dict=test_hercules_dict)
    _ = YawSetpointPassthroughController(interface=test_interface_standin)


def test_LookupBasedWakeSteeringController(test_hercules_dict, test_interface_hercules_ad):

    # No lookup table passed; simply passes through wind direction to yaw angles
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface_hercules_ad,
        input_dict=test_hercules_dict
    )

    # Check that the controller can be stepped
    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_angles = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions)

    # Lookup table that specified 20 degree offset for T000, 10 degree offset for T001 for all
    # wind directions
    test_offsets = np.array([20.0, 10.0])
    df_opt_test = pd.DataFrame(data={
        "wind_direction":[220.0, 220.0, 320.0, 320.0],
        "wind_speed":[0.0, 20.0, 0.0, 20.0],
        "yaw_angles_opt":[test_offsets]*4,
        "turbulence_intensity":[0.06]*4
    })
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface_hercules_ad,
        input_dict=test_hercules_dict,
        df_yaw=df_opt_test
    )

    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_angles = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions - test_offsets)

def test_WindFarmPowerDistributingController(test_hercules_dict, test_interface_hercules_ad):
    test_controller = WindFarmPowerDistributingController(
        interface=test_interface_hercules_ad,
        input_dict=test_hercules_dict
    )

    # Default behavior when no power reference is given
    test_hercules_dict["time"] = 20
    test_hercules_dict["external_signals"] = {}
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(
        test_power_setpoints,
        POWER_SETPOINT_DEFAULT/test_hercules_dict["controller"]["num_turbines"],
    )

    # Test with power reference
    test_hercules_dict["external_signals"]["wind_power_reference"] = 1000
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)
    
def test_WindFarmPowerTrackingController(test_hercules_dict, test_interface_hercules_ad):
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface_hercules_ad,
        input_dict=test_hercules_dict
    )

    # Test no change to power setpoints if producing desired power
    test_hercules_dict["external_signals"]["wind_power_reference"] = 1000
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [500, 500]
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)

    # Test if power exceeds farm reference, power setpoints are reduced
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        <= test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test if power is less than farm reference, power setpoints are increased
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [550, 400]
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        >= test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test that more aggressive control leads to faster response
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface_hercules_ad,
        input_dict=test_hercules_dict,
        proportional_gain=2
    )
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints_a = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (test_power_setpoints_a < test_power_setpoints).all()

def test_HybridSupervisoryControllerBaseline(test_hercules_dict, test_interface_hercules_hybrid_ad):

    # Establish lower controllers
    wind_controller = WindFarmPowerTrackingController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )
    solar_controller = SolarPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )
    battery_controller = BatteryPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )

    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        wind_controller=wind_controller,
        solar_controller=solar_controller,
        battery_controller=battery_controller
    )

    solar_current = 800 
    wind_current = [600, 300]
    power_ref = 1000

    # Simply test the supervisory_control method, for the time being
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = wind_current
    test_hercules_dict["py_sims"]["test_solar"]["outputs"]["power_mw"] = solar_current / 1e3
    test_controller.prev_solar_power = solar_current # To override filtering
    test_controller.prev_wind_power = sum(wind_current) # To override filtering

    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    # Expected outputs
    wind_solar_current = sum(wind_current)+solar_current
    wind_power_cmd = 20000/2 + sum(wind_current)-(wind_solar_current - power_ref)/2
    solar_power_cmd = 20000/2 + solar_current-(wind_solar_current - power_ref)/2
    battery_power_cmd = power_ref - wind_solar_current

    assert np.allclose(
            supervisory_control_output,
            [wind_power_cmd, solar_power_cmd, battery_power_cmd]
        ) # To charge battery

def test_HybridSupervisoryControllerBaseline_subsets(
    test_hercules_dict, test_interface_hercules_hybrid_ad
):
    """
    Tests that the HybridSupervisoryControllerBaseline can be run with only
    some of the wind, solar, and battery controllers.
    """
    test_interface = test_interface_hercules_hybrid_ad

    # Establish lower controllers
    wind_controller = WindFarmPowerTrackingController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )
    solar_controller = SolarPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )
    battery_controller = BatteryPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )

    ## First, try with wind and solar only
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        wind_controller=wind_controller,
        solar_controller=solar_controller,
        battery_controller=None
    )

    solar_current = 800 
    wind_current = [600, 300]
    power_ref = 1000

    # Simply test the supervisory_control method, for the time being
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = wind_current
    test_hercules_dict["py_sims"]["test_solar"]["outputs"]["power_mw"] = solar_current / 1e3
    test_controller.prev_solar_power = solar_current # To override filtering
    test_controller.prev_wind_power = sum(wind_current) # To override filtering

    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    wind_solar_current = sum(wind_current)+solar_current
    wind_power_cmd = sum(wind_current)-(wind_solar_current - power_ref)/2
    solar_power_cmd = solar_current-(wind_solar_current - power_ref)/2
    battery_power_cmd = 0 # No battery controller!

    assert np.allclose(
            supervisory_control_output,
            [wind_power_cmd, solar_power_cmd, battery_power_cmd]
        )

    ## Next, wind and battery only
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
        input_dict=test_hercules_dict,
        wind_controller=wind_controller,
        solar_controller=None,
        battery_controller=battery_controller
    )

    test_controller.prev_solar_power = 0
    test_controller.prev_wind_power = sum(wind_current) # To override filtering
    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    wind_power_cmd = 20000 + power_ref
    solar_power_cmd = 0 # No solar controller!
    battery_power_cmd = power_ref - sum(wind_current)

    assert np.allclose(
        supervisory_control_output,
        [wind_power_cmd, solar_power_cmd, battery_power_cmd]
    )

    ## Finally, solar and battery only
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
        input_dict=test_hercules_dict,
        wind_controller=None,
        solar_controller=solar_controller,
        battery_controller=battery_controller
    )

    test_controller.prev_solar_power = solar_current # To override filtering
    test_controller.prev_wind_power = 0
    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    wind_power_cmd = 0 # No wind controller!
    solar_power_cmd = 20000 + power_ref
    battery_power_cmd = power_ref - solar_current

    assert np.allclose(
        supervisory_control_output,
        [wind_power_cmd, solar_power_cmd, battery_power_cmd]
    )

    ## Either wind or solar controller must be defined
    with pytest.raises(ValueError):
        _ = HybridSupervisoryControllerBaseline(
            interface=test_interface,
            input_dict=test_hercules_dict,
            wind_controller=None,
            solar_controller=None,
            battery_controller=battery_controller
        )

    ## Only wind controller
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
        input_dict=test_hercules_dict,
        wind_controller=wind_controller,
        solar_controller=None,
        battery_controller=None
    )

    test_controller.prev_solar_power = 0
    test_controller.prev_wind_power = sum(wind_current) # To override filtering
    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    wind_power_cmd = power_ref
    solar_power_cmd = 0 # No solar controller!
    battery_power_cmd = 0 # No battery controller!

    assert np.allclose(
        supervisory_control_output,
        [wind_power_cmd, solar_power_cmd, battery_power_cmd]
    )

    ## Only solar controller
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
        input_dict=test_hercules_dict,
        wind_controller=None,
        solar_controller=solar_controller,
        battery_controller=None
    )

    test_controller.prev_solar_power = solar_current # To override filtering
    test_controller.prev_wind_power = 0
    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    wind_power_cmd = 0 # No wind controller!
    solar_power_cmd = power_ref
    battery_power_cmd = 0 # No battery controller!

    assert np.allclose(
        supervisory_control_output,
        [wind_power_cmd, solar_power_cmd, battery_power_cmd]
    )

def test_BatteryPassthroughController(test_hercules_dict, test_interface_hercules_hybrid_ad):
    test_controller = BatteryPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )

    power_ref = 1000
    measurements_dict = {"power_reference": power_ref}
    controls_dict = test_controller.compute_controls(measurements_dict)
    assert controls_dict["power_setpoint"] == power_ref

def test_SolarPassthroughController(test_hercules_dict, test_interface_hercules_hybrid_ad):
    test_controller = SolarPassthroughController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )

    power_ref = 1000
    measurements_dict = {"power_reference": power_ref}
    controls_dict = test_controller.compute_controls(measurements_dict)
    assert controls_dict["power_setpoint"] == power_ref

def test_BatteryController(test_hercules_dict):
    test_interface = HerculesBatteryInterface(test_hercules_dict)
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_batt":0.1})

    # Test when starting with 0 power output
    power_ref = 1000
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.3}
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller.step(test_hercules_dict)
    out_0 = test_controller._controls_dict["power_setpoint"]
    assert 0 < out_0 < power_ref

    # Test that increasing the gain increases the control response
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_batt":0.5})
    test_controller.step(test_hercules_dict)
    out_1 = test_controller._controls_dict["power_setpoint"]
    assert out_0 < out_1 < power_ref

    # Decreasing the gain slows the response
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_batt":0.01})
    test_controller.step(test_hercules_dict)
    out_2 = test_controller._controls_dict["power_setpoint"]
    assert 0 < out_2 < out_0

    # More complex test for smoothing capabilities (mid-low gain)
    power_refs_in = np.tile(np.array([1000.0, -1000.0]), 5)
    power_refs_out = np.zeros_like(power_refs_in)
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_batt":0.1})

    battery_power = 0
    for i, pr_in in enumerate(power_refs_in):
        test_hercules_dict["external_signals"]["plant_power_reference"] = pr_in
        test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"] = -battery_power
        test_hercules_dict["time"] += 1
        out = test_controller.step(test_hercules_dict)
        battery_power = out["py_sims"]["inputs"]["battery_signal"]
        power_refs_out[i] = battery_power

    assert (power_refs_out > -1000.0).all()
    assert (power_refs_out < 1000.0).all()

    # Test SOC-based clipping
    clipping_threshold_0 = [0.0, 0.0, 1.0, 1.0] # No clipping
    clipping_threshold_1 = [0.1, 0.2, 0.8, 0.9] # Clipping at 10%--20% and 80%--90%
    clipping_threshold_2 = [0.0, 0.5, 0.5, 1.0] # Clipping throughout

    # at 30% SOC, all should match if power reference is small
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.3}
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller_0 = BatteryController(
        test_interface,
        test_hercules_dict,
        {"clipping_thresholds":clipping_threshold_0}
    )
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["power_setpoint"]

    test_controller_1 = BatteryController(
        test_interface,
        test_hercules_dict,
        {"clipping_thresholds":clipping_threshold_1}
    )
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["power_setpoint"]

    test_controller_2 = BatteryController(
        test_interface,
        test_hercules_dict,
        {"clipping_thresholds":clipping_threshold_2}
    )
    test_controller_2.step(test_hercules_dict)
    out_2 = test_controller_2._controls_dict["power_setpoint"]

    assert out_0 == out_1
    assert out_0 == out_0

    # Clipping comes into play in 2 when the reference is large
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_2.x = 0
    test_hercules_dict["external_signals"]["plant_power_reference"] = 20000
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["power_setpoint"]
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["power_setpoint"]
    test_controller_2.step(test_hercules_dict)
    out_2 = test_controller_2._controls_dict["power_setpoint"]

    assert out_0 == out_1
    assert out_0 > out_2

    # at 85% SOC and large reference, 1 should be clipped
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.85}
    test_controller_0.x = 0
    test_controller_1.x = 0
    test_controller_0.step(test_hercules_dict)
    out_0 = test_controller_0._controls_dict["power_setpoint"]
    test_controller_1.step(test_hercules_dict)
    out_1 = test_controller_1._controls_dict["power_setpoint"]
    
    assert out_0 > out_1

def test_HydrogenPlantController(test_hercules_dict, test_interface_hercules_hybrid_ad):
    """
    Tests that the HydrogenPlantController outputs a reasonable signal
    """
    ## Test with only wind providing generation
    wind_controller = WindFarmPowerTrackingController(
        test_interface_hercules_hybrid_ad, test_hercules_dict
    )

    test_controller = HydrogenPlantController(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        generator_controller=wind_controller,
    )

    wind_current = [600, 300]
    hyrogen_ref = 0.03
    hydrogen_output = test_hercules_dict["py_sims"]["test_hydrogen"]["outputs"]["H2_mfr"]
    hydrogen_error = hyrogen_ref - hydrogen_output

    # Simply test the supervisory_control method, for the time being
    test_hercules_dict["external_signals"]["hydrogen_reference"] = hyrogen_ref
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = wind_current
    test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"] = 0.0
    test_hercules_dict["py_sims"]["test_solar"]["outputs"]["power_mw"] = 0.0
    test_controller.filtered_power_prev = sum(wind_current) # To override filtering

    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )
    controller_gain = test_hercules_dict["controller"]["nominal_plant_power_kW"] / \
        test_hercules_dict["controller"]["nominal_hydrogen_rate_kgps"] * \
        test_hercules_dict["controller"]["hydrogen_controller_gain"]
    assert controller_gain == test_controller.K

    wind_power_cmd = sum(wind_current) + controller_gain * hydrogen_error

    assert supervisory_control_output == wind_power_cmd

    # Test with a full wind/solar/battery plant
    hybrid_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        wind_controller=wind_controller,
        solar_controller=SolarPassthroughController(
            test_interface_hercules_hybrid_ad, test_hercules_dict
        ),
        battery_controller=BatteryPassthroughController(
            test_interface_hercules_hybrid_ad, test_hercules_dict
        )
    )

    test_controller = HydrogenPlantController(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        generator_controller=hybrid_controller,
    )

    # Set up the dictionary
    solar_current = 1000
    battery_current = 500
    total_current_power = sum(wind_current) + solar_current - battery_current
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = wind_current
    test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"] = battery_current
    test_hercules_dict["py_sims"]["test_solar"]["outputs"]["power_mw"] = solar_current / 1e3
    test_controller.filtered_power_prev = total_current_power # To override filtering

    test_controller.step(test_hercules_dict) # Run the controller once to update measurements
    supervisory_control_output = test_controller.supervisory_control(
        test_controller._measurements_dict
    )

    power_cmd_base = total_current_power + controller_gain * hydrogen_error

    assert supervisory_control_output == power_cmd_base

    # Test instantiation using separate controller parameters
    external_controller_parameters={
        "nominal_plant_power_kW": 10000,
        "nominal_hydrogen_rate_kgps": 0.1,
        "hydrogen_controller_gain": 1.0,
    }

    # Test an error is raised if controller_parameters is passed while also specified on input_dict
    with pytest.raises(KeyError):
        HydrogenPlantController(
            interface=test_interface_hercules_hybrid_ad,
            input_dict=test_hercules_dict,
            generator_controller=hybrid_controller,
            controller_parameters=external_controller_parameters
        )

    # Check instantiation fails if a required parameter is missing from both controller_parameters
    # and input_dict["controller"]
    del test_hercules_dict["controller"]["nominal_plant_power_kW"]
    with pytest.raises(TypeError):
        HydrogenPlantController(
            interface=test_interface_hercules_hybrid_ad,
            input_dict=test_hercules_dict,
            generator_controller=hybrid_controller,
        )

    # Check instantiation proceeds correctly if doubly-specified parameters are avoided
    del test_hercules_dict["controller"]["nominal_hydrogen_rate_kgps"]
    del test_hercules_dict["controller"]["hydrogen_controller_gain"]

    test_controller = HydrogenPlantController(
        interface=test_interface_hercules_hybrid_ad,
        input_dict=test_hercules_dict,
        generator_controller=hybrid_controller,
        controller_parameters=external_controller_parameters
    )

def test_YawSetpointPassthroughController(test_hercules_dict, test_interface_hercules_ad):
    """
    Tests that the YawSetpointPassthroughController simply passes through the yaw setpoints
    from the interface.
    """
    test_controller = YawSetpointPassthroughController(
        test_interface_hercules_ad, test_hercules_dict
    )

    # Check that the controller can be stepped
    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)

    assert np.allclose(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"],
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
