import numpy as np
import pandas as pd
import pytest

# import pandas as pd
from whoc.controllers import (
    BatteryController,
    BatteryPassthroughController,
    HybridSupervisoryControllerBaseline,
    LookupBasedWakeSteeringController,
    SolarPassthroughController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)
from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces import (
    HerculesADInterface,
    HerculesBatteryInterface,
    HerculesHybridADInterface,
)
from whoc.interfaces.interface_base import InterfaceBase


class StandinInterface(InterfaceBase):
    """
    Empty class to test controllers.
    """

    def __init__(self):
        super().__init__()

    def get_measurements(self):
        pass

    def check_controls(self):
        pass

    def send_controls(self):
        pass


test_hercules_dict = {
    "dt": 1,
    "time": 0,
    "controller": {"num_turbines": 2, "initial_conditions": {"yaw": [270.0, 270.0]}},
    "hercules_comms": {
        "amr_wind": {
            "test_farm": {
                "turbine_wind_directions": [271.0, 272.5],
                "turbine_powers": [4000.0, 4001.0],
                "wind_speed": 10.0,
            }
        }
    },
    "py_sims": {
        "test_battery": {
            "outputs": {"power": 10.0, "soc": 0.3},
            "charge_rate":20,
            "discharge_rate":20
        },
        "test_solar": {"outputs": {"power_mw": 1.0, "dni": 1000.0, "aoi": 30.0}},
        "inputs": {},
    },
    "external_signals": {"wind_power_reference": 1000.0, "plant_power_reference": 1000.0},
}


def test_controller_instantiation():
    """
    Tests whether all controllers can be imported correctly and that they
    each implement the required methods specified by ControllerBase.
    """
    test_interface = StandinInterface()

    _ = LookupBasedWakeSteeringController(interface=test_interface, input_dict=test_hercules_dict)
    _ = WindFarmPowerDistributingController(interface=test_interface, input_dict=test_hercules_dict)
    _ = WindFarmPowerTrackingController(interface=test_interface, input_dict=test_hercules_dict)
    _ = HybridSupervisoryControllerBaseline(
        interface=test_interface,
        input_dict=test_hercules_dict,
        wind_controller=1, # Override error raised for empty controllers
    )
    _ = SolarPassthroughController(interface=test_interface, input_dict=test_hercules_dict)
    _ = BatteryPassthroughController(interface=test_interface, input_dict=test_hercules_dict)
    _ = BatteryController(interface=test_interface, input_dict=test_hercules_dict)


def test_LookupBasedWakeSteeringController():
    test_interface = HerculesADInterface(test_hercules_dict)

    # No lookup table passed; simply passes through wind direction to yaw angles
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface,
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
        "wind_direction":[220.0, 320.0, 220.0, 320.0],
        "wind_speed":[0.0, 0.0, 20.0, 20.0],
        "yaw_angles_opt":[test_offsets]*4,
        "turbulence_intensity":[0.06]*4
    })
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface,
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

def test_WindFarmPowerDistributingController():
    test_interface = HerculesADInterface(test_hercules_dict)
    test_controller = WindFarmPowerDistributingController(
        interface=test_interface,
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
    
def test_WindFarmPowerTrackingController():
    test_interface = HerculesADInterface(test_hercules_dict)
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface,
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
        interface=test_interface,
        input_dict=test_hercules_dict,
        proportional_gain=2
    )
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_hercules_dict_out = test_controller.step(input_dict=test_hercules_dict)
    test_power_setpoints_a = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (test_power_setpoints_a < test_power_setpoints).all()

def test_HybridSupervisoryControllerBaseline():
    test_interface = HerculesHybridADInterface(test_hercules_dict)

    # Establish lower controllers
    wind_controller = WindFarmPowerTrackingController(test_interface, test_hercules_dict)
    solar_controller = SolarPassthroughController(test_interface, test_hercules_dict)
    battery_controller = BatteryPassthroughController(test_interface, test_hercules_dict)

    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
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
    supervisory_control_output = test_controller.supervisory_control()

    # Expected outputs
    wind_solar_current = sum(wind_current)+solar_current
    wind_power_cmd = 20000/2 + sum(wind_current)-(wind_solar_current - power_ref)/2
    solar_power_cmd = 20000/2 + solar_current-(wind_solar_current - power_ref)/2
    battery_power_cmd = wind_solar_current - power_ref

    assert np.allclose(
            supervisory_control_output,
            [wind_power_cmd, solar_power_cmd, battery_power_cmd]
        ) # To charge battery

def test_HybridSupervisoryControllerBaseline_subsets():
    """
    Tests that the HybridSupervisoryControllerBaseline can be run with only
    some of the wind, solar, and battery controllers.
    """
    test_interface = HerculesHybridADInterface(test_hercules_dict)

    # Establish lower controllers
    wind_controller = WindFarmPowerTrackingController(test_interface, test_hercules_dict)
    solar_controller = SolarPassthroughController(test_interface, test_hercules_dict)
    battery_controller = BatteryPassthroughController(test_interface, test_hercules_dict)

    ## First, try with wind and solar only
    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface,
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
    supervisory_control_output = test_controller.supervisory_control()

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
    supervisory_control_output = test_controller.supervisory_control()

    wind_power_cmd = 20000 + power_ref
    solar_power_cmd = 0 # No solar controller!
    battery_power_cmd = sum(wind_current) - power_ref

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
    supervisory_control_output = test_controller.supervisory_control()

    wind_power_cmd = 0 # No wind controller!
    solar_power_cmd = 20000 + power_ref
    battery_power_cmd = solar_current - power_ref

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
    supervisory_control_output = test_controller.supervisory_control()

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
    supervisory_control_output = test_controller.supervisory_control()

    wind_power_cmd = 0 # No wind controller!
    solar_power_cmd = power_ref
    battery_power_cmd = 0 # No battery controller!

    assert np.allclose(
        supervisory_control_output,
        [wind_power_cmd, solar_power_cmd, battery_power_cmd]
    )

def test_BatteryPassthroughController():
    test_interface = HerculesHybridADInterface(test_hercules_dict)
    test_controller = BatteryPassthroughController(test_interface, test_hercules_dict)

    power_ref = 1000
    test_controller.measurements_dict["power_reference"] = power_ref
    test_controller.compute_controls()
    assert test_controller.controls_dict["power_setpoint"] == power_ref

def test_SolarPassthroughController():
    test_interface = HerculesHybridADInterface(test_hercules_dict)
    test_controller = SolarPassthroughController(test_interface, test_hercules_dict)

    power_ref = 1000
    test_controller.measurements_dict["solar_power_reference"] = power_ref
    test_controller.compute_controls()
    assert test_controller.controls_dict["power_setpoint"] == power_ref

def test_BatteryController():
    test_interface = HerculesBatteryInterface(test_hercules_dict)
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_p":1, "k_d":0})

    # Test when starting with 0 power output
    power_ref = 1000
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.3}
    test_hercules_dict["external_signals"]["plant_power_reference"] = power_ref
    test_controller.step(test_hercules_dict)
    assert test_controller.controls_dict["power_setpoint"] == power_ref

    # Test when starting with nonzero power output
    test_hercules_dict["py_sims"]["test_battery"]["outputs"]["power"] = -200
    test_controller.step(test_hercules_dict)
    assert test_controller.controls_dict["power_setpoint"] == power_ref

    # k_p = 2 (fast control)
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_p_max":2})
    test_controller.step(test_hercules_dict)
    assert test_controller.controls_dict["power_setpoint"] == 2 * (power_ref - 200) + 200

    # k_p = 0.3 (slow control)
    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_p_max":0.3})
    test_controller.step(test_hercules_dict)
    assert test_controller.controls_dict["power_setpoint"] == 0.3 * (power_ref - 200) + 200

    # More complex test for smoothing capabilities
    power_refs_in = np.tile(np.array([1000.0, -1000.0]), 5)
    power_refs_out = np.zeros_like(power_refs_in)

    test_controller = BatteryController(test_interface, test_hercules_dict, {"k_p_max":0.5})

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

    # Test SOC-based gain scheduling
    k_p_max = 0.8
    k_p_min = 0.2
    socs = np.linspace(0, 1, 11)
    gains = BatteryController.quadratic_gain_schedule(k_p_max, k_p_min, socs)
    assert np.isclose(gains[0], k_p_min)
    assert np.isclose(gains[-1], k_p_min)
    assert np.isclose(gains[5], k_p_max)
    assert np.allclose(gains[:6], gains[11:4:-1])
    assert (np.diff(gains[:6]) > 0).all()
    assert (np.diff(gains[6:]) < 0).all()

    k_p_max = 1.0
    test_controller = BatteryController(
        test_interface,
        test_hercules_dict,
        {"k_p_max":k_p_max, "k_p_min":k_p_min}
    )

    pow_ref = 1000
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.5}
    test_hercules_dict["external_signals"]["plant_power_reference"] = pow_ref
    test_controller.step(test_hercules_dict)
    power_setpoint_mid_soc = test_controller.controls_dict["power_setpoint"]
    
    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.8}
    test_controller.step(test_hercules_dict)
    power_setpoint_midhigh_soc = test_controller.controls_dict["power_setpoint"]

    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.2}
    test_controller.step(test_hercules_dict)
    power_setpoint_midlow_soc = test_controller.controls_dict["power_setpoint"]

    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 1.0}
    test_controller.step(test_hercules_dict)
    power_setpoint_high_soc = test_controller.controls_dict["power_setpoint"]

    test_hercules_dict["py_sims"]["test_battery"]["outputs"] = {"power": 0, "soc": 0.0}
    test_controller.step(test_hercules_dict)
    power_setpoint_low_soc = test_controller.controls_dict["power_setpoint"]

    assert power_setpoint_mid_soc == pow_ref
    assert power_setpoint_midhigh_soc < power_setpoint_mid_soc
    assert power_setpoint_midlow_soc < power_setpoint_mid_soc
    assert power_setpoint_midhigh_soc == power_setpoint_midlow_soc
    assert power_setpoint_high_soc < power_setpoint_midhigh_soc
    assert power_setpoint_low_soc < power_setpoint_midlow_soc
