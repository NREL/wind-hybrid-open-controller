# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation
import numpy as np
import pandas as pd

# import pandas as pd
from whoc.controllers import (
    HybridSupervisoryControllerBaseline,
    HybridSupervisoryControllerBaseline_ForecastDemo,
    LookupBasedWakeSteeringController,
    WindBatteryController,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)
from whoc.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from whoc.interfaces import (
    HerculesADInterface,
    HerculesHybridADInterface,
    HerculesWindBatteryInterface,
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
        "test_battery": {"outputs": {"power": 10.0, "soc": 0.3}, "charge_rate":20},
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
    _ = WindBatteryController(interface=test_interface, input_dict=test_hercules_dict)
    _ = WindFarmPowerDistributingController(interface=test_interface, input_dict=test_hercules_dict)
    _ = WindFarmPowerTrackingController(interface=test_interface, input_dict=test_hercules_dict)
    _ = HybridSupervisoryControllerBaseline(interface=test_interface, input_dict=test_hercules_dict)
    _ = HybridSupervisoryControllerBaseline_ForecastDemo(
        interface=test_interface, input_dict=test_hercules_dict
    )


def test_LookupBasedWakeSteeringController():
    test_interface = HerculesADInterface(test_hercules_dict)

    # No lookup table passed; simply passes through wind direction to yaw angles
    test_controller = LookupBasedWakeSteeringController(
        interface=test_interface,
        input_dict=test_hercules_dict
    )

    # Check that the controller can be stepped
    test_hercules_dict["time"] = 20
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
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
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_angles = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_yaw_angles"]
    )
    wind_directions = np.array(
        test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_wind_directions"]
    )
    assert np.allclose(test_angles, wind_directions - test_offsets)


def test_WindBatteryController():
    test_interface = HerculesWindBatteryInterface(test_hercules_dict)
    test_controller = WindBatteryController(test_interface, test_hercules_dict)

    # Check the low level methods behave as expected
    test_controller._receive_measurements(test_hercules_dict)

    wind_setpoints = test_controller.calc_wind_setpoints()
    assert not wind_setpoints  # wind setpoints should be empty

    battery_setpoints = test_controller.calc_battery_setpoints()
    assert battery_setpoints["signal"] == -500  # battery setpoints should not be empty

    test_controller.compute_controls()
    assert test_controller.setpoints_dict == {"wind": wind_setpoints, "battery": battery_setpoints}

    # Test step
    # We will need to change these cases when the wind_battery_controller has more general behavior
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [450, 450]
    hercules_dict_out = test_controller.step(test_hercules_dict)
    assert hercules_dict_out["py_sims"]["inputs"]["battery_signal"] == 900

    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [550, 550]
    hercules_dict_out = test_controller.step(test_hercules_dict)
    assert hercules_dict_out["py_sims"]["inputs"]["battery_signal"] == -500


def test_WindFarmPowerDistributingController():
    test_interface = HerculesADInterface(test_hercules_dict)

    test_controller = WindFarmPowerDistributingController(
        interface=test_interface, input_dict=test_hercules_dict
    )

    # Default behavior when no power reference is given
    test_hercules_dict["time"] = 20
    test_hercules_dict["external_signals"] = {}
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(
        test_power_setpoints,
        POWER_SETPOINT_DEFAULT / test_hercules_dict["controller"]["num_turbines"],
    )

    # Test with power reference
    test_hercules_dict["external_signals"]["wind_power_reference"] = 1000
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)


def test_WindFarmPowerTrackingController():
    test_interface = HerculesADInterface(test_hercules_dict)

    test_controller = WindFarmPowerTrackingController(
        interface=test_interface, input_dict=test_hercules_dict
    )

    # Test no change to power setpoints if producing desired power
    test_hercules_dict["external_signals"]["wind_power_reference"] = 1000
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [500, 500]
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert np.allclose(test_power_setpoints, 500)

    # Test if power exceeds farm reference, power setpoints are reduced
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        <= test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test if power is less than farm reference, power setpoints are increased
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [550, 400]
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (
        test_power_setpoints
        >= test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"]
    ).all()

    # Test that more aggressive control leads to faster response
    test_controller = WindFarmPowerTrackingController(
        interface=test_interface, input_dict=test_hercules_dict, proportional_gain=2
    )
    test_hercules_dict["hercules_comms"]["amr_wind"]["test_farm"]["turbine_powers"] = [600, 600]
    test_hercules_dict_out = test_controller.step(hercules_dict=test_hercules_dict)
    test_power_setpoints_a = np.array(
        test_hercules_dict_out["hercules_comms"]["amr_wind"]["test_farm"]["turbine_power_setpoints"]
    )
    assert (test_power_setpoints_a < test_power_setpoints).all()

def test_HybridSupervisoryControllerBaseline():
    test_interface = HerculesHybridADInterface(test_hercules_dict)

    test_controller = HybridSupervisoryControllerBaseline(
        interface=test_interface, input_dict=test_hercules_dict
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

def test_HybridSupervisoryControllerBaseline_ForecastDemo():
    test_interface = HerculesHybridADInterface(test_hercules_dict)

    test_controller = HybridSupervisoryControllerBaseline_ForecastDemo(
        interface=test_interface, input_dict=test_hercules_dict
    )

    test_controller.step(test_hercules_dict)
    # test_hercules_dict doesn't have forecast data, so should return empty dict
    assert test_controller.return_forecast() == {}

    # Add forecast data to test_hercules_dict
    test_forecast_dict = {
        "forecast_ws_mean_0": 8.0,
        "forecast_ws_mean_1": 8.1,
        "forecast_ws_std_0": 2.0,
        "forecast_ws_std_1": 2.1,
        "ws_median_0": 0.0, # Doesn't contain "forecast", will be ignored
    }
    test_hercules_dict["external_signals"] = test_forecast_dict
    test_controller.step(test_hercules_dict)
    forecast_dict = test_controller.return_forecast()

    del test_forecast_dict["ws_median_0"]

    assert forecast_dict == test_forecast_dict
