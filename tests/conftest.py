import pytest
from hycon.interfaces import HerculesADInterface, HerculesHybridADInterface, HerculesInterface
from hycon.interfaces.interface_base import InterfaceBase


@pytest.fixture
def test_hercules_v1_dict():
    return {
        "dt": 1,
        "time": 0,
        "controller": {
            "num_turbines": 2,
            "initial_conditions": {"yaw": [270.0, 270.0]},
            "nominal_plant_power_kW": 10000,
            "nominal_hydrogen_rate_kgps": 0.1,
            "hydrogen_controller_gain": 1.0,
        },
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
                "charge_rate": 20,
                "discharge_rate": 20,
            },
            "test_solar": {"outputs": {"power_mw": 1.0, "dni": 1000.0, "aoi": 30.0}},
            "test_hydrogen": {"outputs": {"H2_mfr": 0.03}},
            "inputs": {},
        },
        "external_signals": {
            "wind_power_reference": 1000.0,
            "plant_power_reference": 1000.0,
            "hydrogen_reference": 0.02,
        },
    }


@pytest.fixture
def test_hercules_dict():
    return {
        "dt": 1,
        "time": 0,
        "plant": {"interconnect_limit": None},
        "controller": {
            "test_controller_parameter": 1.0,
        },
        "wind_farm": {
            "n_turbines": 2,
            "capacity": 10000.0,
            "wind_direction_mean": 271.0,
            "turbine_powers": [4000.0, 4001.0],
            "power": 8001.0,
            "wind_speed": 10.0,
            "component_type": "WindFarm",
            "component_category": "generator",
        },
        "solar_farm": {
            "capacity": 1000.0,
            "power": 1000.0,  # kW
            "dni": 1000.0,
            "aoi": 30.0,
            "component_type": "SolarPySAMPVWatts",
            "component_category": "generator",
        },
        "battery": {
            "size": 10.0e3,
            "energy_capacity": 40.0e3,
            "power": 10.0e3,
            "soc": 0.3,
            "charge_rate": 20e3,
            "discharge_rate": 15e3,
            "component_type": "BatterySimple",
            "component_category": "storage",
        },
        "electrolyzer": {
            "H2_mfr": 0.03,
            "component_type": "ElectrolyzerPlant",
            "component_category": "load",
            "power": 500.0,
        },
        "external_signals": {
            "wind_power_reference": 1000.0,
            "solar_power_reference": 800.0,
            "battery_power_reference": 0.0,
            "plant_power_reference": 1000.0,
            "forecast_ws_mean_0": 8.0,
            "forecast_ws_mean_1": 8.1,
            "ws_median_0": 8.1,
            "hydrogen_reference": 0.02,
        },
        "component_names": ["wind_farm", "solar_farm", "battery", "electrolyzer"],
    }


class StandinInterface(InterfaceBase):
    """
    Empty class to test controllers.
    """

    def __init__(self):
        super().__init__()
        self.dt = 1.0
        # Set up stand-in plant parameters and controller parameters
        self.plant_parameters = {"n_turbines": 2}
        self.controller_parameters = {}

    def get_measurements(self):
        pass

    def check_controls(self):
        pass

    def send_controls(self):
        pass


@pytest.fixture
def test_interface_standin():
    return StandinInterface()


@pytest.fixture
def test_interface_hercules(test_hercules_dict):
    """
    Fixture to create a Hercules v2 dictionary for testing.
    """
    return HerculesInterface(test_hercules_dict)


@pytest.fixture
def test_interface_hercules_ad(test_hercules_v1_dict):
    """
    Fixture to create a HerculesADInterface for testing.
    """
    return HerculesADInterface(test_hercules_v1_dict)


@pytest.fixture
def test_interface_hercules_hybrid_ad(test_hercules_v1_dict):
    """
    Fixture to create a HerculesHybridADInterface for testing.
    """
    test_hercules_v1_dict["controller"]["num_batteries"] = 1
    test_hercules_v1_dict["controller"]["num_solar"] = 1
    return HerculesHybridADInterface(test_hercules_v1_dict)


@pytest.fixture
def floris_dict():
    """
    Fixture to create a FLORIS dictionary for testing.
    """
    return {
        "name": "test_input",
        "description": "Two-turbine farm for testing",
        "floris_version": "v4",
        "logging": {
            "console": {"enable": False, "level": "WARNING"},
            "file": {"enable": False, "level": "WARNING"},
        },
        "solver": {"type": "turbine_grid", "turbine_grid_points": 3},
        "farm": {
            "layout_x": [0.0, 500.0],
            "layout_y": [0.0, 0.0],
            "turbine_type": ["nrel_5MW"],
        },
        "flow_field": {
            "air_density": 1.225,
            "reference_wind_height": 90.0,
            "turbulence_intensities": [0.06],
            "wind_directions": [270.0],
            "wind_shear": 0.12,
            "wind_speeds": [8.0],
            "wind_veer": 0.0,
        },
        "wake": {
            "model_strings": {
                "combination_model": "sosfs",
                "deflection_model": "gauss",
                "turbulence_model": "crespo_hernandez",
                "velocity_model": "gauss",
            },
            "enable_secondary_steering": True,
            "enable_yaw_added_recovery": True,
            "enable_active_wake_mixing": True,
            "enable_transverse_velocities": True,
            "wake_deflection_parameters": {
                "gauss": {
                    "ad": 0.0,
                    "alpha": 0.58,
                    "bd": 0.0,
                    "beta": 0.077,
                    "dm": 1.0,
                    "ka": 0.38,
                    "kb": 0.004,
                },
            },
            "wake_velocity_parameters": {
                "gauss": {"alpha": 0.58, "beta": 0.077, "ka": 0.38, "kb": 0.004},
            },
            "wake_turbulence_parameters": {
                "crespo_hernandez": {
                    "initial": 0.01,
                    "constant": 0.9,
                    "ai": 0.83,
                    "downstream": -0.25,
                }
            },
        },
    }
