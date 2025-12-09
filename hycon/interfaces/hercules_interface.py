import copy

from hycon.controllers.wind_farm_power_tracking_controller import POWER_SETPOINT_DEFAULT
from hycon.interfaces.interface_base import InterfaceBase


class HerculesInterface(InterfaceBase):
    """
    Class for interfacing with Hercules v2 simulator.
    """
    def __init__(self, h_dict):
        super().__init__()
        self.dt = h_dict["dt"]

        # Controller parameters
        if "controller" in h_dict and h_dict["controller"] is not None:
            self.controller_parameters = copy.deepcopy(h_dict["controller"])
        else:
            self.controller_parameters = {}

        # Plant parameters
        if "plant" in h_dict and h_dict["plant"] is not None:
            self.plant_parameters = copy.deepcopy(h_dict["plant"])
        else:
            self.plant_parameters = {}

        # Determine which components are present in the simulation
        self._has_wind_component = "wind_farm" in h_dict
        self._has_solar_component = "solar_farm" in h_dict
        self._has_battery_component = "battery" in h_dict
        self._has_hydrogen_component = "electrolyzer" in h_dict

        # Wind farm parameters
        if self._has_wind_component:
            self.plant_parameters["wind_farm"] = {
                "capacity": h_dict["wind_farm"]["capacity"],
                "n_turbines": h_dict["wind_farm"]["n_turbines"],
                "turbines": range(h_dict["wind_farm"]["n_turbines"]),
            }
            self._n_turbines = self.plant_parameters["wind_farm"]["n_turbines"]
        else:
            self._n_turbines = 0

        # Solar farm parameters
        if self._has_solar_component:
            self.plant_parameters["solar_farm"] = {
                "capacity": h_dict["solar_farm"]["capacity"]
            }

        # Battery parameters
        if self._has_battery_component:
            self.plant_parameters["battery"] = {
                "power_capacity": h_dict["battery"]["size"],
                "energy_capacity": h_dict["battery"]["energy_capacity"],
                "charge_rate": h_dict["battery"]["charge_rate"],
                "discharge_rate": h_dict["battery"]["discharge_rate"],
                "allow_grid_power_consumption": h_dict["battery"].get(
                    "allow_grid_power_consumption",
                    False
                )
            }

        # Electrolyzer parameters (placeholder for future electrolyzer parameters)
        if self._has_hydrogen_component:
            self.plant_parameters["hydrogen"] = {}

        # Pre-compute LMP keys to avoid string formatting in get_measurements
        self._lmp_da_keys = tuple(f"lmp_da_{h:02d}" for h in range(24))

    def check_controls(self, controls_dict):
        available_controls = [
            "wind_power_setpoints",
            "solar_power_setpoint",
            "battery_power_setpoint",
        ]

        for k in controls_dict.keys():
            if k not in available_controls:
                raise ValueError("Setpoint " + k + " is not available in this configuration.")
            if k == "wind_power_setpoints":
                if len(controls_dict[k]) != self._n_turbines:
                    raise ValueError(
                        "Number of wind power setpoints ({0})".format(len(controls_dict[k])) +
                        " must match number of turbines ({0}).".format(self._n_turbines)
                    )

    def get_measurements(self, h_dict):
        time = h_dict["time"]

        # Set up placeholder dictionary
        measurements = {
            "time": time,
            "forecast": {},
        }

        total_power = 0.0

        # Basic wind quantities
        if self._has_wind_component:
            measurements["wind_farm"] = {
                "turbine_powers": h_dict["wind_farm"]["turbine_powers"],
                "wind_directions": [h_dict["wind_farm"]["wind_direction_mean"]]*self._n_turbines,
                # TODO: wind_speeds?
            }
            total_power += sum(measurements["wind_farm"]["turbine_powers"])

        # Basic solar quantities
        if self._has_solar_component:
            measurements["solar_farm"] = {
                "power": h_dict["solar_farm"]["power"],
                "direct_normal_irradiance": h_dict["solar_farm"]["dni"],
                "angle_of_incidence": h_dict["solar_farm"]["aoi"],
            }
            total_power += measurements["solar_farm"]["power"]

        # Basic battery quantities
        if self._has_battery_component:
            measurements["battery"] = {
                "power": h_dict["battery"]["power"],
                "state_of_charge": h_dict["battery"]["soc"],
            }
            total_power += measurements["battery"]["power"]

        # Basic hydrogen quantities
        if self._has_hydrogen_component:
            measurements["hydrogen"] = {
                "production_rate": h_dict["electrolyzer"]["H2_mfr"],
            }

        # Handle external signals (parse and pass to individual components)
        if "external_signals" in h_dict:
            external_signals = h_dict["external_signals"]

            if "plant_power_reference" in external_signals:
                measurements["plant_power_reference"] = external_signals["plant_power_reference"]

            if "wind_power_reference" in external_signals and self._has_wind_component:
                measurements["wind_farm"]["power_reference"] = external_signals[
                    "wind_power_reference"
                ]

            if "solar_power_reference" in external_signals and self._has_solar_component:
                measurements["solar_farm"]["power_reference"] = external_signals[
                    "solar_power_reference"
                ]

            if self._has_battery_component:
                if "battery_power_reference" in external_signals:
                    measurements["battery"]["power_reference"] = external_signals[
                        "battery_power_reference"
                    ]

            if "hydrogen_reference" in external_signals and self._has_hydrogen_component:
                measurements["hydrogen"]["power_reference"] = external_signals["hydrogen_reference"]

            # Grid price information (using pre-computed keys for performance)
            if "lmp_da_00" in external_signals:
                measurements["DA_LMP_24hours"] = [external_signals[k] for k in self._lmp_da_keys]
            if "lmp_da" in external_signals:
                measurements["DA_LMP"] = external_signals["lmp_da"]
            if "lmp_rt" in external_signals:
                measurements["RT_LMP"] = external_signals["lmp_rt"]

            # Special handling for forecast elements
            for k, v in external_signals.items():
                if "forecast" in k:
                    measurements["forecast"][k] = v

        measurements["total_power"] = total_power

        return measurements

    def send_controls(
            self,
            h_dict,
            wind_power_setpoints=None,
            solar_power_setpoint=None,
            battery_power_setpoint=None
        ):
        if wind_power_setpoints is None:
            wind_power_setpoints = [POWER_SETPOINT_DEFAULT] * self._n_turbines
        if solar_power_setpoint is None:
            solar_power_setpoint = POWER_SETPOINT_DEFAULT
        if battery_power_setpoint is None:
            battery_power_setpoint = 0.0

        if self._has_wind_component:
            # Set wind power setpoints
            h_dict["wind_farm"]["turbine_power_setpoints"] = wind_power_setpoints

        if self._has_solar_component:
            # Set solar power setpoint
            h_dict["solar_farm"]["power_setpoint"] = solar_power_setpoint

        if self._has_battery_component:
            # Set battery power setpoint (positive for discharge)
            h_dict["battery"]["power_setpoint"] = battery_power_setpoint

        return h_dict
