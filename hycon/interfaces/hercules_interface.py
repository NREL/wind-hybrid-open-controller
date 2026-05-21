import copy

from hycon.interfaces.interface_base import InterfaceBase

# List of channels that may be present in the hercules component data that the controller needs.
# Key: Hercules name. Value: Name to use in controller measurements dictionary
hercules_data_channel_map = {
    "power": "power",
    "power_reference": "power_reference",
    "soc": "state_of_charge",
    "turbine_powers": "turbine_powers",
    "turbine_speeds": "turbine_speeds",
    "wind_direction_mean": "wind_direction_mean",
    "dni": "direct_normal_irradiance",
    "aoi": "angle_of_incidence",
    "H2_mfr": "production_rate",
}

# List of valid Hercules component types recognized by Hycon
hercules_wind_types = ["WindFarm"]
hercules_solar_types = ["SolarPySAMPVWatts"]
hercules_battery_types = ["BatteryLithiumIon", "BatterySimple"]
hercules_hydrogen_types = ["ElectrolyzerPlant"]
hercules_thermal_types = ["HardCoalSteamTurbine", "OpenCycleGasTurbine"]


class HerculesInterface(InterfaceBase):
    """
    Class for interfacing with Hercules v2 simulator.
    """

    def __init__(self, h_dict):
        super().__init__()
        self.dt = h_dict["dt"]

        # Plant parameters
        if "plant" in h_dict and h_dict["plant"] is not None:
            self.plant_parameters = copy.deepcopy(h_dict["plant"])
        else:
            self.plant_parameters = {}

        # Determine which components are present in the simulation
        self.component_names = h_dict["component_names"]
        self.component_types = {c: h_dict[c]["component_type"] for c in self.component_names}

        # Extract parameters for various component types
        for c in self.component_names:
            c_type = self.component_types[c]
            if c_type in hercules_wind_types:
                self.plant_parameters[c] = {
                    "type": "wind",  # needed?
                    "component_category": "generator",
                    "capacity": h_dict[c]["capacity"],
                    "n_turbines": h_dict[c]["n_turbines"],
                    "turbines": range(h_dict[c]["n_turbines"]),
                }
            elif c_type in hercules_solar_types:
                self.plant_parameters[c] = {
                    "type": "solar",
                    "component_category": "generator",
                    "capacity": h_dict[c]["capacity"],
                }
            elif c_type in hercules_battery_types:
                self.plant_parameters[c] = {
                    "type": "battery",
                    "component_category": "storage",
                    "power_capacity": h_dict[c]["size"],
                    "energy_capacity": h_dict[c]["energy_capacity"],
                    "charge_rate": h_dict[c]["charge_rate"],
                    "discharge_rate": h_dict[c]["discharge_rate"],
                    "allow_grid_charging": h_dict[c].get("allow_grid_power_consumption", True),
                    "state_of_charge_max": h_dict[c].get("max_SOC", 1.0),
                    "state_of_charge_min": h_dict[c].get("min_SOC", 0.0),
                }
            elif c_type in hercules_hydrogen_types:
                self.plant_parameters[c] = {"type": "hydrogen", "component_category": "load"}
            elif c_type in hercules_thermal_types:
                self.plant_parameters[c] = {"type": "thermal",
                                            "component_category": "generator",
                                            "capacity": h_dict[c]["rated_capacity"],
                                            "min_stable_load": h_dict[c].get("min_stable_load_fraction", 0.0)\
                                                * h_dict[c]["rated_capacity"],
                                            }
            else:
                raise ValueError(f"Component '{c}' has unrecognized type '{c_type}' for Hycon.")

        # # Coal plant parameters
        # if self._has_coal_component:
        #     self.plant_parameters["coal_plant"] = {
        #         "capacity": h_dict["coal_plant"]["rated_capacity"],
        #         "min_stable_load": h_dict["coal_plant"]["min_stable_load_fraction"] * h_dict["coal_plant"]["rated_capacity"]
        #         }

        # Pre-compute LMP keys to avoid string formatting in get_measurements
        self._lmp_da_keys = tuple(f"lmp_da_{h:02d}" for h in range(24))

    def check_controls(self, controls_dict):
        available_controls = [
            "power_setpoint",
        ]

        # Check valid control keys _for each component_ on the hybrid plant
        for c in controls_dict.keys():
            for k in controls_dict[c].keys():
                if k not in available_controls:
                    raise ValueError("Setpoint " + k + " is not available in this configuration.")

    def get_measurements(self, h_dict):
        time = h_dict["time"]

        # Set up placeholder dictionary
        measurements = {
            "time": time,
            "forecast": {},
        }

        total_power = 0.0
        local_power = 0.0

        # Loop over components in simulation
        for c in h_dict["component_names"]:
            component_power = h_dict[c]["power"]
            total_power += component_power
            if self.plant_parameters[c]["component_category"] in ["generator", "storage"]:
                # TODO: Do we need another that excludes storage?
                local_power += component_power
            component_measurements = {"power": component_power}
            for k, v in hercules_data_channel_map.items():
                if k in h_dict[c]:
                    component_measurements[v] = h_dict[c][k]

            # Assign to main measurements dictionary
            measurements[c] = component_measurements

        # Record total power
        measurements["total_power"] = total_power
        measurements["local_power"] = local_power

        ## Handle external signals (somewhat hardcoded; can add more as needed)
        measurements["plant_power_reference"] = h_dict["external_signals"].get(
            "plant_power_reference", None
        )

        # Special handling for wind directions and thermal plant state
        for c in h_dict["component_names"]:
            if self.component_types[c] in hercules_wind_types:
                measurements[c]["wind_directions"] = [
                    h_dict[c]["wind_direction_mean"]
                ] * self.plant_parameters[c]["n_turbines"]
            elif self.component_types[c] in hercules_thermal_types:
                measurements[c]["state"] = h_dict[c]["state"]

        # Handle a variety of external_signals
        if "hydrogen_reference" in h_dict["external_signals"]:
            for c in h_dict["component_names"]:
                if self.component_types[c] in hercules_hydrogen_types:
                    measurements[c]["hydrogen_production_reference"] = h_dict["external_signals"][
                        "hydrogen_reference"
                    ]

        # Handle coal plant specific external signals
        if "plant_status" in h_dict["external_signals"]:
            for c in h_dict["component_names"]:
                if self.component_types[c] in hercules_thermal_types:
                    measurements[c]["status_reference"] = h_dict["external_signals"][
                            "plant_status"
                        ]
        # TODO: @Misha, is there a better way to do this with the new interface?
        if "coal_power_reference" in h_dict["external_signals"]:
            for c in h_dict["component_names"]:
                if self.component_types[c] in hercules_thermal_types:
                    measurements[c]["power_reference"] = h_dict["external_signals"][
                            "coal_power_reference"
                        ]

        # Grid price information (using pre-computed keys for performance)
        if "lmp_da_00" in h_dict["external_signals"]:
            measurements["DA_LMP_24hours"] = [
                h_dict["external_signals"][k] for k in self._lmp_da_keys
            ]
        measurements["DA_LMP"] = h_dict["external_signals"].get("lmp_da", None)  # TODO: used?
        measurements["RT_LMP"] = h_dict["external_signals"].get("lmp_rt", None)

        # Special handling for forecast elements
        for k in h_dict["external_signals"].keys():
            if "forecast" in k:
                measurements["forecast"][k] = h_dict["external_signals"][k]

        # TODO: How to prescribe an override signal for one or more components?

        return measurements

    def send_controls(
        self,
        h_dict,
        controls_dict,
    ):
        controls_dict = copy.deepcopy(controls_dict)
        # Translate controls_dict as needed
        for c in self.component_names:
            if c in controls_dict:
                c_type = self.component_types[c]
                if c_type in hercules_wind_types:
                    if "power_setpoint" not in controls_dict[c]:
                        raise ValueError(
                            "Missing required control 'power_setpoint' for wind component "
                            + c
                            + "."
                        )
                    controls_dict[c]["turbine_power_setpoints"] = controls_dict[c].pop(
                        "power_setpoint"
                    )
                h_dict[c] = h_dict[c] | controls_dict[c]

        return h_dict
