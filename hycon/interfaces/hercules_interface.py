import copy

from hycon.interfaces.interface_base import InterfaceBase

# List of channels that may be present in the hercules component data that the controller needs.
# Key: Hercules name. Value: Name to use in controller measurements dictionary
hercules_data_channel_map = {
    "power" : "power",
    "soc" : "state_of_charge",
    "turbine_powers" : "turbine_powers",
    "turbine_speeds" : "turbine_speeds",
    "wind_direction_mean" : "wind_direction_mean",
    "dni" : "direct_normal_irradiance",
    "aoi" : "angle_of_incidence",
    "H2_mfr" : "production_rate",
}


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
            self.plant_parameters["solar_farm"] = {"capacity": h_dict["solar_farm"]["capacity"]}

        # Battery parameters
        if self._has_battery_component:
            self.plant_parameters["battery"] = {
                "power_capacity": h_dict["battery"]["size"],
                "energy_capacity": h_dict["battery"]["energy_capacity"],
                "charge_rate": h_dict["battery"]["charge_rate"],
                "discharge_rate": h_dict["battery"]["discharge_rate"],
                "allow_grid_power_consumption": h_dict["battery"].get(
                    "allow_grid_power_consumption", False
                ),
            }

        # Electrolyzer parameters (placeholder for future electrolyzer parameters)
        if self._has_hydrogen_component:
            self.plant_parameters["hydrogen"] = {}

        # Pre-compute LMP keys to avoid string formatting in get_measurements
        self._lmp_da_keys = tuple(f"lmp_da_{h:02d}" for h in range(24))

    def check_controls(self, controls_dict):
        available_controls = [
            "power_setpoint",
            "power_setpoints",
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

        # Loop over components in simulation
        for c in h_dict["component_names"]:
            component_power = h_dict[c]["power"]
            total_power += component_power
            component_measurements = {"power" : component_power}
            for k, v in hercules_data_channel_map.items():
                if k in h_dict[c]:
                    component_measurements[v] = h_dict[c][k]

            # Assign to main measurements dictionary
            measurements[c] = component_measurements

        # Record total power
        measurements["total_power"] = total_power

        ## Handle external signals (somewhat hardcoded; can add more as needed)
        measurements["plant_power_reference"] = h_dict["external_signals"].get(
            "plant_power_reference", None
        )

        # TODO: how to pass hydrogen reference to the particular component?
        # measurements["hydrogen"]["power_reference"] = h_dict["external_signals"].get(
        #     "hydrogen_reference", 0
        # )

        # Grid price information (using pre-computed keys for performance)
        if "lmp_da_00" in h_dict["external_signals"]:
            measurements["DA_LMP_24hours"] = [
                h_dict["external_signals"][k] for k in self._lmp_da_keys
            ]
        measurements["DA_LMP"] = h_dict["external_signals"].get("lmp_da", None) # TODO: used?
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
        # Overwrite h_dict elements with controls_dict
        h_dict = h_dict | controls_dict

        return h_dict
