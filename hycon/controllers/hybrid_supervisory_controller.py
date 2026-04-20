import numpy as np

from hycon.controllers.controller_base import ControllerBase


class HybridSupervisoryControllerBase(ControllerBase):
    """
    Base class for hybrid supervisory controllers, implementing shared functionality.
    """

    def __init__(
        self,
        interface,
        input_dict,
        wind_controller=None,
        solar_controller=None,
        battery_controller=None,
        verbose=False,
    ):
        super().__init__(interface=interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have

        # Assign the individual asset controllers
        self.wind_controller = wind_controller
        self.solar_controller = solar_controller
        self.battery_controller = battery_controller

        self._has_solar_controller = solar_controller is not None
        self._has_wind_controller = wind_controller is not None
        self._has_battery_controller = battery_controller is not None

        # Initialize power references
        self.wind_reference = 0
        self.solar_reference = 0
        self.battery_reference = 0
        self.prev_battery_power = 0
        self.prev_wind_power = 0
        self.prev_solar_power = 0

    def compute_controls(self, measurements_dict):
        # Run supervisory control logic
        wind_reference, solar_reference, battery_reference = self.supervisory_control(
            measurements_dict
        )

        # Package the controls for the individual controllers, step, and return
        controls_dict = {}
        if self._has_wind_controller:
            measurements_dict["wind_farm"]["power_reference"] = wind_reference
            wind_controls_dict = self.wind_controller.compute_controls(measurements_dict)
            controls_dict["wind_power_setpoints"] = wind_controls_dict["power_setpoints"]
        if self._has_solar_controller:
            measurements_dict["solar_farm"]["power_reference"] = solar_reference
            solar_controls_dict = self.solar_controller.compute_controls(measurements_dict)
            controls_dict["solar_power_setpoint"] = solar_controls_dict["power_setpoint"]
        if self._has_battery_controller:
            measurements_dict["battery"]["power_reference"] = battery_reference
            battery_controls_dict = self.battery_controller.compute_controls(measurements_dict)
            controls_dict["battery_power_setpoint"] = battery_controls_dict["power_setpoint"]

        return controls_dict


class HybridSupervisoryControllerBaseline(HybridSupervisoryControllerBase):
    def __init__(
        self,
        interface,
        input_dict,
        wind_controller=None,
        solar_controller=None,
        battery_controller=None,
        verbose=False,
    ):
        super().__init__(
            interface=interface,
            input_dict=input_dict,
            wind_controller=wind_controller,
            solar_controller=solar_controller,
            battery_controller=battery_controller,
            verbose=verbose,
        )

        if not self._has_wind_controller and not self._has_solar_controller:
            raise ValueError(
                "The HybridSupervisoryControllerBaseline requires that either a solar_controller"
                " or a wind_controller be provided."
            )

    def supervisory_control(self, measurements_dict):
        # Extract measurements sent
        if self._has_wind_controller:
            wind_power = np.array(measurements_dict["wind_farm"]["turbine_powers"]).sum()
        else:
            wind_power = 0

        if self._has_solar_controller:
            solar_power = measurements_dict["solar_farm"]["power"]
        else:
            solar_power = 0

        if self._has_battery_controller:
            battery_power = measurements_dict["battery"]["power"]
            battery_soc = measurements_dict["battery"]["state_of_charge"]
        else:
            battery_power = 0
            battery_soc = 0

        # Handle power_reference or plant_power_reference keys
        if (
            "power_reference" in measurements_dict
            and "plant_power_reference" not in measurements_dict
        ):
            measurements_dict["plant_power_reference"] = measurements_dict["power_reference"]
            del measurements_dict["power_reference"]
        elif (
            "power_reference" not in measurements_dict
            and "plant_power_reference" not in measurements_dict
        ):
            raise KeyError(
                "Either 'power_reference' or 'plant_power_reference' must be provided"
                " in measurements_dict."
            )
        elif (
            "power_reference" in measurements_dict and "plant_power_reference" in measurements_dict
        ):
            raise KeyError(
                "Found both 'power_reference' and 'plant_power_reference' in measurements_dict."
            )
        plant_power_reference = measurements_dict["plant_power_reference"]

        # Filter the wind and solar power measurements to reduce noise and improve closed-loop
        # controller damping
        a = 0.1
        wind_power = (1 - a) * self.prev_wind_power + a * wind_power
        solar_power = (1 - a) * self.prev_solar_power + a * solar_power

        # Calculate battery reference value
        if self._has_battery_controller:
            battery_reference = plant_power_reference - (wind_power + solar_power)
            battery_charge_rate = self.plant_parameters["battery"]["charge_rate"]
        else:
            battery_reference = 0
            battery_charge_rate = 0

        # Decide control gain:
        if (wind_power + solar_power) < (
            plant_power_reference + battery_charge_rate
        ) and battery_power <= 0:
            if battery_soc > 0.89:
                K = ((wind_power + solar_power) - plant_power_reference) / 2
            else:
                K = ((wind_power + solar_power) - (plant_power_reference + battery_charge_rate)) / 2
        else:
            K = ((wind_power + solar_power) - plant_power_reference) / 2

        if not (self._has_wind_controller & self._has_solar_controller):
            # Only one type of generation available, double the control gain
            K = 2 * K

        if (wind_power + solar_power) > (plant_power_reference + battery_charge_rate) or (
            (wind_power + solar_power) > (plant_power_reference) and battery_soc > 0.89
        ):
            # go down
            wind_reference = wind_power - K
            solar_reference = solar_power - K
        else:
            # go up
            # Is the resource saturated?
            if self.solar_reference > (self.prev_solar_power + 0.05 * self.solar_reference):
                solar_reference = self.solar_reference
            else:
                # If not, ask for more power
                solar_reference = solar_power - K

            if self.wind_reference > (self.prev_wind_power + 0.05 * self.wind_reference):
                wind_reference = self.wind_reference
            else:
                wind_reference = wind_power - K

        # Reset references for invalid controllers
        if not self._has_wind_controller:
            wind_reference = 0
        if not self._has_solar_controller:
            solar_reference = 0

        self.prev_solar_power = solar_power
        self.prev_wind_power = wind_power
        self.prev_battery_power = battery_power
        self.wind_reference = wind_reference
        self.solar_reference = solar_reference
        self.battery_reference = battery_reference

        return wind_reference, solar_reference, battery_reference


class HybridSupervisoryControllerMultiRef(HybridSupervisoryControllerBase):
    """
    Modified version of HybridSupervisoryControllerBaseline that accepts
    individual references for wind and solar generation and respects an
    interconnection limit.
    """

    def __init__(
        self,
        interface,
        input_dict,
        wind_controller=None,
        solar_controller=None,
        battery_controller=None,
        verbose=False,
    ):
        super().__init__(
            interface=interface,
            input_dict=input_dict,
            wind_controller=wind_controller,
            solar_controller=solar_controller,
            battery_controller=battery_controller,
            verbose=verbose,
        )

        # Extract interconnection limit
        if "interconnect_limit" in self.plant_parameters:
            if (
                not isinstance(self.plant_parameters["interconnect_limit"], (float, int))
                or self.plant_parameters["interconnect_limit"] <= 0
            ):
                raise ValueError("interconnect_limit must be a positive value.")
        else:
            raise KeyError("interconnect_limit must be specified to use this controller.")

        # Establish curtailment protocols
        default_curtailment_order = ["battery", "solar", "wind"]
        default_curtailment_order = [
            c
            for c, a in zip(
                default_curtailment_order,
                [
                    self._has_battery_controller,
                    self._has_solar_controller,
                    self._has_wind_controller,
                ],
            )
            if a
        ]
        if "curtailment_order" in self.controller_parameters:
            # Check that curtailment order does not contain any invalid components
            for component in self.controller_parameters["curtailment_order"]:
                if component not in default_curtailment_order:
                    raise ValueError(
                        f"Invalid component {component} in curtailment_order. "
                        "Valid components based on configuration provided are: "
                        ", ".join(default_curtailment_order)
                    )
            self.curtailment_order = self.controller_parameters["curtailment_order"]
        else:
            self.curtailment_order = default_curtailment_order

    def supervisory_control(self, measurements_dict):
        """
        Overwrite HybridSupervisoryControllerBaseline.supervisory_control()
        with controller that follows separate setpoints and curtails in order.
        """

        # Extract measurements sent
        if self._has_wind_controller:
            wind_power = np.array(measurements_dict["wind_farm"]["turbine_powers"]).sum()
            wind_reference = measurements_dict["wind_farm"].get(
                "power_reference", self.plant_parameters["wind_farm"]["capacity"]
            )
            wind_reference = np.minimum(
                wind_reference, self.plant_parameters["wind_farm"]["capacity"]
            )
        else:
            wind_power = 0
            wind_reference = 0

        if self._has_solar_controller:
            solar_power = measurements_dict["solar_farm"]["power"]
            solar_reference = measurements_dict["solar_farm"].get(
                "power_reference", self.plant_parameters["solar_farm"]["capacity"]
            )
            solar_reference = np.minimum(
                solar_reference, self.plant_parameters["solar_farm"]["capacity"]
            )
        else:
            solar_power = 0
            solar_reference = 0

        if self._has_battery_controller:
            battery_power = measurements_dict["battery"]["power"]
            if "power_reference" in measurements_dict["battery"]:
                battery_reference = measurements_dict["battery"].get("power_reference", 0)
            else:
                battery_reference = 0
            battery_reference = np.minimum(
                battery_reference, self.plant_parameters["battery"]["discharge_rate"]
            )
            battery_reference = np.maximum(
                battery_reference, -1 * self.plant_parameters["battery"]["charge_rate"]
            )
        else:
            battery_power = 0
            battery_reference = 0

        # Filter the wind and solar power measurements to reduce noise and improve closed-loop
        # controller damping
        # TODO RECONSIDER THIS MAYBE MAKE MORE DEPENDENT ON THE TIME STEP
        a = 1.0  # 0.1 # FORCE THE FILTER TO BE 100% DEPENDENT ON THE CURRENT TIME STEP
        wind_power = (1 - a) * self.prev_wind_power + a * wind_power
        solar_power = (1 - a) * self.prev_solar_power + a * solar_power
        battery_power = (1 - a) * self.prev_battery_power + a * battery_power

        # Loop over the curtailment order in reverse order to progressively reduce the reference
        # of the first component in the order
        unconstrained_power = 0.0

        # If battery power is negative (charging), immediately include it in the unconstrained power
        if battery_power < 0:
            unconstrained_power += battery_power

        for component in reversed(self.curtailment_order):
            if component == "wind":
                wind_reference = np.minimum(
                    wind_reference,
                    self.plant_parameters["interconnect_limit"] - unconstrained_power,
                )
                unconstrained_power += wind_power
            elif component == "solar":
                solar_reference = np.minimum(
                    solar_reference,
                    self.plant_parameters["interconnect_limit"] - unconstrained_power,
                )
                unconstrained_power += solar_reference
            elif component == "battery":
                battery_reference = np.minimum(
                    battery_reference,
                    self.plant_parameters["interconnect_limit"] - unconstrained_power,
                )
                if battery_power < 0:  # Make sure not to double count battery power when charging
                    unconstrained_power += battery_power
            else:
                raise ValueError(f"Invalid generation type {component} in curtailment_order.")

        self.prev_solar_power = solar_power
        self.prev_wind_power = wind_power
        self.prev_battery_power = battery_power
        self.wind_reference = wind_reference
        self.solar_reference = solar_reference
        self.battery_reference = battery_reference

        return wind_reference, solar_reference, battery_reference

    # TODO: Need to add it's own compute_controls method that ensures interconnect is satisfied



class HybridSupervisoryController_DGL_PriceSOCBattery(HybridSupervisoryControllerBase):
    """
    This controller includes DGL as in HybridSupervisoryController_DGL_IntegratedBattery, but
    it is meant to work with the price-based BatteryPriceSOCController_ChargeFromPlant controller
    rather then the simpler integrated battery controller.
    """

    def __init__(
        self,
        interface,
        input_dict,
        wind_controller=None,
        solar_controller=None,
        battery_controller=None,
        controller_parameters={},
        verbose=False,
    ):
        super().__init__(
            interface=interface,
            input_dict=input_dict,
            wind_controller=wind_controller,
            solar_controller=solar_controller,
            battery_controller=battery_controller,
            verbose=verbose,
        )

        # Extract interconnection limit
        if "interconnect_limit" in self.plant_parameters:
            if (
                not isinstance(
                    self.plant_parameters["interconnect_limit"], (float, int)
                )
                or self.plant_parameters["interconnect_limit"] <= 0
            ):
                raise ValueError("interconnect_limit must be a positive value.")
        else:
            self.plant_parameters["interconnect_limit"] = np.inf

        # Initialize the dynamic interconnect limit to the interconnect limit
        self.dynamic_grid_limit = self.plant_parameters["interconnect_limit"]
        self.dgl_prev_state = 1  # -1 / 0 / 1 (negative price / 0 / positive price)

        # Get the input controller parameters
        input_controller_parameters = input_dict["controller"]
        if input_controller_parameters is None:
            input_controller_parameters = {}

        # Check that parameters are not specified both in input file
        # and in controller_parameters
        for cp in controller_parameters.keys():
            if cp in input_controller_parameters:
                raise KeyError(
                    'Found key "' + cp + '" in both input_dict["controller"] and'
                    " in controller_parameters."
                )
        controller_parameters = {**controller_parameters, **input_controller_parameters}
        self.set_controller_parameters(**controller_parameters)

        # Set initial values for filtered powers
        self._wind_ref_filtered = 0.0
        self._solar_ref_filtered = 0.0
        self._batt_ref_filtered = 0.0

    def set_controller_parameters(
        self,
        filter_time_constant=0.0,
        curtailment_order=None,
        allow_grid_charging=False,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        # Establish curtailment protocols
        default_curtailment_order = ["battery", "solar", "wind"]
        default_curtailment_order = [
            c
            for c, a in zip(
                default_curtailment_order,
                [
                    self._has_battery_controller,
                    self._has_solar_controller,
                    self._has_wind_controller,
                ],
                strict=False,
            )
            if a
        ]
        if curtailment_order is not None:
            # Check that curtailment order does not contain any invalid components
            for component in curtailment_order:
                if component not in default_curtailment_order:
                    raise ValueError(
                        f"Invalid component {component} in curtailment_order. "
                        "Valid components based on configuration provided are: "
                        ", ".join(default_curtailment_order)
                    )
            self.curtailment_order = curtailment_order
        else:
            self.curtailment_order = default_curtailment_order
        # TODO: consider adding a price-ordered version of curtailment order.

        self._b = self.dt / (filter_time_constant + self.dt)
        self._a = filter_time_constant / (filter_time_constant + self.dt)

        self.allow_grid_charging = allow_grid_charging

    def _update_dynamic_grid_limit(self, measurements_dict, total_power):
        """
        Update the dynamic interconnection limit (DIL) based on the measurements.
        """
        NEAR_ZERO_PRICE = 0.1  # $/MWh
        RAMP_RATE = 40000  # kW / 5 minutes

        # If this is a 5 minute step
        # TODO: ASSUMING WE START ON AN EVEN TIME AND DT IS SUCH
        # THAT 300 IS AN INTEGER NUMBER OF STEPS
        if measurements_dict["time"] % 300 == 0:
            if measurements_dict["RT_LMP"] < -1 * NEAR_ZERO_PRICE:
                # DECREASING CONDITION
                if self.dgl_prev_state == 1:  # Coming from positive price
                    self.dynamic_grid_limit = total_power - RAMP_RATE
                else:  # continuing negative or 0 price
                    self.dynamic_grid_limit = self.dynamic_grid_limit - RAMP_RATE
                self.dgl_prev_state = -1
            elif measurements_dict["RT_LMP"] < NEAR_ZERO_PRICE:
                # 0 price condition
                if self.dgl_prev_state == 1:  # Coming from positive price
                    self.dynamic_grid_limit = total_power - RAMP_RATE
                # Otherwise no update in 0 price condition
                self.dgl_prev_state = 0
            else:
                self.dynamic_grid_limit = self.dynamic_grid_limit + RAMP_RATE
                self.dgl_prev_state = 1

            # Limit the DIL to the [0, interconnect limit] range
            self.dynamic_grid_limit = np.maximum(self.dynamic_grid_limit, 0.0)
            self.dynamic_grid_limit = np.minimum(
                self.dynamic_grid_limit, self.plant_parameters["interconnect_limit"]
            )

    def supervisory_control(self, measurements_dict, battery_power_setpoint):
        """
        Overwrite HybridSupervisoryControllerBaseline.supervisory_control()
        with controller that follows separate setpoints and curtails in order.
        """

        # # Temporarily hard code these values
        # BATTERY_CHARGE_PRICE = 0 # $/MWh
        # BATTERY_DISCHARGE_PRICE = 5 # $/MWh
        NEAR_ZERO_PRICE = 0.1  # $/MWh

        # Get current power production of the various components
        wind_power = (
            np.array(measurements_dict["wind_farm"]["turbine_powers"]).sum()
            if self._has_wind_controller
            else 0
        )
        solar_power = (
            measurements_dict["solar_farm"]["power"]
            if self._has_solar_controller
            else 0
        )
        battery_power = (
            measurements_dict["battery"]["power"] if self._has_battery_controller else 0
        )

        total_power = wind_power + solar_power + battery_power
        total_power_generating_power = wind_power + solar_power  # + battery_power

        # Update the dynamic interconnect limit
        self._update_dynamic_grid_limit(measurements_dict, total_power)

        # Identify the headroom before account for battery
        dgl_headroom_pre_battery_charging = (
            self.dynamic_grid_limit - total_power_generating_power
        )

        # Compute the dgl_headroom accounting the battery_power_setpoint (instead of actual battery
        # power use the reference to avoid chicken-and-egg)
        if not self.allow_grid_charging or battery_power_setpoint >= 0:
            # If grid charging is not allowed or the battery power setpoint is above zero, then the dgl_headroom is the dgl_headroom_pre_battery_charging
            # minus the battery_power_setpoint
            dgl_headroom = dgl_headroom_pre_battery_charging - battery_power_setpoint
        else:
            # If grid charging is allowed and the setpoint is negative, then the dgl_headroom is the dgl_headroom_pre_battery_charging
            # since the battery is better off charging from the grid than from the local power
            dgl_headroom = dgl_headroom_pre_battery_charging

        # Initialize references
        wind_reference = 0.0
        solar_reference = 0.0
        battery_curtailed_setpoint = 0.0

        for component in self.curtailment_order:
            if component == "wind":
                wind_reference = np.maximum(0.0, wind_power + dgl_headroom)
                dgl_headroom += wind_power
            elif component == "solar":
                solar_reference = np.maximum(0.0, solar_power + dgl_headroom)
                dgl_headroom += solar_power
            elif component == "battery":
                # Limit power only if discharging
                if battery_power_setpoint >= 0:
                    battery_curtailed_setpoint = np.maximum(
                        0.0, battery_power_setpoint + dgl_headroom
                    )
                    battery_curtailed_setpoint = np.minimum(
                        battery_curtailed_setpoint, battery_power_setpoint
                    )
                    dgl_headroom += battery_power_setpoint
                else:
                    battery_curtailed_setpoint = battery_power_setpoint

        # Apply filtering to set points sent back to component controllers
        self._wind_ref_filtered = (
            self._a * self._wind_ref_filtered + self._b * wind_reference
        )
        self._solar_ref_filtered = (
            self._a * self._solar_ref_filtered + self._b * solar_reference
        )
        # self._batt_ref_filtered = (
        #     self._a * self._batt_ref_filtered + self._b * battery_curtailed_setpoint
        # )
        # self._batt_charge_lim_filtered = (
        #     self._a * self._batt_charge_lim_filtered + self._b * battery_charge_limit
        # )
        # self._batt_discharge_lim_filtered = (
        #     self._a * self._batt_discharge_lim_filtered
        #     + self._b * battery_discharge_limit
        # )

        return (
            self._wind_ref_filtered,
            self._solar_ref_filtered,
            battery_curtailed_setpoint,
        )

    def compute_controls(self, measurements_dict):
        # Initialize the controls_dict
        controls_dict = {}

        # Compute the battery setpoint ahead of the supervisory controller
        if self._has_battery_controller:
            # measurements_dict["battery"]["power_reference"] = battery_reference
            battery_controls_dict = self.battery_controller.compute_controls(
                measurements_dict
            )

            controls_dict["battery_power_setpoint"] = battery_controls_dict[
                "power_setpoint"
            ]
        else:
            # No battery controller, so battery setpoint is 0
            controls_dict["battery_power_setpoint"] = 0.0

        # Run supervisory control logic
        wind_reference, solar_reference, battery_curtailed_setpoint = (
            self.supervisory_control(
                measurements_dict,
                battery_power_setpoint=controls_dict["battery_power_setpoint"],
            )
        )

        # Package the controls for the individual controllers, step, and return
        if self._has_wind_controller:
            measurements_dict["wind_farm"]["power_reference"] = wind_reference
            wind_controls_dict = self.wind_controller.compute_controls(
                measurements_dict
            )
            controls_dict["wind_power_setpoints"] = wind_controls_dict[
                "power_setpoints"
            ]
        if self._has_solar_controller:
            measurements_dict["solar_farm"]["power_reference"] = solar_reference
            solar_controls_dict = self.solar_controller.compute_controls(
                measurements_dict
            )
            controls_dict["solar_power_setpoint"] = solar_controls_dict[
                "power_setpoint"
            ]
        if self._has_battery_controller:
            # Overwrite curtailed version
            controls_dict["battery_power_setpoint"] = battery_curtailed_setpoint

            # Finally limit battery charging to available local power if grid charging is not allowed
            if (
                controls_dict["battery_power_setpoint"] < 0
                and not self.allow_grid_charging
            ):
                wind_power = (
                    np.array(measurements_dict["wind_farm"]["turbine_powers"]).sum()
                    if self._has_wind_controller
                    else 0
                )
                solar_power = (
                    measurements_dict["solar_farm"]["power"]
                    if self._has_solar_controller
                    else 0
                )

                controls_dict["battery_power_setpoint"] = np.max(
                    [
                        controls_dict["battery_power_setpoint"],
                        -1 * (wind_power + solar_power),
                    ]
                )

        return controls_dict
