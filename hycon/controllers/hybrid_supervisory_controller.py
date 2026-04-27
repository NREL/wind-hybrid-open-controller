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
            controls_dict.update(self.wind_controller.compute_controls(measurements_dict))
        if self._has_solar_controller:
            measurements_dict["solar_farm"]["power_reference"] = solar_reference
            controls_dict.update(self.solar_controller.compute_controls(measurements_dict))
        if self._has_battery_controller:
            measurements_dict["battery"]["power_reference"] = battery_reference
            controls_dict.update(self.battery_controller.compute_controls(measurements_dict))

        return controls_dict


class HybridSupervisoryControllerGeneric(ControllerBase):
    """
    HybridSupervisoryControllerGeneric is a supervisory controller for a hybrid
    plant with an arbitrary set of components. These components may be heterogeneous
    or homogeneous (e.g. multiple solar farms), or a mixture (e.g. two solar farms combined with
    one natural gas plant).
    """

    def __init__(
        self,
        interface,
        input_dict,
        cname="supervisor",
        component_controllers=[],
        curtailment_order=None,
        verbose=False,
    ):
        """

        Args:
            interface: The controller's interface to the plant.
            input_dict: Dictionary containing any additional information needed to initialize the
                controller.
            component_controllers: List of controllers for the individual components in the plant.
                Must be in the same order as the components are listed in the plant parameters.
            curtailment_order: List of component names corresponding to the order in which
                components should be curtailed to satisfy interconnection limits. The first element
                in the list will be curtailed first.
            verbose: Whether to print additional information during controller operation.
        """
        super().__init__(interface=interface, cname=cname, verbose=verbose)

        # Check valid component_controllers
        if len(component_controllers) == 0:
            raise ValueError(
                "component_controllers cannot be empty. "
                "At least one component controller must be provided."
            )
        else:
            self.component_controllers = component_controllers

        # Check valid curtailment_order
        if curtailment_order is None:
            # Default is reverse order of component_controllers
            self.curtailment_order = list(range(len(component_controllers) - 1, -1, -1))
        elif len(curtailment_order) != len(component_controllers):
            raise ValueError("curtailment_order must be the same length as component_controllers.")
        elif not all([type(c) is int and c >= 0 for c in curtailment_order]):
            raise ValueError(
                "All entries in curtailment_order must be non-negative integers corresponding to "
                "indices of component_controllers."
            )
        elif (
            max(curtailment_order) != len(component_controllers) - 1 or min(curtailment_order) != 0
        ):
            raise ValueError(
                "curtailment_order must contain integers corresponding to indices of "
                "component_controllers."
            )
        elif len(curtailment_order) != len(set(curtailment_order)):
            raise ValueError("curtailment_order must not contain duplicate entries.")
        else:
            self.curtailment_order = curtailment_order

        # Extract interconnection limit, if specified
        self.static_interconnect_limit = self.plant_parameters.get("interconnect_limit", np.inf)
        if self.static_interconnect_limit == -1 or self.static_interconnect_limit is None:
            self.static_interconnect_limit = np.inf
        if (
            not isinstance(self.static_interconnect_limit, (float, int))
            or self.static_interconnect_limit < -1
        ):
            raise ValueError(
                "interconnect_limit must be a positive value (or -1, indicating no limit)."
            )

    def compute_controls(self, measurements_dict):
        """
        Pass necessary information to each component controller, and apply power
        capping/curtailment.
        """

        # Compute available storage for charging
        total_available_storage_for_charging = 0.0
        for cc in self.component_controllers:
            if cc.plant_parameters[cc.cname]["component_category"] == "storage" and not np.isclose(
                measurements_dict[cc.cname]["state_of_charge"],
                cc.plant_parameters[cc.cname].get("state_of_charge_max", 1.0),
                atol=1e-2,  # Within 1% of max SOC, assume storage is fully charged
            ):
                # TODO: Check if the controller _wants_ to charge
                total_available_storage_for_charging += cc.plant_parameters[cc.cname]["charge_rate"]

        # Establish dynamic upper limit

        # Get overall reference, and remove from measurements_dict to avoid confusion for
        # component controllers.
        if "plant_power_reference" in measurements_dict:
            provided_power_reference = measurements_dict.pop("plant_power_reference")
        elif "power_reference" in measurements_dict:
            provided_power_reference = measurements_dict.pop("power_reference")
        else:
            provided_power_reference = None

        power_reference_total = min(
            self.static_interconnect_limit,
            measurements_dict.get("dynamic_interconnect_limit", np.inf),
            provided_power_reference if provided_power_reference is not None else np.inf,
        )
        power_reference_with_storage = power_reference_total + total_available_storage_for_charging

        # Initialize overall quantities
        power_export_total = 0.0
        controls_dict = {}

        # Loop over curtailment order in reverse to bring in power for each component until we hit
        # the interconnection limit, then curtail as needed according to the order.
        for cidx in self.curtailment_order[::-1]:
            cc = self.component_controllers[cidx]

            if cc.plant_parameters[cc.cname]["component_category"] == "generator":
                power_reference_component = power_reference_with_storage - power_export_total
            elif cc.plant_parameters[cc.cname]["component_category"] == "storage":
                power_reference_component = power_reference_total - power_export_total

            # Assign power_reference_component the upper limit for the component's power output,
            # as well as the power reference. Component controllers can then chose which to use.
            measurements_dict[cc.cname]["power_limit_upper"] = power_reference_component
            measurements_dict[cc.cname]["power_reference"] = power_reference_component

            controls_dict.update(cc.compute_controls(measurements_dict))

            power_export_total += measurements_dict[cc.cname]["power"]

        return controls_dict
