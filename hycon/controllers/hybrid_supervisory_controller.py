import copy

import numpy as np

from hycon.controllers.controller_base import ControllerBase


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
        cname="supervisor",
        controller_parameters={},
        verbose=False,
    ):
        """
        Instantiate HybridSupervisoryControllerGeneric.

        Args:
            interface: The controller's interface to the plant.
            cname: The name of the controller, which should correspond to a key in the plant
                parameters dictionary. Defaults to "supervisor".
            controller_parameters: Dictionary of controller parameters. Should include keys
                "component_controllers" and "curtailment_order". See set_controller_parameters for
                details. Defaults to empty dictionary.
            verbose: Whether to print additional information during controller operation.
        """
        super().__init__(interface=interface, cname=cname, verbose=verbose)

        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

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

    def set_controller_parameters(self, component_controllers=[], curtailment_order=None):
        """
        Set controller parameters for HybridSupervisoryControllerGeneric.

        Args:
            component_controllers: List of component controllers to coordinate. Should be
                instantiated Hycon-compatible controllers with cnames corresponding to the plant
                components in the simulation.
            curtailment_order: List of integers corresponding to the order in which to curtail
                components when the overall power reference exceeds the interconnection limit.
        """

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
                # Ask to charge at the full charge rate---this will then indicate whether the
                # storage would like to charge, for more complex battery controllers.
                standin_measurements_dict = copy.deepcopy(measurements_dict)
                standin_measurements_dict[cc.cname]["power_reference"] = -cc.plant_parameters[
                    cc.cname
                ]["charge_rate"]
                total_available_storage_for_charging += np.maximum(
                    0,
                    -cc.compute_controls_without_updating_state(standin_measurements_dict)[
                        cc.cname
                    ]["power_setpoint"],
                )

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
        locally_generated_power_total = 0.0
        controls_dict = {}

        # Compute total locally generated power from generators
        locally_generated_power_total = sum(
            [
                measurements_dict[cc.cname]["power"]
                for cc in self.component_controllers
                if cc.plant_parameters[cc.cname]["component_category"] == "generator"
            ]
        )

        # Loop over curtailment order in reverse to bring in power for each component until we hit
        # the interconnection limit, then curtail as needed according to the order.
        for cidx in self.curtailment_order[::-1]:
            cc = self.component_controllers[cidx]

            if cc.plant_parameters[cc.cname]["component_category"] == "generator":
                power_reference_component = power_reference_with_storage - power_export_total
            elif cc.plant_parameters[cc.cname]["component_category"] == "storage":
                if cc.plant_parameters[cc.cname].get("allow_grid_charging", True):
                    power_reference_component = power_reference_total - power_export_total
                else:
                    power_reference_component = max(
                        power_reference_total - power_export_total,
                        -locally_generated_power_total,
                    )
                    # Reduce or increase the available power to store
                    locally_generated_power_total += measurements_dict[cc.cname]["power"]

            # Assign power_reference_component the upper limit for the component's power output,
            # as well as the power reference. Component controllers can then chose which to use.
            measurements_dict[cc.cname]["power_limit_upper"] = power_reference_component  # Not used
            measurements_dict[cc.cname]["power_reference"] = power_reference_component

            controls_dict.update(cc.compute_controls(measurements_dict))

            power_export_total += measurements_dict[cc.cname]["power"]

        return controls_dict
