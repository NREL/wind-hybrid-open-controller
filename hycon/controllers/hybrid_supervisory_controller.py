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
        self._interconnect_limit = self.plant_parameters.get("interconnect_limit", np.inf)
        if self._interconnect_limit == -1 or self._interconnect_limit is None:
            self._interconnect_limit = np.inf
        if not isinstance(self._interconnect_limit, (float, int)) or self._interconnect_limit < -1:
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
            self._interconnect_limit,
            measurements_dict.get("dynamic_interconnect_limit", np.inf),
            provided_power_reference if provided_power_reference is not None else np.inf,
        )
        power_reference_with_storage = power_reference_total + total_available_storage_for_charging

        # Initialize overall quantities
        power_export_total = 0.0
        locally_generated_power_total = 0.0
        controls_dict = {}
        curtailment_status = {}
        # Remove keys that are not part of the control output that are returned by thermal
        #   plant controllers (or technologies with similar constraints, e.g. ramp rates or
        #   minimum stable loads)
        keys_to_remove = ["uncurtailable", "ramp_rate"]


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
                    measurements_dict[cc.cname]["power_limit_lower"] = -np.inf
                    measurements_dict[cc.cname]["power_limit_upper"] = power_reference_component
                else:
                    power_reference_component = max(
                        power_reference_total - power_export_total,
                        -locally_generated_power_total,
                    )
                    measurements_dict[cc.cname][
                        "power_limit_lower"
                    ] = -locally_generated_power_total
                    measurements_dict[cc.cname]["power_limit_upper"] = power_reference_component
                    # Reduce or increase the available power to store
                    locally_generated_power_total += measurements_dict[cc.cname]["power"]

            # Assign power_reference_component for use by lower level controller
            measurements_dict[cc.cname]["power_reference"] = power_reference_component

            # Two step process to remove uncurtailable and ramp_rate variables from
            #   compute controls output
            computed_controls = cc.compute_controls(measurements_dict)
            for key in keys_to_remove:
                if key in computed_controls[cc.cname]:
                    if cc.cname not in curtailment_status.keys():
                        curtailment_status[cc.cname] = {}
                    key_value = computed_controls[cc.cname].pop(key)
                    curtailment_status[cc.cname].update({key: key_value})

            controls_dict.update(computed_controls)


            # If component is a flexible generator, base on current power output rather
            #   than power setpoint
            # Need this so that other technologies can fill in for the flexible generator
            #   if it is not producing at full capacity
            if cc.plant_parameters[cc.cname]["component_category"] == "generator" and \
                cc.cname not in curtailment_status.keys():
                power_export_total += measurements_dict[cc.cname]["power"]
            else:
                power_export_total += controls_dict[cc.cname]["power_setpoint"]

        # Check if we are at risk of exceeding interconnection limit
        # This includes checking if we would exceed the interconnection limit with the current
        #   controls, as well as checking if we are close to the interconnection limit and have
        #   ramping limited assets that could cause us to exceed the interconnection limit in
        #   the next time step.
        if [cc for cc in self.component_controllers if \
            cc.plant_parameters[cc.cname]["component_category"] == "generator" and \
            cc.cname in curtailment_status.keys()]:
            ramp_limited_asset_power = sum(
                [
                    measurements_dict[cc.cname]["power"] for cc in self.component_controllers
                    if cc.plant_parameters[cc.cname]["component_category"] == "generator" \
                        and cc.cname in curtailment_status.keys()
                ]            )
            max_ramp_rate_change = sum(
                [
                    curtailment_status[cc.cname]["ramp_rate"] for cc in self.component_controllers
                    if cc.plant_parameters[cc.cname]["component_category"] == "generator" \
                        and cc.cname in curtailment_status.keys()
                ]
            )
            if ramp_limited_asset_power + sum(
                [controls_dict[cc.cname]["power_setpoint"] for cc in self.component_controllers
                if cc.plant_parameters[cc.cname]["component_category"] == "generator" \
                    and cc.cname not in curtailment_status.keys()
                ]
            ) > self._interconnect_limit:
                run_interconnect_curtailment = True
            else:
                run_interconnect_curtailment = False
        else:
            run_interconnect_curtailment = False

        # Run if interconnect curtailment is needed, and if so, curtail according
        #   to curtailment order
        if run_interconnect_curtailment or power_export_total > self._interconnect_limit:
            for cidx in self.curtailment_order:
                cc = self.component_controllers[cidx]
                # Check if component is further curtailable---if so, apply curtailment and
                #   update controls_dict, then check if we're below the interconnection limit.
                # If not curtailable, (like for a thermal unit with ramping constraints or a
                #   minimum stable load), move on to the next component in the curtailment order.
                uncurtailable = curtailment_status.get(cc.cname, False)
                if not uncurtailable:
                    excess_power = power_export_total - self._interconnect_limit
                    # If component is a flexible generator, base on current power 
                    #   output rather than power setpoint
                    if cc.plant_parameters[cc.cname]["component_category"] == "generator" \
                                and cc.cname not in curtailment_status.keys():
                        current_power = measurements_dict[cc.cname]["power"]
                    else:
                        current_power = controls_dict[cc.cname]["power_setpoint"]

                    curtailed_power = max((current_power - excess_power) + max_ramp_rate_change, 0)

                    # Assign power_reference_component for use by lower level controller
                    if cc.plant_parameters[cc.cname]["component_category"] == "generator":
                        measurements_dict[cc.cname]["power_reference"] = curtailed_power
                    elif cc.plant_parameters[cc.cname]["component_category"] == "storage":
                        measurements_dict[cc.cname]["power_limit_upper"] = curtailed_power

                    # Run controller step again for curtailed component
                    # Two step process to remove uncurtailable variable from compute controls output
                    computed_controls = cc.compute_controls(measurements_dict)
                    for key in keys_to_remove:
                        if key in computed_controls[cc.cname]:
                            curtailment_status[cc.cname] = \
                            computed_controls[cc.cname].pop(key, None)

                    controls_dict.update(computed_controls)

                    power_export_total -= (current_power -
                                           controls_dict[cc.cname]["power_setpoint"])

                    if power_export_total <= self._interconnect_limit:
                        break

        return controls_dict
