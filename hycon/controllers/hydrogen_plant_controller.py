import copy

from hycon.controllers.controller_base import ControllerBase


class HydrogenPlantController(ControllerBase):
    def __init__(
        self,
        interface,
        cname="hydrogen",
        controller_parameters={},
        verbose=False,
    ):
        """
        Constructor for HydrogenPlantController.

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            cname (str): Name of the controller. Defaults to "hydrogen".
            controller_parameters (dict): Dictionary of controller parameters. See
                set_controller_parameters for details.
            verbose (bool): Verbosity flag. Defaults to False.
        """
        super().__init__(interface, cname=cname, verbose=verbose)

        # Check that parameters are not specified both in input file
        # and in controller_parameters
        self.check_controller_parameters(controller_parameters)
        self.set_controller_parameters(**controller_parameters)

        # Initialize filter
        self.filtered_power_prev = 0

    def set_controller_parameters(
        self,
        nominal_plant_power_kW,
        nominal_hydrogen_rate_kgps,
        generator_controller,
        hydrogen_controller_gain=1.0,
        **_,  # <- Allows arbitrary additional parameters to be passed, which are ignored
    ):
        """
        Set gains and threshold limits for HydrogenPlantController.

        nominal_plant_power_kW and nominal_hydrogen_rate_kgps are the nominal power of the
        power generation plant in kW and the nominal hydrogen production rate of the electrolyzer
        in kg/s, respectively. These are used to scale the control action.

        hydrogen_controller_gain is a gain applied to the difference between the hydrogen reference
        and hydrogen production rate to adjust the responsiveness of the controller.

        Args:
            nominal_plant_power_kW (float): Nominal power of the plant in kW.
            nominal_hydrogen_rate_kgps (float): Nominal hydrogen production rate in kg/s.
            generator_controller (ControllerBase): Controller for the generator. This controller
                should accept a power reference as an input and output appropriate generator
                controls.
            hydrogen_controller_gain (float): Gain for the hydrogen controller. Defaults to 1.0.
        """

        # Assign the power component controller
        self.generator_controller = generator_controller

        # Set K from plant inputs
        self.K = nominal_plant_power_kW / nominal_hydrogen_rate_kgps * hydrogen_controller_gain

    def compute_controls(self, measurements_dict):
        # Run supervisory control logic
        power_reference = self.supervisory_control(measurements_dict)

        # Package the controls for the individual controllers, step, and return
        if self.generator_controller:
            # Create exhaustive generator measurements dict to handle variety
            # of possible lower-level controllers
            generator_measurements_dict = copy.deepcopy(measurements_dict)
            generator_measurements_dict["power_reference"] = power_reference
            # Remove any external power reference
            if "plant_power_reference" in generator_measurements_dict:
                del generator_measurements_dict["plant_power_reference"]

            # Compute controls for generator
            generator_controls_dict = self.generator_controller.compute_controls(
                generator_measurements_dict
            )

            # Clean up returned controls
            if "yaw_angles" in generator_controls_dict:
                del generator_controls_dict["yaw_angles"]
            if "power_setpoints" in generator_controls_dict:
                generator_controls_dict["wind_power_setpoints"] = generator_controls_dict[
                    "power_setpoints"
                ]
                del generator_controls_dict["power_setpoints"]

        return generator_controls_dict

    def supervisory_control(self, measurements_dict):
        # Extract measurements sent
        current_power = measurements_dict["local_power"]
        hydrogen_output = measurements_dict[self.cname]["production_rate"]
        hydrogen_reference = measurements_dict[self.cname]["hydrogen_production_reference"]

        # Input filtering
        a = 0.05
        filtered_power = (1 - a / self.dt) * self.filtered_power_prev + a / self.dt * current_power

        # Calculate difference between hydrogen reference and hydrogen actual
        hydrogen_error = hydrogen_reference - hydrogen_output

        # Apply gain to generator power output
        power_reference = filtered_power + self.K * hydrogen_error

        if power_reference < 0:
            power_reference = 0

        self.filtered_power_prev = filtered_power

        return power_reference
