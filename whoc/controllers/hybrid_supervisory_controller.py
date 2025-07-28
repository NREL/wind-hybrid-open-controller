import numpy as np

from whoc.controllers.controller_base import ControllerBase


class HybridSupervisoryControllerBaseline(ControllerBase):
    def __init__(
            self,
            interface,
            input_dict,
            wind_controller=None,
            solar_controller=None,
            battery_controller=None,
            verbose=False
        ):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have

        # Assign the individual asset controllers
        self.wind_controller = wind_controller
        self.solar_controller = solar_controller
        self.battery_controller = battery_controller

        self._has_solar_controller = solar_controller is not None
        self._has_wind_controller = wind_controller is not None
        self._has_battery_controller = battery_controller is not None

        # Must provide a controller for one type of generation
        if not self._has_wind_controller and not self._has_solar_controller:
            raise ValueError(
                "The HybridSupervisoryControllerBaseline requires that either a solar_controller"
                " or a wind_controller be provided."
            )

        # Set constants
        # TODO: Remove this direct py_sims dependence, shouldn't be needed.
        # This will break the code and need to be fixed shortly.
        py_sims = input_dict["py_sim_names"]
        if self.battery_controller:
            battery_name = [ps for ps in py_sims if "battery" in ps][0]
            self.battery_charge_rate = input_dict[battery_name]["charge_rate"]*1000
        else:
            self.battery_charge_rate = 0
        # Change battery charge rate to kW

        # Initialize Power references
        self.wind_reference = 0
        self.solar_reference = 0
        self.battery_reference = 0

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
            wind_measurements_dict = {
                "power_reference": wind_reference,
                "wind_turbine_powers": measurements_dict["wind_turbine_powers"]
            }
            wind_controls_dict = self.wind_controller.compute_controls(wind_measurements_dict)
            controls_dict["wind_power_setpoints"] = wind_controls_dict["wind_power_setpoints"]
        if self._has_solar_controller:
            solar_measurements_dict = {"power_reference": solar_reference}
            solar_controls_dict = self.solar_controller.compute_controls(solar_measurements_dict)
            controls_dict["solar_power_setpoint"] = solar_controls_dict["power_setpoint"]
        if self._has_battery_controller:
            battery_measurements_dict = {
                "time": measurements_dict["time"],
                "power_reference": battery_reference,
                "battery_power": measurements_dict["battery_power"],
                "battery_soc": measurements_dict["battery_soc"]
            }
            battery_controls_dict = self.battery_controller.compute_controls(
                battery_measurements_dict
            )
            controls_dict["battery_power_setpoint"] = battery_controls_dict["power_setpoint"]

        return controls_dict

    def supervisory_control(self, measurements_dict):
        # Extract measurements sent
        time = measurements_dict["time"] # noqa: F841 
        if self._has_wind_controller:
            wind_power = np.array(measurements_dict["wind_turbine_powers"]).sum()
            wind_speed = measurements_dict["wind_speed"] # noqa: F841
        else:
            wind_power = 0
            wind_speed = 0 # noqa: F841

        if self._has_solar_controller:
            solar_power = measurements_dict["solar_power"]
            solar_dni = measurements_dict["solar_dni"] # direct normal irradiance # noqa: F841
            solar_aoi = measurements_dict["solar_aoi"] # angle of incidence # noqa: F841
        else:
            solar_power = 0
            solar_dni = 0 # noqa: F841
            solar_aoi = 0 # noqa: F841

        if self._has_battery_controller:
            battery_power = measurements_dict["battery_power"]
            battery_soc = measurements_dict["battery_soc"]
        else:
            battery_power = 0
            battery_soc = 0

        plant_power_reference = measurements_dict["power_reference"]

        # Filter the wind and solar power measurements to reduce noise and improve closed-loop
        # controller damping
        a = 0.1
        wind_power = (1-a)*self.prev_wind_power + a*wind_power
        solar_power = (1-a)*self.prev_solar_power + a*solar_power

        # Temporary print statements (note that negative battery indicates discharging)
        print("Measured powers (wind, solar, battery):", wind_power, solar_power, battery_power)
        print("Reference power:", plant_power_reference)

        # Calculate battery reference value
        if self._has_battery_controller:
            battery_reference = plant_power_reference - (wind_power + solar_power)
        else:
            battery_reference = 0

        # Decide control gain:
        if (wind_power + solar_power) < (plant_power_reference+self.battery_charge_rate)\
            and battery_power <= 0:
            if battery_soc>0.89:
                K = ((wind_power + solar_power) - plant_power_reference) / 2
            else:
                K = ((wind_power+solar_power) - (plant_power_reference+self.battery_charge_rate))/2
        else:
            K = ((wind_power + solar_power) - plant_power_reference) / 2

        if not (self._has_wind_controller & self._has_solar_controller):
            # Only one type of generation available, double the control gain
            K = 2*K

        if (wind_power + solar_power) > (plant_power_reference+self.battery_charge_rate) or \
            ((wind_power + solar_power) > (plant_power_reference) and battery_soc>0.89):
            
            # go down
            wind_reference = wind_power - K
            solar_reference = solar_power - K
        else: 
            # go up
            # Is the resource saturated?
            if self.solar_reference > (self.prev_solar_power+0.05*self.solar_reference):
                solar_reference = self.solar_reference
            else:
                # If not, ask for more power
                solar_reference = solar_power - K

            if self.wind_reference > (self.prev_wind_power+0.05*self.wind_reference):
                wind_reference = self.wind_reference
            else:
                wind_reference = wind_power - K

        # Reset references for invalid controllers
        if not self._has_wind_controller:
            wind_reference = 0
        if not self._has_solar_controller:
            solar_reference = 0

        print(
            "Power reference values (wind, solar, battery)",
            wind_reference, solar_reference, battery_reference
        )

        self.prev_solar_power = solar_power
        self.prev_wind_power = wind_power
        self.wind_reference = wind_reference
        self.solar_reference = solar_reference
        self.battery_reference = battery_reference

        return wind_reference, solar_reference, battery_reference


class HybridSupervisoryControllerMultiRef(HybridSupervisoryControllerBaseline):
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
            verbose=False
        ):
        super().__init__(
            interface=interface,
            input_dict=input_dict,
            wind_controller=wind_controller,
            solar_controller=solar_controller,
            battery_controller=battery_controller,
            verbose=verbose
        )

        # Extract interconnection limit
        if "interconnect_limit" in self.plant_parameters:
            self.interconnect_limit = self.plant_parameters["interconnect_limit"]
            if self.interconnect_limit <= 0:
                raise ValueError("interconnect_limit must be positive.")
        else:
            raise KeyError("interconnect_limit must be specified to use this controller.")

        if "curtailment_order" in self.controller_parameters:
            self.curtailment_order = self.controller_parameters["curtailment_order"]
        else:
            self.curtailment_order = ["solar", "wind"]

    def supervisory_control(self, measurements_dict):
        """
        Overwrite HybridSupervisoryControllerBaseline.supervisory_control()
        with controller that follows separate setpoints and curtails in order.
        """

        # Extract measurements sent
        if self._has_wind_controller:
            wind_power = np.array(measurements_dict["wind_turbine_powers"]).sum()
            wind_reference = measurements_dict["wind_power_reference"]

        if self._has_solar_controller:
            solar_power = measurements_dict["solar_power"]
            solar_reference = measurements_dict["solar_power_reference"]

        if self._has_battery_controller:
            raise NotImplementedError("Logic for battery component not yet added.")

        # Filter the wind and solar power measurements to reduce noise and improve closed-loop
        # controller damping
        #TODO RECONSIDER THIS MAYBE MAKE MORE DEPENDENT ON THE TIME STEP
        a = 1.0 #0.1 # FORCE THE FILTER TO BE 100% DEPENDENT ON THE CURRENT TIME STEP
        wind_power = (1-a)*self.prev_wind_power + a*wind_power
        solar_power = (1-a)*self.prev_solar_power + a*solar_power

        # Adjust references to respect capacities and interconnection limits
        wind_reference = np.minimum(wind_reference, self.plant_parameters["wind_capacity"])
        solar_reference = np.minimum(solar_reference, self.plant_parameters["solar_capacity"])
        if self.curtailment_order[0] == "solar":
            # Give whole interconnection to wind if necessary
            wind_reference = np.minimum(wind_reference, self.interconnect_limit)
            solar_ref_temp = np.maximum(self.interconnect_limit - wind_power, 0)
            solar_reference = np.minimum(solar_reference, solar_ref_temp)
        elif self.curtailment_order[0] == "wind":
            solar_reference = np.minimum(solar_reference, self.interconnect_limit)
            wind_ref_temp = np.maximum(self.interconnect_limit - solar_power, 0)
            wind_reference = np.minimum(wind_reference, wind_ref_temp)
        else:
            raise ValueError("Invalid generation type in curtailment_order.")

        # TODO: add battery option
        battery_reference = 0

        self.prev_solar_power = solar_power
        self.prev_wind_power = wind_power
        self.wind_reference = wind_reference
        self.solar_reference = solar_reference
        self.battery_reference = battery_reference

        return wind_reference, solar_reference, battery_reference
