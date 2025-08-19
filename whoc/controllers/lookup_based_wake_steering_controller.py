from __future__ import annotations

import numpy as np
import pandas as pd
from floris.utilities import wrap_180

from whoc.controllers.controller_base import ControllerBase
from whoc.design_tools.wake_steering_design import get_yaw_angles_interpolant
from whoc.interfaces.interface_base import InterfaceBase


class LookupBasedWakeSteeringController(ControllerBase):
    def __init__(
            self,
            interface: InterfaceBase,
            input_dict: dict,
            df_yaw: pd.DataFrame | None = None,
            hysteresis_dict: dict | None = None,
            verbose: bool = False
        ):
        """
        Constructor for LookupBasedWakeSteeringController.

        Args:
            interface (InterfaceBase): Interface object for communicating with the plant.
            input_dict (dict): Dictionary of input parameters.
            df_yaw (pd.DataFrame): DataFrame of yaw offsets. May be produced using tools in 
                whoc.design_tools.wake_steering_design. Defaults to None.
            hysteresis_dict (dict): Dictionary of hysteresis zones. May be produced using
                compute_hysteresis_zones function in whoc.design_tools.wake_steering_design.
                Defaults to None.
            verbose (bool): Verbosity flag.
        """
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Handle yaw optimizer object
        if df_yaw is None:
            if hysteresis_dict is not None:
                raise ValueError(
                    "Hysteresis zones provided without yaw offsets. "
                    "Please provide yaw offsets."
                )
            if self.verbose:
                print("No offsets received; assuming nominal aligned control.")
            self.wake_steering_interpolant = None
        else:
            self.wake_steering_interpolant = get_yaw_angles_interpolant(df_yaw)

        if isinstance(hysteresis_dict, dict) and len(hysteresis_dict) == 0:
            print((
                "Received empty hysteresis dictionary. Assuming no hysteresis."
                "This may happen if yaw offsets are one-sided."
            ))
            hysteresis_dict = None

        self.hysteresis_dict = hysteresis_dict

        # Set initial conditions
        yaw_IC = input_dict["controller"]["initial_conditions"]["yaw"]
        if hasattr(yaw_IC, "__len__"):
            if len(yaw_IC) == self.n_turbines:
                self.controls_dict = {"yaw_angles": yaw_IC}
            else:
                raise TypeError(
                    "yaw initial condition should be a float or "
                    + "a list of floats of length num_turbines."
                )
        else:
            self.controls_dict = {"yaw_angles": [yaw_IC] * self.n_turbines}

        # For startup
        self.wd_store = [270.]*self.n_turbines # TODO: update this?
        self.yaw_store = yaw_IC


    def compute_controls(self, measurements_dict):
        return self.wake_steering_angles(measurements_dict["wind_directions"])

    def wake_steering_angles(self, wind_directions):

        # Handle possible bad data
        wind_speeds = [8.0]*self.n_turbines # TODO: enable extraction of wind speed in Hercules
        if not wind_directions: # Received empty or None
            if self.verbose:
                print("Bad wind direction measurement received, reverting to previous measurement.")
            wind_directions = self.wd_store
        else:
            self.wd_store = wind_directions

        # Look up wind direction
        if self.wake_steering_interpolant is None:
            yaw_setpoint = wind_directions
        else:
            interpolated_angles = self.wake_steering_interpolant(
                wind_directions,
                wind_speeds,
                None
            )
            yaw_offsets = np.diag(interpolated_angles)
            yaw_setpoint = (np.array(wind_directions) - yaw_offsets).tolist()

        # Apply hysteresis
        if self.hysteresis_dict is not None:
            for t in range(self.n_turbines):
                for zone in self.hysteresis_dict["T{:03d}".format(t)]:
                    if (
                        (zone[0] < wind_directions[t] < zone[1])
                        or (wrap_180(zone[0]) < wrap_180(wind_directions[t]) < wrap_180(zone[1]))
                        ):
                        # In hysteresis zone, overwrite yaw angle with previous setpoint
                        yaw_setpoint[t] = self.yaw_store[t]

        self.yaw_store = yaw_setpoint

        self.controls_dict = {"yaw_angles": yaw_setpoint}

        return {"yaw_angles": yaw_setpoint}


class YawSetpointPassthroughController(ControllerBase):
    """
    YawSetpointPassthroughController is a simple controller that passes through wind directions
    as yaw setpoints without modification.
    """
    def __init__(self, interface: InterfaceBase, verbose: bool = False):
        super().__init__(interface, verbose=verbose)

    def compute_controls(self, measurements_dict):
        # Simply pass through the yaw setpoints as the received wind directions
        return {"yaw_angles": measurements_dict["wind_directions"]}
