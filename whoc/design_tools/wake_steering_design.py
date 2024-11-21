import numpy as np
from floris import FlorisModel, WindRose
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

# TODO
# basic wake steering design tool, will require an instantiated FlorisModel as input

# wake steering designer that uses uncertainty (?)

# Slopes across wind speeds (how to do?)

# Spline approximation

# Maximum slope constraint approach

def build_simple_wake_steering_lookup_table(
        fmodel: FlorisModel,
        wd_resolution: float = 5,
        wd_min: float = 0,
        wd_max: float = 360,
        ws_resolution: float = 1,
        ws_min: float = 8,
        ws_max: float = 8,
        ti: float = 0.06,
        minimum_yaw_angle: float = 0.0,
        maximum_yaw_angle: float = 25.0,
    ):
    """
    Build a simple wake steering lookup table for a given FlorisModel using the Serial Refine
    method.

    Args:
        fm (FlorisModel): An instantiated FlorisModel object.
        wd_resolution (float, optional): The resolution of the wind direction in degrees.
            Defaults to 5.

    TODO:
    - single wind speed or all wind speeds? 
    - uncertainty?
    - wd resolution?
    """
    if wd_min == 0 and wd_max == 360:
        wd_max = wd_max - wd_resolution
    wind_directions = np.arange(wd_min, wd_max+wd_resolution, wd_resolution)
    
    wind_speeds = np.arange(ws_min, ws_max+ws_resolution, ws_resolution)

    wind_rose = WindRose(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        ti_table=ti,
    )

    fmodel.set(wind_data=wind_rose)

    yaw_opt = YawOptimizationSR(
        fmodel=fmodel,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    return yaw_opt.optimize()