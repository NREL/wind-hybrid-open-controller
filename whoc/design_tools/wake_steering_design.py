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

    wind_directions = np.arange(0, 360, wd_resolution)
    wind_rose = WindRose(
        wind_speeds=np.array([8.0]),
        wind_directions=wind_directions,
        ti_table=0.06,
    )

    fmodel.set(wind_data=wind_rose)

    yaw_opt = YawOptimizationSR(
        fmodel=fmodel,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    return yaw_opt.optimize()