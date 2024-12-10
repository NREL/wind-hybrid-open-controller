import numpy as np
import pandas as pd
from floris import FlorisModel, UncertainFlorisModel, WindRose
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
        wd_min (float, optional): The minimum wind direction in degrees. Defaults to 0.
        wd_max (float, optional): The maximum wind direction in degrees. Defaults to 360.
        ws_resolution (float, optional): The resolution of the wind speed in m/s.
            Defaults to 1.
        ws_min (float, optional): The minimum wind speed in m/s. Defaults to 8.
        ws_max (float, optional): The maximum wind speed in m/s. Defaults to 8.
        ti (float, optional): The turbulence intensity for wake steering design. Defaults to 0.06.
        minimum_yaw_angle (float, optional): The minimum allowable misalignment in degrees.
            Defaults to 0.0.
        maximum_yaw_angle (float, optional): The maximum allowable misalignment in degrees.
            Defaults to 25.0.
    """
    wind_rose = create_uniform_wind_rose(
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        ti=ti,
    )

    fmodel.set(wind_data=wind_rose)

    yaw_opt = YawOptimizationSR(
        fmodel=fmodel,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    return yaw_opt.optimize()

def build_uncertain_wake_steering_lookup_table(
    fmodel: FlorisModel,
    wd_std: float,
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
    method, with uncertainty in the wind direction.

    Args:
        fm (FlorisModel): An instantiated FlorisModel object.
        wd_resolution (float, optional): The resolution of the wind direction in degrees.
            Defaults to 5.
        wd_min (float, optional): The minimum wind direction in degrees. Defaults to 0.
        wd_max (float, optional): The maximum wind direction in degrees. Defaults to 360.
        ws_resolution (float, optional): The resolution of the wind speed in m/s.
            Defaults to 1.
        ws_min (float, optional): The minimum wind speed in m/s. Defaults to 8.
        ws_max (float, optional): The maximum wind speed in m/s. Defaults to 8.
        ti (float, optional): The turbulence intensity for wake steering design. Defaults to 0.06.
        minimum_yaw_angle (float, optional): The minimum allowable misalignment in degrees.
            Defaults to 0.0.
        maximum_yaw_angle (float, optional): The maximum allowable misalignment in degrees.
            Defaults to 25.0.
    """
    wind_rose = create_uniform_wind_rose(
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        ti=ti,
    )

    fmodel.set(wind_data=wind_rose)

    ufmodel = UncertainFlorisModel(
        configuration=fmodel.core.as_dict(),
        wd_std=wd_std,
    )

    yaw_opt = YawOptimizationSR(
        fmodel=ufmodel,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    return yaw_opt.optimize()

def apply_static_rate_limits(
    df_opt: pd.DataFrame,
    wd_rate_limit: float = 5,
    ws_rate_limit: float = 10
):
    """
    Apply static rate limits to a yaw offset lookup table.

    Args:
        df_opt (pd.DataFrame): A yaw offset lookup table.
        wd_rate_limit (float, optional): The maximum rate of change in yaw offset per degree change
            in wind direction [deg / deg]. Defaults to 5.
        ws_rate_limit (float, optional): The maximum rate of change in yaw offset per change in
            wind speed [deg / m/s]. Defaults to 10.
    """

    offsets_all = np.vstack(df_opt.yaw_angles_opt.to_numpy()).transpose()

    wd_array = np.unique(df_opt.wind_direction)
    ws_array = np.unique(df_opt.wind_speed)
    ti_array = np.unique(df_opt.turbulence_intensity)

    wd_step = wd_array[1] - wd_array[0]
    ws_step = ws_array[1] - ws_array[0]

    # 4D array, with dimensions: (turbine, wd, ws, ti)
    # TODO: will this ordering always work? Or not?
    offsets_array = offsets_all.reshape(
        (offsets_all.shape[0], len(wd_array), len(ws_array), len(ti_array))
    )

    # Apply wd rate limits
    offsets_limited_lr = offsets_array.copy()
    for i in range(1, len(wd_array)):
        delta_yaw = offsets_limited_lr[:, i, :, :] - offsets_limited_lr[:, i-1, :, :]
        delta_yaw = np.clip(delta_yaw, -wd_rate_limit*wd_step, wd_rate_limit*wd_step)
        offsets_limited_lr[:, i, :, :] = offsets_limited_lr[:, i-1, :, :] + delta_yaw
    offsets_limited_rl = offsets_array.copy()
    for i in range(len(wd_array)-2, -1, -1):
        delta_yaw = offsets_limited_rl[:, i, :, :] - offsets_limited_rl[:, i+1, :, :]
        delta_yaw = np.clip(delta_yaw, -wd_rate_limit*wd_step, wd_rate_limit*wd_step)
        offsets_limited_rl[:, i, :, :] = offsets_limited_rl[:, i+1, :, :] + delta_yaw
    offsets_array = (offsets_limited_lr + offsets_limited_rl) / 2

    # Apply ws rate limits (increasing ws)
    for j in range(1, len(ws_array)):
        delta_yaw = offsets_array[:, :, j, :] - offsets_array[:, :, j-1, :]
        delta_yaw = np.clip(delta_yaw, -ws_rate_limit*ws_step, ws_rate_limit*ws_step)
        offsets_array[:, :, j, :] = offsets_array[:, :, j-1, :] + delta_yaw

    # Flatten array back into 2D array for dataframe
    offsets_shape = offsets_array.shape
    offsets_all_limited = offsets_array.reshape(
        (offsets_shape[0], offsets_shape[1]*offsets_shape[2]*offsets_shape[3])
    ).transpose()
    df_opt_rate_limited = df_opt.copy()
    df_opt_rate_limited["yaw_angles_opt"] = [*offsets_all_limited]

    return df_opt_rate_limited

def create_linear_spline_approximation(
        df_opt: pd.DataFrame,
):
    """
    Create a linear spline approximation to a yaw offset lookup table.

    Args:
        df_opt (pd.DataFrame): A yaw offset lookup table.
    """

    df_opt_spline = df_opt.copy()

    return df_opt_spline

def compute_hysterisis_zones(
    df_opt: pd.DataFrame,
):
    
    return 0

def apply_wind_speed_ramps(
    df_opt: pd.DataFrame,
    ws_wake_steering_cut_in: float = 3,
    ws_wake_steering_fully_engaged_low: float = 5,
    ws_wake_steering_fully_engaged_high: float = 10,
    ws_wake_steering_cut_out: float = 13,
):
    """
    Apply wind speed ramps to a yaw offset lookup table.

    Args:
        df_opt (pd.DataFrame): A yaw offset lookup table.
    """

    # Check valid ordering of wind speeds
    if ws_wake_steering_cut_in <= ws_wake_steering_fully_engaged_low <= ws_wake_steering_fully_engaged_high <= ws_wake_steering_cut_out:
        pass
    else:
        raise ValueError("Wind speed ramp values must be in the order: cut in, fully engaged low, fully engaged high, cut out.")

    # Check if there is more than one wind speed specified
    if len(df_opt["wind_speed"].unique()) > 1:
        raise ValueError("Wind speed ramps can only be applied to a dataframe with a single wind speed.")
    
    # Check that all wind speeds are between the fully engaged limits
    if (df_opt["wind_speed"].unique() < ws_wake_steering_fully_engaged_low).any():
        raise ValueError("All wind speeds must be greater than or equal to the lower fully engaged wind speed.")
    if (df_opt["wind_speed"].unique() > ws_wake_steering_fully_engaged_high).any():
        raise ValueError("All wind speeds must be less than or equal to the higher fully engaged wind speed.")

    df_opt_ramped = df_opt.copy()

def create_uniform_wind_rose(
    wd_resolution: float = 5,
    wd_min: float = 0,
    wd_max: float = 360,
    ws_resolution: float = 1,
    ws_min: float = 8,
    ws_max: float = 8,
    ti: float = 0.06,
):

    if wd_min == 0 and wd_max == 360:
        wd_max = wd_max - wd_resolution
    wind_directions = np.arange(wd_min, wd_max+0.001, wd_resolution)
    
    wind_speeds = np.arange(ws_min, ws_max+0.001, ws_resolution)

    return WindRose(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        ti_table=ti,
    )