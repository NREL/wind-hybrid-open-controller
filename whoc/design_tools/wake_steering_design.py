import numpy as np
import pandas as pd
from floris import FlorisModel, UncertainFlorisModel, WindRose
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from scipy.interpolate import interp1d, LinearNDInterpolator

# TODO
# DONE basic wake steering design tool, will require an instantiated FlorisModel as input

# DONE wake steering designer that uses uncertainty (?)

# TODO Slopes across wind speeds (how to do?)

# TODO Interpolator (from FLASC)

# TODO Spline approximation

# TODO Dynamic controller that uses hysteresis

# DONE Maximum slope constraint approach

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
    ws_rate_limit: float = 10,
    ti_rate_limit: float = 1
):
    """
    Apply static rate limits to a yaw offset lookup table.

    Args:
        df_opt (pd.DataFrame): A yaw offset lookup table.
        wd_rate_limit (float, optional): The maximum rate of change in yaw offset per degree change
            in wind direction [deg / deg]. Defaults to 5.
        ws_rate_limit (float, optional): The maximum rate of change in yaw offset per change in
            wind speed [deg / m/s]. Defaults to 10.
        ti_rate_limit (float, optional): The maximum rate of change in yaw offset per change in
            turbulence intensity [deg / -]. Defaults to 1.
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

    # Apply ti rate limits (increasing ti)
    for k in range(1, len(ti_array)):
        delta_yaw = offsets_array[:, :, :, k] - offsets_array[:, :, :, k-1]
        delta_yaw = np.clip(delta_yaw, -ti_rate_limit, ti_rate_limit)
        offsets_array[:, :, :, k] = offsets_array[:, :, :, k-1] + delta_yaw

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

def compute_hysteresis_zones(
    df_opt: pd.DataFrame,
):
    
    return 0

def apply_wind_speed_ramps(
    df_opt: pd.DataFrame,
    ws_resolution: float = 1,
    ws_min: float = 0,
    ws_max: float = 30,
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
    if (ws_wake_steering_cut_in
        <= ws_wake_steering_fully_engaged_low
        <= ws_wake_steering_fully_engaged_high
        <= ws_wake_steering_cut_out):
        pass
    else:
        raise ValueError(
            "Wind speed ramp values must be in the order: cut in, fully engaged low, "
            "fully engaged high, cut out."
        )

    # Check if there is more than one wind speed specified
    if len(df_opt["wind_speed"].unique()) > 1:
        raise ValueError(
            "Wind speed ramps can only be applied to a dataframe with a single wind speed."
        )
    else:
        ws_specified = df_opt["wind_speed"].unique()

    # Check that provided wind speed is between the fully engaged limits
    if (ws_specified < ws_wake_steering_fully_engaged_low
        or ws_specified > ws_wake_steering_fully_engaged_high):
        raise ValueError(
            "Provided wind speed must be between fully engaged limits."
        )

    offsets_specified = np.vstack(df_opt.yaw_angles_opt.to_numpy())[None,:,:]

    # Pack offsets with zero values at the cut in, start, finish, and cut out wind speeds
    offsets_ramps = np.concatenate(
        (
            np.zeros_like(offsets_specified),
            np.zeros_like(offsets_specified),
            offsets_specified,
            offsets_specified,
            np.zeros_like(offsets_specified),
            np.zeros_like(offsets_specified)
        ),
        axis=0
    )
    wind_speed_ramps = np.array([
        ws_min,
        ws_wake_steering_cut_in,
        ws_wake_steering_fully_engaged_low,
        ws_wake_steering_fully_engaged_high,
        ws_wake_steering_cut_out,
        ws_max
    ])

    # Build interpolator and interpolate to desired wind speeds
    interp = interp1d(
        wind_speed_ramps,
        offsets_ramps,
        axis=0,
        bounds_error=False,
        fill_value=np.zeros_like(offsets_ramps[0,:,:])
    )
    wind_speed_all = np.arange(ws_min, ws_max, ws_resolution)
    offsets_stacked = interp(wind_speed_all).reshape(-1, offsets_ramps.shape[2])

    wind_direction_stacked = np.tile(df_opt.wind_direction, len(wind_speed_all))
    wind_speed_stacked = np.repeat(wind_speed_all, len(df_opt))
    turbulence_intensity_stacked = np.tile(df_opt.turbulence_intensity, len(wind_speed_all))

    return pd.DataFrame({
        "wind_direction": wind_direction_stacked,
        "wind_speed": wind_speed_stacked,
        "turbulence_intensity": turbulence_intensity_stacked,
        "yaw_angles_opt": [offsets_stacked[i,:] for i in range(offsets_stacked.shape[0])]
    })

def get_yaw_angles_interpolant(df_opt):
    """Get an interpolant for the optimal yaw angles from a dataframe.

    Create an interpolant for the optimal yaw angles from a dataframe
    'df_opt', which contains the rows 'wind_direction', 'wind_speed',
    'turbulence_intensity', and 'yaw_angles_opt'. This dataframe is typically
    produced automatically from a FLORIS yaw optimization using Serial Refine
    or SciPy. One can additionally apply a ramp-up and ramp-down region
    to transition between non-wake-steering and wake-steering operation.

    Note that, in contrast to the previous implementation in FLASC, this implementation
    does not allow ramp_up_ws, ramp_down_ws, minimum_yaw_angle, or maximum_yaw_angle.
    Minimum and maximum yaw angles should be specified during the design optimization,
    while ramping with wind speed is now handled by apply_wind_speed_ramps.

    Args:
        df_opt (pd.DataFrame): Dataframe containing the rows 'wind_direction',
            'wind_speed', 'turbulence_intensity', and 'yaw_angles_opt'.

    Returns:
        LinearNDInterpolator: An interpolant function which takes the inputs
            (wind_directions, wind_speeds, turbulence_intensities), all of equal
            dimensions, and returns the yaw angles for all turbines. This function
            incorporates the ramp-up and ramp-down regions.
    """

    # Load data and set up a linear interpolant
    points = df_opt[["wind_direction", "wind_speed", "turbulence_intensity"]]
    values = np.vstack(df_opt["yaw_angles_opt"])

    # Expand wind direction range to cover 0 deg to 360 deg
    points_copied = points[points["wind_direction"] == 0.0].copy()
    points_copied.loc[points_copied.index, "wind_direction"] = 360.0
    values_copied = values[points["wind_direction"] == 0.0, :]
    points = np.vstack([points, points_copied])
    values = np.vstack([values, values_copied])

    # Copy lowest wind speed / TI solutions to -1.0 to create lower bound
    for col in [1, 2]:
        ids_to_copy_lb = points[:, col] == np.min(points[:, col])
        points_copied = np.array(points[ids_to_copy_lb, :], copy=True)
        values_copied = np.array(values[ids_to_copy_lb, :], copy=True)
        points_copied[:, col] = -1.0  # Lower bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

        # Copy highest wind speed / TI solutions to 999.0
        ids_to_copy_ub = points[:, col] == np.max(points[:, col])
        points_copied = np.array(points[ids_to_copy_ub, :], copy=True)
        values_copied = np.array(values[ids_to_copy_ub, :], copy=True)
        points_copied[:, col] = 999.0  # Upper bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

    # Now create a linear interpolant for the yaw angles
    interpolant = LinearNDInterpolator(points=points, values=values, fill_value=np.nan)

    # Now create a wrapper function with ramp-up and ramp-down
    def yaw_angle_interpolant(wd_array, ws_array, ti_array=None):
        # Deal with missing ti_array
        if ti_array is None:
            ti_ref = float(np.median(interpolant.points[:, 2]))
            ti_array = np.ones(np.shape(wd_array), dtype=float) * ti_ref

        # Format inputs
        wd_array = np.array(wd_array, dtype=float)
        ws_array = np.array(ws_array, dtype=float)
        ti_array = np.array(ti_array, dtype=float)
        return np.array(interpolant(wd_array, ws_array, ti_array), dtype=float)

    return yaw_angle_interpolant

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