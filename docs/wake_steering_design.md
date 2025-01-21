# Wake steering design

The `whoc.design_tools.wake_steering_design` module provides various tools for the design of yaw
offset lookup tables for "open-loop" wake steering. The two primary functions are `build_simple_wake_steering_lookup_table` and `build_uncertain_wake_steering_lookup_table`, both of
which take an instantiated
[`FlorisModel`](https://nrel.github.io/floris/_autosummary/floris.floris_model.html),
along with various design parameters, and return a pandas DataFrame `df_opt` containing the optimal
yaw offset angles for each wind turbine. Under the hood, both functions run an optimization using
FLORIS'
[`YawOptimizerSR`](https://nrel.github.io/floris/_autosummary/floris.optimization.yaw_optimization.yaw_optimizer_sr.html) methodology. The `uncertain` version takes into account wind direction
uncertainty via the second required argument `wd_std`, representing the wind direction standard
deviation.

The output DataFrame of optimal yaw angles `df_opt` can then be passed to the
{ref}`controllers_luwakesteer`
upon its instantiation.

___

As well as these primary functions, several other functions are provided that modify the
FLORIS-produced optimal offsets for various practical purposes, as described below.

### Geometric ramping with wind speed

Generally speaking, while the dependency of optimal offsets on wind direction is complex,
the dependency on wind speed is simpler. The `apply_wind_speed_ramps()` function allows users to
define a simple heuristic relationship of the optimal offsets on wind speeds, given a set of
yaw angles optimized for a single "middle" wind speed. Offsets are linearly "ramped up" between
`ws_wake_steering_cut_in` and `ws_wake_steering_fully_engaged_low`; take the optimized value between
`ws_wake_steering_fully_engaged_low` and `ws_wake_steering_fully_engaged_high`; and are linearly
ramped down again between `ws_wake_steering_fully_engaged_high` and `ws_wake_steering_cut_out`.
Additionally, the DataFrame produced pads with zero offsets between `ws_min` and
`ws_wake_steering_cut_in` and between `ws_wake_steering_cut_out` and `ws_max`.

### Yaw offset rate limits

To limit the rate of change of the yaw offset as a function of wind direction, wind speed, or 
turbulence intensity, use the `apply_static_rate_limits()` function. This will "smooth" the yaw
offsets to avoid high sensitivity to changes in the lookup table inputs (most obviously, to
wind direction near the perfectly aligned directions, where optimal offsets often change from
being at largest negative value to being at their largest positive value over a small wind
direction window). Resulting offsets, however, will be suboptimal and produce significant
performance reduction.

___

Two other wake steering-based utilities are provided in the `whoc.design_tools.wake_steering_design` module.

### Finding hysteresis zones

If excessive yaw maneuvers are of concern, an alternative to
[rate limiting](#yaw-offset-rate-limits)
the optimal offsets is to apply a hysteresis approach around the transition wind directions. This
holds the yaw offset at a fixed value across the transition between negative and positive values,
until a certain "large-enough" change in wind direction occurs. This prevents chattering between
large positive and negative values. The `compute_hysteresis_zones()` function takes in a DataFrame
of optimal yaw angles, and returns a list of wind direction bands where hysteresis should be applied
based on transitions exceeding the `yaw_rate_threshold`. The list of hysteresis zones can be
provided to the {ref}`controllers_luwakesteer`
on instantiation, along with `df_opt`, to apply
dynamic hysteresis to the yaw offsets. However, again, care should be taken with this approach as it
can result in "wrong-way steering".

### Converting yaw offsets DataFrame to an interpolant function

In many cases, including in the
{ref}`controllers_luwakesteer`,
it is more convenient for the precomputed yaw offsets lookup table to be defined as a an
interpolant than a DataFrame of optimal values. The `get_yaw_angles_interpolant` function takes in
a DataFrame `df_opt` and returns a function that can be queried at any
wind direction, wind speed, turbulence intensity combination to provide an interpolated set of
offsets. Additionally, the wind direction, wind speed, and turbulence intensity can be queried
using arrays of equal length to interpolate in a vectorized manner.

Note that in {ref}`controllers_luwakesteer`,
the construction of the interpolator happens automatically based on the `df_opt` passed in on
instantiation.

___

### Wake steering offset visualization

Visualization tools for wake steering lookup tables are provided in the 
`whoc.design_tools.wake_steering_visualization` module. There are currently two functions:

- `plot_offsets_wswd_heatmap()` creates a heatmap of offsets by wind speed and wind direction based
on `df_opt` for a given turbine index `turb_id`.
- `plot_offsets_wd()` plots the offsets from `df_opt` for turbine `turb_id` at a specified (set of)
wind speeds `wd_plot`. 

Both functions, as well as many of the design functions described here, are demonstrated in the
compare_yaw_offset_designs.py python script provided in {ref}`examples_luwakesteer`.

