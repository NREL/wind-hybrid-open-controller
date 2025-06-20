from pathlib import Path

import numpy as np
import pytest
from floris import FlorisModel
from whoc.design_tools.wake_steering_design import (
    apply_static_rate_limits,
    apply_wind_speed_ramps,
    build_simple_wake_steering_lookup_table,
    build_uncertain_wake_steering_lookup_table,
    check_df_opt_ordering,
    compute_hysteresis_zones,
    consolidate_hysteresis_zones,
    create_uniform_wind_rose,
    get_yaw_angles_interpolant,
)

TEST_DATA = Path(__file__).resolve().parent
YAML_INPUT = TEST_DATA / "floris_input.yaml"

def generic_df_opt(
        wd_resolution=4.0,
        wd_min=220.0,
        wd_max=310.0,
        ws_resolution=0.5,
        ws_min=8.0,
        ws_max=10.0,
        ti_resolution=0.02,
        ti_min=0.06,
        ti_max=0.08,
        minimum_yaw_angle=-20,
        maximum_yaw_angle=20,
        wd_std=None,
        kwargs_UncertainFlorisModel = {},
    ):

    fmodel_test = FlorisModel(YAML_INPUT)

    if wd_std is None:
        return build_simple_wake_steering_lookup_table(
            fmodel_test,
            wd_resolution=wd_resolution,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_resolution=ws_resolution,
            ws_min=ws_min,
            ws_max=ws_max,
            ti_resolution=ti_resolution,
            ti_min=ti_min,
            ti_max=ti_max,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
        )
    else:
        return build_uncertain_wake_steering_lookup_table(
            fmodel_test,
            wd_std=wd_std,
            wd_resolution=wd_resolution,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_resolution=ws_resolution,
            ws_min=ws_min,
            ws_max=ws_max,
            ti_resolution=ti_resolution,
            ti_min=ti_min,
            ti_max=ti_max,
            kwargs_UncertainFlorisModel=kwargs_UncertainFlorisModel,
        )

def test_build_simple_wake_steering_lookup_table():

    # Start with the simple case
    wd_resolution = 4.0
    wd_min = 220.0
    wd_max = 310.0
    ws_resolution = 0.5
    ws_min = 8.0
    ws_max = 10.0
    ti_resolution = 0.02
    ti_min = 0.06
    ti_max = 0.08
    minimum_yaw_angle = -20
    maximum_yaw_angle = 20
    df_opt = generic_df_opt(
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        ti_resolution=ti_resolution,
        ti_min=ti_min,
        ti_max=ti_max,
    )


    df_opt = generic_df_opt()

    opt_yaw_angles = np.vstack(df_opt["yaw_angles_opt"])

    n_conditions = (
        ((ws_max-ws_min)//ws_resolution+1)
        * ((wd_max-wd_min)//wd_resolution+1)
        * ((ti_max-ti_min)//ti_resolution+1)
    )

    assert opt_yaw_angles.shape == (n_conditions, 2)
    assert (opt_yaw_angles >= minimum_yaw_angle).all()
    assert (opt_yaw_angles <= maximum_yaw_angle).all()

    # More complex case (include 360, minimum yaw greater than zero)
    wd_min = 0.0
    wd_max = 360.0
    minimum_yaw_angle = -5 # Positive numbers DO NOT WORK. FLORIS bug?
    df_opt = generic_df_opt(
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
        ti_resolution=ti_resolution,
        ti_min=ti_min,
        ti_max=ti_max,
    )

    opt_yaw_angles = np.vstack(df_opt["yaw_angles_opt"])

    n_conditions = (
        ((ws_max-ws_min)//ws_resolution+1)
        * ((wd_max-wd_min)//wd_resolution)
        * ((ti_max-ti_min)//ti_resolution+1)
    )

    assert opt_yaw_angles.shape == (n_conditions, 2)
    assert (opt_yaw_angles >= minimum_yaw_angle).all()
    assert (opt_yaw_angles <= maximum_yaw_angle).all()

    # Also check case that doesn't include 0/360
    wd_min = 2.0
    wd_max = 360.0 # Shouldn't appear in output; max should be 358.0
    df_opt = generic_df_opt(
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
    )
    assert df_opt.wind_direction.min() == wd_min
    assert df_opt.wind_direction.max() == 358.0

def test_build_uncertain_wake_steering_lookup_table():

    max_yaw_angle = 35 # To force split between basic and uncertain

    df_opt_simple = generic_df_opt(wd_std=None, maximum_yaw_angle=max_yaw_angle)
    df_opt_uncertain = generic_df_opt(wd_std=3.0, maximum_yaw_angle=max_yaw_angle)

    max_offset_simple = df_opt_simple.yaw_angles_opt.apply(lambda x: np.max(x)).max()
    max_offset_uncertain = df_opt_uncertain.yaw_angles_opt.apply(lambda x: np.max(x)).max()
    assert max_offset_simple > max_offset_uncertain

    # Check that kwargs are passed correctly (results not identical)
    df_opt_uncertain_fixed = generic_df_opt(
        wd_std=3.0,
        maximum_yaw_angle=max_yaw_angle,
        kwargs_UncertainFlorisModel={"fix_yaw_to_nominal_direction": True}
    )
    assert not np.allclose(df_opt_uncertain.farm_power_opt, df_opt_uncertain_fixed.farm_power_opt)

def test_apply_static_rate_limits():
    eps = 1e-4

    wd_resolution = 4
    ws_resolution = 0.5
    ti_resolution = 0.01
    df_opt = generic_df_opt(
        wd_resolution=wd_resolution,
        ws_resolution=ws_resolution,
        ti_resolution=ti_resolution
    )

    wd_rate_limit = 4
    ws_rate_limit = 4
    ti_rate_limit = 200
    df_opt_rate_limited = apply_static_rate_limits(
        df_opt,
        wd_rate_limit,
        ws_rate_limit,
        ti_rate_limit
    )

    # Check that the rate limits are applied
    offsets = np.vstack(df_opt_rate_limited.yaw_angles_opt.values).reshape(
        len(np.unique(df_opt.wind_direction)),
        len(np.unique(df_opt.wind_speed)),
        len(np.unique(df_opt.turbulence_intensity)),
        2
    )
    assert (np.abs(np.diff(offsets, axis=0)) <= wd_rate_limit*wd_resolution+eps).all()
    assert (np.abs(np.diff(offsets, axis=1)) <= ws_rate_limit*ws_resolution+eps).all()
    assert (np.abs(np.diff(offsets, axis=2)) <= ti_rate_limit*ti_resolution+eps).all()

    # Check wd test would have failed before rate limits applied
    offsets_unlimited = np.vstack(df_opt.yaw_angles_opt.values).reshape(
        len(np.unique(df_opt.wind_direction)),
        len(np.unique(df_opt.wind_speed)),
        len(np.unique(df_opt.turbulence_intensity)),
        2
    )
    assert not (np.abs(np.diff(offsets_unlimited, axis=0)) <= wd_rate_limit*wd_resolution).all()
    assert not (np.abs(np.diff(offsets_unlimited, axis=1)) <= ws_rate_limit*ws_resolution).all()
    assert not (np.abs(np.diff(offsets_unlimited, axis=2)) <= ti_rate_limit*ti_resolution).all()

def test_apply_wind_speed_ramps():

    ws_specified = 8.0
    ws_wake_steering_cut_out = 13.0
    ws_wake_steering_fully_engaged_high = 10.0
    df_opt_single_ws = generic_df_opt(ws_min=ws_specified, ws_max=ws_specified)

    df_opt_ramps = apply_wind_speed_ramps(
        df_opt_single_ws,
        ws_resolution=0.5,
        ws_wake_steering_fully_engaged_high=ws_wake_steering_fully_engaged_high,
        ws_wake_steering_cut_out=ws_wake_steering_cut_out,
    )

    # Check that the dataframes match at the specified wind speed
    assert np.allclose(
        np.vstack(df_opt_single_ws.yaw_angles_opt.values),
        np.vstack(df_opt_ramps[df_opt_ramps.wind_speed == ws_specified].yaw_angles_opt.values)
    )

    # Check that above wake steering cut out, all values are zero
    assert np.allclose(
        np.vstack(
            df_opt_ramps[df_opt_ramps.wind_speed >= ws_wake_steering_cut_out]
            .yaw_angles_opt.values
        ),
        0.0
    )

    # Check that between the cut out and fully engaged, values are linearly interpolated
    ws_midpoint = (ws_wake_steering_fully_engaged_high + ws_wake_steering_cut_out)/2
    assert np.allclose(
        np.vstack(df_opt_ramps[df_opt_ramps.wind_speed == ws_midpoint].yaw_angles_opt.values),
        (np.vstack(df_opt_ramps[df_opt_ramps.wind_speed == ws_wake_steering_cut_out]
                   .yaw_angles_opt.values)
         + np.vstack(df_opt_ramps[df_opt_ramps.wind_speed == ws_wake_steering_fully_engaged_high]
                     .yaw_angles_opt.values)
        )/2
    )

def test_wake_steering_interpolant():

    df_opt = generic_df_opt()

    yaw_interpolant = get_yaw_angles_interpolant(df_opt)

    # Confirm the yaw interpolant matches the lookup table
    offsets_interp = yaw_interpolant(
        df_opt.wind_direction.values,
        df_opt.wind_speed.values,
        df_opt.turbulence_intensity.values,
    )
    assert np.allclose(offsets_interp, np.vstack(df_opt.yaw_angles_opt.values))

    # Check interpolation at a specific point
    # (data at wd (268, 272) ws (8.0, 8.5) ti (0.06, 0.08))
    interpolated_offset = yaw_interpolant(271, 8.25, 0.06) 
    data = np.vstack(df_opt[
        (df_opt.wind_direction >= 268)
        & (df_opt.wind_direction <= 272)
        & (df_opt.wind_speed >= 8.0)
        & (df_opt.wind_speed <= 8.5)
    ].yaw_angles_opt.values).reshape(2,2,2,2)
    temp = 0.25*data[0,:,:,:] + 0.75*data[1,:,:,:] # wd interp
    temp = 0.5*temp[0,:,:] + 0.5*temp[1,:,:] # ws interp
    base = 1.0*temp[0,:] + 0.0*temp[1,:] # ti interp
    assert np.allclose(interpolated_offset, base)

    # Check extrapolation
    with pytest.raises(ValueError):
        _ = yaw_interpolant(200.0, 8.0, 0.06) # min specified wd is 220

    # Check wrapping works
    df_0_270 = generic_df_opt(wd_min=0.0, wd_max=270.0, wd_resolution=10.0) # Includes 0 degree WD
    yaw_interpolant = get_yaw_angles_interpolant(df_0_270)
    _ = yaw_interpolant(0.0, 8.0, 0.06)
    _ = yaw_interpolant(355.0, 8.0, 0.06)
    _ = yaw_interpolant(360.0, 8.0, 0.06)
    with pytest.raises(ValueError):
        _ = yaw_interpolant(-1.0, 8.0, 0.06)
    with pytest.raises(ValueError):
        _ = yaw_interpolant(361.0, 8.0, 0.06)

def test_hysteresis_zones():

    df_opt = generic_df_opt()
    min_zone_width = 4.0

    hysteresis_dict_base = {"T000": [(270-min_zone_width/2, 270+min_zone_width/2)]}

    # Calculate hysteresis regions
    hysteresis_dict_test = compute_hysteresis_zones(df_opt, min_zone_width=min_zone_width)
    assert hysteresis_dict_test == hysteresis_dict_base

    # Check angle wrapping works (runs through)
    df_opt = generic_df_opt(wd_min=0.0, wd_max=360.0)
    hysteresis_dict_test = compute_hysteresis_zones(df_opt, min_zone_width=min_zone_width)
    assert hysteresis_dict_test["T000"] == hysteresis_dict_base["T000"]

    # Limited wind directions that span 360/0 \
    df_opt_2 = generic_df_opt()
    df_opt_2.wind_direction = (df_opt_2.wind_direction + 90.0) % 360.0
    df_opt_2 = df_opt_2.sort_values(by=["wind_direction", "wind_speed", "turbulence_intensity"])
    hysteresis_dict_test = compute_hysteresis_zones(df_opt_2, min_zone_width=min_zone_width)
    assert ((np.array(hysteresis_dict_test["T000"][0]) - 90.0) % 360.0
            == np.array(hysteresis_dict_base["T000"][0])).all()

    # Check 0 low end, less than 360 upper end
    df_opt = generic_df_opt(wd_min=0.0, wd_max=300.0)
    hysteresis_dict_test = compute_hysteresis_zones(df_opt, min_zone_width=min_zone_width)
    assert hysteresis_dict_test["T000"] == hysteresis_dict_base["T000"]

    # Check nonzero low end, 360 upper end
    df_opt = generic_df_opt(wd_min=200.0, wd_max=360.0)
    hysteresis_dict_test = compute_hysteresis_zones(df_opt, min_zone_width=min_zone_width)
    assert hysteresis_dict_test["T000"] == hysteresis_dict_base["T000"]

    # Close to zero low end, 360 upper end
    df_opt = generic_df_opt(wd_min=2.0, wd_max=360.0)
    _ = compute_hysteresis_zones(df_opt)

    # Check grouping of regions by reducing yaw rate threshold
    df_opt = generic_df_opt()
    hysteresis_dict_test = compute_hysteresis_zones(
        df_opt,
        min_zone_width=3*min_zone_width, # Force regions to be grouped
        yaw_rate_threshold=1.0
    )
    # Check actual grouping occurs (not purely due to larger region width)
    assert (
        hysteresis_dict_test["T000"][0][1] - hysteresis_dict_test["T000"][0][0]
        > 3*min_zone_width
    )
    # Check new region covers original region
    assert hysteresis_dict_test["T000"][0][0] < hysteresis_dict_base["T000"][0][0]
    assert hysteresis_dict_test["T000"][0][1] > hysteresis_dict_base["T000"][0][1]

    # Make sure this works over the 360 degree wrap
    df_opt_2 = df_opt.copy()
    df_opt_2.wind_direction = (df_opt_2.wind_direction + 90.0) % 360.0
    df_opt_2 = df_opt_2.sort_values(by=["wind_direction", "wind_speed", "turbulence_intensity"])
    hysteresis_dict_test = compute_hysteresis_zones(
        df_opt_2,
        min_zone_width=3*min_zone_width,
        yaw_rate_threshold=1.0,
        verbose=True
    )
    # Check actual grouping occurs (not purely due to larger region width)
    assert (
        (hysteresis_dict_test["T000"][0][1] - hysteresis_dict_test["T000"][0][0]) % 360.0
        > 3*min_zone_width
    )
    # Check new region covers original region
    assert (hysteresis_dict_test["T000"][0][0] - 90.0) % 360.0 < hysteresis_dict_base["T000"][0][0]
    assert (hysteresis_dict_test["T000"][0][1] - 90.0) % 360.0 > hysteresis_dict_base["T000"][0][1]


def test_consolidate_hysteresis_zones():

    # Check basic grouping
    hysteresis_wds_base = [(10, 30)]
    hysteresis_wds_unconsolidated = [(10, 20), (18, 25), (25, 30)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # # Check 360 degree wrap
    hysteresis_wds_base = [(350, 10)]
    hysteresis_wds_unconsolidated = [(350, 355), (355, 5), (5, 10)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # # Only last one crosses 0/360 divide
    hysteresis_wds_base = [(350, 10)]
    hysteresis_wds_unconsolidated = [(350, 355), (354, 10)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # # Only first one crosses 0/360 divide
    hysteresis_wds_base = [(350, 10)]
    hysteresis_wds_unconsolidated = [(350, 5), (5, 10)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # Larger set
    width = 6.0
    hysteresis_centers = [  8.,  12.,  16., 344., 348., 352., 359.]
    hysteresis_wds_base = [(344.0-width, 16.0+width)]
    hysteresis_wds_unconsolidated = [
        (hysteresis_centers[0]-width, hysteresis_centers[0]+width),
        (hysteresis_centers[1]-width, hysteresis_centers[1]+width),
        (hysteresis_centers[2]-width, hysteresis_centers[2]+width),
        (hysteresis_centers[3]-width, hysteresis_centers[3]+width),
        (hysteresis_centers[4]-width, hysteresis_centers[4]+width),
        (hysteresis_centers[5]-width, hysteresis_centers[5]+width),
        (hysteresis_centers[6]-width, (hysteresis_centers[6]+width)%360.0)
    ]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base
    
    # Two crossing 0/360 divide
    hysteresis_wds_base = [(350, 10)]
    hysteresis_wds_unconsolidated = [(350, 355), (355, 5), (357, 8), (7, 10)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # Three crossing 0/360 divide
    hysteresis_wds_base = [(350, 10)]
    hysteresis_wds_unconsolidated = [(350, 355), (355, 5), (357, 8), (358, 9), (7, 10)]
    hysteresis_wds_test = consolidate_hysteresis_zones(hysteresis_wds_unconsolidated)
    assert hysteresis_wds_test == hysteresis_wds_base

    # A few more edge cases
    hysteresis_wds_unconsolidated = [(15, 30), (35, 50), (340, 355), (345, 0), (355, 10)]
    hysteresis_wds_base = [(15, 30), (35, 50), (340, 10)]
    assert consolidate_hysteresis_zones(hysteresis_wds_unconsolidated) == hysteresis_wds_base

    hysteresis_wds_unconsolidated = [(15, 30), (35, 50)]
    hysteresis_wds_base = hysteresis_wds_unconsolidated
    assert consolidate_hysteresis_zones(hysteresis_wds_unconsolidated) == hysteresis_wds_base

    hysteresis_wds_unconsolidated = [
        (0, 15), (10, 25), (15, 30), (35, 50), (340, 355), (345, 0), (355, 10)
    ]
    hysteresis_wds_base = [(340, 30), (35, 50)]
    assert consolidate_hysteresis_zones(hysteresis_wds_unconsolidated) == hysteresis_wds_base

def test_create_uniform_wind_rose():
    wind_rose = create_uniform_wind_rose()
    frequencies = wind_rose.unpack_freq()
    assert (frequencies == frequencies[0]).all()

def test_check_df_opt_ordering():

    # Pass tests
    df_opt = generic_df_opt()
    check_df_opt_ordering(df_opt)

    # Remove a row so that not all data is present
    with pytest.raises(ValueError):
        check_df_opt_ordering(df_opt.drop(0))

    # Artificially create bad ordering by swapping columns
    df_opt_2 = df_opt.copy()
    df_opt_2.wind_speed = df_opt_2.wind_direction
    df_opt_2.wind_direction = df_opt.wind_speed
    with pytest.raises(ValueError):
        check_df_opt_ordering(df_opt_2)
