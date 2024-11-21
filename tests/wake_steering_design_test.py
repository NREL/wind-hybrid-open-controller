from pathlib import Path

import numpy as np
from floris import FlorisModel
from whoc.design_tools.wake_steering_design import build_simple_wake_steering_lookup_table

TEST_DATA = Path(__file__).resolve().parent
YAML_INPUT = TEST_DATA / "floris_input.yaml"

def test_build_simple_wake_steering_lookup_table():
    fmodel_test = FlorisModel(YAML_INPUT)

    # Start with the simple case
    wd_resolution = 4.0
    wd_min = 220.0
    wd_max = 310.0
    ws_resolution = 0.5
    ws_min = 8.0
    ws_max = 10.0
    minimum_yaw_angle = -20
    maximum_yaw_angle = 20
    df_opt = build_simple_wake_steering_lookup_table(
        fmodel_test,
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    opt_yaw_angles = np.vstack(df_opt["yaw_angles_opt"])

    n_conditions = ((ws_max-ws_min)//ws_resolution+1) * ((wd_max-wd_min)//wd_resolution+1)

    assert opt_yaw_angles.shape == (n_conditions, 2)
    assert (opt_yaw_angles >= minimum_yaw_angle).all()
    assert (opt_yaw_angles <= maximum_yaw_angle).all()

    # More complex case (include 360, minimum yaw greater than zero)
    wd_min = 0.0
    wd_max = 360.0
    minimum_yaw_angle = -5 # Positive numbers DO NOT WORK. FLORIS bug?
    df_opt = build_simple_wake_steering_lookup_table(
        fmodel_test,
        wd_resolution=wd_resolution,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_resolution=ws_resolution,
        ws_min=ws_min,
        ws_max=ws_max,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
    )

    opt_yaw_angles = np.vstack(df_opt["yaw_angles_opt"])

    n_conditions = ((ws_max-ws_min)//ws_resolution+1) * ((wd_max-wd_min)//wd_resolution)

    assert opt_yaw_angles.shape == (n_conditions, 2)
    assert (opt_yaw_angles >= minimum_yaw_angle).all()
    assert (opt_yaw_angles <= maximum_yaw_angle).all()
