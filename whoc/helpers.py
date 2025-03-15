import numpy as np

def cluster_turbines(fi, wind_direction=None, wake_slope=0.30, radius_of_influence=None):
    """Separate a wind farm into separate clusters in which the turbines in
    each subcluster only affects the turbines in its cluster and has zero
    interaction with turbines from other clusters, both ways (being waked,
    generating wake), This allows the user to separate the control setpoint
    optimization in several lower-dimensional optimization problems, for
    example. This function assumes a very simplified wake function where the
    wakes are assumed to have a linearly diverging profile. In comparisons
    with the FLORIS GCH model, the wake_slope matches well with the FLORIS'
    wake profiles for a value of wake_slope = 0.5 * turbulence_intensity, where
    turbulence_intensity is an input to the FLORIS model at the default
    GCH parameterization. Note that does not include wind direction variability.
    To be conservative, the user is recommended to use the rule of thumb:
    `wake_slope = turbulence_intensity`. Hence, the default value for
    `wake_slope=0.30` should be conservative for turbulence intensities up to
    0.30 and is likely to provide valid estimates of which turbines are
    downstream until a turbulence intensity of 0.50. This simple model saves
    time compared to FLORIS.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wind_direction (float): The wind direction in the FLORIS frame
        of reference for which the downstream turbines are to be determined.
        wake_slope (float, optional): linear slope of the wake (dy/dx)
        plot_lines (bool, optional): Enable plotting wakes/turbines.
        Defaults to False.

    Returns:
        clusters (iterable): A list in which each entry contains a list
        of turbine numbers that together form a cluster which
        exclusively interact with one another and have zero
        interaction with turbines outside of this cluster.
    """

    if wind_direction is None:
        wind_direction = np.mean(fi.floris.farm.wind_direction)

    # Get farm layout
    x = fi.layout_x
    y = fi.layout_y
    D = np.array([t.rotor_diameter for t in fi.floris.farm.turbines])
    n_turbs = len(x)

    # Rotate farm and determine freestream/waked turbines
    is_downstream = [False for _ in range(n_turbs)]
    # TODO fix this rotation
    x_rot = (
        np.cos((wind_direction - 270.0) * np.pi / 180.0) * x
        - np.sin((wind_direction - 270.0) * np.pi / 180.0) * y
    )
    y_rot = (
        np.sin((wind_direction - 270.0) * np.pi / 180.0) * x
        + np.cos((wind_direction - 270.0) * np.pi / 180.0) * y
    )

    srt = np.argsort(x_rot)
    usrt = np.argsort(srt)
    x_rot_srt = x_rot[srt]
    y_rot_srt = y_rot[srt]
    affected_by_turbs = np.tile(False, (n_turbs, n_turbs))
    # TODO only consider "affected" if the turbine is within one unit crossstream/downstream
    # TODO return list of dictionaries {"Pivot turbine": {"x": x, "y": y, "immediate_ds_turbines":[]}}
    for ii in range(n_turbs):
        # get coordinate of "pivot" turbine
        x0 = x_rot_srt[ii]
        y0 = y_rot_srt[ii]

        def wake_profile_ub_turbii(x):
            y = (y0 + D[ii]) + (x - x0) * wake_slope
            if isinstance(y, (float, np.float64, np.float32)):
                if x < (x0 + 0.01):
                    y = -np.Inf
            else:
                y[x < x0 + 0.01] = -np.Inf
            return y

        def wake_profile_lb_turbii(x):
            y = (y0 - D[ii]) - (x - x0) * wake_slope
            if isinstance(y, (float, np.float64, np.float32)):
                if x < (x0 + 0.01):
                    y = -np.Inf
            else:
                y[x < x0 + 0.01] = -np.Inf
            return y

        def determine_if_in_wake(xt, yt):
            return (yt < wake_profile_ub_turbii(xt)) & (yt > wake_profile_lb_turbii(xt))

        # Get most downstream turbine
        is_downstream[ii] = not any(
            determine_if_in_wake(x_rot_srt[iii], y_rot_srt[iii]) for iii in range(n_turbs)
        )
        # Determine which turbines are affected by this turbine ('ii')
        # TODO only if within radius
        affecting_following_turbs = [
                determine_if_in_wake(x_rot_srt[iii], y_rot_srt[iii])
                for iii in range(n_turbs)
        ]

        # Determine by which turbines this turbine ('ii') is affected
        for aft in np.where(affecting_following_turbs)[0]:
            affected_by_turbs[aft, ii] = True

    # Rearrange into initial frame of reference
    affected_by_turbs = affected_by_turbs[:, usrt][usrt, :]
    for ii in range(n_turbs):
        affected_by_turbs[ii, ii] = True  # Add self to turb_list_affected
    affected_by_turbs = [np.where(c)[0] for c in affected_by_turbs]

    # List of downstream turbines
    turbs_downstream = [is_downstream[i] for i in usrt]
    turbs_downstream = list(np.where(turbs_downstream)[0])

    # Initialize one cluster for each turbine and all the turbines its affected by
    clusters = affected_by_turbs

    # Iteratively merge clusters if any overlap between turbines
    ci = 0
    while ci < len(clusters):
        # Compare current row to the ones to the right of it
        cj = ci + 1
        merged_column = False
        while cj < len(clusters):
            if any(y in clusters[ci] for y in clusters[cj]):
                # Merge
                clusters[ci] = np.hstack([clusters[ci], clusters[cj]])
                clusters[ci] = np.array(np.unique(clusters[ci]), dtype=int)
                clusters.pop(cj)
                merged_column = True
            else:
                cj = cj + 1
        if not merged_column:
            ci = ci + 1

    return clusters
