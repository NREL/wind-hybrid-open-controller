"""Module for visualizing yaw optimizer results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_offsets_wdws_heatmap(
    df_opt,
    turb_id,
    ti_plot=None,
    vmin=None,
    vmax=None,
    cmap="coolwarm",
    ax=None
):
    """Plot offsets for a single turbine as a heatmap in wind speed.

    df_opt should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbulence_intensity
       - yaw_angles_opt

    Produces a heat map of the offsets for all wind directions and
    wind speeds for turbine specified by turb_id. df_opt is assumed
    to be in the form produced by FLORIS yaw optimization routines (or
    functions in WHOC's wake_steering_design module).

    Args:
        df_opt (pd.DataFrame): dataframe with offsets
        turb_id (int or str): turbine id or column name
        ti_plot (float): turbulence intensity to plot. If None, assumes only one turbulence
           intensity in df_opt. Defaults to None.
        vmin (float): minimum value for color scale. Defaults to None.
        vmax (float): maximum value for color scale. Defaults to None.
        cmap (str): colormap to use. Defaults to "coolwarm".
        ax (matplotlib.axes.Axes): axis to plot on.  If None, a new figure is created.
            Default is None.

    Returns:
        A tuple containing a matplotlib.axes.Axes object and a matplotlib.colorbar.Colorbar

    """
    if "yaw_angles_opt" not in df_opt.columns:
        raise ValueError("df_opt must contain yaw_angles_opt column.")
    else:
        offsets_all = np.vstack(df_opt.yaw_angles_opt.to_numpy())[:, turb_id]

    if ti_plot is None:
        ti_plot = np.unique(df_opt.turbulence_intensity)
        if ti_plot.size > 1:
            raise ValueError(
                "Multiple turbulence intensities present in df_opt. Must specify ti_plot."
            )

    ws_array = np.unique(df_opt.wind_speed)
    wd_array = np.unique(df_opt.wind_direction)

    # Construct array of offsets
    offsets_array = np.zeros((len(ws_array), len(wd_array)))
    for i, ws in enumerate(ws_array):
        offsets_array[i, :] = offsets_all[
            (df_opt.wind_speed == ws) & (df_opt.turbulence_intensity == ti_plot)
        ]

    if ax is None:
        _, ax = plt.subplots(1, 1)
    d_wd = (wd_array[1] - wd_array[0]) / 2
    d_ws = (ws_array[1] - ws_array[0]) / 2
    im = ax.imshow(
        offsets_array,
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[wd_array[0] - d_wd, wd_array[-1] + d_wd, ws_array[0] - d_ws, ws_array[-1] + d_ws],
        aspect="auto",
        origin="lower",
    )
    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Wind speed")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Yaw offset")

    return ax, cbar


def plot_offsets_wd(
    df_opt,
    turb_id,
    ws_plot,
    ti_plot = None,
    color = "black",
    linestyle = "-",
    alpha = 1.0,
    label = None,
    ax = None
):
    """Plot offsets for a single turbine as a function of wind direction.

    df_opt should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbulence_intensity
       - yaw_angles_opt

    If ws_plot is scalar, only that wind speed is plotted. If ws_plot is
    a two-element tuple or list, that range of wind speeds is plotted.

    If ti_plot is None, assumes only one turbulence intensity is present in df_opt.
    If ti_plot is scalar, only that turbulence intensity is plotted. If ti_plot is
    a two-element tuple or list, that range of turbulence intensities is plotted.

    label only allowed if single wind speed is given.

    Args:
        df_opt (pd.DataFrame): dataframe with offsets, as produced by FLORIS yaw optimizer
        turb_id (int or str): index of the turbine to plot
        ws_plot (float or list): wind speed to plot
        ti_plot (float or list): turbulence intensity to plot
        color (str): color of line
        alpha (float): transparency of line
        label (str): label for line
        ax (matplotlib.axes.Axes): axis to plot on. If None, a new figure is created.
            Default is None.
    """
    if "yaw_angles_opt" not in df_opt.columns:
        raise ValueError("df_opt must contain yaw_angles_opt column.")
    else:
        offsets_all = np.vstack(df_opt.yaw_angles_opt.to_numpy())[:, turb_id]

    if ti_plot is None:
        ti_plot = np.unique(df_opt.turbulence_intensity)
        if ti_plot.size > 1:
            raise ValueError(
                "Multiple turbulence intensities present in df_opt. Must specify ti_plot."
            )
    elif not hasattr(ti_plot, "__len__"):
        ti_plot = [ti_plot]

    if not hasattr(ws_plot, "__len__"):
        ws_plot = [ws_plot]

    if len(ws_plot) > 1 and label is not None:
        label = None
        print("label option can only be used for single wind speed plot.")

    if len(ti_plot) > 1 and label is not None:
        label = None
        print("label option can only be used for single turbulence intensity plot.")

    if set(ws_plot) <= set(df_opt.wind_speed):
        pass
    else:
        raise ValueError("One or more ws_plot values not found in df_opt.wind_speed.")
    
    if set(ti_plot) <= set(df_opt.turbulence_intensity):
        pass
    else:
        raise ValueError("One or more ti_plot values not found in df_opt.turbulence_intensity.")

    wd_array = np.unique(df_opt.wind_direction)
    ws_array = np.unique(df_opt.wind_speed)
    ti_array = np.unique(df_opt.turbulence_intensity)

    offsets_list = []
    for ws in ws_array:
        if ws >= ws_plot[0] and ws <= ws_plot[-1]:
            for ti in ti_plot:
                if ti >= ti_array[0] and ti <= ti_array[-1]:
                    offsets_list.append(
                        offsets_all[(df_opt.wind_speed == ws) & (df_opt.turbulence_intensity == ti)]
                    )

    if ax is None:
        _, ax = plt.subplots(1, 1)

    for offsets in offsets_list:
        ax.plot(wd_array, offsets, color=color, linestyle=linestyle, alpha=alpha, label=label)

    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Yaw offset")

    return ax