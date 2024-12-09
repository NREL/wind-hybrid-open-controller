"""Module for visualizing yaw optimizer results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_offsets_wswd_heatmap(df_opt, turb_id, ax=None):
    """Plot offsets for a single turbine as a heatmap in wind speed.

    df_opt should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbulence_intensity
       - yaw_angles_opt

    Produces a heat map of the offsets for all wind directions and
    wind speeds for turbine specified by turb_id. Dataframe is assumed
    to contain individual turbine offsets in distinct columns (unlike
    the yaw_angles_opt column from FLORIS.

    Args:
        df_offsets (pd.DataFrame): dataframe with offsets
        turb_id (int or str): turbine id or column name
        ax (matplotlib.axes.Axes): axis to plot on.  If None, a new figure is created.
            Default is None.

    Returns:
        A tuple containing a matplotlib.axes.Axes object and a matplotlib.colorbar.Colorbar

    """
    if "yaw_angles_opt" not in df_opt.columns:
        raise ValueError("df_opt must contain yaw_angles_opt column.")
    else:
        offsets_all = np.vstack(df_opt.yaw_angles_opt.to_numpy())[:, turb_id]

    ws_array = np.unique(df_opt.wind_speed)
    wd_array = np.unique(df_opt.wind_direction)

    # Construct array of offets
    offsets_array = np.zeros((len(ws_array), len(wd_array)))
    for i, ws in enumerate(ws_array):
        offsets_array[-i, :] = offsets_all[df_opt.wind_speed == ws]

    if ax is None:
        _, ax = plt.subplots(1, 1)
    d_wd = (wd_array[1] - wd_array[0]) / 2
    d_ws = (ws_array[1] - ws_array[0]) / 2
    im = ax.imshow(
        offsets_array,
        interpolation=None,
        extent=[wd_array[0] - d_wd, wd_array[-1] + d_wd, ws_array[0] - d_ws, ws_array[-1] + d_ws],
        aspect="auto",
    )
    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Wind speed")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Yaw offset")

    return ax, cbar


def plot_offsets_wd(df_opt, turb_id, ws_plot, color="black", alpha=1.0, label=None, ax=None):
    """Plot offsets for a single turbine as a function of wind direction.

    df_opt should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbulence_intensity
       - yaw_angles_opt

    if ws_plot is scalar, only that wind speed is plotted. If ws_plot is
    a two-element tuple or list, that range of wind speeds is plotted.

    label only allowed if single wind speed is given.

    Args:
        df_opt (pd.DataFrame): dataframe with offsets, as produced by FLORIS yaw optimizer
        turb_id (int or str): index of the turbine to plot
        ws_plot (float or list): wind speed to plot
        color (str): color of line
        alpha (float): transparency of line
        label (str): label for line
        ax (matplotlib.axes.Axes): axis to plot on.  If None, a new figure is created.
            Default is None.
    """
    if "yaw_angles_opt" not in df_opt.columns:
        raise ValueError("df_opt must contain yaw_angles_opt column.")
    else:
        offsets_all = np.vstack(df_opt.yaw_angles_opt.to_numpy())[:, turb_id]
    
    if hasattr(ws_plot, "__len__") and label is not None:
        label = None
        print("label option can only be used for single wind speed plot.")

    ws_array = np.unique(df_opt.wind_speed)
    wd_array = np.unique(df_opt.wind_direction)

    if hasattr(ws_plot, "__len__"):
        offsets_list = []
        for ws in ws_array:
            if ws >= ws_plot[0] and ws <= ws_plot[-1]:
                offsets_list.append(offsets_all[df_opt.wind_speed == ws])
    else:
        offsets_list = [offsets_all[df_opt.wind_speed == ws_plot]]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    for offsets in offsets_list:
        ax.plot(wd_array, offsets, color=color, alpha=alpha, label=label)

    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Yaw offset")

    return ax