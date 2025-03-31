import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_ts(df, fig_dir, include_transformation=False):
    if not include_transformation:
        sns.set(font_scale=1.5) # for single column figure
    # Plot vs. time
    fig_ts, ax_ts = plt.subplots(2, 2 if include_transformation else 1, sharex=True)  # len(case_list), 5)
    # fig_ts.set_size_inches(12, 6)
    if hasattr(ax_ts, '__len__'):
        ax_ts = ax_ts.flatten()
    else:
        ax_ts = [ax_ts]
        
    df["time"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
     
    sns.lineplot(data=df, hue="WindSeed", x="time", y="FreestreamWindSpeedU", ax=ax_ts[0])
    sns.lineplot(data=df, hue="WindSeed", x="time", y="FreestreamWindSpeedV", ax=ax_ts[1])
    if include_transformation:
        sns.lineplot(data=df, hue="WindSeed", x="time", y="FreestreamWindMag", ax=ax_ts[2])
        sns.lineplot(data=df, hue="WindSeed", x="time", y="FreestreamWindDir", ax=ax_ts[3])

    ax_ts[0].set(title='Downwind Freestream Wind Speed, U [m/s]', ylabel="")
    ax_ts[1].set(title='Crosswind Freestream Wind Speed, V [m/s]', ylabel="")
    if include_transformation:
        ax_ts[2].set(title='Freestream Wind Magnitude [m/s]', ylabel="")
        ax_ts[3].set(title='Freestream Wind Direction [$^\\circ$]', ylabel="")

    # handles, labels, kwargs = mlegend._parse_legend_args([ax_ts[0]], ncol=2, title="Wind Seed")
    # ax_ts[0].legend_ = mlegend.Legend(ax_ts[0], handles, labels, **kwargs)
    # ax_ts[0].legend_.set_ncols(2)
    for i in range(0, len(ax_ts)):
        ax_ts[i].legend([], [], frameon=False)
    
    time = df.loc[df["WindSeed"] == 0, "time"]
    for i in range(len(ax_ts) - 2, len(ax_ts)):
        ax_ts[i].set(xticks=time.iloc[0:-1:int(60 * 12 // (time.iloc[1] - time.iloc[0]))].astype(int), 
                     xlabel='Time [s]', 
                     xlim=(0, 3600))
    
    # for seed in pd.unique(df["WindSeed"]):
    #     seed_df = df.loc[df["WindSeed"] == seed, :]
    #     time = seed_df['Time']
    #     freestream_wind_speed_u = seed_df['FreestreamWindSpeedU'].to_numpy()
    #     freestream_wind_speed_v = seed_df['FreestreamWindSpeedV'].to_numpy()
    #     freestream_wind_mag = seed_df['FreestreamWindMag'].to_numpy()
    #     freestream_wind_dir = seed_df['FreestreamWindDir'].to_numpy()
    #     # freestream_wind_mag = np.linalg.norm(np.vstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=0)
    #     # freestream_wind_dir = np.arctan(freestream_wind_speed_u / freestream_wind_speed_v) * (180 / np.pi) + 180
        
    #     ax_ts[0].plot(time, freestream_wind_speed_u, label=f"Seed {seed}")
    #     ax_ts[1].plot(time, freestream_wind_speed_v)
    #     ax_ts[2].plot(time, freestream_wind_mag)
    #     ax_ts[3].plot(time, freestream_wind_dir)
        
    #     ax_ts[0].set(title='Freestream Wind Speed, U [m/s]')
    #     ax_ts[1].set(title='Freestream Wind Speed, V [m/s]')
    #     ax_ts[2].set(title='Freestream Wind Magnitude [m/s]')
    #     ax_ts[3].set(title='Freestream Wind Direction [$^\circ$]')
        
    #     for ax in ax_ts[2:]:
    #         ax.set(xticks=time.iloc[0:-1:int(60 * 12 // (time.iloc[1] - time.iloc[0]))], xlabel='Time [s]', xlim=(time.iloc[0], time.iloc[-1]))
    
    # ax_ts[0].legend(ncol=2)
    plt.tight_layout()
    fig_ts.savefig(os.path.join(fig_dir, f'wind_field_ts.png'))
    # fig_ts.show()

def plot_distribution_samples(df, n_preview_steps, fig_dir):
    # Plot vs. time
    
    freestream_wind_speed_u = df[[f'FreestreamWindSpeedU_{j}' for j in range(n_preview_steps)]].to_numpy()
    freestream_wind_speed_v = df[[f'FreestreamWindSpeedV_{j}' for j in range(n_preview_steps)]].to_numpy()
    freestream_wind_mag = df[[f'FreestreamWindMag_{j}' for j in range(n_preview_steps)]].to_numpy()
    freestream_wind_dir = df[[f'FreestreamWindDir_{j}' for j in range(n_preview_steps)]].to_numpy()
    
    n_samples = freestream_wind_speed_u.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, n_samples))
    preview_time = np.arange(n_preview_steps)

    fig_scatter, ax_scatter = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
    if hasattr(ax_scatter, '__len__'):
        ax_scatter = ax_scatter.flatten()
    else:
        ax_scatter = [ax_scatter]
    
    fig_plot, ax_plot = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
    if hasattr(ax_plot, '__len__'):
        ax_plot = ax_plot.flatten()
    else:
        ax_plot = [ax_plot]
    
    for i in range(n_samples):
        ax_scatter[0].scatter(preview_time, freestream_wind_speed_u[i, :], marker='o', color=colors[i])
        ax_scatter[1].scatter(preview_time, freestream_wind_speed_v[i, :], marker='o', color=colors[i])
        ax_scatter[2].scatter(preview_time, freestream_wind_mag[i, :], marker='o', color=colors[i])
        ax_scatter[3].scatter(preview_time, freestream_wind_dir[i, :], marker='o', color=colors[i])
    
    
    for i, c in zip(range(n_samples), cycle(colors)):
        ax_plot[0].plot(preview_time, freestream_wind_speed_u[i, :], color=c)
        ax_plot[1].plot(preview_time, freestream_wind_speed_v[i, :], color=c)
        ax_plot[2].plot(preview_time, freestream_wind_mag[i, :], color=c)
        ax_plot[3].plot(preview_time, freestream_wind_dir[i, :], color=c)
    
    for axs in [ax_scatter, ax_plot]:
        axs[0].set(title='Freestream Wind Speed, U [m/s]')
        axs[1].set(title='Freestream Wind Speed, V [m/s]')
        axs[2].set(title='Freestream Wind Magnitude [m/s]')
        axs[3].set(title='Freestream Wind Direction [deg]')
    
    for ax in chain(ax_scatter, ax_plot):
        ax.set(xticks=preview_time, xlabel='Preview Time-Steps')
    
    # fig_scatter.show()
    # fig_plot.show()
    fig_scatter.savefig(os.path.join(fig_dir, f'wind_field_preview_samples1.png'))
    fig_plot.savefig(os.path.join(fig_dir, f'wind_field_preview_samples2.png'))


def plot_distribution_ts(wf, n_preview_steps):
    # Plot vs. time
    fig_scatter, ax_scatter = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
    if hasattr(ax_scatter, '__len__'):
        ax_scatter = ax_scatter.flatten()
    else:
        ax_scatter = [ax_scatter]
    
    fig_plot, ax_plot = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
    if hasattr(ax_plot, '__len__'):
        ax_plot = ax_plot.flatten()
    else:
        ax_plot = [ax_plot]
    
    time = wf.df['Time'].to_numpy()
    freestream_wind_speed_u = wf.df[[f'FreestreamWindSpeedU_{i}' for i in range(n_preview_steps)]].to_numpy()
    freestream_wind_speed_v = wf.df[[f'FreestreamWindSpeedV_{i}' for i in range(n_preview_steps)]].to_numpy()
    freestream_wind_mag = (freestream_wind_speed_u ** 2 + freestream_wind_speed_v ** 2) ** 0.5
    
    # # compute directions
    freestream_wind_dir = np.arctan2(freestream_wind_speed_u, freestream_wind_speed_v)
    freestream_wind_dir = (180.0 + np.rad2deg(freestream_wind_dir)) % 360.0
    
    colors = cm.rainbow(np.linspace(0, 1, n_preview_steps))
    
    idx = slice(600)
    for i in range(n_preview_steps):
        ax_scatter[0].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_speed_u[idx, i],
                              marker='o', color=colors[i])
        ax_scatter[1].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_speed_v[idx, i],
                              marker='o', color=colors[i])
        ax_scatter[2].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_mag[idx, i], marker='o',
                              color=colors[i])
        ax_scatter[3].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_dir[idx, i], marker='o',
                              color=colors[i])
    
    idx = slice(10)
    for k, c in zip(range(len(time[idx])), cycle(colors)):
        # i = (np.arange(k * DT, k * DT + wf.wind_speed_preview_time, wf.wind_speed_sampling_time_step) * (
        # 		1 // DT)).astype(int)
        i = slice(k, k + int(wf.wind_speed_preview_time // wf.simulation_dt), 1)
        ax_plot[0].plot(time[i], freestream_wind_speed_u[k, :], color=c)
        ax_plot[1].plot(time[i], freestream_wind_speed_v[k, :], color=c)
        ax_plot[2].plot(time[i], freestream_wind_mag[k, :], color=c)
        ax_plot[3].plot(time[i], freestream_wind_dir[k, :], color=c)
    
    for axs in [ax_scatter, ax_plot]:
        axs[0].set(title='Freestream Wind Speed, U [m/s]')
        axs[1].set(title='Freestream Wind Speed, V [m/s]')
        axs[2].set(title='Freestream Wind Magnitude [m/s]')
        axs[3].set(title='Freestream Wind Direction [deg]')
    
    for ax in chain(ax_scatter, ax_plot):
        ax.set(xticks=time[idx][0:-1:int(60 // wf.dt)], xlabel='Time [s]')
    
    # fig_scatter.show()
    # fig_plot.show()
    fig_scatter.savefig(os.path.join(wf.fig_dir, f'wind_field_preview_ts1.png'))
    fig_plot.savefig(os.path.join(wf.fig_dir, f'wind_field_preview_ts2.png'))

