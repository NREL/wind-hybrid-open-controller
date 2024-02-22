import matplotlib.pyplot as plt
import numpy as np

def plot_power_vs_speed(df):
	# Plot power and AEP uplift across wind speed
	fig, ax = plt.subplots(nrows=3, sharex=True)
	
	df_avg = df.groupby("ws").mean().reset_index(drop=False)
	mean_power_uplift = 100.0 * (df_avg["farm_power_relative"] - 1.0)
	ax[0].bar(
		x=df_avg["ws"],
		height=mean_power_uplift,
		color="darkgray",
		edgecolor="black",
		width=0.95,
	)
	ax[0].set_title("Mean power uplift [%]")
	ax[0].grid(True)
	
	dist = df.groupby("ws").sum().reset_index()
	ax[1].bar(
		x=dist["ws"],
		height=100 * dist["rel_energy_uplift"],
		color="darkgray",
		edgecolor="black",
		width=0.95,
	)
	ax[1].set_title("Contribution to AEP uplift [%]")
	ax[1].grid(True)
	
	ax[2].bar(
		x=dist["ws"],
		height=dist["freq_val"],
		color="darkgray",
		edgecolor="black",
		width=0.95,
	)
	ax[2].set_xlabel("Wind speed [m/s]")
	ax[2].set_title("Frequency of occurrence [-]")
	ax[2].grid(True)
	plt.tight_layout()
	plt.show()

	return fig


def plot_yaw_vs_dir(yaw_offsets_interpolant, n_turbines):
	# Now plot yaw angle distributions over wind direction up to first three turbines
	wd_plot = np.arange(0.0, 360.001, 1.0)
	ws_plot = [6.0, 9.0, 12.0]
	wd_grid, ws_grid = np.meshgrid(wd_plot, ws_plot, indexing="ij")
	colors = ["maroon", "dodgerblue", "grey"]
	styles = ["-o", "-v", "-o"]
	yaw_offsets = yaw_offsets_interpolant(wd_grid, ws_grid)
	figs = []
	
	for i in range(min(n_turbines, 3)):
		fig, ax = plt.subplots()
		for ws_idx, ws in enumerate(ws_plot):
			ax.plot(
				wd_plot,
				yaw_offsets[:, ws_idx, i],
				styles[ws_idx],
				color=colors[ws_idx],
				markersize=3,
				label=f"For wind speed of {ws:.1f} m/s",
			)
		ax.set_xlabel("Wind direction [deg]")
		ax.set_title(f"Assigned yaw offsets for Turbine {i:d} [deg]")
		ax.grid(True)
		ax.legend()
		plt.tight_layout()
		figs.append(fig)
	
	plt.show()
	return figs


def plot_power_vs_dir(df, wind_directions):
	# Plot power and AEP uplift across wind direction
	wd_step = np.diff(wind_directions)[0]  # Useful variable for plotting
	fig, ax = plt.subplots(nrows=3, sharex=True)
	
	df_8ms = df[df["ws"] == 8.0].reset_index(drop=True)
	pow_uplift = 100 * (
		df_8ms["farm_power_opt"] / df_8ms["farm_power_baseline"] - 1
	)
	ax[0].bar(
		x=df_8ms["wd"],
		height=pow_uplift,
		color="darkgray",
		edgecolor="black",
		width=wd_step,
	)
	ax[0].set_title("Power uplift at 8 m/s [%]")
	ax[0].grid(True)
	
	dist = df.groupby("wd").sum().reset_index()
	ax[1].bar(
		x=dist["wd"],
		height=100 * dist["rel_energy_uplift"],
		color="darkgray",
		edgecolor="black",
		width=wd_step,
	)
	ax[1].set_title("Contribution to AEP uplift [%]")
	ax[1].grid(True)
	
	ax[2].bar(
		x=dist["wd"],
		height=dist["freq_val"],
		color="darkgray",
		edgecolor="black",
		width=wd_step,
	)
	ax[2].set_xlabel("Wind direction [deg]")
	ax[2].set_title("Frequency of occurrence [-]")
	ax[2].grid(True)
	plt.tight_layout()
	plt.show()

	return fig