import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_inputs = True # Save generated inputs to csv files
show_inputs = False # Plot generated inputs for inspection

dt = 1.0 # time step in seconds
total_time = 600 # total time in seconds
time = np.arange(0, total_time, dt)

# Create a varied reference signal for hydrogen production by overlaying a constant value
# and two sine waves of different frequencies and amplitudes.
hydrogen_base_value = 0.03 # in kg/s
sine1_frequency = 5 / total_time # Hz
sine1_amplitude = 0.1 # as a percentage of the reference value
sine2_frequency = 3 / total_time # Hz
sine2_amplitude = 0.2 # as a percentage of the reference value
sine_start_time = 100 # seconds

hydrogen_reference = hydrogen_base_value * np.ones(len(time))
sine1 = np.sin(2 * np.pi * sine1_frequency * time) * hydrogen_base_value * sine1_amplitude
sine2 = np.sin(2 * np.pi * sine2_frequency * time) * hydrogen_base_value * sine2_amplitude
hydrogen_reference[round(sine_start_time/dt):] += (
    sine1[:-round(sine_start_time/dt)] + 
    sine2[:-round(sine_start_time/dt)]
)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time, hydrogen_reference)
ax[0].set_ylabel("Full hydrogen\nreference [kg/s]")
ax[0].grid()

ax[1].plot([time[0], time[-1]], [hydrogen_base_value, hydrogen_base_value])
ax[1].plot(time[round(sine_start_time/dt):], sine1[:-round(sine_start_time/dt)])
ax[1].plot(time[round(sine_start_time/dt):], sine2[:-round(sine_start_time/dt)])
ax[1].set_ylabel("Hydrogen reference\ncomponents [kg/s]")
ax[1].set_xlabel("Time [s]")
ax[1].grid()

if save_inputs:
    df = pd.DataFrame(data={"time":time, "hydrogen_reference":hydrogen_reference})
    df.to_csv("inputs/hydrogen_ref_signal.csv", index=False, header=True)

# Generate (constant) wind speed and direction inputs
wind_speed = 10.0 # m/s
wind_direction = 240.0 # degrees

if save_inputs:
    df = pd.DataFrame(
        data={
            "time":time,
            "amr_wind_speed":wind_speed*np.ones_like(time),
            "amr_wind_direction":wind_direction*np.ones_like(time)
        }
    )
    df.to_csv("inputs/amr_standin_data.csv", index=False, header=True)

if show_inputs:
    plt.show()
