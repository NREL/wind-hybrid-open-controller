import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_inputs = False # Save generated inputs to csv files
show_inputs = False # Plot generated inputs for inspection

dt = 1.0 # time step in seconds
total_time = 600 # total time in seconds
time = np.linspace(dt, 600, num=round(total_time/dt))

# External signal for hydrogen reference: Step change
hydrogen_reference_value = 0.03 # in kg/dt
second_value = 0.04
third_value = 0.02
hydrogen_reference = hydrogen_reference_value*np.ones(600)

hydrogen_reference[150:] = second_value
hydrogen_reference[300:] = third_value
hydrogen_reference[450:] = second_value

fig1 = plt.figure(figsize=(8,4))
plt.plot(time, hydrogen_reference, 'b')
plt.ylabel("Full hydrogen\nreference [kg/s]")
plt.xlabel("Time")
plt.grid()


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
