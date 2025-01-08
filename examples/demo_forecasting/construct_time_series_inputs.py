import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create a demonstration wind speed forecast based on the actual wind speed
# that will occur with some noise added.

np.random.seed(0)

df_truth = pd.read_csv("inputs/floris_standin_data_fixedwd.csv", index_col=0)

cov = np.diag(np.linspace(0.2, 1.0, 5))
r = 0.8
for i in range(5):
    for j in range(5):
       cov[i,j] = cov[i,j] + r**np.abs(i-j)

ws_pred = np.zeros((len(df_truth), 5))

for t in range(len(df_truth)):
    ws_forecast = (
        df_truth.loc[t, "amr_wind_speed"]
        + np.random.multivariate_normal(np.zeros(5), cov)
    )
    ws_pred[t, :] = ws_forecast

fig, ax = plt.subplots(1,2,sharey=True)

# Plot all predictions for all time steps
for t in range(len(df_truth)-5):
   ax[0].plot(df_truth.time.loc[t:t+4], ws_pred[t,:], color="lightgray", alpha=0.5)
ax[0].plot(df_truth.time, df_truth.amr_wind_speed, color="red", linestyle="--")
ax[0].set_ylabel("Wind Speed [m/s]")
ax[0].set_xlabel("Time [s]")
ax[0].set_xlim([0, 600])

# Plot only the prediction for the first time step, as well as the true wind speed
ax[1].plot(df_truth.time.loc[0:4], ws_pred[0,:], color="lightgray")
ax[1].plot(df_truth.time, df_truth.amr_wind_speed, color="red", linestyle="--")
ax[1].set_xlim([0, 2])
ax[1].set_xlabel("Time [s]")

# Save off the data (mean and standard dev)
df_pred = pd.DataFrame(ws_pred, columns=[f"forecast_ws_mean_{i}" for i in range(5)])
df_pred[[f"forecast_ws_std_{i}" for i in range(5)]] = np.sqrt(np.diag(cov))
df_pred["time"] = df_truth.time

# Add a plant_power_reference so that simulation runs through
df_pred["plant_power_reference"] = 75e3 # 75 MW

df_pred.to_csv("inputs/hercules_time_series_input.csv")

plt.show()
