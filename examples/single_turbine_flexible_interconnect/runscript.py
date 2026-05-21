import matplotlib.pyplot as plt
import pandas as pd
from hercules.hercules_model import HerculesModel
from hercules.utilities import load_hercules_input
from hercules.utilities_examples import prepare_output_directory
from hycon.controllers import (
    BatteryController,
    HybridSupervisoryControllerGeneric,
    WindFarmPowerTrackingController,
)
from hycon.interfaces import HerculesInterface
from plot_outputs import plot_outputs

generate_output_plots = True

prepare_output_directory()

# Generate the dynamic interconnect limit over a 24-hour period
time_hours = pd.date_range(start="2018-05-10 00:00:00", periods=25, freq="1H", tz="UTC")
df = pd.DataFrame(
    {
        "time_utc": time_hours,
        "plant_power_reference": [
            1374.7,
            1366.1,
            1366.1,
            1376.4,
            1376.4,
            1390.9,
            1390.9,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1401.2,
            1406.9,
            1406.9,
        ],
    }
)
df2 = df.copy(deep=True)
df2["time_utc"] = df2["time_utc"] + pd.Timedelta(seconds=3599)
df = pd.merge(df, df2, how="outer").sort_values("time_utc").reset_index(drop=True)
df.to_csv("flexible_interconnect_limit.csv", index=False)

### Run base case
h_dict = load_hercules_input("hercules_input.yaml")
del h_dict["battery"]
del h_dict["external_data"]  # Remove wind reference
h_dict["output_file"] = "outputs/hercules_output_baseline.h5"

hmodel = HerculesModel(h_dict)
interface = HerculesInterface(hmodel.h_dict)
wind_controller = WindFarmPowerTrackingController(interface, "distributed_wind")
controller = HybridSupervisoryControllerGeneric(
    interface=interface,
    cname="supervisory_controller",
    controller_parameters={"component_controllers": [wind_controller]},
)
hmodel.assign_controller(controller)

print("Running baseline case.")
hmodel.run()
print("Finished running baseline case.")

### Run case with wind only reference tracking
h_dict = load_hercules_input("hercules_input.yaml")
del h_dict["battery"]
h_dict["output_file"] = "outputs/hercules_output_wind_only.h5"

hmodel = HerculesModel(h_dict)
interface = HerculesInterface(hmodel.h_dict)
controller = HybridSupervisoryControllerGeneric(
    interface=interface,
    cname="supervisory_controller",
    controller_parameters={"component_controllers": [wind_controller]},
)
hmodel.assign_controller(controller)

print("Running wind-only reference tracking case.")
hmodel.run()
print("Finished running wind-only reference tracking case.")

### Run case with battery included
h_dict = load_hercules_input("hercules_input.yaml")
h_dict["output_file"] = "outputs/hercules_output_with_battery.h5"
hmodel = HerculesModel(h_dict)
interface = HerculesInterface(hmodel.h_dict)
controller = HybridSupervisoryControllerGeneric(
    interface=interface,
    cname="supervisory_controller",
    controller_parameters={
        "component_controllers": [
            wind_controller,
            BatteryController(interface, "battery", {"k_batt": 0.1}),
        ]
    },
)
hmodel.assign_controller(controller)

print("Running case with battery included.")
hmodel.run()
print("Finished running case with battery included.")

### Plot results
if generate_output_plots:
    plot_outputs()
    plt.show()
