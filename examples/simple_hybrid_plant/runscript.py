import matplotlib.pyplot as plt
from hercules.hercules_model import HerculesModel
from hercules.utilities import load_hercules_input
from hercules.utilities_examples import prepare_output_directory
from hycon.controllers import (
    BatteryController,
    HybridSupervisoryControllerGeneric,
    SolarPassthroughController,
    WindFarmPowerTrackingController,
)
from hycon.interfaces import HerculesInterface
from plot_outputs import plot_outputs

generate_output_plots = True

prepare_output_directory()

h_dict = load_hercules_input("hercules_input.yaml")

# User options
include_solar = True
include_battery = True

# Load all inputs, remove solar and/or battery as desired
if not include_solar:
    del h_dict["solar_farm"]
if not include_battery:
    del h_dict["battery"]

# Establish the Hercules model without a controller
hmodel = HerculesModel(h_dict)

# Establish controllers based on options
interface = HerculesInterface(hmodel.h_dict)
print("Setting up controller.")
wind_controller = WindFarmPowerTrackingController(interface, hmodel.h_dict, "wind_farm")
solar_controller = (
    SolarPassthroughController(interface, hmodel.h_dict, "solar_farm") if include_solar else None
)
battery_controller = (
    BatteryController(interface, hmodel.h_dict, "battery", {"k_batt": 0.1})
    if include_battery
    else None
)
component_controllers = [wind_controller]
if include_solar:
    component_controllers.append(solar_controller)
if include_battery:
    component_controllers.append(battery_controller)

# Set up main supervisory controller
controller = HybridSupervisoryControllerGeneric(
    interface,
    hmodel.h_dict,
    component_controllers=component_controllers,
)

hmodel.assign_controller(controller)

# Run the simulation
hmodel.run()

hmodel.logger.info("Simulation completed successfully")

if generate_output_plots:
    plot_outputs()
    plt.show()
