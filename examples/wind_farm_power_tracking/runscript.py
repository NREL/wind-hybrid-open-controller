import matplotlib.pyplot as plt
from hercules.hercules_model import HerculesModel
from hercules.utilities import load_hercules_input
from hercules.utilities_examples import prepare_output_directory
from hycon.controllers import (
    HybridSupervisoryControllerGeneric,
    WindFarmPowerDistributingController,
    WindFarmPowerTrackingController,
)
from hycon.interfaces import HerculesInterface
from plot_outputs import plot_outputs

generate_output_plots = True
ramp_rate_limit = 200  # kW/s, set to None for no ramp rate limit

prepare_output_directory()

### Open loop controller run
h_dict = load_hercules_input("hercules_input.yaml")
h_dict["output_file"] = "outputs/hercules_output_ol.h5"

hmodel = HerculesModel(h_dict)

interface = HerculesInterface(hmodel.h_dict)

print("Running open-loop controller...")
controller = HybridSupervisoryControllerGeneric(
    interface=interface,
    input_dict=hmodel.h_dict,
    component_controllers=[
        WindFarmPowerDistributingController(interface, hmodel.h_dict, "wind_farm", ramp_rate_limit)
    ],
)
hmodel.assign_controller(controller)

hmodel.run()
print("Finished running open-loop controller.")

### Closed loop controller run
h_dict = load_hercules_input("hercules_input.yaml")
h_dict["output_file"] = "outputs/hercules_output_cl.h5"

hmodel = HerculesModel(h_dict)

interface = HerculesInterface(hmodel.h_dict)

print("Running closed-loop controller...")
controller = HybridSupervisoryControllerGeneric(
    interface=interface,
    input_dict=hmodel.h_dict,
    component_controllers=[
        WindFarmPowerTrackingController(
            interface,
            hmodel.h_dict,
            "wind_farm",
            controller_parameters={"ramp_rate_limit": ramp_rate_limit},
        )
    ],
)
hmodel.assign_controller(controller)

hmodel.run()
print("Finished running closed-loop controller.")

if generate_output_plots:
    plot_outputs()
    plt.show()
