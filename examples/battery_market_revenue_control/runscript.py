# Requires Hercules v2

import matplotlib.pyplot as plt
import pandas as pd
from hercules.grid.grid_utilities import (
    generate_locational_marginal_price_dataframe_from_gridstatus,
)
from hercules.hercules_model import HerculesModel
from hercules.utilities_examples import prepare_output_directory
from hycon.controllers import BatteryPriceSOCController, HybridSupervisoryControllerGeneric
from hycon.interfaces import HerculesInterface
from plot_outputs import plot_outputs

generate_output_plots = True

prepare_output_directory()

# Generate the LMP data needed for the simulation
df_lmp = generate_locational_marginal_price_dataframe_from_gridstatus(
    pd.read_csv("../example_inputs/lmp_da.csv"), pd.read_csv("../example_inputs/lmp_rt.csv")
)
df_lmp.to_csv("lmp_data.csv", index=False)

# Load the input file and establish the Hercules model
hmodel = HerculesModel("hercules_input.yaml")

# Establish the interface and controller, assign to the Hercules model
interface = HerculesInterface(hmodel.h_dict)
controller = HybridSupervisoryControllerGeneric(
    interface=HerculesInterface(hmodel.h_dict),
    input_dict=hmodel.h_dict,
    component_controllers=[
        BatteryPriceSOCController(interface=interface, input_dict=hmodel.h_dict, cname="battery")
    ],
)
hmodel.assign_controller(controller)

# Run the simulation
hmodel.run()

hmodel.logger.info("Process completed successfully")

if generate_output_plots:
    plot_outputs()
    plt.show()
