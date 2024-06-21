import numpy as np
from memory_profiler import profile
from floris import FlorisModel, TimeSeries
 
@profile
def run_floris():
  # Load the Floris model
    fmodel = FlorisModel("/Users/ahenry/Documents/toolboxes/floris/examples/inputs/gch.yaml")

    N = 100000
    np.random.seed(0)

    # Set up inflow wind conditions
    time_series = TimeSeries(
        wind_directions=270 + 30 * np.random.randn(N),
        wind_speeds=8 + 2 * np.random.randn(N),
        turbulence_intensities=0.06 + 0.02 * np.random.randn(N),
    )

    # Set the wind conditions for the model
    fmodel.set(wind_data=time_series)

    # Run the calculations
    fmodel.run()
    fmodel.run()

    # Extract turbine and farm powers
    turbine_powers = fmodel.get_turbine_powers() / 1000.0
    farm_power = fmodel.get_farm_power() / 1000.0
    print(turbine_powers.shape)
    print(farm_power.shape)

if __name__ == "__main__":
    run_floris()