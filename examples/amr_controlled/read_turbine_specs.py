from hercules.utilities import load_yaml

turbine_path = "/Users/ahenry/Documents/toolboxes/floris/floris/turbine_library/nrel_5MW.yaml"

turbine_dict = load_yaml(turbine_path)

wind_speeds = turbine_dict["power_thrust_table"]["wind_speed"]
thrust_coeff = turbine_dict["power_thrust_table"]["thrust_coefficient"]
# rpm
print(" ".join([str(ws) for ws in wind_speeds]))
print(" ".join([str(tc) for tc in thrust_coeff]))