import sys

from hercules.amr_wind_standin import launch_amr_wind_standin

# Check that one command line argument was given
if len(sys.argv) != 2:
    raise Exception("Usage: python hercules_runscript_amr_standin.py <amr_input_file>")

# # Get the first command line argument
# This is the name of the file to read
amr_input_file = sys.argv[1]
print(f"Running AMR-Wind standin with input file: {amr_input_file}")


launch_amr_wind_standin(amr_input_file)
