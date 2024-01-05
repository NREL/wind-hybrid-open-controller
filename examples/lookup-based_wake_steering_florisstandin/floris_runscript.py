import sys

from hercules.floris_standin import launch_floris

# Check that one command line argument was given
if len(sys.argv) < 2:
    raise Exception("Usage: python floris_runscript.py <amr_input_file>")

# # Get the first command line argument
# This is the name of the file to read
amr_input_file = sys.argv[1]
print(f"Running FLORIS standin with input file: {amr_input_file}")
if len(sys.argv) > 2:
    amr_standin_data_file = sys.argv[2]
    print(f"Using standin data for AMR-Wind from file: {amr_standin_data_file}")
else:
    amr_standin_data_file = None

launch_floris(amr_input_file, amr_standin_data_file)
