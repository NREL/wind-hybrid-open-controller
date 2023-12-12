import sys
from hercules.dummy_amr_wind import launch_dummy_amr_wind

# Check that one command line argument was given
if len(sys.argv) != 2:
    raise Exception("Usage: python emu_runscript_dummy_amr.py <amr_input_file>")
                    
# # Get the first command line argument
# This is the name of the file to read
amr_input_file = sys.argv[1]
print(f"Running AMR-Wind dummy with input file: {amr_input_file}")


launch_dummy_amr_wind(amr_input_file)