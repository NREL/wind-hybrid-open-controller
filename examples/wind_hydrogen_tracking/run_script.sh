#!/bin/bash

# Locate the scripts folder
SCRIPTS_DIR="../../../hercules/scripts"

# Kill any active helics jobs by calling the find_and_kill_helics script
# within the scripts folder
source $SCRIPTS_DIR/find_and_kill_helics.sh

# Determine the base path for Conda initialization
if [ -f "/home/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
    # Common path for Anaconda on Linux
    CONDA_PATH="/home/$USER/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/Users/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
    # Common path for Anaconda on macOS
    CONDA_PATH="/Users/$USER/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    # Alternative system-wide installation path
    CONDA_PATH="/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "opt/miniconda3/etc/profile.d/conda.sh" ]; then
    # Alternative system-wide installation path
    CONDA_PATH="/opt/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # If conda is in PATH, use the which command to find conda location
    CONDA_PATH=$(dirname "$(which conda)")/../etc/profile.d/conda.sh
else
    echo "Conda installation not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

# Kill any active helics jobs by calling the find_and_kill_helics script
# within the scripts folder
source $SCRIPTS_DIR/find_and_kill_helics.sh

# Run the activate CONDA script within the scripts folder
# to ensure the Hercules environment is active
source $SCRIPTS_DIR/activate_conda.sh

# Identify an available port for the HELICS broker.  This should
# be the first in a sequence of 10 available ports
# In case of comms trouble can be useful to change the first port
# to check for availability
first_port=32000
source $SCRIPTS_DIR/get_helics_port.sh $first_port

# Clean up existing outputs
if [ -d outputs ]; then rm -r outputs; fi
mkdir -p outputs

# Generate input time series for hydrogen reference and wind resource
echo "Generating input time series"
python generate_input_timeseries.py

# Set up the helics broker
echo "Connecting helics broker to port $HELICS_PORT"
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
python hercules_runscript.py inputs/hercules_input.yaml $HELICS_PORT >> outputs/loghercules_cl.log 2>&1 &
python floris_runscript.py inputs/amr_input.inp inputs/amr_standin_data.csv $HELICS_PORT >> outputs/logfloris_cl.log 2>&1

# Clean up helics output if there
# Search for a file that begins with the current year
# And ends with csv
# If the file exists, move to outputs folder
current_year=$(date +"%Y")
for file in ${current_year}*.csv; do
    if [ -f "$file" ]; then
        mv "$file" outputs/
    fi
done

# If everything is successful
echo "Finished running hercules"
echo "Plotting outputs"

python plot_output_data.py

exit 0
