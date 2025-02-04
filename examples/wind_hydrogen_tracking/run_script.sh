#!/bin/bash

# Locate the scripts folder
SCRIPTS_DIR="../../Hercules_WHOC_dev/hercules/scripts"

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
HELICS_PORT=32000
# source $SCRIPTS_DIR/get_helics_port.sh $first_port

# Clean up existing outputs
if [ -d outputs ]; then rm -r outputs; fi
mkdir -p outputs



# # Generate floris standin data
# python ../../hercules/tools/generate_amr_standin_data.py floris_standin_data.csv


# Set up the helics broker
echo "Connecting helics broker to port $HELICS_PORT"
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
#helics_broker -f 2 --consoleloglevel=trace --loglevel=debug --local_port=$HELICS_PORT >> loghelics &


python hercules_runscript.py inputs/hercules_input.yaml $HELICS_PORT >> outputs/loghercules_cl.log 2>&1 &
python floris_runscript.py inputs/amr_input.inp inputs/floris_standin_data_ws10_wd240.csv $HELICS_PORT >> outputs/logfloris_cl.log 2>&1

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
exit 0
