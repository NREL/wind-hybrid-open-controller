#!/bin/bash

# Ensure hercules conda or venv is activated before running this script

# Run this script via the command: 
#    bash batch_script.sh
#    ./batch_script.sh

# Set the helics port to use: 
#make sure you use the same port number in the amr_input.inp and hercules_input_000.yaml files. 
export HELICS_PORT=32000

# Delete the logs within the outputs folder (if the output folder exists)
if [ -d "outputs" ]; then
  rm -f outputs/*.log
fi

# Create the outputs folder
mkdir -p outputs

# Set up the helics broker
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
# For debugging add --consoleloglevel=trace

# Start the controller center and pass in input file
echo "Starting hercules"
# python3 hercules_runscript.py hercules_input_000.yaml >> outputs/loghercules.log 2>&1 &
python3 hercules_runscript.py hercules_controller_input_000.yaml >> outputs/loghercules.log 2>&1 &

# Start the floris standin
echo "Starting floris"
python3 floris_runscript.py inputs/amr_input.inp inputs/floris_standin_data_fixedwd.csv >> outputs/logfloris.log 2>&1

# If everything is successful
echo "Finished running hercules"
exit 0



