# Example bash for running things locally
# I just run these one at a t time

# A lot of modules and conda stuff
conda activate hercules

# Set the helics port to use: 
export HELICS_PORT=32000

#make sure you use the same port number in the amr_input.inp and hercules_input_000.yaml files. 

# Clear old log files for clarity
rm loghercules_ol logfloris_ol loghercules_cl logfloris_cl

# Set up the helics broker and run the open-loop control simulation
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
python3 hercules_runscript_OLcontrol.py hercules_input_000.yaml >> loghercules_ol 2>&1 &
python3 floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris_ol 2>&1

# Wait for the open-loop control simulation to finish and then run the closed-loop simulation
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
python3 hercules_runscript_CLcontrol.py hercules_input_000.yaml >> loghercules_cl 2>&1 &
python3 floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris_cl 2>&1
