# Example bash for running things locally
# I just run these one at a t time

# A lot of modules and conda stuff
conda activate whoc

# Set the helics port to use: 
export HELICS_PORT=32000
export NUM_TURBINES=25
export WIND_CASE_IDX=0

#make sure you use the same port number in the amr_input.inp and hercules_input_000.yaml files. 

# Clear old log files for clarity
rm loghercules logfloris

# Set up the helics broker
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 
#helics_broker -f 2 --consoleloglevel=trace --loglevel=debug --local_port=$HELICS_PORT >> loghelics &

# Need to set this to your hercules folder
# cd /home/pfleming/hercules/hercules
python hercules_runscript.py hercules_input_001.yaml $WIND_CASE_IDX >> loghercules 2>&1 & # Start the controller center and pass in input file

python floris_runscript.py amr_input_$NUM_TURBINES.inp amr_standin_data_$WIND_CASE_IDX.csv >> logfloris 2>&1

# Now go back to scratch folder and launch the job

# cd /scratch/pfleming/c2c/example_sim_02
# mpirun -n 72 /home/pfleming/amr-wind/build/amr_wind amr_input.inp >> logamr 
