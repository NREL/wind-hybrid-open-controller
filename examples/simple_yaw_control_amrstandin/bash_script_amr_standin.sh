# Example bash for running things locally
# I just run these one at a t time

# A lot of modules and conda stuff
conda activate hercules

export HELICS_PORT=32000

# Set up the helics broker
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT & 

# Need to set this to your emu_python folder
# cd /home/pfleming/emu_python/emu_python
python hercules_runscript.py hercules_input_shortsim.yaml >> loghercules 2>&1 & # Start the controller center and pass in input file


python3 hercules_runscript_amr_standin.py amr_input.inp >> logstandin 2>&1
# Now go back to scratch folder and launch the job

# cd /scratch/pfleming/c2c/example_sim_02
# mpirun -n 72 /home/pfleming/amr-wind/build/amr_wind amr_input.inp >> logamr 