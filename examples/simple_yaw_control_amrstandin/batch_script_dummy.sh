# Example bash for running things locally
# I just run these one at a t time

# A lot of modules and conda stuff
conda activate hercules

# Set up the helics broker
helics_broker -t zmq  -f 2 --loglevel="debug" & 

# Need to set this to your emu_python folder
# cd /home/pfleming/emu_python/emu_python
python hercules_runscript.py hercules_input_shortsim.yaml >> loghercules 2>&1 & # Start the controller center and pass in input file


python hercules_runscript_dummy_amr.py amr_input.inp >> logdummy 2>&1
# Now go back to scratch folder and launch the job

# cd /scratch/pfleming/c2c/example_sim_02
# mpirun -n 72 /home/pfleming/amr-wind/build/amr_wind amr_input.inp >> logamr 