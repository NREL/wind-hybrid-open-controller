# Example bash for running things locally
# I just run these one at a t time
from subprocess import call, Popen, run
from os import system

# A lot of modules and conda stuff
# system('conda activate whoc; export HELICS_PORT=32000; rm loghercules logfloris; helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT &; python hercules_runscript.py hercules_input_001.yaml >> loghercules 2>&1; python floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris 2>&1')
# run(". /Applications/anaconda3/etc/profile.d/conda.sh && conda activate whoc", shell=True)
# Popen("conda activate whoc", shell=True, executable="/bin/zsh")

# Set the helics port to use: 
# run("export HELICS_PORT=32000", shell=True)

# #make sure you use the same port number in the amr_input.inp and hercules_input_000.yaml files. 

# # Clear old log files for clarity
# run("rm loghercules logfloris", shell=True)

# # Set up the helics broker
Popen('rm loghercules logfloris && export HELICS_PORT=32000 && /Users/ahenry/Documents/toolboxes/Helics_3.1.0/bin/helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT &', shell=True)
#helics_broker -f 2 --consoleloglevel=trace --loglevel=debug --local_port=$HELICS_PORT >> loghelics &

# Need to set this to your hercules folder
# # cd /home/pfleming/hercules/hercules
# Popen(["python", "hercules_runscript.py", "hercules_input_001.yaml", ">>", "loghercules", "2>&1"], shell=True, close_fds=True) # Start the controller center and pass in input file
Popen(". /Applications/anaconda3/etc/profile.d/conda.sh && conda activate whoc && python hercules_runscript.py hercules_input_001.yaml >> loghercules 2>&1 &", shell=True)
Popen(". /Applications/anaconda3/etc/profile.d/conda.sh && conda activate whoc && python floris_runscript.py amr_input.inp amr_standin_data.csv >> logfloris 2>&1", shell=True)


# Popen(["python", "floris_runscript.py", "amr_input.inp", "amr_standin_data.csv", ">>", "logfloris", "2>&1"], shell=True, close_fds=True)
# Now go back to scratch folder and launch the job

# cd /scratch/pfleming/c2c/example_sim_02
# mpirun -n 72 /home/pfleming/amr-wind/build/amr_wind amr_input.inp >> logamr 
