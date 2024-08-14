#!/bin/bash
#SBATCH --job-name=greedy_wake_steering_amr_only_0
#SBATCH --time=01:00:00
##SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=52
#SBATCH --account=ssc

# A lot of modules and conda stuff
module purge
ml intel-oneapi-mpi
ml netcdf/4.9.2-intel-oneapi-mpi-intel
ml mamba
mamba activate whoc
#helics_broker -t zmq -f 2 --loglevel="debug" --local_port=32001 &
#python hercules_runscript.py hercules_input_000.yaml 0 &

# Set the helics port to use: 
export HELICS_PORT=32001
export NUM_TURBINES=9
export WIND_CASE_IDX=0

PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/build
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/helics/build/bin
# PATH=$PATH:/projects/ssc/ahenry/whoc/amr_controlled/greedy/wf_$WIND_CASE_IDX
input_file="amr_input_${NUM_TURBINES}_${WIND_CASE_IDX}.inp"
echo $input_file

cp /projects/ssc/ahenry/whoc/amr_controlled/greedy/hercules_input_000.yaml ./
cp /projects/ssc/ahenry/whoc/amr_controlled/greedy/hercules_runscript.py ./
cp /projects/ssc/ahenry/whoc/amr_controlled/$input_file ./

echo "Starting AMR-Wind job at: " $(date)

rm -rf post_processing/ outputs/
rm logamr loghercules
rm -rf chk* plt* *.out *.csv

# Set up the helics broker
helics_broker -t zmq -f 2 --loglevel="debug" --local_port=$HELICS_PORT &

# Need to set this to your hercules folder
# cd /home/pfleming/hercules/hercules
python hercules_runscript.py hercules_input_000.yaml $WIND_CASE_IDX >> loghercules 2>&1 & # Start the controller center and pass in input file

# Now go back to scratch folder and launch the job
srun --distribution=cyclic:cyclic --cpu_bind=cores amr_wind $input_file >> logamr 2>&1
#srun -n 104 s amr_wind $input_file >> logamr 2>&1
echo "Finished AMR-Wind job at: " $(date)
