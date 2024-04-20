#!/bin/bash
#SBATCH --job-name=greedy_wake_steering_amr_only
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --account=ssc
#SBATCH --mail-user=aoife.henry@colorado.edu
#SBATCH --mail-type=ALL

# A lot of modules and conda stuff
module purge

export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier
export MPICH_COLL_OPT_OFF=mpi_allreduce

export SPACK_MANAGER="/home/ahenry/toolboxes/spack-manager"
source $SPACK_MANAGER/start.sh
spack-start
quick-activate /home/ahenry/toolboxes/whoc_env
PATH=$PATH:/home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy
spack load amr-wind+helics+openfast

module load conda
conda activate whoc

echo "Starting AMR-Wind job at: " $(date)

# Set the helics port to use: 
export HELICS_PORT=32000
export NUM_TURBINES=25
export WIND_CASE_IDX=0

rm logamr loghercules

# Set up the helics broker
helics_broker -t zmq  -f 2 --loglevel="debug" --local_port=$HELICS_PORT &

# Need to set this to your hercules folder
# cd /home/pfleming/hercules/hercules
python3 hercules_runscript.py hercules_input_000.yaml $WIND_CASE_IDX >> loghercules 2>&1  & # Start the controller center and pass in input file

# Now go back to scratch folder and launch the job
srun -n 72 /home/ahenry/toolboxes/whoc_env/amr-wind/spack-build-bmx2pfy/amr_wind amr_input.inp >> logamr 2>&1
echo "Finished AMR-Wind job at: " $(date)
