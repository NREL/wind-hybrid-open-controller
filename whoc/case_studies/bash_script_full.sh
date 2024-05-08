#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=104
#SBATCH --account=ssc

module purge
ml restore system
ml mamba
conda activate whoc
#env MPICC=cc pip install dask-mpi mpi4py

rm -rf /projects/ssc/ahenry/whoc/floris_case_studies
export MAXWORKERS=`echo $(($SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES))`
python initialize_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
srun -n $MAXWORKERS python simulate_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
python process_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
