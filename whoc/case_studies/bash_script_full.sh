#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --partition=debug
#SBATCH --account=ssc

# load modules
ml restore && ml mamba

# define number of workers
export MAXWORKERS=`echo $(($SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES))`

conda activate whoc
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py


#echo "running initialize_case_studies.py"
#python initialize_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
#echo "running simulate_case_studies.py"
#srun -n $MAXWORKERS python simulate_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
#echo "running process_case_studies.py"
#python process_case_studies.py nodebug mpi parallel 0 1 2 3 4 5 6 7
#srun -n $MAXWORKERS python run_case_studies.py debug mpi parallel 0 1 2 3 4 5 6 7
srun -n $MAXWORKERS  python run_case_studies.py debug mpi noparallel 0
