#!/bin/bash
#SBATCH --job-name=initialize_floris_case_studies.py
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

module purge
ml restore system
ml mamba
conda activate whoc
#module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
#MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

export MAXWORKERS=`echo $(($SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES))`
srun python run_case_studies.py nodebug cf parallel 0 2 10

