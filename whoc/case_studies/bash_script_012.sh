#!/bin/bash
#SBATCH --job-name=debug_floris_case_studies.py
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --account=ssc

module purge
module load conda
module load openmpi
conda activate whoc

mpirun -np $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py debug 0 1 2
# srun python run_case_studies.py
