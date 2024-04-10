#!/bin/bash
#SBATCH --job-name=floris_case_studies.py
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --account=ssc

module purge
module load conda
module load openmpi/4.0.0
conda activate whoc

mpirun -np $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py
# srun python run_case_studies.py