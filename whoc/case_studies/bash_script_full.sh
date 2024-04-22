#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=48:00:00
<<<<<<< HEAD
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
=======
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
>>>>>>> 779f27e193dde4a170e547e68a38be4e5fc84703
#SBATCH --account=ssc

module purge
module load conda
module load openmpi
conda activate whoc

# srun python run_case_studies.py debug nompi 0 1 2 3 4 5 6 7
<<<<<<< HEAD
mpirun -np $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py nodebug mpi 0 1 2 3 4 5 6 7
=======
srun -n $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py nodebug mpi 0 1 2 3 4 5 6 7
>>>>>>> 779f27e193dde4a170e547e68a38be4e5fc84703
# srun python run_case_studies.py
