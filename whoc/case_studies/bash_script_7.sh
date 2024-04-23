#!/bin/bash
#SBATCH --job-name=7_floris_case_studies.py
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH -cpu-bind=none
##SBATCH --distribution=cyclic:cyclic
##SBATCH --cpu_bind=cores
#SBATCH --account=ssc

export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier
export MPICH_COLL_OPT_OFF=mpi_allreduce

module purge
module load cray-python
module load anaconda3
#module load openmpi
conda activate whoc

srun -n $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py nodebug mpi parallel 7
#srun -n $SLURM_NTASKS python -m mpi4py.futures run_case_studies.py nodebug mpi parallel 7
