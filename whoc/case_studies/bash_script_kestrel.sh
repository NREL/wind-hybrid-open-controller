#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

# load modules
module purge
module load mamba
mamba activate whoc
#module load intel/2022.1.2 impi/2021.5.0
module load intel
export LD_LIBRARY_PATH=/home/ahenry/.conda-envs/whoc/lib
echo $SLURMP_NTASKS
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 9 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
mpirun -np $SLURM_NTASKS python run_case_studies.py 1 2 3 4 5 6 7 8 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
