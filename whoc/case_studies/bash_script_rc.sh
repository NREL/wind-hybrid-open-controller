#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
##SBATCH --time=01:00:00
##SBATCH --partition=debug


# load modules
module purge
module load mambaforge
mamba activate whoc
#module load gcc/10.3 openmpi
# module load openmpi/4.1.4
module load intel/2022.1.2 impi/2021.5.0
export LD_LIBRARY_PATH=/projects/aohe7145/software/anaconda/envs/whoc/lib

#conda activate whoc
echo $SLURMP_NTASKS
#export SLURM_EXPORT_ENV=ALL
#python run_case_studies.py debug cf parallel 0 1 2 3 4 5 6 7
mpirun -np $SLURM_NTASKS python run_case_studies.py 0 9 -rs -st 3600 -ns 1 -p -m mpi -sd /projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies
