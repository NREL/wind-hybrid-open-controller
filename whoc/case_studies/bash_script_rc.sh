#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
##SBATCH --mem=0
##SBATCH --nodes=4
##SBATCH --ntasks-per-node=64
##SBATCH --time=24:00:00
##SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=01:00:00
#SBATCH --partition=atesting

# load modules
module purge
module load miniforge 
conda activate whoc
module load intel impi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/aohe7145/software/anaconda/envs/whoc/lib

echo $SLURM_NTASKS
mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 120 -ns 1 -p -m mpi -sd /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/floris_case_studies -wcnf /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf floris
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/aohe7145/toolboxes/wind-hybrid-open-controller/whoc/floris_case_studies
