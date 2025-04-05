#!/bin/bash
#SBATCH --job-name=full_floris_case_studies.py
##SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --qos=long
##SBATCH --time=01:00:00
##SBATCH --partition=atesting

# load modules
module purge
module load miniforge 
mamba activate wind_forecasting
#module load gcc/10.3 openmpi
# module load openmpi/4.1.4
module load intel/2022.1.2 impi/2021.5.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/stth7454/software/anaconda/envs/whoc/lib
module load intel impi


echo $SLURM_NTASKS
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 120 -ns 1 -p -m mpi -sd /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/floris_case_studies -wcnf /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf floris
mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/floris_case_studies -wcnf /projects/aohe7145/toolboxes/whoc_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf floris 
