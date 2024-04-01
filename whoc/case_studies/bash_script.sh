#!/bin/bash
#SBATCH --job-name=floris_case_studies.py
#SBATCH --time=6:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --account=ssc

module purge
module load conda
conda activate whoc

python3 run_case_studies.py