#!/bin/bash
#SBATCH --job-name=preview_floris_case_studies.py
#SBATCH --time=36:00:00
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=104
#SBATCH --account=ssc

module purge
module load mamba
mamba activate wind_forecasting
module load intel
export LD_LIBRARY_PATH=/home/ahenry/.conda-envs/whoc/lib
echo $SLURM_NTASKS
mpirun -np $SLURM_NTASKS python run_case_studies.py 15 -rs -rrs -st 3600 -ns 3 -p -m mpi \
        -sd /projects/ssc/ahenry/whoc/floris_case_studies \
        -mcnf wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml \
        -wf scada
#mpirun -np $SLURM_NTASKS python run_case_studies.py 0 1 2 3 4 5 6 7 8 9 -rs -st 3600 -ns 6 -p -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
#mpirun -np $SLURM_NTASKS python run_case_studies.py 5 6 -p -rs -st 3600 -ns 6 -m mpi -sd /projects/ssc/ahenry/whoc/floris_case_studies
