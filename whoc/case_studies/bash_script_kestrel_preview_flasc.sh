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
export LD_LIBRARY_PATH=/projects/ssc/ahenry/conda/envs/wind_forecasting/lib
echo $SLURM_NTASKS
mpirun --mca opal_warn_on_missing_libcuda 0 \
       -np $SLURM_NTASKS python run_case_studies.py 15 -rs -rrs -st 3600 -ns 3 -p -m mpi \
       -sd /projects/ssc/ahenry/whoc/floris_case_studies \
       -mcnf /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_flasc.yaml \
       -wf scada
