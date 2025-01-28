#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --account=ssc
#SBATCH --output=model_tuning_%j.out
#SBATCH --array=0-103
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

srun python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "${1}_tuning" --model $1