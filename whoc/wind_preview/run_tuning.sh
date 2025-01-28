#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --account=ssc
#SBATCH --output=model_tuning_%j.out
#SBATCH --ntasks=104
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

# salloc --account=ssc --job-name=model_tuning  --ntasks=104 --cpus-per-task=1 --time=01:00:00 --partition=debug
# python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr"
ml mamba
ml PrgEnv-intel

mamba activate wind_forecasting

for i in $(seq 0 103); do
    srun --exclusive -n 1 python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "${1}_tuning" --model $1 &
done