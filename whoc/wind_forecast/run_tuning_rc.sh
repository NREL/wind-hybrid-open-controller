#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --output=model_tuning_%j.out
##SBATCH --nodes=4
##SBATCH --ntasks-per-node=64
##SBATCH --time=12:00:00
##SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=01:00:00
#SBATCH --partition=atesting

#  srun -n 1 --exclusive python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr" &
# salloc --account=ssc --job-name=model_tuning  --ntasks=104 --cpus-per-task=1 --time=01:00:00 --partition=debug
# python tuning.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel.yaml --study_name "svr_tuning" --model "svr"

module purge
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list


# Configure how many workers to run per GPU
NUM_WORKERS_PER_CPU=1
# NUM_MODELS=$2

# Used to track process IDs for all workers
declare -a WORKER_PIDS=()

export MODEL_CONFIG="/projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_rc_flasc.yaml"
export DATA_CONFIG="/projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_rc_flasc.yaml"

# prepare training data first
echo "=== STARTING DATA PREPARATION ==="
date +"%Y-%m-%d %H:%M:%S"
module purge
module load miniforge
mamba activate wind_forecasting
python tuning.py \
            --model_config $MODEL_CONFIG \
            --data_config $DATA_CONFIG \
            --study_name "${1}_tuning" \
            --prepare_data \
            --model $1 \
            --seed ${WORKER_SEED}
wait
echo "=== DATA PREPARATION COMPLETE ==="

echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"
# for m in $(seq 0 $((${NUM_MODELS}-1))); do
for i in $(seq 0 $((${SLURM_NTASKS}-1))); do
    for j in $(seq 0 $((${NUM_WORKERS_PER_CPU}-1))); do
        # The restart flag should only be set for the very first worker (i=0, j=0)
        if [ $i -eq 0 ] && [ $j -eq 0 ]; then
            export RESTART_FLAG="--restart_tuning"
        else
            export RESTART_FLAG=""
        fi

        # Create a unique seed for each worker to ensure they explore different areas
        export WORKER_SEED=$((42 + i*10 + j))

        # Calculate worker index for logging
        WORKER_INDEX=$((i*NUM_WORKERS_PER_CPU + j))

        echo "Starting worker ${WORKER_INDEX} on GPU ${i} with seed ${WORKER_SEED}"
        
        # Launch worker with environment settings
        # CUDA_VISIBLE_DEVICES ensures each worker sees only one GPU
        # The worker ID (SLURM_PROCID) helps Optuna identify workers
        # srun --exclusive -n 1 --export=ALL,CUDA_VISIBLE_DEVICES=$i,SLURM_PROCID=${WORKER_INDEX},WANDB_DIR=${WANDB_DIR} \
        nohup bash -c "
        module purge
        module load miniforge
        mamba activate wind_forecasting
        python tuning.py \
            --model_config $MODEL_CONFIG \
            --data_config $DATA_CONFIG \
            --study_name "${1}_tuning" \
            --model $1 \
            --seed ${WORKER_SEED} \
            ${RESTART_FLAG}" &

        # Store the process ID
        WORKER_PIDS+=($!)

        # Add a small delay between starting workers on the same GPU
        # to avoid initialization conflicts
        sleep 2
    done
done
echo "Started ${#WORKER_PIDS[@]} worker processes for model ${m}"
echo "Process IDs: ${WORKER_PIDS[@]}"

# Wait for all workers to complete
wait
# done

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="
