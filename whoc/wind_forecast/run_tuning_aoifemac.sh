#!/bin/zsh

export NTASKS_PER_TUNER=4
export CPU_COUNT=12
NTUNERS=$((CPU_COUNT / NTASKS_PER_TUNER))

# Used to track process IDs for all workers
declare -a WORKER_PIDS=()

export MODEL_CONFIG="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml"
export DATA_CONFIG="/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml"
export TMPDIR="/tmp/scratch/${SLURM_JOB_ID}/"
export STUDY_NAME="${1}_${2}_tuning"

# prepare training data first
echo "=== STARTING DATA PREPARATION ==="
date +"%Y-%m-%d %H:%M:%S"
# conda init
source activate base
conda activate wind_forecasting_env
python tuning.py \
            --model_config $MODEL_CONFIG \
            --data_config $DATA_CONFIG \
            --study_name $STUDY_NAME \
            --initialize \
            --model $1 \
            --seed 0
wait
echo "=== DATA PREPARATION COMPLETE ==="

echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"
# for m in $(seq 0 $((${NUM_MODELS}-1))); do
for i in $(seq 0 $((${NTUNERS}-1))); do
    # for j in $(seq 0 $((${NUM_WORKERS_PER_CPU}-1))); do
        # The restart flag should only be set for the very first worker (i=0, j=0)
        if [ $i -eq 0 ]; then
            export RESTART_FLAG="--restart_tuning"
        else
            export RESTART_FLAG=""
        fi

        # Create a unique seed for each worker to ensure they explore different areas
        export WORKER_SEED=$((42 + i*10))

        # Calculate worker index for logging
        WORKER_INDEX=$((i))

        echo "Starting worker ${WORKER_INDEX} on CPU ${i} with seed ${WORKER_SEED}"
        
        nohup bash -c "
        conda activate wind_forecasting
        python tuning.py \
            --model_config $MODEL_CONFIG \
            --data_config $DATA_CONFIG \
            --study_name $STUDY_NAME \
            --model $1 \
            --multiprocessor cf \
            --seed ${WORKER_SEED} \
            ${RESTART_FLAG}" &

        # Store the process ID
        WORKER_PIDS+=($!)

        # Add a small delay between starting workers on the same GPU
        # to avoid initialization conflicts
        sleep 2
    # done
done
echo "Started ${#WORKER_PIDS[@]} worker processes for model ${m}"
echo "Process IDs: ${WORKER_PIDS[@]}"

# Wait for all workers to complete
wait
# done

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="
