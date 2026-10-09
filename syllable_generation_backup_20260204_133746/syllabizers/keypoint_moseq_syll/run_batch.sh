#!/bin/bash

# Auto-logging setup
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch_run_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to log file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to $LOG_FILE"


# Configuration
NUM_RUNS=10
BASE_SEED=42

echo "Starting batch run of $NUM_RUNS iterations..."
echo "Note: INIT_KAPPA is hardcoded in run_pipeline.py to 120000"

for ((i=1; i<=NUM_RUNS; i++))
do
    CURRENT_SEED=$((BASE_SEED + i))
    MODEL_NAME="amph_prod_k120k_run_${i}"
    
    echo "----------------------------------------------------------------"
    echo "Running Iteration $i / $NUM_RUNS"
    echo "Model Name: $MODEL_NAME"
    echo "Seed: $CURRENT_SEED"
    echo "----------------------------------------------------------------"

    # Running the python script with MODEL_NAME and SEED (Kappa is hardcoded in py)
    /home/fsp585/miniforge3/envs/kpms_gpu/bin/python run_pipeline.py "$MODEL_NAME" "$CURRENT_SEED"
    
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully."
    else
        echo "Run $i FAILED."
    fi
    
    echo "----------------------------------------------------------------"
    sleep 2 # Brief pause between runs
done

echo "Batch execution finished."
