#!/bin/bash

# Configuration
NTFY_TOPIC="woody_videos_seb"
ENV_PYTHON="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

# Change directory to script location
cd "$(dirname "$0")"

export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Redirect stdout and stderr to log file automatically
exec >> final_karen_model_k130k.log 2>&1

# 1. Notify start
echo "Sending start notification..."
curl -d "Starting final_karen_model_k130k (kappa=130k) training run now." "https://ntfy.sh/$NTFY_TOPIC" || true

# 2. Run pipeline
echo "Starting KPMS pipeline..."
$ENV_PYTHON -u run_pipeline.py "final_karen_model_k130k" 42

# 3. Notify completion
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Sending success notification..."
    curl -d "SUCCESS: final_karen_model_k130k training run completed." "https://ntfy.sh/$NTFY_TOPIC" || true
else
    echo "Sending failure notification..."
    curl -d "FAILED: final_karen_model_k130k training run exited with code $EXIT_CODE." "https://ntfy.sh/$NTFY_TOPIC" || true
fi

echo "Done."
