#!/bin/bash

# Configuration
NTFY_TOPIC="woody_videos_seb"
ENV_PYTHON="/home/fsp585/miniforge3/envs/kpms_gpu/bin/python"

# 1. Notify start
echo "Sending start notification..."
curl -d "Starting optimized_flip_activity_k100k training run now." "ntfy.sh/$NTFY_TOPIC" || true

# 2. Run pipeline
echo "Starting KPMS pipeline..."
# Parameters:
# --model_name "optimized_flip_activity_k100k"
# --flipping_mode "optimized"
# --subset_method "activity"
# --subset_fraction 0.5
# --init_kappa 100000
# --downsample_rate 2

$ENV_PYTHON run_pipeline.py \
    "optimized_flip_activity_k100k" \
    42 \
    "optimized" \
    "activity" \
    "optimized_flip_project"

# 3. Notify completion
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Sending success notification..."
    curl -d "SUCCESS: optimized_flip_activity_k100k training run completed." "ntfy.sh/$NTFY_TOPIC" || true
else
    echo "Sending failure notification..."
    curl -d "FAILED: optimized_flip_activity_k100k training run exited with code $EXIT_CODE." "ntfy.sh/$NTFY_TOPIC" || true
fi

echo "Done."
