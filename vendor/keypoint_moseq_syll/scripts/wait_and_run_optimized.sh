#!/bin/bash
# scripts/wait_and_run_optimized.sh
# This script waits for the current Keypoint MoSeq training process to finish
# and then automatically starts the Optimized Flipping run for hemisphere alignment.

PID=1720894
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu/bin/python"
LOG_DIR="logs"

mkdir -p $LOG_DIR

echo "[$(date)] Chained Run: Waiting for PID $PID..."

# Wait loop
while ps -p $PID > /dev/null; do
    sleep 60
done

echo "[$(date)] Process $PID has finished."

# Safety delay to allow GPU to defragment and clear memory
COOLDOWN_MINS=10
echo "[$(date)] Waiting $COOLDOWN_MINS minutes for GPU cooldown..."
sleep 600 

echo "[$(date)] Launching Optimized Flipping Run (Hemisphere Alignment) in NEW project..."

# Setup new project folder
NEW_PROJECT="optimized_flip_project"
mkdir -p $NEW_PROJECT
cp flipped_double_project/config.yml $NEW_PROJECT/

# Run command: model_name, seed, flipping_mode, subset_method, project_dir, pca_mode
# Using 'fit' for PCA mode to ensure a fresh, aligned PCA is created.
nohup $PYTHON_BIN run_pipeline.py optimized_flip_activity_k20k 42 optimized activity $NEW_PROJECT fit > $LOG_DIR/optimized_run.log 2>&1 &

echo "[$(date)] Run launched with PID $!. Monitoring log: $LOG_DIR/optimized_run.log"
