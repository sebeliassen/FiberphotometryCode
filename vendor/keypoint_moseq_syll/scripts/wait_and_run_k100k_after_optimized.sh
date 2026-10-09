#!/bin/bash

# Target PID to wait for (the "optimized_flip_activity_k100k" run)
TARGET_PID=413812
LOG_DIR="logs"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu/bin/python"

echo "[$(date)] Waiting for PID $TARGET_PID (optimized_flip_activity_k100k) to finish..."

while ps -p $TARGET_PID > /dev/null; do
    sleep 60
done

echo "[$(date)] Process $TARGET_PID has finished."

# Safety delay to allow GPU to defragment and clear memory
COOLDOWN_MINS=2
echo "[$(date)] Waiting $COOLDOWN_MINS minutes for GPU cooldown..."
sleep 120

echo "[$(date)] Launching Flipped Double k100k Run (after optimized)..."

# Run command: model_name, seed, flipping_mode, subset_method, project_dir, pca_mode
# We use load_or_fit for PCA mode.
# Using 'augmentation' mode and 'activity' subsetting for the double project
nohup $PYTHON_BIN run_pipeline.py double_flip_activity_k100k 42 augmentation activity flipped_double_project load_or_fit > $LOG_DIR/double_flip_activity_k100k.log 2>&1 &

echo "[$(date)] Run launched with PID $!. Monitoring log: $LOG_DIR/double_flip_activity_k100k.log"
