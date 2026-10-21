#!/bin/bash
# Wrapper: wait for the current dense sweep (task-948) to finish, then run confined sweep.
# Usage: nohup bash run_confined_after_dense.sh &

PYTHON="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
WORKDIR="/home/fsp585/Desktop/FiberphotometryCode"
DENSE_SCRIPT="syllable_generation/analysis/language_models/pipelines/test_skeptical_audit_deep.py"
CONFINED_SCRIPT="syllable_generation/analysis/language_models/pipelines/8_run_confined_eta2_sweep.py"

echo "[$(date)] Waiting for any running python sweep to finish..."

# Wait until no test_skeptical_audit_deep.py is running
while pgrep -f "$DENSE_SCRIPT" > /dev/null 2>&1; do
    sleep 10
done

echo "[$(date)] Dense sweep finished. Starting confined manifold sweep..."
cd "$WORKDIR"
$PYTHON "$CONFINED_SCRIPT"
echo "[$(date)] Confined sweep complete."
