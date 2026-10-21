#!/usr/bin/env bash
# ==============================================================================
# KPMS Studio Launch Script
# Uses the requested kpms_gpu_cuml environment
# ==============================================================================

CONDA_ENV="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
STUDIO_DIR="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/kpms_studio"

cd "$STUDIO_DIR" || exit 1
"$CONDA_ENV" "$STUDIO_DIR/launch_gui.py" "$@"
