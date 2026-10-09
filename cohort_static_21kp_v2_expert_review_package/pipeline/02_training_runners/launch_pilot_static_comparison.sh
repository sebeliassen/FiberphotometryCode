#!/usr/bin/env bash
# =============================================================================
# launch_pilot_static_comparison.sh
# Launches the pilot static comparison (Arm A: 21kp, Arm B: 23kp) in background via nohup.
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
RUNNER="${EXP_DIR}/02_training_runners/run_pilot_static_comparison.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
LOG_FILE="${LOG_DIR}/pilot_static_comparison_$(date +'%Y%m%d_%H%M%S').log"
LATEST_LINK="${LOG_DIR}/pilot_static_comparison_latest.log"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

ARM="${1:-both}"

echo "============================================================================="
echo "🚀 LAUNCHING PILOT STATIC COMPARISON TRAINING RUN"
echo "Arm(s):     ${ARM}"
echo "Runner:     ${RUNNER}"
echo "Log File:   ${LOG_FILE}"
echo "============================================================================="

nohup "${PYTHON_BIN}" -u "${RUNNER}" \
    --arm "${ARM}" \
    --num-ar-iters 50 \
    --num-slds-iters 100 \
    --init-kappa 1.0e6 \
    --decrease-kappa-factor 10.0 \
    --seed 404 \
    > "${LOG_FILE}" 2>&1 &

PID=$!
disown -h "${PID}"
echo "${PID}" > "${LOG_DIR}/pilot_static_comparison.pid"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Process launched with PID: ${PID}"
echo "To monitor live logs:"
echo "  tail -f ${LATEST_LINK}"
echo "============================================================================="
