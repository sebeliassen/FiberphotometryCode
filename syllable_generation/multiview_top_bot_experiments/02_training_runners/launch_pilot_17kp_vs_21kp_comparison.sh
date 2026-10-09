#!/usr/bin/env bash
# =============================================================================
# launch_pilot_17kp_vs_21kp_comparison.sh
# Launches the controlled 17kp vs 21kp comparison in background via nohup.
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
RUNNER="${EXP_DIR}/02_training_runners/run_pilot_17kp_vs_21kp_comparison.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

NUM_AR="${1:-50}"
NUM_SLDS="${2:-150}"
SUFFIX="${3:-cohort}"
SAVE_N="${4:-25}"
ARMS="${5:-17kp 21kp_static 21kp_dynamic}"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/pilot_17kp_vs_21kp_${SUFFIX}_${TIMESTAMP}.log"
LATEST_LINK="${LOG_DIR}/pilot_17kp_vs_21kp_latest.log"
PID_FILE="${LOG_DIR}/pilot_17kp_vs_21kp.pid"

echo "============================================================================="
echo "🚀 LAUNCHING 17KP VS 21KP BENCHMARK"
echo "AR Iterations:   ${NUM_AR}"
echo "SLDS Iterations: ${NUM_SLDS}"
echo "Save Interval:   ${SAVE_N}"
echo "Model Suffix:    ${SUFFIX}"
echo "Arms:            ${ARMS}"
echo "Log File:        ${LOG_FILE}"
echo "============================================================================="

nohup "${PYTHON_BIN}" -u "${RUNNER}" \
    --arms ${ARMS} \
    --num-ar-iters "${NUM_AR}" \
    --num-slds-iters "${NUM_SLDS}" \
    --init-kappa 1.0e6 \
    --slds-kappa 1.5e12 \
    --save-every-n "${SAVE_N}" \
    --model-suffix "${SUFFIX}" \
    --seed 404 \
    > "${LOG_FILE}" 2>&1 &

PID=$!
disown -h "${PID}"
echo "${PID}" > "${PID_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Process launched with PID: ${PID}"
echo "To monitor live logs:"
echo "  tail -f ${LATEST_LINK}"
echo "============================================================================="
