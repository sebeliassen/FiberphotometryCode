#!/usr/bin/env bash
# =============================================================================
# launch_cohort_calibrated_retrain_seed101.sh
# Launches the full cohort calibrated retrain run (Seed 101) in background.
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
RUNNER="${EXP_DIR}/02_training_runners/run_cohort_calibrated_retrain_seed101.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
LOG_FILE="${LOG_DIR}/cohort_calibrated_retrain_seed101_$(date +'%Y%m%d_%H%M%S').log"
LATEST_LINK="${LOG_DIR}/cohort_calibrated_retrain_seed101_latest.log"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

echo "============================================================================="
echo "🚀 LAUNCHING CALIBRATED TANDEM COHORT RETRAINING (SEED 101)"
echo "Runner:     ${RUNNER}"
echo "Log File:   ${LOG_FILE}"
echo "============================================================================="

nohup "${PYTHON_BIN}" -u "${RUNNER}" \
    --seed 101 \
    --num-ar-iters 50 \
    --num-slds-iters 200 \
    --save-every-n-iters 25 \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/cohort_calibrated_retrain_seed101.pid"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Process launched with PID: ${PID}"
echo "To monitor live logs:"
echo "  tail -f ${LATEST_LINK}"
echo "============================================================================="
