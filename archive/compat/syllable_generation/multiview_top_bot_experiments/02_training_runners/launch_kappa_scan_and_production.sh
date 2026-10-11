#!/usr/bin/env bash
# =============================================================================
# launch_kappa_scan_and_production.sh
# Launches the automated kappa scan & production pipeline in background via nohup.
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
RUNNER="${EXP_DIR}/02_training_runners/run_kappa_scan_and_production_pipeline.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/kappa_scan_and_production_${TIMESTAMP}.log"
LATEST_LINK="${LOG_DIR}/kappa_scan_and_production_latest.log"
PID_FILE="${LOG_DIR}/kappa_scan_and_production.pid"

echo "============================================================================="
echo "🚀 LAUNCHING KAPPA SCAN & PRODUCTION BENCHMARK (NOHUP)"
echo "Runner:      ${RUNNER}"
echo "Python:      ${PYTHON_BIN}"
echo "Log File:    ${LOG_FILE}"
echo "Latest Link: ${LATEST_LINK}"
echo "============================================================================="

chmod +x "${RUNNER}"

nohup "${PYTHON_BIN}" -u "${RUNNER}" > "${LOG_FILE}" 2>&1 &

PID=$!
disown -h "${PID}"
echo "${PID}" > "${PID_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Process launched with PID: ${PID}"
echo "To monitor live logs:"
echo "  tail -f ${LATEST_LINK}"
echo "============================================================================="
