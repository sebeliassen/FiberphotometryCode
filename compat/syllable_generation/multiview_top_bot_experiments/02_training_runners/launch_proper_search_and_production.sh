#!/usr/bin/env bash
# =============================================================================
# launch_proper_search_and_production.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
SCRIPT="${EXP_DIR}/02_training_runners/run_proper_search_and_production.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/proper_search_and_production_${TIMESTAMP}.log"
LATEST_LINK="${LOG_DIR}/proper_search_and_production_latest.log"
PID_FILE="${LOG_DIR}/proper_search_and_production.pid"

chmod +x "${SCRIPT}"

echo "============================================================================="
echo "🚀 LAUNCHING EXTENDED 21KP_STATIC SCAN & PRODUCTION PIPELINE (NOHUP)"
echo "Script:      ${SCRIPT}"
echo "Python:      ${PYTHON_BIN}"
echo "Log File:    ${LOG_FILE}"
echo "Latest Link: ${LATEST_LINK}"
echo "============================================================================="

"${PYTHON_BIN}" -u "${SCRIPT}" > "${LOG_FILE}" 2>&1
