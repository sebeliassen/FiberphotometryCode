#!/usr/bin/env bash
# =============================================================================
# launch_calibrated400_production.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
SCRIPT="${EXP_DIR}/02_training_runners/run_calibrated400_production_all_arms.sh"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/calibrated400_production_${TIMESTAMP}.log"
LATEST_LINK="${LOG_DIR}/calibrated400_production_latest.log"
PID_FILE="${LOG_DIR}/calibrated400_production.pid"

chmod +x "${SCRIPT}"

echo "============================================================================="
echo "🚀 DETACHING CALIBRATED PRODUCTION PIPELINE VIA NOHUP"
echo "Script:      ${SCRIPT}"
echo "Log File:    ${LOG_FILE}"
echo "Latest Link: ${LATEST_LINK}"
echo "============================================================================="

nohup "${SCRIPT}" < /dev/null > "${LOG_FILE}" 2>&1 &

PID=$!
disown -h "${PID}" 2>/dev/null || true
echo "${PID}" > "${PID_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Process detached with PID: ${PID}"
echo "To monitor live logs:"
echo "  tail -f ${LATEST_LINK}"
echo "============================================================================="
