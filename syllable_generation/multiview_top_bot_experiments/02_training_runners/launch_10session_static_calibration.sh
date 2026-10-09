#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/static_10session_calibration_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/static_10session_calibration.pid"
LATEST_LINK="${LOG_DIR}/static_10session_calibration_latest.log"

PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
RUNNER="${SCRIPT_DIR}/run_10session_static_calibration.py"

echo "================================================================================"
echo "🚀 LAUNCHING 10-SESSION STATIC 21KP KAPPA CALIBRATION & DORSAL MOVIE PIPELINE"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  Log File:   ${LOG_FILE}"
echo "  Python Bin: ${PYTHON_BIN}"
echo "  Runner:     ${RUNNER}"
echo "================================================================================"

nohup "${PYTHON_BIN}" -u "${RUNNER}" > "${LOG_FILE}" 2>&1 &
RUNNER_PID=$!

echo "${RUNNER_PID}" > "${PID_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Started in background with PID: ${RUNNER_PID}"
echo "Logs: tail -f ${LOG_FILE}"
