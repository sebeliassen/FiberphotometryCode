#!/usr/bin/env bash
# =============================================================================
# launch_cohort_static_21kp_v2_production.sh
# =============================================================================
# Detached launcher for clean V2 production run across full 118-session cohort:
# 50 AR + 300 SLDS = 350 iterations (Seed 101)
# Automatic real-time telemetry alerts pushed to https://ntfy.sh/woody_videos_seb
# every 50 iterations.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/cohort_static_21kp_v2_production_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/cohort_static_21kp_v2_production.pid"
LATEST_LINK="${LOG_DIR}/cohort_static_21kp_v2_production_latest.log"

PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
RUNNER="${SCRIPT_DIR}/run_cohort_static_21kp_v2_production.py"

chmod +x "${RUNNER}"

echo "================================================================================"
echo "🚀 LAUNCHING CLEAN FULL COHORT STATIC 21KP V2 PRODUCTION RUN (DETACHED NOHUP)"
echo "  Timestamp:    ${TIMESTAMP}"
echo "  Log File:     ${LOG_FILE}"
echo "  Python Bin:   ${PYTHON_BIN}"
echo "  Runner:       ${RUNNER}"
echo "  Regimen:      50 AR + 300 SLDS = 350 iters (Seed 101)"
echo "  ntfy Channel: https://ntfy.sh/woody_videos_seb (every 50 iters)"
echo "================================================================================"

setsid nohup "${PYTHON_BIN}" -u "${RUNNER}" ${@+"$@"} > "${LOG_FILE}" 2>&1 < /dev/null &
RUNNER_PID=$!

echo "${RUNNER_PID}" > "${PID_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LINK}"

echo "Started in background with PID: ${RUNNER_PID}"
echo "Logs: tail -f ${LOG_FILE}"
echo "Latest: tail -f ${LATEST_LINK}"
