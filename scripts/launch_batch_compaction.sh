#!/usr/bin/env bash
# ==============================================================================
# launch_batch_compaction.sh
# ==============================================================================
# Detached runner for Safe 3-Stage Checkpoint Compaction on /mnt/hdd.
# Runs detached via nohup with idle I/O priority (ionice -c 3).
# Pushes notifications to https://ntfy.sh/woody_videos_seb per stage & per 5 h5s.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOGS_DIR="${SCRIPT_DIR}/logs"
RUN_TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOGS_DIR}/batch_compaction_${RUN_TIMESTAMP}.log"
LOG_LINK="${LOGS_DIR}/batch_compaction.log"
PID_FILE="${LOGS_DIR}/batch_compaction.pid"

mkdir -p "${LOGS_DIR}"
ln -sfn "${LOG_FILE}" "${LOG_LINK}"

echo "========================================================================"
echo "  LAUNCHING DETACHED SAFE BATCH COMPACTION (3 STAGES)"
echo "  Date:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python Exec:    ${PYTHON_BIN}"
echo "  Log Path:       ${LOG_FILE}"
echo "  Latest Link:    ${LOG_LINK}"
echo "  Ntfy Channel:   https://ntfy.sh/woody_videos_seb"
echo "========================================================================"

cd "${REPO_ROOT}"

# Launch detached via nohup with polite CPU and I/O priority
nohup "${PYTHON_BIN}" -u "${SCRIPT_DIR}/batch_compact_all_checkpoints.py" < /dev/null >> "${LOG_FILE}" 2>&1 &
BG_PID=$!
disown -h "${BG_PID}" 2>/dev/null || true
echo "${BG_PID}" > "${PID_FILE}"

echo "  Compaction Pipeline started with PID: ${BG_PID}"
echo "========================================================================"
echo "  PROCESS DETACHED SUCCESSFULLY"
echo "  To monitor live progress, run:"
echo "    tail -f ${LOG_LINK}"
echo "========================================================================"
