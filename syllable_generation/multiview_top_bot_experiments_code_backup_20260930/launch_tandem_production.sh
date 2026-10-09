#!/usr/bin/env bash
# ==============================================================================
# launch_tandem_production.sh
# ==============================================================================
# Detached production launcher for Tandem 23-Keypoint KPMS Training
# (50 AR + 200 SLDS Iterations, Kappa 7.5e13 -> 7.5e12).
#
# Robustness & Fault-Tolerance Features:
# - Completely detached via `nohup` (survives SSH/terminal/IDE disconnects)
# - Checkpoints every 25 iterations with automatic resume if re-launched
# - Live unbuffered logging to logs/tandem_production_50ar_200slds.log
# - Real-time progress updates & alerts pushed to ntfy.sh/woody_videos_seb
# - Real-time resource watchdog (RAM/VRAM/Disk) running alongside
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOGS_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOGS_DIR}/tandem_production_50ar_200slds.log"
PID_FILE="${LOGS_DIR}/tandem_production.pid"
WATCHDOG_LOG="${LOGS_DIR}/production_watchdog.log"
WATCHDOG_SCRIPT="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/production_watchdog.py"

mkdir -p "${LOGS_DIR}"

# Check disk space (> 40 GB required)
FREE_DISK_GB=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
if [ "${FREE_DISK_GB}" -lt 40 ]; then
    echo "ERROR: Insufficient disk space on / (${FREE_DISK_GB} GB free < 40 GB required)."
    exit 1
fi

# Export runtime environment flags
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export USE_MEMORY_PATCH=1
export CUDA_VISIBLE_DEVICES=0

cd "${SCRIPT_DIR}"

echo "========================================================================"
echo "  LAUNCHING DETACHED TANDEM PRODUCTION RUN (50 AR + 200 SLDS)"
echo "  Date:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python Exec:    ${PYTHON_EXEC}"
echo "  Log Path:       ${LOG_FILE}"
echo "  Free Disk:      ${FREE_DISK_GB} GB"
echo "  Ntfy Channel:   https://ntfy.sh/woody_videos_seb"
echo "========================================================================"

# Launch the training script detached using nohup
nohup "${PYTHON_EXEC}" -u "${SCRIPT_DIR}/run_tandem_production_50ar_200slds.py" < /dev/null >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
disown -h "${TRAIN_PID}" 2>/dev/null || true
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "  Training Process started with PID: ${TRAIN_PID}"

# Launch resource watchdog tied to the training PID
if [ -f "${WATCHDOG_SCRIPT}" ]; then
    nohup "${PYTHON_EXEC}" -u "${WATCHDOG_SCRIPT}" --parent-pid "${TRAIN_PID}" < /dev/null >> "${WATCHDOG_LOG}" 2>&1 &
    WATCHDOG_PID=$!
    disown -h "${WATCHDOG_PID}" 2>/dev/null || true
    echo "  Resource Watchdog started with PID: ${WATCHDOG_PID}"
fi

# Launch 30-minute GPU Idle Watchdog
IDLE_WATCHDOG_SCRIPT="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/nvidia_idle_watchdog.py"
IDLE_WATCHDOG_LOG="${LOGS_DIR}/nvidia_idle_watchdog.log"
IDLE_PID_FILE="${LOGS_DIR}/nvidia_idle_watchdog.pid"

if [ -f "${IDLE_WATCHDOG_SCRIPT}" ]; then
    nohup "${PYTHON_EXEC}" -u "${IDLE_WATCHDOG_SCRIPT}" --timeout-mins 30.0 < /dev/null >> "${IDLE_WATCHDOG_LOG}" 2>&1 &
    IDLE_PID=$!
    disown -h "${IDLE_PID}" 2>/dev/null || true
    echo "${IDLE_PID}" > "${IDLE_PID_FILE}"
    echo "  GPU Idle Watchdog (30m threshold) started with PID: ${IDLE_PID}"
fi

echo "========================================================================"
echo "  EXECUTION DETACHED SUCCESSFULLY"
echo "  To monitor output live, run:"
echo "    tail -f ${LOG_FILE}"
echo "========================================================================"

