#!/usr/bin/env bash
# =============================================================================
# launch_tandem_cohort_production.sh
# Detached production launcher for Full Tandem Cohort (118 Sessions, 24M Frames)
# 50 AR + 200 SLDS (250 Gibbs iterations) with Checkpointing and Idle Watchdog.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
MAIN_LOG="$LOG_DIR/tandem_cohort_production_${RUN_TIMESTAMP}.log"
WATCHDOG_LOG="$LOG_DIR/nvidia_idle_watchdog_${RUN_TIMESTAMP}.log"

# Create symlinks to 'latest' logs for easy tailing
ln -sf "$MAIN_LOG" "$LOG_DIR/tandem_cohort_production.log"
ln -sf "$WATCHDOG_LOG" "$LOG_DIR/nvidia_idle_watchdog.log"

echo "================================================================================"
echo "🚀 LAUNCHING FULL TANDEM COHORT PRODUCTION TRAINING (118 SESSIONS)"
echo "================================================================================"
echo "  Date & Time:    $(date)"
echo "  Python Env:     $PYTHON_BIN"
echo "  Main Log:       $MAIN_LOG (Symlinked: $LOG_DIR/tandem_cohort_production.log)"
echo "  Watchdog Log:   $WATCHDOG_LOG"
echo "  Ntfy Endpoint:  https://ntfy.sh/woody_videos_seb"
echo "================================================================================"

# Hardware optimization flags
export TF_CPP_MIN_LOG_LEVEL="3"
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
export CUDA_VISIBLE_DEVICES="0"
export USE_MEMORY_PATCH="1"
export PYTHONUNBUFFERED="1"

# 1. Launch Main Training Runner with nohup
nohup "$PYTHON_BIN" -u "$SCRIPT_DIR/run_tandem_cohort_production_50ar_200slds.py" \
    --init-kappa 7.5e13 \
    --decrease-kappa-factor 10.0 \
    --num-ar-iters 50 \
    --num-slds-iters 200 \
    --save-every-n-iters 25 \
    --model-name "tandem_cohort_118sess_50ar_200slds" \
    >> "$MAIN_LOG" 2>&1 &

MAIN_PID=$!
disown -h "$MAIN_PID"

echo "✅ Main KPMS Production Job launched successfully!"
echo "   PID: $MAIN_PID"
echo "   Live tail command: tail -f $LOG_DIR/tandem_cohort_production.log"

# 2. Launch 30-Minute Idle Watchdog with nohup
WATCHDOG_SCRIPT="$REPO_ROOT/syllable_generation/syllabizers/keypoint_moseq_syll/nvidia_idle_watchdog.py"
nohup "$PYTHON_BIN" -u "$WATCHDOG_SCRIPT" \
    --timeout-mins 30.0 \
    --check-interval 30.0 \
    >> "$WATCHDOG_LOG" 2>&1 &

WATCHDOG_PID=$!
disown -h "$WATCHDOG_PID"

echo "🛡️ 30-Minute NVIDIA / Python Idle Watchdog armed successfully!"
echo "   PID: $WATCHDOG_PID"
echo "   Watchdog log: tail -f $LOG_DIR/nvidia_idle_watchdog.log"
echo "================================================================================"
echo "Both background tasks are completely detached and protected from disconnects."
echo "================================================================================"
