#!/usr/bin/env bash
# =============================================================================
# launch_tandem_cohort_seed_queue.sh
# Sequential Multi-Seed Production Queue for Full Tandem Cohort (118 Sessions)
#
# Runs 4 additional seeds (101, 202, 303, 404) at 50 AR + 200 SLDS iterations
# sequentially on the RTX 4090, followed by automated Expected Marginal
# Likelihood (EML) model ranking across all 5 models (including Seed 42).
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
QUEUE_LOG="$LOG_DIR/tandem_cohort_seed_queue_${RUN_TIMESTAMP}.log"
QUEUE_LINK="$LOG_DIR/tandem_cohort_seed_queue.log"

# Create symlink to latest log
ln -sfn "$QUEUE_LOG" "$QUEUE_LINK"

# Redirect stdout and stderr to both the queue log and terminal
exec > >(tee -a "$QUEUE_LOG") 2>&1

NTFY_URL="https://ntfy.sh/woody_videos_seb"

notify() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    local tags="${4:-robot}"
    curl -s -H "Title: $title" -H "Priority: $priority" -H "Tags: $tags" -d "$msg" "$NTFY_URL" >/dev/null 2>&1 || true
}

# Environment flags for optimal GPU memory and stability
export TF_CPP_MIN_LOG_LEVEL="3"
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
export CUDA_VISIBLE_DEVICES="0"
export USE_MEMORY_PATCH="1"
export PYTHONUNBUFFERED="1"

# The 4 candidate seeds to evaluate
SEEDS=(101 202 303 404)
TOTAL_SEEDS=${#SEEDS[@]}

echo "================================================================================"
echo "🚀 LAUNCHING 4-SEED TANDEM COHORT PRODUCTION QUEUE"
echo "  Date & Time:       $(date)"
echo "  Python Env:        $PYTHON_BIN"
echo "  Queue Log:         $QUEUE_LOG"
echo "  Latest Log Link:   $QUEUE_LINK"
echo "  Seeds to execute:  ${SEEDS[*]}"
echo "  Plan per model:    50 AR + 200 SLDS = 250 iterations"
echo "================================================================================"

notify "KPMS 4-Seed Queue Launched 🚀" "Running 4 seeds (${SEEDS[*]}) at 50+200 iters sequentially on RTX 4090." "default" "rocket,brain"

# Launch 30-Minute Idle Watchdog in background
WATCHDOG_SCRIPT="$REPO_ROOT/syllable_generation/syllabizers/keypoint_moseq_syll/nvidia_idle_watchdog.py"
WATCHDOG_LOG="$LOG_DIR/nvidia_idle_watchdog_${RUN_TIMESTAMP}.log"
ln -sfn "$WATCHDOG_LOG" "$LOG_DIR/nvidia_idle_watchdog.log"

nohup "$PYTHON_BIN" -u "$WATCHDOG_SCRIPT" \
    --timeout-mins 30.0 \
    --check-interval 30.0 \
    >> "$WATCHDOG_LOG" 2>&1 &
WATCHDOG_PID=$!
disown -h "$WATCHDOG_PID"
echo "🛡️ NVIDIA Idle Watchdog armed (PID: $WATCHDOG_PID, Log: $WATCHDOG_LOG)"

# Sequential Runner Loop
CURRENT_INDEX=0
for SEED in "${SEEDS[@]}"; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    MODEL_NAME="tandem_cohort_118sess_50ar_200slds_seed${SEED}"
    MODEL_LOG="$LOG_DIR/${MODEL_NAME}.log"

    echo ""
    echo "================================================================================"
    echo "[$CURRENT_INDEX/$TOTAL_SEEDS] STARTING RUN FOR SEED $SEED ($MODEL_NAME)"
    echo "  Start Time: $(date)"
    echo "  Model Log:  $MODEL_LOG"
    echo "================================================================================"

    notify "Seed $SEED Starting ($CURRENT_INDEX/$TOTAL_SEEDS)" "Model: $MODEL_NAME (50 AR + 200 SLDS)" "default" "hourglass_flowing_sand"

    t0_run=$(date +%s)

    # Run the production script for this seed
    "$PYTHON_BIN" -u "$SCRIPT_DIR/run_tandem_cohort_production_50ar_200slds.py" \
        --init-kappa 7.5e13 \
        --decrease-kappa-factor 10.0 \
        --num-ar-iters 50 \
        --num-slds-iters 200 \
        --save-every-n-iters 25 \
        --model-name "$MODEL_NAME" \
        --seed "$SEED" \
        2>&1 | tee -a "$MODEL_LOG"

    RUN_EXIT_CODE=${PIPESTATUS[0]}
    t1_run=$(date +%s)
    run_duration=$((t1_run - t0_run))
    run_mins=$((run_duration / 60))

    if [ $RUN_EXIT_CODE -ne 0 ]; then
        echo "❌ FATAL: Run for Seed $SEED failed with exit code $RUN_EXIT_CODE!"
        notify "Seed $SEED FAILED ❌" "Model $MODEL_NAME exited with code $RUN_EXIT_CODE. Queue halted." "urgent" "x,warning"
        exit $RUN_EXIT_CODE
    fi

    echo "✅ Seed $SEED ($MODEL_NAME) completed successfully in $run_mins minutes!"
    notify "Seed $SEED Complete 🎉 ($CURRENT_INDEX/$TOTAL_SEEDS)" "Finished 250 iters in ${run_mins}m. Moving to next seed." "default" "white_check_mark"

    # Optional: Automatically archive large checkpoint to /mnt/hdd to preserve NVMe headroom
    MODEL_DIR="$SCRIPT_DIR/kpms_tandem_cohort_production/$MODEL_NAME"
    ARCHIVE_TARGET="/mnt/hdd/old_projects/kpms_checkpoints_archive/tandem_cohort_checkpoints/$MODEL_NAME"
    if [ -f "$MODEL_DIR/checkpoint.h5" ] && [ -w "/mnt/hdd/old_projects/kpms_checkpoints_archive" ]; then
        echo "  Archiving checkpoint.h5 to /mnt/hdd cold storage..."
        mkdir -p "$ARCHIVE_TARGET"
        mv "$MODEL_DIR/checkpoint.h5" "$ARCHIVE_TARGET/checkpoint.h5"
        ln -sfn "$ARCHIVE_TARGET/checkpoint.h5" "$MODEL_DIR/checkpoint.h5"
        echo "  Checkpoint symlink created: $MODEL_DIR/checkpoint.h5 -> $ARCHIVE_TARGET/checkpoint.h5"
    fi

    # Small pause to allow memory reclaim
    sleep 10
done

echo ""
echo "================================================================================"
echo "🎉 ALL 4 SEED RUNS COMPLETED! STARTING EML MODEL SELECTION..."
echo "================================================================================"

EML_SCRIPT="$SCRIPT_DIR/compute_tandem_cohort_eml_rankings.py"
if [ -f "$EML_SCRIPT" ]; then
    echo "Running 5-Model Pairwise EML Evaluation (Seeds 42, 101, 202, 303, 404)..."
    "$PYTHON_BIN" -u "$EML_SCRIPT"
fi

notify "All 4 Seeds & EML Complete! 🏆" "The 5-model ensemble has finished training and EML ranking. Best model crowned!" "default" "trophy,party_popper"
