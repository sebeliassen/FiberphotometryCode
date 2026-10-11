#!/usr/bin/env bash
# =============================================================================
# queue_additional_seeds.sh
# Resilient, Cgroup-Isolated Multi-Seed Production Queue for Full Tandem Cohort
#
# Status:
#   - Seed 101: Finished (250 iters)
#   - Seed 202: Finished (250 iters)
#   - Seed 303: Finished (250 iters)
#   - Seed 404: Checkpointed at iter 75 -> Resuming 75 -> 250 iters
#   - Seed 505: Pending -> Fresh 250 iters
#
# Protection:
#   Runs training under transient systemd scope with MemoryMax=80G
#   (guaranteeing 45 GB headroom for host and language server).
#   Executes standalone zero-OOM checkpoint compaction after each seed.
# =============================================================================
set -euo pipefail

repo=/home/fsp585/Desktop/FiberphotometryCode
python=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
runner="$repo/syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py"
project="$repo/syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_continuous_calibrated"
logs="$repo/syllable_generation/multiview_top_bot_experiments/02_training_runners/logs"
compact_script="$repo/diagnostics/oom_harness/compact_standalone.py"
mkdir -p "$logs"

exec 9>"$project/additional_seed_queue.lock"
flock -n 9 || { echo "Another seed queue is active."; exit 1; }

NTFY_URL="https://ntfy.sh/woody_videos_seb"
notify() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    local tags="${4:-robot}"
    curl -s -H "Title: $title" -H "Priority: $priority" -H "Tags: $tags" -d "$msg" "$NTFY_URL" >/dev/null 2>&1 || true
}

echo "================================================================================"
echo "🚀 RESUMING TANDEM COHORT PRODUCTION QUEUE (SEEDS 404 -> 505)"
echo "  Date:       $(date)"
echo "  Python:     $python"
echo "  Completed:  Seeds 101, 202, 303 (all at 250 iters)"
echo "  To Execute: Seed 404 (resume from iter 75), Seed 505 (fresh 250 iters)"
echo "================================================================================"

notify "KPMS Queue Resuming" "Resuming Seed 404 from iter 75 -> 250, followed by Seed 505." "default" "rocket,brain"

# Seeds to run: (seed:model_name)
SEEDS_TO_RUN=(
    "404:tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073"
    "505:tandem_calibrated_50ar_200slds_seed505_20261002_012237_940073"
)

for item in "${SEEDS_TO_RUN[@]}"; do
    IFS=":" read -r seed name <<< "$item"

    if pgrep -f '[r]un_cohort_calibrated_retrain_seed101.py' >/dev/null; then
        echo "STOP: another cohort training process is active."
        exit 1
    fi

    free_bytes=$(df -PB1 "$project" | awk 'NR==2 {print $4}')
    if (( free_bytes < 80000000000 )); then
        echo "STOP: less than 80 GB disk space available."
        exit 1
    fi

    log="$logs/$name.log"
    echo ""
    echo "================================================================================"
    echo "START seed=$seed model=$name log=$log"
    echo "================================================================================"

    notify "Seed $seed Starting" "Model: $name (running to 250 iters)" "default" "hourglass_flowing_sand"

    # Launch under isolated transient systemd unit with MemoryMax=80G
    systemd-run --user --unit="kpms-seed${seed}" --service-type=simple \
        -p MemoryMax=80G -p MemorySwapMax=0 -p OOMPolicy=kill -p Restart=no \
        -p StandardOutput=journal -p StandardError=journal \
        "$python" -u "$runner" \
        --seed "$seed" --num-ar-iters 50 --num-slds-iters 200 \
        --save-every-n-iters 25 --model-name "$name" --fit-only

    # Stream journal logs to file
    journalctl --user -u "kpms-seed${seed}.service" -f --no-pager >> "$log" 2>&1 &
    LOG_PID=$!

    # Wait for service to finish
    while systemctl --user is-active --quiet "kpms-seed${seed}.service"; do
        sleep 15
    done
    kill "$LOG_PID" 2>/dev/null || true

    # Check exit status
    RESULT=$(systemctl --user show "kpms-seed${seed}.service" -p Result --value)
    EXIT_CODE=$(systemctl --user show "kpms-seed${seed}.service" -p ExecMainStatus --value)

    if [ "$RESULT" != "success" ] || [ "$EXIT_CODE" != "0" ]; then
        echo "STOP: seed $seed failed with Result=$RESULT, ExitCode=$EXIT_CODE. Check $log"
        notify "Seed $seed FAILED ❌" "Result=$RESULT, ExitCode=$EXIT_CODE. Check $log" "urgent" "x,warning"
        exit 1
    fi

    if ! grep -Fq '[FIT COMPLETE] Iteration 250;' "$log"; then
        echo "STOP: seed $seed lacks completion marker in $log"
        notify "Seed $seed Incomplete ❌" "No completion marker in $log" "urgent" "x"
        exit 1
    fi

    echo "✅ COMPLETE seed=$seed (250 iterations reached)"
    notify "Seed $seed Complete 🎉" "Reached 250 iterations! Starting compaction." "default" "white_check_mark"

    # Standalone zero-OOM compaction in a fresh process
    CKPT="$project/$name/checkpoint.h5"
    if [ -f "$CKPT" ]; then
        echo "  [compaction] Running standalone checkpoint compaction for seed $seed..."
        systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
            "$python" "$compact_script" "$CKPT" --keep-iters 50 250 --chunk-mb 64
        echo "  [compaction] Seed $seed checkpoint successfully compacted!"
    fi
done

echo ""
echo "================================================================================"
echo "🎉 ALL SEEDS (101, 202, 303, 404, 505) FINISHED AND COMPACTED!"
echo "================================================================================"

notify "Tandem Cohort Production Queue Complete 🏆" "All 5 seeds (101, 202, 303, 404, 505) finished training & compaction!" "high" "trophy,tada"
