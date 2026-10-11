#!/usr/bin/env bash
# ==============================================================================
# Automated Sequential KPMS 10-Run Production Queue: AMPH Dataset
# ==============================================================================
# 10 runs at 50 AR + 150 SLDS (200 total iters), init_kappa=150k (slds_kappa=15k)
# Checkpointing interval: every 50 iterations (conserves disk space)
# Hardware: RTX 4090 with Banded GMRF + Batched CPU HMM + GPU Kernels
#
# Detached execution:
#   nohup ./queue_amph_k150k_runs.sh > logs/queue_amph_k150k.log 2>&1 &
# ==============================================================================

set -u

NTFY_TOPIC="woody_videos_seb"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

notify() {
    local msg="$1"
    local title="$2"
    local tags="${3:-robot}"
    local priority="${4:-default}"
    curl -s -H "Title: ${title:-MoSeq Orchestrator}" \
            -H "Tags: ${tags}" \
            -H "Priority: ${priority}" \
            -d "${msg}" "${NTFY_URL}" > /dev/null 2>&1 || true
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# GPU & Memory Environment Variables
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export USE_MEMORY_PATCH=1
export CUDA_VISIBLE_DEVICES=0

# Define Continuation Production Runs: (MODEL_NAME:SEED)
RUNS=(
    "amph_prod_k150k_run_11:52"
    "amph_prod_k150k_run_12:53"
    "amph_prod_k150k_run_13:54"
    "amph_prod_k150k_run_14:55"
    "amph_prod_k150k_run_15:56"
)

TOTAL_RUNS=${#RUNS[@]}
INITIAL_DISK=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')

echo "========================================================================"
echo "  AMPH PRODUCTION BATCH: RUNS 11..15 (50 AR + 150 SLDS, Kappa=150k)"
echo "  Date:               $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total Models:       ${TOTAL_RUNS}"
echo "  Checkpoint every:   50 iterations (Disk safe)"
echo "  Initial Disk Free:  ${INITIAL_DISK} GB"
echo "  Log Directory:      ${LOGS_DIR}"
echo "========================================================================"

notify "Launching continuation AMPH batch (Runs 11..15, 50 AR + 150 SLDS @ kappa=150k). Free disk: ${INITIAL_DISK}G." \
       "MoSeq Queue 11..15 Started" "rocket" "default"

# Launch Production Resource & OOM Watchdog
${PYTHON_EXEC} -u "${SCRIPT_DIR}/production_watchdog.py" --parent-pid $$ > "${LOGS_DIR}/production_watchdog.log" 2>&1 &
WATCHDOG_PID=$!
echo "Resource & OOM Watchdog active (PID: ${WATCHDOG_PID})"
echo "Watchdog Log:       ${LOGS_DIR}/production_watchdog.log"

cleanup() {
    echo "Stopping Watchdog (PID: ${WATCHDOG_PID})..."
    kill ${WATCHDOG_PID} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

QUEUE_START=$(date +%s)
RUN_IDX=1

for ITEM in "${RUNS[@]}"; do
    IFS=":" read -r MODEL_NAME SEED <<< "${ITEM}"
    LOG_FILE="${LOGS_DIR}/${MODEL_NAME}.log"
    
    # 0. Check if model is already completed (e.g. run_10 finished previously)
    SUMMARY_FILE="${SCRIPT_DIR}/amph_project/${MODEL_NAME}/model_summary.json"
    if [ -f "${SUMMARY_FILE}" ]; then
        ITERS_DONE=$(${PYTHON_EXEC} -c "import json; print(json.load(open('${SUMMARY_FILE}')).get('iters_done', 0))" 2>/dev/null || echo "0")
        if [ "${ITERS_DONE}" -ge 200 ]; then
            echo ""
            echo "========================================================================"
            echo " [${RUN_IDX}/${TOTAL_RUNS}] SKIPPING: ${MODEL_NAME} (Already completed ${ITERS_DONE}/200 iters)"
            echo " Time:      $(date '+%Y-%m-%d %H:%M:%S')"
            echo " Summary:   ${SUMMARY_FILE}"
            echo "========================================================================"
            notify "SKIP: ${MODEL_NAME} already finished (${ITERS_DONE}/200 iters done). Proceeding to next model." \
                   "MoSeq Run Skipped" "fast_forward" "low"
            RUN_IDX=$((RUN_IDX + 1))
            continue
        fi
    fi

    # 1. Disk space safeguard (< 75 GB free -> abort safely before starting next model)
    CURRENT_DISK=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${CURRENT_DISK}" -lt 75 ]; then
        HALT_MSG="CRITICAL DISK WARNING: Only ${CURRENT_DISK}G free on /. Stopped before launching run ${RUN_IDX}/${TOTAL_RUNS} (${MODEL_NAME})."
        echo "[ABORT] ${HALT_MSG}"
        notify "${HALT_MSG}" "MoSeq Queue Halted (Low Disk)" "warning" "urgent"
        exit 1
    fi

    echo ""
    echo "========================================================================"
    echo " [${RUN_IDX}/${TOTAL_RUNS}] LAUNCHING: ${MODEL_NAME} (Seed: ${SEED})"
    echo " Time:      $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Disk Free: ${CURRENT_DISK} GB"
    echo " Log:       ${LOG_FILE}"
    echo "========================================================================"

    notify "Run [${RUN_IDX}/${TOTAL_RUNS}] started: ${MODEL_NAME} (Seed: ${SEED}). Disk: ${CURRENT_DISK}G free." \
           "MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Started" "hourglass_flowing_sand" "default"

    RUN_START=$(date +%s)
    cd "${SCRIPT_DIR}"
    
    # Run production script: save_every_n_iters = 50
    ${PYTHON_EXEC} -u run_amph_production.py "${MODEL_NAME}" "${SEED}" 50 > "${LOG_FILE}" 2>&1
    EXIT_CODE=$?

    RUN_END=$(date +%s)
    ELAPSED_MIN=$(( (RUN_END - RUN_START) / 60 ))
    DISK_AFTER=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ${MODEL_NAME} completed in ${ELAPSED_MIN} min. Disk: ${DISK_AFTER} GB."
        notify "SUCCESS: ${MODEL_NAME} finished in ${ELAPSED_MIN} min (${RUN_IDX}/${TOTAL_RUNS}). Disk: ${DISK_AFTER}G free." \
               "MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Complete" "white_check_mark" "default"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE} after ${ELAPSED_MIN} min."
        notify "ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE} after ${ELAPSED_MIN} min. Check ${LOG_FILE}" \
               "MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Failed" "x" "high"
    fi

    RUN_IDX=$((RUN_IDX + 1))
    sleep 5
done

QUEUE_END=$(date +%s)
TOTAL_ELAPSED_HOURS=$(( (QUEUE_END - QUEUE_START) / 3600 ))
TOTAL_ELAPSED_MIN=$(( ((QUEUE_END - QUEUE_START) % 3600) / 60 ))
FINAL_DISK=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')

echo ""
echo "========================================================================"
echo " ALL SCHEDULED AMPH PRODUCTION RUNS COMPLETED"
echo " Total Duration: ${TOTAL_ELAPSED_HOURS}h ${TOTAL_ELAPSED_MIN}m"
echo " Final Disk Free: ${FINAL_DISK} GB (Started: ${INITIAL_DISK} GB)"
echo " Finished at:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

notify "All continuation AMPH production runs (11..15) have completed! Total time: ${TOTAL_ELAPSED_HOURS}h ${TOTAL_ELAPSED_MIN}m. Disk free: ${FINAL_DISK}G." \
       "MoSeq Continuation Batch Finished" "party_popper" "high"
