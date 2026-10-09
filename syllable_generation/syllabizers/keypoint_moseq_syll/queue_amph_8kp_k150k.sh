#!/usr/bin/env bash
# ==============================================================================
# Automated Sequential KPMS 5-Run Production Queue: 8-Keypoint AMPH Dataset
# ==============================================================================
# 5 runs at 50 AR + 150 SLDS (200 total iters), init_kappa=150k (slds_kappa=15k)
# 8 Informative Keypoints, Latent Dim = 6 (95.52% EVR)
# Checkpointing interval: every 50 iterations (conserves disk space)
# Hardware: RTX 4090 with Banded GMRF + Batched CPU HMM + GPU Kernels
#
# Detached execution:
#   nohup ./queue_amph_8kp_k150k.sh > logs/queue_amph_8kp.log 2>&1 &
# ==============================================================================

set -u

NTFY_TOPIC="woody_videos_seb"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

notify() {
    local msg="$1"
    local title="$2"
    local tags="${3:-robot}"
    local priority="${4:-default}"
    curl -s -H "Title: ${title:-MoSeq 8KP Orchestrator}" \
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

# Define 5 8-Keypoint Production Runs: (MODEL_NAME:SEED)
RUNS=(
    "amph_8kp_k150k_run_01:42"
    "amph_8kp_k150k_run_02:101"
    "amph_8kp_k150k_run_03:202"
    "amph_8kp_k150k_run_04:303"
    "amph_8kp_k150k_run_05:404"
)

TOTAL_RUNS=${#RUNS[@]}
INITIAL_DISK=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')

echo "========================================================================"
echo "  AMPH 8-KEYPOINT PRODUCTION BATCH (5 Models, 50 AR + 150 SLDS @ 150k)"
echo "  Date:               $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total Models:       ${TOTAL_RUNS}"
echo "  Keypoints:          8 (Informative core skeleton)"
echo "  Latent Dim:         6"
echo "  Checkpoint every:   50 iterations (Disk safe)"
echo "  Initial Disk Free:  ${INITIAL_DISK} GB"
echo "  Log Directory:      ${LOGS_DIR}"
echo "========================================================================"

notify "Launching 8-Keypoint AMPH Batch (5 Models, 50 AR + 150 SLDS, Dim=6, K=150k). Free disk: ${INITIAL_DISK}G." \
       "8KP MoSeq Queue Started" "rocket" "default"

QUEUE_START=$(date +%s)
RUN_IDX=1

for ITEM in "${RUNS[@]}"; do
    IFS=":" read -r MODEL_NAME SEED <<< "${ITEM}"
    LOG_FILE="${LOGS_DIR}/${MODEL_NAME}.log"
    
    # 0. Check if model is already completed
    SUMMARY_FILE="${SCRIPT_DIR}/amph_project_8kp/${MODEL_NAME}/model_summary.json"
    if [ -f "${SUMMARY_FILE}" ]; then
        ITERS_DONE=$(${PYTHON_EXEC} -c "import json; print(json.load(open('${SUMMARY_FILE}')).get('iters_done', 0))" 2>/dev/null || echo "0")
        if [ "${ITERS_DONE}" -ge 200 ]; then
            echo ""
            echo "========================================================================"
            echo " [${RUN_IDX}/${TOTAL_RUNS}] SKIPPING: ${MODEL_NAME} (Already completed ${ITERS_DONE}/200 iters)"
            echo " Time:      $(date '+%Y-%m-%d %H:%M:%S')"
            echo " Summary:   ${SUMMARY_FILE}"
            echo "========================================================================"
            notify "SKIP: ${MODEL_NAME} already finished (${ITERS_DONE}/200 iters done)." \
                   "8KP MoSeq Skipped" "fast_forward" "low"
            RUN_IDX=$((RUN_IDX + 1))
            continue
        fi
    fi

    # 1. Disk space safeguard (< 75 GB free -> abort safely)
    CURRENT_DISK=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${CURRENT_DISK}" -lt 75 ]; then
        HALT_MSG="CRITICAL DISK WARNING: Only ${CURRENT_DISK}G free on /. Stopped before launching run ${RUN_IDX}/${TOTAL_RUNS} (${MODEL_NAME})."
        echo "[ABORT] ${HALT_MSG}"
        notify "${HALT_MSG}" "8KP MoSeq Queue Halted (Low Disk)" "warning" "urgent"
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
           "8KP MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Started" "hourglass_flowing_sand" "default"

    RUN_START=$(date +%s)
    cd "${SCRIPT_DIR}"
    
    # Run 8-keypoint production script: save_every_n_iters = 50
    ${PYTHON_EXEC} -u run_amph_8kp_production.py "${MODEL_NAME}" "${SEED}" 50 > "${LOG_FILE}" 2>&1
    EXIT_CODE=$?

    RUN_END=$(date +%s)
    ELAPSED_MIN=$(( (RUN_END - RUN_START) / 60 ))
    DISK_AFTER=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ${MODEL_NAME} completed in ${ELAPSED_MIN} min. Disk: ${DISK_AFTER} GB."
        notify "SUCCESS: ${MODEL_NAME} finished in ${ELAPSED_MIN} min (${RUN_IDX}/${TOTAL_RUNS}). Disk: ${DISK_AFTER}G free." \
               "8KP MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Complete" "white_check_mark" "default"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE} after ${ELAPSED_MIN} min."
        notify "ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE} after ${ELAPSED_MIN} min. Check ${LOG_FILE}" \
               "8KP MoSeq Run ${RUN_IDX}/${TOTAL_RUNS} Failed" "x" "high"
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
echo " ALL 5 8-KEYPOINT PRODUCTION RUNS COMPLETED"
echo " Total Duration: ${TOTAL_ELAPSED_HOURS}h ${TOTAL_ELAPSED_MIN}m"
echo " Final Disk Free: ${FINAL_DISK} GB (Started: ${INITIAL_DISK} GB)"
echo " Finished at:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

notify "All 5 8-Keypoint AMPH production runs completed! Total time: ${TOTAL_ELAPSED_HOURS}h ${TOTAL_ELAPSED_MIN}m. Disk free: ${FINAL_DISK}G." \
       "8KP MoSeq Batch Finished" "party_popper" "high"
