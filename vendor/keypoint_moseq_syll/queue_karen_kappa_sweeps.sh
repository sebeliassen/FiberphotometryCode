#!/usr/bin/env bash
# ==============================================================================
# Automated Sequential KPMS Kappa Sweep Queue & GPU Runner
# ==============================================================================
# Runs models sequentially on RTX 4090 using the kpms_gpu_cuml environment:
# 1. final_karen_model_k130k (130k) -> ~425ms target
# 2. final_karen_model_k100k (100k) -> ~350ms target
# 3. final_karen_model_k150k (150k) -> ~460ms target
# 4. final_karen_model_k180k (180k) -> ~520ms target
# ==============================================================================

NTFY_TOPIC="woody_videos_seb"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

notify() {
    local msg="$1"
    local title="$2"
    local tags="${3:-robot}"
    curl -s -H "Title: ${title:-KPMS Orchestrator}" -H "Tags: ${tags}" -d "${msg}" "${NTFY_URL}" > /dev/null 2>&1 || true
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
PROJECT_DIR="${SCRIPT_DIR}/karen_project_2"

# GPU & Memory Environment Variables
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export USE_MEMORY_PATCH=1
export CUDA_VISIBLE_DEVICES=0

# Sweep Queue: (MODEL_NAME:INIT_KAPPA)
SWEEP_MODELS=(
    "final_karen_model_k180k:180000"
    "final_karen_model_k200k:200000"
    "final_karen_model_k240k:240000"
)

TOTAL_RUNS=${#SWEEP_MODELS[@]}
RUN_IDX=1

notify "Starting automated KPMS sweep queue: ${TOTAL_RUNS} models (k180k [resume] -> k200k -> k240k)." "KPMS Queue Started" "rocket"

for ITEM in "${SWEEP_MODELS[@]}"; do
    IFS=":" read -r MODEL_NAME KAPPA_VAL <<< "${ITEM}"
    
    LOG_FILE="${SCRIPT_DIR}/${MODEL_NAME}.log"
    START_TIME=$(date +%s)
    
    echo ""
    echo "========================================================================"
    echo " [${RUN_IDX}/${TOTAL_RUNS}] LAUNCHING: ${MODEL_NAME} (Init Kappa = ${KAPPA_VAL})"
    echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Log:  ${LOG_FILE}"
    echo "========================================================================"
    
    notify "Launching [${RUN_IDX}/${TOTAL_RUNS}]: ${MODEL_NAME} (kappa=${KAPPA_VAL})." "KPMS Run Started" "hourglass_flowing_sand"
    
    export KPMS_INIT_KAPPA=${KAPPA_VAL}
    
    cd "${SCRIPT_DIR}"
    echo -e "\n\n=== [RESUME/START] $(date '+%Y-%m-%d %H:%M:%S') ===\n" >> "${LOG_FILE}"
    ${PYTHON_EXEC} -u run_pipeline.py "${MODEL_NAME}" 42 None spread "${PROJECT_DIR}" load_or_fit >> "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    
    END_TIME=$(date +%s)
    ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ${MODEL_NAME} finished in ${ELAPSED_MIN} minutes."
        notify "SUCCESS: ${MODEL_NAME} (kappa=${KAPPA_VAL}) finished in ${ELAPSED_MIN} min! (${RUN_IDX}/${TOTAL_RUNS})" "KPMS Run Succeeded" "white_check_mark"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE} after ${ELAPSED_MIN} minutes."
        notify "ERROR: ${MODEL_NAME} failed with exit code ${EXIT_CODE}. Check ${LOG_FILE}" "KPMS Run Failed" "x"
    fi
    
    RUN_IDX=$((RUN_IDX + 1))
    sleep 10
done

echo ""
echo "========================================================================"
echo " [ALL QUEUED RUNS FINISHED] Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
notify "All ${TOTAL_RUNS} queued KPMS kappa sweeps have completed successfully!" "KPMS Sweeps Complete" "party_popper"
