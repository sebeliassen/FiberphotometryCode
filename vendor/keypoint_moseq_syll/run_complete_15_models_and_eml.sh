#!/usr/bin/env bash
# ==============================================================================
# Complete Incomplete KPMS Runs (Run 02 & Run 03) and Compute 15-Model EML Rankings
# ==============================================================================
set -e

SCRIPT_DIR="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export USE_MEMORY_PATCH=1
export CUDA_VISIBLE_DEVICES=0

cd "${SCRIPT_DIR}"

notify() {
    local msg="$1"
    local title="$2"
    curl -s -H "Title: ${title:-MoSeq 15-Model Pipeline}" \
            -H "Tags: hourglass_flowing_sand,robot" \
            -d "${msg}" "https://ntfy.sh/woody_videos_seb" > /dev/null 2>&1 || true
}

notify "Starting completion of run_02 (50 iters) and run_03 (100 iters), followed by full 15-model EML evaluation." "MoSeq Resumption Started"

# --- STEP 1: Complete run_02 (Seed 43) ---
echo "========================================================================"
echo " STEP 1: RESUME & COMPLETE amph_prod_k150k_run_02 (Seed 43, 50 iters left)"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
${PYTHON_EXEC} -u run_amph_production.py amph_prod_k150k_run_02 43 50 2>&1 | tee "${LOGS_DIR}/amph_prod_k150k_run_02_resume.log"
echo "[SUCCESS] Run 02 finished at $(date '+%Y-%m-%d %H:%M:%S')"
notify "Run 02 reached 200 iterations! Proceeding to run 03." "Run 02 Complete"

# --- STEP 2: Complete run_03 (Seed 44) ---
echo "========================================================================"
echo " STEP 2: RESUME & COMPLETE amph_prod_k150k_run_03 (Seed 44, 100 iters left)"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
${PYTHON_EXEC} -u run_amph_production.py amph_prod_k150k_run_03 44 50 2>&1 | tee "${LOGS_DIR}/amph_prod_k150k_run_03_resume.log"
echo "[SUCCESS] Run 03 finished at $(date '+%Y-%m-%d %H:%M:%S')"
notify "Run 03 reached 200 iterations! All 15 models now at 200 iters. Starting EML ranking." "Run 03 Complete"

# --- STEP 3: Full 15-Model EML Evaluation & Ranking ---
echo "========================================================================"
echo " STEP 3: RUN RESULTS EXTRACTION & 15-MODEL EML EVALUATION (210 PAIRS)"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
${PYTHON_EXEC} -u compute_amph_eml_rankings.py 2>&1 | tee "${LOGS_DIR}/compute_amph_eml_rankings_all15.log"
echo "[SUCCESS] All 15 models finished and ranked at $(date '+%Y-%m-%d %H:%M:%S')"
notify "Full 15-model EML evaluation and rankings complete! Check report and CSV." "MoSeq 15 Models Complete"
