#!/usr/bin/env bash
# ==============================================================================
# Sequential Parity & Equivalence Verification Suite
# Option A (5 Parity Tests) followed by Option B (Continuous Sampler Equivalence)
# Live log stream destination: /tmp/kpms_live.log
# ==============================================================================

set -u
LIVE_LOG="/tmp/kpms_live.log"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
DIAG_DIR="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/diagnostics"

echo "========================================================================" | tee "${LIVE_LOG}"
echo "  KEYPOINT-MOSEQ PARITY SUITE (OPTION A) & EQUIVALENCE AUDIT (OPTION B)" | tee -a "${LIVE_LOG}"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LIVE_LOG}"
echo "  Target Model: amph_prod_k150k_run_01 (Checkpoint: Iteration 150)" | tee -a "${LIVE_LOG}"
echo "========================================================================" | tee -a "${LIVE_LOG}"
echo "" | tee -a "${LIVE_LOG}"

cd "${DIAG_DIR}/.."

run_test() {
    local test_num="$1"
    local test_name="$2"
    local script_file="$3"

    echo "" | tee -a "${LIVE_LOG}"
    echo "========================================================================" | tee -a "${LIVE_LOG}"
    echo " [${test_num}] RUNNING: ${test_name}" | tee -a "${LIVE_LOG}"
    echo " Script: ${script_file}" | tee -a "${LIVE_LOG}"
    echo " Time:   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LIVE_LOG}"
    echo "========================================================================" | tee -a "${LIVE_LOG}"

    ${PYTHON_EXEC} -u "${DIAG_DIR}/${script_file}" 2>&1 | tee -a "${LIVE_LOG}"
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [PASS] ${test_name} PASSED successfully." | tee -a "${LIVE_LOG}"
    else
        echo ">>> [FAIL] ${test_name} FAILED with exit code ${exit_code}!" | tee -a "${LIVE_LOG}"
        exit ${exit_code}
    fi
}

# -----------------------------------------------------------------------------
# PART 1: OPTION A (PARITY TEST SUITE)
# -----------------------------------------------------------------------------
echo ">>> STARTING OPTION A: CORE PARITY AUDIT SUITE..." | tee -a "${LIVE_LOG}"

run_test "1/6" "Kernel Semantic Parity" "test_kernel_semantic_parity.py"
run_test "2/6" "Double-Buffered FFBS Parity" "test_double_buffer_ffbs.py"
run_test "3/6" "Consolidated NaN Checker" "test_consolidated_nan_checker.py"
run_test "4/6" "Checkpoint Resume & Fused Transform" "test_checkpoint_resume_and_fused_transform.py"
run_test "5/6" "CPU Cache Validity & Invalidation" "test_cpu_cache_validity.py"

echo "" | tee -a "${LIVE_LOG}"
echo ">>> [OPTION A COMPLETE] All 5 core parity test suites PASSED!" | tee -a "${LIVE_LOG}"
echo "" | tee -a "${LIVE_LOG}"

# -----------------------------------------------------------------------------
# PART 2: OPTION B (CONTINUOUS SAMPLER EQUIVALENCE AUDIT)
# -----------------------------------------------------------------------------
echo ">>> STARTING OPTION B: CONTINUOUS LATENT SAMPLER EQUIVALENCE AUDIT (100 DRAWS)..." | tee -a "${LIVE_LOG}"

run_test "6/6" "Continuous Latent Sampler Equivalence (Numba vs JAX Kalman)" "test_continuous_sampler_equivalence.py"

echo "" | tee -a "${LIVE_LOG}"
echo "========================================================================" | tee -a "${LIVE_LOG}"
echo " ALL 6 PARITY & EQUIVALENCE AUDITS PASSED CLEANLY!" | tee -a "${LIVE_LOG}"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LIVE_LOG}"
echo "========================================================================" | tee -a "${LIVE_LOG}"
