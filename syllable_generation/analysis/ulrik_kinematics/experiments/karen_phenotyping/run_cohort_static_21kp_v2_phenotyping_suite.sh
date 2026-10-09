#!/usr/bin/env bash
# ==============================================================================
# run_cohort_static_21kp_v2_phenotyping_suite.sh
# ==============================================================================
# Executes the full 5-Tier Systematic Behavioral Phenotyping Suite, Genotype
# Comparison, Permutation Tests, and Longitudinal Mixed-Effects Models on the
# new Cohort Static 21kp V2 Seed 101 discovered syllables.
# ==============================================================================

set -eo pipefail

echo "================================================================================"
echo "🧬 STARTING FULL COHORT STATIC 21KP V2 SEED 101 PHENOTYPING SUITE"
echo "Start Time: $(date)"
echo "Host:       $(hostname)"
echo "User:       $(whoami)"
echo "================================================================================"

PYTHON_EXE="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
SCRIPT_DIR="${REPO_ROOT}/syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping"

DATA_PKL="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl"
OUT_BASE="${REPO_ROOT}/syllable_generation/analysis/ulrik_kinematics/output/cohort_static_21kp_v2_seed101_phenotyping"

mkdir -p "$OUT_BASE"

echo ""
echo "Data PKL:   $DATA_PKL"
echo "Out Base:   $OUT_BASE"
echo "FPS:        60.0"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Comprehensive Genotype Behavioral Comparison (5 Dimensions)
# ------------------------------------------------------------------------------
echo "▶ [STEP 1/5] Running 5-Dimension Genotype Behavioral Comparison..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/run_karen_genotype_analysis.py" \
  --data-pkl "$DATA_PKL" \
  --out-dir "${OUT_BASE}/01_genotype_comparison" \
  --fps 60.0

# ------------------------------------------------------------------------------
# STEP 2: 5-Tier Systematic Behavioral Phenotyping Suite (Sections A-D)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 2/5] Running 5-Tier Systematic Phenotyping Suite..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/run_karen_5tier_phenotyping_suite.py" \
  --data-pkl "$DATA_PKL" \
  --out-dir "${OUT_BASE}/02_5tier_phenotyping" \
  --fps 60.0

# ------------------------------------------------------------------------------
# STEP 3: Non-Parametric Label Permutation Testing (10,000 Shuffles)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 3/5] Running 10,000 Label-Shuffling Permutation Tests..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/run_karen_permutation_tests.py" \
  --data-pkl "$DATA_PKL" \
  --out-dir "${OUT_BASE}/03_permutation_tests" \
  --fps 60.0 \
  --n-permutations 10000

# ------------------------------------------------------------------------------
# STEP 4: Longitudinal Linear Mixed-Effects Models (LMMs across 5 waves)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 4/5] Running Longitudinal Linear Mixed-Effects Models..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/run_karen_longitudinal_lmms.py" \
  --data-pkl "$DATA_PKL" \
  --out-dir "${OUT_BASE}/04_longitudinal_lmms" \
  --fps 60.0

# ------------------------------------------------------------------------------
# STEP 5: Macro Behavior Summary
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 5/5] Running Macro Behavior Summary..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/generate_macro_behavior_summary.py" \
  --data-pkl "$DATA_PKL" \
  --out-dir "${OUT_BASE}/05_macro_behavior" \
  --fps 60.0

echo ""
echo "================================================================================"
echo "🎉 ALL PHENOTYPING SUITE MODULES COMPLETED SUCCESSFULLY!"
echo "End Time:   $(date)"
echo "Output Dir: $OUT_BASE"
echo "================================================================================"
