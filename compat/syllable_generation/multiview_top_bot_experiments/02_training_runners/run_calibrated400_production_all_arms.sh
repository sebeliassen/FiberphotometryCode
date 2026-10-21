#!/usr/bin/env bash
# =============================================================================
# run_calibrated400_production_all_arms.sh
# =============================================================================
# Trains all 3 arms on all 10 pilot sessions for 200 iterations (50 AR + 150 SLDS)
# using their exact calibrated 3x scout kappas targeting ~400ms dwell time:
#   - 17kp:        1.183e+05 (scout: 3.942e+04 -> 416.7ms)
#   - 21kp_static: 2.102e+16 (scout: 7.006e+15 -> 316.7ms)
#   - 21kp_dynamic:1.292e+15 (scout: 4.307e+14 -> 366.7ms)
#
# Each arm executes in its own isolated fresh Python process to guarantee
# clean VRAM allocation without JIT cache fragmentation.
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
EXP_DIR="${REPO_ROOT}/syllable_generation/multiview_top_bot_experiments"
RUNNER="${EXP_DIR}/02_training_runners/run_pilot_17kp_vs_21kp_comparison.py"
EVALUATOR="${EXP_DIR}/03_evaluation_and_eml/evaluate_17kp_vs_21kp_comparison.py"
LOG_DIR="${EXP_DIR}/02_training_runners/logs"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"

mkdir -p "${LOG_DIR}"

KAPPA_17KP="1.18262e5"
KAPPA_21KP_STATIC="2.10166e16"
KAPPA_21KP_DYNAMIC="1.29222e15"
SUFFIX="calibrated400"

echo "============================================================================="
echo "🚀 LAUNCHING FULL CALIBRATED PRODUCTION RUN (3x KAPPA) ACROSS ALL ARMS"
echo "Date:               $(date '+%Y-%m-%d %H:%M:%S')"
echo "Suffix:             ${SUFFIX}"
echo "Arm 1 (17kp):        SLDS Kappa = ${KAPPA_17KP}"
echo "Arm 2 (21kp_static): SLDS Kappa = ${KAPPA_21KP_STATIC}"
echo "Arm 3 (21kp_dynamic):SLDS Kappa = ${KAPPA_21KP_DYNAMIC}"
echo "Regimen:            50 AR + 150 SLDS = 200 Gibbs Iterations per arm"
echo "============================================================================="

# -----------------------------------------------------------------------------
# ARM 1: 17KP
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/3] LAUNCHING PRODUCTION ARM 1: 17KP (SLDS Kappa = ${KAPPA_17KP}) <<<"
"${PYTHON_BIN}" -u "${RUNNER}" \
    --arms 17kp \
    --num-ar-iters 50 \
    --num-slds-iters 150 \
    --init-kappa 1.0e6 \
    --slds-kappa "${KAPPA_17KP}" \
    --save-every-n 25 \
    --model-suffix "${SUFFIX}" \
    --seed 404

echo "✅ [1/3] Arm 1 (17kp) completed successfully!"

# -----------------------------------------------------------------------------
# ARM 2: 21KP_STATIC
# -----------------------------------------------------------------------------
echo ""
echo ">>> [2/3] LAUNCHING PRODUCTION ARM 2: 21KP_STATIC (SLDS Kappa = ${KAPPA_21KP_STATIC}) <<<"
"${PYTHON_BIN}" -u "${RUNNER}" \
    --arms 21kp_static \
    --num-ar-iters 50 \
    --num-slds-iters 150 \
    --init-kappa 1.0e6 \
    --slds-kappa "${KAPPA_21KP_STATIC}" \
    --save-every-n 25 \
    --model-suffix "${SUFFIX}" \
    --seed 404

echo "✅ [2/3] Arm 2 (21kp_static) completed successfully!"

# -----------------------------------------------------------------------------
# ARM 3: 21KP_DYNAMIC
# -----------------------------------------------------------------------------
echo ""
echo ">>> [3/3] LAUNCHING PRODUCTION ARM 3: 21KP_DYNAMIC (SLDS Kappa = ${KAPPA_21KP_DYNAMIC}) <<<"
"${PYTHON_BIN}" -u "${RUNNER}" \
    --arms 21kp_dynamic \
    --num-ar-iters 50 \
    --num-slds-iters 150 \
    --init-kappa 1.0e6 \
    --slds-kappa "${KAPPA_21KP_DYNAMIC}" \
    --save-every-n 25 \
    --model-suffix "${SUFFIX}" \
    --seed 404

echo "✅ [3/3] Arm 3 (21kp_dynamic) completed successfully!"

# -----------------------------------------------------------------------------
# PHASE 3: EVALUATION & MASTER SCORECARD
# -----------------------------------------------------------------------------
echo ""
echo "============================================================================="
echo "📊 GENERATING DURATION-MATCHED COMPARATIVE SCORECARD"
echo "============================================================================="

SCORECARD_CSV="${EXP_DIR}/04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_scorecard.csv"
METRICS_JSON="${EXP_DIR}/04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_metrics.json"

"${PYTHON_BIN}" "${EVALUATOR}" \
    --suffix "${SUFFIX}" \
    --output-csv "${SCORECARD_CSV}" \
    --output-json "${METRICS_JSON}"

echo ""
echo "============================================================================="
echo "🎉 ALL CALIBRATED ARMS & SCORECARD COMPLETED SUCCESSFULLY!"
echo "Date:               $(date '+%Y-%m-%d %H:%M:%S')"
echo "Scorecard:          ${SCORECARD_CSV}"
echo "Metrics:            ${METRICS_JSON}"
echo "============================================================================="
