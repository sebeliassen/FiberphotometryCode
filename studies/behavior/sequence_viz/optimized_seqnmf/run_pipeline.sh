#!/usr/bin/env bash
set -e

REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu/bin/python"
OPTIMIZED_DIR="${REPO_ROOT}/syllable_generation/analysis/sequence_viz/optimized_seqnmf"
LOG_FILE="${OPTIMIZED_DIR}/seqnmf_optimization.log"

echo "================================================================================"
echo "LAUNCHING QUANTITATIVELY OPTIMIZED CONVOLUTIONAL SEQNMF PIPELINE"
echo "Logging to: ${LOG_FILE}"
echo "To follow along in real time, run:"
echo "  tail -f ${LOG_FILE}"
echo "================================================================================"

# Run Phase 1: Quantitative Sweeps
echo "[1/2] Running Quantitative Hyperparameter Optimization Sweeps..."
${PYTHON_BIN} "${OPTIMIZED_DIR}/run_seqnmf_quantitative_sweep.py"

# Run Phase 2: Production Model Fitting & Episode Extraction
echo "[2/2] Fitting Final Production Model & Extracting Verified Episodes..."
${PYTHON_BIN} "${OPTIMIZED_DIR}/fit_optimized_seqnmf_model.py"

echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================================"
