#!/usr/bin/env bash
# ==============================================================================
# run_overnight_full_visualizations.sh
# ==============================================================================
# Complete, autonomous overnight rendering pipeline for Cohort Static 21kp V2:
# 1. Canonical KPMS Median Trajectories (Side-by-Side & Superimposed, all qualifying bouts)
# 2. Dual-View Density Grid Movies (Active-RMS reranked)
# 3. Dual-View Random Grid Movies (Uniform spread, both cameras active & clean)
# 4. Interactive HTML Dashboard & Ethological Catalog Compilation
#
# Designed to run detached via nohup:
#   nohup ./run_overnight_full_visualizations.sh > overnight_visualizations.log 2>&1 &
# ==============================================================================

set -eo pipefail

echo "================================================================================"
echo "🌙 STARTING OVERNIGHT FULL VISUALIZATION PIPELINE"
echo "Start Time: $(date)"
echo "Host:       $(hostname)"
echo "User:       $(whoami)"
echo "================================================================================"

# CPU-safe environment to prevent CUDA/ptxas compiler errors with OpenCV/FFmpeg
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

PYTHON_EXE="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
EXP_DIR="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments"
SCRIPT_DIR="${EXP_DIR}/04_reports_and_figures"
VIZ_DIR="${SCRIPT_DIR}/visualizations_cohort_static_21kp_v2_seed101_v4"

cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# STEP 1: Standard KPMS Median Trajectory Plots (Side-by-Side & Superimposed)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 1/4] Rendering Canonical Median Trajectory Plots (all qualifying bouts)..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/render_all_cohort_static_21kp_v2_seed101_trajectories.py" --overwrite

# ------------------------------------------------------------------------------
# STEP 2: Dual-View Density Grid Movies (Active-RMS Reranked)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 2/4] Verifying / Rendering Dual-View Density Grid Movies..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/render_all_cohort_static_21kp_v2_seed101_grid_movies.py"

# ------------------------------------------------------------------------------
# STEP 3: Dual-View Random Grid Movies (High Confidence, Both Cameras Active)
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 3/4] Rendering Dual-View Random Grid Movies (all active syllables)..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/render_all_cohort_static_21kp_v2_seed101_random_grid_movies.py" --overwrite

# ------------------------------------------------------------------------------
# STEP 4: Build Interactive HTML Labeling Dashboard & Markdown Catalog
# ------------------------------------------------------------------------------
echo ""
echo "▶ [STEP 4/4] Compiling Interactive HTML Labeling Dashboard & Catalog..."
echo "  Timestamp: $(date)"
"$PYTHON_EXE" "${SCRIPT_DIR}/build_cohort_static_21kp_v2_seed101_labeling_gallery.py"

# ------------------------------------------------------------------------------
# Create / update finalized V5 alias
# ------------------------------------------------------------------------------
echo ""
echo "Linking final output to visualizations_cohort_static_21kp_v2_seed101_v5..."
ln -sfn "$VIZ_DIR" "${SCRIPT_DIR}/visualizations_cohort_static_21kp_v2_seed101_v5"

echo ""
echo "================================================================================"
echo "🎉 OVERNIGHT FULL VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY!"
echo "End Time:   $(date)"
echo "Gallery:    ${VIZ_DIR}/labeling_gallery.html"
echo "================================================================================"
