#!/bin/bash
set -e

# ==============================================================================
# MASTER RUNNER: TANDEM MULTI-VIEW VISUALIZATIONS (COHORT STATIC 21KP SEED 101)
# ==============================================================================
# Runs:
# 1. Trajectories (Superimposed + Tandem Side-by-Side PDFs and GIFs, with density sampling)
# 2. Grid Movies (WITH density sampling) -> grid_movies_density/
# 3. Grid Movies (WITHOUT density sampling / random uniform) -> grid_movies_random/
# 4. Interactive Syllable Labeling Gallery HTML & Markdown Catalog
# 5. Pushes ntfy notifications to https://ntfy.sh/woody_videos_seb
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/fsp585/miniforge3/envs/keypoint_moseq/bin/python"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

NTFY_URL="https://ntfy.sh/woody_videos_seb"

send_alert() {
    local title="$1"
    local message="$2"
    curl -s \
        -H "Title: $title" \
        -H "Tags: movie_camera,paw_prints" \
        -d "$message" \
        "$NTFY_URL" > /dev/null 2>&1 || true
}

echo "================================================================================"
echo "🚀 STARTING COHORT STATIC 21kp SEED 101 VISUALIZATION PIPELINE"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory: $SCRIPT_DIR"
echo "Python Executable: $PYTHON"
echo "================================================================================"

send_alert "Seed 101 Tandem Viz Started 🎬" "Launching CPU rendering suite for 74 active syllables (Trajectories + Density/Random Grid Movies + HTML Gallery)."

# PHASE 1: Trajectory Visualizations (Superimposed & Side-by-Side PDFs + GIFs)
echo ""
echo "================================================================================"
echo "▶ PHASE 1: RENDERING TRAJECTORIES (ALL ACTIVE SYLLABLES, DENSITY SAMPLING)"
echo "================================================================================"
$PYTHON -u render_all_cohort_static_21kp_seed101_trajectories.py "$@"

send_alert "Seed 101 Trajectories Rendered 📐" "Side-by-side (Dorsal Spine vs Ventral Paws) & Superimposed PDFs/GIFs complete. Starting Grid Movies."

# PHASE 2: Grid Movies (WITH Density Sampling)
echo ""
echo "================================================================================"
echo "▶ PHASE 2: RENDERING GRID MOVIES (WITH DENSITY SAMPLING)"
echo "================================================================================"
$PYTHON -u render_all_cohort_static_21kp_seed101_grid_movies.py --mode density "$@"

send_alert "Seed 101 Density Grid Movies Done 🎥" "Phase 2 complete. Starting Random Uniform Grid Movies."

# PHASE 3: Grid Movies (WITHOUT Density Sampling / Random Uniform)
echo ""
echo "================================================================================"
echo "▶ PHASE 3: RENDERING GRID MOVIES (WITHOUT DENSITY SAMPLING / RANDOM)"
echo "================================================================================"
$PYTHON -u render_all_cohort_static_21kp_seed101_grid_movies.py --mode random "$@"

# PHASE 4: Build Interactive Labeling Gallery & Catalog
echo ""
echo "================================================================================"
echo "▶ PHASE 4: GENERATING INTERACTIVE LABELING GALLERY & CATALOG"
echo "================================================================================"
$PYTHON -u build_cohort_static_21kp_seed101_labeling_gallery.py

echo ""
echo "================================================================================"
echo "🎉 ALL PHASES COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

send_alert "Seed 101 Full Viz Suite COMPLETE! 🎉" "All 74 syllables rendered (Trajectories, Dual-View Grid Movies, Interactive HTML Dashboard & Markdown Catalog)."
