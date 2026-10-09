#!/bin/bash
set -e

# ==============================================================================
# MASTER RUNNER: TANDEM MULTI-VIEW VISUALIZATIONS (COHORT STATIC 21KP V2 SEED 101)
# ==============================================================================
# Pipeline:
# 1. Unified Selection Manifest with 3-Tier Prototype Strategy:
#    - Tier 1: N > 50 (Built-in KPMS Density in 4D PCA)
#    - Tier 2: 12 < N <= 50 (Compact-Neighborhood Selection: min mean pairwise distance)
#    - Tier 3: N <= 12 (Complete Bout Enumeration + Blank Placeholders)
# 2. Dual-View 3x4 Grid Movies (TopCam + BotCam with Basler crystal sync & mirror parity)
# 3. 21kp Trajectory Panels (Side-by-Side & Superimposed PDFs and animated GIFs)
# 4. Interactive HTML Ethologist Gallery & Markdown Catalog
# 5. ntfy.sh alerts to https://ntfy.sh/woody_videos_seb
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
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
echo "🚀 STARTING COHORT STATIC 21kp V2 SEED 101 VISUALIZATION SUITE"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory: $SCRIPT_DIR"
echo "Python Executable: $PYTHON"
echo "================================================================================"

send_alert "V2 Seed 101 Viz Pipeline Started 🎬" "Launching 3-tier prototype suite (Manifest + Grid Movies + Trajectories + HTML Gallery)."

# PHASE 1: Generate or Update Unified Syllable Selection Manifest
echo ""
echo "================================================================================"
echo "▶ PHASE 1: GENERATING UNIFIED SYLLABLE SELECTION MANIFEST (3-TIER)"
echo "================================================================================"
$PYTHON -u select_cohort_static_21kp_v2_syllable_manifest.py --grid_size 12

send_alert "V2 Manifest Built 📋" "Manifest compiled with Tier 1 (Density), Tier 2 (Compact-Neighborhood), and Tier 3 (Bout Enumeration)."

# PHASE 2: Render All Tandem Dual-View Grid Movies
echo ""
echo "================================================================================"
echo "▶ PHASE 2: RENDERING DUAL-VIEW 3x4 GRID MOVIES FROM MANIFEST"
echo "================================================================================"
$PYTHON -u render_all_cohort_static_21kp_v2_seed101_grid_movies.py "$@"

send_alert "V2 Grid Movies Complete 🎥" "Dual-view grid movies rendered. Starting Trajectory Panels."

# PHASE 3: Render Trajectories (Side-by-Side & Superimposed)
echo ""
echo "================================================================================"
echo "▶ PHASE 3: RENDERING 21KP OBSERVED TRAJECTORY PANELS (PDF + GIF)"
echo "================================================================================"
$PYTHON -u render_all_cohort_static_21kp_v2_seed101_trajectories.py "$@"

# PHASE 4: Build Interactive Labeling Gallery & Catalog
echo ""
echo "================================================================================"
echo "▶ PHASE 4: GENERATING INTERACTIVE LABELING GALLERY & CATALOG"
echo "================================================================================"
$PYTHON -u build_cohort_static_21kp_v2_seed101_labeling_gallery.py

echo ""
echo "================================================================================"
echo "🎉 ALL PHASES COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

send_alert "V2 Seed 101 Full Viz Suite COMPLETE! 🏆" "All active syllables rendered with 3-tier prototypes (Movies, Trajectories, Interactive HTML Gallery & Catalog)."
