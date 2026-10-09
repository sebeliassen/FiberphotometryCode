#!/bin/bash
set -e

# ==============================================================================
# MASTER RUNNER: TANDEM MULTI-VIEW VISUALIZATIONS (CHAMPION SEED 404)
# ==============================================================================
# Runs:
# 1. Trajectories (Superimposed + Tandem Side-by-Side PDFs and GIFs, with density sampling)
# 2. Grid Movies (WITH density sampling) -> grid_movies_density/
# 3. Grid Movies (WITHOUT density sampling / random uniform) -> grid_movies_random/
# 4. Interactive Syllable Labeling Gallery HTML & Markdown Catalog
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo "🚀 STARTING COMPREHENSIVE SEED 404 VISUALIZATION PIPELINE"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory: $SCRIPT_DIR"
echo "Python Executable: $PYTHON"
echo "================================================================================"

# PHASE 1: Trajectory Visualizations (Superimposed & Side-by-Side PDFs + GIFs)
echo ""
echo "================================================================================"
echo "▶ PHASE 1: RENDERING TRAJECTORIES (ALL SYLLABLES, DENSITY SAMPLING)"
echo "================================================================================"
$PYTHON -u render_all_seed404_trajectories.py "$@"

# PHASE 2: Grid Movies (WITH Density Sampling)
echo ""
echo "================================================================================"
echo "▶ PHASE 2: RENDERING GRID MOVIES (WITH DENSITY SAMPLING)"
echo "================================================================================"
$PYTHON -u render_all_seed404_grid_movies.py --mode density "$@"

# PHASE 3: Grid Movies (WITHOUT Density Sampling / Random Uniform)
echo ""
echo "================================================================================"
echo "▶ PHASE 3: RENDERING GRID MOVIES (WITHOUT DENSITY SAMPLING / RANDOM)"
echo "================================================================================"
$PYTHON -u render_all_seed404_grid_movies.py --mode random "$@"

# PHASE 4: Build Interactive Labeling Gallery & Catalog
echo ""
echo "================================================================================"
echo "▶ PHASE 4: GENERATING INTERACTIVE LABELING GALLERY & CATALOG"
echo "================================================================================"
$PYTHON -u build_seed404_labeling_gallery.py

echo ""
echo "================================================================================"
echo "🎉 ALL PHASES COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
