#!/bin/bash
# rerun_all_plots.sh
# This script reruns all 3 plot types after the signal_meta extraction fix.

export PYTHONPATH=/home/fsp585/Desktop/FiberphotometryCode
PYTHON_EXEC=/home/fsp585/miniforge3/envs/kpms_gpu/bin/python
BASE_DIR=/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/deep_research_experiments

echo "========================================================"
echo "1. Re-extracting dual-fiber traces for Standard Projects"
echo "========================================================"
$PYTHON_EXEC $BASE_DIR/peth_lateralized_da.py

echo "========================================================"
echo "2. Generating Standard Lateralization Plots"
echo "========================================================"
$PYTHON_EXEC $BASE_DIR/plot_lateralized_da.py

echo "========================================================"
echo "3. Generating Regional Side-by-Side Plots (DLS vs DMS)"
echo "========================================================"
$PYTHON_EXEC $BASE_DIR/plot_lateralized_da_regions.py

echo "========================================================"
echo "4. Running Optimized Flip Extraction & Plots"
echo "========================================================"
$PYTHON_EXEC $BASE_DIR/peth_lateralized_da_optimized.py

echo "========================================================"
echo "5. Generating Virtual Hemisphere Averages (Flipped + Non-Flipped)"
echo "========================================================"
$PYTHON_EXEC $BASE_DIR/peth_lateralized_da_virtual.py

echo "========================================================"
echo "All extraction and plotting pipelines are complete!"
echo "========================================================"
