#!/bin/bash

# Master Replication Pipeline for Panel 1f
# Runs AMPH and IFDN pipelines sequentially to prevent OOM errors.
# Usage: ./run_replication_pipeline.sh <suffix>

SUFFIX=$1

if [ -z "$SUFFIX" ]; then
    echo "Usage: $0 <suffix> [midfi|hifi|std]"
    exit 1
fi

CONDA_ENV="kpms_gpu"
REPO_ROOT="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation"
DATTA_DIR="$REPO_ROOT/analysis/replications/datta"

echo "=========================================================="
echo "Starting Full Replication Pipeline with suffix: $SUFFIX"
echo "=========================================================="

# 1. AMPH Pipeline
echo "[1/6] Running AMPH Extraction..."
conda run -n $CONDA_ENV python $DATTA_DIR/run_panel_1f.py

echo "[2/6] Running AMPH Digestion ($SUFFIX)..."
conda run -n $CONDA_ENV python $DATTA_DIR/digest_panel_1f.py $SUFFIX

echo "[3/6] Running AMPH Plotting (suffix: $SUFFIX)..."
conda run -n $CONDA_ENV python $DATTA_DIR/plot_panel_1f.py $SUFFIX

# 2. IFDN Pipeline
echo "[4/6] Running IFDN Extraction..."
conda run -n $CONDA_ENV python $DATTA_DIR/run_panel_1f_ifdn.py

echo "[5/6] Running IFDN Digestion ($SUFFIX)..."
conda run -n $CONDA_ENV python $DATTA_DIR/digest_panel_1f_ifdn.py $SUFFIX

echo "[6/6] Running IFDN Plotting (suffix: $SUFFIX)..."
conda run -n $CONDA_ENV python $DATTA_DIR/plot_panel_1f_ifdn.py $SUFFIX

echo "=========================================================="
echo "Pipeline Complete!"
echo "AMPH Plots: analysis/replications/datta/output/panel_1f_amph/plots_$SUFFIX/"
echo "IFDN Plots: analysis/replications/datta/output/panel_1f_ifdn/plots_$SUFFIX/"
echo "=========================================================="
