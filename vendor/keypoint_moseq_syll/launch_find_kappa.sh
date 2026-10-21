#!/bin/bash
cd /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll
source /home/fsp585/miniforge3/etc/profile.d/conda.sh
conda activate kpms_gpu_cuml
nohup python -u find_kappa.py > find_kappa.log 2>&1 &
PID=$!
disown $PID
echo "Started find_kappa.py in background with PID $PID"
