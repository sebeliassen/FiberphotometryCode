#!/usr/bin/env bash
set -euo pipefail

# Directory with your notebook and panels/
cd /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/replications/datta

# CPU limit per notebook run (500% = up to ~5 cores worth of CPU time per job)
CPU_QUOTA_PER_JOB=500

run_nb() {
  local variant="$1"
  local mode="$2"

  echo "Running 1e.ipynb with DATA_VARIANT=$variant, SCALAR_MODE=$mode"

  systemd-run --user --scope --quiet \
    -p CPUQuota=${CPU_QUOTA_PER_JOB}% \
    env DATA_VARIANT="$variant" SCALAR_MODE="$mode" \
      jupyter nbconvert --to notebook --execute panels/1e.ipynb \
        --output "1e_${variant}_${mode}.ipynb"
}

# Check that systemd-run is available
if ! command -v systemd-run >/dev/null 2>&1; then
  echo "Error: systemd-run not found. This script requires systemd (user mode) for CPU limiting." >&2
  exit 1
fi

########## FIRST SET (2 in parallel) ##########
for mode in mean absmean; do
  run_nb "raw" "$mode" &
done

# Wait for the first two to finish
wait
echo "First set (raw, mean/absmean) finished."

########## SECOND SET (2 in parallel) ##########
for mode in mean absmean; do
  run_nb "denoised" "$mode" &
done

# Wait for the second two to finish
wait
echo "Second set (denoised, mean/absmean) finished."
echo "All notebook runs finished."
