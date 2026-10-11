#!/usr/bin/env bash
# =============================================================================
# launch_eml_evaluation.sh
# Cgroup-Isolated EML Model Selection Evaluation for Continuous-Calibrated Cohort
# =============================================================================
set -euo pipefail

repo="/home/fsp585/Desktop/FiberphotometryCode"
python="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
script="$repo/syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py"
project="$repo/syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_continuous_calibrated"
output="$repo/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures"
log="$output/eml_evaluation_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$output"

echo "================================================================================"
echo "🚀 LAUNCHING CGROUP-ISOLATED EML EVALUATION (RTX 4090, 5 SEEDS)"
echo "  Date:       $(date)"
echo "  Project:    $project"
echo "  Output:     $output"
echo "  Log File:   $log"
echo "================================================================================"

# Ensure previous unit is not active
systemctl --user stop kpms-eml.service 2>/dev/null || true

# Run under isolated transient systemd unit with MemoryMax=60G
systemd-run --user --unit="kpms-eml" --service-type=simple \
    -p MemoryMax=60G -p MemorySwapMax=0 -p OOMPolicy=kill -p Restart=no \
    -p StandardOutput=journal -p StandardError=journal \
    "$python" -u "$script" \
    --project-dir "$project" \
    --output-dir "$output"

# Stream journal logs to file and stdout
journalctl --user -u "kpms-eml.service" -f --no-pager | tee "$log" &
LOG_PID=$!

# Wait for service to finish
while systemctl --user is-active --quiet "kpms-eml.service"; do
    sleep 3
done
kill "$LOG_PID" 2>/dev/null || true

# Check exit status
RESULT=$(systemctl --user show "kpms-eml.service" -p Result --value)
EXIT_CODE=$(systemctl --user show "kpms-eml.service" -p ExecMainStatus --value)

echo ""
echo "================================================================================"
if [ "$RESULT" = "success" ] && [ "$EXIT_CODE" = "0" ]; then
    echo "🎉 EML EVALUATION FINISHED SUCCESSFULLY!"
    echo "================================================================================"
    exit 0
else
    echo "❌ EML EVALUATION FAILED: Result=$RESULT, ExitCode=$EXIT_CODE"
    echo "Check log: $log"
    echo "================================================================================"
    exit 1
fi
