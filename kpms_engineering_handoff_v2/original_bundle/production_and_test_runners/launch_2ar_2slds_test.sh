#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/test_2ar_2slds_cohort.log"
rm -f "$LOG_FILE"

echo "================================================================================"
echo "🔬 LAUNCHING 2 AR + 2 SLDS EMPIRICAL VALIDATION (FULL 118 SESSIONS)"
echo "================================================================================"
echo "  Date:       $(date)"
echo "  Python:     $PYTHON_BIN"
echo "  Log File:   $LOG_FILE"
echo "================================================================================"

nohup "$PYTHON_BIN" -u "$SCRIPT_DIR/run_cohort_2ar_2slds_test.py" > "$LOG_FILE" 2>&1 &
TEST_PID=$!
disown -h "$TEST_PID"

echo "✅ Test Runner launched successfully (PID: $TEST_PID)!"
echo "   Monitor with: tail -f $LOG_FILE"
