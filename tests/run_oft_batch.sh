#!/bin/sh
set -eu

# If you ever want the script to self-activate, uncomment this portable block:
# CONDA_BASE="$(conda info --base 2>/dev/null || true)"
# if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
#   . "$CONDA_BASE/etc/profile.d/conda.sh"
#   conda activate keypoint_moseq
# fi

NB="oft_session_deep_analyzis.ipynb"
OUTDIR="runs"
PROJECT_DIR="./kpms_few_bp_150ms_frac_1_ld3_jn24"
MODEL_PREFIX="final_model_1.21e+04_idx"
LOAD_FILE="/home/fsp585/Desktop/FiberphotometryCode/tests/kpms_few_bp_150ms_frac_1_ld3_jn24.pkl"

# Models 0..9
MODELS="0 1 2 3 4 5 6 7 8 9"

mkdir -p "$OUTDIR"

# --- sanity checks & info ---
for cmd in papermill jupyter date; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' not found. Activate the correct conda env (e.g., 'conda activate keypoint_moseq') and try again." >&2
    exit 1
  fi
done
echo "Using papermill: $(command -v papermill)"
echo "Using jupyter:   $(command -v jupyter)"
echo "Python:          $(command -v python)"

human() {
  s=$1
  d=$((s/86400))
  h=$(( (s%86400)/3600 ))
  m=$(( (s%3600)/60 ))
  s=$(( s%60 ))
  printf "%dd %02dh %02dm %02ds" "$d" "$h" "$m" "$s"
}

START_TS=$(date +%s)
TOTAL=0; for m in $MODELS; do TOTAL=$((TOTAL+1)); done
COUNT=0; TOTAL_ELAPSED=0

echo "==== Batch start: $(date) ===="
echo "Notebook: $NB"
echo "Output dir: $OUTDIR"
echo "Models: $MODELS"
echo "Project dir: $PROJECT_DIR"
echo "Load file: $LOAD_FILE"
echo "================================"

for m in $MODELS; do
  COUNT=$((COUNT+1))
  RUN_START=$(date +%s)
  PCT=$((100*COUNT/TOTAL))
  MODEL_NAME="${MODEL_PREFIX}${m}"
  OUT_NB="$OUTDIR/out_idx${m}.ipynb"
  OUT_HTML_BASENAME="out_idx${m}.html"

  echo ""
  echo "---- [$COUNT/$TOTAL] Starting idx ${m} (${PCT}%) at $(date) ----"
  echo "Params: PROJECT_DIR='$PROJECT_DIR', MODEL_NAME='$MODEL_NAME', load_filename='$LOAD_FILE'"

  papermill "$NB" "$OUT_NB" \
    -p PROJECT_DIR "$PROJECT_DIR" \
    -p MODEL_NAME "$MODEL_NAME" \
    -p load_filename "$LOAD_FILE"

  # Use --output-dir to avoid path quirks
  jupyter nbconvert --to html "$OUT_NB" --output-dir "$OUTDIR" --output "$OUT_HTML_BASENAME"

  RUN_END=$(date +%s)
  DUR=$((RUN_END - RUN_START))
  TOTAL_ELAPSED=$((TOTAL_ELAPSED + DUR))
  AVG=$((TOTAL_ELAPSED / COUNT))
  REMAINING=$(((TOTAL - COUNT) * AVG))
  ETA_EPOCH=$(( $(date +%s) + REMAINING ))

  if date -d @"$ETA_EPOCH" >/dev/null 2>&1; then
    ETA_FMT=$(date -d @"$ETA_EPOCH" "+%Y-%m-%d %H:%M:%S %Z")
  else
    ETA_FMT=$(date -r "$ETA_EPOCH" "+%Y-%m-%d %H:%M:%S %Z")
  fi

  echo "Completed idx ${m} in $(human $DUR)"
  echo "Progress: $COUNT/$TOTAL (${PCT}%)"
  echo "Avg/run so far: $(human $AVG)"
  echo "Estimated remaining: $(human $REMAINING)"
  echo "Estimated finish by: $ETA_FMT"
  echo "Outputs: $OUT_NB, $OUTDIR/$OUT_HTML_BASENAME"
done

END_TS=$(date +%s)
TOTAL_DUR=$((END_TS - START_TS))
echo ""
echo "==== Batch done: $(date) ===="
echo "Total time: $(human $TOTAL_DUR)"
echo "All outputs in: $OUTDIR"
