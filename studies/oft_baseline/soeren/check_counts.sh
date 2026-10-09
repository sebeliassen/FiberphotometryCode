#!/usr/bin/env bash
set -euo pipefail

# Configurable tolerances
PCT_TOL=${PCT_TOL:-0.5}   # percent
ABS_TOL=${ABS_TOL:-5}     # frames

header() {
  printf "%-8s %12s %12s %10s %9s  %s\n" "T#" "MP4_FRAMES" "CSV_FRAMES" "DIFF" "PCT(%)" "STATUS"
  printf "%-8s %12s %12s %10s %9s  %s\n" "--------" "------------" "------------" "----------" "---------" "------"
}

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# FAST probe order:
# 1) ffprobe nb_frames (container metadata)
# 2) mediainfo FrameCount (if available)
# 3) ffprobe nb_read_packets (heuristic: often ~ frame count; no decode)
mp4_frames_fast() {
  local f="$1" v

  # 1) nb_frames from container metadata
  v="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
      -of default=nokey=1:noprint_wrappers=1 "$f" 2>/dev/null || true)"
  if [[ -n "$v" && "$v" != "N/A" && "$v" != "0" ]]; then
    echo "$v"; return 0
  fi

  # 2) mediainfo (if installed)
  if command -v mediainfo >/dev/null 2>&1; then
    v="$(mediainfo --Inform='Video;%FrameCount%' "$f" 2>/dev/null || true)"
    if [[ -n "$v" && "$v" != "N/A" && "$v" != "0" ]]; then
      echo "$v"; return 0
    fi
  fi

  # 3) count packets (no full decode; quick heuristic)
  v="$(ffprobe -v error -count_packets -select_streams v:0 \
      -show_entries stream=nb_read_packets \
      -of default=nokey=1:noprint_wrappers=1 "$f" 2>/dev/null || true)"
  if [[ -n "$v" && "$v" != "N/A" && "$v" != "0" ]]; then
    echo "$v"; return 0
  fi

  # Give up quickly — no slow fallbacks
  echo "NA"
  return 1
}

csv_frames_minus_header() {
  local csv="$1" lines
  lines=$(wc -l < "$csv")
  (( lines > 0 )) && echo $((lines - 1)) || echo 0
}

status_line() {
  local tag="$1" vf="$2" cf="$3"

  if ! is_int "$vf"; then
    printf "%-8s %12s %12d %10s %9s  %s\n" "$tag" "-" "$cf" "-" "-" "NO_VIDEO_COUNT"
    return
  fi

  local diff=$(( vf - cf ))
  local adiff=$(( diff<0 ? -diff : diff ))
  local pct="0.0"
  if (( vf > 0 )); then
    pct="$(awk -v a="$adiff" -v b="$vf" 'BEGIN{printf "%.3f", (a*100.0)/b}')"
  fi

  local pct_thresh
  pct_thresh="$(awk -v p="$PCT_TOL" -v b="$vf" 'BEGIN{printf "%.3f", (p*b)/100.0}')"
  local pct_thresh_int
  pct_thresh_int="$(awk -v x="$pct_thresh" 'BEGIN{printf "%d", (x==int(x)?x:int(x)+1)}')"
  local tol=$(( ABS_TOL > pct_thresh_int ? ABS_TOL : pct_thresh_int ))

  local status="OK"
  (( adiff > tol )) && status="MISMATCH"

  printf "%-8s %12d %12d %10d %9s  %s\n" "$tag" "$vf" "$cf" "$diff" "$pct" "$status"
}

main() {
  shopt -s nullglob
  header
  local any=0
  for mp4 in video_files/T*_*.mp4; do
    any=1
    local base tag
    base="$(basename "$mp4")"
    if [[ "$base" =~ ^(T[0-9]+)_ ]]; then
      tag="${BASH_REMATCH[1]}"
    else
      printf "WARN: Could not parse tag from %s\n" "$base" >&2
      continue
    fi

    # Match csvs/<tag_*>/TrackingData_*.csv (prefer *Salient*)
    local csv
    csv=(csvs/"${tag}"_*/TrackingData_*.csv)
    if [[ ${#csv[@]} -eq 0 ]]; then
      printf "%-8s %12s %12s %10s %9s  %s\n" "$tag" "-" "-" "-" "-" "NO_CSV_FOUND"
      continue
    fi
    if [[ ${#csv[@]} -gt 1 ]]; then
      for c in "${csv[@]}"; do
        if [[ "$c" == *"TrackingData_Salient_"* ]]; then csv=("$c"); break; fi
      done
    fi
    csv="${csv[0]}"

    # Fast probes only
    vf="$(mp4_frames_fast "$mp4")" || true
    cf="$(csv_frames_minus_header "$csv")"

    status_line "$tag" "$vf" "$cf"
  done

  (( any == 0 )) && echo "No mp4 files found in video_files/. Check your paths."
}

main "$@"
