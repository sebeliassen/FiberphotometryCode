#!/bin/bash

# Karen Project: Split videos for all timepoints using split_videos.sh
# Uses GPU acceleration (h264_nvenc) and sends ntfy notifications.

NTFY_URL="https://ntfy.sh/woody_videos_seb"
SPLIT_SCRIPT="/home/fsp585/Desktop/FiberphotometryCode/scripts/split_videos.sh"
DST_BASE="/mnt/hdd/karen_project"
OUT_DIR="${DST_BASE}/all_videos_split"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

notify() {
    local message="$1"
    echo "$message"
    curl -H "Title: Karen Video Split" -d "$message" "$NTFY_URL"
}

notify "Starting video splitting process for all timepoints..."

# Timepoints to process
TIMEPOINTS=("3_week_PI" "8_weeks_PI" "24_weeks_PI" "L_dopa")

for tp in "${TIMEPOINTS[@]}"; do
    notify "Splitting videos in timepoint [$tp]..."
    
    # Call split_videos.sh:
    # -o specifies the output directory
    # we target the subdirectory on HDD
    "$SPLIT_SCRIPT" -o "$OUT_DIR" "${DST_BASE}/${tp}" >> "${DST_BASE}/split_process_${tp}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        notify "✅ Splitting of [$tp] completed successfully!"
    else
        notify "❌ Splitting of [$tp] encountered errors! Check log: ${DST_BASE}/split_process_${tp}.log"
    fi
done

notify "Video splitting process completed for all 4 timepoints!"
