#!/bin/bash

# Karen Project: Rsync new timepoints from N-drive to HDD
# Skips bottom camera videos to save space.
# Sends ntfy notifications upon progress/completion.

NTFY_URL="https://ntfy.sh/woody_videos_seb"
SRC_BASE="/mnt/neuropharm/Common/Lab member folder/Jamila Lilja/Alpha-synuclein PFF mice/Behavior cohort 1/a-syn_Fiber photo"
DST_BASE="/mnt/hdd/karen_project"

# Array of source folder names and corresponding local target folder names
declare -A TIMEPOINTS
TIMEPOINTS=(
    ["3 week PI"]="3_week_PI"
    ["8 weeks PI"]="8_weeks_PI"
    ["24 weeks PI"]="24_weeks_PI"
    ["L-dopa+Benserazide -fiber photometry"]="L_dopa"
)

# Function to send ntfy notification
notify() {
    local message="$1"
    echo "$message"
    curl -H "Title: Karen Rsync" -d "$message" "$NTFY_URL"
}

# Ensure destination base exists
mkdir -p "$DST_BASE"

notify "Starting Rsync of Karen project data timepoints..."

# Loop through each timepoint and run rsync
for src_folder in "${!TIMEPOINTS[@]}"; do
    dst_folder="${TIMEPOINTS[$src_folder]}"
    
    src_path="${SRC_BASE}/${src_folder}/"
    dst_path="${DST_BASE}/${dst_folder}/"
    
    notify "Rsyncing [${src_folder}] -> [${dst_folder}]..."
    
    # Run rsync:
    # -a: archive mode (preserves permissions, times, etc.)
    # -v: verbose
    # --progress: show progress
    # --partial: keep partially transferred files
    # --exclude="*BotCamVid*.avi": exclude bottom camera video files
    rsync -av --progress --partial --exclude="*BotCamVid*.avi" "$src_path" "$dst_path" >> "${DST_BASE}/rsync_${dst_folder}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        notify "✅ Rsync of [${src_folder}] completed successfully!"
    else
        notify "❌ Rsync of [${src_folder}] failed! Check log: ${DST_BASE}/rsync_${dst_folder}.log"
    fi
done

notify "Rsync process completed for all 4 new timepoints!"
