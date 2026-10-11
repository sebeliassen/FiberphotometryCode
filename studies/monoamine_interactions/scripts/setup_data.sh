#!/bin/bash
# setup_data.sh
# Mirrors CSVs from the network drive and creates symlinks for HDD videos.
# Usage: ./setup_data.sh [run]

SOURCE_DIR="/mnt/neuropharm/Common/Lab member folder/Jonathan Korn/1st_cohort/OFT"
DEST_ROOT="/home/fsp585/Desktop/FiberphotometryCode/src_mono_inter/jonathan_oft_baseline"
HDD_VIDEOS="/mnt/hdd/mono_inter/jonathan_oft_baseline/all_videos"

# Subdirectories
CSV_DEST="$DEST_ROOT/csvs"
VIDEO_DEST="$DEST_ROOT/video_files"
H5_DEST="$DEST_ROOT/h5_files"

# Default to DRY RUN
DRY_RUN=true
if [ "$1" == "run" ]; then
    DRY_RUN=false
    echo "=== EXECUTING LIVE RUN ==="
else
    echo "=== DRY RUN (No changes will be made) ==="
fi

do_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY] $@"
    else
        eval "$@"
    fi
}

# 1. Create Directories
echo "Step 0: Creating directories..."
do_cmd mkdir -p "$CSV_DEST"
do_cmd mkdir -p "$VIDEO_DEST"
do_cmd mkdir -p "$H5_DEST"

# 2. Mirror CSVs (Preserve structure)
echo "Step 1: Mirroring CSVs from network drive..."
# --include='*/'     : Mirror directory structure
# --include='*.csv'  : Copy CSV files
# --exclude='*'      : Ignore everything else
do_cmd rsync -av --include='*/' --include='*.csv' --exclude='*' --prune-empty-dirs \"$SOURCE_DIR/\" \"$CSV_DEST/\"

# 3. Create Symlinks for Videos (from HDD)
echo "Step 2: Linking videos from HDD..."
if [ -d "$HDD_VIDEOS" ]; then
    do_cmd ln -sf \"$HDD_VIDEOS/*.avi\" \"$VIDEO_DEST/\"
else
    echo "WARNING: HDD video folder not found: $HDD_VIDEOS"
    echo "Did you run setup_videos.sh yet?"
fi

echo ""
echo "Done. Paths are now set up for build_sessions.py."
