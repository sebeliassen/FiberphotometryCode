#!/bin/bash
# setup_videos.sh
# Copies OFT videos from network drive to HDD and creates flattened symlinks.
#
# Source: /mnt/neuropharm/.../OFT
# Target: /mnt/hdd/mono_inter/jonathan_oft_baseline
# Links:  /mnt/hdd/mono_inter/jonathan_oft_baseline/all_videos

# --- Configuration ---
SOURCE_DIR="/mnt/neuropharm/Common/Lab member folder/Jonathan Korn/1st_cohort/OFT"
TARGET_ROOT="/mnt/hdd/mono_inter/jonathan_oft_baseline"
LINK_DIR="$TARGET_ROOT/all_videos"

echo "=== Video Setup Script ==="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_ROOT"
echo "Links:  $LINK_DIR"
echo "=========================="

# NEW: Disk space check
TARGET_PARTITION=$(df -P "$TARGET_ROOT" | tail -1 | awk '{print $1}')
FREE_SPACE=$(df -h "$TARGET_ROOT" | tail -1 | awk '{print $4}')
echo "Target partition: $TARGET_PARTITION"
echo "Free space:      $FREE_SPACE"
echo ""
# Default to DRY RUN unless 'run' is passed as the first argument
if [ "$1" != "run" ]; then
    echo "Usage: ./setup_videos.sh run"
    echo "Running in DRY MODE (echoing commands)"
fi

do_cmd() {
    if [ "$1" == "run" ]; then
        eval "$2"
    else
        echo "[DRY] $2"
    fi
}

# 1. Create Target Directory
# 'mkdir -p' automatically creates parent directories if they don't exist.
echo ""
echo "Step 0: Creating target directories..."
do_cmd "$1" "mkdir -p \"$TARGET_ROOT\""

# 2. RSYNC Videos
#   -a                 : Archive mode (recursive, preserves attributes)
#   -v                 : Verbose
#   -P                 : Show progress bar and partial files (resume capability)
#   -h                 : Human-readable sizes
#   --include='*/'     : Enter all directories
#   --include='*.avi'  : Include avi files
#   --exclude='*'      : Exclude everything else
#   --prune-empty-dirs : Don't create empty directories
echo ""
echo "Step 1: Rsyncing videos (this may take time)..."
do_cmd "$1" "rsync -avPh --include='*/' --include='*.avi' --exclude='*' --prune-empty-dirs \"$SOURCE_DIR/\" \"$TARGET_ROOT/\""

# 3. Create Flattened Symlinks
echo ""
echo "Step 2: Creating symbolic links in $LINK_DIR..."
do_cmd "$1" "mkdir -p \"$LINK_DIR\""

if [ "$1" == "run" ]; then
    # Find all AVIs in TARGET_ROOT, excluding the LINK_DIR itself
    find "$TARGET_ROOT" -type f -name "*.avi" | while read -r filepath; do
        # Skip if file is inside the link directory
        if [[ "$filepath" == "$LINK_DIR"* ]]; then
            continue
        fi

        # Get relative path from TARGET_ROOT
        # e.g. /mnt/hdd/mono_inter/jonathan_oft_baseline/day1/T1/vid.avi -> day1/T1/vid.avi
        relpath="${filepath#$TARGET_ROOT/}"
        
        # Replace slashes with underscores for flat name
        # day1/T1/vid.avi -> day1_T1_vid.avi
        flatname=$(echo "$relpath" | sed 's/\//_/g')
        
        target_link="$LINK_DIR/$flatname"
        
        if [ ! -L "$target_link" ]; then
            ln -s "$filepath" "$target_link"
            echo "Linked: $flatname"
        fi
    done
else
    echo "[DRY] find \"$TARGET_ROOT\" -type f -name \"*.avi\" ..."
    echo "[DRY] For each file: ln -s \$filepath \"$LINK_DIR/path_to_flat_name\""
fi

echo ""
echo "Done."
