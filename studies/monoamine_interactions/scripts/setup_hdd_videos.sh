#!/bin/bash
# setup_hdd_videos.sh
# Copies OFT videos from network drive to HDD and creates flattened symlinks.
# Target: /mnt/hdd/mono_inter/jonathan_oft_baseline

SOURCE_DIR="/mnt/neuropharm/Common/Lab member folder/Jonathan Korn/1st_cohort/OFT"
TARGET_ROOT="/mnt/hdd/mono_inter/jonathan_oft_baseline"
LINK_DIR="$TARGET_ROOT/all_videos"

echo "=== Video Setup Script ==="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_ROOT"
echo "Links:  $LINK_DIR"
echo "=========================="

if [ "$1" != "run" ]; then
    echo "Usage: ./setup_hdd_videos.sh run"
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
do_cmd "$1" "mkdir -p \"$TARGET_ROOT\""

# 2. RSYNC Videos
#   --include='*/'     : Enter all directories
#   --include='*.avi'  : Include avi files
#   --exclude='*'      : Exclude everything else (files that are not avi)
#   --prune-empty-dirs : Don't create empty directories
echo "Step 1: Rsyncing videos (this may take time)..."
do_cmd "$1" "rsync -av --include='*/' --include='*.avi' --exclude='*' --prune-empty-dirs \"$SOURCE_DIR/\" \"$TARGET_ROOT/\""

# 3. Create Flattened Symlinks
echo "Step 2: Creating symbolic links..."
do_cmd "$1" "mkdir -p \"$LINK_DIR\""

# Determine logic for iterating
# We want to find all .avi files in TARGET_ROOT (excluding all_videos itself)
# and link them into LINK_DIR with flattened names.

if [ "$1" == "run" ]; then
    find "$TARGET_ROOT" -type f -name "*.avi" | while read -r filepath; do
        # Ignore files already inside LINK_DIR to avoid recursion loops
        if [[ "$filepath" == "$LINK_DIR"* ]]; then
            continue
        fi

        # Get relative path from TARGET_ROOT
        relpath="${filepath#$TARGET_ROOT/}"
        
        # Replace slashes with underscores for flat name
        # e.g. day1/T1.../Vid.avi -> day1_T1..._Vid.avi
        flatname=$(echo "$relpath" | sed 's/\//_/g')
        
        target_link="$LINK_DIR/$flatname"
        
        if [ ! -L "$target_link" ]; then
            ln -s "$filepath" "$target_link"
            echo "Linked: $flatname"
        fi
    done
else
    echo "[DRY] find \"$TARGET_ROOT\" -type f -name \"*.avi\" ..."
    echo "[DRY] For each file: ln -s \$filepath \"$LINK_DIR/\$(basename_with_parents)\""
fi

echo "Done."
