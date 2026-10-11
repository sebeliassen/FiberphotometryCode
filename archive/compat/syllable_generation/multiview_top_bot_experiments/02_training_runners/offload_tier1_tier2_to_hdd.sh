#!/usr/bin/env bash
# =============================================================================
# offload_tier1_tier2_to_hdd.sh
# Safely migrates Tier 1 & Tier 2 KPMS directories from NVMe SSD to /mnt/hdd
# and leaves transparent symlinks on the SSD so all existing paths continue
# to function identically.
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/hdd_offload_${TIMESTAMP}.log"
LOG_LINK="$LOG_DIR/hdd_offload.log"

# Create/update symlink to latest log
ln -sfn "$LOG_FILE" "$LOG_LINK"

# Redirect stdout and stderr to both the log file and the console
exec > >(tee -a "$LOG_FILE") 2>&1

NTFY_URL="https://ntfy.sh/woody_videos_seb"

notify() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    local tags="${4:-package}"
    curl -s -H "Title: $title" -H "Priority: $priority" -H "Tags: $tags" -d "$msg" "$NTFY_URL" >/dev/null 2>&1 || true
}

BASE_SRC="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll"
BASE_DEST="/mnt/hdd/old_projects/kpms_checkpoints_archive"

echo "================================================================================"
echo "🚀 STARTING TIER 1 & TIER 2 MIGRATION TO /mnt/hdd"
echo "  Date & Time:    $(date)"
echo "  Log File:       $LOG_FILE"
echo "  Symlink:        $LOG_LINK"
echo "  Source Base:    $BASE_SRC"
echo "  Target Base:    $BASE_DEST"
echo "================================================================================"
echo "Initial NVMe Disk Status:"
df -h /
echo "HDD Storage Status:"
df -h /mnt/hdd
echo "================================================================================"

notify "NVMe -> HDD Migration Started 🚚" "Beginning migration of ~570 GB of KPMS archives to /mnt/hdd." "default" "truck,storage"

# Verify HDD is writable
mkdir -p "$BASE_DEST"
if [ ! -w "$BASE_DEST" ]; then
    echo "❌ FATAL: Destination $BASE_DEST is not writable!"
    notify "Migration FAILED" "Destination $BASE_DEST is not writable." "high" "x"
    exit 1
fi

# List of folders to migrate (order: smallest/fastest first to free immediate space)
# 1. karen_project_2 (~20 GB)
# 2. amph_project_8kp (~38 GB)
# 3. demo_project (~171 GB)
# 4. amph_project (~344 GB)
FOLDERS=(
    "karen_project_2"
    "amph_project_8kp"
    "demo_project"
    "amph_project"
)

TOTAL_FOLDERS=${#FOLDERS[@]}
INDEX=0

for FOLDER in "${FOLDERS[@]}"; do
    INDEX=$((INDEX + 1))
    SRC="$BASE_SRC/$FOLDER"
    DEST="$BASE_DEST/$FOLDER"

    echo ""
    echo "================================================================================"
    echo "[$INDEX/$TOTAL_FOLDERS] Processing '$FOLDER'..."
    echo "================================================================================"

    if [ ! -e "$SRC" ]; then
        echo "⚠️ Source $SRC does not exist. Skipping."
        continue
    fi

    if [ -L "$SRC" ]; then
        echo "ℹ️ $SRC is already a symlink -> $(readlink -f "$SRC"). Skipping."
        continue
    fi

    SRC_SIZE="$(du -sh "$SRC" | cut -f1)"
    echo "  Source size: $SRC_SIZE"
    echo "  Copying to:  $DEST"
    t_start=$(date +%s)

    mkdir -p "$DEST"

    # Rsync with stats and progress
    echo "  Running rsync (preserving all attributes, permissions, timestamps)..."
    rsync -avh --progress "$SRC/" "$DEST/"

    RSYNC_STATUS=$?
    t_end=$(date +%s)
    duration=$((t_end - t_start))

    if [ $RSYNC_STATUS -ne 0 ]; then
        echo "❌ FATAL: rsync for $FOLDER failed with code $RSYNC_STATUS!"
        notify "Migration Error: $FOLDER" "rsync failed with exit code $RSYNC_STATUS. Source intact." "high" "warning"
        exit 1
    fi

    # Verification: Ensure destination is non-empty
    DEST_COUNT=$(find "$DEST" -mindepth 1 | wc -l)
    if [ "$DEST_COUNT" -le 0 ]; then
        echo "❌ FATAL: Destination $DEST is empty after rsync! Aborting removal."
        notify "Migration Verification Failed" "Destination $DEST is empty. Source intact." "urgent" "x"
        exit 1
    fi

    DEST_SIZE="$(du -sh "$DEST" | cut -f1)"
    echo "  ✅ Verification passed: $DEST contains $DEST_COUNT items ($DEST_SIZE)."
    echo "  Transfer completed in $((duration / 60)) min $((duration % 60)) sec."

    # Remove NVMe source directory and replace with transparent symlink
    echo "  Removing original NVMe folder to reclaim space..."
    rm -rf "$SRC"

    echo "  Creating transparent symlink: $SRC -> $DEST..."
    ln -sfn "$DEST" "$SRC"

    # Verify symlink resolves cleanly
    if [ -d "$SRC" ]; then
        echo "  ✅ Symlink verified: $SRC resolves to $(readlink -f "$SRC")"
    else
        echo "❌ FATAL: Symlink $SRC is broken!"
        notify "Symlink Broken: $FOLDER" "Symlink does not resolve to directory." "urgent" "x"
        exit 1
    fi

    FREE_SPACE="$(df -h / | awk 'NR==2 {print $4}')"
    echo "  🎉 Successfully offloaded '$FOLDER'! NVMe available space now: $FREE_SPACE"
    notify "Offloaded $FOLDER ($SRC_SIZE)" "Transferred in $((duration / 60))m. NVMe free: $FREE_SPACE" "default" "white_check_mark"
done

echo ""
echo "================================================================================"
echo "🎉 ALL TIER 1 & TIER 2 MIGRATIONS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Final NVMe Disk Status:"
df -h /
echo "HDD Storage Status:"
df -h /mnt/hdd
echo "================================================================================"

FINAL_FREE="$(df -h / | awk 'NR==2 {print $4}')"
notify "Migration Complete! 🎉" "All folders offloaded. NVMe free space is now $FINAL_FREE (was 67 GB)." "default" "tada,storage"
