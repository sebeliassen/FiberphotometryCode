#!/bin/bash

# Define variables
SOURCE_DIR="/home/fsp585/Desktop/FiberphotometryCode"
DEST_DIR="/mnt/neuropharm/Common/Lab member folder/Sebastian/code_backup"
DATE=$(date +%Y-%m-%d_%H-%M)
ZIP_NAME="code_backup_$DATE.zip"

echo "[$(date)] Starting backup for $DATE..."

# Go to the source directory
cd "$SOURCE_DIR" || { echo "Failed to cd to $SOURCE_DIR"; exit 1; }

# Check if destination directory exists (is mounted)
if [ ! -d "$DEST_DIR" ]; then
    echo "[$(date)] ERROR: Destination directory $DEST_DIR does not exist. Is the drive mounted?"
    NTFY_TOPIC="woody_videos_seb"
    curl -H "Priority: urgent" -H "Tags: rotating_light,warning" -d "Backup FAILED: Network drive not mounted!" "ntfy.sh/$NTFY_TOPIC" || true
    exit 1
fi

# Find and zip the files, suppressing output, saving directly to the destination
find . -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" -o ! -name "*.*" \) \
    -size -1048576c -not -path "*/.*" -not -path "*/__pycache__/*" | \
    zip -q "$DEST_DIR/$ZIP_NAME" -@

BACKUP_STATUS=$?
NTFY_TOPIC="woody_videos_seb"

if [ $BACKUP_STATUS -eq 0 ]; then
    echo "[$(date)] Backup successfully saved to: $DEST_DIR/$ZIP_NAME"
    curl -d "Code backup successful: $ZIP_NAME" "ntfy.sh/$NTFY_TOPIC" || true
else
    echo "[$(date)] Backup failed with exit code $BACKUP_STATUS."
    curl -H "Priority: high" -H "Tags: warning" -d "Code backup FAILED! (Exit code: $BACKUP_STATUS)" "ntfy.sh/$NTFY_TOPIC" || true
fi
