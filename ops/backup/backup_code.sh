#!/bin/bash
# Compatibility entry point: local HDD backup, explicit exclusions, no notifications.
set -euo pipefail
SOURCE_DIR="${BACKUP_SOURCE_DIR:-/home/fsp585/Desktop/FiberphotometryCode}"
DEST_DIR="${BACKUP_DEST_DIR:-/mnt/hdd/backups/FiberphotometryCode}"
PYTHON="${BACKUP_PYTHON:-python3}"
exec "$PYTHON" "$SOURCE_DIR/ops/backup/snapshot.py" create \
    --root "$SOURCE_DIR" --destination "$DEST_DIR" \
    --policy "$SOURCE_DIR/configs/backup.yaml" "$@"
