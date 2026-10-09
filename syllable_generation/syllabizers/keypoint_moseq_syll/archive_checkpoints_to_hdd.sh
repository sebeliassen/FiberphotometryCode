#!/usr/bin/env bash
# ==============================================================================
# Script: archive_checkpoints_to_hdd.sh
# Purpose: Safely rsync large checkpoint.h5 files (~83 GB each) from NVMe to HDD,
#          verify file size and HDF5 integrity, remove NVMe copy, and symlink back.
# ==============================================================================

set -eo pipefail

WORKSPACE_PROJECT="/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2"
ARCHIVE_BASE="/mnt/hdd/old_projects/kpms_checkpoints_archive/karen_project_2"
PYTHON_EXEC="/home/fsp585/miniforge3/envs/keypoint_moseq/bin/python"

MODELS=(
    "final_karen_model_k100k"
    "final_karen_model_k130k"
    "final_karen_model_k150k"
    "final_karen_model_k180k"
    "final_karen_model_k200k"
    "final_karen_model_k240k"
)

echo "========================================================================"
echo " Starting Checkpoint Archiving to HDD"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Source Workspace: $WORKSPACE_PROJECT"
echo " Destination Base: $ARCHIVE_BASE"
echo "========================================================================"

mkdir -p "$ARCHIVE_BASE"

TOTAL_MODELS=${#MODELS[@]}
CURRENT_IDX=0

for MODEL in "${MODELS[@]}"; do
    CURRENT_IDX=$((CURRENT_IDX + 1))
    SRC_DIR="$WORKSPACE_PROJECT/$MODEL"
    SRC_CKPT="$SRC_DIR/checkpoint.h5"
    DST_DIR="$ARCHIVE_BASE/$MODEL"
    DST_CKPT="$DST_DIR/checkpoint.h5"

    echo ""
    echo "========================================================================"
    echo " [$CURRENT_IDX/$TOTAL_MODELS] Processing Model: $MODEL"
    echo " Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"

    if [ ! -f "$SRC_CKPT" ] && [ ! -L "$SRC_CKPT" ]; then
        echo "[SKIP] Source checkpoint not found: $SRC_CKPT"
        continue
    fi

    if [ -L "$SRC_CKPT" ]; then
        TARGET=$(readlink -f "$SRC_CKPT")
        echo "[SKIP] Source is already a symlink -> $TARGET"
        continue
    fi

    SRC_SIZE_BYTES=$(stat -c%s "$SRC_CKPT")
    SRC_SIZE_GB=$(awk -v size="$SRC_SIZE_BYTES" 'BEGIN { printf "%.2f", size / 1073741824 }')
    echo " Source File Size: $SRC_SIZE_GB GB ($SRC_SIZE_BYTES bytes)"
    echo " Target: $DST_CKPT"

    mkdir -p "$DST_DIR"

    echo ">>> Running rsync with progress..."
    rsync --progress -av "$SRC_CKPT" "$DST_CKPT"

    echo ">>> Verifying byte size match..."
    DST_SIZE_BYTES=$(stat -c%s "$DST_CKPT")
    if [ "$SRC_SIZE_BYTES" -ne "$DST_SIZE_BYTES" ]; then
        echo "ERROR: File size mismatch for $MODEL! Source: $SRC_SIZE_BYTES, Dest: $DST_SIZE_BYTES"
        exit 1
    fi
    echo "    Size verified: $DST_SIZE_BYTES bytes matching."

    echo ">>> Verifying destination HDF5 structure & snapshots..."
    "$PYTHON_EXEC" -c "
import h5py, sys
try:
    with h5py.File('$DST_CKPT', 'r') as f:
        assert 'model_snapshots' in f, 'Missing model_snapshots'
        snaps = len(f['model_snapshots'])
        assert snaps > 0, 'No snapshots found'
        print(f'    HDF5 verification OK: {snaps} snapshots detected.')
except Exception as e:
    print(f'ERROR: Corrupt HDF5: {e}', file=sys.stderr)
    sys.exit(1)
"

    echo ">>> Replacing source file with symlink..."
    rm -f "$SRC_CKPT"
    ln -s "$DST_CKPT" "$SRC_CKPT"

    echo ">>> Testing symlink read access..."
    "$PYTHON_EXEC" -c "
import h5py, sys
try:
    with h5py.File('$SRC_CKPT', 'r') as f:
        _ = list(f.keys())
    print('    Symlink read test passed successfully.')
except Exception as e:
    print(f'ERROR: Symlink failed read test: {e}', file=sys.stderr)
    sys.exit(1)
"

    ls -la "$SRC_CKPT"
    echo ">>> Successfully archived and symlinked $MODEL at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================================================"
echo " ARCHIVING COMPLETE! Summary of Disk Storage:"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
df -h / /mnt/hdd
echo "========================================================================"
