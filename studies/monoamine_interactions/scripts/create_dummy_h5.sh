#!/bin/bash
# create_dummy_h5.sh
# Creates dummy H5 files based on video files to satisfy the session discovery logic.

VIDEO_DIR="/home/fsp585/Desktop/FiberphotometryCode/src_mono_inter/jonathan_oft_baseline/video_files"
H5_DIR="/home/fsp585/Desktop/FiberphotometryCode/src_mono_inter/jonathan_oft_baseline/h5_files"

echo "Creating dummy H5s in $H5_DIR from videos in $VIDEO_DIR"

# Regex pattern for video: Arena..._Camset..._Timestamp.avi
# We want: Arena..._Camset..._Timestamp_TokenDLC_resnet50.h5
# Example Video: Arena1234_CenterTopCamVid_2026-01-20T15_07_45.avi
# Example H5:    Arena1234_CenterTopCamVid_2026-01-20T15_07_45_dummyDLC_resnet50.h5

# Note: The loader expects regex: ...(?P<token>.+?)DLC_.*\.h5$
# So we need 'DLC_' in the name.

find "$VIDEO_DIR" -name "*.avi" | while read -r vidpath; do
    vidname=$(basename "$vidpath")
    # Strip extension
    base="${vidname%.*}"
    
    # Create valid H5 name
    h5name="${base}_dummyDLC_resnet50.h5"
    
    touch "$H5_DIR/$h5name"
    echo "Created: $h5name"
done
