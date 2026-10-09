#!/bin/bash
#
# This script finds all .mp4 files in the current folder
# and uses ffprobe to count the total number of frames in each video.

echo "--- Analyzing MP4 files ---"

# A counter for the number of files found
file_count=0

# Loop through all files ending in .mp4
for file in *.mp4; do
  # Check if a matching file actually exists
  if [ -f "$file" ]; then
    # Use ffprobe to get the frame count from the video stream's metadata.
    frame_count=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "$file")
    
    # Check if frame_count is a valid number
    if [[ -n "$frame_count" && "$frame_count" =~ ^[0-9]+$ ]]; then
        echo "File: $file, Frames: $frame_count"
    else
        echo "File: $file, Frames: Could not determine frame count."
    fi
    
    file_count=$((file_count + 1))
  fi
done

# If no files were found, print a message
if [ "$file_count" -eq 0 ]; then
  echo "No .mp4 files found in this directory."
fi

echo "--- Done ---"
