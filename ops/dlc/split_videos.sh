#!/bin/bash

# ================================
# Configuration
# ================================

# Define buffer percentage (e.g., 5 for 5% overlap)
BUFFER_PERCENT=1

# Define suffixes to ignore
ignored_suffixes=("_upper" "_lower" "_top_left" "_top_right" "_bottom_left" "_bottom_right")

# Default values
TARGET_DIR=""
OUTPUT_DIR=""
DRY_RUN=false
JOBS=1
NO_SKIP=false

# ================================
# Parse Arguments
# ================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift # Remove --dry-run from processing
            ;;
        --no-skip)
            NO_SKIP=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 [--dry-run] [--no-skip] [-j jobs] [-o output_dir] <directory_with_videos>"
    exit 1
fi

TARGET_DIR="${TARGET_DIR%/}"
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${TARGET_DIR}_split"
fi
OUTPUT_DIR="${OUTPUT_DIR%/}"

# ================================
# Helper Functions
# ================================

should_ignore() {
    local file="$1"
    for suffix in "${ignored_suffixes[@]}"; do
        if [[ "$file" == *"$suffix".avi ]]; then
            return 0
        fi
    done
    return 1
}

# Returns 0 (true) if we SHOULD process it, 1 (false) if we should SKIP
should_process() {
    local file="$1"
    if [[ -f "$file" ]]; then
        # Check size in bytes
        local size
        size=$(stat -c %s "$file")
        if (( size > 10000000 )); then
            # File is larger than 10MB, skip it
            return 1
        fi
    fi
    return 0
}

# ================================
# Splitting Functions
# ================================

split_upper_lower() {
    local video="$1"
    local base_name
    base_name=$(basename "$video" .avi)

    # If the video is in a subdirectory of TARGET_DIR, use the parent subdirectory's name
    # to prefix the output flat filename.
    local parent_dir
    parent_dir=$(basename "$(dirname "$video")")
    local out_prefix=""
    if [[ "$parent_dir" != "$(basename "$TARGET_DIR")" ]]; then
        out_prefix="${parent_dir}_"
    fi

    # Determine chambers configuration if not skipping
    local c1="" c2="" c3="" c4=""
    if [ "$NO_SKIP" != true ]; then
        local parse_src=""
        if [[ "$base_name" == *"_Arena"* ]]; then
            parse_src="${base_name%%_Arena*}"
        else
            parse_src="$parent_dir"
        fi
        local chambers_str
        if [[ $(echo "$parse_src" | tr -cd '_' | wc -c) -ge 2 ]]; then
            chambers_str=$(echo "$parse_src" | cut -d'_' -f2)
        else
            chambers_str="${parse_src##*_}"
        fi
        IFS='.' read -r c1 c2 c3 c4 <<< "$chambers_str"
    fi

    local process_upper=false
    local process_lower=false
    
    local out_upper="${OUTPUT_DIR}/${out_prefix}${base_name}_upper.avi"
    local out_lower="${OUTPUT_DIR}/${out_prefix}${base_name}_lower.avi"

    if [ "$NO_SKIP" = true ]; then
        if [[ "$base_name" == *"Arena1and2_Left"* || "$base_name" == *"Arena3and4_Right"* ]]; then
            if should_process "$out_upper"; then process_upper=true; fi
            if should_process "$out_lower"; then process_lower=true; fi
        fi
    else
        if [[ "$base_name" == *"Arena1and2_Left"* ]]; then
            # C1 is upper, C2 is lower
            if [[ -n "$c1" && "$c1" != "e" ]] && should_process "$out_upper"; then process_upper=true; fi
            if [[ -n "$c2" && "$c2" != "e" ]] && should_process "$out_lower"; then process_lower=true; fi
        elif [[ "$base_name" == *"Arena3and4_Right"* ]]; then
            # C4 is upper, C3 is lower
            if [[ -n "$c4" && "$c4" != "e" ]] && should_process "$out_upper"; then process_upper=true; fi
            if [[ -n "$c3" && "$c3" != "e" ]] && should_process "$out_lower"; then process_lower=true; fi
        fi
    fi

    if ! $process_upper && ! $process_lower; then
        echo "Skipping $base_name (already processed or empty chambers)..."
        return
    fi

    # Calculate Dimensions
    local height width
    height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$video")
    width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$video")

    local half_height=$(( height / 2 ))
    local overlap=$(( height * BUFFER_PERCENT / 100 / 2 ))
    local top_crop_height=$(( half_height + overlap ))
    local bottom_crop_height=$(( half_height + overlap ))
    local top_offset=0
    local bottom_offset=$(( half_height - overlap ))

    echo "Processing $video -> $OUTPUT_DIR ..."

    if $process_upper; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would process: $out_upper"
        else
            echo "  Generating: $out_upper"
            ffmpeg -nostdin -y -hwaccel cuda -i "$video" \
                -vf "crop=${width}:${top_crop_height}:0:${top_offset}" \
                -c:v h264_nvenc -preset p5 -b:v 5M "$out_upper"
        fi
    fi

    if $process_lower; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would process: $out_lower"
        else
            echo "  Generating: $out_lower"
            ffmpeg -nostdin -y -hwaccel cuda -i "$video" \
                -vf "crop=${width}:${bottom_crop_height}:0:${bottom_offset}" \
                -c:v h264_nvenc -preset p5 -b:v 5M "$out_lower"
        fi
    fi
}

split_quadrants() {
    local video="$1"
    local base_name
    base_name=$(basename "$video" .avi)

    # If the video is in a subdirectory of TARGET_DIR, use the parent subdirectory's name
    # to prefix the output flat filename.
    local parent_dir
    parent_dir=$(basename "$(dirname "$video")")
    local out_prefix=""
    if [[ "$parent_dir" != "$(basename "$TARGET_DIR")" ]]; then
        out_prefix="${parent_dir}_"
    fi

    # Determine chambers configuration if not skipping
    local c1="" c2="" c3="" c4=""
    if [ "$NO_SKIP" != true ]; then
        local parse_src=""
        if [[ "$base_name" == *"_Arena"* ]]; then
            parse_src="${base_name%%_Arena*}"
        else
            parse_src="$parent_dir"
        fi
        local chambers_str
        if [[ $(echo "$parse_src" | tr -cd '_' | wc -c) -ge 2 ]]; then
            chambers_str=$(echo "$parse_src" | cut -d'_' -f2)
        else
            chambers_str="${parse_src##*_}"
        fi
        IFS='.' read -r c1 c2 c3 c4 <<< "$chambers_str"
    fi

    local out_tl="${OUTPUT_DIR}/${out_prefix}${base_name}_top_left.avi"
    local out_tr="${OUTPUT_DIR}/${out_prefix}${base_name}_top_right.avi"
    local out_bl="${OUTPUT_DIR}/${out_prefix}${base_name}_bottom_left.avi"
    local out_br="${OUTPUT_DIR}/${out_prefix}${base_name}_bottom_right.avi"
    
    local p_tl=false; local p_tr=false; local p_bl=false; local p_br=false
    
    if [ "$NO_SKIP" = true ]; then
        if should_process "$out_tl"; then p_tl=true; fi
        if should_process "$out_tr"; then p_tr=true; fi
        if should_process "$out_bl"; then p_bl=true; fi
        if should_process "$out_br"; then p_br=true; fi
    else
        # C1 = Top Left (tl)
        if [[ -n "$c1" && "$c1" != "e" ]] && should_process "$out_tl"; then p_tl=true; fi
        # C2 = Bottom Left (bl)
        if [[ -n "$c2" && "$c2" != "e" ]] && should_process "$out_bl"; then p_bl=true; fi
        # C3 = Bottom Right (br)
        if [[ -n "$c3" && "$c3" != "e" ]] && should_process "$out_br"; then p_br=true; fi
        # C4 = Top Right (tr)
        if [[ -n "$c4" && "$c4" != "e" ]] && should_process "$out_tr"; then p_tr=true; fi
    fi
    
    if ! $p_tl && ! $p_tr && ! $p_bl && ! $p_br; then
        echo "Skipping $base_name (already processed or empty chambers)..."
        return
    fi
    
    # Calculate Dimensions
    local height width
    height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$video")
    width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$video")

    local half_width=$(( width / 2 ))
    local half_height=$(( height / 2 ))
    local overlap_w=$(( width * BUFFER_PERCENT / 100 / 2 ))
    local overlap_h=$(( height * BUFFER_PERCENT / 100 / 2 ))
    local out_w=$((half_width + overlap_w))
    local out_h=$((half_height + overlap_h))

    echo "Processing $video -> $OUTPUT_DIR ..."

    local filter_parts=""
    
    if $p_tl; then filter_parts+="[0:v]crop=${out_w}:${out_h}:0:0[tl]; "; fi
    if $p_tr; then filter_parts+="[0:v]crop=${out_w}:${out_h}:$((half_width - overlap_w)):0[tr]; "; fi
    if $p_bl; then filter_parts+="[0:v]crop=${out_w}:${out_h}:0:$((half_height - overlap_h))[bl]; "; fi
    if $p_br; then filter_parts+="[0:v]crop=${out_w}:${out_h}:$((half_width - overlap_w)):$((half_height - overlap_h))[br]; "; fi
    
    filter_parts="${filter_parts%; }"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would process $base_name with filter: $filter_parts"
    else
        local cmd=(ffmpeg -nostdin -y -hwaccel cuda -i "$video" -filter_complex "$filter_parts")
        
        if $p_tl; then cmd+=("-map" "[tl]" "-c:v" "h264_nvenc" "-preset" "p5" "-b:v" "5M" "$out_tl"); fi
        if $p_tr; then cmd+=("-map" "[tr]" "-c:v" "h264_nvenc" "-preset" "p5" "-b:v" "5M" "$out_tr"); fi
        if $p_bl; then cmd+=("-map" "[bl]" "-c:v" "h264_nvenc" "-preset" "p5" "-b:v" "5M" "$out_bl"); fi
        if $p_br; then cmd+=("-map" "[br]" "-c:v" "h264_nvenc" "-preset" "p5" "-b:v" "5M" "$out_br"); fi
        
        "${cmd[@]}"
    fi
}

# ================================
# Main Processing
# ================================

if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN MODE ACTIVATED ---"
    echo "Input Folder:  $TARGET_DIR"
    echo "Output Folder: $OUTPUT_DIR"
    echo "No files will be created."
    echo ""
else
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

running_jobs=0

while read video; do
    if [[ -f "$video" && ! $(should_ignore "$video") ]]; then
        if [ "$JOBS" -gt 1 ]; then
            split_upper_lower "$video" &
            ((running_jobs++))
            if (( running_jobs >= JOBS )); then
                wait -n
                ((running_jobs--))
            fi
        else
            split_upper_lower "$video"
        fi
    fi
done < <(find -L "$TARGET_DIR" -type f -name "*and*.avi" -not -path "*split*" | sort)
wait

running_jobs=0

while read video; do
    if [[ -f "$video" && ! $(should_ignore "$video") ]]; then
        if [ "$JOBS" -gt 1 ]; then
            split_quadrants "$video" &
            ((running_jobs++))
            if (( running_jobs >= JOBS )); then
                wait -n
                ((running_jobs--))
            fi
        else
            split_quadrants "$video"
        fi
    fi
done < <(find -L "$TARGET_DIR" -type f -name "*Arena1234*.avi" -not -path "*split*" | sort)
wait

if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN COMPLETE ---"
else
    echo "Video splitting completed!"
fi
