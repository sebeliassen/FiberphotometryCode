#!/usr/bin/env bash
set -euo pipefail

# Explicit paths/arguments keep selector, movie, plots and gallery on one manifest.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${KPMS_PYTHON:-/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python}"
VIZ_DIR="$SCRIPT_DIR/visualizations_cohort_static_21kp_v2_seed101_v4"
SYLLABLE_ARGS=()
OVERWRITE_ARGS=()
OVERLAY_ARGS=()
while (($#)); do
    case "$1" in
        --viz_dir) VIZ_DIR="$2"; shift 2 ;;
        --syllables)
            shift
            SYLLABLE_ARGS+=(--syllables)
            while (($#)) && [[ "$1" =~ ^[0-9]+$ ]]; do SYLLABLE_ARGS+=("$1"); shift; done
            if ((${#SYLLABLE_ARGS[@]} < 2)); then echo "--syllables needs integer IDs" >&2; exit 2; fi
            ;;
        --overwrite) OVERWRITE_ARGS=(--overwrite); shift ;;
        --overlay_keypoints) OVERLAY_ARGS=(--overlay_keypoints); shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR"

"$PYTHON" -u select_cohort_static_21kp_v2_syllable_manifest.py \
    --grid_size 12 --score_pre_seconds 0.166667 --score_post_seconds 0.5 \
    --movie_pre_seconds 1 --movie_post_seconds 2 --output_dir "$VIZ_DIR" "${SYLLABLE_ARGS[@]}"
MANIFEST="$VIZ_DIR/unified_syllable_selection_manifest.json"
"$PYTHON" -u render_all_cohort_static_21kp_v2_seed101_grid_movies.py \
    --manifest "$MANIFEST" --out_dir "$VIZ_DIR/grid_movies_density" \
    "${SYLLABLE_ARGS[@]}" "${OVERWRITE_ARGS[@]}" "${OVERLAY_ARGS[@]}"
"$PYTHON" -u render_all_cohort_static_21kp_v2_seed101_trajectories.py \
    --manifest "$MANIFEST" --out_dir "$VIZ_DIR/trajectories" \
    "${SYLLABLE_ARGS[@]}" "${OVERWRITE_ARGS[@]}"
"$PYTHON" -u build_cohort_static_21kp_v2_seed101_labeling_gallery.py --viz_dir "$VIZ_DIR"
