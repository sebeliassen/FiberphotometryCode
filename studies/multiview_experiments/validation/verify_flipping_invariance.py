import os
import sys

import cv2
import numpy as np
import pandas as pd

from fpcode.common.paths import resolve_data_path

# Add parent directory to path to find utils and local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor/keypoint_moseq_syll')))

# Import from local modules
import keypoint_moseq as kpms
from utils.flipping import flip_data
from utils.paths import ref_pairs_from_sessions

# Configuration
SESSIONS_PKL = str(resolve_data_path('data/sessions/jonathan_ifdn_sessions_dirty.pkl'))
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"

def calc_distances(coords):
    """
    Calculate Euclidean distance traveled per frame for each bodypart.
    coords: (T, N, 2)
    Returns: (T-1, N)
    """
    diffs = np.diff(coords, axis=0) # (T-1, N, 2)
    dists = np.linalg.norm(diffs, axis=2) # (T-1, N)
    return dists

def verify_invariance():
    print(f"Loading sessions from {SESSIONS_PKL}...")
    try:
        sessions = pd.read_pickle(SESSIONS_PKL)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # Use all sessions
    session_subset = sessions

    print("Deriving paths...")
    keypoint_paths, video_paths = ref_pairs_from_sessions(session_subset, subroutine=SUBROUTINE, cam=CAM)

    if not video_paths:
        print("No videos found.")
        return

    print(f"Loading keypoints for {len(keypoint_paths)} sessions...")
    # Load all at once to match pipeline behavior, or loop?
    # Pipeline loads all, then flips.
    # But flip_data takes a single 'width' or we need to handle variable widths?
    # Current flip_data implementation takes a SINGLE width.
    # This implies the current pipeline implementation assumes CONSTANT width for all videos if passed a single integer.
    #
    # Wait. kpms_pipeline.py:
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  <-- TAKES FIRST VIDEO ONLY.
    #
    # So if I run this on all sessions, and they have different widths, the pipeline (and this test) is WRONG.
    # This test will expose that flaw if it exists.

    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, DETECTOR)

    # Get Width from FIRST video (matching pipeline logic)
    if not video_paths:
        return

    cap = cv2.VideoCapture(video_paths[0])
    first_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    print(f"Pipeline Width (from 1st video): {first_width}")

    # Perform Augmentation
    print("Flipping data (Augmentation Mode)...")
    new_coords, new_confs = flip_data(coordinates, confidences, bodyparts, first_width, mode='augmentation')

    # Verification Loop
    print("\nVerifying Invariance across all videos...")

    # We need to check per-video if the width assumption holds or if symmetry holds regardless.
    # If a video has width != first_width, then (first_width - x) will shift the animal.
    # If the animal is centered, shifting might be okay for relative pose but terrible for absolute location.
    # BUT we want to verify DISTANCE traveled.
    # Distance of (C - x) is same as distance of x. |d(C-x)/dt| = |-dx/dt| = |dx/dt|.
    # So Distance Invariance should HOLD even if width is "wrong"!
    #
    # Let's verify that.

    original_keys = list(coordinates.keys())
    failure_count = 0

    for key in original_keys:
        flipped_key = f"{key}_flipped"
        if flipped_key not in new_coords:
            print(f"ERROR: {flipped_key} not found.")
            continue

        orig = new_coords[key]      # (T, N, 2)
        flipped = new_coords[flipped_key] # (T, N, 2)

        # Calculate Frame-to-Frame Distances
        dist_orig = calc_distances(orig)     # (T-1, N)
        dist_flip = calc_distances(flipped)  # (T-1, N)

        # Check 1: Central Bodyparts
        for bp_name in ['head_midpoint', 'mid_back', 'tail_base', 'nose']:
            if bp_name in bodyparts:
                idx = bodyparts.index(bp_name)
                diff = np.abs(dist_orig[:, idx] - dist_flip[:, idx])
                if np.max(diff) > 1e-4:
                     print(f"  [FAIL] {key} - {bp_name}: Max Diff {np.max(diff):.2e}")
                     failure_count += 1

        # Check 2: Symmetric Pairs
        pairs = [('left_ear', 'right_ear'), ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')]
        for left, right in pairs:
            if left in bodyparts and right in bodyparts:
                idx_l = bodyparts.index(left)
                idx_r = bodyparts.index(right)

                # Compare Orig Left vs Flipped Right
                diff_lr = np.abs(dist_orig[:, idx_l] - dist_flip[:, idx_r])
                if np.max(diff_lr) > 1e-4:
                     print(f"  [FAIL] {key} - {left}->{right}: Max Diff {np.max(diff_lr):.2e}")
                     failure_count += 1

                # Compare Orig Right vs Flipped Left
                diff_rl = np.abs(dist_orig[:, idx_r] - dist_flip[:, idx_l])
                if np.max(diff_rl) > 1e-4:
                     print(f"  [FAIL] {key} - {right}->{left}: Max Diff {np.max(diff_rl):.2e}")
                     failure_count += 1

    if failure_count == 0:
        print("\nAll videos passed invariance checks!")
    else:
        print(f"\nVerification FAILED with {failure_count} errors.")

if __name__ == "__main__":
    verify_invariance()
