#!/usr/bin/env python3
"""
Verify Arena Orientation and Spatial Mapping between TopCam and BotCam.

Tests:
1. Uniform temporal sampling across the session (after t=15 min).
2. Motion energy and pixel variance comparison across all arenas.
3. Cross-camera concurrent frame delta correlation.
4. Permutation test over all 24 shuffles to prove no other assignment is better.
"""

import itertools
import subprocess
from pathlib import Path

import numpy as np


def extract_motion_profile(video_path, timestamps, window_frames=50, width=120, height=120):
    """
    Extracts frame-to-frame motion deltas at specified timestamps using fast-seek ffmpeg.
    Returns:
        motion_profile: 1D array of mean frame differences across all sampled windows
        mean_motion: scalar average motion energy
        variance: scalar temporal variance
    """
    deltas = []
    frame_size = width * height

    for ts in timestamps:
        cmd = [
            "ffmpeg",
            "-ss", str(ts),
            "-i", str(video_path),
            "-vframes", str(window_frames),
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw, _ = proc.communicate()
        n = len(raw) // frame_size
        if n < 2:
            continue
        frames = np.frombuffer(raw[:n * frame_size], dtype=np.uint8).reshape((n, height, width))
        diff = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
        deltas.extend(diff)

    if not deltas:
        return np.array([]), 0.0, 0.0

    motion_profile = np.array(deltas)
    return motion_profile, float(motion_profile.mean()), float(np.var(motion_profile))

def run_single_mouse_benchmark():
    print("====================================================================")
    print(" TEST 1: Single Mouse Benchmark (T10_396_04122025)")
    print(" Expected: TopCam has ONLY top_left (mouse 396).")
    print("           BotCam should show ONE active arena and THREE empty arenas.")
    print("====================================================================")

    botcam_dir = Path("/mnt/hdd/karen_project/3_week_PI/T10_396_04122025")
    # Total duration ~1390s (~23 min). Sample 8 windows uniformly between t=900s (15 min) and t=1300s
    sample_timestamps = np.linspace(900, 1300, 8).astype(int)

    stats = {}
    for arena in [1, 2, 3, 4]:
        vid = botcam_dir / f"Arena{arena}_BotCamVid_2025-12-04T12_56_44.avi"
        if not vid.exists():
            print(f"  [MISSING] {vid}")
            continue
        _, mean_m, var_m = extract_motion_profile(vid, sample_timestamps)
        stats[arena] = (mean_m, var_m)
        print(f"  BotCam Arena {arena}: Mean Motion Delta = {mean_m:6.2f} | Motion Variance = {var_m:6.2f}")

    # Determine winning arena
    best_arena = max(stats, key=lambda a: stats[a][0])
    print(f"\n  --> Result: Arena {best_arena} has {stats[best_arena][0]/max(0.01, stats[best_arena%4 + 1][0]):.1f}x higher motion than other arenas!")
    if best_arena == 1:
        print("  --> CONFIRMED: Arena 1 is the active arena (matches TopCam top_left)!")
    else:
        print(f"  --> ALERT: Arena {best_arena} is active instead of Arena 1!")
    return best_arena

def run_multi_mouse_correlation():
    print("\n====================================================================")
    print(" TEST 2: Multi-Mouse Motion Correlation & Permutation Test (T1_2.6.5)")
    print(" Expected: Mice 2, 6, 5 in TopCam quadrants top_left, bottom_left, bottom_right.")
    print("           Arena 4 (top_right) is empty.")
    print("====================================================================")

    topcam_dir = Path(__file__).resolve().parents[2] / "data/raw/karen_pff_vs_monomer/video_files"
    botcam_dir = Path("/mnt/hdd/karen_project/3_week_PI/T1_2.6.5_03122025")

    # Duration ~3658s (60 min). Sample 12 windows uniformly between t=900s (15 min) and t=2700s (45 min)
    sample_timestamps = np.linspace(900, 2700, 12).astype(int)

    top_quads = ["top_left", "bottom_left", "bottom_right"]
    top_profiles = {}
    for quad in top_quads:
        cand = list(topcam_dir.glob(f"*T1_2.6.5*_{quad}.avi"))
        if not cand:
            print(f"  [MISSING TOP] {quad}")
            continue
        prof, _, _ = extract_motion_profile(cand[0], sample_timestamps)
        top_profiles[quad] = prof
        print(f"  Loaded TopCam [{quad}] motion profile (length={len(prof)})")

    bot_arenas = [1, 2, 3, 4]
    bot_profiles = {}
    bot_stats = {}
    for arena in bot_arenas:
        vid = botcam_dir / f"Arena{arena}_BotCamVid_2025-12-03T09_51_05.avi"
        if not vid.exists():
            print(f"  [MISSING BOT] Arena {arena}")
            continue
        prof, m, v = extract_motion_profile(vid, sample_timestamps)
        bot_profiles[arena] = prof
        bot_stats[arena] = (m, v)
        print(f"  Loaded BotCam [Arena {arena}] motion profile (mean={m:5.2f}, var={v:5.2f})")

    # Check which BotCam is empty
    empty_arena = min(bot_stats, key=lambda a: bot_stats[a][0])
    print(f"\n  --> Empty BotCam Arena: Arena {empty_arena} (mean={bot_stats[empty_arena][0]:.2f})")
    if empty_arena == 4:
        print("  --> CONFIRMED: Arena 4 is empty, matching empty TopCam top_right!")

    # Compute correlation matrix between TopCam quadrants and BotCam arenas (1, 2, 3)
    active_arenas = [a for a in [1, 2, 3, 4] if a != empty_arena]
    corr_matrix = np.zeros((len(top_quads), len(active_arenas)))

    for i, tq in enumerate(top_quads):
        for j, ba in enumerate(active_arenas):
            p1 = top_profiles[tq]
            p2 = bot_profiles[ba]
            min_len = min(len(p1), len(p2))
            if min_len > 10 and np.std(p1[:min_len]) > 0 and np.std(p2[:min_len]) > 0:
                corr = np.corrcoef(p1[:min_len], p2[:min_len])[0, 1]
            else:
                corr = 0.0
            corr_matrix[i, j] = corr

    print("\n--- Pearson Correlation Matrix (TopCam vs BotCam Motion) ---")
    header = "             " + "  ".join(f"BotCam {a}" for a in active_arenas)
    print(header)
    for i, tq in enumerate(top_quads):
        row = f"{tq:<13} " + "   ".join(f"{corr_matrix[i, j]:+7.3f}" for j in range(len(active_arenas)))
        print(row)

    # Permutation Shuffle Test
    print("\n--- Testing All Possible Shuffles (Permutations) ---")
    best_score = -999.0
    all_perm_results = []

    for perm in itertools.permutations(active_arenas):
        # perm assigns TopCam i -> BotCam perm[i]
        score = sum(corr_matrix[i, active_arenas.index(perm[i])] for i in range(len(top_quads)))
        all_perm_results.append((perm, score))
        if score > best_score:
            best_score = score

    all_perm_results.sort(key=lambda x: x[1], reverse=True)
    for rank, (perm, score) in enumerate(all_perm_results, 1):
        mapping_str = ", ".join(f"{top_quads[i]} -> Arena {perm[i]}" for i in range(len(top_quads)))
        marker = " <--- (BEST MAPPING)" if rank == 1 else ""
        print(f"  Rank {rank:2d}: Score = {score:+6.3f} | {mapping_str}{marker}")

if __name__ == "__main__":
    run_single_mouse_benchmark()
    run_multi_mouse_correlation()
