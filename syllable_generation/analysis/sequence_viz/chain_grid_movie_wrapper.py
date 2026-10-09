#!/usr/bin/env python3
"""
chain_grid_movie_wrapper.py
=============================================================================
Extensible Grid Movie Generator for Recurrent Multi-Syllable Behavioral Chains.

Key Features & Enhancements:
1. Multi-Method Ingest:
   - Supports episodes discovered across all sequence methods (Exact N-grams,
     SEQUITUR, Cluster-RLE, Weighted Alignment, Profile HMMs, seqNMF,
     PST Variable-Order Markov, and Continuous DTW).
2. Dynamic Concurrent-Token Color Palette:
   - Evaluates the K unique syllables occurring concurrently across the movie's
     instances and stages (typically 3-5 syllables).
   - Maps them to K maximally separated, vibrant, high-contrast colors from a
     curated palette.
   - Synchronously applied across the centroid marker, trajectory trail, and HUD.
3. 2D Centroid Trajectory Inset:
   - Renders a mini 2D trajectory inset in each tile relative to onset heading θ₀.
   - Traces the animal's arena path segment-by-segment in the active token's color,
     with a glowing current-position marker, revealing spatial displacement that
     the centroid-following camera centers out.
4. Explicit Alignment Engine (Indels):
   - 'show_variations': Preserves inserted footage. Panels lacking that insertion
     wait on a dimmed frame marked 'Alignment wait: +X'. Missing stages show a
     marked gap slate 'No token: [B]'. All panels resume at the next shared stage.
   - 'compact_template': Skips inserted intervals with a brief cut badge
     'Skipped X: 0.24s' so the template sequence is easy to follow.
5. Original Durations & Transparency Multipliers:
   - Displays real natural durations in seconds/frames in the HUD.
   - When time-warped, displays the playback multiplier (e.g. '1.2x speed').
6. Branch-Centred Grid Mode:
   - Synchronizes multiple continuation branches at the exact exit frame of a
     shared prefix (e.g. syll_014 -> syll_007 -> [syll_000, syll_001, syll_038]),
     making divergent kinematic continuations directly watchable.
7. Similarity Panel:
   - Optional objective badges for prototype substitution cost and continuous
     kinematic DTW trajectory distance.
8. Hardware Acceleration:
   - GPU NVENC H.264 encoding via FFmpeg pipe with graceful multithreaded CPU
     and OpenCV fallbacks.
=============================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2
import h5py
from pathlib import Path

# Force CPU execution for JAX/TF to prevent GPU VRAM clashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
DEFAULT_PROJECT_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
DEFAULT_MODEL_NAME = "amph_prod_k150k_run_12"
DEFAULT_VIDEO_DIR = REPO_ROOT / "src_oft/amph_vids_ssd"

# Human Syllable Labels (Run 12)
HUMAN_LABELS_PATH = REPO_ROOT / "syllable_generation/analysis/sequence_viz/run12_human_labels.csv"
HUMAN_LABELS = {}
if HUMAN_LABELS_PATH.exists():
    try:
        import pandas as _pd
        _df = _pd.read_csv(HUMAN_LABELS_PATH)
        for _, _r in _df.iterrows():
            if _pd.notna(_r.get("syllable_id")) and _pd.notna(_r.get("description")):
                HUMAN_LABELS[int(_r["syllable_id"])] = str(_r["description"]).strip()
    except Exception:
        pass

# Fallback ethological descriptions for rarer syllables based on Level 4 clustering
META_FALLBACK = {
    45: "sniff", 46: "walk fwd", 47: "sprint", 48: "rest", 50: "walk right",
    51: "turn right", 53: "turn left", 54: "walk fwd", 56: "turn right",
    57: "walk left", 59: "groom", 64: "rest", 67: "turn right",
    68: "rest", 69: "sniff", 70: "rear"
}
for _sid, _desc in META_FALLBACK.items():
    if _sid not in HUMAN_LABELS:
        HUMAN_LABELS[_sid] = _desc

# Compact stage box abbreviations so multi-word human labels fit cleanly without ellipsis
SHORT_BOX_LABELS = {
    "back down + turn left": "back+turn L",
    "back down + turn right": "back+turn R",
    "look left (from right)": "look left",
    "scrunched sniffing": "scrunch sniff",
    "run + turn right": "run+turn R",
    "sniff upwards": "sniff up",
    "hunched rear": "hunch rear",
}



# ==============================================================================
# CURATED HIGH-CONTRAST PALETTE FOR CONCURRENT TOKENS
# ==============================================================================
CURATED_PALETTE = [
    (50, 130, 255),   # Vivid Coral / Amber Orange
    (240, 190, 40),   # Bright Sky Cyan
    (70, 220, 100),   # Vibrant Emerald Green
    (220, 80, 230),   # Electric Magenta / Pink
    (40, 220, 255),   # Radiant Gold / Yellow
    (255, 120, 140),  # Bright Lavender / Violet
    (80, 240, 240),   # Soft Lemon Lime
    (240, 140, 60),   # Electric Cerulean Blue
]


def build_concurrent_token_colormap(unique_tokens):
    """
    Given a list of unique tokens present concurrently across instances/stages,
    assigns maximally separated, high-contrast, visually pleasant colors in BGR.
    """
    colormap = {}
    n = len(unique_tokens)
    for i, tok in enumerate(unique_tokens):
        if n <= len(CURATED_PALETTE):
            color = CURATED_PALETTE[i]
        else:
            hue = int((i * 180) / n)
            hsv = np.uint8([[[hue, 220, 240]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        colormap[tok] = color
    return colormap


def get_contrasting_text_color(bgr_color):
    """Returns black or white text depending on background luminance."""
    b, g, r = bgr_color
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if lum > 145 else (255, 255, 255)


# ==============================================================================
# FAST OPENCV VIDEO READER WITH FRAME CACHING
# ==============================================================================
class FastVideoReader:
    """High-performance frame reader maintaining file pointer across seeks."""
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.current_frame = -1
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        gap = frame_idx - self.current_frame
        if 0 < gap <= 15:
            # Fast sequential grab
            for _ in range(gap - 1):
                self.cap.grab()
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
        if ret:
            self.current_frame = frame_idx
            return frame
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# RESOURCE LOADER
# ==============================================================================
def load_chain_movie_resources(
    project_dir=DEFAULT_PROJECT_DIR,
    model_name=DEFAULT_MODEL_NAME,
    video_dir=DEFAULT_VIDEO_DIR
):
    """
    Loads results.h5, video_frame_indexes.npz, and maps available video paths.
    """
    project_dir = Path(project_dir)
    video_dir = Path(video_dir)
    res_path = project_dir / model_name / "results.h5"
    vfi_path = project_dir / model_name / "video_frame_indexes.npz"

    print(f"[RESOURCES] Loading modeling results from {res_path.name}...", flush=True)
    results = {}
    with h5py.File(res_path, "r") as f:
        for k in f.keys():
            results[k] = {
                "centroid": f[k]["centroid"][:],
                "heading": f[k]["heading"][:],
                "syllable": f[k]["syllables"][:] if "syllables" in f[k] else f[k]["syllable"][:],
                "latent_state": f[k]["latent_state"][:] if "latent_state" in f[k] else None
            }

    print(f"[RESOURCES] Loading video frame indexes from {vfi_path.name}...", flush=True)
    vfi_data = np.load(vfi_path)
    video_frame_indexes = {k: vfi_data[k] for k in vfi_data.files}

    # Map video files
    video_paths = {}
    all_avis = list(video_dir.glob("*.avi")) if video_dir.exists() else []
    for rk in results.keys():
        stem_parts = rk.split("DLC_")[0]
        v_matches = [p for p in all_avis if stem_parts in p.name]
        if not v_matches:
            tokens = stem_parts.split("_")
            day_tok = tokens[0]
            quad_tok = tokens[-1] if len(tokens) > 1 else ""
            v_matches = [p for p in all_avis if day_tok in p.name and quad_tok in p.name]
            
        if v_matches:
            video_paths[rk] = str(v_matches[0])
        else:
            video_paths[rk] = None

    valid_mapped = sum(1 for v in video_paths.values() if v is not None)
    print(f"[RESOURCES] Successfully mapped {valid_mapped} / {len(results)} recordings to video files.", flush=True)

    return {
        "results": results,
        "video_frame_indexes": video_frame_indexes,
        "video_paths": video_paths
    }


# ==============================================================================
# TOKEN ALIGNMENT UTILITY (NEEDLEMAN-WUNSCH DP)
# ==============================================================================
def align_tokens_to_template(observed_tokens, template_tokens):
    """
    Dynamic programming alignment between observed tokens and template tokens.
    Returns list of aligned step dicts:
      [{"type": "match"|"sub"|"ins"|"del",
        "template_token": ..., "observed_token": ...,
        "obs_idx": ..., "tpl_idx": ...}]
    """
    n = len(observed_tokens)
    m = len(template_tokens)
    
    dp = np.zeros((n + 1, m + 1))
    back = {}
    
    for i in range(1, n + 1):
        dp[i, 0] = i * 2.0
        back[(i, 0)] = ("ins", i - 1, None)
    for j in range(1, m + 1):
        dp[0, j] = j * 2.0
        back[(0, j)] = ("del", None, j - 1)
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            obs = observed_tokens[i - 1]
            tpl = template_tokens[j - 1]
            
            sub_cost = 0.0 if obs == tpl else 1.5
            cost_diag = dp[i - 1, j - 1] + sub_cost
            cost_del = dp[i, j - 1] + 2.0
            cost_ins = dp[i - 1, j] + 2.0
            
            min_c = min(cost_diag, cost_del, cost_ins)
            dp[i, j] = min_c
            if min_c == cost_diag:
                op = "match" if obs == tpl else "sub"
                back[(i, j)] = (op, i - 1, j - 1)
            elif min_c == cost_del:
                back[(i, j)] = ("del", None, j - 1)
            else:
                back[(i, j)] = ("ins", i - 1, None)
                
    curr = (n, m)
    ops = []
    while curr != (0, 0):
        op, oi, tj = back[curr]
        ops.append({
            "type": op,
            "obs_idx": oi,
            "tpl_idx": tj,
            "observed_token": observed_tokens[oi] if oi is not None else None,
            "template_token": template_tokens[tj] if tj is not None else None
        })
        if op in ("match", "sub"):
            curr = (curr[0] - 1, curr[1] - 1)
        elif op == "del":
            curr = (curr[0], curr[1] - 1)
        else: # ins
            curr = (curr[0] - 1, curr[1])
    ops.reverse()
    return ops


# ==============================================================================
# INSTANCE STANDARDIZER
# ==============================================================================
def standardize_chain_instances(instances, results=None, video_frame_indexes=None):
    """
    Standardizes varied episode representation dicts/rows into a uniform format:
    {
      'instance_id': str,
      'recording_key': str,
      'start_frame': int (model frame),
      'end_frame': int (model frame),
      'stage_boundaries': [b0, b1, ..., bK],
      'stage_tokens': ['syll_001', ...],
      'animal_id': str,
      'genotype': str,
      'drug': str,
      'proto_cost': float (optional),
      'kin_dist': float (optional),
      'aligned_stages': list of dicts (optional)
    }
    """
    standardized = []
    for idx, inst in enumerate(instances):
        d = dict(inst)
        start_f = int(d.get("start_frame", d.get("start_f", 0)))
        end_f = int(d.get("end_frame", d.get("end_f", start_f + 40)))
        rec_key = d.get("recording_key", d.get("rec_key"))
        
        # Fallback recording resolution
        if rec_key is None and results is not None:
            aid = str(d.get("animal_id", ""))
            drug = str(d.get("drug", ""))
            for k in results.keys():
                if aid in k and (("day1" in k and drug.lower() == "saline") or ("day2" in k and drug.lower() == "amphetamine")):
                    rec_key = k
                    break

        stage_b = d.get("stage_boundaries")
        tokens = d.get("matched_tokens", d.get("stage_tokens", []))
        if isinstance(tokens, str):
            tokens = tokens.replace("[", "").replace("]", "").strip().split()
            tokens = [f"syll_{int(t):03d}" if t.isdigit() else t for t in tokens]
        elif isinstance(tokens, (list, np.ndarray)):
            tokens = [f"syll_{int(t):03d}" if isinstance(t, (int, np.integer)) or (isinstance(t, str) and t.isdigit()) else str(t) for t in tokens]

        # Auto-detect video frame coordinates:
        if results is not None and rec_key in results and video_frame_indexes is not None and rec_key in video_frame_indexes:
            n_model_f = len(results[rec_key]["syllable"])
            if start_f >= n_model_f:
                vfi_arr = video_frame_indexes[rec_key]
                orig_start = start_f
                orig_end = end_f
                start_f = int(np.clip(np.searchsorted(vfi_arr, orig_start), 0, n_model_f - 1))
                end_f = int(np.clip(np.searchsorted(vfi_arr, orig_end), start_f + 1, n_model_f - 1))
                if stage_b is not None:
                    stage_b = [int(np.clip(np.searchsorted(vfi_arr, b), 0, n_model_f - 1)) if b >= n_model_f else int(b) for b in stage_b]

        # Infer stage boundaries from syllable array if not provided
        if stage_b is None and results is not None and rec_key in results:
            syll_arr = results[rec_key]["syllable"][start_f:end_f]
            change_pts = np.where(syll_arr[1:] != syll_arr[:-1])[0] + 1
            stage_b = [start_f] + (start_f + change_pts).tolist() + [end_f]
            if not tokens:
                tokens = [f"syll_{int(syll_arr[p]):03d}" for p in [0] + change_pts.tolist()]
        elif stage_b is None:
            n_stages = max(1, len(tokens))
            step = (end_f - start_f) / n_stages
            stage_b = [int(start_f + s * step) for s in range(n_stages)] + [end_f]

        # Extract similarity values if present
        proto_cost = d.get("proto_cost", d.get("substitution_cost", d.get("cost", None)))
        kin_dist = d.get("kin_dist", d.get("dtw_dist", d.get("distance", None)))
        if proto_cost is not None:
            try: proto_cost = float(proto_cost)
            except: proto_cost = None
        if kin_dist is not None:
            try: kin_dist = float(kin_dist)
            except: kin_dist = None

        standardized.append({
            "instance_id": str(d.get("match_id", d.get("instance_id", f"inst_{idx:03d}"))),
            "recording_key": rec_key,
            "start_frame": start_f,
            "end_frame": end_f,
            "stage_boundaries": [int(b) for b in stage_b],
            "stage_tokens": tokens,
            "animal_id": str(d.get("animal_id", "Unknown")),
            "genotype": str(d.get("genotype", "WT")),
            "drug": str(d.get("drug", "Saline")),
            "method": str(d.get("method", "Discovered_Motif")),
            "tier": str(d.get("tier", "Exact")),
            "proto_cost": proto_cost,
            "kin_dist": kin_dist,
            "aligned_stages": d.get("aligned_stages", None)
        })
    return standardized


# ==============================================================================
# OUTLIER REMOVAL AND CENTRALITY RANKING ENGINE
# ==============================================================================
def filter_chain_outliers(
    instances,
    results,
    min_bout_frames=5,
    max_bout_frames=65,
    max_velocity_jump=22.0,
    min_path_displacement=10.0,
    iqr_multiplier=1.5,
    max_instances_per_animal=2,
    target_count=6,
    video_frame_indexes=None
):
    """
    Automated Outlier Rejection & Representative Centrality Ranking.
    Removes tracking glitches, micro-flicker bouts, and stuck states that throw
    off sequential kinematic continuity (chaos theory mitigation).

    Parameters:
    -----------
    instances : list of dicts
        Candidate chain instances (raw or standardized).
    results : dict
        Loaded results.h5 containing 'centroid', 'heading', 'syllable'.
    min_bout_frames : int (default 5 = 125ms at 40fps)
        Minimum duration for every constituent syllable. Rejects micro-flickers.
    max_bout_frames : int (default 65 = 1.625s at 40fps)
        Maximum duration for every constituent syllable. Rejects stuck/lost tracking bouts.
    max_velocity_jump : float (default 22.0 px/frame)
        Max permissible frame-to-frame centroid displacement. Rejects teleportation jumps.
    min_path_displacement : float (default 10.0 px)
        Minimum cumulative path length. Rejects motionless bouts.
    iqr_multiplier : float (default 1.5)
        IQR threshold for statistical duration and speed outlier rejection.
    max_instances_per_animal : int (default 2)
        Enforces cross-animal diversity in selected panels.
    target_count : int (default 6)
        Target number of top clean instances to return.
    video_frame_indexes : dict (optional)
        Mapping dictionary for coordinate space normalization if needed.

    Returns:
    --------
    selected_instances : list of dicts
    audit_stats : dict with rejection counts and retention metrics
    """
    std_insts = standardize_chain_instances(instances, results, video_frame_indexes)
    stats = {
        "total_candidates": len(std_insts),
        "flicker_outliers": 0,
        "stuck_outliers": 0,
        "velocity_jump_outliers": 0,
        "immobile_outliers": 0,
        "iqr_distribution_outliers": 0,
        "retained_clean_count": 0,
        "selected_count": 0
    }

    stage_filtered = []
    for inst in std_insts:
        sb = inst["stage_boundaries"]
        durs = [sb[k+1] - sb[k] for k in range(len(sb) - 1)]
        
        # 1. Micro-flicker check (<125ms state noise)
        if any(d < min_bout_frames for d in durs):
            stats["flicker_outliers"] += 1
            continue
            
        # 2. Prolonged / stuck state check (>1.625s tracking freeze)
        if any(d > max_bout_frames for d in durs):
            stats["stuck_outliers"] += 1
            continue

        # 3. Kinematic centroid tracking check
        rk = inst["recording_key"]
        if rk not in results or "centroid" not in results[rk]:
            stage_filtered.append(inst)
            continue

        cents = results[rk]["centroid"][inst["start_frame"]:inst["end_frame"]]
        if len(cents) > 1:
            vels = np.linalg.norm(np.diff(cents, axis=0), axis=1)
            max_v = float(np.max(vels))
            sum_v = float(np.sum(vels))
            mean_v = float(np.mean(vels))
        else:
            max_v, sum_v, mean_v = 0.0, 0.0, 0.0

        # Teleportation jump filter
        if max_v > max_velocity_jump:
            stats["velocity_jump_outliers"] += 1
            continue

        # Immobility filter
        if sum_v < min_path_displacement:
            stats["immobile_outliers"] += 1
            continue

        inst["_total_dur"] = inst["end_frame"] - inst["start_frame"]
        inst["_mean_speed"] = mean_v
        inst["_path_len"] = sum_v
        stage_filtered.append(inst)

    # 4. Statistical Distribution Outlier Filter (IQR on total duration and mean velocity)
    if len(stage_filtered) >= 4:
        durs = np.array([x["_total_dur"] for x in stage_filtered])
        spds = np.array([x["_mean_speed"] for x in stage_filtered])
        q1_d, q3_d = np.percentile(durs, [25, 75])
        iqr_d = q3_d - q1_d
        dur_min, dur_max = q1_d - iqr_multiplier * iqr_d, q3_d + iqr_multiplier * iqr_d

        q1_s, q3_s = np.percentile(spds, [25, 75])
        iqr_s = q3_s - q1_s
        spd_min, spd_max = q1_s - iqr_multiplier * iqr_s, q3_s + iqr_multiplier * iqr_s

        iqr_clean = []
        for x in stage_filtered:
            if (dur_min <= x["_total_dur"] <= dur_max) and (spd_min <= x["_mean_speed"] <= spd_max):
                iqr_clean.append(x)
            else:
                stats["iqr_distribution_outliers"] += 1
    else:
        iqr_clean = stage_filtered

    # 5. Representative Centrality Scoring
    med_dur = np.median([x["_total_dur"] for x in iqr_clean]) if iqr_clean else 40
    std_dur = np.std([x["_total_dur"] for x in iqr_clean]) if iqr_clean else 10
    med_spd = np.median([x["_mean_speed"] for x in iqr_clean]) if iqr_clean else 2.5
    std_spd = np.std([x["_mean_speed"] for x in iqr_clean]) if iqr_clean else 0.5

    for x in iqr_clean:
        x["centrality"] = ((x["_total_dur"] - med_dur) / (std_dur + 1e-4))**2 + ((x["_mean_speed"] - med_spd) / (std_spd + 1e-4))**2
    iqr_clean.sort(key=lambda x: x["centrality"])

    # 6. Animal Diversity Enforcement
    selected = []
    counts_per_animal = {}
    for x in iqr_clean:
        aid = x.get("animal_id", x["recording_key"].split("_")[2][:8])
        if counts_per_animal.get(aid, 0) < max_instances_per_animal:
            selected.append(x)
            counts_per_animal[aid] = counts_per_animal.get(aid, 0) + 1
            if len(selected) >= target_count:
                break

    # Relax animal limit if needed to reach target_count
    if len(selected) < target_count and len(iqr_clean) > len(selected):
        for x in iqr_clean:
            if x not in selected:
                selected.append(x)
                if len(selected) >= target_count:
                    break

    stats["retained_clean_count"] = len(iqr_clean)
    stats["selected_count"] = len(selected)
    return selected, stats


# ==============================================================================
# 2D CENTROID TRAJECTORY INSET
# ==============================================================================
def draw_trajectory_inset(
    tile_img,
    all_centroids,
    all_tokens,
    curr_frame_idx,
    onset_heading,
    token_colormap,
    inset_size=74,
    margin=8
):
    """
    Renders a mini 2D trajectory inset in the top-right corner of tile_img.
    Traces arena displacement rotated to onset heading θ₀.
    Line segments are coloured by the active token at each frame.
    A glowing dot marks current frame position.
    """
    h, w = tile_img.shape[:2]
    x0 = w - inset_size - 8
    y0 = 8
    x1 = x0 + inset_size
    y1 = y0 + inset_size
    
    # Semi-transparent dark background
    overlay = tile_img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 24), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (70, 70, 75), 1)
    # North / onset orientation tick at top center
    mid_inset_x = x0 + inset_size // 2
    cv2.line(overlay, (mid_inset_x, y0 + 2), (mid_inset_x, y0 + 6), (200, 200, 200), 1)
    cv2.addWeighted(overlay, 0.78, tile_img, 0.22, 0, tile_img)
    
    if len(all_centroids) < 2:
        return
        
    c0 = all_centroids[0]
    rel_c = all_centroids - c0
    R = np.array([
        [np.cos(onset_heading), np.sin(onset_heading)],
        [-np.sin(onset_heading), np.cos(onset_heading)]
    ])
    # Rotated relative coordinates: (0, 0) is onset, facing up (-Y)
    rot_c = (R @ rel_c.T).T
    
    # Invert Y so forward movement points upward on the screen
    rx = rot_c[:, 0]
    ry = -rot_c[:, 1]
    
    min_x, max_x = rx.min(), rx.max()
    min_y, max_y = ry.min(), ry.max()
    span = max(max_x - min_x, max_y - min_y, 25.0)
    
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    
    avail = inset_size - 2 * margin
    scale = avail / span
    
    px = np.clip((mid_inset_x + (rx - mid_x) * scale).astype(int), x0 + 2, x1 - 2)
    py = np.clip((y0 + inset_size // 2 + (ry - mid_y) * scale).astype(int), y0 + 2, y1 - 2)
    
    # Draw trajectory line segments
    n_pts = len(px)
    for t in range(n_pts - 1):
        tok = all_tokens[t] if t < len(all_tokens) else None
        seg_color = token_colormap.get(tok, (160, 160, 160))
        cv2.line(tile_img, (px[t], py[t]), (px[t+1], py[t+1]), seg_color, 2, cv2.LINE_AA)
        
    # Current position marker
    curr_idx = min(n_pts - 1, max(0, curr_frame_idx))
    curr_tok = all_tokens[curr_idx] if curr_idx < len(all_tokens) else None
    c_curr = token_colormap.get(curr_tok, (255, 255, 255))
    cv2.circle(tile_img, (px[curr_idx], py[curr_idx]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(tile_img, (px[curr_idx], py[curr_idx]), 2, c_curr, -1, cv2.LINE_AA)


# ==============================================================================
# DRAW HUD / TIMELINE UNDER TILE
# ==============================================================================
def draw_stage_hud(
    tile_w,
    hud_h,
    tokens,
    token_colormap,
    current_stage_idx,
    is_transition_frame,
    elapsed_sec,
    total_sec,
    stage_durations=None,
    status_text="PLAYING",
    tile_title="",
    similarity_info=None,
    blind_labels=False,
    cut_badge=None,
    alignment_badge=None
):
    """
    Renders an enhanced stage progress bar HUD underneath each video tile.
    Includes consistent token colors, original stage durations, playback multipliers,
    alignment indicators (Alignment wait / No token / Skipped), and similarity badges.
    """
    hud = np.full((hud_h, tile_w, 3), 20, dtype=np.uint8) # Dark background
    
    # Border line on top
    cv2.line(hud, (0, 0), (tile_w, 0), (60, 60, 60), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status / Time string
    if cut_badge:
        time_str = f"[{cut_badge}]"
        status_color = (40, 200, 255) # Amber cut indicator
    elif alignment_badge:
        time_str = f"[{alignment_badge}]"
        status_color = (0, 230, 255) # Cyan alignment wait
    elif status_text == "WAITING":
        time_str = f"{elapsed_sec:.1f}s [{status_text}]"
        status_color = (0, 220, 255)
    elif status_text == "FINISHED":
        time_str = f"FINISHED [{total_sec:.1f}s]"
        status_color = (150, 150, 150)
    else:
        time_str = f"{elapsed_sec:.1f}s / {total_sec:.1f}s"
        status_color = (110, 240, 110)

    # Optional Similarity info next to status (compact)
    sim_str = ""
    if similarity_info and not blind_labels:
        k_d = similarity_info.get("kin_dist")
        if k_d is not None:
            sim_str = f" | K:{k_d:.2f}"

    full_status_str = f"{time_str}{sim_str}"
    (tw, _), _ = cv2.getTextSize(full_status_str, font, 0.30, 1)
    status_x = tile_w - tw - 6
    cv2.putText(hud, full_status_str, (status_x, 14), font, 0.30, status_color, 1, cv2.LINE_AA)

    # 1. Top text row: Title + Active Syllable Description (prominent human label)
    active_desc = ""
    if current_stage_idx < len(tokens):
        curr_tok = tokens[current_stage_idx]
        c_sid_str = curr_tok.replace("syll_", "").lstrip("0") or "0"
        c_sid = int(c_sid_str) if c_sid_str.isdigit() else None
        c_desc = HUMAN_LABELS.get(c_sid, "")
        if c_desc:
            active_desc = f" | {c_desc}"
        else:
            active_desc = f" | {curr_tok}"

    header_str = "Blinded Episode" if blind_labels else (tile_title + active_desc)
    avail_header_w = status_x - 10
    f_header = 0.32
    (hw, _), _ = cv2.getTextSize(header_str, font, f_header, 1)
    if hw > avail_header_w:
        f_header = 0.28
        (hw, _), _ = cv2.getTextSize(header_str, font, f_header, 1)
    if hw > avail_header_w:
        f_header = 0.25
        (hw, _), _ = cv2.getTextSize(header_str, font, f_header, 1)

    while hw > avail_header_w and len(header_str) > 6:
        header_str = header_str[:-4] + "..."
        (hw, _), _ = cv2.getTextSize(header_str, font, f_header, 1)
    cv2.putText(hud, header_str, (6, 14), font, f_header, (220, 220, 220), 1, cv2.LINE_AA)

    # 2. Stage Boxes row
    n_tokens = max(1, len(tokens))
    box_y = 20
    box_h = 22
    total_margin = 8
    avail_w = tile_w - total_margin * 2
    spacing = 4
    box_w = max(10, int((avail_w - (n_tokens - 1) * spacing) / n_tokens))

    for k, tok in enumerate(tokens):
        x0 = total_margin + k * (box_w + spacing)
        x1 = x0 + box_w
        tok_c = token_colormap.get(tok, (180, 180, 180))
        text_c = get_contrasting_text_color(tok_c)
        
        # Color coding: active stage vs past vs future
        if k == current_stage_idx:
            # Active box
            if is_transition_frame:
                box_color = (0, 255, 255) # Flash cyan on transition
                border_color = (255, 255, 255)
                text_c = (0, 0, 0)
            else:
                box_color = tok_c
                border_color = (255, 255, 255)
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), box_color, -1)
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), border_color, 2 if is_transition_frame else 1)
        elif k < current_stage_idx:
            # Completed box: dark fill with colored outline
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), (35, 35, 38), -1)
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), tok_c, 1)
            text_c = (170, 170, 170)
        else:
            # Upcoming box: subtle outline
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), (25, 25, 28), -1)
            cv2.rectangle(hud, (x0, box_y), (x1, box_y + box_h), (60, 60, 65), 1)
            text_c = (110, 110, 110)

        # Token text: Direct human ethological label (e.g. 'run right' instead of '00')
        sid_str = tok.replace("syll_", "")
        c_sid_val = sid_str.lstrip("0") or "0"
        sid = int(c_sid_val) if c_sid_val.isdigit() else None
        desc = HUMAN_LABELS.get(sid, "")

        if desc:
            label = SHORT_BOX_LABELS.get(desc, desc)
            f_scale = 0.27
        else:
            label = sid_str
            f_scale = 0.30

        (lw, lh), _ = cv2.getTextSize(label, font, f_scale, 1)
        if lw > box_w - 4:
            f_scale = 0.23
            (lw, lh), _ = cv2.getTextSize(label, font, f_scale, 1)
        if lw > box_w - 4:
            f_scale = 0.19
            (lw, lh), _ = cv2.getTextSize(label, font, f_scale, 1)

        while lw > box_w - 4 and len(label) > 3:
            label = label[:-3] + ".."
            (lw, lh), _ = cv2.getTextSize(label, font, f_scale, 1)

        tx = x0 + max(2, (box_w - lw) // 2)
        ty = box_y + (box_h + lh) // 2
        cv2.putText(hud, label, (tx, ty), font, f_scale, text_c, 1, cv2.LINE_AA)

    # 3. Bottom Row: Stage Durations, Warp Multipliers, and Syllable Turn Angles
    if stage_durations is not None and hud_h >= 56:
        dur_y = box_y + box_h + 12
        for k, dur_info in enumerate(stage_durations):
            if k >= len(tokens):
                break
            x0 = total_margin + k * (box_w + spacing)
            if len(dur_info) == 3:
                dur_s, warp_mult, turn_str = dur_info
            else:
                dur_s, warp_mult = dur_info[:2]
                turn_str = None
            
            # Format: '0.28s (+75dL)' or '0.28s [1.2x]' or '0.28s'
            if turn_str and turn_str not in ["0d", "0°", "Fwd"]:
                d_str = f"{dur_s:.2f}s ({turn_str})"
            elif warp_mult is not None and abs(warp_mult - 1.0) > 0.06:
                d_str = f"{dur_s:.2f}s ({warp_mult:.1f}x)"
            else:
                d_str = f"{dur_s:.2f}s"
                
            (dw, dh), _ = cv2.getTextSize(d_str, font, 0.28, 1)
            dx = x0 + max(0, (box_w - dw) // 2)
            cv2.putText(hud, d_str, (dx, dur_y), font, 0.28, (140, 140, 140), 1, cv2.LINE_AA)

    return hud


# ==============================================================================
# MAIN WRAPPER CLASS: ChainGridMovieGenerator
# ==============================================================================
class ChainGridMovieGenerator:
    """
    High-performance wrapper generating synchronized multi-syllable chain grid movies.
    """
    def __init__(
        self,
        results,
        video_frame_indexes,
        video_paths,
        centroid_mode="follow_centroid",
        sync_mode="time_warp",
        alignment_mode="show_variations", # 'show_variations' vs 'compact_template'
        template_tokens=None,
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=0.75,
        fps=40,
        dot_radius=5,
        transition_flash_color=(0, 255, 255), # Cyan
        transition_flash_frames=4,
        show_hud=True,
        hud_height=58,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        blind_labels=False,
        use_gpu_encoder=True,
        reset_heading_per_syllable=False,
        suptitle_text=None
    ):
        self.results = results
        self.video_frame_indexes = video_frame_indexes
        self.video_paths = video_paths
        
        self.centroid_mode = centroid_mode
        self.sync_mode = sync_mode
        self.alignment_mode = alignment_mode
        self.template_tokens = template_tokens
        self.suptitle_text = suptitle_text
        
        self.rows = rows
        self.cols = cols
        self.n_tiles = rows * cols
        
        self.window_size = window_size
        self.scaled_window_size = scaled_window_size
        self.fps = fps
        self.pre_frames = int(pre_sec * fps)
        self.post_frames = int(post_sec * fps)
        
        self.dot_radius = dot_radius
        self.transition_flash_color = transition_flash_color
        self.transition_flash_frames = transition_flash_frames
        
        self.show_hud = show_hud
        self.hud_height = hud_height if show_hud else 0
        self.show_trajectory_inset = show_trajectory_inset
        self.show_similarity_panel = show_similarity_panel
        self.blind_labels = blind_labels
        self.use_gpu_encoder = use_gpu_encoder
        self.reset_heading_per_syllable = reset_heading_per_syllable

    def _build_temporal_schedule(self, instances):
        """
        Builds frame-by-frame source playback indices for each tile according
        to the chosen synchronization and alignment modes.
        """
        schedules = []
        
        # Check if template alignment is active
        is_aligned_mode = False
        if self.template_tokens is not None or any(inst.get("aligned_stages") for inst in instances):
            is_aligned_mode = True

        if is_aligned_mode and self.template_tokens is not None:
            # Build global alignment across instances
            tpl = self.template_tokens
            alignments = []
            for inst in instances:
                if inst.get("aligned_stages"):
                    alignments.append(inst["aligned_stages"])
                else:
                    ops = align_tokens_to_template(inst["stage_tokens"], tpl)
                    alignments.append(ops)

            # Determine global stages
            # We construct a canonical sequence: for each template anchor, check for preceding insertions
            global_stages = []
            for j, t_tok in enumerate(tpl):
                # Check for insertions before t_tok
                ins_tokens = set()
                for ops in alignments:
                    for op in ops:
                        if op["type"] == "ins" and op["tpl_idx"] == j:
                            ins_tokens.add(op["observed_token"])
                for itok in sorted(list(ins_tokens)):
                    global_stages.append({"kind": "insertion", "token": itok, "anchor_idx": j})
                # Add template anchor
                global_stages.append({"kind": "anchor", "token": t_tok, "anchor_idx": j})

            # Check suffix insertions after last anchor
            suffix_ins = set()
            for ops in alignments:
                for op in ops:
                    if op["type"] == "ins" and op["tpl_idx"] is None:
                        suffix_ins.add(op["observed_token"])
            for itok in sorted(list(suffix_ins)):
                global_stages.append({"kind": "insertion", "token": itok, "anchor_idx": len(tpl)})

            # In compact_template mode, drop insertions
            if self.alignment_mode == "compact_template":
                active_global_stages = [gs for gs in global_stages if gs["kind"] == "anchor"]
            else:
                active_global_stages = global_stages

            # Compute stage durations
            stage_lens = []
            for inst, ops in zip(instances, alignments):
                lens_dict = {}
                sb = inst["stage_boundaries"]
                for k in range(len(sb) - 1):
                    lens_dict[k] = max(1, sb[k+1] - sb[k])
                stage_lens.append(lens_dict)

            # Target duration per active global stage
            target_stage_durations = []
            for gs in active_global_stages:
                dur_samples = []
                for inst, ops in zip(instances, alignments):
                    sb = inst["stage_boundaries"]
                    for op in ops:
                        if gs["kind"] == "anchor" and op["template_token"] == gs["token"] and op["type"] in ("match", "sub"):
                            oi = op["obs_idx"]
                            dur_samples.append(sb[oi+1] - sb[oi])
                        elif gs["kind"] == "insertion" and op["observed_token"] == gs["token"] and op["type"] == "ins":
                            oi = op["obs_idx"]
                            dur_samples.append(sb[oi+1] - sb[oi])
                target_stage_durations.append(int(np.median(dur_samples)) if dur_samples else 15)

            total_chain_display = sum(target_stage_durations)
            total_movie_frames = self.pre_frames + total_chain_display + self.post_frames

            # Build schedules per instance
            for inst, ops in zip(instances, alignments):
                sb = inst["stage_boundaries"]
                sched = []
                start_f = sb[0]
                end_f = sb[-1]
                max_valid_f = len(self.results[inst["recording_key"]]["syllable"]) - 1

                # 1. Pre-padding
                for p in range(self.pre_frames, 0, -1):
                    sched.append({
                        "model_frame": max(0, start_f - p),
                        "stage_idx": -1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "LEAD-IN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })

                # 2. Playback stages
                last_frame = start_f
                for gs, target_d in zip(active_global_stages, target_stage_durations):
                    # Find if this instance participates in gs
                    matched_op = None
                    for op in ops:
                        if gs["kind"] == "anchor" and op["template_token"] == gs["token"] and op["type"] in ("match", "sub"):
                            matched_op = op
                            break
                        elif gs["kind"] == "insertion" and op["observed_token"] == gs["token"] and op["type"] == "ins":
                            matched_op = op
                            break

                    if matched_op is not None:
                        # Instance performs this stage!
                        oi = matched_op["obs_idx"]
                        orig_start = sb[oi]
                        orig_len = sb[oi+1] - sb[oi]
                        
                        # Check cut badge (if compact_template skipped preceding insertion)
                        cut_badge = None
                        if self.alignment_mode == "compact_template":
                            # Check if preceding insertion was skipped
                            for prev_op in ops:
                                if prev_op["type"] == "ins" and prev_op["obs_idx"] < oi:
                                    ins_tok = prev_op["observed_token"]
                                    ins_dur = (sb[prev_op["obs_idx"]+1] - sb[prev_op["obs_idx"]]) / self.fps
                                    cut_badge = f"Skipped +{ins_tok.replace('syll_', '')}: {ins_dur:.2f}s"

                        for tau in range(target_d):
                            src_offset = min(orig_len - 1, int(tau * orig_len / target_d))
                            curr_f = orig_start + src_offset
                            last_frame = curr_f
                            sched.append({
                                "model_frame": curr_f,
                                "stage_idx": oi,
                                "is_motif": True,
                                "is_transition": (tau < self.transition_flash_frames),
                                "status": "PLAYING",
                                "cut_badge": cut_badge if tau < 12 else None,
                                "alignment_badge": None
                            })
                    elif gs["kind"] == "insertion":
                        # Panel lacks this insertion -> continue smooth fluid footage without freezing
                        ins_tok = gs["token"].replace("syll_", "")
                        ins_sid = int(ins_tok.lstrip("0") or "0") if ins_tok.isdigit() else None
                        ins_desc = SHORT_BOX_LABELS.get(HUMAN_LABELS.get(ins_sid, ""), ins_tok) if ins_sid is not None else ins_tok
                        for tau in range(target_d):
                            adv_f = min(max_valid_f, last_frame + tau + 1)
                            sched.append({
                                "model_frame": adv_f,
                                "stage_idx": -1,
                                "is_motif": False,
                                "is_transition": False,
                                "status": "POST-CHAIN",
                                "cut_badge": None,
                                "alignment_badge": f"+{ins_desc}"
                            })
                    else: # Anchor deleted by this instance -> continue smooth fluid footage without freezing
                        del_tok = gs["token"].replace("syll_", "")
                        del_sid = int(del_tok.lstrip("0") or "0") if del_tok.isdigit() else None
                        del_desc = SHORT_BOX_LABELS.get(HUMAN_LABELS.get(del_sid, ""), del_tok) if del_sid is not None else del_tok
                        for tau in range(target_d):
                            adv_f = min(max_valid_f, last_frame + tau + 1)
                            sched.append({
                                "model_frame": adv_f,
                                "stage_idx": -1,
                                "is_motif": False,
                                "is_transition": False,
                                "status": "POST-CHAIN",
                                "cut_badge": None,
                                "alignment_badge": f"-{del_desc}"
                            })

                # 3. Post-padding (continuous natural footage)
                for p in range(self.post_frames):
                    sched.append({
                        "model_frame": min(max_valid_f, end_f + p),
                        "stage_idx": len(sb) - 1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "POST-CHAIN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                schedules.append(sched)

            return schedules, total_movie_frames

        # Standard non-aligned schedules (time_warp, wait_for_others, or nothing)
        n_stages_list = [len(inst["stage_boundaries"]) - 1 for inst in instances]
        max_stages = max(n_stages_list)

        stage_lens = []
        for inst in instances:
            sb = inst["stage_boundaries"]
            lens = [max(1, sb[k+1] - sb[k]) for k in range(len(sb) - 1)]
            stage_lens.append(lens)

        if self.sync_mode == "time_warp":
            target_stage_durations = []
            for k in range(max_stages):
                k_durations = [lens[k] for lens in stage_lens if k < len(lens)]
                target_stage_durations.append(int(np.median(k_durations)) if k_durations else 15)

            total_chain_display = sum(target_stage_durations)
            total_movie_frames = self.pre_frames + total_chain_display + self.post_frames

            for inst, lens in zip(instances, stage_lens):
                sb = inst["stage_boundaries"]
                sched = []
                start_f = sb[0]
                end_f = sb[-1]
                max_valid_f = len(self.results[inst["recording_key"]]["syllable"]) - 1

                for p in range(self.pre_frames, 0, -1):
                    sched.append({
                        "model_frame": max(0, start_f - p),
                        "stage_idx": -1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "LEAD-IN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                    
                for k, target_d in enumerate(target_stage_durations):
                    if k < len(lens):
                        orig_start = sb[k]
                        orig_len = lens[k]
                        for tau in range(target_d):
                            src_offset = min(orig_len - 1, int(tau * orig_len / target_d))
                            sched.append({
                                "model_frame": orig_start + src_offset,
                                "stage_idx": k,
                                "is_motif": True,
                                "is_transition": (tau < self.transition_flash_frames),
                                "status": "PLAYING",
                                "cut_badge": None,
                                "alignment_badge": None
                            })
                    else:
                        # Smooth continuous forward lead-out rather than freezing on the last frame
                        for tau in range(target_d):
                            sched.append({
                                "model_frame": min(max_valid_f, end_f + tau + 1),
                                "stage_idx": -1,
                                "is_motif": False,
                                "is_transition": False,
                                "status": "POST-CHAIN",
                                "cut_badge": None,
                                "alignment_badge": None
                            })
                            
                for p in range(self.post_frames):
                    sched.append({
                        "model_frame": min(max_valid_f, end_f + p),
                        "stage_idx": len(lens),
                        "is_motif": False,
                        "is_transition": False,
                        "status": "POST-CHAIN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                schedules.append(sched)

        elif self.sync_mode == "wait_for_others":
            target_stage_durations = []
            for k in range(max_stages):
                k_durations = [lens[k] for lens in stage_lens if k < len(lens)]
                target_stage_durations.append(max(k_durations) if k_durations else 15)

            total_chain_display = sum(target_stage_durations)
            total_movie_frames = self.pre_frames + total_chain_display + self.post_frames

            for inst, lens in zip(instances, stage_lens):
                sb = inst["stage_boundaries"]
                sched = []
                start_f = sb[0]
                end_f = sb[-1]
                max_valid_f = len(self.results[inst["recording_key"]]["syllable"]) - 1

                for p in range(self.pre_frames, 0, -1):
                    sched.append({
                        "model_frame": max(0, start_f - p),
                        "stage_idx": -1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "LEAD-IN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                    
                for k, target_d in enumerate(target_stage_durations):
                    if k < len(lens):
                        orig_start = sb[k]
                        orig_len = lens[k]
                        for tau in range(target_d):
                            curr_f = min(max_valid_f, orig_start + tau)
                            is_m = (tau < orig_len)
                            sched.append({
                                "model_frame": curr_f,
                                "stage_idx": k if is_m else -1,
                                "is_motif": is_m,
                                "is_transition": (tau < self.transition_flash_frames) if is_m else False,
                                "status": "PLAYING" if is_m else "POST-CHAIN",
                                "cut_badge": None,
                                "alignment_badge": None
                            })
                    else:
                        for tau in range(target_d):
                            sched.append({
                                "model_frame": min(max_valid_f, end_f + tau + 1),
                                "stage_idx": -1,
                                "is_motif": False,
                                "is_transition": False,
                                "status": "POST-CHAIN",
                                "cut_badge": None,
                                "alignment_badge": None
                            })
                            
                for p in range(self.post_frames):
                    sched.append({
                        "model_frame": min(max_valid_f, end_f + p),
                        "stage_idx": len(lens),
                        "is_motif": False,
                        "is_transition": False,
                        "status": "POST-CHAIN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                schedules.append(sched)

        else: # "nothing" / natural speed
            total_durations = [self.pre_frames + (inst["end_frame"] - inst["start_frame"]) + self.post_frames for inst in instances]
            total_movie_frames = max(total_durations)

            for inst in instances:
                sb = inst["stage_boundaries"]
                start_f = sb[0]
                end_f = sb[-1]
                dur = end_f - start_f
                sched = []
                max_valid_f = len(self.results[inst["recording_key"]]["syllable"]) - 1

                for p in range(self.pre_frames, 0, -1):
                    sched.append({
                        "model_frame": max(0, start_f - p),
                        "stage_idx": -1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "LEAD-IN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })

                curr_stage = 0
                for f_rel in range(dur):
                    curr_model_f = start_f + f_rel
                    while curr_stage < len(sb) - 2 and curr_model_f >= sb[curr_stage + 1]:
                        curr_stage += 1
                    is_trans = (curr_model_f - sb[curr_stage] < self.transition_flash_frames)
                    sched.append({
                        "model_frame": curr_model_f,
                        "stage_idx": curr_stage,
                        "is_motif": True,
                        "is_transition": is_trans,
                        "status": "PLAYING",
                        "cut_badge": None,
                        "alignment_badge": None
                    })

                for p in range(self.post_frames):
                    sched.append({
                        "model_frame": min(max_valid_f, end_f + p),
                        "stage_idx": len(sb) - 1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "LEAD-OUT",
                        "cut_badge": None,
                        "alignment_badge": None
                    })

                # Fluid continuous natural progression rather than freezing on the final frame
                extra_p = 0
                while len(sched) < total_movie_frames:
                    extra_p += 1
                    sched.append({
                        "model_frame": min(max_valid_f, end_f + self.post_frames + extra_p),
                        "stage_idx": -1,
                        "is_motif": False,
                        "is_transition": False,
                        "status": "POST-CHAIN",
                        "cut_badge": None,
                        "alignment_badge": None
                    })
                schedules.append(sched)

        return schedules, total_movie_frames

    def render(self, raw_instances, output_path, schedules=None):
        """
        Renders the grid movie and saves as MP4.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        instances = standardize_chain_instances(raw_instances, self.results, self.video_frame_indexes)[:self.n_tiles]
        if len(instances) < self.n_tiles:
            while len(instances) < self.n_tiles:
                instances.append(instances[len(instances) % len(raw_instances)])

        # Build dynamic color palette from unique concurrent tokens in this movie
        all_movie_tokens = []
        if self.template_tokens:
            all_movie_tokens.extend(self.template_tokens)
        for inst in instances:
            all_movie_tokens.extend(inst["stage_tokens"])
        unique_tokens = list(dict.fromkeys(all_movie_tokens))
        token_colormap = build_concurrent_token_colormap(unique_tokens)

        # Build playback schedule
        if schedules is None:
            schedules, total_frames = self._build_temporal_schedule(instances)
        else:
            total_frames = len(schedules[0])

        # Precalculate stage durations, warp factors, and relative turn angles for HUD
        instance_stage_durations = []
        for inst in instances:
            sb = inst["stage_boundaries"]
            rk = inst["recording_key"]
            dur_list = []
            for k in range(len(sb) - 1):
                orig_len = max(1, sb[k+1] - sb[k])
                dur_s = orig_len / self.fps
                # Determine warp factor
                warp_factor = 1.0
                if self.sync_mode == "time_warp":
                    # target duration
                    k_all = [max(1, inst_other["stage_boundaries"][k+1] - inst_other["stage_boundaries"][k])
                             for inst_other in instances if k < len(inst_other["stage_boundaries"]) - 1]
                    target_len = int(np.median(k_all)) if k_all else orig_len
                    warp_factor = orig_len / target_len
                    
                # Relative turn angle
                turn_str = None
                if self.reset_heading_per_syllable:
                    if k == 0:
                        turn_str = "0d"
                    else:
                        h_prev = self.results[rk]["heading"][sb[k-1]]
                        h_curr = self.results[rk]["heading"][sb[k]]
                        diff = (h_curr - h_prev + np.pi) % (2 * np.pi) - np.pi
                        deg = float(np.degrees(diff))
                        if abs(deg) < 15.0:
                            turn_str = "Fwd"
                        elif deg > 0:
                            turn_str = f"+{deg:.0f}dL"
                        else:
                            turn_str = f"-{abs(deg):.0f}dR"
                            
                dur_list.append((dur_s, warp_factor, turn_str))
            instance_stage_durations.append(dur_list)

        # Precompute trajectories and onset headers
        onset_headings = []
        onset_centroids = []
        inst_all_centroids = []
        inst_all_tokens = []
        for inst in instances:
            rk = inst["recording_key"]
            start_f = inst["start_frame"]
            end_f = inst["end_frame"]
            h0 = float(self.results[rk]["heading"][start_f])
            c0 = self.results[rk]["centroid"][start_f]
            onset_headings.append(h0)
            onset_centroids.append(c0)
            
            # Trajectory slice
            c_slice = self.results[rk]["centroid"][start_f:end_f]
            s_slice = self.results[rk]["syllable"][start_f:end_f]
            inst_all_centroids.append(c_slice)
            inst_all_tokens.append([f"syll_{int(s):03d}" for s in s_slice])

        # Tile and grid dimensions
        tile_w = self.scaled_window_size
        tile_video_h = self.scaled_window_size
        tile_total_h = tile_video_h + self.hud_height
        
        grid_w = self.cols * tile_w
        suptitle_h = 32 if self.suptitle_text else 0
        grid_h = self.rows * tile_total_h + suptitle_h

        suptitle_img = None
        if self.suptitle_text:
            suptitle_img = np.full((suptitle_h, grid_w, 3), 18, dtype=np.uint8)
            cv2.line(suptitle_img, (0, suptitle_h - 1), (grid_w, suptitle_h - 1), (60, 60, 65), 1)
            s_font = cv2.FONT_HERSHEY_SIMPLEX
            s_scale = 0.36
            (sw, sh), _ = cv2.getTextSize(self.suptitle_text, s_font, s_scale, 1)
            if sw > grid_w - 20:
                s_scale = 0.32
                (sw, sh), _ = cv2.getTextSize(self.suptitle_text, s_font, s_scale, 1)
            if sw > grid_w - 20:
                s_scale = 0.28
                (sw, sh), _ = cv2.getTextSize(self.suptitle_text, s_font, s_scale, 1)
            sx = max(10, (grid_w - sw) // 2)
            sy = (suptitle_h + sh) // 2
            cv2.putText(suptitle_img, self.suptitle_text, (sx, sy), s_font, s_scale, (230, 230, 230), 1, cv2.LINE_AA)

        print(f"[RENDER] Rendering {output_path.name} | Sync: {self.sync_mode} | Alignment: {self.alignment_mode} | Camera: {self.centroid_mode} | Frames: {total_frames} ({total_frames/self.fps:.1f}s)", flush=True)

        # Open video readers
        readers = {}
        for inst in instances:
            rk = inst["recording_key"]
            if rk not in readers:
                vpath = self.video_paths.get(rk)
                if not vpath or not Path(vpath).exists():
                    raise FileNotFoundError(f"Video file for {rk} not found: {vpath}")
                readers[rk] = FastVideoReader(vpath)

        # Encoder setup: NVENC -> CPU libx264 -> OpenCV mp4v
        use_ffmpeg = False
        ffmpeg_proc = None
        out_writer = None

        if self.use_gpu_encoder:
            ffmpeg_env = os.environ.copy()
            ffmpeg_env.pop("CUDA_VISIBLE_DEVICES", None)
            try:
                ffmpeg_cmd_nvenc = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{grid_w}x{grid_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(self.fps),
                    "-i", "-",
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",
                    "-cq", "22",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ]
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd_nvenc,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=ffmpeg_env
                )
                use_ffmpeg = True
                print("[RENDER] Hardware-accelerated GPU NVENC encoder initialized.", flush=True)
            except Exception as e:
                print(f"[RENDER] NVENC init skipped: {e}", flush=True)
                use_ffmpeg = False

        if not use_ffmpeg:
            try:
                ffmpeg_cmd_cpu = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{grid_w}x{grid_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(self.fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ]
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd_cpu,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                use_ffmpeg = True
                print("[RENDER] Multithreaded CPU libx264 encoder initialized.", flush=True)
            except Exception:
                use_ffmpeg = False

        if not use_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (grid_w, grid_h))
            print("[RENDER] Standard OpenCV VideoWriter initialized.", flush=True)

        t0 = time.time()

        # Frame Generation Loop
        for frame_idx in range(total_frames):
            grid_tiles = []
            
            for tile_idx, (inst, sched) in enumerate(zip(instances, schedules)):
                rk = inst["recording_key"]
                reader = readers[rk]
                vfi = self.video_frame_indexes[rk]
                
                step = sched[frame_idx]
                mod_f = step["model_frame"]
                vid_f = int(vfi[min(len(vfi) - 1, mod_f)])
                
                raw_frame = reader.get_frame(vid_f)
                
                h0 = onset_headings[tile_idx]
                if self.reset_heading_per_syllable:
                    curr_s_idx = step["stage_idx"]
                    sb = inst["stage_boundaries"]
                    if 0 <= curr_s_idx < len(sb) - 1:
                        st_f = sb[curr_s_idx]
                        h_frame = float(self.results[rk]["heading"][st_f])
                    elif curr_s_idx < 0:
                        h_frame = h0
                    else:
                        st_f = sb[-2] if len(sb) >= 2 else inst["start_frame"]
                        h_frame = float(self.results[rk]["heading"][st_f])
                else:
                    h_frame = h0

                if self.centroid_mode == "follow_centroid":
                    c_curr = self.results[rk]["centroid"][mod_f]
                else: # lock_centroid
                    c_curr = onset_centroids[tile_idx]

                r_mat = np.float32([[np.cos(h_frame), np.sin(h_frame)], [-np.sin(h_frame), np.cos(h_frame)]])
                c_trans = r_mat @ c_curr - self.window_size // 2
                M = np.float32([
                    [np.cos(h_frame), np.sin(h_frame), -c_trans[0]],
                    [-np.sin(h_frame), np.cos(h_frame), -c_trans[1]]
                ])

                warped = cv2.warpAffine(raw_frame, M, (self.window_size, self.window_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                tile_img = cv2.resize(warped, (self.scaled_window_size, self.scaled_window_size))

                if self.reset_heading_per_syllable:
                    cv2.putText(tile_img, "[HEADING RESET]", (8, self.scaled_window_size - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

                # Active syllable indicator dot directly on mouse centroid (KPMS standard)
                curr_s_idx = step["stage_idx"]
                if step["is_motif"] and self.dot_radius > 0:
                    curr_tok = inst["stage_tokens"][curr_s_idx] if 0 <= curr_s_idx < len(inst["stage_tokens"]) else None
                    dot_c = token_colormap.get(curr_tok, (255, 255, 255))
                    dot_r = self.dot_radius

                    # Centroid dot directly following the mouse in its syllable color
                    c_actual = self.results[rk]["centroid"][mod_f]
                    scale_factor = self.scaled_window_size / float(self.window_size)
                    c_warped = (r_mat @ c_actual - c_trans) * scale_factor
                    cx = int(round(c_warped[0]))
                    cy = int(round(c_warped[1]))
                    if 0 <= cx < self.scaled_window_size and 0 <= cy < self.scaled_window_size:
                        cv2.circle(tile_img, (cx, cy), dot_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(tile_img, (cx, cy), dot_r, dot_c, -1, cv2.LINE_AA)

                # 2D Trajectory Inset in top-right corner
                if self.show_trajectory_inset:
                    rel_f_idx = mod_f - inst["start_frame"]
                    draw_trajectory_inset(
                        tile_img=tile_img,
                        all_centroids=inst_all_centroids[tile_idx],
                        all_tokens=inst_all_tokens[tile_idx],
                        curr_frame_idx=rel_f_idx,
                        onset_heading=h0,
                        token_colormap=token_colormap,
                        inset_size=74,
                        margin=8
                    )

                # HUD row
                if self.show_hud:
                    elapsed_s = max(0.0, (mod_f - inst["start_frame"]) / self.fps)
                    total_s = max(0.1, (inst["end_frame"] - inst["start_frame"]) / self.fps)
                    drug_raw = str(inst.get("drug", ""))
                    drug_short = "Sal" if "sal" in drug_raw.lower() else ("Amp" if "amp" in drug_raw.lower() else drug_raw[:3])
                    tile_title = f"M{inst['animal_id']} ({inst['genotype']}, {drug_short})"
                    
                    sim_dict = None
                    if self.show_similarity_panel:
                        sim_dict = {}
                        if inst.get("proto_cost") is not None: sim_dict["proto_cost"] = inst["proto_cost"]
                        if inst.get("kin_dist") is not None: sim_dict["kin_dist"] = inst["kin_dist"]
                    
                    hud_img = draw_stage_hud(
                        tile_w=tile_w,
                        hud_h=self.hud_height,
                        tokens=inst["stage_tokens"],
                        token_colormap=token_colormap,
                        current_stage_idx=step["stage_idx"],
                        is_transition_frame=step["is_transition"],
                        elapsed_sec=elapsed_s,
                        total_sec=total_s,
                        stage_durations=instance_stage_durations[tile_idx],
                        status_text=step["status"],
                        tile_title=tile_title,
                        similarity_info=sim_dict,
                        blind_labels=self.blind_labels,
                        cut_badge=step.get("cut_badge"),
                        alignment_badge=step.get("alignment_badge")
                    )
                    full_tile = np.vstack([tile_img, hud_img])
                else:
                    full_tile = tile_img

                grid_tiles.append(full_tile)

            # Assemble grid (rows x cols)
            row_strips = []
            for r in range(self.rows):
                row_tiles = grid_tiles[r * self.cols : (r + 1) * self.cols]
                row_strips.append(np.hstack(row_tiles))
            full_frame = np.vstack(row_strips)
            if suptitle_img is not None:
                full_frame = np.vstack([suptitle_img, full_frame])

            # Write frame
            if use_ffmpeg:
                try:
                    ffmpeg_proc.stdin.write(full_frame.tobytes())
                except (BrokenPipeError, IOError):
                    use_ffmpeg = False
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (grid_w, grid_h))
                    out_writer.write(full_frame)
            else:
                out_writer.write(full_frame)

        # Cleanup
        if use_ffmpeg and ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            try:
                ffmpeg_proc.wait(timeout=10)
            except Exception:
                ffmpeg_proc.kill()
        elif out_writer is not None:
            out_writer.release()

        for r in readers.values():
            r.close()

        total_wall = time.time() - t0
        print(f"[SUCCESS] Movie rendered in {total_wall:.2f}s ({total_frames / total_wall:.1f} FPS) -> {output_path}", flush=True)
        return str(output_path)


# ==============================================================================
# BRANCH-CENTRED GRID GENERATOR
# ==============================================================================
def generate_branch_grid_movie(
    prefix_tokens,
    branch_groups,
    output_path,
    resources=None,
    centroid_mode="follow_centroid",
    rows=2,
    cols=3,
    window_size=240,
    scaled_window_size=320,
    pre_sec=0.75,
    post_sec=0.75,
    fps=40,
    dot_radius=5,
    use_gpu_encoder=True
):
    """
    Generates a branch-centred grid movie synchronized at the exact exit frame
    of a shared behavioral prefix.
    """
    if resources is None:
        resources = load_chain_movie_resources()

    results = resources["results"]
    vfi = resources["video_frame_indexes"]
    video_paths = resources["video_paths"]

    # Flatten branch groups into grid tiles arranged by column
    ordered_instances = []
    branch_keys = list(branch_groups.keys())[:cols]
    for r in range(rows):
        for c, b_tok in enumerate(branch_keys):
            inst_list = branch_groups[b_tok]
            inst = inst_list[r % len(inst_list)]
            ordered_instances.append(inst)

    std_instances = standardize_chain_instances(ordered_instances, results, vfi)
    
    # Extract unique tokens
    all_toks = list(prefix_tokens) + branch_keys
    colormap = build_concurrent_token_colormap(all_toks)

    # Prefix duration: median duration of prefix across all instances
    n_prefix = len(prefix_tokens)
    pre_frames = int(pre_sec * fps)
    post_frames = int(post_sec * fps)
    
    prefix_durations = []
    for k in range(n_prefix):
        k_durs = [inst["stage_boundaries"][k+1] - inst["stage_boundaries"][k] for inst in std_instances]
        prefix_durations.append(int(np.median(k_durs)) if k_durs else 15)
        
    branch_target_dur = 25 # ~0.6s continuation
    total_frames = pre_frames + sum(prefix_durations) + branch_target_dur + post_frames
    branch_sync_frame = pre_frames + sum(prefix_durations)

    # Build schedules synchronized at branch point
    schedules = []
    for inst in std_instances:
        sb = inst["stage_boundaries"]
        sched = []
        max_valid_f = len(results[inst["recording_key"]]["syllable"]) - 1
        
        # 1. Pre-padding
        start_f = sb[0]
        for p in range(pre_frames, 0, -1):
            sched.append({
                "model_frame": max(0, start_f - p),
                "stage_idx": -1,
                "is_motif": False,
                "is_transition": False,
                "status": "LEAD-IN",
                "is_branch": False
            })
            
        # 2. Resampled prefix
        for k, target_d in enumerate(prefix_durations):
            orig_start = sb[k]
            orig_len = max(1, sb[k+1] - sb[k])
            for tau in range(target_d):
                src_off = min(orig_len - 1, int(tau * orig_len / target_d))
                sched.append({
                    "model_frame": orig_start + src_off,
                    "stage_idx": k,
                    "is_motif": True,
                    "is_transition": (tau < 4),
                    "status": "PREFIX",
                    "is_branch": False
                })
                
        # 3. Continuation branch
        branch_start = sb[n_prefix] if len(sb) > n_prefix else sb[-1]
        branch_end = sb[n_prefix+1] if len(sb) > n_prefix + 1 else min(max_valid_f, branch_start + branch_target_dur)
        branch_len = max(1, branch_end - branch_start)
        for tau in range(branch_target_dur):
            src_off = min(branch_len - 1, int(tau * branch_len / branch_target_dur))
            sched.append({
                "model_frame": branch_start + src_off,
                "stage_idx": n_prefix,
                "is_motif": True,
                "is_transition": (tau < 6), # Branch flash
                "status": "BRANCH",
                "is_branch": True
            })
            
        # 4. Post-padding (continuous ongoing footage)
        end_f = branch_end
        for p in range(post_frames):
            sched.append({
                "model_frame": min(max_valid_f, end_f + p),
                "stage_idx": -1,
                "is_motif": False,
                "is_transition": False,
                "status": "POST-BRANCH",
                "is_branch": False
            })
        schedules.append(sched)

    # Use generator render loop with branch titles
    gen = ChainGridMovieGenerator(
        results=results,
        video_frame_indexes=vfi,
        video_paths=video_paths,
        centroid_mode=centroid_mode,
        sync_mode="time_warp",
        rows=rows,
        cols=cols,
        window_size=window_size,
        scaled_window_size=scaled_window_size,
        pre_sec=pre_sec,
        post_sec=post_sec,
        fps=fps,
        dot_radius=dot_radius,
        use_gpu_encoder=use_gpu_encoder
    )
    
    # Custom titles per tile indicating the branch
    for idx, inst in enumerate(std_instances):
        b_tok = inst["stage_tokens"][-1] if len(inst["stage_tokens"]) > n_prefix else "Continuation"
        inst["animal_id"] = f"-> {b_tok.replace('syll_', '')}"
        inst["drug"] = inst["drug"][:3] # 'Sal' or 'Amp'

    return gen.render(std_instances, output_path, schedules=schedules)


# ==============================================================================
# FUNCTIONAL CONVENIENCE INTERFACE
# ==============================================================================
def generate_chain_grid_movie(
    instances,
    output_path,
    resources=None,
    centroid_mode="follow_centroid",
    sync_mode="time_warp",
    alignment_mode="show_variations",
    template_tokens=None,
    rows=2,
    cols=3,
    window_size=240,
    scaled_window_size=320,
    pre_sec=0.75,
    post_sec=0.75,
    fps=40,
    dot_radius=5,
    show_hud=True,
    show_trajectory_inset=False,
    show_similarity_panel=True,
    blind_labels=False,
    use_gpu_encoder=True,
    filter_outliers=False,
    reset_heading_per_syllable=False,
    suptitle_text=None
):
    """
    Convenience wrapper function matching the plan specification.
    """
    if resources is None:
        resources = load_chain_movie_resources()

    if filter_outliers and len(instances) > rows * cols:
        instances, _ = filter_chain_outliers(
            instances,
            resources["results"],
            target_count=rows * cols,
            video_frame_indexes=resources.get("video_frame_indexes")
        )
        
    generator = ChainGridMovieGenerator(
        results=resources["results"],
        video_frame_indexes=resources["video_frame_indexes"],
        video_paths=resources["video_paths"],
        centroid_mode=centroid_mode,
        sync_mode=sync_mode,
        alignment_mode=alignment_mode,
        template_tokens=template_tokens,
        rows=rows,
        cols=cols,
        window_size=window_size,
        scaled_window_size=scaled_window_size,
        pre_sec=pre_sec,
        post_sec=post_sec,
        fps=fps,
        dot_radius=dot_radius,
        show_hud=show_hud,
        show_trajectory_inset=show_trajectory_inset,
        show_similarity_panel=show_similarity_panel,
        blind_labels=blind_labels,
        use_gpu_encoder=use_gpu_encoder,
        reset_heading_per_syllable=reset_heading_per_syllable,
        suptitle_text=suptitle_text
    )
    return generator.render(instances, output_path)
