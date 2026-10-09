#!/usr/bin/env python3
"""
render_comparison_gallery.py
=============================================================================
Renders the 24 Primary 1x3 Widescreen Comparison Movies:
- 3 Themes: Theme A, Theme B, Theme C
- 2 Methods: seqNMF (Arm 1) vs. Mixed Probabilistic Episode Model (Arm 3)
- 4 Operating Points per combination:
    1. Canonical Exemplar (Top similarity / lowest dispersion)
    2. P80 Representative (80th percentile of continuous movement similarity)
    3. P99 Representative (99th percentile of continuous movement similarity)
    4. Ordinary / Random Member (Unconditioned representative from population)

Features:
- Native 40 fps fluid playback without pauses
- Per-panel HUD displaying the ACTUAL observed syllable tokens and durations
  for that specific clip (enabling direct inspection of HSMM's flexible substitutions
  vs. seqNMF's rigid chains)
- Multi-animal panel diversity (panels 1, 2, 3 selected from distinct mice)
- Real-time 2D spatial trajectory inset
- Generates thumbnail PNG and lightweight animated GIF preview for each movie
=============================================================================
"""

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# Compatibility bridge for NumPy unpickling
import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import (
    DATA_DIR,
    MOVIES_DIR,
    THEMES,
    THUMBS_DIR,
)

# Load Human Syllable Labels
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


def select_panels_for_target(candidates, target_type, target_val=None, rng_seed=42):
    """
    Selects up to 3 panels from distinct animals matching the 3 standardized tiers:
    - Strictly restricts candidates to 3-5 syllable chains
    1. 'exemplar': top similarity / closest to prototypical motif (sorted descending)
    2. 'p80': 80th percentile operating point (demonstrating top 20% cohesive core)
    3. 'random': unbiased random draw across distinct animals from the unpruned population
    """
    # Strictly enforce 3-5 syllable chains
    valid_cands = [c for c in candidates if 3 <= len(c.get("stage_tokens", [])) <= 5]
    if len(valid_cands) >= 3:
        candidates = valid_cands
    elif len(candidates) == 0:
        return []

    sims = np.array([c["movement_similarity"] for c in candidates])

    if target_type == "exemplar":
        # Top similarity (closest to exemplar / density mode)
        order = np.argsort(-sims)
    elif target_type == "p80":
        # Operating point near the 80th percentile threshold
        diff = np.abs(sims - target_val)
        order = np.argsort(diff)
    elif target_type == "random":
        # Unbiased random draw across distinct animals from the unpruned population
        rng = np.random.RandomState(rng_seed)
        indices = np.arange(len(candidates))
        rng.shuffle(indices)
        order = indices
    else:
        order = np.arange(len(candidates))

    selected = []
    seen_animals = set()

    for idx in order:
        c = candidates[idx]
        aid = c["animal_id"]
        if aid not in seen_animals:
            c_copy = dict(c)
            c_copy["proto_cost"] = float(abs(c["movement_similarity"] - (target_val if target_val is not None else 1.0)))
            c_copy["kin_dist"] = float(c["movement_similarity"])
            selected.append(c_copy)
            seen_animals.add(aid)
            if len(selected) >= 3:
                break

    # Fallback if fewer than 3 distinct animals
    if len(selected) < 3:
        for idx in order:
            c = candidates[idx]
            if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in selected}:
                c_copy = dict(c)
                c_copy["proto_cost"] = float(abs(c["movement_similarity"] - (target_val if target_val is not None else 1.0)))
                c_copy["kin_dist"] = float(c["movement_similarity"])
                selected.append(c_copy)
                if len(selected) >= 3:
                    break

    return selected


def render_gallery():
    print("=" * 80)
    print("RENDERING 18 PRIMARY 1x3 WIDESCREEN COMPARISON MOVIES (3 STANDARDIZED TIERS)")
    print("=" * 80)
    t0 = time.time()

    # Load master candidates
    master_pkl = DATA_DIR / "pilot_master_candidates.pkl"
    with open(master_pkl, "rb") as f:
        data = pickle.load(f)

    arm1_cands = data["arm1_candidates"]
    arm3_cands = data["arm3_candidates"]

    # Load movie resources
    print("[1/3] Loading video rendering resources...")
    res = load_chain_movie_resources()

    gallery_manifest = []

    # Standardized 3 visual tiers
    operating_points = [
        {"tag": "exemplar", "title": "Exemplar (Closest to Prototypical Motif)", "type": "exemplar"},
        {"tag": "p80", "title": "P80 Representative (80th Percentile Core)", "type": "p80", "percentile": 80},
        {"tag": "random", "title": "Random Sampling (Unpruned Population)", "type": "random"}
    ]

    methods = [
        {"method_id": "seqnmf", "name": "seqNMF (Level 2 Soft)", "cands": arm1_cands},
        {"method_id": "prob_hsmm", "name": "Mixed HSMM (Raw Syllables)", "cands": arm3_cands}
    ]

    total_movies = len(THEMES) * len(methods) * len(operating_points)
    movie_idx = 0

    print(f"[2/3] Rendering {total_movies} 1x3 Widescreen Movies (Native 40 fps)...")

    for th_id, th_meta in THEMES.items():
        th_title = th_meta["title"]

        for m_info in methods:
            m_id = m_info["method_id"]
            m_name = m_info["name"]
            # Filter pool strictly for 3-5 syllable chains
            pool = [c for c in m_info["cands"] if c["theme_id"] == th_id and 3 <= len(c.get("stage_tokens", [])) <= 5]

            if not pool:
                print(f"  [WARN] No 3-5 syllable candidates for {th_id} in {m_name}, skipping.")
                continue

            sims = np.array([c["movement_similarity"] for c in pool])

            for op in operating_points:
                movie_idx += 1
                op_tag = op["tag"]
                op_title = op["title"]

                # Determine target value
                if op["type"] == "p80":
                    target_sim = float(np.percentile(sims, op["percentile"]))
                elif op["type"] == "exemplar":
                    target_sim = float(np.max(sims))
                else:
                    target_sim = None

                panels = select_panels_for_target(pool, op["type"], target_sim)
                if len(panels) < 3:
                    print(f"  [{movie_idx}/{total_movies}] {th_id} | {m_id} | {op_tag}: Insufficient panels ({len(panels)})")
                    continue

                movie_name = f"movie_1x3_{th_id}_{m_id}_{op_tag}.mp4"
                movie_path = MOVIES_DIR / movie_name
                thumb_name = f"thumb_{th_id}_{m_id}_{op_tag}.png"
                thumb_path = THUMBS_DIR / thumb_name
                gif_name = f"preview_{th_id}_{m_id}_{op_tag}.gif"
                gif_path = THUMBS_DIR / gif_name
                skip_render = False

                if skip_render:
                    print(f"  [{movie_idx:2d}/{total_movies}] Keeping existing seqNMF: {movie_name}")
                else:
                    print(f"  [{movie_idx:2d}/{total_movies}] Rendering: {movie_name} (Animals: {[p['animal_id'] for p in panels]}, Sims: {[round(p['movement_similarity'], 2) for p in panels]})")

                    # Generate 1x3 widescreen video with idealized chain suptitle banner
                    th_meta = THEMES[th_id]
                    idealized_str = th_meta.get("idealized_chain", "")
                    suptitle_str = f"{th_title} | Idealized: {idealized_str} | {m_name} [{op_title}]"

                    generate_chain_grid_movie(
                        instances=panels,
                        output_path=movie_path,
                        resources=res,
                        centroid_mode="follow_centroid",
                        sync_mode="nothing",
                        rows=1,
                        cols=3,
                        window_size=240,
                        scaled_window_size=320,
                        pre_sec=0.5,
                        post_sec=0.75,
                        fps=40,
                        dot_radius=5,
                        show_hud=True,
                        show_trajectory_inset=False,
                        show_similarity_panel=False,
                        use_gpu_encoder=True,
                        suptitle_text=suptitle_str
                    )

                    # Generate Thumbnail & GIF
                    cap = cv2.VideoCapture(str(movie_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    target_thumb_f = min(25, max(0, total_frames - 1))
                    gif_frames = []
                    frame_gap = max(1, int(40 / 12))  # ~12 fps for GIF
                    count = 0
                    thumb_frame = None

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if count == target_thumb_f and thumb_frame is None:
                            thumb_frame = frame.copy()
                        if count % frame_gap == 0 and len(gif_frames) < 30:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w = rgb.shape[:2]
                            scaled = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                            gif_frames.append(scaled)
                        count += 1
                    cap.release()

                    if thumb_frame is not None:
                        cv2.imwrite(str(thumb_path), thumb_frame)

                    if gif_frames:
                        from PIL import Image
                        pil_frames = [Image.fromarray(f) for f in gif_frames]
                        pil_frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=pil_frames[1:],
                            duration=int(1000 / 12),
                            loop=0,
                            optimize=True
                        )

                gallery_manifest.append({
                    "movie_index": movie_idx,
                    "theme_id": th_id,
                    "theme_title": th_title,
                    "method_id": m_id,
                    "method_name": m_name,
                    "operating_point": op_tag,
                    "operating_title": op_title,
                    "target_similarity": round(target_sim if target_sim is not None else float(np.mean([p["movement_similarity"] for p in panels])), 3),
                    "movie_filename": movie_name,
                    "movie_path": str(movie_path),
                    "thumbnail_filename": thumb_name,
                    "thumbnail_path": str(thumb_path),
                    "gif_filename": gif_name,
                    "gif_path": str(gif_path),
                    "panel_animals": [p["animal_id"] for p in panels],
                    "panel_similarities": [round(p["movement_similarity"], 3) for p in panels],
                    "panel_durations_sec": [round(p["duration_sec"], 2) for p in panels],
                    "panel_tokens": [p["stage_tokens"] for p in panels],
                    "panel_human_tokens": [
                        [
                            HUMAN_LABELS.get(int(t.replace('syll_', '').lstrip('0') or '0'), 'misc')
                            for t in p["stage_tokens"]
                        ]
                        for p in panels
                    ],
                    "panel_chain_descriptions": [
                        " -> ".join([
                            HUMAN_LABELS.get(int(t.replace('syll_', '').lstrip('0') or '0'), 'misc')
                            for t in p["stage_tokens"]
                        ])
                        for p in panels
                    ]
                })

    # Save gallery manifest
    manifest_path = DATA_DIR / "video_gallery_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(gallery_manifest, f, indent=2)

    print(f"\n[3/3] Successfully rendered {len(gallery_manifest)} videos! Manifest saved to: {manifest_path}")
    print(f"Total time: {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    render_gallery()
