#!/usr/bin/env python3
"""
render_2x3_saline_exemplar_videos.py
=============================================================================
Generates 3 Candidate Screening 2x3 Grid Movies (6 panels each) for:
1. Theme 1: Wall-Rear & Descent (lean/rear -> sniff -> descent)
2. Theme 2: Sprint & Turning Redirection (turn left -> walk/run -> turn left)
3. Theme 3: Stationary / Olfactory Inspection & Rest (sniff -> rest -> sniff)

Model: Latest seqNMF baseline (amph_prod_k150k_run_12 / pilot_master_candidates)
Cohort: Saline WT (with top-ranked candidate fallbacks across distinct mice)
Layout: 2 rows x 3 columns (6 instances per video)
Purpose: User will review the 6 subpanels in each video, pick their top 3,
         and narrow them down to final 1x3 widescreen videos.
=============================================================================
"""

import os
import pickle
import sys
import time
from pathlib import Path

import cv2

# Set CPU for JAX/TF safety
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# NumPy compatibility
import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    HUMAN_LABELS,
    generate_chain_grid_movie,
    load_chain_movie_resources,
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/movies_2x3_saline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/probabilistic_episode_pilot/data/pilot_master_candidates.pkl"


def get_theme1_candidates(cands):
    """Theme 1: lean/rear -> sniff -> descent"""
    th_c = [c for c in cands if c['theme_id'] == 'theme_A' and 3 <= len(c.get('stage_tokens', [])) <= 5]
    sal_wt = [c for c in th_c if c.get('genotype') == 'WT' and c.get('drug') == 'Saline']
    sal_any = [c for c in th_c if c.get('drug') == 'Saline']

    sal_wt.sort(key=lambda x: -x['movement_similarity'])
    sal_any.sort(key=lambda x: -x['movement_similarity'])

    selected = []
    seen_animals = set()
    for c in sal_wt:
        aid = c['animal_id']
        if aid not in seen_animals:
            seen_animals.add(aid)
            selected.append(c)
            if len(selected) == 6:
                break

    if len(selected) < 6:
        for c in sal_any:
            aid = c['animal_id']
            if aid not in seen_animals:
                seen_animals.add(aid)
                selected.append(c)
                if len(selected) == 6:
                    break

    return selected


def get_theme2_candidates(cands):
    """Theme 2: turn left -> walk/run -> turn left"""
    th_c = [c for c in cands if c['theme_id'] == 'theme_B' and 3 <= len(c.get('stage_tokens', [])) <= 5]

    # Left-turn starting and left-turn ending in Saline
    # 4: back down + turn left, 6: turn left, 3: walk left, 17: run left, 29: sharp left, 26: dash left
    left_both = [
        c for c in th_c
        if c.get('drug') == 'Saline'
        and any(c['stage_tokens'][0].endswith(x) for x in ['_004', '_006', '_003', '_017', '_029', '_026', '_007'])
        and any(c['stage_tokens'][-1].endswith(x) for x in ['_004', '_006', '_003', '_017', '_029', '_026'])
    ]
    left_both.sort(key=lambda x: -x['movement_similarity'])

    selected = []
    seen_animals = set()
    # First pick distinct WT Saline
    for c in left_both:
        aid = c['animal_id']
        if aid not in seen_animals and c.get('genotype') == 'WT':
            seen_animals.add(aid)
            selected.append(c)
            if len(selected) == 6:
                break

    # Fill remaining from distinct Saline animals
    for c in left_both:
        aid = c['animal_id']
        if aid not in seen_animals:
            seen_animals.add(aid)
            selected.append(c)
            if len(selected) == 6:
                break

    # If still fewer than 6, relax animal uniqueness
    if len(selected) < 6:
        for c in left_both:
            if c not in selected:
                selected.append(c)
                if len(selected) == 6:
                    break

    return selected


def get_theme3_candidates(cands):
    """Theme 3: stationary / sensory rest / sniff -> rest -> sniff"""
    th_c = [c for c in cands if c['theme_id'] == 'theme_C' and 3 <= len(c.get('stage_tokens', [])) <= 5]
    sal_wt = [c for c in th_c if c.get('genotype') == 'WT' and c.get('drug') == 'Saline']
    sal_any = [c for c in th_c if c.get('drug') == 'Saline']

    sal_wt.sort(key=lambda x: -x['movement_similarity'])
    sal_any.sort(key=lambda x: -x['movement_similarity'])

    selected = []
    seen_animals = set()
    for c in sal_wt:
        aid = c['animal_id']
        if aid not in seen_animals:
            seen_animals.add(aid)
            selected.append(c)
            if len(selected) == 6:
                break

    if len(selected) < 6:
        for c in sal_any:
            aid = c['animal_id']
            if aid not in seen_animals:
                seen_animals.add(aid)
                selected.append(c)
                if len(selected) == 6:
                    break

    return selected


def generate_thumbnail(video_path, thumb_path):
    """Saves a high-quality poster frame from the middle of the video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_f = max(0, min(total_frames - 1, int(total_frames * 0.4)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(thumb_path), frame)
    cap.release()


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("🎬 RENDERING 3 CANDIDATE SCREENING 2x3 GRID MOVIES (6 PANELS EACH)", flush=True)
    print(f"   Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load candidates
    print("[1/4] Loading master seqNMF candidates...", flush=True)
    with open(MASTER_PKL, "rb") as f:
        data = pickle.load(f)
    seqnmf_cands = data["arm1_candidates"]
    print(f"Loaded {len(seqnmf_cands)} total candidates.", flush=True)

    # 2. Load movie resources
    print("[2/4] Loading video rendering resources (H5 + VFI + Video map)...", flush=True)
    resources = load_chain_movie_resources()

    # 3. Select candidates
    print("[3/4] Selecting 6 exemplar panels per theme...", flush=True)
    theme1_panels = get_theme1_candidates(seqnmf_cands)
    theme2_panels = get_theme2_candidates(seqnmf_cands)
    theme3_panels = get_theme3_candidates(seqnmf_cands)

    def tag_panels(panels):
        tagged = []
        for idx, p in enumerate(panels):
            p_copy = dict(p)
            p_copy["instance_id"] = f"Panel {idx+1}: M{p['animal_id']}"
            p_copy["animal_id"] = f"P{idx+1}-M{p['animal_id']}"
            tagged.append(p_copy)
        return tagged

    t1_tagged = tag_panels(theme1_panels)
    t2_tagged = tag_panels(theme2_panels)
    t3_tagged = tag_panels(theme3_panels)

    print("\n--- Theme 1 (Rear -> Sniff -> Descent) 6 Candidates ---", flush=True)
    for i, p in enumerate(t1_tagged):
        human_readable = [HUMAN_LABELS.get(int(t.replace("syll_", "").lstrip("0") or 0), t) for t in p['stage_tokens']]
        print(f"  Panel {i+1}: Animal {p['animal_id']} ({p['genotype']}, {p['drug']}) | Sim: {p['movement_similarity']:.3f} | Stages: {' -> '.join(human_readable)}", flush=True)

    print("\n--- Theme 2 (Turn Left -> Walk/Run -> Turn Left) 6 Candidates ---", flush=True)
    for i, p in enumerate(t2_tagged):
        human_readable = [HUMAN_LABELS.get(int(t.replace("syll_", "").lstrip("0") or 0), t) for t in p['stage_tokens']]
        print(f"  Panel {i+1}: Animal {p['animal_id']} ({p['genotype']}, {p['drug']}) | Sim: {p['movement_similarity']:.3f} | Stages: {' -> '.join(human_readable)}", flush=True)

    print("\n--- Theme 3 (Stationary / Olfactory Inspection & Rest) 6 Candidates ---", flush=True)
    for i, p in enumerate(t3_tagged):
        human_readable = [HUMAN_LABELS.get(int(t.replace("syll_", "").lstrip("0") or 0), t) for t in p['stage_tokens']]
        print(f"  Panel {i+1}: Animal {p['animal_id']} ({p['genotype']}, {p['drug']}) | Sim: {p['movement_similarity']:.3f} | Stages: {' -> '.join(human_readable)}", flush=True)

    # 4. Render 2x3 movies
    print("\n[4/4] Rendering 2x3 movies...", flush=True)

    # Movie 1: Theme 1
    m1_path = OUT_DIR / "candidate_2x3_theme1_rear_sniff_descent.mp4"
    t1_thumb = OUT_DIR / "candidate_2x3_theme1_rear_sniff_descent.png"
    print(f"\n[A] Rendering Theme 1 -> {m1_path.name}...", flush=True)
    generate_chain_grid_movie(
        instances=t1_tagged,
        output_path=m1_path,
        resources=resources,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
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
        suptitle_text="Theme 1: Wall-Rear & Descent (lean/rear -> sniff -> descent) [2x3 Candidate Selection Grid]"
    )
    generate_thumbnail(m1_path, t1_thumb)
    print(f"  Saved poster frame -> {t1_thumb.name}", flush=True)

    # Movie 2: Theme 2
    m2_path = OUT_DIR / "candidate_2x3_theme2_turn_run_turn.mp4"
    t2_thumb = OUT_DIR / "candidate_2x3_theme2_turn_run_turn.png"
    print(f"\n[B] Rendering Theme 2 -> {m2_path.name}...", flush=True)
    generate_chain_grid_movie(
        instances=t2_tagged,
        output_path=m2_path,
        resources=resources,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
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
        suptitle_text="Theme 2: Turn Left & Sprint (turn left -> run/dash -> turn left) [2x3 Candidate Selection Grid]"
    )
    generate_thumbnail(m2_path, t2_thumb)
    print(f"  Saved poster frame -> {t2_thumb.name}", flush=True)

    # Movie 3: Theme 3
    m3_path = OUT_DIR / "candidate_2x3_theme3_stationary_rest.mp4"
    t3_thumb = OUT_DIR / "candidate_2x3_theme3_stationary_rest.png"
    print(f"\n[C] Rendering Theme 3 -> {m3_path.name}...", flush=True)
    generate_chain_grid_movie(
        instances=t3_tagged,
        output_path=m3_path,
        resources=resources,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
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
        suptitle_text="Theme 3: Olfactory Inspection & Rest (sniff -> stationary -> sniff) [2x3 Candidate Selection Grid]"
    )
    generate_thumbnail(m3_path, t3_thumb)
    print(f"  Saved poster frame -> {t3_thumb.name}", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"✅ ALL 3 CANDIDATE 2x3 MOVIES GENERATED IN {total_time:.1f}s!", flush=True)
    print(f"1. Theme 1: {m1_path}", flush=True)
    print(f"2. Theme 2: {m2_path}", flush=True)
    print(f"3. Theme 3: {m3_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
