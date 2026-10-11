#!/usr/bin/env python3
"""
render_1x3_theme_B_right_turn_options.py
=============================================================================
Renders 3 distinct 1x3 Widescreen (3-panel) videos for:
Theme B Right-Turn: Turn Right -> Run/Walk Right -> Turn Right

Each video tests a different canonical exemplar anchor:
- Video 1: Exemplar Suite 1 (M1749 Anchor: High-speed ballistic arc)
    Turn_R -> Run_R -> Turn_R ([2, 0, 2])
    Panel 1: Mouse 1749 (KI, Saline) | Panel 2: Mouse 1664 (KI, Amp) | Panel 3: Mouse 1759 (WT, Amp)

- Video 2: Exemplar Suite 2 (M1765 Anchor: Extended clean Saline WT arc)
    Turn_R -> Run_R -> Turn_R ([2, 0, 2])
    Panel 1: Mouse 1765 (WT, Saline) | Panel 2: Mouse 1773 (WT, Amp) | Panel 3: Mouse 1767 (WT, Saline)

- Video 3: Exemplar Suite 3 (M1771 Anchor: Extended Dash sprint arc)
    Turn_R -> Run_R -> Dash -> Turn_R ([2, 0, 13, 2])
    Panel 1: Mouse 1771 (KI, Amp) | Panel 2: Mouse 1775 (WT, Saline) | Panel 3: Mouse 1759 (WT, Saline)

Layout: 1x3 widescreen (3 panels horizontally, read left to right).
Preset: show_trajectory_inset=False (trajectory inset HUD removed).
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# NumPy compatibility
import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

from studies.behavior.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from studies.behavior.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_verified_metadata,
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/movies_1x3_final_exemplars"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_thumbnail(video_path, thumb_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_f = max(0, min(total_frames - 1, int(total_frames * 0.4)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(thumb_path), frame)
    cap.release()


def extract_theme_candidates(results, vfi, target_seq, stage_tokens, max_dur=85):
    n_stages = len(target_seq)
    candidates = []
    for rk, data in results.items():
        if "latent_state" not in data or "centroid" not in data or rk not in vfi:
            continue
        cents = data["centroid"]
        heads = data["heading"]
        latents = data["latent_state"]
        sylls = data["syllable"]

        mask = np.concatenate([[True], sylls[1:] != sylls[:-1]])
        c_seq = sylls[mask].tolist()
        c_starts = np.where(mask)[0].tolist()
        c_ends = c_starts[1:] + [len(sylls)]

        aid, geno, drug = get_verified_metadata(rk)

        for i in range(len(c_seq) - n_stages + 1):
            if c_seq[i : i + n_stages] == target_seq:
                sf = c_starts[i]
                ef = c_ends[i + n_stages - 1]
                bounds = [c_starts[i + k] for k in range(n_stages)] + [ef]
                durs = np.diff(bounds)

                if not (all(d >= 2 for d in durs) and all(d <= max_dur for d in durs)):
                    continue
                if ef >= len(cents) or ef >= len(latents):
                    continue

                sub_cents = cents[sf:ef]
                if len(sub_cents) < 6:
                    continue
                step_d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                if np.any(step_d > 25.0):
                    continue

                t_orig = np.linspace(0, 1, ef - sf)
                t_targ = np.linspace(0, 1, 20)
                z_res = interp1d(t_orig, latents[sf:ef], axis=0)(t_targ).flatten()
                v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d * 40.0)(t_targ)
                v_norm = (v_res - 60.0) / 45.0
                feat = np.concatenate([z_res, v_norm * 2.0])

                candidates.append({
                    "recording_key": rk,
                    "start_frame": int(sf),
                    "end_frame": int(ef),
                    "stage_boundaries": [int(b) for b in bounds],
                    "stage_tokens": stage_tokens,
                    "animal_id": aid,
                    "genotype": geno,
                    "drug": drug,
                    "duration_sec": float((ef - sf) / 40.0),
                    "path_length": float(np.sum(step_d)),
                    "net_displacement": float(np.linalg.norm(sub_cents[-1] - sub_cents[0])),
                    "avg_speed": float(np.mean(step_d) * 40.0),
                    "feat": feat
                })
    return candidates


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("🎬 RENDERING 3 DISTINCT 1x3 VIDEOS FOR THEME B RIGHT-TURN", flush=True)
    print("   Theme: Turn Right -> Run/Walk Right -> Turn Right", flush=True)
    print(f"   Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]

    # 1. Triplet Candidates: [2, 0, 2]
    cands_triplet = extract_theme_candidates(
        results, vfi, target_seq=[2, 0, 2],
        stage_tokens=["Turn_R", "Run_R", "Turn_R"], max_dur=75
    )
    pca_triplet = PCA(n_components=4, random_state=42).fit([c["feat"] for c in cands_triplet])
    X_trip = pca_triplet.transform([c["feat"] for c in cands_triplet])

    # 2. Dash Candidates: [2, 0, 13, 2]
    cands_dash = extract_theme_candidates(
        results, vfi, target_seq=[2, 0, 13, 2],
        stage_tokens=["Turn_R", "Run_R", "Dash", "Turn_R"], max_dur=75
    )
    pca_dash = PCA(n_components=4, random_state=42).fit([c["feat"] for c in cands_dash])
    X_dash = pca_dash.transform([c["feat"] for c in cands_dash])

    def build_suite(cands, X, anchor_aid, exclude_aids=set(), max_dur_filter=None):
        anchor_idx = None
        for i, c in enumerate(cands):
            if c["animal_id"] == anchor_aid:
                if max_dur_filter and c["duration_sec"] > max_dur_filter:
                    continue
                anchor_idx = i
                break
        if anchor_idx is None:
            # fallback to first
            for i, c in enumerate(cands):
                if c["animal_id"] == anchor_aid:
                    anchor_idx = i
                    break

        dists = np.linalg.norm(X - X[anchor_idx], axis=1)
        sort_d = np.argsort(dists)
        suite = [cands[anchor_idx].copy()]
        suite[0]["role"] = "CANONICAL_EXEMPLAR"
        suite[0]["proto_cost"] = 0.0

        seen = {anchor_aid} | set(exclude_aids)
        for idx in sort_d:
            if idx == anchor_idx:
                continue
            c = cands[idx]
            if c["animal_id"] not in seen:
                seen.add(c["animal_id"])
                c_copy = c.copy()
                c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                c_copy["proto_cost"] = float(dists[idx])
                suite.append(c_copy)
                if len(suite) == 3:
                    break

        for p_idx, p in enumerate(suite):
            p["instance_id"] = f"M{p['animal_id']}"
            p["panel_label"] = f"Panel {p_idx+1}"
        return suite

    # Suite 1: Triplet anchored on M1749 (KI)
    suite1 = build_suite(cands_triplet, X_trip, anchor_aid="1749")
    # Suite 2: Triplet anchored on M1765 (WT, Saline) with distinct sisters
    suite2 = build_suite(cands_triplet, X_trip, anchor_aid="1765", exclude_aids={"1749"})
    # Suite 3: Dash variant anchored on M1771 (KI)
    suite3 = build_suite(cands_dash, X_dash, anchor_aid="1771")

    suites_to_render = [
        {
            "id": "exemplar_option1_m1749",
            "title": "Theme B (Option 1 - M1749 Anchor): Turn_R -> Run_R -> Turn_R",
            "instances": suite1,
            "filename": "final_1x3_theme_B_right_turn_option1_M1749.mp4"
        },
        {
            "id": "exemplar_option2_m1765",
            "title": "Theme B (Option 2 - M1765 WT Anchor): Turn_R -> Run_R -> Turn_R",
            "instances": suite2,
            "filename": "final_1x3_theme_B_right_turn_option2_M1765.mp4"
        },
        {
            "id": "exemplar_option3_m1771_dash",
            "title": "Theme B (Option 3 - M1771 Dash Anchor): Turn_R -> Run_R -> Dash -> Turn_R",
            "instances": suite3,
            "filename": "final_1x3_theme_B_right_turn_option3_M1771_dash.mp4"
        }
    ]

    rendered_videos = []
    for s in suites_to_render:
        title = s["title"]
        fn = s["filename"]
        insts = s["instances"]
        out_v = OUT_DIR / fn
        out_t = OUT_DIR / fn.replace(".mp4", ".png")

        print("\n" + "-" * 80, flush=True)
        print(f"RENDERING: {title}", flush=True)
        for p_idx, p in enumerate(insts):
            print(f"  Panel {p_idx+1}: Mouse {p['animal_id']} ({p['genotype']}, {p['drug']}) | Role: {p['role']} | Dur: {p['duration_sec']:.2f}s | Dist: {p['proto_cost']:.2f}", flush=True)

        generate_chain_grid_movie(
            instances=insts,
            output_path=out_v,
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
            show_trajectory_inset=False,  # Trajectory inset removed
            show_similarity_panel=True,
            use_gpu_encoder=True,
            suptitle_text=f"{title} [1x3 Suite]"
        )
        generate_thumbnail(out_v, out_t)
        print(f"  Saved poster frame -> {out_t.name}", flush=True)
        rendered_videos.append(out_v)

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"✅ ALL 3 RIGHT-TURN 1x3 VIDEOS GENERATED IN {total_time:.1f}s!", flush=True)
    for v in rendered_videos:
        print(f"  - {v}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
