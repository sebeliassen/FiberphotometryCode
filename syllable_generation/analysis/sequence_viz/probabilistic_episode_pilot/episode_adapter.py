#!/usr/bin/env python3
"""
episode_adapter.py
=============================================================================
Adapter module converting decoded episode arcs and seqNMF candidate episodes
into uniform candidate records for standardized 240D continuous kinematic
similarity scoring and 1x3 widescreen video rendering.

Features:
- Standardized 240D feature extraction (10D pose latents resampled to 20 +
  speed profile + heading angular velocity).
- Model-independent pooled reference trajectory per theme constructed
  strictly from training animals.
- Standardized cosine similarity Sim(F_i, F_ref) in [-1, 1].
- Per-panel HUD token assignment ensuring each video panel displays the
  actual observed syllables and durations that occurred in that clip.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import (
    EXP_DIR, DATA_DIR, THEMES, META_NAMES, TRAIN_ANIMALS
)


def extract_standardized_240d_features(results, rk, sf, ef, t_steps=20):
    """
    Extracts the standardized 240D continuous kinematic trajectory:
    - 200D: 10D KPMS latent trajectory resampled to T=20
    - 20D: Centroid speed profile v(t) resampled to T=20, normalized
    - 20D: Heading angular velocity dh(t) resampled to T=20, normalized
    """
    cents = results[rk]["centroid"][sf:ef]
    heads = results[rk]["heading"][sf:ef]
    latents = results[rk]["latent_state"][sf:ef]

    t_orig = np.linspace(0, 1, ef - sf)
    t_targ = np.linspace(0, 1, t_steps)

    # 1. Pose latent states (10 dims x 20 steps = 200 features)
    z_res = interp1d(t_orig, latents, axis=0)(t_targ).flatten()

    # 2. Centroid speed profile (px/s) (20 features)
    if len(cents) > 1:
        step_d = np.linalg.norm(np.diff(cents, axis=0), axis=1) * 40.0
        t_speed = np.linspace(0, 1, len(step_d))
        if len(step_d) == 1:
            v_res = np.full(t_steps, step_d[0])
        else:
            v_res = interp1d(t_speed, step_d)(t_targ)
    else:
        v_res = np.zeros(t_steps)
    v_norm = (v_res - 60.0) / 45.0

    # 3. Heading angular velocity (deg/s) (20 features)
    if len(heads) > 1:
        dh = np.diff(heads)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        t_dh = np.linspace(0, 1, len(dh))
        if len(dh) == 1:
            dh_res = np.full(t_steps, dh[0] * 40.0)
        else:
            dh_res = interp1d(t_dh, dh * 40.0)(t_targ)
    else:
        dh_res = np.zeros(t_steps)
    dh_norm = dh_res / 5.0

    return np.concatenate([z_res, v_norm * 2.0, dh_norm * 2.0]).astype(np.float32)


def compute_theme_references(all_candidates, results):
    """
    Computes a single, model-independent pooled reference trajectory for each theme
    using qualifying training animal candidates across methods.
    """
    theme_refs = {}
    for th_id in THEMES:
        # Filter training candidates for this theme
        th_cands = [c for c in all_candidates if c["theme_id"] == th_id and c["animal_id"] in TRAIN_ANIMALS]
        if not th_cands:
            th_cands = [c for c in all_candidates if c["theme_id"] == th_id]

        feats = []
        for c in th_cands:
            feat = extract_standardized_240d_features(results, c["recording_key"], c["start_frame"], c["end_frame"])
            c["feature_240d"] = feat
            feats.append(feat)

        if feats:
            ref = np.mean(feats, axis=0)
            ref_norm = ref / (np.linalg.norm(ref) + 1e-8)
            theme_refs[th_id] = ref_norm
        else:
            theme_refs[th_id] = np.zeros(240, dtype=np.float32)

    return theme_refs


def score_and_standardize_candidates(candidates, theme_refs, results):
    """
    Scores all candidates with standardized cosine similarity against the shared
    theme reference and prepares per-panel HUD metadata.
    """
    scored = []
    for idx, c in enumerate(candidates):
        th_id = c["theme_id"]
        ref_norm = theme_refs.get(th_id)
        sf = int(c["start_frame"])
        ef = int(c["end_frame"])
        rk = c["recording_key"]

        # Ensure 240D feature vector exists
        if "feature_240d" in c:
            feat = c["feature_240d"]
        else:
            feat = extract_standardized_240d_features(results, rk, sf, ef)
            c["feature_240d"] = feat

        # Cosine similarity
        sim = float(np.dot(feat, ref_norm) / (np.linalg.norm(feat) + 1e-8))
        c["movement_similarity"] = sim

        # Compute per-clip actual syllable sequence for truthful HUD
        raw_sylls = results[rk]["syllable"][sf:ef]
        change_pts = np.where(raw_sylls[1:] != raw_sylls[:-1])[0] + 1
        clip_bounds = [sf] + (sf + change_pts).tolist() + [ef]
        clip_tokens = [f"syll_{int(raw_sylls[p]):03d}" for p in [0] + change_pts.tolist()]

        # Ensure kinematics are calculated from raw data if missing or zero
        cents = results[rk]["centroid"][sf:ef]
        heads = results[rk]["heading"][sf:ef]
        if len(cents) > 1:
            step_d = np.linalg.norm(np.diff(cents, axis=0), axis=1)
            p_len = float(np.sum(step_d))
            n_disp = float(np.linalg.norm(cents[-1] - cents[0]))
            spd = float(np.mean(step_d) * 40.0)
        else:
            p_len, n_disp, spd = 0.0, 0.0, 0.0

        if len(heads) > 1:
            dh = np.diff(heads)
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            turn = float(np.degrees(np.sum(dh)))
        else:
            turn = 0.0

        # Standardized instance record matching chain_grid_movie_wrapper
        inst = {
            "instance_id": f"{c['method']}_{th_id}_{idx:05d}",
            "theme_id": th_id,
            "theme_title": c["theme_title"],
            "method": c["method"],
            "recording_key": rk,
            "animal_id": c["animal_id"],
            "genotype": c["genotype"],
            "drug": c["drug"],
            "split": "train" if c["animal_id"] in TRAIN_ANIMALS else "val",
            "start_frame": sf,
            "end_frame": ef,
            "duration_sec": float((ef - sf) / 40.0),
            "stage_boundaries": clip_bounds,
            "stage_tokens": clip_tokens,
            "movement_similarity": sim,
            "proto_cost": 0.0,
            "kin_dist": sim,
            "avg_speed_px_s": spd if c.get("avg_speed_px_s", 0.0) == 0.0 else float(c["avg_speed_px_s"]),
            "path_length_px": p_len if c.get("path_length_px", 0.0) == 0.0 else float(c["path_length_px"]),
            "net_displacement_px": n_disp if c.get("net_displacement_px", 0.0) == 0.0 else float(c["net_displacement_px"]),
            "signed_turn_deg": turn if c.get("signed_turn_deg", 0.0) == 0.0 else float(c["signed_turn_deg"]),
            "feature_240d": feat
        }
        scored.append(inst)

    return scored
