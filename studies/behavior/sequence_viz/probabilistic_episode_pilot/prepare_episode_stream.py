#!/usr/bin/env python3
"""
prepare_episode_stream.py
=============================================================================
Constructs aligned per-bout observations combining discrete meta-syllables
with continuous movement data (speed, circular heading change, physical
duration, displacement ratio) and training-fitted PCA-reduced pose latents.

Preserves the exact bout clock and original frame timestamps.
=============================================================================
"""

import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_verified_metadata,
)
from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import (
    DATA_DIR,
    KPMS_RESULTS_PATH,
    TRAIN_ANIMALS,
    VAL_ANIMALS,
    VFI_PATH,
)


def extract_bout_stream():
    print("=" * 80)
    print("PREPARING ALIGNED BOUT STREAM WITH MIXED CONTINUOUS OBSERVATIONS")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Level 2 Conservative Synonym mapping (93 states)
    print("[1/5] Loading Level 2 Conservative Synonym mapping (93 states)...")
    from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import LEVEL2_MAP
    syll_to_meta_arr = LEVEL2_MAP

    # 2. Load KPMS results and video frame indexes
    print("[2/5] Loading KPMS results.h5 and video_frame_indexes.npz...")
    vfi_data = np.load(VFI_PATH)
    valid_keys = set(vfi_data.keys())

    sessions_data = {}
    train_latents_for_pca = []

    with h5py.File(KPMS_RESULTS_PATH, "r") as h5:
        session_keys = sorted([k for k in h5.keys() if k in valid_keys])
        print(f"  Found {len(session_keys)} valid sessions in results.h5")

        for s_idx, rk in enumerate(session_keys):
            grp = h5[rk]
            cents = grp["centroid"][:]
            heads = grp["heading"][:]
            latents = grp["latent_state"][:]
            raw_sylls = grp["syllable"][:]

            aid, geno, drug = get_verified_metadata(rk)
            split = "train" if aid in TRAIN_ANIMALS else ("val" if aid in VAL_ANIMALS else "other")

            # 1. Collapse raw syllables into raw macro-bouts for HMM (unconstrained 100 states)
            mask = np.concatenate([[True], raw_sylls[1:] != raw_sylls[:-1]])
            b_starts = np.where(mask)[0]
            b_ends = np.append(b_starts[1:], len(raw_sylls))
            n_bouts = len(b_starts)

            # Raw syllables (0-99) for HMM's innate clustering ability
            bout_raw = raw_sylls[b_starts]
            # Level 2 conservative synonym mapping for seqNMF input
            bout_l2 = LEVEL2_MAP[bout_raw]

            # Pre-allocate bout features
            bout_meta = bout_raw  # Default discrete observation for HMM
            bout_speeds = np.zeros(n_bouts, dtype=np.float32)
            bout_dh = np.zeros(n_bouts, dtype=np.float32)
            bout_durs_sec = np.zeros(n_bouts, dtype=np.float32)
            bout_disp_ratio = np.zeros(n_bouts, dtype=np.float32)
            bout_path_len = np.zeros(n_bouts, dtype=np.float32)
            bout_net_disp = np.zeros(n_bouts, dtype=np.float32)
            bout_mean_latents = np.zeros((n_bouts, 10), dtype=np.float32)

            for b in range(n_bouts):
                sf = b_starts[b]
                ef = b_ends[b]
                dur_frames = max(1, ef - sf)
                dur_s = dur_frames / 40.0
                bout_durs_sec[b] = dur_s

                # Centroid & speed
                c_slice = cents[sf:ef]
                if len(c_slice) > 1:
                    step_d = np.linalg.norm(np.diff(c_slice, axis=0), axis=1)
                    p_len = float(np.sum(step_d))
                    n_disp = float(np.linalg.norm(c_slice[-1] - c_slice[0]))
                    speed = float(np.mean(step_d) * 40.0)
                else:
                    p_len = 0.0
                    n_disp = 0.0
                    speed = 0.0

                bout_path_len[b] = p_len
                bout_net_disp[b] = n_disp
                bout_disp_ratio[b] = n_disp / (p_len + 1.0)
                bout_speeds[b] = speed

                # Heading circular change
                h_slice = heads[sf:ef]
                if len(h_slice) > 1:
                    dh = np.diff(h_slice)
                    dh = (dh + np.pi) % (2 * np.pi) - np.pi
                    bout_dh[b] = float(np.degrees(np.sum(dh)))
                else:
                    bout_dh[b] = 0.0

                # Mean latent vector
                bout_mean_latents[b] = np.mean(latents[sf:ef], axis=0)

            # Store for training PCA if in train split
            if split == "train":
                train_latents_for_pca.append(bout_mean_latents)

            sessions_data[rk] = {
                "session_idx": s_idx,
                "recording_key": rk,
                "animal_id": aid,
                "genotype": geno,
                "drug": drug,
                "split": split,
                "n_bouts": n_bouts,
                "bout_starts": b_starts,
                "bout_ends": b_ends,
                "bout_meta": bout_meta,
                "bout_l2": bout_l2,
                "bout_durs_sec": bout_durs_sec,
                "bout_speeds": bout_speeds,
                "bout_dh": bout_dh,
                "bout_disp_ratio": bout_disp_ratio,
                "bout_path_len": bout_path_len,
                "bout_net_disp": bout_net_disp,
                "bout_mean_latents": bout_mean_latents,
                "raw_syllables": raw_sylls
            }

    # 3. Fit PCA(3) on training animal latents
    print("[3/5] Fitting 3D PCA on training animal pose latents...")
    train_latents_mat = np.vstack(train_latents_for_pca)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(train_latents_mat)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_} (Sum: {pca.explained_variance_ratio_.sum():.3f})")

    # 4. Standardize continuous features on training data
    print("[4/5] Standardizing 7D continuous feature vectors on training data...")
    # Features: [speed_px_s, signed_heading_deg, log_duration_sec, disp_ratio, pca_0, pca_1, pca_2]
    train_feat_list = []
    for rk, sdata in sessions_data.items():
        if sdata["split"] == "train":
            pca_lat = pca.transform(sdata["bout_mean_latents"])
            log_dur = np.log(sdata["bout_durs_sec"] + 1e-3)
            raw_feats = np.column_stack([
                sdata["bout_speeds"],
                sdata["bout_dh"],
                log_dur,
                sdata["bout_disp_ratio"],
                pca_lat
            ])
            train_feat_list.append(raw_feats)

    train_feats_mat = np.vstack(train_feat_list)
    scaler = StandardScaler()
    scaler.fit(train_feats_mat)

    # Transform all sessions
    total_bouts_all = 0
    for rk, sdata in sessions_data.items():
        pca_lat = pca.transform(sdata["bout_mean_latents"])
        log_dur = np.log(sdata["bout_durs_sec"] + 1e-3)
        raw_feats = np.column_stack([
            sdata["bout_speeds"],
            sdata["bout_dh"],
            log_dur,
            sdata["bout_disp_ratio"],
            pca_lat
        ])
        norm_feats = scaler.transform(raw_feats).astype(np.float32)
        sdata["continuous_features"] = norm_feats
        total_bouts_all += sdata["n_bouts"]

    # 5. Save unified dataset bundle
    out_pkl = DATA_DIR / "episode_stream_dataset.pkl"
    bundle = {
        "sessions": sessions_data,
        "pca_latents": pca,
        "feature_scaler": scaler,
        "feature_names": ["speed_px_s", "signed_heading_deg", "log_duration_sec", "disp_ratio", "pca_lat_0", "pca_lat_1", "pca_lat_2"],
        "total_sessions": len(sessions_data),
        "total_bouts": total_bouts_all,
        "train_animals": TRAIN_ANIMALS,
        "val_animals": VAL_ANIMALS
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[5/5] Saved unified stream dataset to: {out_pkl}")
    print(f"  Total sessions: {len(sessions_data)} (Train: {sum(1 for s in sessions_data.values() if s['split']=='train')}, Val: {sum(1 for s in sessions_data.values() if s['split']=='val')})")
    print(f"  Total macro-bouts: {total_bouts_all:,} across cohort")
    print(f"  Done in {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    extract_bout_stream()
