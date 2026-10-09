#!/usr/bin/env python3
"""
select_cohort_static_21kp_v2_syllable_manifest.py
==================================================
Generates the Unified Syllable Selection Manifest for Cohort Static 21kp V2
with all repairs applied from the Codex audit:
- Grid size: G = 12 (default) or G = 24.
- Canonical half-open bout intervals [start, end) without off-by-one errors.
- Native FPS-derived timing windows: 40 frames @ 60 FPS (10 pre + 30 post = 167 ms + 500 ms).
- Upstream original DLC tracking QC on dorsal landmarks and ventral paws before selection.
- Tier routing determined by eligible bout count AFTER QC:
  1. Tier 1: N_eligible > 50
     - Built-in KPMS density sampling in 4D PCA space with deterministic tie-breaking.
     - Exemplar maximizing local/global density ratio placed first (Tile 0),
       followed by its G - 1 nearest neighbors in PCA space ordered by distance.
  2. Tier 2: G < N_eligible <= 50
     - Compact-neighborhood selection: candidate set with lowest mean pairwise distance.
     - Medoid placed first (Tile 0), followed by remaining G - 1 ordered by distance.
  3. Tier 3: N_eligible <= G
     - Include all eligible bouts, medoid first.
     - Unused (G - N) slots are marked as blank.
- Diagnostics sidecars:
  - `neighbor_diagnostics.json`: 50-neighbor table, active-bout RMS, PCA variance, grid diameter.
  - `qc_threshold_sensitivity.json`: sensitivity of eligible bout counts to QC thresholds.
- Outputs written to versioned directory `visualizations_cohort_static_21kp_v2_seed101_v4`.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence, Set
import numpy as np
import h5py
import pickle
import yaml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import viz_v2_common as vc
from core.pair_matching import find_matched_pairs
from jax_moseq.utils import unbatch

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_300slds_seed101"
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"
PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"
FUSED_DIR = EXP_DIR / "data/cohort_fused_static_21kp_v2"


def main():
    parser = argparse.ArgumentParser(description="Generate Repaired Syllable Selection Manifest (V2 Cohort Static 21kp)")
    parser.add_argument("--grid_size", type=int, default=12, choices=[12, 24],
                        help="Number of tiles in the grid movie (default: 12)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Recording framerate in Hz (default: 60.0)")
    parser.add_argument("--score_pre_seconds", type=float, default=0.166667,
                        help="Seconds before onset for scoring trajectory (default: 0.166667 = 10 frames @ 60 FPS)")
    parser.add_argument("--score_post_seconds", type=float, default=0.5,
                        help="Seconds after onset for scoring trajectory (default: 0.5 = 30 frames @ 60 FPS)")
    parser.add_argument("--movie_pre_seconds", type=float, default=vc.DEFAULT_MOVIE_PRE_SECONDS)
    parser.add_argument("--movie_post_seconds", type=float, default=vc.DEFAULT_MOVIE_POST_SECONDS)
    parser.add_argument("--min_bout_seconds", type=float, default=vc.DEFAULT_MIN_BOUT_SECONDS)
    parser.add_argument("--min_frequency", type=float, default=0.0,
                        help="Minimum frequency fraction in cohort frames (default: 0.0 to cover all syllables)")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Optional specific syllables to process (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for manifest and sidecars")
    parser.add_argument("--selection_policy", type=str, default="active_reranked",
                        choices=["canonical", "active_reranked"],
                        help="Selection policy for Tier 1 candidate pool (default: active_reranked)")
    parser.add_argument("--skip_fused_qc", action="store_true", default=False,
                        help="Bypass fused HDF5 QC checks (for rapid synthetic debugging only)")
    args = parser.parse_args()

    with open(vc.CONFIG_PATH) as f:
        cohort_config = yaml.safe_load(f)
    configured_fps = float(cohort_config["fps"])
    if args.fps is not None and not np.isclose(args.fps, configured_fps):
        raise ValueError(f"Requested FPS {args.fps} differs from model/config FPS {configured_fps}")

    G = args.grid_size
    fps = configured_fps if args.fps is None else args.fps
    win = vc.make_window(fps, args.score_pre_seconds, args.score_post_seconds)
    pre, post, clip_len = win["pre_frames"], win["post_frames"], win["total_frames"]
    movie_win = vc.make_window(fps, args.movie_pre_seconds, args.movie_post_seconds)
    min_bout_frames = max(1, vc.frames_from_seconds(args.min_bout_seconds, fps))

    out_dir = Path(args.output_dir) if args.output_dir else (
        vc.DEFAULT_VIZ_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "unified_syllable_selection_manifest.json"
    diag_path = out_dir / "neighbor_diagnostics.json"
    sens_path = out_dir / "qc_threshold_sensitivity.json"

    print("=" * 80)
    print("🎯 GENERATING REPAIRED SYLLABLE SELECTION MANIFEST (V2 COHORT STATIC 21KP)")
    print(f"Grid Size:         G = {G} (Layout: {vc.grid_layout(G)})")
    print(f"Sampling Framerate: {fps:.1f} FPS")
    print(f"Timing Window:     [-{pre}, +{post}) -> {clip_len} frames ({pre/fps*1000:.1f}ms pre, {post/fps*1000:.1f}ms post)")
    print(f"Output Directory:  {out_dir}")
    print("=" * 80)

    # 1. Load VFI and pair mapping
    print("\n[1/6] Loading Video Frame Indexes (VFI) and matched pairs...", flush=True)
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # 2. Load unbatched coordinates Y
    print("[2/6] Loading and unbatching preprocessed 21kp coordinates...", flush=True)
    t0 = time.time()
    with open(PREPROCESSED_PATH, "rb") as pf:
        data, metadata = pickle.load(pf)
    coords_dict = unbatch(data["Y"], *metadata)
    del data
    print(f"  Coordinates unbatched for {len(coords_dict)} segments in {time.time() - t0:.1f}s.", flush=True)

    # 3. Cache fused confidence tracks for upstream tracking QC
    if list(cohort_config["use_bodyparts"]) != vc.BODYPARTS_21KP:
        raise ValueError("Configured keypoint order differs from visualization/QC order")
    qc_cfg = vc.QCConfig(conf_threshold=float(cohort_config.get("conf_threshold", 0.5)))
    fused_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_fused(base_sid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if base_sid in fused_cache:
            return fused_cache[base_sid]
        try:
            # Only one base session is needed while indexing its consecutive segments.
            fused_cache.clear()
            xy, conf = vc.load_fused_tracks(base_sid, FUSED_DIR)
            fused_cache[base_sid] = (xy, conf)
            return xy, conf
        except Exception as exc:
            raise RuntimeError(f"Cannot load original tracking for {base_sid}; refusing to bypass QC") from exc

    # 4. Index all qualifying bouts and extract ego-centric trajectories
    print("\n[3/6] Indexing qualifying bouts & evaluating tracking QC across results.h5...", flush=True)
    t0_idx = time.time()

    qualifying_by_syllable: Dict[int, List[Dict[str, Any]]] = {}
    eligible_by_syllable: Dict[int, List[Dict[str, Any]]] = {}
    trajectories_by_syllable: Dict[int, List[np.ndarray]] = {}
    raw_trajectories_by_syllable: Dict[int, List[np.ndarray]] = {}  # unflattened (clip_len, 21, 2)
    all_trajectories = []

    total_cohort_frames = 0
    syll_total_frames = {}
    rejection_counts: Dict[int, Dict[str, int]] = {}
    sensitivity_counts: Dict[int, Dict[str, int]] = {}
    sensitivity_cfgs = [(th, fr, vc.QCConfig(conf_threshold=th,
                         min_dorsal_valid_fraction=fr, min_paw_valid_fraction=fr))
                        for th in (0.3, 0.5, 0.7) for fr in (0.7, 0.8, 0.9)]

    with h5py.File(RESULTS_PATH, "r") as res_file:
        sids = sorted(res_file.keys())
        for sid_idx, sid in enumerate(sids):
            if sid not in coords_dict or sid not in vfi_dict:
                continue
            paths = vc.get_session_files(sid, pairs_by_sid)
            if not paths:
                continue

            grp = res_file[sid]
            sylls = grp["syllable"][:]
            centroids = grp["centroid"][:]
            headings = grp["heading"][:]
            Y_arr = coords_dict[sid]
            vfi = vfi_dict[sid]

            n_frames = len(sylls)
            if any(len(x) != n_frames for x in (centroids, headings, Y_arr, vfi)):
                raise ValueError(f"Result/coordinate/VFI length mismatch for {sid}")
            total_cohort_frames += n_frames

            u, c = np.unique(sylls, return_counts=True)
            for val, cnt in zip(u, c):
                val = int(val)
                syll_total_frames[val] = syll_total_frames.get(val, 0) + cnt

            # Circular median filtering on the complete continuous segment once
            c_filt, h_filt = vc.filter_segment_pose(centroids, headings)

            # Canonical half-open bouts [start, end)
            starts, ends = vc.half_open_bouts(sylls)

            base_sid = vc.base_session_id(sid)
            fused_data = None if args.skip_fused_qc else get_fused(base_sid)
            vc.require_fused_qc(fused_data, base_sid, skip=args.skip_fused_qc)

            for s_idx, e_idx in zip(starts, ends):
                dur = int(e_idx - s_idx)
                # Valid qualifying bout: dur >= 5, onset has enough room for timing window
                if dur >= min_bout_frames and s_idx >= pre and (s_idx + post) <= n_frames:
                    s_id = int(sylls[s_idx])
                    qualifying_by_syllable.setdefault(s_id, []).append((sid, s_idx))

                    c0 = c_filt[s_idx]
                    h0 = h_filt[s_idx]
                    Y_slice = Y_arr[s_idx - pre : s_idx + post]

                    if not np.isfinite(c0).all() or not np.isfinite(h0) or not np.isfinite(Y_arr[s_idx-pre:max(e_idx,s_idx+post)]).all():
                        rejection_counts.setdefault(s_id, {}).setdefault("nan_pose", 0)
                        rejection_counts[s_id]["nan_pose"] += 1
                        continue

                    # Upstream fused DLC tracking QC
                    is_qc_ok = True
                    qc_fail_reasons = []
                    summary_score = summary_active = None
                    context_ok = vc.movie_context_ok(s_idx, n_frames, movie_win["pre_frames"], movie_win["post_frames"])
                    if fused_data is not None:
                        xy_fused, conf_fused = fused_data
                        # Map segment frames to TopCam frames via VFI
                        score_top_frames = vfi[s_idx - pre : s_idx + post]
                        active_top_frames = vfi[s_idx:e_idx]
                        both_top_frames = np.r_[score_top_frames, active_top_frames]
                        if (both_top_frames >= 0).all() and (both_top_frames < len(conf_fused)).all():
                            valid_score = vc.valid_track_mask(
                                xy_fused[score_top_frames], conf_fused[score_top_frames], qc_cfg.conf_threshold
                            )
                            summary_score = vc.summarize_window_qc(valid_score)
                            valid_active = vc.valid_track_mask(xy_fused[active_top_frames], conf_fused[active_top_frames], qc_cfg.conf_threshold)
                            summary_active = vc.summarize_window_qc(valid_active)
                            ok_score, reasons_score = vc.qc_decision(summary_score, qc_cfg, fps)
                            ok_active, reasons_active = vc.qc_decision(summary_active, qc_cfg, fps)
                            is_qc_ok = ok_score and ok_active
                            qc_fail_reasons = reasons_score + reasons_active
                            # Actual retention counts, not a list of thresholds labelled as a measurement.
                            summaries = {}
                            for threshold in (0.3, 0.5, 0.7):
                                summaries[threshold] = (
                                    vc.summarize_window_qc(vc.valid_track_mask(xy_fused[score_top_frames], conf_fused[score_top_frames], threshold)),
                                    vc.summarize_window_qc(vc.valid_track_mask(xy_fused[active_top_frames], conf_fused[active_top_frames], threshold)))
                            counts = sensitivity_counts.setdefault(s_id, {f"conf={th:.2f},fraction={fr:.2f}":0 for th,fr,_ in sensitivity_cfgs})
                            for th, fr, cfg in sensitivity_cfgs:
                                if context_ok and all(vc.qc_decision(sm, cfg, fps)[0] for sm in summaries[th]):
                                    counts[f"conf={th:.2f},fraction={fr:.2f}"] += 1
                        else:
                            raise ValueError(f"VFI points outside original fused track rows for {sid}")

                    if not is_qc_ok:
                        reason_bucket = vc.collapse_qc_reason(qc_fail_reasons)
                        rejection_counts.setdefault(s_id, {}).setdefault(reason_bucket, 0)
                        rejection_counts[s_id]["qc_paws" if "paw" in reason_bucket else "qc_dorsal"] += 1
                        continue

                    if not context_ok:
                        rejection_counts.setdefault(s_id, {}).setdefault("movie_context", 0)
                        rejection_counts[s_id]["movie_context"] += 1
                        continue

                    # Ego-centric window (T, 21, 2)
                    ego_raw = vc.egocentric_window(Y_slice, c0, h0)
                    ego_traj = ego_raw.reshape(-1)

                    src_onset = int(vfi[s_idx]) if s_idx < len(vfi) else s_idx
                    src_offset = int(vfi[e_idx - 1]) + 1 if (e_idx - 1) < len(vfi) else e_idx

                    bout_entry = {
                        "bout_id": f"{sid}:{int(s_idx)}",
                        "sid": sid,
                        "base_sid": base_sid,
                        "onset_segment_frame": int(s_idx),
                        "offset_segment_frame": int(e_idx),
                        "duration_frames": int(dur),
                        "source_top_onset": src_onset,
                        "source_top_offset": src_offset,
                        "top_vid": paths["top_vid"],
                        "bot_vid": paths["bot_vid"],
                        "top_h5": paths["top_h5"],
                        "bot_h5": paths["bot_h5"]
                    }
                    bout_entry["qc_score"] = summary_score
                    bout_entry["qc_active"] = summary_active

                    eligible_by_syllable.setdefault(s_id, []).append(bout_entry)
                    trajectories_by_syllable.setdefault(s_id, []).append(ego_traj)
                    raw_trajectories_by_syllable.setdefault(s_id, []).append(ego_raw)
                    all_trajectories.append(ego_traj)

    del coords_dict
    print(f"  Indexed {len(all_trajectories)} QC-eligible bouts across {len(eligible_by_syllable)} syllables in {time.time() - t0_idx:.1f}s.")
    if not all_trajectories:
        raise RuntimeError("No display-eligible bouts; inspect QC/context rejections instead of fitting PCA")

    # Target syllables to process
    min_freq = args.min_frequency
    all_candidate_sylls = [
        s for s, cnt in sorted(syll_total_frames.items(), key=lambda x: x[1], reverse=True)
        if (cnt / total_cohort_frames) >= min_freq and s in eligible_by_syllable and len(eligible_by_syllable[s]) > 0
    ]
    if args.syllables is not None:
        target_sylls = [s for s in args.syllables if s in all_candidate_sylls]
    else:
        target_sylls = all_candidate_sylls
    print(f"  Target syllables to select: {len(target_sylls)} syllables.")

    # 5. Fit Global 4D PCA for Tier 1 Built-in KPMS Density Sampling
    print("\n[4/6] Fitting global 4D PCA for Tier 1 Built-in KPMS density sampling...", flush=True)
    all_trajs_arr = np.array(all_trajectories, dtype=np.float32)
    n_sample_pca = min(50000, len(all_trajs_arr))
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_trajs_arr), n_sample_pca, replace=False)
    pca = PCA(n_components=4, random_state=42).fit(all_trajs_arr[sample_indices])
    all_pca = pca.transform(all_trajs_arr[sample_indices])
    if len(all_pca) < 50 and any(len(x)>50 for x in eligible_by_syllable.values()):
        raise RuntimeError("Insufficient global background for 50-neighbor density scoring")
    all_nbrs = NearestNeighbors(n_neighbors=min(50,len(all_pca))).fit(all_pca)
    print(f"  Global PCA fitted (explained variance ratio: {pca.explained_variance_ratio_.round(4).tolist()}).")

    # 6. Execute 3-Tier Selection per Syllable
    print(f"\n[5/6] Executing 3-Tier Prototype Selection for G={G} tiles across {len(target_sylls)} syllables...", flush=True)
    manifest = {
        "schema_version": vc.SCHEMA_VERSION,
        "model_name": "model_21kp_static_50ar_300slds_seed101",
        "results_path": str(RESULTS_PATH),
        "total_iterations": 350,
        "grid_size": G,
        "fps": fps,
        "movie_window": movie_win,
        "qc_config": {**vc.asdict(qc_cfg), "applied":not args.skip_fused_qc, "intervals":["score_window","full_active_execution"]},
        "alignment": {"filter_size":vc.KPMS_FILTER_SIZE,"heading_filter":"circular_median","origin":"filtered_onset"},
        "trajectory_window": {
            "score_pre_seconds": args.score_pre_seconds,
            "score_post_seconds": args.score_post_seconds,
            "pre_frames": pre,
            "post_frames": post,
            "clip_len_frames": clip_len
        },
        "tiers_definition": {
            "tier_1": f"N_eligible > 50: Built-in KPMS density sampling (4D PCA, exemplar mode)",
            "tier_2": f"{G} < N_eligible <= 50: Compact-neighborhood selection (lowest mean pairwise distance)",
            "tier_3": f"N_eligible <= {G}: Complete bout enumeration (medoid first, remaining ordered, unused blank)"
        },
        "syllables": {}
    }

    tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
    diagnostics_by_syllable = {}

    for s in target_sylls:
        bouts = eligible_by_syllable[s]
        trajs = np.array(trajectories_by_syllable[s], dtype=np.float32)
        raw_trajs = np.array(raw_trajectories_by_syllable[s], dtype=np.float32)
        N_eligible = len(bouts)
        N_qualifying = len(qualifying_by_syllable.get(s, []))
        keys = [(b["sid"], b["onset_segment_frame"]) for b in bouts]
        stable_ranks = vc.stable_ranks(keys)

        tier_name = vc.choose_tier(N_eligible, G)

        if tier_name == "tier_1_density":
            tier_counts["tier_1"] += 1
            X_s_pca = pca.transform(trajs)
            sel_res = vc.select_tier1_density(X_s_pca, all_nbrs, stable_ranks, G=G, n_neighbors=50)
            anchor_idx = sel_res["anchor"]
            chosen_indices = sel_res["chosen"]

            # Compute neighbor diagnostics for exemplar and its 50 nearest neighbors
            nb_indices = sel_res["neighbor_idx"]
            nb_pca_dist = sel_res["neighbor_dist"]
            anchor_raw = raw_trajs[anchor_idx]

            neighbor_table = []
            for rank_nb, (nb_i, pca_d) in enumerate(zip(nb_indices, nb_pca_dist)):
                nb_bout = bouts[nb_i]
                nb_raw = raw_trajs[nb_i]
                full_d = vc.rms_point_difference(anchor_raw, nb_raw, list(range(21)))
                dorsal_d = vc.rms_point_difference(anchor_raw, nb_raw, vc.DORSAL_IDXS)
                paw_d = vc.rms_point_difference(anchor_raw, nb_raw, vc.PAW_IDXS)
                active_d, shared_frames = vc.shared_active_rms(anchor_raw, nb_raw,
                    bouts[anchor_idx]["duration_frames"], nb_bout["duration_frames"], pre, post, list(range(21)))
                active_dorsal, _ = vc.shared_active_rms(anchor_raw, nb_raw,
                    bouts[anchor_idx]["duration_frames"], nb_bout["duration_frames"], pre, post, vc.DORSAL_IDXS)
                active_paw, _ = vc.shared_active_rms(anchor_raw, nb_raw,
                    bouts[anchor_idx]["duration_frames"], nb_bout["duration_frames"], pre, post, vc.PAW_IDXS)

                neighbor_table.append({
                    "neighbor_rank": rank_nb,
                    "bout_id": nb_bout["bout_id"],
                    "sid": nb_bout["sid"],
                    "duration_frames": nb_bout["duration_frames"],
                    "pca_distance": float(pca_d),
                    "full_21kp_rms_px": float(full_d),
                    "dorsal_rms_px": float(dorsal_d),
                    "paw_rms_px": float(paw_d),
                    "is_in_grid_12": (nb_i in chosen_indices),
                    "is_in_grid": (nb_i in chosen_indices),
                    "shared_active_frames": shared_frames,
                    "shared_active_full_rms_px": active_d,
                    "shared_active_dorsal_rms_px": active_dorsal,
                    "shared_active_paw_rms_px": active_paw,
                    "score_execution_counts":vc.score_execution_counts(nb_bout["duration_frames"],pre,post),
                    "retained_pair_difference_fraction": float(pca_d*pca_d/(full_d*full_d*clip_len*21)) if full_d>0 else None,
                    "qc_score":nb_bout["qc_score"], "qc_active":nb_bout["qc_active"],
                })

            if args.selection_policy == "active_reranked":
                candidates = []
                for rank_nb, (nb_i, pca_d) in enumerate(zip(nb_indices, nb_pca_dist)):
                    if nb_i == anchor_idx:
                        continue
                    row = neighbor_table[rank_nb]
                    active_rms = row["shared_active_full_rms_px"] if row["shared_active_full_rms_px"] is not None else float("inf")
                    candidates.append((active_rms, int(stable_ranks[nb_i]), nb_i))
                candidates.sort()
                chosen_indices = [anchor_idx] + [c[2] for c in candidates[: G - 1]]
                chosen_set = set(chosen_indices)
                for rank_nb, row in enumerate(neighbor_table):
                    in_grid = (nb_indices[rank_nb] in chosen_set)
                    row["is_in_grid_12"] = in_grid
                    row["is_in_grid"] = in_grid

            chosen_raw = raw_trajs[chosen_indices]
            D_chosen = pairwise_distances(chosen_raw.reshape(len(chosen_indices), -1))
            compactness_radius = float(D_chosen.max())
            mean_pairwise = float(D_chosen[np.triu_indices(len(chosen_indices), k=1)].mean()) if len(chosen_indices) > 1 else 0.0

            diagnostics_by_syllable[str(s)] = {
                "syllable_id": s,
                "tier": tier_name,
                "n_qualifying": N_qualifying,
                "n_eligible": N_eligible,
                "rejection_counts": rejection_counts.get(s, {}),
                "exemplar_bout_id": bouts[anchor_idx]["bout_id"],
                "density_ratio": sel_res["density_ratio"],
                "local_density": sel_res["local_density"],
                "global_density": sel_res["global_density"],
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "grid_full_feature_diameter": compactness_radius,
                "grid_full_feature_mean_pairwise_distance": mean_pairwise,
                "grid_diameter_point_rms_px": compactness_radius / np.sqrt(clip_len*21),
                "grid_mean_pairwise_point_rms_px": mean_pairwise / np.sqrt(clip_len*21),
                "neighbors_50": neighbor_table,
            }

        elif tier_name == "tier_2_compact_neighborhood":
            tier_counts["tier_2"] += 1
            sel_res = vc.select_tier2_compact(trajs, stable_ranks, G=G)
            anchor_idx = sel_res["anchor"]
            chosen_indices = sel_res["chosen"]
            diagnostics_by_syllable[str(s)] = {
                "syllable_id": s,
                "tier": tier_name,
                "n_qualifying": N_qualifying,
                "n_eligible": N_eligible,
                "rejection_counts": rejection_counts.get(s, {}),
                "medoid_bout_id": bouts[anchor_idx]["bout_id"],
                "compactness_score": sel_res["compactness_score"],
            }

        else:
            tier_counts["tier_3"] += 1
            sel_res = vc.select_tier3_all(trajs, stable_ranks)
            anchor_idx = sel_res["anchor"]
            chosen_indices = sel_res["chosen"]
            diagnostics_by_syllable[str(s)] = {
                "syllable_id": s,
                "tier": tier_name,
                "n_qualifying": N_qualifying,
                "n_eligible": N_eligible,
                "rejection_counts": rejection_counts.get(s, {}),
                "medoid_bout_id": bouts[anchor_idx]["bout_id"],
            }

        # Assemble selected bout list with blank tile placeholders
        selected_bouts = []
        for rank, b_idx in enumerate(chosen_indices):
            bout = bouts[b_idx]
            is_anchor = (b_idx == anchor_idx)
            selected_bouts.append({
                "tile_index": rank,
                "is_blank": False,
                "is_medoid": is_anchor,
                "is_anchor": is_anchor,
                "anchor_type": "density_exemplar" if tier_name == "tier_1_density" else "medoid",
                "rank": rank,
                "bout_id": bout["bout_id"],
                "segment_name": bout["sid"],
                "onset_segment_frame": bout["onset_segment_frame"],
                "offset_segment_frame": bout["offset_segment_frame"],
                "duration_frames": bout["duration_frames"],
                "source_top_onset": bout["source_top_onset"],
                "source_top_offset": bout["source_top_offset"],
                "top_vid": bout["top_vid"],
                "bot_vid": bout["bot_vid"],
                "top_h5": bout["top_h5"],
                "bot_h5": bout["bot_h5"]
            })
            selected_bouts[-1]["qc_score"] = bout["qc_score"]
            selected_bouts[-1]["qc_active"] = bout["qc_active"]

        # Add explicit blank tiles if N_eligible < G
        for blank_rank in range(len(chosen_indices), G):
            selected_bouts.append({
                "tile_index": blank_rank,
                "is_blank": True,
                "is_medoid": False,
                "is_anchor": False,
                "anchor_type": None,
                "rank": blank_rank,
                "bout_id": None,
                "segment_name": None,
                "duration_frames": 0
            })

        # Assert no duplicate bout IDs
        vc.assert_unique_bout_ids(selected_bouts, context=f"syllable {s}")

        usage_pct = float((syll_total_frames.get(s, 0) / total_cohort_frames) * 100.0)
        manifest["syllables"][str(s)] = {
            "syllable_id": s,
            "total_cohort_frames": syll_total_frames.get(s, 0),
            "total_qualifying_bouts": N_qualifying,
            "total_cohort_bouts": N_eligible,
            "usage_pct": round(usage_pct, 4),
            "is_active_0p1pct": bool(usage_pct >= 0.1),
            "tier": tier_name,
            "selection_policy": args.selection_policy,
            "num_active_tiles": len(chosen_indices),
            "num_blank_tiles": G - len(chosen_indices),
            "bouts": selected_bouts
        }

    # Bind the selection to the actual arrays and save the PCA used for diagnostics.
    hash_cache = vc.HashCache(out_dir / ".hash_cache.json")
    manifest["input_hashes"] = vc.source_input_hashes(hash_cache)
    manifest["selection_code_sha256"] = vc.hash_files([Path(__file__).resolve(),SCRIPT_DIR/"viz_v2_common.py"])
    pca_path = out_dir / "density_pca.npz"
    with pca_path.with_name(pca_path.name+".tmp").open("wb") as pca_file:
        np.savez_compressed(pca_file, components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_,
                            score_shape=np.array([clip_len,21,2]), seed=np.array(42))
    os.replace(pca_path.with_name(pca_path.name+".tmp"),pca_path)
    manifest["density_pca_sha256"] = vc.sha256_file(pca_path)
    manifest["selection_summary"] = {"model_labels":len(syll_total_frames),"selected_labels":len(target_sylls),
        "labels_without_eligible_bouts":[int(s) for s in syll_total_frames if s not in eligible_by_syllable],
        "rejections_by_label":rejection_counts}
    # Write files atomically
    vc.write_json_atomic(manifest_path, manifest)
    vc.write_json_atomic(diag_path, diagnostics_by_syllable)

    # 6. Sensitivity Matrix
    sensitivity_matrix = {
        "conf_thresholds": [0.3, 0.5, 0.7],
        "min_valid_fractions": [0.7, 0.8, 0.9],
        "notes": "Sensitivity of qualifying bout retention across QC thresholds on dorsal and 4 paws."
    }
    sensitivity_matrix.update({"measured":not args.skip_fused_qc,
        "scope":"scorable finite-pose bouts; candidate must pass movie bounds and QC on score plus full active interval",
        "eligible_counts_by_syllable":sensitivity_counts,
        "qualifying_counts_by_syllable":{str(s):len(v) for s,v in qualifying_by_syllable.items()},
        "max_gap_seconds":0.1})
    vc.write_json_atomic(sens_path, sensitivity_matrix)

    print("\n" + "=" * 80)
    print("✅ REPAIRED SELECTION MANIFEST & DIAGNOSTICS COMPLETE!")
    print(f"Total Processed Syllables: {len(target_sylls)}")
    print(f"  Tier 1 (N > 50):       {tier_counts['tier_1']} syllables (Built-in KPMS Density)")
    print(f"  Tier 2 (G < N <= 50):  {tier_counts['tier_2']} syllables (Compact-Neighborhood Selection)")
    print(f"  Tier 3 (N <= G):       {tier_counts['tier_3']} syllables (Full Enumeration + Blanks)")
    print(f"Manifest:    {manifest_path}")
    print(f"Diagnostics: {diag_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
