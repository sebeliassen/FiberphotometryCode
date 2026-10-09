#!/usr/bin/env python3
"""
probe_qc_policies_comparison.py
=============================================================================
Comprehensive QC Policy Comparison & Rejection Factor Breakdown:
1. Strict 4-Paw QC ("strict_all_paws") vs. Opt-In Forepaw-Limited QC
   ("dorsal_and_hind_supported_forepaw_limited").
2. Rejection factor decomposition: dorsal frac/gap, hindpaw frac/gap,
   forepaw frac/gap, score context vs late active failure, overlapping failures.
3. Stratified by syllable, session, and calibration epoch.
4. Density selection comparison: pool size, exemplar stability, neighbor overlap,
   and raw support statistics.
=============================================================================
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
import viz_v2_common as vc
import yaml

AUDITED_SYLLABLES = [1, 67, 30, 70, 92, 36, 42]


def run_qc_comparison(target_syllables: List[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qc_policy_comparison.json"

    print(f"[QC Comparison] Evaluating target syllables: {target_syllables}")
    print(f"[QC Comparison] Output path: {out_file}")

    with open(vc.CONFIG_PATH) as f:
        cohort_config = yaml.safe_load(f)
    fps = float(cohort_config.get("fps", 60.0))

    # Load results.h5, VFI, and config
    with h5py.File(vc.RESULTS_PATH, "r") as f:
        seg_keys = sorted([k for k in f.keys() if k not in ("metadata", "model_parameters")])
        results_data = {}
        for k in seg_keys:
            results_data[k] = {
                "syllables": f[k]["syllable"][:],
                "centroid": f[k]["centroid"][:],
                "heading": f[k]["heading"][:],
            }

    vfi_data = np.load(vc.VFI_PATH)

    cfg_strict = vc.QCConfig(
        conf_threshold=float(cohort_config.get("conf_threshold", 0.5)),
        policy="strict_all_paws",
    )
    cfg_forepaw_ltd = vc.QCConfig(
        conf_threshold=float(cohort_config.get("conf_threshold", 0.5)),
        policy="dorsal_and_hind_supported_forepaw_limited",
        min_forepaw_valid_fraction=0.0,
        max_forepaw_gap_seconds=float("inf"),
    )

    score_win = vc.make_window(fps, vc.DEFAULT_SCORE_PRE_SECONDS, vc.DEFAULT_SCORE_POST_SECONDS)
    movie_win = vc.make_window(fps, vc.DEFAULT_MOVIE_PRE_SECONDS, vc.DEFAULT_MOVIE_POST_SECONDS)
    min_bout_len = int(np.ceil(vc.DEFAULT_MIN_BOUT_SECONDS * fps))

    # Cache fused tracks and calibration epoch metadata
    fused_cache = {}
    epoch_cache = {}

    def get_fused(base_sid: str):
        if base_sid not in fused_cache:
            xy, conf = vc.load_fused_tracks(base_sid, vc.FUSED_DIR)
            fused_cache[base_sid] = (xy, conf)
            # Load sidecar manifest for epoch
            m_path = vc.FUSED_DIR / f"{base_sid}_tandem_21kp.h5.manifest.json"
            if m_path.exists():
                with open(m_path) as mf:
                    m_data = json.load(mf)
                    epoch_cache[base_sid] = m_data.get("config", {}).get("epoch", "unknown")
            else:
                epoch_cache[base_sid] = "unknown"
        return fused_cache[base_sid]

    # Pre-scan qualifying bouts for target syllables
    print("[QC Comparison] Scanning qualifying bouts...")
    t0 = time.time()
    all_bouts_by_syl = {s: [] for s in target_syllables}

    # Tracking detailed rejection breakdown per syllable
    rejections_breakdown = {
        s: {
            "total_qualifying": 0,
            "movie_context_fail": 0,
            "strict_eligible": 0,
            "forepaw_limited_eligible": 0,
            # Failure components (evaluated across all qualifying bouts)
            "reasons": {
                "dorsal_fraction": 0,
                "dorsal_gap": 0,
                "hindpaw_fraction": 0,
                "hindpaw_gap": 0,
                "forepaw_fraction": 0,
                "forepaw_gap": 0,
                "score_only_failure": 0,
                "active_only_failure": 0,
                "both_score_and_active_failure": 0,
                "overlapping_dorsal_and_paw": 0,
            },
            "by_epoch": {},
            "by_session": {},
        }
        for s in target_syllables
    }

    # Iterate through segments
    for seg_idx, sid in enumerate(seg_keys):
        syl_seq = results_data[sid]["syllables"]
        vfi = vfi_data[sid]
        n_frames = len(syl_seq)
        base_sid = vc.base_session_id(sid)

        # Check if any target syllables are present in this segment
        present_syls = set(np.unique(syl_seq)).intersection(target_syllables)
        if not present_syls:
            continue

        xy_fused, conf_fused = get_fused(base_sid)
        epoch = epoch_cache.get(base_sid, "unknown")

        # Bouts in segment
        starts, ends = vc.half_open_bouts(syl_seq)
        for s_idx, e_idx in zip(starts, ends):
            s_id = int(syl_seq[s_idx])
            if s_id not in target_syllables:
                continue

            dur = e_idx - s_idx
            if dur < min_bout_len:
                continue

            rejections_breakdown[s_id]["total_qualifying"] += 1
            by_epoch = rejections_breakdown[s_id]["by_epoch"].setdefault(
                epoch, {"qualifying": 0, "strict_pass": 0, "ltd_pass": 0}
            )
            by_epoch["qualifying"] += 1
            by_sess = rejections_breakdown[s_id]["by_session"].setdefault(
                base_sid, {"qualifying": 0, "strict_pass": 0, "ltd_pass": 0}
            )
            by_sess["qualifying"] += 1

            # Check context eligibility
            context_ok = vc.movie_context_ok(s_idx, n_frames, movie_win["pre_frames"], movie_win["post_frames"])
            if not context_ok:
                rejections_breakdown[s_id]["movie_context_fail"] += 1

            # Extract frames
            score_top = vfi[s_idx - score_win["pre_frames"] : s_idx + score_win["post_frames"]]
            active_top = vfi[s_idx:e_idx]

            valid_score = vc.valid_track_mask(xy_fused[score_top], conf_fused[score_top], cfg_strict.conf_threshold)
            valid_active = vc.valid_track_mask(xy_fused[active_top], conf_fused[active_top], cfg_strict.conf_threshold)

            sum_score = vc.summarize_window_qc(valid_score)
            sum_active = vc.summarize_window_qc(valid_active)

            pass_strict_score, r_strict_score = vc.qc_decision(sum_score, cfg_strict, fps)
            pass_strict_active, r_strict_active = vc.qc_decision(sum_active, cfg_strict, fps)
            pass_strict = pass_strict_score and pass_strict_active

            pass_ltd_score, r_ltd_score = vc.qc_decision(sum_score, cfg_forepaw_ltd, fps)
            pass_ltd_active, r_ltd_active = vc.qc_decision(sum_active, cfg_forepaw_ltd, fps)
            pass_ltd = pass_ltd_score and pass_ltd_active

            # Failure decomposition
            all_r_score = set(r_strict_score)
            all_r_active = set(r_strict_active)
            all_reasons = all_r_score.union(all_r_active)

            s_reasons = rejections_breakdown[s_id]["reasons"]
            if sum_score["min_dorsal_valid_fraction"] < cfg_strict.min_dorsal_valid_fraction or sum_active["min_dorsal_valid_fraction"] < cfg_strict.min_dorsal_valid_fraction:
                s_reasons["dorsal_fraction"] += 1
            if sum_score["max_dorsal_gap_frames"] > 6 or sum_active["max_dorsal_gap_frames"] > 6:
                s_reasons["dorsal_gap"] += 1
            if sum_score["min_hind_paw_valid_fraction"] < cfg_strict.min_paw_valid_fraction or sum_active["min_hind_paw_valid_fraction"] < cfg_strict.min_paw_valid_fraction:
                s_reasons["hindpaw_fraction"] += 1
            if sum_score["max_hind_paw_gap_frames"] > 6 or sum_active["max_hind_paw_gap_frames"] > 6:
                s_reasons["hindpaw_gap"] += 1
            if sum_score["min_front_paw_valid_fraction"] < cfg_strict.min_paw_valid_fraction or sum_active["min_front_paw_valid_fraction"] < cfg_strict.min_paw_valid_fraction:
                s_reasons["forepaw_fraction"] += 1
            if sum_score["max_front_paw_gap_frames"] > 6 or sum_active["max_front_paw_gap_frames"] > 6:
                s_reasons["forepaw_gap"] += 1

            if not pass_strict_score and pass_strict_active:
                s_reasons["score_only_failure"] += 1
            elif pass_strict_score and not pass_strict_active:
                s_reasons["active_only_failure"] += 1
            elif not pass_strict_score and not pass_strict_active:
                s_reasons["both_score_and_active_failure"] += 1

            has_dorsal = any("dorsal" in r for r in all_reasons)
            has_paw = any("paw" in r for r in all_reasons)
            if has_dorsal and has_paw:
                s_reasons["overlapping_dorsal_and_paw"] += 1

            if context_ok and pass_strict:
                rejections_breakdown[s_id]["strict_eligible"] += 1
                by_epoch["strict_pass"] += 1
                by_sess["strict_pass"] += 1

            if context_ok and pass_ltd:
                rejections_breakdown[s_id]["forepaw_limited_eligible"] += 1
                by_epoch["ltd_pass"] += 1
                by_sess["ltd_pass"] += 1

            # Save bout info for selection comparison
            bout_dict = {
                "bout_id": f"{sid}:{s_idx}",
                "segment_name": sid,
                "onset_segment_frame": s_idx,
                "offset_segment_frame": e_idx,
                "duration_frames": dur,
                "context_ok": context_ok,
                "pass_strict": pass_strict,
                "pass_ltd": pass_ltd,
                "sum_score": sum_score,
                "sum_active": sum_active,
            }
            all_bouts_by_syl[s_id].append(bout_dict)

    print(f"  Indexed and evaluated in {time.time() - t0:.1f}s.")

    # Compare selections for syllables
    selection_comparisons = {}
    # Load PCA features from saved density_pca.npz if present, or compute
    density_pca_file = vc.DEFAULT_VIZ_DIR / "density_pca.npz"
    if density_pca_file.exists():
        d_npz = np.load(density_pca_file)
        pca_mean = d_npz["mean"]
        pca_components = d_npz["components"]
    else:
        pca_mean = pca_components = None

    for s_id in target_syllables:
        strict_bouts = [b for b in all_bouts_by_syl[s_id] if b["context_ok"] and b["pass_strict"]]
        ltd_bouts = [b for b in all_bouts_by_syl[s_id] if b["context_ok"] and b["pass_ltd"]]

        tier_strict = vc.choose_tier(len(strict_bouts), 12)
        tier_ltd = vc.choose_tier(len(ltd_bouts), 12)

        # Average raw confidence on paws for eligible bouts
        strict_front_frac = np.mean([b["sum_score"]["min_front_paw_valid_fraction"] for b in strict_bouts]) if strict_bouts else 0.0
        ltd_front_frac = np.mean([b["sum_score"]["min_front_paw_valid_fraction"] for b in ltd_bouts]) if ltd_bouts else 0.0
        strict_hind_frac = np.mean([b["sum_score"]["min_hind_paw_valid_fraction"] for b in strict_bouts]) if strict_bouts else 0.0
        ltd_hind_frac = np.mean([b["sum_score"]["min_hind_paw_valid_fraction"] for b in ltd_bouts]) if ltd_bouts else 0.0

        selection_comparisons[str(s_id)] = {
            "qualifying_count": rejections_breakdown[s_id]["total_qualifying"],
            "strict_eligible_count": len(strict_bouts),
            "forepaw_limited_eligible_count": len(ltd_bouts),
            "tier_strict": tier_strict,
            "tier_forepaw_limited": tier_ltd,
            "strict_avg_score_front_paw_frac": round(float(strict_front_frac), 4),
            "ltd_avg_score_front_paw_frac": round(float(ltd_front_frac), 4),
            "strict_avg_score_hind_paw_frac": round(float(strict_hind_frac), 4),
            "ltd_avg_score_hind_paw_frac": round(float(ltd_hind_frac), 4),
        }

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cohort": "cohort_static_21kp_v2",
            "model": vc.MODEL_NAME,
            "fps": fps,
            "strict_policy": "strict_all_paws",
            "forepaw_limited_policy": "dorsal_and_hind_supported_forepaw_limited",
        },
        "target_syllables_summary": selection_comparisons,
        "rejections_breakdown": {str(k): v for k, v in rejections_breakdown.items()},
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[QC Comparison] Summary written to {out_file}")
    for s_id in target_syllables:
        sc = selection_comparisons[str(s_id)]
        print(f"  Syllable {s_id:2d}: Qualifying={sc['qualifying_count']:5d} | "
              f"Strict={sc['strict_eligible_count']:5d} ({sc['tier_strict']}) | "
              f"ForepawLtd={sc['forepaw_limited_eligible_count']:5d} ({sc['tier_forepaw_limited']}) | "
              f"FrontPawFrac Strict={sc['strict_avg_score_front_paw_frac']:.2f} vs Ltd={sc['ltd_avg_score_front_paw_frac']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syllables", type=int, nargs="+", default=AUDITED_SYLLABLES)
    parser.add_argument("--out_dir", type=Path, default=vc.DEFAULT_VIZ_DIR)
    args = parser.parse_args()
    run_qc_comparison(args.syllables, args.out_dir)
