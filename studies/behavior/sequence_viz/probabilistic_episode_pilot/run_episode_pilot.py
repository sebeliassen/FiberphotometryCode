#!/usr/bin/env python3
"""
run_episode_pilot.py
=============================================================================
Master Driver for the seqNMF-Adjacent Probabilistic Behavioral Episode Pilot.

Executes the 3-Arm Bounded Comparison:
- Arm 1: seqNMF Baseline (Clustered seqNMF L=5 retrieval)
- Arm 2: Categorical Episode Model (Syllables only, trained on train cohort)
- Arm 3: Mixed Episode Model (Syllables + Kinematics + Latents, trained on train cohort)

Exports:
- Unified master candidate tables across train and validation cohorts
- Cross-arm overlap and candidate scaling statistics
- Shared pooled references and scored candidates
=============================================================================
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric

import pandas as pd

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import load_chain_movie_resources
from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_adapter import (
    compute_theme_references,
    score_and_standardize_candidates,
)
from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import DATA_DIR, THEMES
from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.probabilistic_episode_model import (
    ProbabilisticEpisodeModel,
)


def main():
    print("=" * 80)
    print("STARTING 3-ARM PROBABILISTIC BEHAVIORAL EPISODE PILOT")
    print("=" * 80)
    t0 = time.time()

    # 1. Load data streams
    print("[1/5] Loading Level 2 bout stream dataset...")
    stream_pkl = DATA_DIR / "episode_stream_dataset.pkl"
    with open(stream_pkl, "rb") as f:
        stream_bundle = pickle.load(f)

    sessions_data = stream_bundle["sessions"]
    print(f"  Loaded {len(sessions_data)} sessions ({stream_bundle['total_bouts']:,} total Level 2 macro-bouts)")

    # Load resources for video and continuous kinematics
    print("[2/5] Loading chain movie video and trajectory resources...")
    res = load_chain_movie_resources()
    results = res["results"]

    # 2. Extract Arm 1 (seqNMF on Level 2 input with soft activation)
    print("[3/5] Extracting Arm 1 (seqNMF on Level 2 with Soft Activation diff <= 1)...")
    arm1_cands = []
    theme_keys = ["theme_A", "theme_B", "theme_C"]

    for th_id in theme_keys:
        th_meta = THEMES[th_id]
        target = np.array(th_meta["target_seq_l2"])
        L = len(target)

        for rk, sdata in sessions_data.items():
            if rk not in results:
                continue
            cents = results[rk]["centroid"]
            bouts = sdata.get("bout_l2", sdata["bout_meta"])

            # Collapse consecutive identical Level 2 synonyms into macro-bouts for seqNMF
            mask_l2 = np.concatenate([[True], bouts[1:] != bouts[:-1]])
            c_bouts = bouts[mask_l2]
            c_starts = sdata["bout_starts"][mask_l2]
            change_indices = np.where(mask_l2)[0]
            raw_ends = sdata["bout_ends"]
            c_ends = np.array([raw_ends[change_indices[m + 1] - 1] for m in range(len(change_indices) - 1)] + [raw_ends[-1]], dtype=int)

            last_end_b = -1
            for i in range(len(c_bouts) - L):
                sub = c_bouts[i : i + L]
                diff = int(np.sum(sub != target))

                # Soft activation threshold: at most 1 substitution (err <= 0.44 < 0.65)
                if diff <= 1:
                    sf = int(c_starts[i])
                    ef = int(c_ends[i + L - 1])
                    dur_s = (ef - sf) / 40.0

                    if not (0.6 <= dur_s <= 10.0):
                        continue
                    if ef >= len(cents):
                        continue

                    # Glitch exclusion: no single-frame centroid jumps > 25px
                    sub_cents = cents[sf:ef]
                    if len(sub_cents) > 1:
                        d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                        if np.any(d > 25.0):
                            continue

                    # Non-maximum suppression across nearby bouts
                    if i < last_end_b:
                        continue
                    last_end_b = i + L

                    peak_act = float((L - diff) / L)
                    template_err = float(1.0 - (peak_act ** 2))

                    arm1_cands.append({
                        "theme_id": th_id,
                        "theme_title": th_meta["title"],
                        "method": "seqNMF",
                        "recording_key": rk,
                        "animal_id": sdata["animal_id"],
                        "genotype": sdata["genotype"],
                        "drug": sdata["drug"],
                        "split": sdata["split"],
                        "start_frame": sf,
                        "end_frame": ef,
                        "duration_sec": dur_s,
                        "peak_activation": peak_act,
                        "template_error": template_err
                    })

    print(f"  Arm 1 (seqNMF) Candidates: {len(arm1_cands)} across Themes A, B, C (all 15 animals)")

    # 3. Construct Level 2 Template Priors for HSMM
    W_l2 = np.zeros((100, 3, 4), dtype=np.float64)
    for k, th_id in enumerate(theme_keys):
        targ = THEMES[th_id]["target_seq_l2"]
        for s, syll in enumerate(targ):
            W_l2[syll, k, s] = 1.0

    train_sessions = [s for s in sessions_data.values() if s["split"] == "train"]
    all_train_meta = np.concatenate([s["bout_meta"] for s in train_sessions])
    all_train_cont = np.vstack([s["continuous_features"] for s in train_sessions])
    train_bouts_df_mixed = pd.DataFrame({
        "meta_syllable": all_train_meta,
        "continuous_features": list(all_train_cont)
    })

    # Fit Arm 2 (Categorical Episode Model - Level 2 Syllables Only)
    print("\n[4/5] Training Arm 2 (Categorical Episode Model, Level 2 Syllables on Train Cohort)...")
    train_seqs_arm2 = [{"discrete_obs": s["bout_meta"]} for s in train_sessions]

    model_arm2 = ProbabilisticEpisodeModel(
        n_episodes=3, n_phases=4, n_vocab=100, n_continuous=7, mode="categorical", random_state=42
    )
    model_arm2.initialize_from_seqnmf(W_l2, train_bouts_df_mixed, [0, 1, 2])
    model_arm2.fit(train_seqs_arm2, max_iter=8, verbose=True)

    # Fit Arm 3 (Mixed Episode Model - Level 2 Syllables + Kinematics + Pose Latents)
    print("\n[5/5] Training Arm 3 (Mixed Episode Model, Syllables + Kinematics + Latents on Train Cohort)...")
    train_seqs_arm3 = [
        {"discrete_obs": s["bout_meta"], "cont_obs": s["continuous_features"]}
        for s in train_sessions
    ]

    model_arm3 = ProbabilisticEpisodeModel(
        n_episodes=3, n_phases=4, n_vocab=100, n_continuous=7, mode="mixed", random_state=42
    )
    model_arm3.initialize_from_seqnmf(W_l2, train_bouts_df_mixed, [0, 1, 2])
    model_arm3.fit(train_seqs_arm3, max_iter=8, verbose=True)

    # 5. Decode All 30 Sessions with Arm 2 & Arm 3
    print("\n[6/6] Decoding All 30 Sessions (Train + Validation Cohorts) & Scoring...")
    arm2_cands = []
    arm3_cands = []

    for rk, sdata in sessions_data.items():
        disc = sdata["bout_meta"]
        cont = sdata["continuous_features"]

        # Decode Arm 2 (Categorical)
        states_arm2 = model_arm2.viterbi_decode(disc)
        episodes_arm2 = model_arm2.extract_decoded_episodes(states_arm2, sdata)
        arm2_cands.extend(episodes_arm2)

        # Decode Arm 3 (Mixed with Option 1: Calibrated Background Competition, nudged to 7.9)
        states_arm3 = model_arm3.viterbi_decode(disc, cont, bg_competition_boost=7.9)
        episodes_arm3 = model_arm3.extract_decoded_episodes(states_arm3, sdata)
        arm3_cands.extend(episodes_arm3)

    print(f"  Decoded Raw Episodes -> Arm 2 (Categorical): {len(arm2_cands)} | Arm 3 (Mixed Option 1): {len(arm3_cands)}")

    # 6. Compute Shared Model-Independent Pooled References
    print("\n  Computing shared model-independent pooled references across training animals...")
    all_cands_pool = arm1_cands + arm2_cands + arm3_cands
    theme_refs = compute_theme_references(all_cands_pool, results)

    # Score and standardize candidates across all 3 arms
    print("  Scoring all candidates with 240D standardized continuous movement similarity...")
    scored_arm1 = score_and_standardize_candidates(arm1_cands, theme_refs, results)
    scored_arm2 = score_and_standardize_candidates(arm2_cands, theme_refs, results)
    scored_arm3 = score_and_standardize_candidates(arm3_cands, theme_refs, results)

    # Combine into master records
    all_scored = scored_arm1 + scored_arm2 + scored_arm3

    # Save models and candidates
    out_master_pkl = DATA_DIR / "pilot_master_candidates.pkl"
    with open(out_master_pkl, "wb") as f:
        pickle.dump({
            "model_arm2": model_arm2,
            "model_arm3": model_arm3,
            "theme_refs": theme_refs,
            "arm1_candidates": scored_arm1,
            "arm2_candidates": scored_arm2,
            "arm3_candidates": scored_arm3,
            "all_scored": all_scored
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Export master CSV (excluding heavy raw 240D vectors for readability)
    csv_rows = []
    for c in all_scored:
        row = {k: v for k, v in c.items() if k != "feature_240d"}
        csv_rows.append(row)
    df_master = pd.DataFrame(csv_rows)
    master_csv_path = DATA_DIR / "pilot_episodes_master.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"  [SAVED] Master candidates saved to: {out_master_pkl}")
    print(f"  [SAVED] Master CSV ({len(df_master)} rows) saved to: {master_csv_path}")

    # Print summary breakdown
    print("\n" + "=" * 80)
    print("3-ARM EPISODE DISCOVERY SUMMARY MATRIX")
    print("=" * 80)
    summary_data = []
    for th_id in THEMES:
        th_title = THEMES[th_id]["title"]
        c1 = [c for c in scored_arm1 if c["theme_id"] == th_id]
        c2 = [c for c in scored_arm2 if c["theme_id"] == th_id]
        c3 = [c for c in scored_arm3 if c["theme_id"] == th_id]

        summary_data.append({
            "Theme": f"{th_id}: {th_title}",
            "Arm 1 (seqNMF) Total": len(c1),
            "Arm 1 Train (Val)": f"{sum(1 for c in c1 if c['split']=='train')} ({sum(1 for c in c1 if c['split']=='val')})",
            "Arm 1 Animals": len(set(c['animal_id'] for c in c1)),
            "Arm 2 (Cat HSMM) Total": len(c2),
            "Arm 2 Train (Val)": f"{sum(1 for c in c2 if c['split']=='train')} ({sum(1 for c in c2 if c['split']=='val')})",
            "Arm 2 Animals": len(set(c['animal_id'] for c in c2)),
            "Arm 3 (Mixed HSMM) Total": len(c3),
            "Arm 3 Train (Val)": f"{sum(1 for c in c3 if c['split']=='train')} ({sum(1 for c in c3 if c['split']=='val')})",
            "Arm 3 Animals": len(set(c['animal_id'] for c in c3)),
            "Arm 3/1 Scaling": f"{len(c3) / max(1, len(c1)):.2f}x"
        })

    df_sum = pd.DataFrame(summary_data)
    print(df_sum.to_string(index=False))

    # Audit Level 2 themes and candidate counts
    print("\n" + "=" * 80)
    print("LEVEL 2 THEMES AUDIT & SOFT ACTIVATION CANDIDATES")
    print("=" * 80)
    for th_id, th_meta in THEMES.items():
        n1 = sum(1 for c in scored_arm1 if c["theme_id"] == th_id)
        n3 = sum(1 for c in scored_arm3 if c["theme_id"] == th_id)
        print(f"  {th_meta['title']}:")
        print(f"    Idealized: {th_meta['idealized_chain']}")
        print(f"    Arm 1 (seqNMF L2 soft): {n1} | Arm 3 (Mixed HSMM): {n3} (Ratio: {n3/max(1, n1):.2f}x)")

    print(f"\nCompleted in {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
