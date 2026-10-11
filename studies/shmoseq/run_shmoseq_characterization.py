#!/usr/bin/env python3
"""
run_shmoseq_characterization.py
=============================================================================
Objective Characterization of Discovered shMoSeq High-Level States.

Methodology:
- Operates condition-blind on all 30 sessions using decoded MAP trajectories.
- Censoring: Boundary-touching episodes (first and last episode of each session)
  are censored from duration statistics to prevent truncation bias.
- Analyzes:
  1. Empirical episode duration distributions (median seconds, IQR).
  2. State-to-state exit transition matrix P(Z_{t+1}=j | Z_t=i, j != i).
  3. Syllable emission enrichment (fold-change over global base rate).
  4. Exports representative high-confidence timestamps across diverse animals
     for blinded ethological video review.
- Caches all computed characteristics for rapid downstream access.
=============================================================================
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "models", "selected_exploratory_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output")
CACHE_FILE = os.path.join(OUTPUT_DIR, "data", "shmoseq_state_characteristics_cache.pkl")
FIG_FILE = os.path.join(OUTPUT_DIR, "figures", "fig3_shmoseq_state_characteristics.png")
TIMESTAMPS_TABLE = os.path.join(OUTPUT_DIR, "tables", "representative_state_episodes.csv")
STATE_SUMMARY_TABLE = os.path.join(OUTPUT_DIR, "tables", "state_characterization_summary.csv")

def extract_episodes(state_seq, probs_seq=None):
    """
    Extract contiguous runs of identical states, marking boundary episodes.
    Returns list of dicts: {state, start, end, length, is_boundary, mean_prob}
    """
    episodes = []
    T = len(state_seq)
    if T == 0:
        return episodes

    cur_state = state_seq[0]
    cur_start = 0

    for t in range(1, T):
        if state_seq[t] != cur_state:
            length = t - cur_start
            is_boundary = (cur_start == 0)
            mean_prob = float(probs_seq[cur_start:t, cur_state].mean()) if probs_seq is not None else 1.0
            episodes.append({
                "state": int(cur_state),
                "start": cur_start,
                "end": t,
                "length": length,
                "is_boundary": is_boundary,
                "mean_prob": mean_prob,
            })
            cur_state = state_seq[t]
            cur_start = t

    # Final episode
    length = T - cur_start
    mean_prob = float(probs_seq[cur_start:T, cur_state].mean()) if probs_seq is not None else 1.0
    episodes.append({
        "state": int(cur_state),
        "start": cur_start,
        "end": T,
        "length": length,
        "is_boundary": True,
        "mean_prob": mean_prob,
    })
    return episodes

def main():
    print("=" * 80)
    print("RUNNING SHMOSEQ OBJECTIVE STATE CHARACTERIZATION")
    print("=" * 80)

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(TIMESTAMPS_TABLE), exist_ok=True)

    print(f"Loading model bundle from: {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)

    print(f"Loading data bundle from: {DATA_FILE}")
    with open(DATA_FILE, "rb") as f:
        data_bundle = pickle.load(f)

    states_dict = model_bundle["states_dict"]
    probs_dict = model_bundle["probs_dict"]
    metadata_dict = model_bundle["metadata_dict"]
    hypparams = model_bundle["hypparams"]
    fps = model_bundle.get("fps", 40.0125)
    N_STATES = hypparams["n_states"]
    n_syllables = hypparams["n_syllables"]

    print(f"Total sessions: {len(states_dict)}, States: {N_STATES}, Syllable vocab: {n_syllables}, FPS: {fps:.2f}")

    # -------------------------------------------------------------------------
    # 1. Episode Segmentation & Duration Analysis
    # -------------------------------------------------------------------------
    print("\nSegmenting continuous state trajectories into episodes...")
    all_episodes = []
    uncensored_durations = {k: [] for k in range(N_STATES)}
    all_durations = {k: [] for k in range(N_STATES)}

    for session_key, state_seq in states_dict.items():
        probs_seq = probs_dict.get(session_key, None)
        meta = metadata_dict.get(session_key, {})
        inj_f = meta.get("injection_frame", 0)

        episodes = extract_episodes(state_seq, probs_seq)
        for ep in episodes:
            ep["session_key"] = session_key
            ep["animal_id"] = meta.get("animal_id", "")
            ep["genotype"] = meta.get("genotype", "")
            ep["drug"] = meta.get("drug", "")
            ep["cohort"] = meta.get("cohort", "")
            ep["duration_sec"] = ep["length"] / fps
            ep["abs_start_frame"] = inj_f + ep["start"] * 2  # mapped back to raw recording clock
            ep["abs_end_frame"] = inj_f + ep["end"] * 2

            all_episodes.append(ep)
            all_durations[ep["state"]].append(ep["duration_sec"])
            if not ep["is_boundary"]:
                uncensored_durations[ep["state"]].append(ep["duration_sec"])

    df_episodes = pd.DataFrame(all_episodes)
    print(f"Extracted {len(df_episodes)} total episodes across 30 sessions.")
    print(f"Censored {df_episodes['is_boundary'].sum()} boundary episodes to prevent truncation bias.")

    # -------------------------------------------------------------------------
    # 2. Exit Transition Matrix Between Macro-States
    # -------------------------------------------------------------------------
    print("\nComputing state-to-state exit transition routing matrix...")
    exit_counts = np.zeros((N_STATES, N_STATES), dtype=np.float64)

    for session_key, state_seq in states_dict.items():
        episodes = extract_episodes(state_seq)
        for i in range(len(episodes) - 1):
            src = episodes[i]["state"]
            dst = episodes[i+1]["state"]
            if src != dst:
                exit_counts[src, dst] += 1.0

    row_sums = exit_counts.sum(axis=1, keepdims=True)
    exit_routing_matrix = np.divide(exit_counts, row_sums, out=np.zeros_like(exit_counts), where=row_sums > 0)

    # -------------------------------------------------------------------------
    # 3. Syllable Emission Enrichment
    # -------------------------------------------------------------------------
    print("\nComputing syllable emissions and fold-enrichment across states...")
    syllables_dict = data_bundle["syllables_dict"]

    # Global syllable distribution
    all_sylls_flat = np.concatenate(list(syllables_dict.values()))
    global_syll_counts = np.bincount(all_sylls_flat, minlength=n_syllables).astype(np.float64)
    global_syll_prob = (global_syll_counts + 1.0) / (global_syll_counts.sum() + n_syllables)

    # State-specific syllable distributions
    state_syll_counts = np.zeros((N_STATES, n_syllables), dtype=np.float64)
    for session_key, state_seq in states_dict.items():
        syll_seq = syllables_dict[session_key]
        min_len = min(len(state_seq), len(syll_seq))
        for t in range(min_len):
            st = state_seq[t]
            syl = syll_seq[t]
            state_syll_counts[st, syl] += 1.0

    state_syll_probs = np.zeros((N_STATES, n_syllables), dtype=np.float64)
    state_syll_enrichment = np.zeros((N_STATES, n_syllables), dtype=np.float64)
    top_syllables_per_state = {}

    for k in range(N_STATES):
        row = state_syll_counts[k]
        total = row.sum()
        if total > 0:
            p_k = (row + 1.0) / (total + n_syllables)
            state_syll_probs[k] = p_k
            # Log2 fold enrichment over global baseline
            enrichment = np.log2(p_k / global_syll_prob)
            state_syll_enrichment[k] = enrichment
            # Top 5 most enriched syllables
            top_idx = np.argsort(enrichment)[::-1][:5]
            top_syllables_per_state[k] = [(int(idx), float(enrichment[idx]), float(p_k[idx]*100)) for idx in top_idx]
        else:
            top_syllables_per_state[k] = []

    # -------------------------------------------------------------------------
    # 4. Summary Metrics Table
    # -------------------------------------------------------------------------
    summary_rows = []
    total_frames = sum(len(s) for s in states_dict.values())

    print("\nHigh-Level State Summary:")
    print("-" * 85)
    print(f"{'State':<7} | {'Frames %':<9} | {'Episodes':<9} | {'Median Dur (s)':<15} | {'IQR Dur (s)':<12} | {'Top Enriched Syllables'}")
    print("-" * 85)

    for k in range(N_STATES):
        frames_k = sum((s == k).sum() for s in states_dict.values())
        prop_k = (frames_k / total_frames) * 100.0
        durs_k = uncensored_durations[k]
        n_ep = len(durs_k)
        med_dur = float(np.median(durs_k)) if n_ep > 0 else 0.0
        iqr_dur = float(np.percentile(durs_k, 75) - np.percentile(durs_k, 25)) if n_ep > 0 else 0.0
        top_s_str = ", ".join([f"Syll{s} (+{fc:.1f}b)" for s, fc, _ in top_syllables_per_state[k][:3]])

        print(f"State {k:<2} | {prop_k:6.2f}%   | {n_ep:<9} | {med_dur:6.2f}s          | {iqr_dur:6.2f}s       | {top_s_str}")

        summary_rows.append({
            "state_id": k,
            "occupancy_percent": prop_k,
            "total_frames": frames_k,
            "num_episodes_uncensored": n_ep,
            "median_duration_sec": med_dur,
            "iqr_duration_sec": iqr_dur,
            "mean_duration_sec": float(np.mean(durs_k)) if n_ep > 0 else 0.0,
            "top_enriched_syllables": top_s_str,
        })
    print("-" * 85)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(STATE_SUMMARY_TABLE, index=False)
    print(f"Saved state summary table to: {STATE_SUMMARY_TABLE}")

    # -------------------------------------------------------------------------
    # 5. Export Representative Timestamps for Blinded Video Review
    # -------------------------------------------------------------------------
    print("\nSelecting representative, high-confidence episodes across diverse animals...")
    rep_episodes = []

    for k in range(N_STATES):
        # Pick uncensored episodes of state k with duration > median and high posterior confidence
        k_eps = [ep for ep in all_episodes if ep["state"] == k and not ep["is_boundary"] and ep["length"] >= 20]
        if not k_eps:
            continue
        # Sort by posterior probability and duration balance
        k_eps_sorted = sorted(k_eps, key=lambda x: (x["mean_prob"] * np.log1p(x["duration_sec"])), reverse=True)
        # Select top 5 episodes ensuring animal diversity
        seen_animals = set()
        chosen = 0
        for ep in k_eps_sorted:
            aid = ep["animal_id"]
            if aid not in seen_animals or len(seen_animals) >= 4:
                seen_animals.add(aid)
                rep_episodes.append({
                    "state_id": k,
                    "session_key": ep["session_key"],
                    "animal_id": ep["animal_id"],
                    "genotype": ep["genotype"],
                    "drug": ep["drug"],
                    "duration_sec": round(ep["duration_sec"], 2),
                    "mean_posterior_prob": round(ep["mean_prob"], 3),
                    "start_sec_in_session": round(ep["start"] / fps, 2),
                    "end_sec_in_session": round(ep["end"] / fps, 2),
                    "raw_start_frame": ep["abs_start_frame"],
                    "raw_end_frame": ep["abs_end_frame"],
                })
                chosen += 1
                if chosen >= 5:
                    break

    df_rep = pd.DataFrame(rep_episodes)
    df_rep.to_csv(TIMESTAMPS_TABLE, index=False)
    print(f"Exported {len(df_rep)} representative episodes ({len(df_rep)//N_STATES} per state) to:\n  {TIMESTAMPS_TABLE}")

    # -------------------------------------------------------------------------
    # 6. Cache Characteristics
    # -------------------------------------------------------------------------
    cache_bundle = {
        "summary_df": df_summary,
        "uncensored_durations": uncensored_durations,
        "exit_routing_matrix": exit_routing_matrix,
        "state_syll_probs": state_syll_probs,
        "state_syll_enrichment": state_syll_enrichment,
        "top_syllables_per_state": top_syllables_per_state,
        "representative_episodes": df_rep,
        "fps": fps,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_bundle, f)
    print(f"Cached all computed characteristics to: {CACHE_FILE}")

    # -------------------------------------------------------------------------
    # 7. Diagnostic Figures
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # Panel A: Episode Durations (Boxplot on log-scale)
    ax = axes[0]
    box_data = [uncensored_durations[k] for k in range(N_STATES)]
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#4c72b0", color="black"),
                    medianprops=dict(color="red", lw=2))
    ax.set_xticklabels([f"S{k}" for k in range(N_STATES)], fontweight="bold")
    ax.set_ylabel("Episode Duration (seconds)", fontweight="bold")
    ax.set_xlabel("High-Level State ID", fontweight="bold")
    ax.set_title("A. State Duration Distribution (Uncensored)", fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    # Panel B: Exit Transition Routing Matrix Heatmap
    ax = axes[1]
    im = ax.imshow(exit_routing_matrix, cmap="Blues", vmin=0.0, vmax=0.4)
    ax.set_xticks(np.arange(N_STATES))
    ax.set_yticks(np.arange(N_STATES))
    ax.set_xticklabels([f"S{k}" for k in range(N_STATES)], fontweight="bold")
    ax.set_yticklabels([f"S{k}" for k in range(N_STATES)], fontweight="bold")
    ax.set_xlabel("Destination State (Exit)", fontweight="bold")
    ax.set_ylabel("Source State", fontweight="bold")
    ax.set_title("B. High-Level Transition Routing Matrix", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Routing Probability")

    # Panel C: Top Enriched Syllables Matrix
    ax = axes[2]
    # Pick the union of all top 3 enriched syllables across states
    unique_top_sylls = sorted(list(set(s for k in range(N_STATES) for s, _, _ in top_syllables_per_state[k][:3])))
    sub_enrich = state_syll_enrichment[:, unique_top_sylls]
    im2 = ax.imshow(sub_enrich, cmap="coolwarm", vmin=-2.0, vmax=3.0, aspect="auto")
    ax.set_yticks(np.arange(N_STATES))
    ax.set_yticklabels([f"S{k}" for k in range(N_STATES)], fontweight="bold")
    ax.set_xticks(np.arange(len(unique_top_sylls)))
    ax.set_xticklabels([str(s) for s in unique_top_sylls], rotation=90, fontsize=8)
    ax.set_xlabel("Distinctive Syllable ID", fontweight="bold")
    ax.set_ylabel("High-Level State", fontweight="bold")
    ax.set_title("C. Distinctive Syllable Log2 Enrichment", fontweight="bold")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Log2 Fold-Change")

    plt.tight_layout()
    plt.savefig(FIG_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved state characterization figure to: {FIG_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
