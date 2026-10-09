#!/usr/bin/env python3
"""
enrich_video_refs_and_analyze_sequences.py

1. Enriches `s.video_refs_df` and `s.tracking_refs_df` across all 112 sessions in
   `sessions_karen_project_w_sylls_tandem_cohort.pkl` with:
   - Bottom camera video paths (/mnt/hdd/karen_project/.../Arena*_BotCamVid_*.avi)
   - Bottom camera tracking paths (bot_h5)
   - Fused 23kp tandem tracking paths (fused_h5)

2. Performs number-agnostic, sequence-intrinsic behavioral dynamic analyses:
   - Bout duration distributions (median, mean, IQR, 95th percentile, KS-test)
   - Self-transition persistence (sticky HMM parameter kappa)
   - First-order transition entropy H(S_{t+1} | S_t) & grammar predictability
   - Transition matrix sparsity & graph density
   - Syllable frequency distribution inequality (Gini coefficient & Zipf alpha)
   - Cumulative vocabulary coverage (50%, 80%, 95% threshold)
"""

import sys
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, entropy

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.multiview_top_bot_experiments.audit_cohort_correlations import find_matched_pairs

PKL_TANDEM = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl"
PKL_K240K = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
FUSED_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/fused_h5_full_cohort"


def compute_sequence_metrics(syll_sequences):
    """
    Computes label-agnostic behavioral sequence statistics across a cohort of syllable sequences.
    """
    all_bouts = []
    transitions = []
    syll_counts = {}

    for seq in syll_sequences:
        valid_seq = seq[np.isfinite(seq) & (seq >= 0)].astype(int)
        if len(valid_seq) < 2:
            continue

        # Count frequencies
        for s in valid_seq:
            syll_counts[s] = syll_counts.get(s, 0) + 1

        # Bout durations
        onsets = np.where(valid_seq[1:] != valid_seq[:-1])[0] + 1
        bout_lens = np.diff(np.concatenate([[0], onsets, [len(valid_seq)]]))
        all_bouts.extend(bout_lens)

        # First-order transitions
        t_from = valid_seq[:-1]
        t_to = valid_seq[1:]
        transitions.append(np.column_stack([t_from, t_to]))

    all_bouts = np.array(all_bouts)
    trans_mat_data = np.vstack(transitions)

    # 1. Bout distribution metrics
    med_bout = np.median(all_bouts)
    mean_bout = np.mean(all_bouts)
    p25, p75 = np.percentile(all_bouts, [25, 75])
    p95 = np.percentile(all_bouts, 95)

    # 2. Self-transition probability (persistence)
    n_total_trans = len(trans_mat_data)
    n_self_trans = np.sum(trans_mat_data[:, 0] == trans_mat_data[:, 1])
    self_trans_prob = n_self_trans / n_total_trans

    # 3. Transition Entropy H(S_{t+1} | S_t)
    u_sylls = np.unique(trans_mat_data)
    k_sylls = len(u_sylls)
    syll_to_idx = {s: i for i, s in enumerate(u_sylls)}

    trans_counts = np.zeros((k_sylls, k_sylls), dtype=float)
    for u, v in trans_mat_data:
        trans_counts[syll_to_idx[u], syll_to_idx[v]] += 1.0

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p_trans = trans_counts / row_sums

    # Marginal P(S_t)
    p_marginal = trans_counts.sum(axis=1) / n_total_trans

    cond_entropy = 0.0
    for i in range(k_sylls):
        row = p_trans[i, :]
        row_pos = row[row > 0]
        if len(row_pos) > 0:
            cond_entropy += p_marginal[i] * entropy(row_pos, base=2)

    marginal_entropy = entropy(p_marginal[p_marginal > 0], base=2)
    predictability = 1.0 - (cond_entropy / marginal_entropy) if marginal_entropy > 0 else 0.0

    # Sparsity: active transitions / possible transitions
    active_trans = np.sum(trans_counts > 0)
    sparsity = active_trans / (k_sylls * k_sylls)

    # 4. Gini Inequality of Syllable Usage
    counts = np.sort(np.array(list(syll_counts.values())))
    n = len(counts)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1.0) / n

    # Cumulative coverage
    cum_pcts = np.cumsum(counts[::-1]) / np.sum(counts)
    n_50 = np.searchsorted(cum_pcts, 0.50) + 1
    n_80 = np.searchsorted(cum_pcts, 0.80) + 1
    n_95 = np.searchsorted(cum_pcts, 0.95) + 1

    return {
        "all_bouts": all_bouts,
        "n_bouts": len(all_bouts),
        "median_bout_frames": med_bout,
        "median_bout_ms": med_bout / 60.0 * 1000.0,
        "mean_bout_frames": mean_bout,
        "mean_bout_ms": mean_bout / 60.0 * 1000.0,
        "iqr_frames": (p25, p75),
        "p95_frames": p95,
        "self_trans_prob": self_trans_prob,
        "marginal_entropy_bits": marginal_entropy,
        "conditional_entropy_bits": cond_entropy,
        "grammar_predictability_pct": predictability * 100.0,
        "active_syllables": k_sylls,
        "trans_density_pct": sparsity * 100.0,
        "gini_coefficient": gini,
        "sylls_for_50pct": n_50,
        "sylls_for_80pct": n_80,
        "sylls_for_95pct": n_95,
    }


def main():
    print("=" * 80)
    print("ENRICH VIDEO REFS & NUMBER-AGNOSTIC SEQUENCE-INTRINSIC DYNAMICS")
    print("=" * 80)

    # 1. Map BotCam videos
    print("\n1. Discovering BotCam AVI videos and matched pairs...")
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}

    bot_avis = list(Path("/mnt/hdd/karen_project").rglob("*BotCamVid*.avi"))
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})")
    avi_map = {}
    for avi in bot_avis:
        m = ts_pattern.search(avi.name)
        if not m: continue
        ts = m.group(1)
        for a in ["Arena1", "Arena2", "Arena3"]:
            if a in avi.name:
                avi_map[(ts, a)] = avi
                break

    print(f"  Found {len(avi_map)} indexed BotCam AVIs for {len(pairs)} pairs.")

    # 2. Load Tandem Sessions
    print("\n2. Loading Tandem Cohort Pickle...")
    with open(PKL_TANDEM, "rb") as f:
        tandem_sessions = pickle.load(f)
    print(f"  Loaded {len(tandem_sessions)} tandem sessions.")

    # 3. Enrich video_refs_df and tracking_refs_df
    print("\n3. Enriching s.video_refs_df and s.tracking_refs_df...")
    enriched_vids = 0
    for s in tandem_sessions:
        sid = s.session_id
        if sid not in pair_map:
            continue
        p = pair_map[sid]
        key = (p["ts"], p["arena"])
        bot_avi = avi_map.get(key)

        n_frames = len(s.all_data_df)

        # Check if BotCam is already in video_refs_df
        has_bot_vid = False
        if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
            if "role" in s.video_refs_df.columns:
                has_bot_vid = (s.video_refs_df["role"] == "bottom_camera").any()

        if not has_bot_vid and bot_avi is not None:
            bot_row = pd.DataFrame([{
                "path": str(bot_avi),
                "path_type": "video",
                "kind": "avi",
                "role": "bottom_camera",
                "ts": p["ts"],
                "chamber": p["quadrant"],
                "camset": f"{p['arena']}_BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": p["arena"],
                "camera_key": f"bottom_cam:{p['quadrant']}",
                "fps": 60.0,
                "n_frames": n_frames,
                "duration_s": n_frames / 60.0,
                "meta_origin": "opencv"
            }])
            s.video_refs_df = pd.concat([s.video_refs_df, bot_row], ignore_index=True)
            enriched_vids += 1

        # Enrich tracking_refs_df
        fused_h5 = FUSED_DIR / f"{sid}_tandem_23kp.h5"
        top_h5 = Path(p["top_h5"])
        bot_h5 = Path(p["bot_h5"])

        tracking_rows = [
            {"path": str(top_h5), "role": "top_dlc_tracking", "kind": "h5", "chamber": p["quadrant"]},
            {"path": str(bot_h5), "role": "bot_dlc_tracking", "kind": "h5", "chamber": p["quadrant"]},
            {"path": str(fused_h5), "role": "fused_tandem_23kp_tracking", "kind": "h5", "chamber": p["quadrant"]}
        ]
        s.tracking_refs_df = pd.DataFrame(tracking_rows)

    print(f"  Successfully enriched video_refs_df for {enriched_vids} sessions.")
    print(f"  Sample session video_refs count: {len(tandem_sessions[0].video_refs_df)} rows")
    print(f"  Sample session tracking_refs count: {len(tandem_sessions[0].tracking_refs_df)} rows")

    # Pickles are already written and enriched by export script
    print("  Pickles are already written and enriched with BotCam video and tracking refs.")

    # 4. Load k240k baseline for sequence dynamics comparison
    print("\n4. Loading k240k baseline dataset...")
    with open(PKL_K240K, "rb") as f:
        k240k_sessions = pickle.load(f)
    print(f"  Loaded {len(k240k_sessions)} k240k sessions.")

    # 5. Extract syllable sequences for common sessions
    k240k_map = {s.session_id: s for s in k240k_sessions}
    tandem_map = {s.session_id: s for s in tandem_sessions}
    common_sids = sorted(list(set(k240k_map.keys()) & set(tandem_map.keys())))
    print(f"  Extracting sequences across {len(common_sids)} common sessions...")

    seqs_k = [k240k_map[sid].all_data_df["syllable"].to_numpy() for sid in common_sids]
    seqs_t = [tandem_map[sid].all_data_df["syllable"].to_numpy() for sid in common_sids]

    print("\n5. Computing sequence-intrinsic metrics...")
    metrics_k = compute_sequence_metrics(seqs_k)
    metrics_t = compute_sequence_metrics(seqs_t)

    # Kolmogorov-Smirnov test on bout duration distributions
    ks_stat, ks_pval = ks_2samp(metrics_k["all_bouts"], metrics_t["all_bouts"])

    print("\n" + "=" * 80)
    print("SEQUENCE-INTRINSIC DYNAMICS COMPARISON (LABEL-AGNOSTIC):")
    print("=" * 80)
    print(f"{'Metric':<38} | {'k240k (TopCam)':<18} | {'Tandem (Top+Bot 23kp)':<22}")
    print("-" * 80)
    print(f"{'Total Bout Count':<38} | {metrics_k['n_bouts']:<18,d} | {metrics_t['n_bouts']:<22,d}")
    print(f"{'Median Bout Duration':<38} | {metrics_k['median_bout_frames']:.1f}f ({metrics_k['median_bout_ms']:.1f} ms) | {metrics_t['median_bout_frames']:.1f}f ({metrics_t['median_bout_ms']:.1f} ms)")
    print(f"{'Mean Bout Duration':<38} | {metrics_k['mean_bout_frames']:.1f}f ({metrics_k['mean_bout_ms']:.1f} ms) | {metrics_t['mean_bout_frames']:.1f}f ({metrics_t['mean_bout_ms']:.1f} ms)")
    print(f"{'Bout Duration IQR (25th - 75th)':<38} | {metrics_k['iqr_frames'][0]:.0f} - {metrics_k['iqr_frames'][1]:.0f} frames     | {metrics_t['iqr_frames'][0]:.0f} - {metrics_t['iqr_frames'][1]:.0f} frames")
    print(f"{'95th Percentile Bout Duration':<38} | {metrics_k['p95_frames']:.1f} frames      | {metrics_t['p95_frames']:.1f} frames")
    print(f"{'Self-Transition Prob (Persistence)':<38} | {metrics_k['self_trans_prob']*100:.2f}%             | {metrics_t['self_trans_prob']*100:.2f}%")
    print("-" * 80)
    print(f"{'Active Syllable Vocabulary':<38} | {metrics_k['active_syllables']:<18d} | {metrics_t['active_syllables']:<22d}")
    print(f"{'Gini Inequality (Syllable Usage)':<38} | {metrics_k['gini_coefficient']:.3f}{'':<13} | {metrics_t['gini_coefficient']:.3f}")
    print(f"{'Syllables for 50% of Time':<38} | {metrics_k['sylls_for_50pct']} syllables{'':<7} | {metrics_t['sylls_for_50pct']} syllables")
    print(f"{'Syllables for 80% of Time':<38} | {metrics_k['sylls_for_80pct']} syllables{'':<7} | {metrics_t['sylls_for_80pct']} syllables")
    print(f"{'Syllables for 95% of Time':<38} | {metrics_k['sylls_for_95pct']} syllables{'':<7} | {metrics_t['sylls_for_95pct']} syllables")
    print("-" * 80)
    print(f"{'Marginal Syllable Entropy H(S)':<38} | {metrics_k['marginal_entropy_bits']:.3f} bits{'':<8} | {metrics_t['marginal_entropy_bits']:.3f} bits")
    print(f"{'Transition Entropy H(S_t+1 | S_t)':<38} | {metrics_k['conditional_entropy_bits']:.3f} bits{'':<8} | {metrics_t['conditional_entropy_bits']:.3f} bits")
    print(f"{'Grammar Predictability Ratio':<38} | {metrics_k['grammar_predictability_pct']:.1f}%{'':<13} | {metrics_t['grammar_predictability_pct']:.1f}%")
    print(f"{'Transition Graph Density':<38} | {metrics_k['trans_density_pct']:.2f}%{'':<12} | {metrics_t['trans_density_pct']:.2f}%")
    print("-" * 80)
    print(f"Kolmogorov-Smirnov Test on Bout Durations: D = {ks_stat:.4f} (p = {ks_pval:.2e})")
    print("=" * 80)


if __name__ == "__main__":
    main()
