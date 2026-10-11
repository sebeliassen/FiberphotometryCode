#!/usr/bin/env python3
"""
run_karen_pff_vs_monomer_master_pipeline.py
=============================================================================
Comprehensive Multi-Scale Syllable & Chain Phenotyping Suite for
Karen PFF vs. Monomer Longitudinal Dataset across 5 Waves:
  3w_PI, 8w_PI, 16w_PI, 24w_PI, L_dopa

Executes:
  Module 1: Discrete Syllable Repertoire Dynamics & Bigrams
  Module 2: 4-Tier Hierarchical Clustering (Levels 1 to 4)
  Module 3: Inter-Chain Organization (Perseveration, Hazard, Syntax)
  Module 4: Intra-Chain Execution Fidelity (Progression, Stage Hazards, DTW)
  Module 5: Intra-Syllable & Intra-Frame Micro-Kinematics (Phase tau in [0, 1])
  Module 6: Behavioral Transition Networks & Session-Time Motor Fatigue
=============================================================================
"""

import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

# Set repo root
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BASE_OUT = REPO_ROOT / "syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study"
PICKLE_PATH = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl"
LABELS_CSV = REPO_ROOT / "syllable_generation/metadata/karen_static_21kp_v2_syllable_labels.csv"

COND_ORDER = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]
GENO_ORDER = ["Monomer", "PFF"]
FPS = 60.0

# -----------------------------------------------------------------------------
# 1. SETUP MAPPINGS & CANDIDATE CHAINS
# -----------------------------------------------------------------------------
def build_hierarchical_mappings():
    """
    Constructs mappings from 100 raw syllables to Levels 1, 2, 3, 4.
    """
    # Level 1: Raw 100 states
    m1 = np.arange(100)

    # Level 2: Conservative Synonym Merging (93 states)
    m2 = np.arange(100)
    m2[4] = 3    # dash fast -> dash
    m2[14] = 3   # dash -> dash
    m2[42] = 3   # dash -> dash
    m2[45] = 3   # dash -> dash
    m2[51] = 50  # sniff -> sniff
    m2[52] = 50  # sniff -> sniff
    m2[60] = 58  # scrunch -> scrunch
    u2, inv2 = np.unique(m2, return_inverse=True)
    m2 = inv2  # 93 distinct states

    # Level 3: Meso-Scale Functional Categories (22 states)
    l3_dict = {
        "Dash_Fwd": [3, 4, 14, 42, 45],
        "Sprint": [24, 38],
        "Dash_Lat": [80, 94],
        "Stepped_Dash": [13, 19],
        "Walk_R": [10, 55, 75],
        "Walk_L": [23, 82],
        "Turn_R": [20, 34, 46, 57, 85],
        "Turn_L": [8, 71],
        "Look_R": [6, 66, 88, 90, 96],
        "Look_L": [48, 56],
        "Look_Fwd": [53],
        "Sniff_Fwd": [15, 43, 50, 51, 52],
        "Sniff_Lat": [41],
        "Stationary": [25],
        "Rear": [1],
        "Retreat": [16, 22, 54],
        "Scrunch": [58, 60],
        "Groom_Paw": [5, 9, 78],
        "Groom_Body": [28, 59, 67],
        "Rest": [37],
        "Dump_State": [12, 18, 29, 32, 49],
    }
    m3 = np.full(100, len(l3_dict), dtype=int) # unassigned is last
    l3_names = list(l3_dict.keys()) + ["Other_Unassigned"]
    for g_idx, (g_name, s_ids) in enumerate(l3_dict.items()):
        for sid in s_ids:
            if sid < 100:
                m3[sid] = g_idx

    # Level 4: Broad Functional Families (14 states)
    l4_dict = {
        "Locomotion_Dash": [3, 4, 13, 14, 19, 24, 38, 42, 45, 80, 94],
        "Locomotion_Walk": [10, 23, 55, 75, 82],
        "Steering_Turn": [8, 20, 34, 46, 57, 71, 85],
        "Orientation_Look": [6, 48, 53, 56, 66, 88, 90, 96],
        "Sniff_Exploration": [15, 41, 43, 50, 51, 52],
        "Stationary_Arrest": [25],
        "Rearing_Wall": [1],
        "Retreat_Backoff": [16, 22, 54],
        "Postural_Scrunch": [58, 60],
        "Grooming_Maintain": [5, 9, 78],
        "Scratching_Body": [28, 59, 67],
        "Quiescent_Rest": [37],
        "Dump_Outlier": [12, 18, 29, 32, 49]
    }
    m4 = np.full(100, len(l4_dict), dtype=int)
    l4_names = list(l4_dict.keys()) + ["Other_Unassigned"]
    for g_idx, (g_name, s_ids) in enumerate(l4_dict.items()):
        for sid in s_ids:
            if sid < 100:
                m4[sid] = g_idx

    return {
        "L1": {"map": m1, "n_states": 100, "names": [f"Syll_{i}" for i in range(100)]},
        "L2": {"map": m2, "n_states": len(u2), "names": [f"Syn_{i}" for i in range(len(u2))]},
        "L3": {"map": m3, "n_states": len(l3_names), "names": l3_names},
        "L4": {"map": m4, "n_states": len(l4_names), "names": l4_names}
    }


# Canonical Candidate Motifs for Karen's 21kp Model
CANDIDATE_CHAINS = [
    {"name": "K1_Rear_Settle", "tokens": [1, 37, 1], "desc": "Rear -> Rest -> Rear"},
    {"name": "K2_Pause_Scan", "tokens": [20, 48, 25], "desc": "Turn Right -> Look Left -> Stationary Arrest"},
    {"name": "K3_Vigilance_Pause", "tokens": [50, 25, 50], "desc": "Sniff -> Stationary Arrest -> Sniff"},
    {"name": "K4_Look_Scrunch", "tokens": [6, 58, 6], "desc": "Look Right -> Scrunch -> Look Right"},
    {"name": "K5_Sniff_Step_Sniff", "tokens": [50, 13, 50], "desc": "Sniff -> Short Dash -> Sniff"},
    {"name": "K6_Dash_Steer_Dash", "tokens": [42, 8, 42], "desc": "Dash -> Turn Left -> Dash"}
]


def rle_encode(arr):
    """Run-length encode a 1D array into values and lengths."""
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(arr[1:] != arr[:-1])[0] + 1
    idx = np.concatenate(([0], change_idx, [n]))
    vals = arr[idx[:-1]]
    lengths = np.diff(idx)
    return vals, lengths, idx[:-1], idx[1:]


# -----------------------------------------------------------------------------
# MAIN MASTER EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STARTING KAREN PFF VS. MONOMER LONGITUDINAL MASTER STUDY")
    print("=" * 80)

    t0_global = time.time()
    mappings = build_hierarchical_mappings()

    # Load raw session pickle
    print(f"\n[Phase 1] Loading master pickle: {PICKLE_PATH.name}...")
    t_load = time.time()
    with open(PICKLE_PATH, "rb") as f:
        all_sessions = pickle.load(f)
    print(f"Loaded {len(all_sessions)} sessions in {time.time() - t_load:.2f}s")

    # Filter to 99 longitudinal sessions
    sessions = [
        s for s in all_sessions
        if s.metadata.get("genotype") in GENO_ORDER
        and s.metadata.get("condition") in COND_ORDER
    ]
    print(f"Filtered to {len(sessions)} experimental longitudinal sessions across {len(COND_ORDER)} waves.")

    # Pre-extract session data to memory
    print("\n[Phase 1.1] Ingesting and preprocessing session streams...")
    session_data = []
    for s in sessions:
        meta = s.metadata
        aid = str(meta.get("animal_id"))
        cond = meta.get("condition")
        geno = meta.get("genotype")
        sid = s.session_id

        df = s.all_data_df
        if df is None:
            continue

        # Extract valid frames
        df_valid = df.dropna(subset=["syllable", "velocity_px_s"]).copy()
        if len(df_valid) < 1000:
            continue

        sylls = df_valid["syllable"].values.astype(int)
        vel = df_valid["velocity_px_s"].values.astype(float)
        ang = df_valid["angular_velocity"].values.astype(float) if "angular_velocity" in df_valid else np.zeros_like(vel)
        frame_idx = df_valid["frame_index"].values.astype(int)

        # Heading wrap-around fix for angular velocity
        if "heading" in df_valid.columns:
            h = df_valid["heading"].values
            dh = np.diff(h)
            dh_fixed = np.arctan2(np.sin(dh), np.cos(dh))
            ang = np.concatenate(([0.0], dh_fixed * FPS))

        session_data.append({
            "session_id": sid,
            "animal_id": aid,
            "condition": cond,
            "genotype": geno,
            "n_frames": len(df_valid),
            "syllables": sylls,
            "velocity": vel,
            "angular_velocity": ang,
            "frame_index": frame_idx
        })
    print(f"Successfully processed {len(session_data)} active sessions.")

    # -------------------------------------------------------------------------
    # MODULE 1: REPERTOIRE DYNAMICS ACROSS WAVES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 1: REPERTOIRE-LEVEL SYLLABLE DYNAMICS ACROSS WAVES")
    print("=" * 80)
    out_dir_m1 = BASE_OUT / "01_repertoire_dynamics"
    out_dir_m1.mkdir(parents=True, exist_ok=True)

    sess_summary_list = []
    syll_counts_by_sess = []

    for item in session_data:
        sylls = item["syllables"]
        vel = item["velocity"]
        n_tot = len(sylls)

        counts = Counter(sylls)
        probs = np.zeros(100)
        for s_id, cnt in counts.items():
            if 0 <= s_id < 100:
                probs[s_id] = cnt / n_tot

        # Shannon entropy and participation ratio
        p_active = probs[probs > 0]
        h_shannon = -np.sum(p_active * np.log2(p_active)) if len(p_active) > 0 else 0.0
        n_eff = 1.0 / np.sum(probs ** 2) if np.sum(probs ** 2) > 0 else 0.0

        # RLE bouts
        vals, lens, starts, ends = rle_encode(sylls)
        bout_durations = lens / FPS
        mean_bout_dur = np.mean(bout_durations) if len(bout_durations) > 0 else 0.0
        median_bout_dur = np.median(bout_durations) if len(bout_durations) > 0 else 0.0
        switch_rate = len(vals) / (n_tot / FPS / 60.0) # switches per min

        # Bigram transition entropy rate
        if len(vals) > 1:
            bg_pairs = list(zip(vals[:-1], vals[1:]))
            bg_counts = Counter(bg_pairs)
            tot_bg = len(bg_pairs)
            row_totals = defaultdict(int)
            for (s1, s2), c in bg_counts.items():
                row_totals[s1] += c
            h_trans = 0.0
            for (s1, s2), c in bg_counts.items():
                p_joint = c / tot_bg
                p_cond = c / row_totals[s1]
                h_trans -= p_joint * np.log2(p_cond)
        else:
            h_trans = 0.0

        immobility_frac = np.mean(vel < 5.0)
        mean_speed = np.mean(vel)
        median_speed = np.median(vel)
        v95_speed = np.percentile(vel, 95)

        sess_summary_list.append({
            "session_id": item["session_id"],
            "animal_id": item["animal_id"],
            "condition": item["condition"],
            "genotype": item["genotype"],
            "n_frames": n_tot,
            "mean_speed": mean_speed,
            "median_speed": median_speed,
            "v95_speed": v95_speed,
            "immobility_frac": immobility_frac,
            "mean_bout_dur_s": mean_bout_dur,
            "median_bout_dur_s": median_bout_dur,
            "switch_rate_per_min": switch_rate,
            "shannon_entropy": h_shannon,
            "participation_ratio": n_eff,
            "transition_entropy_rate": h_trans
        })

        for s_id in range(100):
            syll_counts_by_sess.append({
                "session_id": item["session_id"],
                "animal_id": item["animal_id"],
                "condition": item["condition"],
                "genotype": item["genotype"],
                "syllable_id": s_id,
                "usage_frac": probs[s_id],
                "usage_pct": probs[s_id] * 100.0
            })

    df_sess_summary = pd.DataFrame(sess_summary_list)
    df_sess_summary.to_csv(out_dir_m1 / "session_repertoire_metrics.csv", index=False)

    df_syll_usage = pd.DataFrame(syll_counts_by_sess)
    df_syll_usage.to_csv(out_dir_m1 / "per_session_syllable_usage.csv", index=False)

    # Animal-level aggregation per condition & syllable
    df_anim_syll = df_syll_usage.groupby(["animal_id", "condition", "genotype", "syllable_id"])["usage_pct"].mean().reset_index()

    # Compute stats per wave for each syllable
    syll_wave_stats = []
    for cond in COND_ORDER:
        sub_c = df_anim_syll[df_anim_syll["condition"] == cond]
        for sid in range(100):
            m_vals = sub_c[(sub_c["genotype"] == "Monomer") & (sub_c["syllable_id"] == sid)]["usage_pct"].values
            p_vals = sub_c[(sub_c["genotype"] == "PFF") & (sub_c["syllable_id"] == sid)]["usage_pct"].values

            m_mean = np.mean(m_vals) if len(m_vals) > 0 else 0.0
            p_mean = np.mean(p_vals) if len(p_vals) > 0 else 0.0
            m_sem = stats.sem(m_vals) if len(m_vals) > 1 else 0.0
            p_sem = stats.sem(p_vals) if len(p_vals) > 1 else 0.0

            delta = p_mean - m_mean
            ratio = (p_mean + 1e-4) / (m_mean + 1e-4)
            log2fc = np.log2(ratio)

            if len(m_vals) >= 2 and len(p_vals) >= 2:
                _, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                s_pool = np.sqrt(((len(m_vals)-1)*np.var(m_vals, ddof=1) + (len(p_vals)-1)*np.var(p_vals, ddof=1)) / (len(m_vals)+len(p_vals)-2))
                d = delta / (s_pool + 1e-9)
            else:
                p_t, p_u, d = 1.0, 1.0, 0.0

            syll_wave_stats.append({
                "condition": cond,
                "syllable_id": sid,
                "monomer_mean_pct": m_mean,
                "monomer_sem_pct": m_sem,
                "pff_mean_pct": p_mean,
                "pff_sem_pct": p_sem,
                "delta_pct": delta,
                "log2fc": log2fc,
                "cohens_d": d,
                "p_ttest": p_t,
                "p_mwu": p_u
            })

    df_syll_wave_stats = pd.DataFrame(syll_wave_stats)
    for cond in COND_ORDER:
        idx = df_syll_wave_stats["condition"] == cond
        pvals = df_syll_wave_stats.loc[idx, "p_mwu"].values
        order = np.argsort(pvals)
        ranked_p = pvals[order]
        n_tests = len(pvals)
        qvals = np.zeros(n_tests)
        min_q = 1.0
        for i in range(n_tests - 1, -1, -1):
            q = ranked_p[i] * n_tests / (i + 1)
            min_q = min(min_q, q)
            qvals[i] = min(1.0, min_q)
        qvals_orig = np.zeros(n_tests)
        qvals_orig[order] = qvals
        df_syll_wave_stats.loc[idx, "q_fdr"] = qvals_orig

    df_syll_wave_stats.to_csv(out_dir_m1 / "longitudinal_syllable_statistics_per_wave.csv", index=False)
    print(f"Exported Repertoire CSVs to {out_dir_m1}")

    # -------------------------------------------------------------------------
    # MODULE 2: MULTI-RESOLUTION HIERARCHICAL CLUSTERING (LEVELS 1 TO 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 2: 4-TIER HIERARCHICAL CLUSTERING (LEVELS 1 TO 4)")
    print("=" * 80)
    out_dir_m2 = BASE_OUT / "02_clustering_levels"
    out_dir_m2.mkdir(parents=True, exist_ok=True)

    clustering_results = []

    for level_key in ["L1", "L2", "L3", "L4"]:
        lvl_map = mappings[level_key]["map"]
        n_states = mappings[level_key]["n_states"]
        print(f"  Analyzing {level_key} (Vocabulary size: {n_states} states)...")

        for item in session_data:
            sylls_raw = item["syllables"]
            sylls_mapped = lvl_map[np.clip(sylls_raw, 0, 99)]

            vals, lens, starts, ends = rle_encode(sylls_mapped)

            for N in [4, 5, 6]:
                if len(vals) < N:
                    continue
                ngrams = [tuple(vals[i:i+N]) for i in range(len(vals) - N + 1)]
                durations_s = [np.sum(lens[i:i+N]) / FPS for i in range(len(vals) - N + 1)]

                clustering_results.append({
                    "session_id": item["session_id"],
                    "animal_id": item["animal_id"],
                    "condition": item["condition"],
                    "genotype": item["genotype"],
                    "level": level_key,
                    "horizon": N,
                    "n_episodes": len(ngrams),
                    "mean_duration_s": np.mean(durations_s),
                    "median_duration_s": np.median(durations_s),
                    "unique_strings": len(set(ngrams))
                })

    df_clust = pd.DataFrame(clustering_results)
    df_clust.to_csv(out_dir_m2 / "hierarchical_clustering_per_session.csv", index=False)

    cohort_level_coverage = []
    for level_key in ["L1", "L2", "L3", "L4"]:
        lvl_map = mappings[level_key]["map"]
        for cond in COND_ORDER:
            for geno in GENO_ORDER:
                sub_sess = [s for s in session_data if s["condition"] == cond and s["genotype"] == geno]
                if not sub_sess:
                    continue

                for N in [4, 5, 6]:
                    ngram_animal_map = defaultdict(set)
                    ngram_episode_count = defaultdict(int)
                    ngram_occurrences = []

                    for s in sub_sess:
                        s_mapped = lvl_map[np.clip(s["syllables"], 0, 99)]
                        vals, lens, starts, ends = rle_encode(s_mapped)
                        if len(vals) < N:
                            continue
                        for i in range(len(vals) - N + 1):
                            tok = tuple(vals[i:i+N])
                            ngram_animal_map[tok].add(s["animal_id"])
                            ngram_episode_count[tok] += 1
                            ngram_occurrences.append((s["session_id"], starts[i], ends[i+N-1], tok))

                    n_distinct_animals = len(set(s["animal_id"] for s in sub_sess))
                    req_animals = min(5, n_distinct_animals)
                    recurrent_families = set(
                        k for k, count in ngram_episode_count.items()
                        if count >= 15 and len(ngram_animal_map[k]) >= req_animals
                    )

                    sess_coverages = []
                    for s in sub_sess:
                        s_episodes = [(st, en) for sid, st, en, tok in ngram_occurrences if sid == s["session_id"] and tok in recurrent_families]
                        if not s_episodes:
                            sess_coverages.append(0.0)
                            continue
                        s_episodes.sort()
                        merged = []
                        for st, en in s_episodes:
                            if not merged or st > merged[-1][1]:
                                merged.append([st, en])
                            else:
                                merged[-1][1] = max(merged[-1][1], en)
                        covered_frames = sum(en - st for st, en in merged)
                        sess_coverages.append(covered_frames / s["n_frames"] * 100.0)

                    cohort_level_coverage.append({
                        "condition": cond,
                        "genotype": geno,
                        "level": level_key,
                        "horizon": N,
                        "distinct_observed": len(ngram_episode_count),
                        "recurrent_families": len(recurrent_families),
                        "mean_population_coverage_pct": np.mean(sess_coverages),
                        "sem_population_coverage_pct": stats.sem(sess_coverages) if len(sess_coverages) > 1 else 0.0
                    })

    df_cohort_cov = pd.DataFrame(cohort_level_coverage)
    df_cohort_cov.to_csv(out_dir_m2 / "hierarchical_clustering_cohort_coverage.csv", index=False)
    print(f"Exported Hierarchical Clustering CSVs to {out_dir_m2}")

    # -------------------------------------------------------------------------
    # MODULE 3: INTER-CHAIN ORGANIZATION ACROSS WAVES (PROJECT B)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 3: HIGHER-ORDER INTER-CHAIN ORGANIZATION ACROSS WAVES")
    print("=" * 80)
    out_dir_m3 = BASE_OUT / "03_inter_chain"
    out_dir_m3.mkdir(parents=True, exist_ok=True)

    inter_chain_summary = []

    for item in session_data:
        sylls = item["syllables"]
        vel = item["velocity"]
        vals, lens, starts, ends = rle_encode(sylls)

        chain_occurrences = []
        for c_idx, cand in enumerate(CANDIDATE_CHAINS):
            pat = cand["tokens"]
            L = len(pat)
            for i in range(len(vals) - L + 1):
                if list(vals[i:i+L]) == pat:
                    chain_occurrences.append({
                        "chain_idx": c_idx,
                        "chain_name": cand["name"],
                        "start_frame": starts[i],
                        "end_frame": ends[i+L-1],
                        "bout_start": i,
                        "bout_end": i + L
                    })

        chain_occurrences.sort(key=lambda x: (x["start_frame"], -(x["end_frame"] - x["start_frame"])))
        resolved_chains = []
        last_end = -1
        for ch in chain_occurrences:
            if ch["start_frame"] >= last_end:
                resolved_chains.append(ch)
                last_end = ch["end_frame"]

        gap_threshold_s = 5.0
        n_repeats = 0
        n_transitions = 0
        chain_repeat_counts = defaultdict(int)
        chain_out_counts = defaultdict(int)

        bout_lengths = []
        curr_run_len = 1

        for i in range(len(resolved_chains) - 1):
            c1 = resolved_chains[i]
            c2 = resolved_chains[i+1]
            gap_s = (c2["start_frame"] - c1["end_frame"]) / FPS

            if 0.0 <= gap_s <= gap_threshold_s:
                n_transitions += 1
                chain_out_counts[c1["chain_idx"]] += 1
                if c1["chain_idx"] == c2["chain_idx"]:
                    n_repeats += 1
                    chain_repeat_counts[c1["chain_idx"]] += 1
                    curr_run_len += 1
                else:
                    bout_lengths.append(curr_run_len)
                    curr_run_len = 1
            else:
                bout_lengths.append(curr_run_len)
                curr_run_len = 1
        if curr_run_len > 1:
            bout_lengths.append(curr_run_len)

        p_persev = n_repeats / n_transitions if n_transitions > 0 else 0.0

        n_ge_1 = len(resolved_chains)
        n_ge_2 = sum(1 for b in bout_lengths if b >= 2)
        n_ge_3 = sum(1 for b in bout_lengths if b >= 3)
        p_cont_1_to_2 = n_ge_2 / n_ge_1 if n_ge_1 > 0 else 0.0
        p_cont_2_to_3 = n_ge_3 / n_ge_2 if n_ge_2 > 0 else 0.0

        inter_chain_summary.append({
            "session_id": item["session_id"],
            "animal_id": item["animal_id"],
            "condition": item["condition"],
            "genotype": item["genotype"],
            "total_chains_detected": len(resolved_chains),
            "chain_transitions": n_transitions,
            "chain_repeats": n_repeats,
            "p_perseveration": p_persev,
            "p_continue_1_to_2": p_cont_1_to_2,
            "p_continue_2_to_3": p_cont_2_to_3,
            "max_repeat_bout": max(bout_lengths) if bout_lengths else 1
        })

    df_inter = pd.DataFrame(inter_chain_summary)
    df_inter.to_csv(out_dir_m3 / "inter_chain_organization_per_session.csv", index=False)

    inter_wave_stats = []
    for cond in COND_ORDER:
        sub_c = df_inter[df_inter["condition"] == cond]
        for metric in ["p_perseveration", "p_continue_1_to_2", "p_continue_2_to_3", "total_chains_detected"]:
            m_vals = sub_c[sub_c["genotype"] == "Monomer"][metric].values
            p_vals = sub_c[sub_c["genotype"] == "PFF"][metric].values
            if len(m_vals) >= 2 and len(p_vals) >= 2:
                _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                _, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
            else:
                p_u, p_t = 1.0, 1.0
            inter_wave_stats.append({
                "condition": cond,
                "metric": metric,
                "monomer_mean": np.mean(m_vals) if len(m_vals) else 0.0,
                "monomer_sem": stats.sem(m_vals) if len(m_vals) > 1 else 0.0,
                "pff_mean": np.mean(p_vals) if len(p_vals) else 0.0,
                "pff_sem": stats.sem(p_vals) if len(p_vals) > 1 else 0.0,
                "delta": (np.mean(p_vals) - np.mean(m_vals)) if len(m_vals) and len(p_vals) else 0.0,
                "p_mwu": p_u,
                "p_ttest": p_t
            })
    pd.DataFrame(inter_wave_stats).to_csv(out_dir_m3 / "inter_chain_wave_statistics.csv", index=False)
    print(f"Exported Inter-Chain CSVs to {out_dir_m3}")

    # -------------------------------------------------------------------------
    # MODULE 4: INTRA-CHAIN EXECUTION WITHIN ROUTINES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 4: HIGHER-ORDER INTRA-CHAIN EXECUTION FIDELITY")
    print("=" * 80)
    out_dir_m4 = BASE_OUT / "04_intra_chain"
    out_dir_m4.mkdir(parents=True, exist_ok=True)

    intra_chain_rows = []

    for item in session_data:
        sylls = item["syllables"]
        vel = item["velocity"]
        vals, lens, starts, ends = rle_encode(sylls)
        global_mean_speed = np.mean(vel)

        for c_idx, cand in enumerate(CANDIDATE_CHAINS):
            pat = cand["tokens"]
            L = len(pat)
            prefix = pat[0]

            prefix_indices = np.where(vals == prefix)[0]
            n_prefix_attempts = len(prefix_indices)

            completed_instances = []
            aborted_at_stage_1 = 0
            aborted_at_stage_2 = 0

            for idx_p in prefix_indices:
                if idx_p + L <= len(vals):
                    sub = list(vals[idx_p:idx_p+L])
                    if sub == pat:
                        st_f = starts[idx_p]
                        en_f = ends[idx_p+L-1]
                        dur_s = (en_f - st_f) / FPS
                        mean_v = np.mean(vel[st_f:en_f]) if en_f > st_f else 0.0
                        completed_instances.append({
                            "duration_s": dur_s,
                            "mean_velocity": mean_v,
                            "vel_profile": vel[st_f:en_f]
                        })
                    elif sub[1] != pat[1]:
                        aborted_at_stage_1 += 1
                    elif L > 2 and sub[2] != pat[2]:
                        aborted_at_stage_2 += 1
                else:
                    aborted_at_stage_1 += 1

            n_complete = len(completed_instances)
            prog_rate = n_complete / n_prefix_attempts if n_prefix_attempts > 0 else 0.0
            stage1_exit_hazard = aborted_at_stage_1 / n_prefix_attempts if n_prefix_attempts > 0 else 0.0
            stage2_exit_hazard = aborted_at_stage_2 / (n_prefix_attempts - aborted_at_stage_1) if (n_prefix_attempts - aborted_at_stage_1) > 0 else 0.0

            durs = [x["duration_s"] for x in completed_instances]
            mean_dur = np.mean(durs) if durs else 0.0
            dur_cv = (np.std(durs) / mean_dur) if (mean_dur > 0 and len(durs) > 1) else 0.0

            speeds = [x["mean_velocity"] for x in completed_instances]
            mean_chain_speed = np.mean(speeds) if speeds else 0.0
            speed_mod_ratio = mean_chain_speed / global_mean_speed if global_mean_speed > 0 else 1.0

            if len(completed_instances) >= 2:
                resampled = []
                for inst in completed_instances[:20]:
                    vp = inst["vel_profile"]
                    if len(vp) >= 5:
                        x_old = np.linspace(0, 1, len(vp))
                        x_new = np.linspace(0, 1, 50)
                        f_int = interp1d(x_old, vp, kind="linear")
                        resampled.append(f_int(x_new))
                if len(resampled) >= 2:
                    corr_mat = np.corrcoef(resampled)
                    triu_idx = np.triu_indices_from(corr_mat, k=1)
                    mean_r = np.nanmean(corr_mat[triu_idx])
                else:
                    mean_r = np.nan
            else:
                mean_r = np.nan

            intra_chain_rows.append({
                "session_id": item["session_id"],
                "animal_id": item["animal_id"],
                "condition": item["condition"],
                "genotype": item["genotype"],
                "chain_name": cand["name"],
                "prefix_attempts": n_prefix_attempts,
                "completed_count": n_complete,
                "conditional_progression_rate": prog_rate,
                "stage1_exit_hazard": stage1_exit_hazard,
                "stage2_exit_hazard": stage2_exit_hazard,
                "mean_duration_s": mean_dur,
                "duration_cv": dur_cv,
                "mean_velocity": mean_chain_speed,
                "speed_modulation_ratio": speed_mod_ratio,
                "profile_correlation_r": mean_r
            })

    df_intra = pd.DataFrame(intra_chain_rows)
    df_intra.to_csv(out_dir_m4 / "intra_chain_execution_per_session.csv", index=False)

    intra_wave_stats = []
    for cond in COND_ORDER:
        sub_c = df_intra[df_intra["condition"] == cond]
        for c_cand in CANDIDATE_CHAINS:
            c_name = c_cand["name"]
            sub_chain = sub_c[sub_c["chain_name"] == c_name]
            for metric in ["conditional_progression_rate", "stage1_exit_hazard", "duration_cv", "speed_modulation_ratio"]:
                m_vals = sub_chain[sub_chain["genotype"] == "Monomer"][metric].dropna().values
                p_vals = sub_chain[sub_chain["genotype"] == "PFF"][metric].dropna().values
                if len(m_vals) >= 2 and len(p_vals) >= 2:
                    _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                    _, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                else:
                    p_u, p_t = 1.0, 1.0
                intra_wave_stats.append({
                    "condition": cond,
                    "chain_name": c_name,
                    "metric": metric,
                    "monomer_mean": np.mean(m_vals) if len(m_vals) else 0.0,
                    "pff_mean": np.mean(p_vals) if len(p_vals) else 0.0,
                    "delta": (np.mean(p_vals) - np.mean(m_vals)) if len(m_vals) and len(p_vals) else 0.0,
                    "p_mwu": p_u,
                    "p_ttest": p_t
                })
    pd.DataFrame(intra_wave_stats).to_csv(out_dir_m4 / "intra_chain_wave_statistics.csv", index=False)
    print(f"Exported Intra-Chain CSVs to {out_dir_m4}")

    # -------------------------------------------------------------------------
    # MODULE 5: INTRA-SYLLABLE & INTRA-FRAME MICRO-KINEMATICS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 5: INTRA-SYLLABLE & INTRA-FRAME MICRO-KINEMATICS (PHASE TAU)")
    print("=" * 80)
    out_dir_m5 = BASE_OUT / "05_intra_frame"
    out_dir_m5.mkdir(parents=True, exist_ok=True)

    TARGET_MICRO_SYLLS = [42, 3, 24, 8, 20, 50, 1, 58, 28]

    micro_bout_records = []
    tau_velocity_profiles = defaultdict(list)

    for item in session_data:
        sylls = item["syllables"]
        vel = item["velocity"]
        ang = item["angular_velocity"]
        vals, lens, starts, ends = rle_encode(sylls)
        sess_mean_v = np.mean(vel)

        for i, sid in enumerate(vals):
            if sid in TARGET_MICRO_SYLLS:
                st_f = starts[i]
                en_f = ends[i]
                dur_frames = en_f - st_f

                if dur_frames >= 6:
                    bout_v = vel[st_f:en_f]
                    x_raw = np.linspace(0.0, 1.0, dur_frames)
                    x_grid = np.linspace(0.0, 1.0, 50)

                    f_v = interp1d(x_raw, bout_v, kind="linear")
                    v_tau = f_v(x_grid)

                    launch_speed = v_tau[0]
                    terminal_speed = v_tau[-1]
                    peak_speed = np.max(v_tau)
                    peak_tau = x_grid[np.argmax(v_tau)]
                    braking_rate = (v_tau[-1] - peak_speed) / (1.0 - peak_tau + 1e-5)
                    norm_residual_speed = np.mean(bout_v) - sess_mean_v

                    micro_bout_records.append({
                        "session_id": item["session_id"],
                        "animal_id": item["animal_id"],
                        "condition": item["condition"],
                        "genotype": item["genotype"],
                        "syllable_id": sid,
                        "duration_frames": dur_frames,
                        "duration_s": dur_frames / FPS,
                        "mean_speed": np.mean(bout_v),
                        "norm_residual_speed": norm_residual_speed,
                        "launch_speed": launch_speed,
                        "peak_speed": peak_speed,
                        "peak_tau": peak_tau,
                        "terminal_speed": terminal_speed,
                        "braking_rate": braking_rate
                    })

                    key = (item["condition"], item["genotype"], sid)
                    if len(tau_velocity_profiles[key]) < 500:
                        tau_velocity_profiles[key].append(v_tau)

    df_micro = pd.DataFrame(micro_bout_records)
    df_micro.to_csv(out_dir_m5 / "intra_frame_micro_bouts.csv", index=False)

    profile_rows = []
    x_grid = np.linspace(0.0, 1.0, 50)
    for (cond, geno, sid), plist in tau_velocity_profiles.items():
        if plist:
            arr = np.array(plist)
            mean_curve = np.mean(arr, axis=0)
            sem_curve = stats.sem(arr, axis=0) if len(arr) > 1 else np.zeros_like(mean_curve)
            for pt_idx, tau_val in enumerate(x_grid):
                profile_rows.append({
                    "condition": cond,
                    "genotype": geno,
                    "syllable_id": sid,
                    "tau": tau_val,
                    "mean_velocity": mean_curve[pt_idx],
                    "sem_velocity": sem_curve[pt_idx]
                })
    df_prof = pd.DataFrame(profile_rows)
    df_prof.to_csv(out_dir_m5 / "micro_velocity_tau_profiles.csv", index=False)

    df_anim_micro = df_micro.groupby(["animal_id", "condition", "genotype", "syllable_id"]).mean(numeric_only=True).reset_index()
    micro_wave_stats = []
    for cond in COND_ORDER:
        sub_c = df_anim_micro[df_anim_micro["condition"] == cond]
        for sid in TARGET_MICRO_SYLLS:
            sub_s = sub_c[sub_c["syllable_id"] == sid]
            for metric in ["peak_tau", "launch_speed", "peak_speed", "norm_residual_speed"]:
                m_vals = sub_s[sub_s["genotype"] == "Monomer"][metric].dropna().values
                p_vals = sub_s[sub_s["genotype"] == "PFF"][metric].dropna().values
                if len(m_vals) >= 2 and len(p_vals) >= 2:
                    _, p_u = stats.mannwhitneyu(p_vals, m_vals)
                    _, p_t = stats.ttest_ind(p_vals, m_vals, equal_var=False)
                else:
                    p_u, p_t = 1.0, 1.0
                micro_wave_stats.append({
                    "condition": cond,
                    "syllable_id": sid,
                    "metric": metric,
                    "monomer_mean": np.mean(m_vals) if len(m_vals) else 0.0,
                    "pff_mean": np.mean(p_vals) if len(p_vals) else 0.0,
                    "delta": (np.mean(p_vals) - np.mean(m_vals)) if len(m_vals) and len(p_vals) else 0.0,
                    "p_mwu": p_u,
                    "p_ttest": p_t
                })
    pd.DataFrame(micro_wave_stats).to_csv(out_dir_m5 / "micro_kinematic_wave_statistics.csv", index=False)
    print(f"Exported Intra-Frame CSVs to {out_dir_m5}")

    # -------------------------------------------------------------------------
    # MODULE 6: BEHAVIORAL TRANSITION NETWORKS & SESSION FATIGUE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 6: BEHAVIORAL TRANSITION NETWORKS & SESSION-TIME MOTOR FATIGUE")
    print("=" * 80)
    out_dir_m6 = BASE_OUT / "06_networks_fatigue"
    out_dir_m6.mkdir(parents=True, exist_ok=True)

    TOP_N_NET = 25
    all_syll_counts = Counter()
    for s in session_data:
        vals, _, _, _ = rle_encode(s["syllables"])
        all_syll_counts.update(vals)
    top25_sylls = [s for s, _ in all_syll_counts.most_common(TOP_N_NET)]

    network_matrices = {}
    for cond in COND_ORDER:
        for geno in GENO_ORDER:
            sub = [s for s in session_data if s["condition"] == cond and s["genotype"] == geno]
            mat = np.zeros((TOP_N_NET, TOP_N_NET))
            syll_to_idx = {sid: i for i, sid in enumerate(top25_sylls)}

            for s in sub:
                vals, _, _, _ = rle_encode(s["syllables"])
                for s1, s2 in zip(vals[:-1], vals[1:]):
                    if s1 in syll_to_idx and s2 in syll_to_idx:
                        mat[syll_to_idx[s1], syll_to_idx[s2]] += 1

            row_sums = mat.sum(axis=1, keepdims=True)
            mat_norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
            network_matrices[(cond, geno)] = mat_norm

            df_mat = pd.DataFrame(mat_norm, index=top25_sylls, columns=top25_sylls)
            df_mat.to_csv(out_dir_m6 / f"network_matrix_{cond}_{geno}.csv")

    for cond in COND_ORDER:
        delta_mat = network_matrices[(cond, "PFF")] - network_matrices[(cond, "Monomer")]
        df_delta = pd.DataFrame(delta_mat, index=top25_sylls, columns=top25_sylls)
        df_delta.to_csv(out_dir_m6 / f"delta_network_{cond}_PFF_minus_Monomer.csv")

    dod_prog = (network_matrices[("16w_PI", "PFF")] - network_matrices[("3w_PI", "PFF")]) - \
               (network_matrices[("16w_PI", "Monomer")] - network_matrices[("3w_PI", "Monomer")])
    pd.DataFrame(dod_prog, index=top25_sylls, columns=top25_sylls).to_csv(out_dir_m6 / "dod_progression_16w_vs_3w.csv")

    dod_rescue = (network_matrices[("L_dopa", "PFF")] - network_matrices[("24w_PI", "PFF")]) - \
                 (network_matrices[("L_dopa", "Monomer")] - network_matrices[("24w_PI", "Monomer")])
    pd.DataFrame(dod_rescue, index=top25_sylls, columns=top25_sylls).to_csv(out_dir_m6 / "dod_rescue_Ldopa_vs_24w.csv")

    fatigue_records = []
    window_s = 120.0
    step_s = 60.0
    max_time_s = 40.0 * 60.0

    for item in session_data:
        sylls = item["syllables"]
        vel = item["velocity"]
        n_f = len(sylls)

        t_starts = np.arange(0.0, min(max_time_s, n_f / FPS - window_s), step_s)
        for t_st in t_starts:
            f_st = int(t_st * FPS)
            f_en = int((t_st + window_s) * FPS)
            if f_en <= n_f:
                w_vel = vel[f_st:f_en]
                w_sylls = sylls[f_st:f_en]

                w_speed = np.mean(w_vel)
                w_immob = np.mean(w_vel < 5.0)

                vals, _, _, _ = rle_encode(w_sylls)
                if len(vals) > 1:
                    bg = list(zip(vals[:-1], vals[1:]))
                    cnts = Counter(bg)
                    tot = len(bg)
                    row_t = defaultdict(int)
                    for (a, b), c in cnts.items(): row_t[a] += c
                    w_ent = -sum((c/tot) * np.log2(c/row_t[a]) for (a, b), c in cnts.items() if row_t[a] > 0)
                else:
                    w_ent = 0.0

                fatigue_records.append({
                    "session_id": item["session_id"],
                    "animal_id": item["animal_id"],
                    "condition": item["condition"],
                    "genotype": item["genotype"],
                    "minute_center": (t_st + window_s / 2.0) / 60.0,
                    "velocity_px_s": w_speed,
                    "immobility_frac": w_immob,
                    "transition_entropy": w_ent
                })

    df_fatigue = pd.DataFrame(fatigue_records)
    df_fatigue.to_csv(out_dir_m6 / "session_time_fatigue_curves.csv", index=False)
    print(f"Exported Networks & Fatigue CSVs to {out_dir_m6}")

    print("\n" + "=" * 80)
    print(f"MASTER DATA COMPUTATION COMPLETED IN {time.time() - t0_global:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
