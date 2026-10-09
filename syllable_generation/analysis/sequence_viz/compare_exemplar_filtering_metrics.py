#!/usr/bin/env python3
"""
compare_exemplar_filtering_metrics.py

Compares biological statistical performance across 3 filtering paradigms:
1. Mode 1: Full Population (All Occurrences)
2. Mode 2: Whole-Chain Density-Ratio Exemplars (Top 10% / k-NN High-Density Core)
3. Mode 3: Forward Transition-Bucketed Exemplars (Strict Constituent Chains: min_p >= 0.50)

Evaluates:
- Contrast 1: Saline Genotype Differences (Cohen's d, exact permutation p-value)
- Contrast 2: Paired Drug Shift within Animal (Cohen's d, exact permutation p-value)
- Contrast 3: Genotype x Drug Interaction (Cohen's d, exact permutation p-value)
- Leave-One-Animal-Out (LOAO) Genotype Decoding Balanced Accuracy per method & overall
"""

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import load_chain_movie_resources
from syllable_generation.analysis.sequence_viz.run_24_candidate_transition_exemplars import (
    SYLL_TO_CLUSTER,
    TransitionBucketEvaluator,
)

DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
META_CSV = DATA_DIR / "session_metadata_ground_truth.csv"
RESULTS_PATH = REPO_ROOT / "results.h5"

def exact_2sample_permutation_test(group1, group2):
    obs_diff = float(np.mean(group1) - np.mean(group2))
    n1, n2 = len(group1), len(group2)
    combined = np.concatenate([group1, group2])
    n_total = n1 + n2
    diffs = []
    for idxs in itertools.combinations(range(n_total), n1):
        idx_set = set(idxs)
        idx2 = [i for i in range(n_total) if i not in idx_set]
        diffs.append(np.mean(combined[list(idxs)]) - np.mean(combined[idx2]))
    diffs = np.array(diffs)
    p_val = float(np.mean(np.abs(diffs) >= np.abs(obs_diff) - 1e-12))
    return obs_diff, p_val

def exact_paired_permutation_test(pairs):
    diffs = np.array([float(a - s) for s, a in pairs])
    obs_mean = float(np.mean(diffs))
    n = len(diffs)
    flips = []
    for signs in itertools.product([-1, 1], repeat=n):
        flips.append(np.mean(diffs * np.array(signs)))
    flips = np.array(flips)
    p_val = float(np.mean(np.abs(flips) >= np.abs(obs_mean) - 1e-12))
    return obs_mean, p_val

def load_candidate_definitions():
    return [
        {"id": "A1_01", "method": "Method A1 (Exact N-Grams)", "name": "Locomotor Alternation Triplet", "type": "exact", "pattern": [1, 11, 1], "tokens": "[1, 11, 1]"},
        {"id": "A1_02", "method": "Method A1 (Exact N-Grams)", "name": "Orienting Turn Triplet", "type": "exact", "pattern": [6, 17, 2], "tokens": "[6, 17, 2]"},
        {"id": "A1_03", "method": "Method A1 (Exact N-Grams)", "name": "Rearing-Descent Triplet", "type": "exact", "pattern": [14, 7, 34], "tokens": "[14, 7, 34]"},
        {"id": "A2_01", "method": "Method A2 (SEQUITUR Grammar)", "name": "Locomotor Subroutine Rule_1", "type": "exact", "pattern": [1, 11, 1, 11], "tokens": "Rule_1: [1, 11] + [1, 11]"},
        {"id": "A2_02", "method": "Method A2 (SEQUITUR Grammar)", "name": "Exploration Reorient Rule_2", "type": "exact", "pattern": [16, 4, 0], "tokens": "Rule_2: [16, 4] + [0]"},
        {"id": "A2_03", "method": "Method A2 (SEQUITUR Grammar)", "name": "Rearing Reorient Rule_3", "type": "exact", "pattern": [14, 7, 0], "tokens": "Rule_3: [14, 7] + [0]"},
        {"id": "A3_01", "method": "Method A3 (Cluster-RLE)", "name": "Locomotion-Pause-Locomotion", "type": "cluster_rle", "pattern": [1, 4, 1], "tokens": "Cluster [1 -> 4 -> 1]"},
        {"id": "A3_02", "method": "Method A3 (Cluster-RLE)", "name": "Exploratory-Orient-Exploratory", "type": "cluster_rle", "pattern": [0, 2, 0], "tokens": "Cluster [0 -> 2 -> 0]"},
        {"id": "A3_03", "method": "Method A3 (Cluster-RLE)", "name": "Rearing-Locomotion-Pause", "type": "cluster_rle", "pattern": [3, 1, 4], "tokens": "Cluster [3 -> 1 -> 4]"},
        {"id": "B_01", "method": "Method B (Weighted Alignment)", "name": "tpl_001_alt3 (Locomotion Family)", "type": "db_b", "template_id": "tpl_001_alt3", "tokens": "[1, 11, 1] (soft align)"},
        {"id": "B_02", "method": "Method B (Weighted Alignment)", "name": "tpl_003_ori3 (Orienting Family)", "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "[6, 17, 2] (soft align)"},
        {"id": "B_03", "method": "Method B (Weighted Alignment)", "name": "tpl_005_rear3 (Rearing Family)", "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "[14, 7, 34] (soft align)"},
        {"id": "C_01", "method": "Method C (Profile HMM)", "name": "Locomotor Elastic Family", "type": "exact", "pattern": [1, 11, 12], "tokens": "pHMM Family 1: [1, 11, 12]"},
        {"id": "C_02", "method": "Method C (Profile HMM)", "name": "Orienting Insertion Family", "type": "exact", "pattern": [6, 17, 21], "tokens": "pHMM Family 2: [6, 17, 21]"},
        {"id": "C_03", "method": "Method C (Profile HMM)", "name": "Rearing Exploration Family", "type": "exact", "pattern": [2, 14, 7, 34], "tokens": "pHMM Family 3: [2, 14, 7, 34]"},
        {"id": "D_01", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_05 (Locomotor Motif)", "type": "exact", "pattern": [1, 11, 1, 30], "tokens": "Factor 5: [1, 11, 1, 30]"},
        {"id": "D_02", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_02 (Orienting Factor)", "type": "exact", "pattern": [19, 6, 2], "tokens": "Factor 2: [19, 6, 2]"},
        {"id": "D_03", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_01 (Exploration Factor)", "type": "exact", "pattern": [16, 4, 0, 13], "tokens": "Factor 1: [16, 4, 0, 13]"},
        {"id": "E_01", "method": "Method E (PST Branch)", "name": "context_018 (Rearing Exit Branch)", "type": "exact", "pattern": [14, 7, 0], "tokens": "Context 018: [14, 7] -> 0"},
        {"id": "E_02", "method": "Method E (PST Branch)", "name": "context_029 (Approach Branch)", "type": "exact", "pattern": [16, 4, 17], "tokens": "Context 029: [16, 4] -> 17"},
        {"id": "E_03", "method": "Method E (PST Branch)", "name": "context_023 (Locomotion Pause Branch)", "type": "exact", "pattern": [1, 0, 21], "tokens": "Context 023: [1, 0] -> 21"},
        {"id": "DTW_01", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_alt3 (Kinematic Locomotion)", "type": "db_dtw", "tokens": "DTW Top Matches [1, 11, 1]"},
        {"id": "DTW_02", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_ori3 (Kinematic Orienting)", "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "DTW Top Matches [6, 17, 2]"},
        {"id": "DTW_03", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_rear3 (Kinematic Rearing)", "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "DTW Top Matches [14, 7, 34]"}
    ]

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXEMPLAR FILTERING IMPACT EVALUATION: FULL POPULATION VS MODES 2 & 3")
    print("=" * 70)

    # 1. Load metadata
    meta_df = pd.read_csv(META_CSV)
    meta_df["animal_id"] = meta_df["animal_id"].astype(str)
    meta_df["drug"] = meta_df["drug"].astype(str)
    unique_animals = sorted(list(set(meta_df["animal_id"])))
    animal_genotypes = {str(r["animal_id"]): str(r["genotype"]) for r in meta_df.to_dict(orient="records")}
    y_true = np.array([1 if animal_genotypes[aid] == "KI" else 0 for aid in unique_animals])

    # 2. Load modeling results
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]

    session_bouts = {}
    session_durations_min = {}
    all_bouts = []
    for rk, data in results.items():
        sylls = data["syllable"]
        session_durations_min[rk] = (len(sylls) / 40.0) / 60.0
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = np.array([sylls[0]] + sylls[c_idx].tolist())
        b_starts = np.array([0] + c_idx.tolist())
        b_ends = np.array(c_idx.tolist() + [len(sylls)])
        b_clusters = np.array([SYLL_TO_CLUSTER.get(s, 0) for s in b_sylls])
        session_bouts[rk] = {
            "sylls": b_sylls, "starts": b_starts, "ends": b_ends, "clusters": b_clusters
        }
        for i in range(len(b_sylls)):
            s = int(b_sylls[i])
            sf = int(b_starts[i])
            ef = int(b_ends[i])
            ns = int(b_sylls[i+1]) if i + 1 < len(b_sylls) else -1
            nc = SYLL_TO_CLUSTER.get(ns, -1) if ns != -1 else -1
            all_bouts.append({"rk": rk, "sf": sf, "ef": ef, "dur": ef - sf, "syll": s, "next_s": ns, "next_c": nc})

    evaluator = TransitionBucketEvaluator(results, all_bouts)

    df_b = pd.read_csv(DATA_DIR / "diverse_templates_match_database.csv") if (DATA_DIR / "diverse_templates_match_database.csv").exists() else pd.DataFrame()
    if not df_b.empty:
        df_b["animal_id"] = df_b["animal_id"].astype(str)
        df_b["drug"] = df_b["drug"].astype(str)

    df_dtw = pd.read_csv(DATA_DIR / "top20_retrieved_episodes_alignment.csv") if (DATA_DIR / "top20_retrieved_episodes_alignment.csv").exists() else pd.DataFrame()
    if not df_dtw.empty:
        df_dtw["animal_id"] = df_dtw["animal_id"].astype(str)
        df_dtw["drug"] = df_dtw["drug"].astype(str)

    candidates = load_candidate_definitions()

    # Background trajectories for density ratio (sample 1,000 random background bouts)
    bg_trajs = []
    np.random.seed(42)
    sample_bouts = [all_bouts[i] for i in np.random.choice(len(all_bouts), min(1000, len(all_bouts)), replace=False)]
    for b in sample_bouts:
        c = results[b["rk"]]["latent_state"][b["sf"]:b["ef"]]
        if len(c) >= 5:
            t_orig = np.linspace(0, 1, len(c))
            t_target = np.linspace(0, 1, 20)
            interp = interp1d(t_orig, c, axis=0)
            bg_trajs.append(interp(t_target).flatten())
    X_bg = np.array(bg_trajs)
    pca_global = PCA(n_components=4, random_state=42).fit(X_bg)
    Xpca_bg = pca_global.transform(X_bg)
    bg_nbrs = NearestNeighbors(n_neighbors=min(50, len(Xpca_bg) - 1)).fit(Xpca_bg)

    # Storage for animal-level rates across the 3 modes: mode -> cand_id -> aid -> (sal_rate, amph_rate)
    modes = ["Mode 1 (Full Population)", "Mode 2 (Density Exemplars)", "Mode 3 (Transition Exemplars)"]
    rates_by_mode = {m: {c["id"]: {} for c in candidates} for m in modes}

    print("\nExtracting and scoring instances across all 24 candidates...")
    for c_idx, cand in enumerate(candidates, 1):
        cid = cand["id"]
        cname = cand["name"]
        ctype = cand["type"]

        # Extract raw candidate instances
        raw_candidates = []
        if ctype == "exact":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                m = [r for r in meta_df.to_dict(orient="records") if r["results_dict_key"] == rk][0]
                b_sylls = b_info["sylls"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                for i in range(len(b_sylls) - pat_len + 1):
                    if np.array_equal(b_sylls[i:i+pat_len], pat):
                        sf = int(b_starts[i])
                        ef = int(b_ends[i+pat_len-1])
                        s_bounds = [int(b_starts[i+s]) for s in range(pat_len)] + [ef]
                        raw_candidates.append({
                            "recording_key": rk, "animal_id": str(m["animal_id"]), "drug": str(m["drug"]),
                            "start_frame": sf, "end_frame": ef, "stage_boundaries": s_bounds
                        })

        elif ctype == "cluster_rle":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                m = [r for r in meta_df.to_dict(orient="records") if r["results_dict_key"] == rk][0]
                b_clusters = b_info["clusters"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                c_change = np.where(b_clusters[1:] != b_clusters[:-1])[0] + 1
                c_vals = np.array([b_clusters[0]] + b_clusters[c_change].tolist())
                c_b_start = np.array([0] + c_change.tolist())
                c_b_end = np.array(c_change.tolist() + [len(b_clusters)])
                for i in range(len(c_vals) - pat_len + 1):
                    if np.array_equal(c_vals[i:i+pat_len], pat):
                        start_bout = c_b_start[i]
                        end_bout = c_b_end[i+pat_len-1] - 1
                        sf = int(b_starts[start_bout])
                        ef = int(b_ends[end_bout])
                        s_bounds = [int(b_starts[c_b_start[i+s]]) for s in range(pat_len)] + [ef]
                        raw_candidates.append({
                            "recording_key": rk, "animal_id": str(m["animal_id"]), "drug": str(m["drug"]),
                            "start_frame": sf, "end_frame": ef, "stage_boundaries": s_bounds
                        })

        elif ctype == "db_b":
            tid = cand["template_id"]
            sub = df_b[df_b["template_id"] == tid]
            for _, r in sub.iterrows():
                raw_candidates.append({
                    "recording_key": str(r.get("recording_key", "")), "animal_id": str(r["animal_id"]), "drug": str(r["drug"]),
                    "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"]),
                    "stage_boundaries": [int(r["start_frame"]), int(r["end_frame"])]
                })

        elif ctype == "db_dtw":
            for _, r in df_dtw.iterrows():
                raw_candidates.append({
                    "recording_key": str(r.get("recording_key", "")), "animal_id": str(r["animal_id"]), "drug": str(r["drug"]),
                    "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"]),
                    "stage_boundaries": [int(r["start_frame"]), int(r["end_frame"])]
                })

        # --- MODE 1: All Instances ---
        mode1_occs = raw_candidates

        # --- MODE 2: Whole-Chain Density-Ratio Top Instances ---
        mode2_occs = []
        if len(raw_candidates) >= 5:
            trajs = []
            valid_c = []
            for c in raw_candidates:
                rk = c.get("recording_key")
                if not rk or rk not in results:
                    matching_rks = [r["results_dict_key"] for r in meta_df.to_dict(orient="records") if str(r["animal_id"]) == c["animal_id"] and str(r["drug"]).lower() == c["drug"].lower()]
                    rk = matching_rks[0] if matching_rks else None
                if rk and rk in results:
                    chunk = results[rk]["latent_state"][c["start_frame"]:c["end_frame"]]
                    if len(chunk) >= 5:
                        t_orig = np.linspace(0, 1, len(chunk))
                        t_target = np.linspace(0, 1, 20)
                        interp = interp1d(t_orig, chunk, axis=0)
                        trajs.append(interp(t_target).flatten())
                        valid_c.append(c)
            if len(trajs) >= 5:
                X_cand = np.array(trajs)
                pca = PCA(n_components=min(4, len(X_cand)-1), random_state=42).fit(X_cand)
                Xpca_cand = pca.transform(X_cand)
                k_pool = min(50, max(2, int(np.ceil(0.10 * len(Xpca_cand)))))
                nbrs = NearestNeighbors(n_neighbors=k_pool).fit(Xpca_cand)
                local_dists, local_indices = nbrs.kneighbors(Xpca_cand)
                local_density = 1.0 / np.maximum(1e-6, local_dists.mean(axis=1))

                # Global density
                Xpca_cand_global = pca_global.transform(X_cand)
                global_dists, _ = bg_nbrs.kneighbors(Xpca_cand_global)
                global_density = 1.0 / np.maximum(1e-6, global_dists.mean(axis=1))

                density_ratio = local_density / global_density
                exemplar_idx = int(np.argmax(density_ratio))
                top_indices = set(local_indices[exemplar_idx])
                mode2_occs = [valid_c[i] for i in top_indices]
            else:
                mode2_occs = raw_candidates[:min(50, len(raw_candidates))]
        else:
            mode2_occs = raw_candidates

        # --- MODE 3: Forward Transition Strict Exemplars (min_p >= 0.50) ---
        mode3_occs = []
        for c in raw_candidates:
            rk = c.get("recording_key")
            if not rk or rk not in results:
                matching_rks = [r["results_dict_key"] for r in meta_df.to_dict(orient="records") if str(r["animal_id"]) == c["animal_id"] and str(r["drug"]).lower() == c["drug"].lower()]
                rk = matching_rks[0] if matching_rks else None
            bounds = c["stage_boundaries"]
            n_stages = len(bounds) - 1
            if rk and rk in results and n_stages >= 1:
                sylls = results[rk]["syllable"]
                stage_ps = []
                for k in range(n_stages):
                    sf = bounds[k]
                    ef = bounds[k+1]
                    s = int(sylls[sf]) if sf < len(sylls) else 0
                    if k + 1 < n_stages:
                        next_sf = bounds[k+1]
                        ns = int(sylls[next_sf]) if next_sf < len(sylls) else -1
                    else:
                        ns = int(sylls[ef]) if ef < len(sylls) else -1
                    nc = SYLL_TO_CLUSTER.get(ns, -1) if ns != -1 else -1
                    p, _, _ = evaluator.score_bout(rk, sf, ef, s, ns, nc)
                    stage_ps.append(p)
                if stage_ps and min(stage_ps) >= 0.50:
                    mode3_occs.append(c)

        # Compute animal-level rates for each mode
        for m_name, occ_list in [
            ("Mode 1 (Full Population)", mode1_occs),
            ("Mode 2 (Density Exemplars)", mode2_occs),
            ("Mode 3 (Transition Exemplars)", mode3_occs)
        ]:
            df_curr = pd.DataFrame(occ_list)
            for aid in unique_animals:
                sal_rk = [r["results_dict_key"] for r in meta_df.to_dict(orient="records") if str(r["animal_id"]) == aid and str(r["drug"]).lower() == "saline"][0]
                amph_rk = [r["results_dict_key"] for r in meta_df.to_dict(orient="records") if str(r["animal_id"]) == aid and str(r["drug"]).lower() == "amphetamine"][0]
                sal_min = session_durations_min[sal_rk]
                amph_min = session_durations_min[amph_rk]

                if not df_curr.empty:
                    sal_c = len(df_curr[(df_curr["animal_id"] == aid) & (df_curr["drug"].str.lower() == "saline")])
                    amph_c = len(df_curr[(df_curr["animal_id"] == aid) & (df_curr["drug"].str.lower() == "amphetamine")])
                else:
                    sal_c, amph_c = 0, 0

                rates_by_mode[m_name][cid][aid] = (sal_c / max(0.1, sal_min), amph_c / max(0.1, amph_min))

    # Evaluate Contrasts and LOAO for each mode
    print("\nComputing statistical contrasts and LOAO decoding across modes...")
    mode_results = []

    wt_aids = [aid for aid in unique_animals if animal_genotypes[aid] == "WT"]
    ki_aids = [aid for aid in unique_animals if animal_genotypes[aid] == "KI"]

    def eval_loao(X):
        preds = np.zeros(len(unique_animals))
        for i in range(len(unique_animals)):
            train_idx = [j for j in range(len(unique_animals)) if j != i]
            test_idx = [i]
            X_tr, y_tr = X[train_idx], y_true[train_idx]
            X_te = X[test_idx]
            mu = X_tr.mean(axis=0)
            sigma = X_tr.std(axis=0)
            sigma[sigma < 1e-6] = 1.0
            X_tr_s = (X_tr - mu) / sigma
            X_te_s = (X_te - mu) / sigma
            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            clf.fit(X_tr_s, y_tr)
            preds[i] = clf.predict(X_te_s)[0]
        return float(balanced_accuracy_score(y_true, preds))

    for m_name in modes:
        cand_contrasts = []
        X_sal = np.zeros((len(unique_animals), len(candidates)))

        for c_idx, cand in enumerate(candidates):
            cid = cand["id"]
            wt_sal_rates = [rates_by_mode[m_name][cid][a][0] for a in wt_aids]
            ki_sal_rates = [rates_by_mode[m_name][cid][a][0] for a in ki_aids]
            pairs = [rates_by_mode[m_name][cid][a] for a in unique_animals]

            # Contrast 1: Genotype in Saline
            mean_wt = float(np.mean(wt_sal_rates))
            mean_ki = float(np.mean(ki_sal_rates))
            std_pool = float(np.sqrt((np.var(wt_sal_rates, ddof=1)*(len(wt_aids)-1) + np.var(ki_sal_rates, ddof=1)*(len(ki_aids)-1)) / (len(unique_animals)-2)))
            d_c1 = (mean_ki - mean_wt) / max(1e-6, std_pool)
            _, p_c1 = exact_2sample_permutation_test(ki_sal_rates, wt_sal_rates)

            # Contrast 2: Paired Drug Shift
            d_c2 = float(np.mean([a - s for s, a in pairs])) / max(1e-6, np.std([a - s for s, a in pairs], ddof=1))
            _, p_c2 = exact_paired_permutation_test(pairs)

            # Contrast 3: Genotype x Drug Interaction
            wt_deltas = [rates_by_mode[m_name][cid][a][1] - rates_by_mode[m_name][cid][a][0] for a in wt_aids]
            ki_deltas = [rates_by_mode[m_name][cid][a][1] - rates_by_mode[m_name][cid][a][0] for a in ki_aids]
            std_pool_inter = float(np.sqrt((np.var(wt_deltas, ddof=1)*(len(wt_aids)-1) + np.var(ki_deltas, ddof=1)*(len(ki_aids)-1)) / (len(unique_animals)-2)))
            d_c3 = (float(np.mean(ki_deltas)) - float(np.mean(wt_deltas))) / max(1e-6, std_pool_inter)
            _, p_c3 = exact_2sample_permutation_test(ki_deltas, wt_deltas)

            cand_contrasts.append({
                "id": cid, "method": cand["method"], "d_c1": d_c1, "p_c1": p_c1,
                "d_c2": d_c2, "p_c2": p_c2, "d_c3": d_c3, "p_c3": p_c3
            })

            for a_idx, aid in enumerate(unique_animals):
                X_sal[a_idx, c_idx] = rates_by_mode[m_name][cid][aid][0]

        df_cc = pd.DataFrame(cand_contrasts)
        acc_all = eval_loao(X_sal)

        acc_by_method = {}
        for meth in sorted(list(set(c["method"] for c in candidates))):
            meth_cids = [c["id"] for c in candidates if c["method"] == meth]
            meth_idxs = [i for i, c in enumerate(candidates) if c["method"] == meth]
            X_sub = X_sal[:, meth_idxs]
            acc_by_method[meth] = eval_loao(X_sub)

        mode_results.append({
            "mode": m_name,
            "mean_abs_d_c1": df_cc["d_c1"].abs().mean(),
            "max_abs_d_c1": df_cc["d_c1"].abs().max(),
            "n_sig_c1_p05": (df_cc["p_c1"] < 0.05).sum(),
            "n_trend_c1_p10": (df_cc["p_c1"] < 0.10).sum(),
            "mean_abs_d_c2": df_cc["d_c2"].abs().mean(),
            "max_abs_d_c2": df_cc["d_c2"].abs().max(),
            "n_sig_c2_p05": (df_cc["p_c2"] < 0.05).sum(),
            "mean_abs_d_c3": df_cc["d_c3"].abs().mean(),
            "max_abs_d_c3": df_cc["d_c3"].abs().max(),
            "loao_overall": acc_all,
            "loao_by_method": acc_by_method,
            "df_contrasts": df_cc
        })

    # Print summary
    print("\n" + "=" * 85)
    print(f"{'STATISTICAL METRIC':<35} | {'MODE 1: FULL POPULATION':<22} | {'MODE 2: DENSITY CORE':<22} | {'MODE 3: STRICT TRANSITION':<22}")
    print("=" * 85)
    r1, r2, r3 = mode_results[0], mode_results[1], mode_results[2]

    print(f"{'Total Discovered Candidates':<35} | {24:<22} | {24:<22} | {24:<22}")
    print(f"{'Overall LOAO Genotype Decoding':<35} | {r1['loao_overall']*100:.1f}%{'':<17} | {r2['loao_overall']*100:.1f}%{'':<17} | {r3['loao_overall']*100:.1f}%{'':<17}")
    print(f"{'Contrast 1: Mean |Cohen d| (Genotype)':<35} | {r1['mean_abs_d_c1']:.2f}{'':<18} | {r2['mean_abs_d_c1']:.2f}{'':<18} | {r3['mean_abs_d_c1']:.2f}{'':<18}")
    print(f"{'Contrast 1: Peak |Cohen d| (Genotype)':<35} | {r1['max_abs_d_c1']:.2f}{'':<18} | {r2['max_abs_d_c1']:.2f}{'':<18} | {r3['max_abs_d_c1']:.2f}{'':<18}")
    print(f"{'Significant C1 Candidates (p < 0.05)':<35} | {r1['n_sig_c1_p05']}{'':<21} | {r2['n_sig_c1_p05']}{'':<21} | {r3['n_sig_c1_p05']}{'':<21}")
    print(f"{'Trend C1 Candidates (p < 0.10)':<35} | {r1['n_trend_c1_p10']}{'':<21} | {r2['n_trend_c1_p10']}{'':<21} | {r3['n_trend_c1_p10']}{'':<21}")
    print(f"{'Contrast 2: Mean |Cohen d| (Drug Shift)':<35} | {r1['mean_abs_d_c2']:.2f}{'':<18} | {r2['mean_abs_d_c2']:.2f}{'':<18} | {r3['mean_abs_d_c2']:.2f}{'':<18}")
    print(f"{'Significant C2 Candidates (p < 0.05)':<35} | {r1['n_sig_c2_p05']}{'':<21} | {r2['n_sig_c2_p05']}{'':<21} | {r3['n_sig_c2_p05']}{'':<21}")
    print(f"{'Contrast 3: Mean |Cohen d| (Interaction)':<35} | {r1['mean_abs_d_c3']:.2f}{'':<18} | {r2['mean_abs_d_c3']:.2f}{'':<18} | {r3['mean_abs_d_c3']:.2f}{'':<18}")
    print(f"{'Contrast 3: Peak |Cohen d| (Interaction)':<35} | {r1['max_abs_d_c3']:.2f}{'':<18} | {r2['max_abs_d_c3']:.2f}{'':<18} | {r3['max_abs_d_c3']:.2f}{'':<18}")
    print("-" * 85)
    print("\nLOAO GENOTYPE DECODING ACCURACY BY METHOD:")
    print(f"{'Method':<35} | {'Mode 1 (Full Pop)':<17} | {'Mode 2 (Density)':<17} | {'Mode 3 (Transition)':<17}")
    print("-" * 90)
    for meth in sorted(list(r1["loao_by_method"].keys())):
        print(f"{meth:<35} | {r1['loao_by_method'][meth]*100:.1f}%{'':<11} | {r2['loao_by_method'][meth]*100:.1f}%{'':<11} | {r3['loao_by_method'][meth]*100:.1f}%{'':<11}")
    print("=" * 90)

    # Save detailed CSV comparison
    comp_records = []
    for m_res in mode_results:
        m_name = m_res["mode"]
        for _, r in m_res["df_contrasts"].iterrows():
            rec = dict(r)
            rec["mode"] = m_name
            comp_records.append(rec)
    pd.DataFrame(comp_records).to_csv(DATA_DIR / "exemplar_modes_statistical_contrasts_comparison.csv", index=False)
    print(f"\nSaved detailed comparison to {DATA_DIR / 'exemplar_modes_statistical_contrasts_comparison.csv'}")

if __name__ == "__main__":
    main()
