#!/usr/bin/env python3
"""
compare_clustering_resolutions_separation.py
=============================================================================
Computes exact combinatorial animal-permutation tests (Total Variation distance,
N=6,435 partitions) comparing WT vs. DAT-KI across 4 nested clustering resolutions:
- Level 1: 100 Original Raw Syllables
- Level 2: 93 Conservative Synonyms
- Level 3: 22 Functional Families
- Level 4: 14 Broad Macro-Classes

Evaluates:
1. State Usage Repertoire (0th-order frequency distributions)
2. Joint Bigram Syntax (1st-order transition distributions)

Generates purely quantitative summary tables (STRICTLY NO FIGURES to prevent p-hacking).
=============================================================================
"""

import pickle
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
DATA_PKL = DATA_DIR / "unified_sequence_dataset.pkl"

def get_level_definitions():
    # Level 1: 100 syllables
    m1 = np.arange(100)
    
    # Level 2: 93 states
    m2 = np.arange(100)
    m2[11] = 1; m2[23] = 3; m2[38] = 19; m2[43] = 32; m2[49] = 32; m2[9] = 8; m2[52] = 27
    u2, inv2 = np.unique(m2, return_inverse=True)
    m2 = inv2
    
    # Level 3: 22 states
    L3_GROUPS = {
        "Walk_Sharp_L": [29], "Walk_Sharp_R": [30], "Turn_L": [6], "Turn_R": [2],
        "Descent_L": [4], "Descent_R": [7], "Look_L": [28, 34], "Look_R": [5, 22, 40, 42],
        "Walk_L": [3, 23, 55], "Walk_R": [1, 11], "Walk_Fwd": [19, 38],
        "Sprint": [0, 13, 17, 21, 26, 33, 41], "Rear_Elevated": [12, 14, 31],
        "Wall_Lean": [16], "Stand_Down": [37], "Sniff_Ground": [8, 9, 10, 18, 20, 25],
        "Sniff_Up": [15], "Sniff_Scrunched": [24, 36, 39],
        "Grooming": [27, 44, 52, 58, 60, 61, 63, 65, 66], "Rest": [32, 43, 49], "Artifact": [35]
    }
    n3 = list(L3_GROUPS.keys()) + ["Tail_Unlabeled"]
    m3 = np.full(100, len(n3) - 1, dtype=int)
    for g_idx, (g_name, s_ids) in enumerate(L3_GROUPS.items()):
        for sid in s_ids:
            m3[sid] = g_idx
            
    # Level 4: 14 states
    L4_GROUPS = {
        "Walk_Fwd": [19, 38], "Walk_L": [3, 23, 55], "Walk_R": [1, 11],
        "Sprint": [0, 13, 17, 21, 26, 33, 41], "Turn_R": [2, 30], "Turn_L": [6, 29],
        "Rear": [12, 14, 16, 31], "Descent": [4, 7, 37],
        "Sniff": [8, 9, 10, 15, 18, 20, 24, 25, 36, 39],
        "Look": [5, 22, 28, 34, 40, 42],
        "Grooming": [27, 44, 52, 58, 60, 61, 63, 65, 66], "Rest": [32, 43, 49], "Artifact": [35]
    }
    n4 = list(L4_GROUPS.keys()) + ["Tail_Unlabeled"]
    m4 = np.full(100, len(n4) - 1, dtype=int)
    for g_idx, (g_name, s_ids) in enumerate(L4_GROUPS.items()):
        for sid in s_ids:
            m4[sid] = g_idx
            
    return [
        {"level_id": 1, "name": "Level 1: Original Syllables", "vocab_size": 100, "mapping": m1},
        {"level_id": 2, "name": "Level 2: Conservative Synonyms", "vocab_size": len(u2), "mapping": m2},
        {"level_id": 3, "name": "Level 3: Functional Families", "vocab_size": len(n3), "mapping": m3},
        {"level_id": 4, "name": "Level 4: Broad Macro-Classes", "vocab_size": len(n4), "mapping": m4},
    ]


def main():
    print("=" * 80)
    print("COMPARING GENOTYPE SEPARATION ACROSS CLUSTERING RESOLUTIONS (LEVELS 1-4)")
    print("Exact Combinatorial Animal-Permutation Test (N=6,435 partitions)")
    print("=" * 80)
    
    with open(DATA_PKL, "rb") as f:
        d_bundle = pickle.load(f)
    sessions = d_bundle["sessions"]
    
    anims_gt = {s["animal_id"]: s["genotype"] for s in sessions}
    all_animals = sorted(list(anims_gt.keys()))
    wt_animals = [a for a in all_animals if anims_gt[a] == "WT"]
    ki_animals = [a for a in all_animals if anims_gt[a] == "KI"]
    
    partitions = list(itertools.combinations(all_animals, len(wt_animals)))
    N_partitions = len(partitions)
    all_set = set(all_animals)
    print(f"Cohort setup: {len(all_animals)} animals ({len(wt_animals)} WT, {len(ki_animals)} DAT-KI), N={N_partitions:,} partitions.")
    
    levels = get_level_definitions()
    
    records = []
    
    for lvl in levels:
        lvl_id = lvl["level_id"]
        lvl_name = lvl["name"]
        K = lvl["vocab_size"]
        mapping = lvl["mapping"]
        
        # 1. State Usage Repertoire
        profiles_state = {}
        # 2. Joint Bigram Syntax
        profiles_trans = {}
        
        for s in sessions:
            aid = s["animal_id"]
            drug = s["drug"]
            bdf = s["bouts_df"]
            
            raw_sylls = bdf["syllable"].values
            valid_mask = (raw_sylls >= 0) & (raw_sylls < 100)
            valid_sylls = raw_sylls[valid_mask]
            mapped = mapping[valid_sylls]
            
            # State frequency distribution
            counts = np.bincount(mapped, minlength=K).astype(float)
            p_state = counts / np.sum(counts)
            profiles_state[(aid, drug)] = p_state
            
            # Transition matrix (within segments)
            T_mat = np.zeros((K, K), dtype=float)
            for seg_id, sub in bdf.groupby("segment_id"):
                s_raw = sub["syllable"].values
                s_valid = s_raw[(s_raw >= 0) & (s_raw < 100)]
                s_mapped = mapping[s_valid]
                # Collapse consecutive identical states
                if len(s_mapped) > 0:
                    c_mask = np.concatenate([[True], s_mapped[1:] != s_mapped[:-1]])
                    c_seq = s_mapped[c_mask]
                    if len(c_seq) > 1:
                        for i in range(len(c_seq) - 1):
                            T_mat[c_seq[i], c_seq[i+1]] += 1.0
            T_sum = T_mat.sum()
            p_trans = (T_mat / T_sum).flatten() if T_sum > 0 else np.zeros(K * K)
            profiles_trans[(aid, drug)] = p_trans
            
        def calc_tv(prof_dict, g1, g2, drug=None):
            if drug is not None:
                v1 = [prof_dict[(a, drug)] for a in g1]
                v2 = [prof_dict[(a, drug)] for a in g2]
            else:
                v1 = [0.5 * (prof_dict[(a, "Saline")] + prof_dict[(a, "Amphetamine")]) for a in g1]
                v2 = [0.5 * (prof_dict[(a, "Saline")] + prof_dict[(a, "Amphetamine")]) for a in g2]
            m1 = np.mean(v1, axis=0)
            m2 = np.mean(v2, axis=0)
            return 0.5 * np.sum(np.abs(m1 - m2))
            
        # Run permutation tests for State Repertoire
        for metric_name, prof_dict in [("State_Usage_Repertoire", profiles_state), ("Joint_Bigram_Syntax", profiles_trans)]:
            for cond_name, drug_arg in [("Saline", "Saline"), ("Amphetamine", "Amphetamine"), ("Global_Pooled", None)]:
                tv_obs = calc_tv(prof_dict, wt_animals, ki_animals, drug_arg)
                null_dist = np.zeros(N_partitions)
                for p_idx, p_wt in enumerate(partitions):
                    p_ki = list(all_set - set(p_wt))
                    null_dist[p_idx] = calc_tv(prof_dict, p_wt, p_ki, drug_arg)
                p_val = float(np.mean(null_dist >= tv_obs))
                
                records.append({
                    "level_id": lvl_id,
                    "level_name": lvl_name,
                    "vocab_size": K,
                    "behavioral_tier": metric_name,
                    "condition": cond_name,
                    "tv_observed": tv_obs,
                    "exact_p_val": p_val,
                    "is_significant": p_val < 0.05,
                    "n_partitions": N_partitions
                })
                print(f"[{lvl_name} | {metric_name} | {cond_name:12s}] TV={tv_obs:.4f}, exact p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
                
    res_df = pd.DataFrame(records)
    out_csv = DATA_DIR / "clustering_resolution_genotype_separation_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved comprehensive cross-resolution separation table to {out_csv}")

if __name__ == "__main__":
    main()
