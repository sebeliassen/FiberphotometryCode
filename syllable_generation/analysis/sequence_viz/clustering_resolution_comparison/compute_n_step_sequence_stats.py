#!/usr/bin/env python3
"""
compute_n_step_sequence_stats.py
=============================================================================
Computes exact empirical statistics on N-step sequence discovery:
- Combinatorial capacity vs observed repertoire size
- Recurrence frequency (>=20, >=50, >=100 occurrences cohort-wide)
- Cross-animal generalizability (>=5, >=10, all 15 animals)
- Robust shared motifs (>=20 occurrences AND >=8 distinct animals)
- Repertoire concentration (Top 10 and Top 50 motif coverage %)
across the 4 clustering levels:
Level 1: 100 Raw Syllables
Level 2: 93 Conservative Synonyms
Level 3: 22 Functional Asymmetry Preserved
Level 4: 14 Broad Functional Families
For sequence lengths N = 2, 3, 4, 5, 6 steps.
=============================================================================
"""

import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import h5py

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison"))

from run_clustering_resolution_experiment import get_level_definitions, get_verified_metadata

def main():
    t0 = time.time()
    res_path = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/results.h5"
    
    print("[INIT] Loading raw syllable streams from results.h5...", flush=True)
    with h5py.File(res_path, "r") as f:
        session_data = {k: f[f"{k}/syllable"][:] for k in sorted(f.keys()) if "syllable" in f[k]}

    levels = get_level_definitions()
    results_table = []

    print("\n" + "=" * 80, flush=True)
    print("N-STEP SEQUENCE DISCOVERY CAPACITY & REPERTOIRE SIZE ACROSS RESOLUTIONS", flush=True)
    print("=" * 80, flush=True)

    for lvl in levels:
        lvl_name = lvl["name"]
        lvl_short = lvl_name.split(":")[0]
        m_arr = lvl["map_arr"]
        vocab = lvl["vocab_size"]

        # 1. Pre-map and collapse all sessions
        animal_collapsed = defaultdict(list)
        for rk, raw_sylls in session_data.items():
            mapped = m_arr[raw_sylls]
            mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
            c_seq = mapped[mask]
            aid, geno, drug = get_verified_metadata(rk)
            animal_collapsed[aid].append(c_seq)

        for N in [2, 3, 4, 5, 6]:
            t_start = time.time()
            ngram_counts = Counter()
            ngram_animals = defaultdict(set)

            for aid, seq_list in animal_collapsed.items():
                for seq in seq_list:
                    L = len(seq)
                    if L < N:
                        continue
                    # Fast zip slicing
                    slices = [seq[i : L - N + 1 + i] for i in range(N)]
                    tuples = zip(*slices)
                    for gram in tuples:
                        ngram_counts[gram] += 1
                        ngram_animals[gram].add(aid)

            total_occurrences = sum(ngram_counts.values())
            unique_observed = len(ngram_counts)
            theoretical_max = vocab ** N

            # Threshold statistics
            rep_ge_20 = sum(1 for c in ngram_counts.values() if c >= 20)
            rep_ge_50 = sum(1 for c in ngram_counts.values() if c >= 50)
            rep_ge_100 = sum(1 for c in ngram_counts.values() if c >= 100)

            anim_ge_5 = sum(1 for aids in ngram_animals.values() if len(aids) >= 5)
            anim_ge_10 = sum(1 for aids in ngram_animals.values() if len(aids) >= 10)
            anim_all_15 = sum(1 for aids in ngram_animals.values() if len(aids) == 15)

            # Robust cohort motifs: >=20 occurrences AND >=8 animals
            robust_motifs = sum(
                1 for g, c in ngram_counts.items() if c >= 20 and len(ngram_animals[g]) >= 8
            )

            # Cumulative behavior concentration
            top10_occ = sum(c for _, c in ngram_counts.most_common(10))
            top10_cov = (top10_occ / total_occurrences) * 100.0 if total_occurrences > 0 else 0.0

            top50_occ = sum(c for _, c in ngram_counts.most_common(50))
            top50_cov = (top50_occ / total_occurrences) * 100.0 if total_occurrences > 0 else 0.0

            row = {
                "Level": lvl_short,
                "Vocab": vocab,
                "N_Step": N,
                "Theoretical_Max": theoretical_max,
                "Unique_Observed": unique_observed,
                "Repeat_ge_20": rep_ge_20,
                "Repeat_ge_50": rep_ge_50,
                "Repeat_ge_100": rep_ge_100,
                "Shared_ge_5_Animals": anim_ge_5,
                "Shared_ge_10_Animals": anim_ge_10,
                "Universal_15_Animals": anim_all_15,
                "Robust_Shared_Motifs": robust_motifs,
                "Top10_Coverage_Pct": round(top10_cov, 2),
                "Top50_Coverage_Pct": round(top50_cov, 2)
            }
            results_table.append(row)
            print(f"[{lvl_short} | Vocab={vocab:2d}] N={N}: Unique={unique_observed:6,d} | Robust(>=20,>=8anim)={robust_motifs:5,d} | Universal(15anim)={anim_all_15:4,d} | Top10 Coverage={top10_cov:5.1f}% ({time.time()-t_start:.2f}s)", flush=True)

    df_out = pd.DataFrame(results_table)
    out_csv = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/n_step_sequence_statistics.csv"
    df_out.to_csv(out_csv, index=False)
    print("\n" + "=" * 80, flush=True)
    print(f"COMPLETE in {time.time()-t0:.1f}s! Saved to: {out_csv}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
