#!/usr/bin/env python3
"""
fit_clustered_seqnmf_model.py
=============================================================================
Fits Convolutional seqNMF on the 14-Meta-Syllable behavioral stream.
Discovers high-order behavioral macro-strategies (e.g. Foraging, Rearing-Descent,
Slalom Locomotion, Grooming Chains, Vigilance Checks).

Extracts empirical verified episodes and quantifies Genotype (WT vs. KI)
and Drug (Saline vs. Amphetamine) modulation.
=============================================================================
"""

import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.optimized_seqnmf.convolutional_seqnmf import ConvolutionalSeqNMF

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf"
DATA_FILE = OUT_DIR / "clustered_sequence_dataset.pkl"

def main():
    print("=" * 80)
    print("FITTING CONVOLUTIONAL SEQNMF ON 14-META-SYLLABLE CLUSTERED STREAM")
    print("=" * 80)

    # 1. Load clustered dataset
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    meta_defs = bundle["meta_syllable_defs"]
    meta_names = bundle["meta_names"]
    sessions = bundle["sessions"]
    S = len(meta_names)  # 14

    print(f"Loaded {len(sessions)} sessions across {S} meta-syllables:")
    for i, name in enumerate(meta_names):
        print(f"  [{i:2d}] {name}")

    # 2. Assemble continuous stream (Mode 1: Deduplicated Macro-Bouts)
    all_macro_meta = []
    all_durs = []
    all_starts = []
    all_ends = []
    all_aids = []
    all_gts = []
    all_drugs = []
    all_sess = []

    for sess in sessions:
        df = sess["macro_bouts_df"]
        if len(df) == 0:
            continue
        all_macro_meta.extend(df["meta_syllable"].values)
        all_durs.extend(df["duration_sec"].values)
        all_starts.extend(df["start_frame"].values)
        all_ends.extend(df["end_frame"].values)
        all_aids.extend([str(sess["animal_id"])] * len(df))
        all_gts.extend([sess["genotype"]] * len(df))
        all_drugs.extend([sess["drug"]] * len(df))
        all_sess.extend([sess["session_idx"]] * len(df))

    T_full = len(all_macro_meta)
    all_macro_meta = np.array(all_macro_meta, dtype=int)
    all_durs = np.array(all_durs, dtype=float)
    all_starts = np.array(all_starts, dtype=int)
    all_ends = np.array(all_ends, dtype=int)
    all_aids = np.array(all_aids)
    all_gts = np.array(all_gts)
    all_drugs = np.array(all_drugs)
    all_sess = np.array(all_sess, dtype=int)

    print(f"\nTotal Macro-Bouts assembled: {T_full:,}")

    # Build one-hot matrix X (S x T)
    # Use first 50,000 macro-bouts for rapid optimal GPU fitting (covers full cohort diversity)
    N_FIT = min(60000, T_full)
    X = np.zeros((S, N_FIT), dtype=np.float32)
    X[all_macro_meta[:N_FIT], np.arange(N_FIT)] = 1.0

    # 3. Fit Convolutional seqNMF
    # K=8 macro-motifs, L=5 macro-steps (captures ~2.5 - 5 seconds of behavior), lambda=0.5
    K = 8
    L = 5
    lam = 0.5
    max_iter = 100

    print(f"\nFitting Convolutional seqNMF on RTX 4090 GPU (K={K}, L={L}, lambda={lam}, max_iter={max_iter})...")
    t0 = time.time()
    model = ConvolutionalSeqNMF(K=K, L=L, lambda_ortho=lam, max_iter=max_iter, random_seed=42, use_gpu=True, verbose=True)
    model.fit(X)
    fit_time = time.time() - t0

    rel_err = model.relative_reconstruction_error(X)
    redundancy = model.compute_factor_redundancy()
    print(f"\n✓ Model fitted in {fit_time:.2f}s! RelErr = {rel_err:.4f}, Factor Redundancy = {redundancy:.4f}")

    # 4. Extract Macro-Motifs & Retrieve Verified Episodes
    print("\n" + "=" * 80)
    print("EXTRACTING MACRO-MOTIFS & VERIFYING EPISODES")
    print("=" * 80)

    motifs_summary = []
    all_episodes = []

    # Episode retrieval threshold
    for k in range(K):
        # Template W_k shape: (S, L)
        W_k = model.W[:, k, :]
        # Dominant token at each lag
        dom_meta_ids = np.argmax(W_k, axis=0)
        dom_weights = np.max(W_k, axis=0)

        chain_names = [meta_names[idx] for idx in dom_meta_ids]
        chain_str = " -> ".join(chain_names)

        # Mean & max activation
        H_k = model.H[k]
        mean_act = float(np.mean(H_k))
        max_act = float(np.max(H_k))

        # Candidate episode detection: peaks in H_k > threshold
        thresh = mean_act + 2.0 * np.std(H_k)
        peaks = []
        for t in range(1, len(H_k) - 1):
            if H_k[t] > thresh and H_k[t] > H_k[t-1] and H_k[t] >= H_k[t+1]:
                peaks.append(t)

        # Non-maximum suppression within L steps
        suppressed_peaks = []
        last_t = -999
        for p in peaks:
            if p - last_t >= L:
                suppressed_peaks.append(p)
                last_t = p

        # Template verification
        verified_episodes = 0
        wt_count = 0
        ki_count = 0
        sal_count = 0
        amph_count = 0

        norm_W_k = np.linalg.norm(W_k, "fro")

        for p in suppressed_peaks:
            if p + L <= N_FIT:
                Y_t = X[:, p : p + L]
                norm_Y = np.linalg.norm(Y_t, "fro")
                if norm_Y > 0 and norm_W_k > 0:
                    inner = np.sum(Y_t * W_k)
                    a_opt = max(0.0, inner / (norm_W_k ** 2))
                    resid = Y_t - a_opt * W_k
                    err = float(np.linalg.norm(resid, "fro") ** 2 / (norm_Y ** 2))

                    if err < 0.65:  # High confidence match
                        verified_episodes += 1
                        gt = all_gts[p]
                        dr = all_drugs[p]
                        if gt == "WT":
                            wt_count += 1
                        else:
                            ki_count += 1
                        if dr == "Saline":
                            sal_count += 1
                        else:
                            amph_count += 1

                        # Extract observed macro sequence
                        obs_meta = all_macro_meta[p : p + L]
                        obs_names = " -> ".join([meta_names[m] for m in obs_meta])

                        all_episodes.append({
                            "factor_id": k + 1,
                            "template_chain": chain_str,
                            "observed_chain": obs_names,
                            "start_macro_bout": p,
                            "end_macro_bout": p + L,
                            "start_frame": all_starts[p],
                            "end_frame": all_ends[p + L - 1],
                            "total_duration_sec": np.sum(all_durs[p : p + L]),
                            "peak_activation": float(H_k[p]),
                            "template_error": err,
                            "animal_id": all_aids[p],
                            "genotype": gt,
                            "drug": dr,
                            "session_idx": all_sess[p],
                        })

        motifs_summary.append({
            "motif_id": k + 1,
            "template_chain": chain_str,
            "dom_meta_ids": dom_meta_ids.tolist(),
            "dom_weights": np.round(dom_weights, 3).tolist(),
            "mean_activation": mean_act,
            "max_activation": max_act,
            "verified_episodes": verified_episodes,
            "wt_episodes": wt_count,
            "ki_episodes": ki_count,
            "saline_episodes": sal_count,
            "amph_episodes": amph_count,
            "wt_ki_ratio": float(wt_count / max(1, ki_count)),
        })

        print(f"\nMotif {k+1}: {chain_str}")
        print(f"  Weights: {np.round(dom_weights, 3)} | Episodes: {verified_episodes} (WT: {wt_count}, KI: {ki_count} | Saline: {sal_count}, Amph: {amph_count})")

    # Save summary CSV
    summary_df = pd.DataFrame(motifs_summary)
    summary_csv = OUT_DIR / "clustered_seqnmf_motifs_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved motifs summary to {summary_csv}")

    episodes_df = pd.DataFrame(all_episodes)
    episodes_csv = OUT_DIR / "clustered_seqnmf_verified_episodes.csv"
    episodes_df.to_csv(episodes_csv, index=False)
    print(f"✓ Saved {len(episodes_df):,} verified episodes to {episodes_csv}")

    # 5. Plot Clustered Factor Matrices W_k
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    for k in range(K):
        ax = axes[k]
        W_k = model.W[:, k, :]  # Shape: (S=14, L=5)
        sns.heatmap(W_k, ax=ax, cmap="magma", cbar=True,
                    xticklabels=[f"Lag {l}" for l in range(L)],
                    yticklabels=meta_names if k % 4 == 0 else False)
        top_str = motifs_summary[k]["template_chain"].replace(" -> ", "\n-> ")
        ax.set_title(f"Macro-Motif {k+1} (n={motifs_summary[k]['verified_episodes']})\n{top_str}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Macro-Bout Lag Step")
        if k % 4 == 0:
            ax.set_ylabel("Meta-Syllable")

    plt.suptitle("Convolutional seqNMF on 14-Meta-Syllable Stream (RTX 4090 GPU, K=8, L=5)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig_clustered_seqnmf_motifs.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved diagnostic multi-panel figure to {fig_path}")

    # Save full pickle model
    res_bundle = {
        "K": K, "L": L, "lambda": lam,
        "W": model.W, "H": model.H,
        "meta_names": meta_names,
        "summary_df": summary_df,
        "episodes_df": episodes_df,
    }
    pkl_path = OUT_DIR / "clustered_seqnmf_model_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(res_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved full results bundle to {pkl_path}")
    print("\nDONE!")

if __name__ == "__main__":
    main()
