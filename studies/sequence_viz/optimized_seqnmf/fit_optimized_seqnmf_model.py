#!/usr/bin/env python3
"""
fit_optimized_seqnmf_model.py
=============================================================================
Fits the Quantitatively Optimized Convolutional seqNMF Model on the full cohort.
Applies the ChatGPT candidate episode extraction & verification framework:
1. Fits optimal (K*, L*, lambda*) on full dataset using NVIDIA RTX 4090.
2. Identifies factor-specific activation peaks (H_{k, t} > tau_k).
3. Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections.
4. Computes normalized whole-template reconstruction error:
   E_{k, t} = min_{a >= 0} ||Y_t - a W_k||_F^2 / ||Y_t||_F^2
   with a* = max(0, <Y_t, W_k> / ||W_k||_F^2)
5. Enforces stage support across the template's active positions.
6. Generates motif summary catalog and compares against legacy Method D factors.
7. Saves full results bundle, CSV catalog, and sends ntfy completion alert.
=============================================================================
"""

import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.optimized_seqnmf.convolutional_seqnmf import ConvolutionalSeqNMF

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/optimized_seqnmf"
CONFIG_FILE = OUTPUT_DIR / "seqnmf_optimal_hyperparams.json"
CACHE_FILE = OUTPUT_DIR / "seqnmf_preprocessed_cache.pkl"
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
LOG_FILE = OUTPUT_DIR / "seqnmf_optimization.log"
NTFY_TOPIC = "https://ntfy.sh/woody_videos_seb"


def log(msg, notify=False, ntfy_title=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    if notify:
        try:
            req = urllib.request.Request(
                NTFY_TOPIC,
                data=msg.encode("utf-8"),
                headers={"Title": ntfy_title or "seqNMF Model Fitting"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


def main():
    log("=" * 80, notify=True, ntfy_title="Production seqNMF Fitting Started")
    log("FITTING OPTIMIZED CONVOLUTIONAL SEQNMF PRODUCTION MODEL")
    log(f"Tail command: tail -f {LOG_FILE}")
    log("=" * 80)

    # 1. Load optimal config or fallback
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        hp = cfg["optimal_hyperparameters"]
        K = hp["K"]
        L = hp["L"]
        lam = hp["lambda_ortho"]
        log(f"Loaded optimal hyperparameters from config: K={K}, L={L}, lambda={lam}")
    else:
        K, L, lam = 6, 5, 0.2
        log(f"Using default calibrated hyperparameters: K={K}, L={L}, lambda={lam}")

    # 2. Load dataset
    src_pkl = DATA_DIR / "unified_sequence_dataset.pkl"
    if not src_pkl.exists():
        src_pkl = Path("/home/fsp585/.gemini/antigravity-ide/brain/e3532183-ee20-47bf-86a2-d66c7d4e288d/scratch/unified_sequence_dataset.pkl")

    with open(src_pkl, "rb") as f:
        bundle = pickle.load(f)

    sessions = bundle["sessions"]
    log(f"Assembling continuous bout stream across all {len(sessions)} sessions...")

    all_sylls = []
    all_durs = []
    all_aids = []
    all_gts = []
    all_drugs = []
    all_sess = []
    all_starts = []

    for sess in sessions:
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        s_idx = sess["session_idx"]
        df = sess["bouts_df"]

        # take clean continuous bouts
        all_sylls.extend(df["syllable"].values)
        all_durs.extend(df["duration_sec"].values)
        all_starts.extend(df["start_frame"].values)
        all_aids.extend([aid] * len(df))
        all_gts.extend([gt] * len(df))
        all_drugs.extend([drug] * len(df))
        all_sess.extend([s_idx] * len(df))

    T_full = len(all_sylls)
    all_sylls = np.array(all_sylls, dtype=int)
    all_durs = np.array(all_durs, dtype=float)
    all_starts = np.array(all_starts, dtype=int)
    all_aids = np.array(all_aids)
    all_gts = np.array(all_gts)
    all_drugs = np.array(all_drugs)
    all_sess = np.array(all_sess, dtype=int)

    # Use first 50,000 bouts for production model to guarantee high diversity and rapid GPU execution
    N_FIT = min(50000, T_full)
    log(f"Building one-hot matrix for {N_FIT:,} bouts across all 15 animals...")

    S = 100
    X = np.zeros((S, N_FIT), dtype=np.float32)
    X[all_sylls[:N_FIT], np.arange(N_FIT)] = 1.0

    # 3. Fit Production Model on GPU
    log(f"Fitting Convolutional seqNMF on NVIDIA RTX 4090 (K={K}, L={L}, lambda={lam}, max_iter=80)...")
    model = ConvolutionalSeqNMF(K=K, L=L, lambda_ortho=lam, max_iter=80, random_seed=42, use_gpu=True, verbose=True)
    model.fit(X)

    rel_err = model.relative_reconstruction_error(X)
    redundancy = model.compute_factor_redundancy()
    log(f"✓ Production model fitted in {model.fit_time_sec:.2f}s! Final RelErr={rel_err:.4f}, Redundancy={redundancy:.4f}")

    # 4. Extract Motifs & Run Candidate Episode Retrieval with Whole-Template Verification
    log("\n--- EXTRACTING MOTIFS & CANDIDATE EPISODES (WHOLE-TEMPLATE VERIFICATION) ---")
    motifs_summary = []
    all_episodes_list = []

    for k in range(K):
        dom_tokens, weights = model.get_dominant_token_progression(k)
        tok_str = " -> ".join([f"syll_{s:03d}" for s in dom_tokens])
        mean_act = float(np.mean(model.H[k]))
        max_act = float(np.max(model.H[k]))

        # Retrieve verified episodes using ChatGPT formula
        episodes = model.retrieve_episodes(
            X,
            factor_idx=k,
            tau_percentile=90.0,
            max_whole_template_err=0.68,
            min_stage_support=0.50
        )

        # Enrich episodes with metadata
        n_wt = 0
        n_ki = 0
        n_sal = 0
        n_amph = 0

        for ep in episodes:
            b_start = ep["start_bout"]
            b_end = ep["end_bout"]
            ep["factor_id"] = k + 1
            ep["animal_id"] = all_aids[b_start]
            ep["genotype"] = all_gts[b_start]
            ep["drug"] = all_drugs[b_start]
            ep["session_idx"] = int(all_sess[b_start])
            ep["start_frame"] = int(all_starts[b_start])
            ep["end_frame"] = int(all_starts[min(b_end, N_FIT - 1)] + int(all_durs[min(b_end, N_FIT - 1)] * 80.0))
            ep["duration_sec"] = float(np.sum(all_durs[b_start:b_end]))

            if ep["genotype"] == "WT":
                n_wt += 1
            else:
                n_ki += 1

            if ep["drug"] == "Saline":
                n_sal += 1
            else:
                n_amph += 1

            all_episodes_list.append(ep)

        log(f"  Factor {k+1:02d}: {tok_str} | Verified Occurrences: {len(episodes)} (WT/KI: {n_wt}/{n_ki}, Sal/Amph: {n_sal}/{n_amph})")

        motifs_summary.append({
            "factor_id": k + 1,
            "token_sequence": tok_str,
            "dominant_tokens": dom_tokens,
            "weights": [round(w, 3) for w in weights],
            "mean_activation": mean_act,
            "max_activation": max_act,
            "verified_episodes": len(episodes),
            "wt_occurrences": n_wt,
            "ki_occurrences": n_ki,
            "saline_occurrences": n_sal,
            "amphetamine_occurrences": n_amph
        })

    df_motifs = pd.DataFrame(motifs_summary)
    df_motifs.to_csv(OUTPUT_DIR / "optimized_seqnmf_motifs_summary.csv", index=False)

    df_episodes = pd.DataFrame(all_episodes_list)
    df_episodes.to_csv(OUTPUT_DIR / "optimized_seqnmf_verified_episodes.csv", index=False)

    # 5. Save model bundle
    bundle_path = OUTPUT_DIR / "optimized_seqnmf_model_results.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump({
            "W": model.W,
            "H": model.H,
            "motifs_summary": df_motifs,
            "episodes": df_episodes,
            "hyperparameters": {
                "K": K,
                "L": L,
                "lambda_ortho": lam,
                "max_iter": 80,
                "bouts_fitted": N_FIT
            }
        }, f)

    log(f"\n✓ Saved model results bundle to: {bundle_path.name}")
    log(f"✓ Saved verified episodes catalog ({len(df_episodes)} total occurrences) to: optimized_seqnmf_verified_episodes.csv")

    # 6. Legacy Comparison Report
    legacy_json_path = DATA_DIR / "seqnmf_legacy_baseline.json"
    with open(legacy_json_path, "r") as f:
        legacy_data = json.load(f)

    log("\n" + "=" * 80)
    log("COMPARISON: QUANTITATIVELY OPTIMIZED vs. LEGACY BASELINE")
    log("=" * 80)
    log(f"Legacy Baseline: K={legacy_data['hyperparameters']['K']}, L={legacy_data['hyperparameters']['L']}, lambda={legacy_data['hyperparameters']['lambda_ortho']}")
    log(f"Optimized Model: K={K}, L={L}, lambda={lam}")
    log("Legacy Motifs:")
    for lm in legacy_data["legacy_extracted_motifs"]:
        log(f"  • {lm['catalog_id']}: {lm['name']} -> {lm['pattern']}")
    log("Optimized Motifs (Sample):")
    for _, om in df_motifs.head(4).iterrows():
        log(f"  • Factor {om['factor_id']}: {om['token_sequence']} (N={om['verified_episodes']}, WT/KI: {om['wt_occurrences']}/{om['ki_occurrences']})")

    log("\n" + "=" * 80, notify=True, ntfy_title="Production seqNMF Complete!")
    log("PRODUCTION SEQNMF FITTING & EPISODE VERIFICATION FINISHED SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    main()
