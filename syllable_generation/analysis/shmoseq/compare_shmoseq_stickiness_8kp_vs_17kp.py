#!/usr/bin/env python3
"""
compare_shmoseq_stickiness_8kp_vs_17kp.py
=============================================================================
Comparative shMoSeq Stickiness Calibration & Exact Genotype Permutation Test:
Head-to-head comparison between 8KP (Run 03) and 17KP (Run 12) across stickiness regimes.

Evaluation Criteria (per user and shMoSeq paper recommendations):
1. Ethological Timescale (Weinreb et al. 2026 parity):
   - Macro-states should persist for multi-second bouts (~5–30s).
   - Avoids monolithic session locking (max duration << 60 min).
   - Avoids sub-second syllable chatter (median duration > 1.0s).
2. Exact Animal-Level Permutation Test (6,435 combinations):
   - Unit of inference: Animal (8 WT vs 7 DAT-Het KI).
   - Preserves paired Saline and Amphetamine sessions.
   - Measures whether calibrated stickiness preserves or enhances genotype separation.
=============================================================================
"""

import itertools
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_8KP = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "output_8kp_run03" / "data" / "shmoseq_input_data_8kp.pkl"
DATA_17KP = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "output" / "data" / "shmoseq_input_data.pkl"

OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "stickiness_comparison_8kp_vs_17kp"
MODELS_DIR = OUT_DIR / "models"
FIGS_DIR = OUT_DIR / "figures"
TABS_DIR = OUT_DIR / "tables"


def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def l2_distance(p, q):
    return np.linalg.norm(p - q)


def exact_animal_permutation_test(occupancies_by_animal, animals_metadata, n_states=8):
    """
    Exhaustive combinatorial permutation test across all 6,435 partitions (15 choose 8).
    Returns observed statistics and exact non-parametric p-values for Saline and Amphetamine.
    """
    unique_animals = sorted(list(animals_metadata.keys()))
    wt_animals = [a for a in unique_animals if "WT" in animals_metadata[a]["genotype"]]
    ki_animals = [a for a in unique_animals if "KI" in animals_metadata[a]["genotype"] or "Het" in animals_metadata[a]["genotype"]]

    n_total = len(unique_animals)
    n_wt = len(wt_animals)
    n_ki = len(ki_animals)

    # Observed groups
    sal_wt_obs = np.array([occupancies_by_animal[a]["Saline"] for a in wt_animals])
    sal_ki_obs = np.array([occupancies_by_animal[a]["Saline"] for a in ki_animals])
    amph_wt_obs = np.array([occupancies_by_animal[a]["Amphetamine"] for a in wt_animals])
    amph_ki_obs = np.array([occupancies_by_animal[a]["Amphetamine"] for a in ki_animals])

    obs_sal_tv = tv_distance(sal_wt_obs.mean(axis=0), sal_ki_obs.mean(axis=0))
    obs_sal_l2 = l2_distance(sal_wt_obs.mean(axis=0), sal_ki_obs.mean(axis=0))
    obs_drug_delta_l2 = l2_distance(
        (amph_wt_obs - sal_wt_obs).mean(axis=0),
        (amph_ki_obs - sal_ki_obs).mean(axis=0)
    )

    all_sal = np.array([occupancies_by_animal[a]["Saline"] for a in unique_animals])
    all_amph = np.array([occupancies_by_animal[a]["Amphetamine"] for a in unique_animals])
    all_delta = all_amph - all_sal

    # Pre-index all combinations
    combos = list(itertools.combinations(range(n_total), n_wt))
    n_perm = len(combos)

    sal_tv_ge = 0
    sal_l2_ge = 0
    drug_delta_l2_ge = 0

    all_indices = set(range(n_total))
    for c in combos:
        c_set = set(c)
        c_comp = list(all_indices - c_set)
        c_list = list(c)

        # Saline group difference
        mean_wt_perm = all_sal[c_list].mean(axis=0)
        mean_ki_perm = all_sal[c_comp].mean(axis=0)
        perm_sal_tv = tv_distance(mean_wt_perm, mean_ki_perm)
        perm_sal_l2 = l2_distance(mean_wt_perm, mean_ki_perm)

        # Drug delta difference
        delta_wt_perm = all_delta[c_list].mean(axis=0)
        delta_ki_perm = all_delta[c_comp].mean(axis=0)
        perm_delta_l2 = l2_distance(delta_wt_perm, delta_ki_perm)

        if perm_sal_tv >= obs_sal_tv - 1e-9:
            sal_tv_ge += 1
        if perm_sal_l2 >= obs_sal_l2 - 1e-9:
            sal_l2_ge += 1
        if perm_delta_l2 >= obs_drug_delta_l2 - 1e-9:
            drug_delta_l2_ge += 1

    return {
        "obs_saline_tv": float(obs_sal_tv),
        "p_exact_saline_tv": float(sal_tv_ge / n_perm),
        "obs_saline_l2": float(obs_sal_l2),
        "p_exact_saline_l2": float(sal_l2_ge / n_perm),
        "obs_drug_delta_l2": float(obs_drug_delta_l2),
        "p_exact_drug_delta": float(drug_delta_l2_ge / n_perm),
        "total_permutations": n_perm,
    }


def fit_and_evaluate(data_file, skeleton_label, kappa, n_states=8, num_iters=50, seed_val=0):
    print("\n" + "=" * 80)
    print(f"🔬 RUNNING {skeleton_label} | Stickiness kappa={kappa:g} | K={n_states} (50 GPU Gibbs Iters)")
    print("=" * 80, flush=True)

    with open(data_file, "rb") as f:
        bundle = pickle.load(f)

    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    n_syllables = bundle["n_syllables"]

    seqs, mask, batch_meta = sm.batch(syllables_dict)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }

    hypparams = {
        "n_states": n_states,
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": 1.0,
        "trans_beta": 1.0,
        "trans_kappa": float(kappa),
        "n_syllables": n_syllables,
        "emission_gd_iters": 100,
        "emission_gd_lr": 5e-3,
        "random_seed": seed_val,
    }

    seed = jr.PRNGKey(seed_val)
    seed, subseed = jr.split(seed)
    states = jr.randint(subseed, data["syllables"].shape, 0, n_states)
    params = None

    t0 = time.time()
    for it in range(num_iters):
        t_iter = time.time()
        seed, subseed = jr.split(seed)
        params, losses = sm.hhmm_efficient.resample_params(subseed, data, states, hypparams, params)
        seed, subseed = jr.split(seed)
        states, marginal_loglik = sm.hhmm_efficient.resample_states(subseed, data, params, parallel=False)
        log_param_p = sm.hhmm_efficient.log_params_prob(params, hypparams)
        current_log_joint = float(log_param_p + marginal_loglik)
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [{skeleton_label} kappa={kappa:g} | Iter {it+1:02d}/{num_iters}] Log-Joint: {current_log_joint:.2f} ({time.time()-t_iter:.1f}s/it)", flush=True)

    fit_dur_min = (time.time() - t0) / 60.0
    print(f"✓ Model fitting completed in {fit_dur_min:.2f} min.", flush=True)

    # Posterior decoding
    print("Decoding smoothed posterior state trajectories...", flush=True)
    post_probs = sm.hhmm_efficient.smoothed_states(data, params, parallel=False)
    decoded_states = np.argmax(np.array(post_probs), axis=-1)

    # Compute episode timescales using explicit batch_meta keys to guarantee exact alignment
    batch_keys = list(batch_meta[0])
    episode_records = []
    animal_meta = {}
    occupancies_by_animal = {}
    decoded_states_dict = {}

    for i, s_key in enumerate(batch_keys):
        meta = metadata_dict[s_key]
        aid = str(meta["animal_id"])
        drug = str(meta["drug"])
        geno = str(meta["genotype"])
        animal_meta[aid] = {"genotype": geno}

        n_valid = int(data["mask"][i].sum())
        s_seq = decoded_states[i, :n_valid]
        decoded_states_dict[s_key] = s_seq

        # State occupancy vector
        counts = np.bincount(s_seq, minlength=n_states)
        occ = counts / (counts.sum() + 1e-9)
        if aid not in occupancies_by_animal:
            occupancies_by_animal[aid] = {}
        occupancies_by_animal[aid][drug] = occ

        st_arr, dur_arr, starts, ends = run_length_encoding(s_seq)
        dur_sec = dur_arr / 40.0
        for st, d_sec in zip(st_arr, dur_sec):
            episode_records.append({"duration_sec": float(d_sec), "state": int(st)})

    df_episodes = pd.DataFrame(episode_records)
    median_dur_sec = float(df_episodes["duration_sec"].median())
    mean_dur_sec = float(df_episodes["duration_sec"].mean())
    max_dur_min = float(df_episodes["duration_sec"].max() / 60.0)

    # Run Exact Permutation Test
    print("Running exact combinatorial animal permutation test (6,435 combinations)...", flush=True)
    perm_results = exact_animal_permutation_test(occupancies_by_animal, animal_meta, n_states=n_states)

    print(f"\n--- RESULTS: {skeleton_label} (kappa={kappa:g}) ---")
    print(f"  Timescale: Median={median_dur_sec:.2f}s | Mean={mean_dur_sec:.2f}s | Max Lock={max_dur_min:.2f} min")
    print(f"  Baseline Saline Permutation: TV={perm_results['obs_saline_tv']:.4f} (p={perm_results['p_exact_saline_tv']:.4f}) | L2={perm_results['obs_saline_l2']:.4f} (p={perm_results['p_exact_saline_l2']:.4f})")
    print(f"  Amphetamine Response Permutation: Delta L2={perm_results['obs_drug_delta_l2']:.4f} (p={perm_results['p_exact_drug_delta']:.4f})")

    # Generate fast vectorized ethogram heatmap
    fig_path = FIGS_DIR / f"ethogram_{skeleton_label.lower()}_kappa_{kappa:g}.png"
    plot_fast_ethogram(decoded_states, data["mask"], metadata_dict, batch_keys, n_states, fig_path,
                       title=f"{skeleton_label} (Stickiness kappa={kappa:g}) - Median Bout: {median_dur_sec:.2f}s, Max: {max_dur_min:.1f}m")

    # Save model
    model_path = MODELS_DIR / f"model_{skeleton_label.lower()}_kappa_{kappa:g}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "skeleton": skeleton_label,
            "kappa": kappa,
            "params": params,
            "hypparams": hypparams,
            "decoded_states": decoded_states,
            "decoded_states_dict": decoded_states_dict,
            "batch_keys": batch_keys,
            "occupancies_by_animal": occupancies_by_animal,
            "perm_results": perm_results,
        }, f)

    summary_record = {
        "Skeleton": skeleton_label,
        "Stickiness_Kappa": float(kappa),
        "Median_Bout_Sec": median_dur_sec,
        "Mean_Bout_Sec": mean_dur_sec,
        "Max_Bout_Min": max_dur_min,
        "Saline_TV_Obs": perm_results["obs_saline_tv"],
        "Saline_TV_p_exact": perm_results["p_exact_saline_tv"],
        "Saline_L2_Obs": perm_results["obs_saline_l2"],
        "Saline_L2_p_exact": perm_results["p_exact_saline_l2"],
        "Drug_Delta_L2_Obs": perm_results["obs_drug_delta_l2"],
        "Drug_Delta_p_exact": perm_results["p_exact_drug_delta"],
        "Fit_Duration_Min": fit_dur_min,
    }
    return summary_record


def plot_fast_ethogram(decoded_states, mask, metadata_dict, batch_keys, n_states, out_path, title=""):
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True, sharey=False)
    cmap = plt.cm.get_cmap("tab10", n_states)

    cohort_order = [
        ("WT_Saline", "WT - Saline (8 mice)"),
        ("WT_Amphetamine", "WT - Amphetamine (8 mice)"),
        ("KI_Saline", "DAT-Het KI - Saline (7 mice)"),
        ("KI_Amphetamine", "DAT-Het KI - Amphetamine (7 mice)"),
    ]

    session_keys = list(batch_keys)
    max_time_min = 60.0
    max_frames = int(max_time_min * 60 * 40)
    ds_factor = 10
    n_cols = max_frames // ds_factor

    for ax_idx, (cohort_id, cohort_title) in enumerate(cohort_order):
        ax = axes[ax_idx]
        matching_indices = [
            i for i, k in enumerate(session_keys)
            if metadata_dict[k]["cohort"] == cohort_id
        ]
        n_rows = len(matching_indices)
        grid = np.full((n_rows, n_cols), -1, dtype=int)
        y_labels = []

        for row_pos, s_idx in enumerate(matching_indices):
            s_key = session_keys[s_idx]
            meta = metadata_dict[s_key]
            n_valid = min(int(mask[s_idx].sum()), max_frames)
            states_seq = decoded_states[s_idx, :n_valid:ds_factor]
            grid[row_pos, :len(states_seq)] = states_seq
            y_labels.append(f"#{meta['animal_id']}")

        masked_grid = np.ma.masked_equal(grid, -1)
        ax.imshow(
            masked_grid, aspect="auto", cmap=cmap, vmin=0, vmax=n_states - 1,
            extent=[0, max_time_min, n_rows - 0.5, -0.5], interpolation="nearest"
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Animal", fontsize=11, fontweight="bold")
        ax.set_title(cohort_title, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlim(0, max_time_min)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Time Since Return to Arena (Minutes)", fontsize=12, fontweight="bold")
    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved ethogram to: {out_path.name}", flush=True)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABS_DIR.mkdir(parents=True, exist_ok=True)

    # Systematic grid of stickiness values:
    # 1. 1e4: moderate stickiness (expected ~2-5s bouts)
    # 2. 5e4: higher stickiness (expected ~10-25s bouts, matching Weinreb et al. 2026 self-directed tasks)
    # 3. 5e5: original baseline hyper-sticky prior (monolithic session locking)
    kappas = [1e4, 5e4, 5e5]

    all_records = []

    print("=" * 80)
    print("🚀 COMPARATIVE SHMOSEQ CALIBRATION: 8KP VS. 17KP ACROSS STICKINESS REGIMES")
    print(f"   Stickiness grid: {kappas}")
    print(f"   Output Directory: {OUT_DIR}")
    print("=" * 80, flush=True)

    # 1. Evaluate 8KP Pipeline
    for k_val in kappas:
        rec = fit_and_evaluate(DATA_8KP, "8KP", k_val, n_states=8, num_iters=50, seed_val=0)
        all_records.append(rec)

    # 2. Evaluate 17KP Pipeline
    for k_val in kappas:
        rec = fit_and_evaluate(DATA_17KP, "17KP", k_val, n_states=8, num_iters=50, seed_val=0)
        all_records.append(rec)

    df_all = pd.DataFrame(all_records)
    out_table = TABS_DIR / "shmoseq_stickiness_8kp_vs_17kp_comparison.csv"
    df_all.to_csv(out_table, index=False)

    print("\n" + "=" * 80)
    print("🎉 FULL SHMOSEQ STICKINESS COMPARISON COMPLETED:")
    print(df_all.to_string(index=False))
    print(f"\nSaved comprehensive table to:\n  {out_table}")
    print("=" * 80)


if __name__ == "__main__":
    main()
