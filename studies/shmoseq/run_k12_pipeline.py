#!/usr/bin/env python3
"""
run_k12_pipeline.py
=============================================================================
Complete 12-State (K=12) shMoSeq Pipeline on Run 12 Sessions:
1. GPU Condition-Blind HHMM Fit (K=12, 50 Gibbs iterations, RTX 4090).
2. Posterior State Decoding & Episode Extraction.
3. Exact Combinatorial Animal-Level Permutation Test (6,435 partitions).
4. Mouse-Level Phenotypic Evaluation (WT vs DAT-Het, Saline vs Amphetamine).
5. All-Session State Timelines Ethogram Generation (0-60 min).
6. Objective Output Comparison: K=8 vs K=12.
=============================================================================
"""

import itertools
import os
import pickle
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
DATA_FILE = os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl")
OUTPUT_DIR = os.path.join(SHMOSEQ_DIR, "output")
MODEL_OUTPUT_K12 = os.path.join(OUTPUT_DIR, "models", "selected_exploratory_model_k12.pkl")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")

def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def l2_distance(p, q):
    return np.linalg.norm(p - q)

def js_distance(p, q):
    return float(jensenshannon(p, q))

def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends

def main():
    os.makedirs(os.path.dirname(MODEL_OUTPUT_K12), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIG_DIR, "timelines"), exist_ok=True)
    os.makedirs(os.path.join(TAB_DIR, "timelines"), exist_ok=True)

    print("=" * 80)
    print("🚀 RUNNING 12-STATE (K=12) SHMOSEQ PIPELINE ON RUN 12 DATASET")
    print(f"   JAX Backend: {jax.default_backend()} | GPU Devices: {jax.devices()}")
    print("=" * 80, flush=True)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    n_syllables = bundle["n_syllables"]
    fps = bundle.get("fps", 40.0125)

    N_STATES = 12
    TRANS_KAPPA = 5e5
    NUM_ITERS = 50
    RANDOM_SEED = 0

    # -------------------------------------------------------------------------
    # 1. GPU MODEL FITTING
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Batching 30 sessions for K={N_STATES} fitting...", flush=True)
    seqs, mask, batch_metadata = sm.batch(syllables_dict)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }

    hypparams = {
        "n_states": N_STATES,
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": 1.0,
        "trans_beta": 1.0,
        "trans_kappa": TRANS_KAPPA,
        "n_syllables": n_syllables,
        "emission_gd_iters": 100,
        "emission_gd_lr": 5e-3,
        "random_seed": RANDOM_SEED,
    }

    seed = jr.PRNGKey(RANDOM_SEED)
    seed, subseed = jr.split(seed)
    states = jr.randint(subseed, data["syllables"].shape, 0, N_STATES)
    params = None

    log_probs = []
    gd_losses = []
    t_start = time.time()

    print(f"Beginning GPU Gibbs sampling (50 iterations, K={N_STATES})...", flush=True)
    for it in range(NUM_ITERS):
        t0 = time.time()
        seed, subseed = jr.split(seed)
        params, losses = sm.hhmm_efficient.resample_params(subseed, data, states, hypparams, params)
        seed, subseed = jr.split(seed)
        states, marginal_loglik = sm.hhmm_efficient.resample_states(subseed, data, params, parallel=False)

        log_param_p = sm.hhmm_efficient.log_params_prob(params, hypparams)
        current_log_joint = float(log_param_p + marginal_loglik)

        log_probs.append(current_log_joint)
        gd_losses.append(losses[-1])
        elapsed = time.time() - t0

        if (it + 1) % 5 == 0 or it == 0 or it == NUM_ITERS - 1:
            print(f"  [K={N_STATES}] Iter {it+1:02d}/{NUM_ITERS}: Log-Joint = {current_log_joint:.2f} | GD Loss = {losses[-1]:.4f} | Time = {elapsed:.1f}s", flush=True)

    fit_duration = time.time() - t_start
    print(f"✓ K={N_STATES} Fitting completed in {fit_duration/60:.2f} min ({fit_duration:.1f}s).", flush=True)

    # Posterior Decoding
    print("Decoding posterior smoothed states...", flush=True)
    smoothed_probs = sm.hhmm_efficient.smoothed_states(data, params, parallel=False)
    decoded_states = jnp.argmax(smoothed_probs, axis=-1)
    states_dict_k12 = sm.unbatch(np.array(decoded_states), *batch_metadata)

    # Save model bundle
    save_bundle = {
        "params": jax.device_get(params),
        "hypparams": hypparams,
        "n_states": N_STATES,
        "log_probs": np.array(log_probs),
        "gd_losses": np.array(gd_losses),
        "states_dict": states_dict_k12,
        "metadata_dict": metadata_dict,
        "fps": fps,
        "fit_duration_sec": fit_duration,
    }
    with open(MODEL_OUTPUT_K12, "wb") as f:
        pickle.dump(save_bundle, f)
    print(f"Saved K={N_STATES} model to: {MODEL_OUTPUT_K12}", flush=True)

    # Overall State Proportions
    all_dec = np.hstack(list(states_dict_k12.values()))
    counts = np.bincount(all_dec, minlength=N_STATES)
    props = counts / counts.sum()
    print(f"\nInferred State Proportions across all 30 sessions (K={N_STATES}):", flush=True)
    for k in range(N_STATES):
        print(f"  State {k:02d}: {props[k]*100:5.2f}% ({counts[k]:,} frames)", flush=True)

    # -------------------------------------------------------------------------
    # 2. EXACT COMBINATORIAL ANIMAL-LEVEL PERMUTATION TEST (6,435 PARTITIONS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[2/5] EXACT COMBINATORIAL ANIMAL PERMUTATION TEST (K={N_STATES})")
    print("=" * 80, flush=True)

    animals = {}
    for s_key, st_seq in states_dict_k12.items():
        m = metadata_dict[s_key]
        aid = str(m["animal_id"])
        drug = m["drug"]
        geno = m["genotype"]

        if aid not in animals:
            animals[aid] = {"genotype": geno, "saline": None, "amph": None}

        c = np.bincount(st_seq, minlength=N_STATES)
        p = c / float(len(st_seq))
        if drug == "Saline":
            animals[aid]["saline"] = p
        elif drug == "Amphetamine":
            animals[aid]["amph"] = p

    animal_ids = sorted(list(animals.keys()), key=lambda x: (animals[x]["genotype"], x))
    N_ANIMALS = len(animal_ids)
    wt_ids = [aid for aid in animal_ids if animals[aid]["genotype"] == "WT"]
    ki_ids = [aid for aid in animal_ids if animals[aid]["genotype"] == "KI"]
    n_wt, n_ki = len(wt_ids), len(ki_ids)

    saline_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    amph_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    delta_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    genotypes = [animals[aid]["genotype"] for aid in animal_ids]

    for i, aid in enumerate(animal_ids):
        sal = animals[aid]["saline"]
        amp = animals[aid]["amph"]
        dlt = amp - sal
        saline_matrix[i] = sal
        amph_matrix[i] = amp
        delta_matrix[i] = dlt

    wt_mask = np.array([g == "WT" for g in genotypes])
    ki_mask = np.array([g == "KI" for g in genotypes])

    wt_sal_mean = saline_matrix[wt_mask].mean(axis=0)
    ki_sal_mean = saline_matrix[ki_mask].mean(axis=0)

    obs_sal_tv = tv_distance(wt_sal_mean, ki_sal_mean)
    obs_sal_l2 = l2_distance(wt_sal_mean, ki_sal_mean)
    obs_sal_js = js_distance(wt_sal_mean, ki_sal_mean)

    wt_dlt_mean = delta_matrix[wt_mask].mean(axis=0)
    ki_dlt_mean = delta_matrix[ki_mask].mean(axis=0)
    obs_dlt_l2 = l2_distance(wt_dlt_mean, ki_dlt_mean)
    obs_dlt_l1 = np.sum(np.abs(wt_dlt_mean - ki_dlt_mean))

    print(f"Observed Separation (K={N_STATES}):")
    print(f"  - Saline TV Distance: {obs_sal_tv:.4f} ({obs_sal_tv*100:.1f}% shifted)")
    print(f"  - Saline L2 Distance: {obs_sal_l2:.4f}")
    print(f"  - Saline JS Distance: {obs_sal_js:.4f}")
    print(f"  - Drug Delta L2 Dist: {obs_dlt_l2:.4f}")

    all_indices = list(range(N_ANIMALS))
    combos = list(itertools.combinations(all_indices, n_wt))
    N_COMBOS = len(combos)

    perm_sal_tv = np.zeros(N_COMBOS)
    perm_sal_l2 = np.zeros(N_COMBOS)
    perm_sal_js = np.zeros(N_COMBOS)
    perm_dlt_l2 = np.zeros(N_COMBOS)
    perm_state_diffs = np.zeros((N_COMBOS, N_STATES))
    perm_dlt_state_diffs = np.zeros((N_COMBOS, N_STATES))

    all_set = set(all_indices)
    for idx, combo in enumerate(combos):
        idx_a = np.array(combo)
        idx_b = np.array(list(all_set - set(combo)))

        mean_a_sal = saline_matrix[idx_a].mean(axis=0)
        mean_b_sal = saline_matrix[idx_b].mean(axis=0)
        perm_sal_tv[idx] = tv_distance(mean_a_sal, mean_b_sal)
        perm_sal_l2[idx] = l2_distance(mean_a_sal, mean_b_sal)
        perm_sal_js[idx] = js_distance(mean_a_sal, mean_b_sal)
        perm_state_diffs[idx] = mean_a_sal - mean_b_sal

        mean_a_dlt = delta_matrix[idx_a].mean(axis=0)
        mean_b_dlt = delta_matrix[idx_b].mean(axis=0)
        perm_dlt_l2[idx] = l2_distance(mean_a_dlt, mean_b_dlt)
        perm_dlt_state_diffs[idx] = mean_a_dlt - mean_b_dlt

    p_val_tv = float(np.mean(perm_sal_tv >= obs_sal_tv))
    p_val_l2 = float(np.mean(perm_sal_l2 >= obs_sal_l2))
    p_val_js = float(np.mean(perm_sal_js >= obs_sal_js))
    p_val_dlt = float(np.mean(perm_dlt_l2 >= obs_dlt_l2))

    print(f"\nExact Combinatorial Permutation Results (K={N_STATES}):")
    print(f"  - Saline TV exact p-value: {p_val_tv:.5f} (Null median: {np.median(perm_sal_tv):.4f})")
    print(f"  - Saline L2 exact p-value: {p_val_l2:.5f} (Null median: {np.median(perm_sal_l2):.4f})")
    print(f"  - Saline JS exact p-value: {p_val_js:.5f} (Null median: {np.median(perm_sal_js):.4f})")
    print(f"  - Delta  L2 exact p-value: {p_val_dlt:.5f} (Null median: {np.median(perm_dlt_l2):.4f})")

    # Per-state exact p-values
    state_results = []
    for k in range(N_STATES):
        obs_diff = wt_sal_mean[k] - ki_sal_mean[k]
        p_state = float(np.mean(np.abs(perm_state_diffs[:, k]) >= np.abs(obs_diff)))

        obs_dlt_diff = wt_dlt_mean[k] - ki_dlt_mean[k]
        p_dlt_state = float(np.mean(np.abs(perm_dlt_state_diffs[:, k]) >= np.abs(obs_dlt_diff)))

        state_results.append({
            "state_id": k,
            "state_name": f"Z_{k}",
            "wt_saline_mean": float(wt_sal_mean[k]),
            "ki_saline_mean": float(ki_sal_mean[k]),
            "saline_diff": float(obs_diff),
            "saline_exact_p": p_state,
            "wt_delta_mean": float(wt_dlt_mean[k]),
            "ki_delta_mean": float(ki_dlt_mean[k]),
            "delta_diff": float(obs_dlt_diff),
            "delta_exact_p": p_dlt_state,
        })

    df_k12_states = pd.DataFrame(state_results)

    # Benjamini-Hochberg FDR
    df_k12_states = df_k12_states.sort_values("saline_exact_p").reset_index(drop=True)
    df_k12_states["fdr_rank"] = np.arange(1, N_STATES + 1)
    df_k12_states["fdr_threshold"] = (df_k12_states["fdr_rank"] / N_STATES) * 0.05
    df_k12_states["saline_fdr_q_value"] = df_k12_states["saline_exact_p"] * N_STATES / df_k12_states["fdr_rank"]
    # Monotonicity
    for i in range(len(df_k12_states) - 2, -1, -1):
        df_k12_states.loc[i, "saline_fdr_q_value"] = min(
            df_k12_states.loc[i, "saline_fdr_q_value"],
            df_k12_states.loc[i + 1, "saline_fdr_q_value"]
        )
    df_k12_states["saline_fdr_sig"] = df_k12_states["saline_fdr_q_value"] < 0.05
    df_k12_states = df_k12_states.sort_values("state_id").reset_index(drop=True)

    df_k12_states.to_csv(os.path.join(TAB_DIR, "per_state_exact_permutation_tests_k12.csv"), index=False)

    print("\nPer-State Summary (K=12):")
    print(df_k12_states[["state_name", "saline_diff", "saline_exact_p", "saline_fdr_q_value", "saline_fdr_sig"]])

    # -------------------------------------------------------------------------
    # 3. GENERATE ALL-SESSION STATE TIMELINES FIGURE (K=12)
    # -------------------------------------------------------------------------
    print("\n[3/5] Generating All-Session State Timelines (0-60 min, K=12)...", flush=True)
    animals_order = wt_ids + ki_ids
    cmap_k12 = plt.cm.tab20(np.linspace(0, 1, N_STATES))

    fig_time, axs_time = plt.subplots(len(animals_order), 2, figsize=(16, 12), dpi=200, sharex=True)
    for row_idx, aid in enumerate(animals_order):
        gt = animals[aid]["genotype"]
        for col_idx, drug in enumerate(["Saline", "Amphetamine"]):
            ax = axs_time[row_idx, col_idx]
            sess_key = f"{aid}_{drug}"
            seq = states_dict_k12[sess_key]
            st, dur, starts, ends = run_length_encoding(seq)

            for b_i in range(len(st)):
                k = st[b_i]
                t_start_min = starts[b_i] / fps / 60.0
                t_end_min = ends[b_i] / fps / 60.0
                ax.axvspan(t_start_min, t_end_min, color=cmap_k12[k], alpha=0.85, edgecolor="none")

            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"{gt} #{aid}", fontsize=9, fontweight="bold", rotation=0, labelpad=35, va="center")
            if row_idx == 0:
                ax.set_title(f"{drug} Sessions (K=12)", fontsize=12, fontweight="bold")
            if row_idx == len(animals_order) - 1:
                ax.set_xlabel("Time Post-Injection (minutes)", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 60)

    handles = [matplotlib.patches.Patch(color=cmap_k12[k], label=f"Z_{k}") for k in range(N_STATES)]
    fig_time.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)
    plt.tight_layout()

    fig_time_path = os.path.join(FIG_DIR, "timelines", "fig_session_state_timelines_k12.png")
    plt.savefig(fig_time_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_time_path}", flush=True)

    # -------------------------------------------------------------------------
    # 4. COMPARISON: K=8 VS K=12
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 DIRECT COMPARISON: K=8 MODEL VS. K=12 MODEL")
    print("=" * 80)
    print("Metric                               | K=8 Model         | K=12 Model")
    print("-" * 75)
    print(f"Multivariate Saline TV Distance      | 0.3297 (p=0.754)  | {obs_sal_tv:.4f} (p={p_val_tv:.5f})")
    print(f"Multivariate Saline L2 Distance      | 0.3022 (p=0.778)  | {obs_sal_l2:.4f} (p={p_val_l2:.5f})")
    print(f"Multivariate Drug Delta L2 Distance  | 0.3115 (p=0.586)  | {obs_dlt_l2:.4f} (p={p_val_dlt:.5f})")

    # Check if any state achieves p < 0.05 or FDR q < 0.05
    k12_sig_raw = df_k12_states[df_k12_states["saline_exact_p"] < 0.05]
    k12_sig_fdr = df_k12_states[df_k12_states["saline_fdr_sig"]]
    print(f"Nominally Sig States (p < 0.05)      | 1 (State 0: p=0.028)| {len(k12_sig_raw)} states: {[r.state_name for _, r in k12_sig_raw.iterrows()]}")
    print(f"FDR Sig States (q < 0.05)            | 0 states          | {len(k12_sig_fdr)} states")
    print("=" * 80)

if __name__ == "__main__":
    main()
