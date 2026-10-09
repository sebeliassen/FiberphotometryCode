#!/usr/bin/env python3
"""
run_capacity_experiments.py
=============================================================================
Runs targeted shMoSeq model fits to test state capacity and condition diversity:
  1. WT Saline only (N=8 sessions, K=8, Seed 0)
  2. Matched Mixed Subset (N=8 sessions, 2 of each cohort, K=8, Seed 0)
  3. Full Pooled Dataset (N=30 sessions, K=12, Seed 0)

Decodes states via posterior smoothing (sm.hhmm_efficient.smoothed_states)
and saves structured model checkpoints.
=============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "models", "capacity")
LOG_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "capacity_experiments.log")

def log_print(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def fit_shmoseq(syllables_dict_sub, metadata_dict_sub, n_syllables, n_states, num_iters=50, kappa=5e5, seed_int=0, fit_name="fit", fps=40.0125):
    log_print("\n" + "=" * 80)
    log_print(f"STARTING FIT: {fit_name}")
    log_print(f"  Sessions: {len(syllables_dict_sub)} | States (K): {n_states} | Kappa: {kappa:.1e} | Seed: {seed_int}")
    log_print("=" * 80)
    
    seqs, mask, batch_metadata = sm.batch(syllables_dict_sub)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }
    n_seqs, n_timesteps = data["syllables"].shape
    log_print(f"  Data shape: {n_seqs} seqs x {n_timesteps} frames | Valid frac: {float(mask.mean()):.4f}")
    
    hypparams = {
        "n_states": n_states,
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": 1.0,
        "trans_beta": 1.0,
        "trans_kappa": kappa,
        "n_syllables": n_syllables,
        "emission_gd_iters": 100,
        "emission_gd_lr": 5e-3,
        "random_seed": seed_int,
    }
    
    seed = jr.PRNGKey(seed_int)
    seed, subseed = jr.split(seed)
    states = jr.randint(subseed, data["syllables"].shape, 0, n_states)
    params = None
    
    log_probs = []
    gd_losses = []
    t_start = time.time()
    
    for it in range(num_iters):
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
        
        if (it + 1) % 10 == 0 or it == 0 or it == num_iters - 1:
            log_print(f"  [{fit_name}] Iter {it+1:02d}/{num_iters}: Log-Joint = {current_log_joint:.2f} | GD Loss = {losses[-1]:.4f} | Time = {elapsed:.1f}s")
            
    fit_duration = time.time() - t_start
    log_print(f"  Fitting completed in {fit_duration/60:.2f} min ({fit_duration:.1f}s).")
    
    # Posterior Decoding
    log_print(f"  Decoding posterior smoothed states...")
    smoothed_probs = sm.hhmm_efficient.smoothed_states(data, params, parallel=False)
    decoded_states = jnp.argmax(smoothed_probs, axis=-1)
    
    states_dict = sm.unbatch(np.array(decoded_states), *batch_metadata)
    
    # Calculate occupancy
    all_dec = np.hstack(list(states_dict.values()))
    counts = np.bincount(all_dec, minlength=n_states)
    props = counts / counts.sum()
    log_print(f"  State Proportions for {fit_name}:")
    for k in range(n_states):
        log_print(f"    State {k:02d}: {props[k]*100:5.2f}% ({counts[k]} frames)")
        
    save_bundle = {
        "fit_name": fit_name,
        "params": jax.device_get(params),
        "hypparams": hypparams,
        "n_states": n_states,
        "log_probs": np.array(log_probs),
        "gd_losses": np.array(gd_losses),
        "states_dict": states_dict,
        "metadata_dict": metadata_dict_sub,
        "batch_metadata": batch_metadata,
        "state_proportions": props,
        "fps": fps,
        "fit_duration_sec": fit_duration,
    }
    
    out_path = os.path.join(OUTPUT_DIR, f"{fit_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(save_bundle, f)
    log_print(f"  Saved model checkpoint to: {out_path}")
    return save_bundle

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Capacity & Diversity Experiments Live Log [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")
        
    log_print("=" * 80)
    log_print("CAPACITY & DIVERSITY EXPERIMENTAL SUITE")
    log_print(f"JAX Backend: {jax.default_backend()} | Devices: {jax.devices()}")
    log_print("=" * 80)
    
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
        
    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    n_syllables = bundle["n_syllables"]
    fps = bundle.get("fps", 40.0125)
    
    # -----------------------------------------------------------------------
    # 1. Prepare Subsets
    # -----------------------------------------------------------------------
    # Subset 1: WT Saline only (N=8)
    wt_sal_keys = [k for k, v in metadata_dict.items() if v["cohort"] == "WT_Saline"]
    syll_wt_sal = {k: syllables_dict[k] for k in wt_sal_keys}
    meta_wt_sal = {k: metadata_dict[k] for k in wt_sal_keys}
    log_print(f"\nSubset 1 (WT Saline): {len(wt_sal_keys)} sessions: {wt_sal_keys}")
    
    # Subset 2: Matched Mixed Subset (N=8 mice, 2 from each cohort)
    # Selected to span 8 distinct animals
    matched_keys = [
        "1775_Saline", "728_Saline",           # WT Saline
        "1767_Amphetamine", "1680_Amphetamine", # WT Amphetamine
        "1781_Saline", "1666_Saline",           # KI Saline
        "1771_Amphetamine", "1658_Amphetamine"  # KI Amphetamine
    ]
    syll_matched = {k: syllables_dict[k] for k in matched_keys}
    meta_matched = {k: metadata_dict[k] for k in matched_keys}
    log_print(f"\nSubset 2 (Matched Mixed): {len(matched_keys)} sessions: {matched_keys}")
    
    # -----------------------------------------------------------------------
    # 2. Run Fits
    # -----------------------------------------------------------------------
    # Fit A: WT Saline only, K=8
    fit_shmoseq(syll_wt_sal, meta_wt_sal, n_syllables, n_states=8, num_iters=50, kappa=5e5, seed_int=0, fit_name="wt_saline_k8_seed0", fps=fps)
    
    # Fit B: Matched Mixed Subset, K=8
    fit_shmoseq(syll_matched, meta_matched, n_syllables, n_states=8, num_iters=50, kappa=5e5, seed_int=0, fit_name="matched_mixed_k8_seed0", fps=fps)
    
    # Fit C: Full Pooled Data, K=12
    fit_shmoseq(syllables_dict, metadata_dict, n_syllables, n_states=12, num_iters=50, kappa=5e5, seed_int=0, fit_name="pooled_k12_seed0", fps=fps)
    
    log_print("\n" + "=" * 80)
    log_print("ALL CAPACITY & DIVERSITY EXPERIMENTAL FITS COMPLETED SUCCESSFULLY!")
    log_print("=" * 80)

if __name__ == "__main__":
    main()
