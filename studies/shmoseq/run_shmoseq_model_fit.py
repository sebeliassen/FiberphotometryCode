#!/usr/bin/env python3
"""
run_shmoseq_model_fit.py
=============================================================================
Exploratory Condition-Blind State Discovery via official state_moseq HHMM.

Methodology:
- Operates on full continuous post-injection syllable streams across all 30 sessions.
- Fits an 8-state Bayesian Hierarchical HMM with sticky Dirichlet prior (kappa = 5e5)
  using GPU-accelerated Gibbs sampling + gradient descent emissions.
- Tracks convergence diagnostics (log-joint probability and gradient descent losses).
- Decodes the final state trajectory using smoothed posterior probabilities
  (sm.hhmm_efficient.smoothed_states) rather than raw last Gibbs sample.
- Saves model parameters, decoded state trajectories, posterior probabilities,
  and diagnostic plots.
- Logs every iteration live to both stdout and:
  syllable_generation/analysis/shmoseq/output/shmoseq_fit.log
  for seamless monitoring via `tail -f`.
=============================================================================
"""

import os
import pickle
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output")
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, "models", "selected_exploratory_model.pkl")
FIG_OUTPUT = os.path.join(OUTPUT_DIR, "figures", "fig2_shmoseq_fitting_convergence.png")
LOG_FILE = os.path.join(OUTPUT_DIR, "shmoseq_fit.log")

def log_print(msg):
    """Print to stdout and append to log file with flush for live tail -f."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def main():
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_OUTPUT), exist_ok=True)

    # Initialize fresh log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== shMoSeq GPU Fitting Live Log [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")

    log_print("=" * 80)
    log_print("SHMOSEQ CONDITION-BLIND MODEL FITTING (50 GIBBS ITERATIONS ON GPU)")
    log_print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    log_print(f"Live Log File: {LOG_FILE}")
    log_print("=" * 80)

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    n_syllables = bundle["n_syllables"]
    fps = bundle.get("fps", 40.0125)

    log_print(f"Loaded {len(syllables_dict)} sessions. Batching at full sequence length...")
    seqs, mask, batch_metadata = sm.batch(syllables_dict)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }

    n_seqs, n_timesteps = data["syllables"].shape
    log_print(f"Data tensor shape: {n_seqs} sequences x {n_timesteps} timesteps")
    log_print(f"Valid frame fraction: {float(mask.mean()):.4f}")

    # Define Hyperparameters
    N_STATES = 8
    TRANS_KAPPA = 5e5  # Stickiness prior calibrated for 40 Hz data
    NUM_ITERS = 50
    RANDOM_SEED = 0

    hypparams = {
        "n_states": N_STATES,
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": 1.0,
        "trans_beta": 1.0,
        "trans_kappa": TRANS_KAPPA,
        "n_syllables": n_syllables,
        "emission_gd_iters": 100,  # 100 GD steps per Gibbs iteration for fast convergence
        "emission_gd_lr": 5e-3,
        "random_seed": RANDOM_SEED,
    }

    log_print("\nModel Configuration:")
    log_print(f"  - Number of High-Level States (K): {N_STATES}")
    log_print(f"  - Number of Syllables: {n_syllables}")
    log_print(f"  - Stickiness Prior (trans_kappa): {TRANS_KAPPA:.1e}")
    log_print(f"  - Total Gibbs Iterations: {NUM_ITERS}")
    log_print(f"  - Emission GD Iterations: {hypparams['emission_gd_iters']}")
    log_print(f"  - Random Seed: {RANDOM_SEED}")

    # Initialize hidden states and parameters
    seed = jr.PRNGKey(RANDOM_SEED)
    seed, subseed = jr.split(seed)
    states = jr.randint(subseed, data["syllables"].shape, 0, N_STATES)
    params = None

    log_probs = []
    all_gd_losses = []
    iter_times = []

    log_print("\nBeginning GPU Gibbs sampling loop...")
    t_start_total = time.time()

    for it in range(NUM_ITERS):
        t0 = time.time()
        seed, subseed = jr.split(seed)

        # 1. Resample parameters via gradient descent on emissions + Dirichlet transitions
        params, losses = sm.hhmm_efficient.resample_params(subseed, data, states, hypparams, params)

        # 2. Resample states via message passing
        seed, subseed = jr.split(seed)
        states, marginal_loglik = sm.hhmm_efficient.resample_states(subseed, data, params, parallel=False)

        # 3. Compute log-joint probability
        log_param_p = sm.hhmm_efficient.log_params_prob(params, hypparams)
        current_log_joint = float(log_param_p + marginal_loglik)

        elapsed = time.time() - t0
        iter_times.append(elapsed)
        log_probs.append(current_log_joint)
        all_gd_losses.append(np.array(losses))

        log_print(f"  Iter {it+1:02d}/{NUM_ITERS}: Log-Joint = {current_log_joint:.2f} | MargLL = {marginal_loglik.item():.2f} | GD Loss = {losses[-1]:.4f} | Time = {elapsed:.1f}s")

    total_duration = time.time() - t_start_total
    log_print(f"\nGibbs fitting completed in {total_duration/60:.2f} minutes (avg {np.mean(iter_times):.1f}s/iter).")

    # -----------------------------------------------------------------------
    # Posterior State Decoding (Smoothed States)
    # -----------------------------------------------------------------------
    log_print("\nDecoding posterior state distribution (sm.hhmm_efficient.smoothed_states)...")
    t_dec_start = time.time()
    smoothed_probs = sm.hhmm_efficient.smoothed_states(data, params, parallel=False)
    decoded_states = jnp.argmax(smoothed_probs, axis=-1)
    log_print(f"Posterior decoding completed in {time.time() - t_dec_start:.2f}s.")

    # Unbatch back to dictionary per session
    log_print("Unbatching decoded states and posterior probabilities...")
    states_dict = sm.unbatch(np.array(decoded_states), *batch_metadata)

    # Unbatch smoothed probabilities
    probs_dict = {}
    smoothed_probs_np = np.array(smoothed_probs)
    keys_out, bounds = batch_metadata
    for key in set(keys_out):
        idx = np.where(keys_out == key)[0]
        idx_sorted = sorted(idx, key=lambda i: bounds[i][0])
        seq_probs = []
        for i in idx_sorted:
            start, end = bounds[i]
            length = end - start
            seq_probs.append(smoothed_probs_np[i, :length, :])
        probs_dict[key] = np.concatenate(seq_probs, axis=0)

    # Calculate state usage
    all_decoded = np.hstack(list(states_dict.values()))
    state_counts = np.bincount(all_decoded, minlength=N_STATES)
    state_proportions = state_counts / state_counts.sum()
    log_print("\nInferred State Proportions across all 30 sessions:")
    for k in range(N_STATES):
        log_print(f"  State {k}: {state_proportions[k]*100:.2f}% ({state_counts[k]} frames)")

    # Save the selected exploratory model
    save_bundle = {
        "params": jax.device_get(params),
        "hypparams": hypparams,
        "log_probs": np.array(log_probs),
        "gd_losses": all_gd_losses,
        "states_dict": states_dict,
        "probs_dict": probs_dict,
        "metadata_dict": metadata_dict,
        "batch_metadata": batch_metadata,
        "state_proportions": state_proportions,
        "fps": fps,
        "total_duration_sec": total_duration,
    }

    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(save_bundle, f)
    log_print(f"\nSaved selected exploratory model to: {MODEL_OUTPUT}")

    # -----------------------------------------------------------------------
    # Convergence Diagnostic Figure
    # -----------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), dpi=300)

    # 1. Log-Joint Probability
    axs[0].plot(log_probs, color="#1f77b4", lw=2)
    axs[0].set_title("A. Log Joint Probability (Gibbs)", fontweight="bold")
    axs[0].set_xlabel("Gibbs Iteration", fontweight="bold")
    axs[0].set_ylabel("Log Joint Probability", fontweight="bold")
    axs[0].grid(True, ls="--", alpha=0.3)

    # 2. Final Emission GD Loss
    final_losses = [l[-1] for l in all_gd_losses]
    axs[1].plot(final_losses, color="#2ca02c", lw=2)
    axs[1].set_title("B. Syllable Emission Loss (GD)", fontweight="bold")
    axs[1].set_xlabel("Gibbs Iteration", fontweight="bold")
    axs[1].set_ylabel("Loss", fontweight="bold")
    axs[1].grid(True, ls="--", alpha=0.3)

    # 3. State Proportions
    axs[2].bar(np.arange(N_STATES), state_proportions * 100, color="#d62728", edgecolor="black", alpha=0.8)
    axs[2].set_title("C. High-Level State Occupancy", fontweight="bold")
    axs[2].set_xlabel("State ID", fontweight="bold")
    axs[2].set_ylabel("% of Total Frames", fontweight="bold")
    axs[2].set_xticks(np.arange(N_STATES))
    axs[2].grid(True, axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_OUTPUT, bbox_inches="tight")
    plt.close()
    log_print(f"Saved diagnostic figure to: {FIG_OUTPUT}")
    log_print("=" * 80)

if __name__ == "__main__":
    main()
