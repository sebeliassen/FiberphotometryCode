#!/usr/bin/env python3
"""
run_lagged_mutual_information.py
=============================================================================
Estimate empirical and surrogate lagged mutual information curves across
lags from 1 second to 300 seconds using official state_moseq functions.

Methodology:
- Operates on full continuous post-injection syllable sequences.
- Calculates empirical lagged MI, simulated 1st-order Markov chain null MI,
  and cross-session shuffled baseline MI.
- Visualizes recording-level trajectories and cohort mean ± SEM.
=============================================================================
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JAX environment configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")
OUTPUT_FIG = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures", "fig1_timescale_lagged_mi.png")
OUTPUT_DATA = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "lagged_mi_results_1000s.pkl")

def main():
    print("=" * 80)
    print("RUNNING LAGGED MUTUAL INFORMATION TIMESCALE DIAGNOSTIC (UP TO 10^3 SECONDS)")
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print("=" * 80)
    
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
        
    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    fps = bundle.get("fps", 40.0125)
    
    print(f"Loaded {len(syllables_dict)} sessions. Formatting full-length batch...")
    
    # Batch sequences at full recording length so 1000s lag is completely unconstrained
    seqs, mask, metadata = sm.batch(syllables_dict)
    seqs = jnp.array(seqs)
    mask = jnp.array(mask)
    
    print(f"Batch shape: {seqs.shape}, mask valid fraction: {float(mask.mean()):.4f}")
    
    # Define lags from 1s to 1000s (logarithmically spaced across 35 lags)
    num_lags = 35
    min_lag = int(fps * 1.0)
    max_lag = int(fps * 1000.0)
    lags = jnp.logspace(jnp.log10(min_lag), jnp.log10(max_lag), num_lags).astype(int)
    lag_seconds = np.array(lags) / fps
    
    if os.path.exists(OUTPUT_DATA):
        print(f"Loading precomputed lagged MI from: {OUTPUT_DATA}")
        with open(OUTPUT_DATA, "rb") as f:
            mi_bundle = pickle.load(f)
        real_mis = mi_bundle["real_mis"]
        markov_mis = mi_bundle["markov_mis"]
        shuff_mis = mi_bundle["shuff_mis"]
        lag_seconds = mi_bundle["lag_seconds"]
    else:
        print(f"Computing lagged MI across {num_lags} lags ({lag_seconds[0]:.2f}s to {lag_seconds[-1]:.2f}s = 10^3s)...")
        real_mis, markov_mis, shuff_mis = sm.lagged_mutual_information(seqs, mask, lags)
        real_mis = np.array(real_mis)
        markov_mis = np.array(markov_mis)
        shuff_mis = np.array(shuff_mis)
        mi_bundle = {
            "lags": np.array(lags),
            "lag_seconds": lag_seconds,
            "real_mis": real_mis,
            "markov_mis": markov_mis,
            "shuff_mis": shuff_mis,
            "keys": list(syllables_dict.keys()),
            "metadata_dict": metadata_dict,
        }
        with open(OUTPUT_DATA, "wb") as f:
            pickle.dump(mi_bundle, f)
        print(f"Saved computed MI arrays to: {OUTPUT_DATA}")
    print(f"Saved computed MI arrays to: {OUTPUT_DATA}")
    
    # Generate Publication Figure (2-Panel: Linear Y scaled to data & Focused Log-Log)
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    colors = {"real": "#D95F02", "markov": "#2C2C2C", "shuff": "#999999"}
    labels = {"real": "Real Data (Continuous Syllables)", "markov": "1st-Order Markov Chain Null", "shuff": "Shuffled Baseline"}
    
    # Panel A: Semi-log (Linear Y scaled to real data: 0 to 1.6 bits)
    ax_a = axs[0]
    for mis, key, color in [(real_mis, "real", colors["real"]), 
                            (markov_mis, "markov", colors["markov"]), 
                            (shuff_mis, "shuff", colors["shuff"])]:
        for s_idx in range(len(mis)):
            ax_a.plot(lag_seconds, np.clip(mis[s_idx], 0, None), color=color, alpha=0.12, linewidth=0.8)
        mean = mis.mean(axis=0)
        sem = mis.std(axis=0) / np.sqrt(len(mis))
        ax_a.plot(lag_seconds, mean, color=color, linewidth=2.4, label=labels[key])
        ax_a.fill_between(lag_seconds, np.maximum(mean - sem, 0), mean + sem, facecolor=color, alpha=0.25)
        
    ax_a.set_xscale("log")
    ax_a.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_a.set_ylim([0.0, 1.6])
    ax_a.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Mutual Information $I(S_t; S_{t+\\tau})$ (bits)", fontsize=11, fontweight="bold")
    ax_a.set_title("A. Linear Y-Scale (Scaled to Data: 0.0 - 1.6 bits)", fontsize=12, fontweight="bold")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)
    
    # Panel B: Focused Log-Log (Y-axis clamped to 0.1 to 2.0 bits)
    ax_b = axs[1]
    for mis, key, color in [(real_mis, "real", colors["real"]), 
                            (markov_mis, "markov", colors["markov"]), 
                            (shuff_mis, "shuff", colors["shuff"])]:
        for s_idx in range(len(mis)):
            ax_b.plot(lag_seconds, np.clip(mis[s_idx], 0.05, None), color=color, alpha=0.12, linewidth=0.8)
        mean = np.clip(mis.mean(axis=0), 0.05, None)
        sem = mis.std(axis=0) / np.sqrt(len(mis))
        ax_b.plot(lag_seconds, mean, color=color, linewidth=2.4, label=labels[key])
        ax_b.fill_between(lag_seconds, np.maximum(mean - sem, 0.05), mean + sem, facecolor=color, alpha=0.25)
        
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim([lag_seconds[0], lag_seconds[-1]])
    ax_b.set_ylim([0.1, 2.0])
    ax_b.set_xlabel("Lag $\\tau$ (seconds, log scale)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Mutual Information (bits, log scale)", fontsize=11, fontweight="bold")
    ax_b.set_title("B. Focused Log-Log Scale (Clamped 0.1 - 2.0 bits)", fontsize=12, fontweight="bold")
    ax_b.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_b.legend(loc="upper right", frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    
    print(f"Figure saved to: {OUTPUT_FIG}")
    print("=" * 80)

if __name__ == "__main__":
    main()
