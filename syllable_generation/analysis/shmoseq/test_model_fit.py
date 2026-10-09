#!/usr/bin/env python3
"""
test_model_fit.py
=============================================================================
Quick test of state_moseq model initialization and 3 Gibbs iterations
to measure exact runtime per iteration on GPU and inspect decoding functions.
=============================================================================
"""

import os
import sys
import pickle
import time
import inspect

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
DATA_FILE = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data", "shmoseq_input_data.pkl")

def main():
    print("=" * 80)
    print("TESTING SHMOSEQ GIBBS SAMPLING & DECODING ON GPU")
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print("=" * 80)
    
    # Check signatures
    print("\nsm.hhmm_efficient.smoothed_states signature:")
    print(inspect.signature(sm.hhmm_efficient.smoothed_states))
    print("\nsm.hhmm_efficient.hmm_posterior_mode signature:")
    if hasattr(sm.hhmm_efficient, "hmm_posterior_mode"):
        print(inspect.signature(sm.hhmm_efficient.hmm_posterior_mode))
        
    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)
        
    syllables_dict = bundle["syllables_dict"]
    n_syllables = bundle["n_syllables"]
    
    # Batch sequences (full length)
    seqs, mask, metadata = sm.batch(syllables_dict)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }
    
    hypparams = {
        "n_states": 8,
        "emission_base_sigma": 1.0,
        "emission_biases_sigma": 1.0,
        "trans_beta": 1.0,
        "trans_kappa": 1e5,
        "n_syllables": n_syllables,
        "emission_gd_iters": 500,
        "emission_gd_lr": 5e-3,
        "random_seed": 0,
    }
    
    print(f"\nInitializing model with K={hypparams['n_states']}, N_sylls={n_syllables}...")
    seed = jr.PRNGKey(hypparams["random_seed"])
    states = jr.randint(seed, data["syllables"].shape, 0, hypparams["n_states"])
    params = None
    
    # Run 3 test iterations
    print("\nRunning 3 test Gibbs iterations to time performance...")
    t0 = time.time()
    for it in range(3):
        t_it_start = time.time()
        seed, subseed = jr.split(seed)
        params, losses = sm.hhmm_efficient.resample_params(subseed, data, states, hypparams, params)
        states, marginal_loglik = sm.hhmm_efficient.resample_states(seed, data, params, parallel=False)
        t_it_end = time.time()
        print(f"  Iteration {it+1}: time = {t_it_end - t_it_start:.2f}s, marginal loglik = {marginal_loglik.item():.2f}")
        
    total_t = time.time() - t0
    avg_per_iter = total_t / 3.0
    print(f"\nAverage time per iteration: {avg_per_iter:.2f}s")
    print(f"Estimated 100 iterations: {avg_per_iter * 100 / 60:.2f} minutes")
    print(f"Estimated 3 seeds x 3 kappas (9 runs): {avg_per_iter * 100 * 9 / 60:.2f} minutes")
    
    # Test decoding
    print("\nTesting posterior state smoothing...")
    probs = sm.hhmm_efficient.smoothed_states(data, params)
    print(f"Smoothed state probabilities shape: {probs.shape}")
    decoded_states = jnp.argmax(probs, axis=-1)
    print(f"Decoded states shape: {decoded_states.shape}, unique states: {np.unique(decoded_states)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
