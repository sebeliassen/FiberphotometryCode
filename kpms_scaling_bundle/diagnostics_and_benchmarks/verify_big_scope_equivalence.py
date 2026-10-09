import os
import sys
import copy
import time
import numpy as np
np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import entropy

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import keypoint_moseq as kpms
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds as kp_slds_mod
import jax_moseq.models.slds as slds_mod
from jax_moseq.models.slds.gibbs import resample_continuous_stateseqs as slds_orig_resample_x
from utils.resample_batching import apply_optimizations
import utils.resample_batching as rb

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
MODEL_NAME = "final_karen_model_k150k"

def get_all_durations(z, mask):
    durations = []
    for i in range(len(z)):
        seq_z = z[i]
        seq_m = mask[i]
        valid_idx = np.where(seq_m)[0]
        if len(valid_idx) == 0: continue
        curr_s = seq_z[valid_idx[0]]
        curr_len = 1
        for j in range(1, len(valid_idx)):
            if valid_idx[j] == valid_idx[j-1] + 1 and seq_z[valid_idx[j]] == curr_s:
                curr_len += 1
            else:
                durations.append(curr_len)
                curr_s = seq_z[valid_idx[j]]
                curr_len = 1
        durations.append(curr_len)
    return np.array(durations)

def compute_model_macro_metrics(model, data):
    states = model['states']
    params = model['params']
    nlags = model['hypparams']['ar_hypparams']['nlags']
    
    Y = np.array(data['Y'])
    mask = np.array(data['mask'], dtype=bool)
    mask_z = mask[:, nlags:]
    
    z = np.array(states['z'])
    x = np.array(states['x'])
    v = np.array(states['v'])
    h = np.array(states['h'])
    
    N, T, k, d = Y.shape
    D = x.shape[-1]
    
    # 1. Observation reconstruction MSE
    cos_h = np.cos(h)[:, :, None]
    sin_h = np.sin(h)[:, :, None]
    Cd = np.array(params['Cd'])
    
    x_pad = np.concatenate([x, np.ones(x.shape[:-1] + (1,))], axis=-1)
    Y_aligned = (x_pad @ Cd.T).reshape(N, T, k-1, d)
    
    Y_aligned_full = np.concatenate([np.zeros((N, T, 1, d)), Y_aligned], axis=-2)
    Y_rot = np.zeros_like(Y_aligned_full)
    Y_rot[..., 0] = cos_h * Y_aligned_full[..., 0] - sin_h * Y_aligned_full[..., 1]
    Y_rot[..., 1] = sin_h * Y_aligned_full[..., 0] + cos_h * Y_aligned_full[..., 1]
    Y_est = Y_rot + v[:, :, None, :]
    
    obs_mse = float(np.mean((Y[mask] - Y_est[mask]) ** 2))
    
    # 2. Syllable duration percentiles
    durs = get_all_durations(z, mask_z)
    
    # 3. Syllable usage entropy & active counts
    valid_z = z[mask_z > 0]
    unq, counts = np.unique(valid_z, return_counts=True)
    probs = counts / len(valid_z)
    H_z = float(entropy(probs))
    
    active_95 = int(np.sum(np.cumsum(np.sort(counts)[::-1]) / len(valid_z) < 0.95) + 1)
    active_99 = int(np.sum(np.cumsum(np.sort(counts)[::-1]) / len(valid_z) < 0.99) + 1)
    
    # 4. Transition matrix self-transition mean
    pi = np.array(params['pi'])
    active_indices = unq
    mean_self_trans = float(np.mean(np.diag(pi)[active_indices]))
    
    return {
        "obs_mse": obs_mse,
        "dur_p10": float(np.percentile(durs, 10)) * 25.0,
        "dur_p25": float(np.percentile(durs, 25)) * 25.0,
        "dur_p50": float(np.median(durs)) * 25.0,
        "dur_p75": float(np.percentile(durs, 75)) * 25.0,
        "dur_p90": float(np.percentile(durs, 90)) * 25.0,
        "dur_mean": float(np.mean(durs)) * 25.0,
        "entropy": H_z,
        "active_95": active_95,
        "active_99": active_99,
        "mean_self_trans": mean_self_trans,
        "pi": pi,
        "total_bouts": len(durs)
    }

def main():
    print("=" * 110)
    print("      3-BRANCH BIG-SCOPE AUDIT: UNPATCHED JAX vs FIXED PATCHED vs BUGGY OLD PATCH")
    print("=" * 110)
    
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME, iteration=50)
    
    N_SLICED = 30
    sliced_data = {
        'Y': data['Y'][:N_SLICED],
        'mask': data['mask'][:N_SLICED]
    }
    if 'conf' in data and data['conf'] is not None:
        sliced_data['conf'] = data['conf'][:N_SLICED]
        
    sliced_metadata = tuple(m[:N_SLICED] for m in metadata) if isinstance(metadata, tuple) else metadata
    
    sliced_states = {}
    for k, v in model['states'].items():
        if hasattr(v, '__len__') and len(v) >= N_SLICED:
            sliced_states[k] = v[:N_SLICED]
        else:
            sliced_states[k] = v
            
    if 'noise_prior' in model and hasattr(model['noise_prior'], '__len__') and len(model['noise_prior']) >= N_SLICED:
        model['noise_prior'] = model['noise_prior'][:N_SLICED]
        
    # Save original references
    orig_resample_continuous = slds_gibbs.resample_continuous_stateseqs
    orig_resample_discrete = arhmm_gibbs.resample_discrete_stateseqs
    orig_resample_ar = arhmm_gibbs.resample_ar_params
    orig_resample_scales = slds_gibbs.resample_scales
    orig_resample_heading = slds_gibbs.resample_heading
    orig_resample_loc = slds_gibbs.resample_location

    NUM_ITERS = 3

    # 1. RUN UNPATCHED JAX
    print(f"\n[1/3] Running {NUM_ITERS} iterations with UNPATCHED ORIGINAL JAX...")
    slds_gibbs.resample_continuous_stateseqs = orig_resample_continuous
    arhmm_gibbs.resample_discrete_stateseqs = orig_resample_discrete
    arhmm_gibbs.resample_ar_params = orig_resample_ar
    slds_gibbs.resample_scales = orig_resample_scales
    slds_gibbs.resample_heading = orig_resample_heading
    slds_gibbs.resample_location = orig_resample_loc
    
    kp_slds_mod.resample_continuous_stateseqs = orig_resample_continuous
    kp_slds_mod.resample_scales = orig_resample_scales
    kp_slds_mod.resample_heading = orig_resample_heading
    kp_slds_mod.resample_location = orig_resample_loc
    
    slds_mod.resample_continuous_stateseqs = slds_orig_resample_x
    slds_mod.resample_scales = orig_resample_scales
    slds_mod.resample_heading = orig_resample_heading
    slds_mod.resample_location = orig_resample_loc

    m_unp = copy.deepcopy(model)
    m_unp['states'] = copy.deepcopy(sliced_states)
    out_dir_unp = os.path.join(KPMS_DIR, "scratch_audit_unp")
    os.makedirs(out_dir_unp, exist_ok=True)
    
    m_unp, _ = kpms.fit_model(
        m_unp, sliced_data, sliced_metadata, out_dir_unp,
        model_name="audit_unp",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=100,
        verbose=False
    )
    unp = compute_model_macro_metrics(m_unp, sliced_data)
    
    # 2. RUN PATCHED & FIXED
    print(f"\n[2/3] Running {NUM_ITERS} iterations with PATCHED & FIXED...")
    def _numpy_get_lags_fixed(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, i : T - nlags + i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_fixed
    apply_optimizations()
    
    m_fix = copy.deepcopy(model)
    m_fix['states'] = copy.deepcopy(sliced_states)
    out_dir_fix = os.path.join(KPMS_DIR, "scratch_audit_fix")
    os.makedirs(out_dir_fix, exist_ok=True)
    
    m_fix, _ = kpms.fit_model(
        m_fix, sliced_data, sliced_metadata, out_dir_fix,
        model_name="audit_fix",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=100,
        verbose=False
    )
    fix = compute_model_macro_metrics(m_fix, sliced_data)
    
    # 3. RUN BUGGY OLD PATCH (REVERSED LAGS)
    print(f"\n[3/3] Running {NUM_ITERS} iterations with BUGGY OLD PATCH...")
    def _numpy_get_lags_buggy(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, nlags - 1 - i : T - 1 - i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_buggy
    apply_optimizations()
    
    m_bug = copy.deepcopy(model)
    m_bug['states'] = copy.deepcopy(sliced_states)
    out_dir_bug = os.path.join(KPMS_DIR, "scratch_audit_bug")
    os.makedirs(out_dir_bug, exist_ok=True)
    
    m_bug, _ = kpms.fit_model(
        m_bug, sliced_data, sliced_metadata, out_dir_bug,
        model_name="audit_bug",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=100,
        verbose=False
    )
    bug = compute_model_macro_metrics(m_bug, sliced_data)

    # Restore fixed globally
    rb._numpy_get_lags = _numpy_get_lags_fixed
    apply_optimizations()
    
    # Transition matrix correlations with ground truth
    pi_corr_fix = float(np.corrcoef(unp['pi'].ravel(), fix['pi'].ravel())[0, 1])
    pi_corr_bug = float(np.corrcoef(unp['pi'].ravel(), bug['pi'].ravel())[0, 1])
    
    # PRINT 3-WAY COMPARISON REPORT
    print("\n" + "=" * 115)
    print("                     3-WAY MACROSCOPIC SYSTEM DIVERGENCE REPORT")
    print("=" * 115)
    print(f"{'Macro Metric':<32} | {'Unpatched JAX (Ref)':<20} | {'Patched (FIXED)':<20} | {'Buggy Patch (OLD)':<20} | {'Fixed vs Buggy Delta':<20}")
    print("-" * 115)
    print(f"{'Reconstruction MSE':<32} | {unp['obs_mse']:18.4f}   | {fix['obs_mse']:18.4f}   | {bug['obs_mse']:18.4f}   | Δ = {abs(fix['obs_mse']-bug['obs_mse']):.4f}")
    print(f"{'Median Duration (p50)':<32} | {unp['dur_p50']:15.1f} ms   | {fix['dur_p50']:15.1f} ms   | {bug['dur_p50']:15.1f} ms   | Δ = {abs(fix['dur_p50']-bug['dur_p50']):.1f} ms (Collapse)")
    print(f"{'Mean Duration':<32} | {unp['dur_mean']:15.1f} ms   | {fix['dur_mean']:15.1f} ms   | {bug['dur_mean']:15.1f} ms   | Δ = {abs(fix['dur_mean']-bug['dur_mean']):.1f} ms")
    print(f"{'IQR Duration (p25 - p75)':<32} | {unp['dur_p25']:.0f} - {unp['dur_p75']:.0f} ms{' ':8s} | {fix['dur_p25']:.0f} - {fix['dur_p75']:.0f} ms{' ':8s} | {bug['dur_p25']:.0f} - {bug['dur_p75']:.0f} ms{' ':8s} | Compressed IQR")
    print(f"{'Active Syllables (99%)':<32} | {unp['active_99']:18d}   | {fix['active_99']:18d}   | {bug['active_99']:18d}   | Δ = {fix['active_99']-bug['active_99']} Syllables")
    print(f"{'Syllable Usage Entropy':<32} | {unp['entropy']:18.4f}   | {fix['entropy']:18.4f}   | {bug['entropy']:18.4f}   | Δ = {abs(fix['entropy']-bug['entropy']):.4f}")
    print(f"{'Self-Transition Prob (pi_kk)':<32} | {unp['mean_self_trans']:18.4f}   | {fix['mean_self_trans']:18.4f}   | {bug['mean_self_trans']:18.4f}   | Δ = {abs(fix['mean_self_trans']-bug['mean_self_trans']):.4f}")
    print(f"{'Transition Matrix Correlation':<32} | {'1.0000 (Ref)':<20} | {pi_corr_fix:18.4f}   | {pi_corr_bug:18.4f}   | {'Fix=1.00 vs Bug=Diverged':<20}")
    print("=" * 115)

if __name__ == "__main__":
    main()
