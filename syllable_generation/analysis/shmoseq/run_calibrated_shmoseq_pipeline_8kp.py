#!/usr/bin/env python3
"""
run_calibrated_shmoseq_pipeline_8kp.py
=============================================================================
Calibrated shMoSeq Model Fitting & Module Ethogram Analysis on 8KP (Run 03) Data.

Key Objectives:
1. Calibrate Stickiness (trans_kappa):
   - Evaluate module bout timescales across candidate stickiness settings
     (e.g., kappa = 5e3, 2e4 vs. previous 5e5) to avoid excessive session-long locking
     while preventing high-frequency syllable-level chatter.
2. Condition-Blind HHMM Fit (K=8, 50 Gibbs iterations on GPU).
3. Posterior State Decoding & Episode Statistics:
   - Median bout duration, IQR, max bout duration, number of switches per session.
4. 30-Session Timeline Ethograms (0-60 min post-return):
   - Grouped by WT Saline, WT Amph, KI Saline, KI Amph.
5. Behavioral Grounding:
   - Identify representative bouts for video clip inspection.
   - Strictly segregated in output_8kp_run03/.
=============================================================================
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import jax
import jax.numpy as jnp
import jax.random as jr
import state_moseq as sm

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_OUTPUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "output_8kp_run03"
DATA_FILE = BASE_OUTPUT_DIR / "data" / "shmoseq_input_data_8kp.pkl"


def run_length_encoding(seq):
    if len(seq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    change_idx = np.where(seq[1:] != seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(seq)]
    durations = ends - starts
    states = seq[starts]
    return states, durations, starts, ends


def fit_and_evaluate_shmoseq(data, metadata_dict, n_syllables, n_states=8, kappa=1e4, num_iters=50, seed_val=0, tag="calibrated"):
    print("\n" + "=" * 80)
    print(f"🔬 FITTING SHMOSEQ HHMM: K={n_states}, trans_kappa={kappa:g}, iters={num_iters}, tag={tag}")
    print("=" * 80, flush=True)

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

    log_probs = []
    gd_losses = []
    t0 = time.time()

    for it in range(num_iters):
        t_iter = time.time()
        seed, subseed = jr.split(seed)
        params, losses = sm.hhmm_efficient.resample_params(subseed, data, states, hypparams, params)
        seed, subseed = jr.split(seed)
        states, marginal_loglik = sm.hhmm_efficient.resample_states(subseed, data, params, parallel=False)
        log_param_p = sm.hhmm_efficient.log_params_prob(params, hypparams)
        current_log_joint = float(log_param_p + marginal_loglik)
        log_probs.append(current_log_joint)
        gd_losses.append(float(losses[-1]))
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [Iter {it+1:02d}/{num_iters}] Log-Joint: {current_log_joint:.2f}, GD-Loss: {float(losses[-1]):.4f} ({time.time()-t_iter:.1f}s/it)", flush=True)

    fit_dur = time.time() - t0
    print(f"✓ Model fitting finished in {fit_dur/60:.1f} minutes.", flush=True)

    # Decode smoothed state probabilities
    print("Decoding smoothed posterior state sequences...", flush=True)
    post_probs = sm.hhmm_efficient.smoothed_states(data, params, parallel=False)
    decoded_states = np.argmax(np.array(post_probs), axis=-1)

    # Compute episode statistics
    session_keys = list(metadata_dict.keys())
    episode_records = []
    session_summaries = []

    for i, s_key in enumerate(session_keys):
        meta = metadata_dict[s_key]
        n_valid = int(data["mask"][i].sum())
        s_seq = decoded_states[i, :n_valid]

        st_arr, dur_arr, starts, ends = run_length_encoding(s_seq)
        dur_sec = dur_arr / 40.0  # 40 Hz frames -> seconds

        for st, d_sec, st_f, end_f in zip(st_arr, dur_sec, starts, ends):
            episode_records.append({
                "session_key": s_key,
                "animal_id": meta["animal_id"],
                "genotype": meta["genotype"],
                "drug": meta["drug"],
                "state": int(st),
                "duration_sec": float(d_sec),
                "duration_frames": int(d_sec * 40),
                "start_frame_40hz": int(st_f),
                "end_frame_40hz": int(end_f),
            })

        session_summaries.append({
            "session_key": s_key,
            "animal_id": meta["animal_id"],
            "genotype": meta["genotype"],
            "drug": meta["drug"],
            "total_episodes": len(st_arr),
            "median_dur_sec": float(np.median(dur_sec)) if len(dur_sec) > 0 else 0.0,
            "mean_dur_sec": float(np.mean(dur_sec)) if len(dur_sec) > 0 else 0.0,
            "max_dur_sec": float(np.max(dur_sec)) if len(dur_sec) > 0 else 0.0,
            "max_dur_min": float(np.max(dur_sec) / 60.0) if len(dur_sec) > 0 else 0.0,
        })

    df_episodes = pd.DataFrame(episode_records)
    df_sessions = pd.DataFrame(session_summaries)

    overall_median_sec = float(df_episodes["duration_sec"].median())
    overall_mean_sec = float(df_episodes["duration_sec"].mean())
    max_lock_min = float(df_episodes["duration_sec"].max() / 60.0)
    median_switches = float(df_sessions["total_episodes"].median())

    print(f"\n--- TIMESCALE METRICS (kappa={kappa:g}) ---")
    print(f"  Median Bout Duration: {overall_median_sec:.2f} seconds ({overall_median_sec*40:.0f} frames)")
    print(f"  Mean Bout Duration:   {overall_mean_sec:.2f} seconds")
    print(f"  Max Bout Duration:    {max_lock_min:.2f} minutes (Check for session locking)")
    print(f"  Median Transitions:   {median_switches:.0f} bouts / session")

    results_bundle = {
        "params": params,
        "hypparams": hypparams,
        "log_probs": log_probs,
        "gd_losses": gd_losses,
        "post_probs": np.array(post_probs),
        "decoded_states": decoded_states,
        "df_episodes": df_episodes,
        "df_sessions": df_sessions,
        "summary": {
            "n_states": n_states,
            "kappa": float(kappa),
            "median_dur_sec": overall_median_sec,
            "mean_dur_sec": overall_mean_sec,
            "max_lock_min": max_lock_min,
            "median_switches": median_switches,
            "fit_dur_min": fit_dur / 60.0,
        }
    }
    return results_bundle


def plot_ethogram_timelines(decoded_states, mask, metadata_dict, n_states, out_path, title_prefix=""):
    """Plot 30-session ethogram heatmap grouped by Cohort (WT/KI, Saline/Amph)."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, sharey=False)
    cmap = plt.cm.get_cmap("tab10", n_states)

    cohort_order = [
        ("WT_Saline", "WT - Saline (8 mice)"),
        ("WT_Amphetamine", "WT - Amphetamine (8 mice)"),
        ("KI_Saline", "DAT-Het KI - Saline (7 mice)"),
        ("KI_Amphetamine", "DAT-Het KI - Amphetamine (7 mice)"),
    ]

    session_keys = list(metadata_dict.keys())
    max_time_min = 60.0  # 60 minutes post-return
    max_frames = int(max_time_min * 60 * 40)

    for ax_idx, (cohort_id, cohort_title) in enumerate(cohort_order):
        ax = axes[ax_idx]
        matching_indices = [
            i for i, k in enumerate(session_keys)
            if metadata_dict[k]["cohort"] == cohort_id
        ]

        n_rows = len(matching_indices)
        # Build 2D raster grid: shape (n_rows, max_frames // 10) downsampled 10x for display
        ds_factor = 10
        n_cols = max_frames // ds_factor
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
        ax.set_title(cohort_title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(0, max_time_min)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Time Since Return to Arena (Minutes)", fontsize=13, fontweight="bold")
    plt.suptitle(f"{title_prefix} - shMoSeq Module Timelines (K={n_states})", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved timeline ethogram to: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Calibrated shMoSeq Pipeline for 8KP Run 03")
    parser.add_argument("--mode", choices=["single", "sweep"], default="sweep", help="Run single fit or stickiness sweep")
    parser.add_argument("--kappa", type=float, default=1e4, help="trans_kappa for single fit")
    args = parser.parse_args()

    out_models = BASE_OUTPUT_DIR / "models"
    out_figs = BASE_OUTPUT_DIR / "figures"
    out_tabs = BASE_OUTPUT_DIR / "tables"
    out_models.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_tabs.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Running extract_shmoseq_inputs_8kp.py first...", flush=True)
        import extract_shmoseq_inputs_8kp
        extract_shmoseq_inputs_8kp.main()

    with open(DATA_FILE, "rb") as f:
        bundle = pickle.load(f)

    syllables_dict = bundle["syllables_dict"]
    metadata_dict = bundle["metadata_dict"]
    n_syllables = bundle["n_syllables"]

    seqs, mask, batch_meta = sm.batch(syllables_dict)
    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask)
    }

    if args.mode == "sweep":
        candidate_kappas = [5e3, 2e4, 5e5]
        sweep_summaries = []

        for k_val in candidate_kappas:
            tag = f"kappa_{k_val:g}"
            res = fit_and_evaluate_shmoseq(
                data, metadata_dict, n_syllables, n_states=8, kappa=k_val, num_iters=50, tag=tag
            )
            sweep_summaries.append(res["summary"])

            # Save model
            model_path = out_models / f"shmoseq_8kp_k8_{tag}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(res, f)

            # Save timeline figure
            fig_path = out_figs / f"fig_module_timelines_{tag}.png"
            plot_ethogram_timelines(
                res["decoded_states"], data["mask"], metadata_dict, n_states=8,
                out_path=fig_path, title_prefix=f"Stickiness kappa={k_val:g}"
            )

        df_sweep = pd.DataFrame(sweep_summaries)
        df_sweep.to_csv(out_tabs / "stickiness_calibration_comparison.csv", index=False)
        print("\n" + "=" * 80)
        print("STICKINESS CALIBRATION SUMMARY:")
        print(df_sweep.to_string(index=False))
        print("=" * 80)
    else:
        k_val = args.kappa
        tag = f"kappa_{k_val:g}"
        res = fit_and_evaluate_shmoseq(
            data, metadata_dict, n_syllables, n_states=8, kappa=k_val, num_iters=50, tag=tag
        )
        with open(out_models / f"shmoseq_8kp_k8_{tag}.pkl", "wb") as f:
            pickle.dump(res, f)
        plot_ethogram_timelines(
            res["decoded_states"], data["mask"], metadata_dict, n_states=8,
            out_path=out_figs / f"fig_module_timelines_{tag}.png", title_prefix=f"Stickiness kappa={k_val:g}"
        )


if __name__ == "__main__":
    main()
