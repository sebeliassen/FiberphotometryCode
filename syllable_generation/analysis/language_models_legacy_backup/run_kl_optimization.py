"""
Sliding Window KL Divergence Optimization Sweep using 2D Bayesian Optimization.
Runner script that utilizes KLOptimizer from the lib submodule.
"""
import argparse
import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

import wandb

# Resolve paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.optimization.optimizer import KLOptimizer
from analysis.ulrik_kinematics.lib import loader

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Sliding Window KL Optimization Sweep")
    parser.add_argument("--dataset", type=str, required=True, choices=["ifdn", "amph"], help="Dataset to load")
    parser.add_argument("--window-type", type=str, required=True, choices=["time", "step"], help="Window step type")
    parser.add_argument("--n-iter", type=str, default="500", help="Total number of iterations")
    parser.add_argument("--verify", action="store_true", help="Verification run (subset size)")
    args = parser.parse_args()

    dataset = args.dataset
    window_type = args.window_type
    n_iter_total = int(args.n_iter)
    verify = args.verify

    print("\n==========================================")
    print(f"KL Divergence Optimization: {dataset.upper()} - {window_type.upper()}")
    if verify:
        print("*** VERIFICATION DRY RUN MODE ***")
    print("==========================================")

    # Load Sessions
    if dataset == "ifdn":
        sessions = loader.load_dataset("ifdn", base_dir=ROOT_DIR, ifdn_phase="5min_end")
    else:
        sessions = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="both")

    if verify:
        # Dry-run uses first 2 sessions and only 5 iterations
        sessions = sessions[:2]
        n_iter_total = 5
        n_init = 3
        n_guided = 2
    else:
        n_init = min(150, int(0.3 * n_iter_total))
        n_guided = n_iter_total - n_init

    print(f"Loaded {len(sessions)} sessions.")

    # Cache session dataframes in memory for speed
    cached_sessions = []
    max_syll = 0
    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns:
            continue
        sylls = df["syllable"].fillna(-1).to_numpy().astype(int)

        # Collapse transitions for RLE
        mask = np.concatenate(([True], sylls[1:] != sylls[:-1]))
        collapsed_sylls = sylls[mask]

        change_indices = np.where(mask)[0]
        durations = np.diff(np.concatenate((change_indices, [len(sylls)])))

        # Drop -1s from collapsed sequence to avoid indexing errors
        valid_mask = collapsed_sylls != -1
        collapsed_sylls = collapsed_sylls[valid_mask]
        durations = durations[valid_mask]

        if len(collapsed_sylls) == 0:
            continue

        max_syll = max(max_syll, int(collapsed_sylls.max()))
        cached_sessions.append({
            "session_id": s.session_id,
            "syllables": sylls,
            "collapsed_sylls": collapsed_sylls,
            "durations": durations
        })

    vocab_size = max_syll + 1
    print(f"Pre-cached {len(cached_sessions)} sessions in memory. Vocabulary size: {vocab_size}")

    # Setup WandB
    run_name = f"kl-opt-{dataset}-{window_type}"
    if verify:
        run_name += "-verify"

    wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name=run_name,
        config={
            "dataset": dataset,
            "window_type": window_type,
            "n_init": n_init,
            "n_guided": n_guided,
            "verify": verify
        }
    )

    def wandb_callback(iter_num, w, r, score):
        run_type = 0 if iter_num <= n_init else 1
        wandb.log({
            "iteration": iter_num,
            "W": w,
            "r": r,
            "score": score,
            "run_type": run_type
        })

    # Instantiate and run KLOptimizer
    optimizer = KLOptimizer(
        dataset=dataset,
        window_type=window_type,
        cached_sessions=cached_sessions,
        vocab_size=vocab_size,
        n_iter=n_iter_total,
        verify=verify,
        callback=wandb_callback
    )

    df_res = optimizer.run()
    best_idx = df_res["score"].idxmax()
    best_row = df_res.loc[best_idx]

    print("\n==========================================")
    print("BEST CONFIGURATION FOUND:")
    print(best_row.to_frame().T.to_markdown(index=False, floatfmt=".6f"))
    print("==========================================")

    # Save configuration pickle to language_models/output/kl
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/kl")
    os.makedirs(out_dir, exist_ok=True)
    pkl_name = f"optimal_kl_params_{dataset}_{window_type}.pkl"
    if verify:
        pkl_name = "verify_" + pkl_name
    pkl_path = os.path.join(out_dir, pkl_name)

    save_dict = {
        "best_W": float(best_row["W"]),
        "best_r": float(best_row["r"]),
        "best_score": float(best_row["score"]),
        "all_results": df_res.to_dict(orient="records")
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved optimal parameters to: {pkl_path}")

    # Generate 2D Scatter plot of visited parameters
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_res["W"], df_res["r"], c=df_res["score"], cmap="viridis", s=30, edgecolor="none")
    ax.scatter(best_row["W"], best_row["r"], color="red", marker="*", s=150, label="Best Config")
    ax.set_xlabel("Window Size (W)")
    ax.set_ylabel("Step Ratio (r)")
    ax.set_title(f"KL Divergence Sweep: {dataset.upper()} - {window_type.upper()}")
    if window_type == "time":
        ax.set_xscale("log")
    fig.colorbar(sc, label="Chance-Corrected KL Score")
    ax.legend()

    plot_name = f"kl_optimization_sweep_{dataset}_{window_type}.png"
    if verify:
        plot_name = "verify_" + plot_name
    plot_path = os.path.join(out_dir, plot_name)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter sweep plot to: {plot_path}")

    # Log final sweep plot to WandB
    wandb.log({"kl_optimization_sweep_plot": wandb.Image(plot_path)})

    wandb.finish()

if __name__ == "__main__":
    main()
