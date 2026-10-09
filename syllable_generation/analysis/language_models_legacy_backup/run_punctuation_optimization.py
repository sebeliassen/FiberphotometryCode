"""
Run Method 1 (Slow Syllable Slicing) Bayesian Hyperparameter Optimization (N=1000)
on IFDN and AMPH individually, with temporal exclusions, and save absolute punctuation indices.
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.preprocessing import optimization
from analysis.language_models.lib.visualization import metrics_viz
from analysis.ulrik_kinematics.lib import loader
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import wandb


def run_dataset_optimization(project_name: str):
    print("\n==================================================")
    print(f"*** Starting Optimization for {project_name.upper()} ***")
    print("==================================================")

    # 1. Load Sessions with custom temporal filtering
    if project_name == "ifdn":
        sessions = loader.load_dataset("ifdn", base_dir=ROOT_DIR, ifdn_phase="5min_end")
    elif project_name == "amph":
        sessions = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="both")
    else:
        raise ValueError(f"Unknown project: {project_name}")

    print(f"Loaded {len(sessions)} {project_name.upper()} sessions.")

    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/bouts")
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparameters for N=1000 optimization
    n_init = 300
    n_iter = 700
    weights = (1.0, 1.0)
    xi = 0.3

    # Initialize wandb run
    wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name=f"method1-opt-{project_name}",
        config={
            "method": 1,
            "goal": "punctuation_optimization",
            "dataset": project_name,
            "n_init": n_init,
            "n_iter": n_iter,
            "weights": weights,
            "exploration_xi": xi,
            "sampler": "LHS",
            "acquisition_portfolio": "EI + UCB Hedging"
        }
    )

    cache_df = optimization.precompute_and_cache_dataset(sessions, project_name)

    # Load syllable velocities
    vel_path = os.path.join(optimization.CACHE_DIR, f"{project_name}_syllable_velocities.parquet")
    vels_df = pd.read_parquet(vel_path)
    sorted_sylls = vels_df.sort_values("mean_velocity")["syllable"].astype(int).tolist()

    # Reconstruct sessions for evaluation using MockSession
    mock_sessions = []
    for _, row in cache_df.iterrows():
        mock_sessions.append(optimization.MockSession(row["session_id"], row["genotype"], row["drug"]))

    # Bounds in log10 space
    log_p_min = np.log10(0.000015)
    log_p_max = np.log10(0.99)

    # Initial sampling using Latin Hypercube Sampling (LHS)
    sampler = qmc.LatinHypercube(d=1, seed=42)
    sample = sampler.random(n=n_init)
    X_log = (sample * (log_p_max - log_p_min) + log_p_min).flatten().tolist()

    results = []
    total_iters = n_init + n_iter

    # Evaluate initial points
    for idx, x_log in enumerate(X_log):
        p = float(10**x_log)
        iter_num = idx + 1
        if iter_num % 100 == 0 or iter_num == 1:
            print(f"LHS Init {iter_num}/{n_init} ({project_name.upper()}): percentile = {p:.6f}")

        S, M1, M3 = optimization.evaluate_candidate_m1(p, sorted_sylls, cache_df, mock_sessions, weights)

        row_data = {
            "iteration": iter_num,
            "percentile": p,
            "log_percentile": x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "type": "random"
        }
        results.append(row_data)

        wandb.log({
            "iteration": iter_num,
            "percentile": p,
            "log_percentile": x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 0
        })

        if iter_num % 100 == 0 or iter_num == total_iters:
            fig = optimization.plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Sequential Guided Optimization
    for i in range(n_iter):
        train_df = pd.DataFrame(results)
        train_df["rounded_log"] = np.round(train_df["log_percentile"], 4)
        train_df = train_df.drop_duplicates(subset=["rounded_log"])

        # Keep GP fitting fast with active training-set sub-sampling
        if len(train_df) > 800:
            best_points = train_df.nlargest(400, "composite_score")
            other_points = train_df.nsmallest(len(train_df) - 400, "composite_score").sample(400, random_state=42)
            train_subset = pd.concat([best_points, other_points])
        else:
            train_subset = train_df

        X_train = train_subset["log_percentile"].to_numpy().reshape(-1, 1)
        y_train = train_subset["composite_score"].to_numpy()

        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )
        gp.fit(X_train, y_train)

        y_best = y_train.max()
        candidates = np.linspace(log_p_min, log_p_max, 5000).reshape(-1, 1)

        if i % 2 == 0:
            acq_val = optimization.expected_improvement(candidates, gp, y_best, xi=xi)
        else:
            acq_val = optimization.upper_confidence_bound(candidates, gp, beta=4.0)

        best_idx = np.argmax(acq_val)
        best_candidate = candidates[best_idx, 0]

        def obj(x):
            x_arr = np.array(x).reshape(-1, 1)
            if i % 2 == 0:
                return -float(optimization.expected_improvement(x_arr, gp, y_best, xi=xi).ravel()[0])
            else:
                return -float(optimization.upper_confidence_bound(x_arr, gp, beta=4.0).ravel()[0])

        res = minimize(obj, x0=[best_candidate], bounds=[(log_p_min, log_p_max)], method="L-BFGS-B")
        if res.success:
            next_x_log = float(res.x[0])
        else:
            next_x_log = float(best_candidate)

        next_p = float(10**next_x_log)
        iter_num = n_init + i + 1

        if iter_num % 100 == 0:
            print(f"Guided Eval {iter_num}/{total_iters} ({project_name.upper()}): percentile = {next_p:.6f}")

        S, M1, M3 = optimization.evaluate_candidate_m1(next_p, sorted_sylls, cache_df, mock_sessions, weights)

        row_data = {
            "iteration": iter_num,
            "percentile": next_p,
            "log_percentile": next_x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "type": "guided"
        }
        results.append(row_data)

        wandb.log({
            "iteration": iter_num,
            "percentile": next_p,
            "log_percentile": next_x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 1
        })

        if iter_num % 100 == 0 or iter_num == total_iters:
            fig = optimization.plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    df_res = pd.DataFrame(results)

    # Locate best parameters
    best_row = df_res.loc[df_res["composite_score"].idxmax()]
    p_opt = float(best_row["percentile"])
    print(f"\nBest configuration found for {project_name.upper()}:")
    print(best_row.to_frame().T.to_markdown(index=False, floatfmt=".6f"))

    # Reconstruct optimal slow syllable boundary IDs
    cutoff_idx = max(1, int(len(sorted_sylls) * p_opt))
    punctuation_syllable_ids = np.array(sorted_sylls[:cutoff_idx])
    print(f"Optimal percentile selects {len(punctuation_syllable_ids)} boundary syllables: {punctuation_syllable_ids}")

    # Build dictionary of absolute indices matching punctuation frames
    session_punctuation_indices = {}
    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns:
            continue
        # Get absolute index values where syllable ID belongs to punctuation set
        punct_mask = df["syllable"].isin(punctuation_syllable_ids)
        punct_idx = df.index[punct_mask].to_numpy().astype(int)
        session_punctuation_indices[s.session_id] = punct_idx

    output_pkl_path = os.path.join(out_dir, f"{project_name}_punctuation_frames.pkl")

    save_dict = {
        "optimal_percentile": p_opt,
        "punctuation_syllable_ids": punctuation_syllable_ids,
        "session_punctuation_indices": session_punctuation_indices
    }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved optimal punctuation frames to: {output_pkl_path}")

    # Plot final landscape
    opt_path = os.path.join(out_dir, f"method1_{project_name}_joint_optimization_sweep.png")
    metrics_viz.plot_optimization_results_m1(df_res, opt_path)
    print(f"Saved final sweep plot to {opt_path}")

    wandb.finish()

if __name__ == "__main__":
    run_dataset_optimization("ifdn")
    run_dataset_optimization("amph")
