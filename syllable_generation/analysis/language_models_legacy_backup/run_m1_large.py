"""
Run Method 1 (Slow Syllable Slicing) with a large 3000-evaluation sweep
and extra exploration (xi=0.3) using joint weights (1.0, 1.0) on the AMPH pre-injection dataset.
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader
from analysis.language_models.lib.preprocessing import optimization
from analysis.language_models.lib.visualization import metrics_viz
import wandb
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def run_large_sweep():
    print("*** Loading AMPH dataset (pre-phase)... ***")
    sessions = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="pre")
    print(f"Loaded {len(sessions)} AMPH sessions.")
    
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/bouts")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3000 iterations: 1000 random LHS initialization, 2000 GP-guided
    n_init = 1000
    n_iter = 2000
    weights = (1.0, 1.0)
    xi = 0.3  # Increased exploration parameter
    
    print(f"\n>>> Running Method 1 Large 3000 Sweep (init={n_init}, iter={n_iter}, xi={xi}) <<<")
    
    # Initialize wandb run
    wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name="method1-large-composite-amph",
        config={
            "method": 1,
            "goal": "composite_large",
            "dataset": "amph",
            "n_init": n_init,
            "n_iter": n_iter,
            "weights": weights,
            "exploration_xi": xi,
            "sampler": "LHS",
            "acquisition_portfolio": "EI + UCB Hedging"
        }
    )

    cache_df = optimization.precompute_and_cache_dataset(sessions, "amph")
    
    # Load syllable velocities
    vel_path = os.path.join(optimization.CACHE_DIR, "amph_syllable_velocities.parquet")
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
            print(f"LHS Init {iter_num}/{n_init} (Method 1): percentile = {p:.6f}")
        
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
        
        # Log to wandb
        wandb.log({
            "iteration": iter_num,
            "percentile": p,
            "log_percentile": x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 0
        })
        
        if iter_num % 250 == 0 or iter_num == total_iters:
            fig = optimization.plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)
        
    # Sequential Guided Optimization
    for i in range(n_iter):
        # Sub-sample training points for GP if N becomes very large to prevent O(N^3) memory/compute slowdown
        # We take all unique points or a representative subset of max 800 points to keep it super fast.
        train_df = pd.DataFrame(results)
        # Drop duplicates in rounded log_percentile to avoid singular covariance matrix
        train_df["rounded_log"] = np.round(train_df["log_percentile"], 4)
        train_df = train_df.drop_duplicates(subset=["rounded_log"])
        
        if len(train_df) > 800:
            # Keep the best performing points and a random subset of other points
            best_points = train_df.nlargest(400, "composite_score")
            other_points = train_df.nsmallest(len(train_df) - 400, "composite_score").sample(400, random_state=42)
            train_subset = pd.concat([best_points, other_points])
        else:
            train_subset = train_df
            
        X_train = train_subset["log_percentile"].to_numpy().reshape(-1, 1)
        y_train = train_subset["composite_score"].to_numpy()
        
        # We use n_restarts_optimizer=2 (instead of 20) for speed in large dimensions
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
        
        # Alternate acquisition function (EI vs UCB) to ensure global exploration
        if i % 2 == 0:
            acq_val = optimization.expected_improvement(candidates, gp, y_best, xi=xi)
        else:
            acq_val = optimization.upper_confidence_bound(candidates, gp, beta=4.0)  # Beta=4.0 for higher exploration
            
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
            print(f"Guided Eval {iter_num}/{total_iters} (Method 1): percentile = {next_p:.6f}")
            
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
        
        # Log to wandb
        wandb.log({
            "iteration": iter_num,
            "percentile": next_p,
            "log_percentile": next_x_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 1
        })
        
        if iter_num % 250 == 0 or iter_num == total_iters:
            fig = optimization.plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)
            
    df = pd.DataFrame(results)
    print("\nMethod 1 Large Optimization Best Results:")
    print(df.nlargest(10, "composite_score").to_markdown(index=False, floatfmt=".6f"))
    
    opt_path = os.path.join(out_dir, "method1_large_joint_optimization_sweep.png")
    metrics_viz.plot_optimization_results_m1(df, opt_path)
    print(f"Finished large optimization. Plot saved to {opt_path}")
    wandb.finish()

if __name__ == "__main__":
    run_large_sweep()
