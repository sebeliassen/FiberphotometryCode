"""
Core logic for sliding window KL divergence optimization.
Includes KLOptimizer class, metric calculators, and notification helpers.
"""
import urllib.request
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm


from fpcode.common.notify import send_ntfy as _unified_send_ntfy

def send_ntfy_notification(message: str, title: str = "KL Optimization"):
    """
    Sends a push notification to https://ntfy.sh/woody_videos_seb.
    """
    _unified_send_ntfy(title=title, msg=message)

def compute_window_kl(counts: np.ndarray, global_dist: np.ndarray) -> float:
    """ Computes mean KL divergence of window counts relative to a global distribution. """
    row_sums = counts.sum(axis=1, keepdims=True)
    eps = 1e-5
    # Laplace smoothing
    probs = (counts + eps) / (row_sums + eps * counts.shape[1])
    kl_matrix = probs * np.log(probs / global_dist)
    kl_vals = kl_matrix.sum(axis=1)
    return float(np.mean(kl_vals))

def evaluate_candidate(
    W_val: float,
    r_val: float,
    cached_sessions: List[dict],
    window_type: str,
    vocab_size: int
) -> float:
    """
    Computes Score = KL_actual - KL_shuffled for a given window size and step ratio.
    """
    actual_counts_list = []
    shuffled_counts_list = []

    if window_type == "time":
        # Time-based windows (W in seconds, converted to frames at 80 Hz)
        W_frames = int(np.round(W_val * 80.0))
        S_frames = max(1, int(np.round(r_val * W_val * 80.0)))

        if W_frames < 2:
            return 0.0

        for s in cached_sessions:
            frames = s["syllables"]
            if len(frames) < W_frames:
                continue

            # Actual sliding windows
            win_view = np.lib.stride_tricks.sliding_window_view(frames, W_frames)[::S_frames]
            M = win_view.shape[0]
            win_view_mapped = np.where(win_view == -1, vocab_size, win_view)
            offset = np.arange(M)[:, None] * (vocab_size + 1)
            flat_idx = win_view_mapped + offset
            counts_flat = np.bincount(flat_idx.ravel(), minlength=M * (vocab_size + 1))
            c = counts_flat.reshape(M, vocab_size + 1)[:, :vocab_size]
            actual_counts_list.append(c)

            # Reconstruct shuffled timeline (preserving block durations but shuffling sequence)
            shuf_idx = np.arange(len(s["collapsed_sylls"]))
            np.random.shuffle(shuf_idx)
            shuf_sylls = s["collapsed_sylls"][shuf_idx]
            shuf_durs = s["durations"][shuf_idx]
            shuf_frames = np.repeat(shuf_sylls, shuf_durs)

            if len(shuf_frames) >= W_frames:
                win_view_shuf = np.lib.stride_tricks.sliding_window_view(shuf_frames, W_frames)[::S_frames]
                M_shuf = win_view_shuf.shape[0]
                win_view_shuf_mapped = np.where(win_view_shuf == -1, vocab_size, win_view_shuf)
                offset_shuf = np.arange(M_shuf)[:, None] * (vocab_size + 1)
                flat_idx_shuf = win_view_shuf_mapped + offset_shuf
                counts_flat_shuf = np.bincount(flat_idx_shuf.ravel(), minlength=M_shuf * (vocab_size + 1))
                c_shuf = counts_flat_shuf.reshape(M_shuf, vocab_size + 1)[:, :vocab_size]
                shuffled_counts_list.append(c_shuf)

    else:
        # Step-based windows (W is number of transitions/syllables)
        W_steps = int(np.round(W_val))
        S_steps = max(1, int(np.round(r_val * W_val)))

        if W_steps < 2:
            return 0.0

        for s in cached_sessions:
            collapsed = s["collapsed_sylls"]
            if len(collapsed) < W_steps:
                continue

            # Actual sliding windows
            win_view = np.lib.stride_tricks.sliding_window_view(collapsed, W_steps)[::S_steps]
            M = win_view.shape[0]
            win_view_mapped = np.where(win_view == -1, vocab_size, win_view)
            offset = np.arange(M)[:, None] * (vocab_size + 1)
            flat_idx = win_view_mapped + offset
            counts_flat = np.bincount(flat_idx.ravel(), minlength=M * (vocab_size + 1))
            c = counts_flat.reshape(M, vocab_size + 1)[:, :vocab_size]
            actual_counts_list.append(c)

            # Shuffled sequence control
            shuf_collapsed = collapsed.copy()
            np.random.shuffle(shuf_collapsed)

            win_view_shuf = np.lib.stride_tricks.sliding_window_view(shuf_collapsed, W_steps)[::S_steps]
            M_shuf = win_view_shuf.shape[0]
            win_view_shuf_mapped = np.where(win_view_shuf == -1, vocab_size, win_view_shuf)
            offset_shuf = np.arange(M_shuf)[:, None] * (vocab_size + 1)
            flat_idx_shuf = win_view_shuf_mapped + offset_shuf
            counts_flat_shuf = np.bincount(flat_idx_shuf.ravel(), minlength=M_shuf * (vocab_size + 1))
            c_shuf = counts_flat_shuf.reshape(M_shuf, vocab_size + 1)[:, :vocab_size]
            shuffled_counts_list.append(c_shuf)

    if not actual_counts_list or not shuffled_counts_list:
        return 0.0

    actual_counts = np.concatenate(actual_counts_list, axis=0)
    shuffled_counts = np.concatenate(shuffled_counts_list, axis=0)

    # Global distribution calculated across actual windows
    global_counts = actual_counts.sum(axis=0)
    eps = 1e-5
    global_dist = (global_counts + eps) / (global_counts.sum() + eps * vocab_size)

    global_counts_shuf = shuffled_counts.sum(axis=0)
    global_dist_shuf = (global_counts_shuf + eps) / (global_counts_shuf.sum() + eps * vocab_size)

    kl_actual = compute_window_kl(actual_counts, global_dist)
    kl_shuffled = compute_window_kl(shuffled_counts, global_dist_shuf)

    # Score is the difference (Structured Information Gain)
    return max(0.0, kl_actual - kl_shuffled)

def expected_improvement_2d(X: np.ndarray, gp: GaussianProcessRegressor, y_best: float, xi: float = 0.05) -> np.ndarray:
    """ Computes Expected Improvement for 2D candidate grid. """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - y_best - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei

def upper_confidence_bound_2d(X: np.ndarray, gp: GaussianProcessRegressor, beta: float = 3.0) -> np.ndarray:
    """ Computes Upper Confidence Bound for 2D candidate grid. """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    return mu + beta * sigma

class KLOptimizer:
    def __init__(
        self,
        dataset: str,
        window_type: str,
        cached_sessions: List[dict],
        vocab_size: int,
        n_iter: int,
        verify: bool = False,
        callback: Callable[[int, float, float, float], None] = None
    ):
        self.dataset = dataset
        self.window_type = window_type
        self.cached_sessions = cached_sessions
        self.vocab_size = vocab_size
        self.n_iter = n_iter
        self.verify = verify
        self.callback = callback

        # Calculate boundaries
        if self.window_type == "time":
            self.W_min, self.W_max = 2.0, 300.0
        else:
            self.W_min, self.W_max = 3.0, 200.0

        self.log_W_min = np.log10(self.W_min)
        self.log_W_max = np.log10(self.W_max)
        self.r_min, self.r_max = 0.01, 0.5

        if self.verify:
            self.n_init = 3
            self.n_guided = 2
        else:
            self.n_init = min(150, int(0.3 * self.n_iter))
            self.n_guided = self.n_iter - self.n_init

        self.results = []

    def run(self) -> pd.DataFrame:
        run_desc = f"{self.dataset.upper()} - {self.window_type.upper()}"
        start_msg = f"Started KL Divergence Optimization for {run_desc}. Total iterations: {self.n_iter}."
        print(start_msg)
        send_ntfy_notification(start_msg, title="KL Optimization Started")

        # Setup Latin Hypercube Sampling for initial exploration
        sampler = qmc.LatinHypercube(d=2, seed=42)
        samples = sampler.random(n=self.n_init)

        log_W_samples = samples[:, 0] * (self.log_W_max - self.log_W_min) + self.log_W_min
        r_samples = samples[:, 1] * (self.r_max - self.r_min) + self.r_min

        # Unified progress bar
        total_steps = self.n_init + self.n_guided
        pbar = tqdm(total=total_steps, desc=f"Optimizing {run_desc}")

        # 1. LHS Phase
        for idx in range(self.n_init):
            log_w = log_W_samples[idx]
            r = r_samples[idx]
            w = float(10**log_w)

            score = evaluate_candidate(w, r, self.cached_sessions, self.window_type, self.vocab_size)

            iter_num = idx + 1
            row_data = {
                "iteration": iter_num,
                "W": w,
                "log_W": log_w,
                "r": r,
                "score": score,
                "type": "random"
            }
            self.results.append(row_data)
            pbar.update(1)

            if self.callback:
                self.callback(iter_num, w, r, score)

        init_done_msg = f"LHS initialization complete for {run_desc} ({self.n_init} iterations done)."
        print(init_done_msg)
        send_ntfy_notification(init_done_msg, title="LHS Phase Complete")

        # 2. GP Guided Phase
        for idx in range(self.n_guided):
            train_df = pd.DataFrame(self.results)
            train_df = train_df.drop_duplicates(subset=["log_W", "r"])

            X_train = train_df[["log_W", "r"]].to_numpy()
            y_train = train_df["score"].to_numpy()

            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5, length_scale=[0.5, 0.1]),
                alpha=1e-5,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=42
            )
            gp.fit(X_train, y_train)

            y_best = y_train.max()

            # Acquisition function candidate grid search
            grid_sampler = qmc.LatinHypercube(d=2, seed=100 + idx)
            grid_samples = grid_sampler.random(n=3000)
            grid_log_W = grid_samples[:, 0] * (self.log_W_max - self.log_W_min) + self.log_W_min
            grid_r = grid_samples[:, 1] * (self.r_max - self.r_min) + self.r_min
            candidates = np.column_stack([grid_log_W, grid_r])

            if idx % 2 == 0:
                acq_vals = expected_improvement_2d(candidates, gp, y_best)
            else:
                acq_vals = upper_confidence_bound_2d(candidates, gp, beta=3.0)

            best_candidate_idx = np.argmax(acq_vals)
            best_cand = candidates[best_candidate_idx]

            def obj(x):
                x_arr = x.reshape(1, -1)
                if idx % 2 == 0:
                    return -float(expected_improvement_2d(x_arr, gp, y_best)[0])
                else:
                    return -float(upper_confidence_bound_2d(x_arr, gp, beta=3.0)[0])

            res = minimize(
                obj,
                x0=best_cand,
                bounds=[(self.log_W_min, self.log_W_max), (self.r_min, self.r_max)],
                method="L-BFGS-B"
            )

            if res.success:
                next_log_w, next_r = res.x
            else:
                next_log_w, next_r = best_cand

            next_w = float(10**next_log_w)
            iter_num = self.n_init + idx + 1

            score = evaluate_candidate(next_w, next_r, self.cached_sessions, self.window_type, self.vocab_size)

            row_data = {
                "iteration": iter_num,
                "W": next_w,
                "log_W": next_log_w,
                "r": next_r,
                "score": score,
                "type": "guided"
            }
            self.results.append(row_data)
            pbar.update(1)

            if self.callback:
                self.callback(iter_num, next_w, next_r, score)

            # Periodic notifications every 100 iterations
            if iter_num % 100 == 0:
                current_best = pd.DataFrame(self.results)["score"].max()
                prog_msg = f"Progress: {iter_num}/{total_steps} iterations for {run_desc}. Best Score: {current_best:.6f}"
                send_ntfy_notification(prog_msg, title="KL Optimization Progress")

        pbar.close()

        df_res = pd.DataFrame(self.results)
        best_row = df_res.loc[df_res["score"].idxmax()]

        final_msg = (
            f"Optimization finished for {run_desc}!\n"
            f"Best Configuration:\n"
            f"  - Window (W): {best_row['W']:.4f}\n"
            f"  - Step Ratio (r): {best_row['r']:.4f}\n"
            f"  - Score: {best_row['score']:.6f}"
        )
        print(final_msg)
        send_ntfy_notification(final_msg, title="KL Optimization Complete ✅")

        return df_res
