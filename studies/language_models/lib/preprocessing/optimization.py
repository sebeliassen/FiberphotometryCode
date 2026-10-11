import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from studies.behavior.language_models.lib.preprocessing import rle, tokenization
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import StratifiedKFold

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

def evaluate_segmentation(sessions: List,
                           segmented_sentences: Dict[str, List[List[int]]],
                           weights: Tuple[float, float] = (1.2, 0.8)) -> Tuple[float, float, float]:
    """
    Computes the weighted composite score S = w1 * M1 + w3 * M3.
    
    Args:
        sessions: List of Session objects.
        segmented_sentences: Dict mapping session_id -> list of sentences.
        weights: (w1, w3) weights for decodability and compression.
        
    Returns:
        Tuple[float, float, float]: (composite_score, decodability_score, compression_score)
    """
    w1, w3 = weights

    # -----------------------------------------------------------------
    # 1. Compute Class Separation Decodability (M1)
    # -----------------------------------------------------------------
    # Collect all unique syllable/token IDs across the segmented sentences
    all_tokens = set()
    for session_id, sents in segmented_sentences.items():
        for sent in sents:
            all_tokens.update(sent)

    token_list = sorted(list(all_tokens))
    if len(token_list) == 0:
        return 0.0, 0.0, 0.0
    token_to_idx = {tok: idx for idx, tok in enumerate(token_list)}

    X_rows = []
    y_labels = []

    # Filter sessions to only those present in segmented_sentences
    active_sessions = [s for s in sessions if s.session_id in segmented_sentences]

    for s in active_sessions:
        sents = segmented_sentences[s.session_id]
        freqs = np.zeros(len(token_list))

        # Build frequency histogram
        for sent in sents:
            for tok in sent:
                if tok in token_to_idx:
                    freqs[token_to_idx[tok]] += 1

        # Normalize to probability distribution
        total = freqs.sum()
        if total > 0:
            freqs = freqs / total

        X_rows.append(freqs)

        # Binary target: WT vs. KI
        geno = str(s.metadata.get("genotype", "WT")).upper()
        y_labels.append(1 if "KI" in geno else 0)

    X = np.array(X_rows)
    y = np.array(y_labels)

    # Handle edge case where we can't do cross validation
    if len(y) < 10 or len(np.unique(y)) < 2:
        M1 = 0.0
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Simple RandomForest with fixed hyperparameters for speed and stability
            clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            accuracies.append(clf.score(X_test, y_test))

        avg_acc = np.mean(accuracies)
        # Normalize relative to chance (0.5 chance threshold for binary classification)
        M1 = max(0.0, (avg_acc - 0.5) / 0.5)

    # -----------------------------------------------------------------
    # 2. Compute Sequence Compression Score (M3)
    # -----------------------------------------------------------------
    all_flat_sents = []
    for session_id, sents in segmented_sentences.items():
        all_flat_sents.extend(sents)

    if not all_flat_sents:
        M3 = 0.0
    else:
        # Train BPE vocabulary of 50 merges
        vocab_size = len(token_list) + 50
        merges = tokenization.train_bpe_tokenizer(all_flat_sents, vocab_size=vocab_size)
        compressed_sents = tokenization.apply_bpe_merges(all_flat_sents, merges)

        len_raw = sum(len(s) for s in all_flat_sents)
        len_bpe = sum(len(s) for s in compressed_sents)

        M3 = 1.0 - (len_bpe / len_raw) if len_raw > 0 else 0.0

    # -----------------------------------------------------------------
    # 3. Compute Weighted Composite Score S
    # -----------------------------------------------------------------
    S = w1 * M1 + w3 * M3
    return S, M1, M3


def precompute_and_cache_dataset(sessions: List, cache_name: str) -> pd.DataFrame:
    """
    Pre-computes RLE collapsed sequences and session info, saving them as a Parquet cache
    to avoid running slow pandas/numpy operations multiple times during sweeps.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_precomputed.parquet")

    # If cache exists, load it
    if os.path.exists(cache_path):
        print(f"Loading precomputed dataset cache from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Precomputing and caching dataset to {cache_path}...")
    rows = []

    # Global mean velocities calculation
    syllable_vels = {}
    syllable_counts = {}
    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns or "velocity_px_s" not in df.columns: continue
        valid = df[["syllable", "velocity_px_s"]].dropna()
        valid = valid[valid["syllable"] != -1]
        grouped = valid.groupby("syllable")["velocity_px_s"].agg(["sum", "count"])
        for syll, row in grouped.iterrows():
            syll_id = int(syll)
            syllable_vels[syll_id] = syllable_vels.get(syll_id, 0.0) + row["sum"]
            syllable_counts[syll_id] = syllable_counts.get(syll_id, 0) + row["count"]

    mean_vels = {sid: (syllable_vels[sid] / syllable_counts[sid]) for sid in syllable_vels if syllable_counts[sid] > 0}

    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns: continue

        collapsed = rle.collapse_transitions(s)
        # Store as string serialization to allow easy parquet storage
        collapsed_str = ",".join(map(str, collapsed.astype(int)))

        # Get frame-by-frame velocity for Method 2
        vel_arr = df["velocity_px_s"].fillna(0.0).to_numpy()
        vel_str = ",".join(map(lambda x: f"{x:.2f}", vel_arr))

        # Save syllable IDs to reconstruct method 2 active bouts
        syll_arr = df["syllable"].fillna(-1).to_numpy().astype(int)
        syll_str = ",".join(map(str, syll_arr))

        rows.append({
            "session_id": s.session_id,
            "genotype": str(s.metadata.get("genotype", "WT")).upper(),
            "drug": str(s.metadata.get("drug", "Saline")),
            "collapsed_seq": collapsed_str,
            "velocity_px_s": vel_str,
            "syllables": syll_str
        })

    cache_df = pd.DataFrame(rows)
    cache_df.to_parquet(cache_path, engine="pyarrow")

    # Save global mean velocities as a metadata file
    vels_df = pd.DataFrame(list(mean_vels.items()), columns=["syllable", "mean_velocity"])
    vels_df.to_parquet(os.path.join(CACHE_DIR, f"{cache_name}_syllable_velocities.parquet"), engine="pyarrow")

    return cache_df

class MockSession:
    def __init__(self, session_id, genotype, drug):
        self.session_id = session_id
        self.metadata = {"genotype": genotype, "drug": drug}

def evaluate_candidate_m1(p: float,
                          sorted_sylls: List[int],
                          cache_df: pd.DataFrame,
                          mock_sessions: List,
                          weights: Tuple[float, float]) -> Tuple[float, float, float]:
    """ Evaluates Method 1 hyperparameter (percentile). """
    # Determine slow syllables
    cutoff_idx = max(1, int(len(sorted_sylls) * p))
    slow_set = set(sorted_sylls[:cutoff_idx])

    # Segment sentences
    segmented_sentences = {}
    for _, row in cache_df.iterrows():
        collapsed = np.array(list(map(int, row["collapsed_seq"].split(","))))

        # Fast inlined segmentation loop
        sentences = []
        current_bout = []
        for state in collapsed:
            if state in slow_set:
                if len(current_bout) > 3:
                    sentences.append(current_bout)
                current_bout = []
            else:
                current_bout.append(state)
        if len(current_bout) > 3:
            sentences.append(current_bout)

        segmented_sentences[row["session_id"]] = sentences

    return evaluate_segmentation(mock_sessions, segmented_sentences, weights)

def evaluate_candidate_m2(v_idle: float,
                          d_idle: float,
                          parsed_cache: List,
                          mock_sessions: List,
                          weights: Tuple[float, float]) -> Tuple[float, float, float]:
    """ Evaluates Method 2 hyperparameters (idle_vel, min_duration). """
    d_idle = int(np.round(d_idle))

    segmented_sentences = {}
    for session_id, vel, syllables in parsed_cache:
        is_idle = vel < v_idle

        # NumPy vectorized boundary search
        padded = np.concatenate(([False], is_idle, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        idle_epochs = []
        for s_pos, e_pos in zip(starts, ends):
            if (e_pos - s_pos) >= d_idle:
                idle_epochs.append((s_pos, e_pos))

        active_bouts = []
        last_idx = 0
        for s_pos, e_pos in idle_epochs:
            if s_pos > last_idx:
                active_bouts.append((last_idx, s_pos))
            last_idx = e_pos
        if last_idx < len(vel):
            active_bouts.append((last_idx, len(vel)))

        sentences = []
        for start, end in active_bouts:
            bout_sylls = syllables[start:end]
            bout_sylls = bout_sylls[bout_sylls != -1]
            if len(bout_sylls) == 0:
                continue

            mask = np.concatenate(([True], bout_sylls[1:] != bout_sylls[:-1]))
            collapsed = bout_sylls[mask].astype(int).tolist()
            if len(collapsed) > 3:
                sentences.append(collapsed)

        segmented_sentences[session_id] = sentences

    return evaluate_segmentation(mock_sessions, segmented_sentences, weights)

def expected_improvement(X: np.ndarray, gp: GaussianProcessRegressor, y_best: float, xi: float = 0.01) -> np.ndarray:
    """ Computes Expected Improvement (EI) for maximizing target. """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - y_best - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei

def run_grid_search_m1(sessions: List,
                       cache_name: str,
                       percentiles: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
                       weights: Tuple[float, float] = (1.2, 0.8)) -> pd.DataFrame:
    """ Sweeps percentile limits for Method 1 using cache. """
    cache_df = precompute_and_cache_dataset(sessions, cache_name)

    # Load syllable velocities
    vel_path = os.path.join(CACHE_DIR, f"{cache_name}_syllable_velocities.parquet")
    vels_df = pd.read_parquet(vel_path)
    sorted_sylls = vels_df.sort_values("mean_velocity")["syllable"].astype(int).tolist()

    # Reconstruct sessions for evaluation using MockSession
    mock_sessions = []
    for _, row in cache_df.iterrows():
        mock_sessions.append(MockSession(row["session_id"], row["genotype"], row["drug"]))

    results = []
    for p in percentiles:
        print(f"Sweeping Method 1: percentile = {p:.3f}")
        S, M1, M3 = evaluate_candidate_m1(p, sorted_sylls, cache_df, mock_sessions, weights)
        results.append({
            "percentile": p,
            "composite_score": S,
            "decodability": M1,
            "compression": M3
        })

    return pd.DataFrame(results)

def run_grid_search_m2(sessions: List,
                       cache_name: str,
                       idle_vels: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0],
                       durations: List[int] = [20, 40, 80, 120, 160],
                       weights: Tuple[float, float] = (1.2, 0.8)) -> pd.DataFrame:
    """ Sweeps idle velocity and duration limits for Method 2 using cache. """
    cache_df = precompute_and_cache_dataset(sessions, cache_name)

    mock_sessions = []
    for _, row in cache_df.iterrows():
        mock_sessions.append(MockSession(row["session_id"], row["genotype"], row["drug"]))

    # Pre-parse cache columns to numpy arrays once for speed
    parsed_cache = []
    for _, row in cache_df.iterrows():
        vels = np.array(list(map(float, row["velocity_px_s"].split(","))))
        sylls = np.array(list(map(int, row["syllables"].split(","))))
        parsed_cache.append((row["session_id"], vels, sylls))

    results = []
    for v_idle in idle_vels:
        for d_idle in durations:
            print(f"Sweeping Method 2: idle_vel = {v_idle:.1f}, min_dur = {d_idle}")
            S, M1, M3 = evaluate_candidate_m2(v_idle, d_idle, parsed_cache, mock_sessions, weights)
            results.append({
                "idle_vel": v_idle,
                "min_duration": d_idle,
                "composite_score": S,
                "decodability": M1,
                "compression": M3
            })

    return pd.DataFrame(results)

def upper_confidence_bound(X: np.ndarray, gp: GaussianProcessRegressor, beta: float = 3.0) -> np.ndarray:
    """ Computes Upper Confidence Bound (UCB) for maximizing target. """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + beta * sigma

def bayesian_optimize_m1(sessions: List,
                         cache_name: str,
                         n_iter: int = 25,
                         n_init: int = 5,
                         weights: Tuple[float, float] = (1.2, 0.8),
                         xi: float = 0.1) -> pd.DataFrame:
    """ Performs log-scale Bayesian Optimization for Method 1. """
    from scipy.optimize import minimize
    from scipy.stats import qmc

    import wandb

    # Determine run name based on weights
    if weights[0] > 0.0 and weights[1] == 0.0:
        goal_name = "decodability"
    elif weights[0] == 0.0 and weights[1] > 0.0:
        goal_name = "compression"
    else:
        goal_name = "composite"

    # Initialize wandb run
    wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name=f"method1-{goal_name}-{cache_name}",
        config={
            "method": 1,
            "goal": goal_name,
            "dataset": cache_name,
            "n_init": n_init,
            "n_iter": n_iter,
            "weights": weights,
            "xi": xi,
            "exploration_xi": xi,
            "sampler": "LHS",
            "acquisition_portfolio": "EI + UCB Hedging"
        }
    )

    cache_df = precompute_and_cache_dataset(sessions, cache_name)

    # Load syllable velocities
    vel_path = os.path.join(CACHE_DIR, f"{cache_name}_syllable_velocities.parquet")
    vels_df = pd.read_parquet(vel_path)
    sorted_sylls = vels_df.sort_values("mean_velocity")["syllable"].astype(int).tolist()

    # Reconstruct sessions for evaluation using MockSession
    mock_sessions = []
    for _, row in cache_df.iterrows():
        mock_sessions.append(MockSession(row["session_id"], row["genotype"], row["drug"]))

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
        print(f"Initial Eval {iter_num}/{n_init} (Method 1): percentile = {p:.6f} (log10 = {x_log:.3f})")
        S, M1, M3 = evaluate_candidate_m1(p, sorted_sylls, cache_df, mock_sessions, weights)

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
            "run_type": 0  # 0 for random
        })

        # Every 50th iteration, plot progress
        if iter_num % 50 == 0 or iter_num == total_iters:
            fig = plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Sequential Guided Optimization
    for i in range(n_iter):
        X_train = np.array([r["log_percentile"] for r in results]).reshape(-1, 1)
        y_train = np.array([r["composite_score"] for r in results])

        # Robust GP with lengthscales optimized over 20 restarts to avoid local minima
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=42
        )
        gp.fit(X_train, y_train)

        y_best = y_train.max()

        # Dense candidate search
        candidates = np.linspace(log_p_min, log_p_max, 5000).reshape(-1, 1)

        # Alternate acquisition function (EI vs UCB) to ensure global exploration
        if i % 2 == 0:
            acq_val = expected_improvement(candidates, gp, y_best, xi=xi)
        else:
            acq_val = upper_confidence_bound(candidates, gp, beta=3.0)

        best_idx = np.argmax(acq_val)
        best_candidate = candidates[best_idx, 0]

        # Local optimization polishing starting from the best candidate
        def obj(x):
            x_arr = np.array(x).reshape(-1, 1)
            if i % 2 == 0:
                return -float(expected_improvement(x_arr, gp, y_best, xi=xi).ravel()[0])
            else:
                return -float(upper_confidence_bound(x_arr, gp, beta=3.0).ravel()[0])

        res = minimize(obj, x0=[best_candidate], bounds=[(log_p_min, log_p_max)], method="L-BFGS-B")
        if res.success:
            next_x_log = float(res.x[0])
        else:
            next_x_log = float(best_candidate)

        next_p = float(10**next_x_log)

        iter_num = n_init + i + 1
        print(f"Guided Eval {iter_num}/{total_iters} (Method 1): percentile = {next_p:.6f} (log10 = {next_x_log:.3f})")
        S, M1, M3 = evaluate_candidate_m1(next_p, sorted_sylls, cache_df, mock_sessions, weights)

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
            "run_type": 1  # 1 for guided
        })

        # Every 50th iteration, plot progress
        if iter_num % 50 == 0 or iter_num == total_iters:
            fig = plot_m1_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Finish wandb run
    wandb.finish()
    return pd.DataFrame(results)

def plot_m1_progress_wandb(df: pd.DataFrame, current_iter: int, total_iters: int):
    """ Generates Method 1 progress sweep plot for wandb with progress in the title. """
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9, 6))

    df_sorted = df.sort_values("percentile")
    ax.plot(df_sorted["percentile"], df_sorted["composite_score"], color="royalblue", alpha=0.5, linestyle="-")

    random_pts = df[df["type"] == "random"]
    guided_pts = df[df["type"] == "guided"]

    ax.scatter(random_pts["percentile"], random_pts["composite_score"], color="dimgray", marker="o", s=80, alpha=0.8, label="Initial Random Points")
    if not guided_pts.empty:
        ax.scatter(guided_pts["percentile"], guided_pts["composite_score"], color="forestgreen", marker="D", s=80, alpha=0.8, label="GP Guided Points")

    opt_idx = df["composite_score"].idxmax()
    opt_p = df.loc[opt_idx, "percentile"]
    opt_s = df.loc[opt_idx, "composite_score"]
    ax.scatter([opt_p], [opt_s], color='red', marker='*', s=250, zorder=10, label=f"Optimal (p={opt_p:.6f}, S={opt_s:.3f})")

    ax.set_xscale("log")
    ax.set_title(f"Method 1 (Slow Syllables Percentile) Sweep - Progress: {current_iter}/{total_iters} iters")
    ax.set_xlabel("Percentile Cutoff (Log Scale)")
    ax.set_ylabel("Score")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    return fig

def bayesian_optimize_m2(sessions: List,
                         cache_name: str,
                         n_iter: int = 25,
                         n_init: int = 5,
                         weights: Tuple[float, float] = (1.2, 0.8),
                         xi: float = 0.1) -> pd.DataFrame:
    """ Performs log-scale Bayesian Optimization for Method 2. """
    from scipy.optimize import minimize
    from scipy.stats import qmc

    import wandb

    # Determine run name based on weights
    if weights[0] > 0.0 and weights[1] == 0.0:
        goal_name = "decodability"
    elif weights[0] == 0.0 and weights[1] > 0.0:
        goal_name = "compression"
    else:
        goal_name = "composite"

    # Initialize wandb run
    wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name=f"method2-{goal_name}-{cache_name}",
        config={
            "method": 2,
            "goal": goal_name,
            "dataset": cache_name,
            "n_init": n_init,
            "n_iter": n_iter,
            "weights": weights,
            "xi": xi,
            "exploration_xi": xi,
            "sampler": "LHS",
            "acquisition_portfolio": "EI + UCB Hedging"
        }
    )

    cache_df = precompute_and_cache_dataset(sessions, cache_name)

    mock_sessions = []
    for _, row in cache_df.iterrows():
        mock_sessions.append(MockSession(row["session_id"], row["genotype"], row["drug"]))

    # Pre-parse cache columns to numpy arrays once for speed
    parsed_cache = []
    for _, row in cache_df.iterrows():
        vels = np.array(list(map(float, row["velocity_px_s"].split(","))))
        sylls = np.array(list(map(int, row["syllables"].split(","))))
        parsed_cache.append((row["session_id"], vels, sylls))

    # Bounds in log10 space
    log_v_min = np.log10(0.005)
    log_v_max = np.log10(300.0)
    log_d_min = np.log10(1.0)
    log_d_max = np.log10(10000.0)

    # Initial sampling using Latin Hypercube Sampling (LHS)
    sampler = qmc.LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=n_init)
    X_log_v = sample[:, 0] * (log_v_max - log_v_min) + log_v_min
    X_log_d = sample[:, 1] * (log_d_max - log_d_min) + log_d_min

    results = []
    total_iters = n_init + n_iter
    rng = np.random.RandomState(42)

    # Evaluate initial points
    for idx in range(n_init):
        v = float(10**X_log_v[idx])
        d = int(np.round(10**X_log_d[idx]))
        iter_num = idx + 1
        print(f"Initial Eval {iter_num}/{n_init} (Method 2): idle_vel = {v:.3f}, min_duration = {d} frames")
        S, M1, M3 = evaluate_candidate_m2(v, d, parsed_cache, mock_sessions, weights)

        row_data = {
            "iteration": iter_num,
            "idle_vel": v,
            "min_duration": d,
            "log_idle_vel": X_log_v[idx],
            "log_min_duration": X_log_d[idx],
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "type": "random"
        }
        results.append(row_data)

        # Log to wandb
        wandb.log({
            "iteration": iter_num,
            "idle_vel": v,
            "min_duration": d,
            "log_idle_vel": X_log_v[idx],
            "log_min_duration": X_log_d[idx],
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 0  # 0 for random
        })

        # Every 50th iteration, plot progress
        if iter_num % 50 == 0 or iter_num == total_iters:
            fig = plot_m2_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Sequential Guided Optimization
    for i in range(n_iter):
        X_train = np.array([[r["log_idle_vel"], r["log_min_duration"]] for r in results])
        y_train = np.array([r["composite_score"] for r in results])

        # Robust GP with lengthscales optimized over 20 restarts to avoid local minima
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2)),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=42
        )
        gp.fit(X_train, y_train)

        y_best = y_train.max()

        # Dense candidate search via random sampling in bounds
        candidates_v = rng.uniform(log_v_min, log_v_max, size=5000)
        candidates_d = rng.uniform(log_d_min, log_d_max, size=5000)
        candidates = np.column_stack((candidates_v, candidates_d))

        # Alternate acquisition function (EI vs UCB) to ensure global exploration
        if i % 2 == 0:
            acq_val = expected_improvement(candidates, gp, y_best, xi=xi)
        else:
            acq_val = upper_confidence_bound(candidates, gp, beta=3.0)

        best_idx = np.argmax(acq_val)
        best_candidate = candidates[best_idx]

        # Local optimization polishing starting from the best candidate
        def obj(x):
            x_arr = np.array(x).reshape(1, 2)
            if i % 2 == 0:
                return -float(expected_improvement(x_arr, gp, y_best, xi=xi).ravel()[0])
            else:
                return -float(upper_confidence_bound(x_arr, gp, beta=3.0).ravel()[0])

        res = minimize(obj, x0=best_candidate, bounds=[(log_v_min, log_v_max), (log_d_min, log_d_max)], method="L-BFGS-B")
        if res.success:
            next_v_log = float(res.x[0])
            next_d_log = float(res.x[1])
        else:
            next_v_log = float(best_candidate[0])
            next_d_log = float(best_candidate[1])

        next_v = float(10**next_v_log)
        next_d = int(np.round(10**next_d_log))

        iter_num = n_init + i + 1
        print(f"Guided Eval {iter_num}/{total_iters} (Method 2): idle_vel = {next_v:.3f}, min_duration = {next_d} frames (log10 = [{next_v_log:.3f}, {next_d_log:.3f}])")
        S, M1, M3 = evaluate_candidate_m2(next_v, next_d, parsed_cache, mock_sessions, weights)

        row_data = {
            "iteration": iter_num,
            "idle_vel": next_v,
            "min_duration": next_d,
            "log_idle_vel": next_v_log,
            "log_min_duration": next_d_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "type": "guided"
        }
        results.append(row_data)

        # Log to wandb
        wandb.log({
            "iteration": iter_num,
            "idle_vel": next_v,
            "min_duration": next_d,
            "log_idle_vel": next_v_log,
            "log_min_duration": next_d_log,
            "composite_score": S,
            "decodability": M1,
            "compression": M3,
            "run_type": 1  # 1 for guided
        })

        # Every 50th iteration, plot progress
        if iter_num % 50 == 0 or iter_num == total_iters:
            fig = plot_m2_progress_wandb(pd.DataFrame(results), iter_num, total_iters)
            wandb.log({"optimization_sweep_plot": wandb.Image(fig)}, step=iter_num)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Finish wandb run
    wandb.finish()
    return pd.DataFrame(results)

def plot_m2_progress_wandb(df: pd.DataFrame, current_iter: int, total_iters: int):
    """ Generates Method 2 progress sweep plot for wandb with progress in the title. """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(df["idle_vel"], df["min_duration"], color="gray", linestyle="--", alpha=0.4, zorder=1)

    sc = ax.scatter(df["idle_vel"], df["min_duration"], c=df["composite_score"], cmap="viridis", s=120, edgecolors='black', zorder=2)
    fig.colorbar(sc, ax=ax, label="Composite Score S")

    for _, row in df.iterrows():
        ax.text(row["idle_vel"] * 1.05, row["min_duration"] * 1.05, str(int(row["iteration"])), fontsize=8, color="black", weight="bold")

    opt_idx = df["composite_score"].idxmax()
    opt_v = df.loc[opt_idx, "idle_vel"]
    opt_d = df.loc[opt_idx, "min_duration"]
    opt_s = df.loc[opt_idx, "composite_score"]

    ax.scatter([opt_v], [opt_d], color='red', marker='*', s=350, zorder=10, edgecolors='black', label=f"Optimal (vel={opt_v:.3f}, dur={opt_d}, S={opt_s:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Method 2 (Kinematic Idleness) 2D Search Trajectory - Progress: {current_iter}/{total_iters} iters")
    ax.set_xlabel("Idle Velocity Cutoff (px/s, Log Scale)")
    ax.set_ylabel("Minimum Idle Duration (Frames, Log Scale)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig
