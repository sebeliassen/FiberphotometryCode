"""
Boilerplate, skeleton, and mathematical scaffolding for the 5 sliding-window optimization metrics.
These functions define the interface and core algorithms to search for optimal window size and POS groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import entropy

def evaluate_discriminability_vs_coherence(
    features: np.ndarray, 
    strategy_labels: np.ndarray, 
    step_size_sec: float,
    window_size_sec: float
) -> Dict[str, float]:
    """
    Metric 1: Twofold Crossing (Discriminability vs. Non-Overlapping Coherence).
    
    Args:
        features: (N_windows, N_features) array of normalized window feature vectors.
        strategy_labels: (N_windows,) array of strategy IDs assigned to each window.
        step_size_sec: Time in seconds between consecutive windows (e.g. 1.0s).
        window_size_sec: Window length in seconds (e.g. 15.0s).
        
    Returns:
        Dict containing:
            - "discriminability": Normalized inter-window variance.
            - "coherence": Temporal persistence of strategy states.
            - "coherence_null": Baseline persistence on shuffled data.
            - "coherence_corrected": Chance-corrected coherence (Cohen's Kappa equivalent).
            - "composite_f1": Harmonic mean of discriminability and corrected coherence.
    """
    # 1. Compute Discriminability: trace of the covariance matrix
    # As W increases, windows average together and Cov(X) trace drops to 0.
    if features.shape[0] > 1:
        cov_matrix = np.cov(features, rowvar=False)
        discriminability = float(np.trace(cov_matrix))
    else:
        discriminability = 0.0

    # 2. Compute Non-Overlapping Coherence
    # We sample adjacent disjoint windows to avoid overlap artifacts.
    # Step size is step_size_sec, window is window_size_sec, so window overlap factor is:
    stride = int(np.round(window_size_sec / step_size_sec))
    if stride < 1:
        stride = 1
        
    disjoint_labels = strategy_labels[::stride]
    
    if len(disjoint_labels) > 1:
        # Calculate transition self-probability P(S_{t+1} == S_t)
        transitions = disjoint_labels[:-1] == disjoint_labels[1:]
        coherence = float(np.mean(transitions))
        
        # Calculate Shuffled Null Control
        shuffled_labels = disjoint_labels.copy()
        np.random.shuffle(shuffled_labels)
        shuffled_transitions = shuffled_labels[:-1] == shuffled_labels[1:]
        coherence_null = float(np.mean(shuffled_transitions))
    else:
        coherence = 0.0
        coherence_null = 0.0

    # 3. Compute Chance-Corrected Coherence (Cohen's Kappa logic)
    if 1.0 - coherence_null > 1e-6:
        coherence_corrected = (coherence - coherence_null) / (1.0 - coherence_null)
    else:
        coherence_corrected = 0.0
        
    # Standardize/normalize values to [0, 1] range for composite score calculation
    # (In the execution script, these will be scaled relative to the sweep boundaries).
    d_scaled = discriminability  # Placeholder for min-max scaling
    c_scaled = max(0.0, coherence_corrected)
    
    if d_scaled + c_scaled > 1e-6:
        composite_f1 = 2 * (d_scaled * c_scaled) / (d_scaled + c_scaled)
    else:
        composite_f1 = 0.0

    return {
        "discriminability": discriminability,
        "coherence": coherence,
        "coherence_null": coherence_null,
        "coherence_corrected": coherence_corrected,
        "composite_f1": composite_f1
    }

def evaluate_downstream_decodability(
    strategy_proportions: np.ndarray, 
    cohort_labels: np.ndarray
) -> float:
    """
    Metric 2: Genotype or Drug Decodability.
    
    Args:
        strategy_proportions: (N_sessions, K_strategies) array representing 
                              the normalized strategy abundance histogram per session.
        cohort_labels: (N_sessions,) binary array of labels (e.g. 0=WT, 1=KI or 0=Saline, 1=AMPH).
        
    Returns:
        float: Mean cross-validated F1-score or prediction accuracy.
    """
    if len(np.unique(cohort_labels)) < 2:
        return 0.0
        
    # Standardize inputs
    X = strategy_proportions
    y = cohort_labels
    
    skf = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Simple Logistic Regression classifier with L2 regularization
        clf = LogisticRegression(C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        scores.append(f1_score(y_test, preds, average="weighted"))
        
    return float(np.mean(scores))

def evaluate_dopamine_alignment(
    strategy_transition_events: np.ndarray, 
    dopamine_signal: np.ndarray,
    fs: float = 80.0
) -> float:
    """
    Metric 3: Neural Photometry Cross-Correlation Alignment.
    
    Args:
        strategy_transition_events: (N_frames,) binary indicator array where 
                                    1.0 indicates a strategy switch.
        dopamine_signal: (N_frames,) continuous z-scored dLight/GRAB-DA signal.
        fs: Photometry and tracking sampling rate (Hz).
        
    Returns:
        float: Peak cross-correlation magnitude within a [-3s, 3s] physiological lag window.
    """
    # Standardize lengths
    n_frames = min(len(strategy_transition_events), len(dopamine_signal))
    transitions = strategy_transition_events[:n_frames]
    da = dopamine_signal[:n_frames]
    
    if np.sum(transitions) == 0 or np.std(da) == 0:
        return 0.0
        
    # Find peak correlation within physiological delay window (-3 to +3 seconds)
    lag_limit_frames = int(3.0 * fs)
    lags = np.arange(-lag_limit_frames, lag_limit_frames + 1)
    correlations = []
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(transitions[-lag:], da[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(transitions[:-lag], da[lag:])[0, 1]
        else:
            corr = np.corrcoef(transitions, da)[0, 1]
            
        correlations.append(corr if not np.isnan(corr) else 0.0)
        
    # Return peak correlation magnitude
    return float(np.max(np.abs(correlations)))

def evaluate_kl_information_gain(
    window_syllables: List[List[int]], 
    vocab_size: int = 40
) -> float:
    """
    Metric 4: Syllable Syntactic Information Gain (KL Divergence relative to baseline).
    Operates directly on Stage 0 raw data (no embeddings or clustering needed).
    
    Args:
        window_syllables: List of syllable lists (each sublist represents syllables in a window).
        vocab_size: The total vocabulary size of syllable IDs (typically max_syll_id + 1).
        
    Returns:
        float: Mean KL divergence across all windows relative to the global frequency baseline.
    """
    if not window_syllables:
        return 0.0
        
    # 1. Compute global baseline syllable distribution
    global_counts = np.zeros(vocab_size)
    for win in window_syllables:
        for syll in win:
            if 0 <= syll < vocab_size:
                global_counts[syll] += 1
                
    total_global = np.sum(global_counts)
    if total_global == 0:
        return 0.0
    global_dist = (global_counts + 1e-5) / (total_global + 1e-5 * vocab_size)
    
    # 2. Compute KL divergence for each window
    kl_values = []
    for win in window_syllables:
        if not win:
            continue
        win_counts = np.zeros(vocab_size)
        for syll in win:
            if 0 <= syll < vocab_size:
                win_counts[syll] += 1
        win_dist = (win_counts + 1e-5) / (len(win) + 1e-5 * vocab_size)
        
        # KL Divergence: sum(P * log(P / Q))
        kl = entropy(win_dist, global_dist)
        kl_values.append(kl)
        
    return float(np.mean(kl_values)) if kl_values else 0.0

def evaluate_transition_predictability(
    strategy_labels: np.ndarray,
    n_strategies: int = 8
) -> float:
    """
    Metric 5: Strategy Markov Transition Predictability (State Perplexity).
    
    Args:
        strategy_labels: (N_windows,) array of strategy IDs assigned to windows.
        n_strategies: Number of discrete strategy clusters.
        
    Returns:
        float: Transition perplexity ratio relative to a shuffled sequence control.
               Lower values indicate a more structured, predictable behavior sequence.
    """
    if len(strategy_labels) < 2:
        return 1.0
        
    def compute_perplexity(labels: np.ndarray) -> float:
        # Build first-order transition matrix
        counts = np.zeros((n_strategies, n_strategies))
        for t in range(len(labels) - 1):
            s_curr, s_next = labels[t], labels[t+1]
            if 0 <= s_curr < n_strategies and 0 <= s_next < n_strategies:
                counts[s_curr, s_next] += 1
                
        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.zeros_like(counts)
        np.divide(counts, row_sums, out=probs, where=row_sums > 0)
        
        # Calculate sequence log-likelihood
        log_lik = 0.0
        total_transitions = 0
        for t in range(len(labels) - 1):
            s_curr, s_next = labels[t], labels[t+1]
            if 0 <= s_curr < n_strategies and 0 <= s_next < n_strategies:
                p = probs[s_curr, s_next]
                if p > 0:
                    log_lik += np.log(p)
                    total_transitions += 1
                    
        if total_transitions == 0:
            return float(n_strategies)
            
        mean_log_lik = log_lik / total_transitions
        return float(np.exp(-mean_log_lik))

    # 1. Compute empirical perplexity
    perplexity_emp = compute_perplexity(strategy_labels)
    
    # 2. Compute shuffled control perplexity
    shuffled_labels = strategy_labels.copy()
    np.random.shuffle(shuffled_labels)
    perplexity_shuf = compute_perplexity(shuffled_labels)
    
    # Return ratio: lower ratio means the sequence has structured transition grammar
    if perplexity_shuf > 0:
        return perplexity_emp / perplexity_shuf
    return 1.0
