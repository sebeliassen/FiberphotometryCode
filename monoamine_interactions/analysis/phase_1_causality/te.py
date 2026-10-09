

import numpy as np

# Note: Ideally use JIDT or IDTxl for robust TE, but that requires Java.
# Here we implement a simple k-nearest neighbor (KSG) estimator for continuous TE
# based on Kraskov et al. (2004) or a discretized binned version.
# For simplicity in pure Python without Java, let's use a discretized approach OR
# a simple Gaussian approximation TE = 0.5 * log( det(Cov(Y_past))/det(Cov(Y_future|Y_past, X_past)) )
# Let's start with Gaussian TE as a baseline (fast, model-based).
# For non-linear, we'd need non-parametric estimators (e.g. from 'nolitsa' or similar).

def compute_transfer_entropy_gaussian(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1
) -> float:
    """
    Computes Transfer Entropy via Gaussian approximation (Linear TE, effectively Granger).
    TE_gauss(X->Y) = 0.5 * log ( Var(Y_future | Y_past) / Var(Y_future | Y_past, X_past) )
    
    This is equivalent to comparison of prediction error variances in AR vs VAR models.
    """

    n = len(source)
    if n < lag + 10:
        return 0.0

    # Build lagged matrices
    # Y_future at t
    y_fut = target[lag:]

    # Y_past at t-lag
    y_past = target[:-lag]

    # X_past at t-lag
    x_past = source[:-lag]

    # 1. Reduced Model: Y_fut ~ Y_past
    # Simple linear regression: y_fut = a*y_past + e1
    # We can use np.linalg.lstsq
    A_reduced = np.vstack([y_past, np.ones(len(y_past))]).T
    res_reduced = np.linalg.lstsq(A_reduced, y_fut, rcond=None)[1]

    # If perfect fit or empty
    if not res_reduced.size:
        var_reduced = np.var(y_fut) # Fallback
    else:
        var_reduced = res_reduced[0] / len(y_fut)

    # 2. Full Model: Y_fut ~ Y_past + X_past
    # y_fut = a*y_past + b*x_past + e2
    A_full = np.vstack([y_past, x_past, np.ones(len(y_past))]).T
    res_full = np.linalg.lstsq(A_full, y_fut, rcond=None)[1]

    if not res_full.size:
        var_full = var_reduced # No improvement
    else:
        var_full = res_full[0] / len(y_fut)

    # 3. TE = 0.5 * log(Var_reduced / Var_full)
    if var_full <= 0 or var_reduced <= 0:
        return 0.0

    te = 0.5 * np.log(var_reduced / var_full)
    return max(0.0, te)  # TE is non-negative
