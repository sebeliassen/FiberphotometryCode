
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Any
import skccm
from skccm.utilities import train_test_split as skccm_split

def run_ccm_analysis(
    source_trace: np.ndarray,
    target_trace: np.ndarray,
    embed_dim: int = 3,
    lag: int = 1,
    predict_lag: int = 0
) -> Tuple[float, float, Any]:
    """
    Runs CCM to see if source causes target (i.e. check if target predicts source).
    In CCM logic: If Source -> Target (Source drives Target), 
    then Target contains info about Source.
    Thus we use TARGET shadow manifold to predict SOURCE.
    
    Returns:
        (rho_source_xmap_target, rho_target_xmap_source, details)
    """
    
    # Embeddings
    e1 = skccm.Embed(source_trace)
    e2 = skccm.Embed(target_trace)
    
    X1 = e1.embed_vectors_1d(lag, embed_dim)
    X2 = e2.embed_vectors_1d(lag, embed_dim)

    # Split train/test
    # For robust CCM, we often just use 100% or split 0.75
    split = int(0.9 * len(X1))
    X1_tr, X1_te, X2_tr, X2_te = X1[:split], X1[split:], X2[:split], X2[split:]

    ccm = skccm.CCM() # Initialize

    # Library lengths
    len_tr = len(X1_tr)
    lib_lens = np.arange(100, len_tr, 200) # Check convergence over length
    
    ccm.fit(X1_tr, X2_tr)
    x1p, x2p = ccm.predict(X1_te, X2_te, lib_lengths=lib_lens)

    # Scores
    # x1p result: Using X1 (Source) to predict X2 (Target) -> implies Target drives Source?
    # x2p result: Using X2 (Target) to predict X1 (Source) -> implies Source drives Target?
    
    sc1, sc2 = ccm.score() # Returns list of R^2 or correlation? Docs say score() returns score for each lib_len
    
    # We take the max score at max lib length as our 'converged' metric
    # sc1: X1 xmap X2 (How well X1 predicts X2?)
    # sc2: X2 xmap X1 (How well X2 predicts X1?)
    
    # Checking convergence: does skill increase with L?
    # We return the final skill (max L)
    
    final_rho_1_xmap_2 = sc1[-1] # X1 predicts X2 (T -> S)
    final_rho_2_xmap_1 = sc2[-1] # X2 predicts X1 (S -> T) <-- This is "Source causes Target" in CCM terms

    return final_rho_1_xmap_2, final_rho_2_xmap_1, {"sc1": sc1, "sc2": sc2, "libs": lib_lens}
