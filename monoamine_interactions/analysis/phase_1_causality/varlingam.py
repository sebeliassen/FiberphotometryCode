
import pandas as pd
import numpy as np
import lingam
from lingam.utils import make_dot

def run_varlingam(
    df: pd.DataFrame, 
    lags: int = 1,
    criterion: str = 'bic'
) -> dict:
    """
    Runs Vector Autoregressive LiNGAM on multi-channel data.
    
    Args:
        df: DataFrame where columns are the time series (e.g. VTA_NE, NAc_DA)
        lags: Number of lags to include in VAR model.
        criterion: Model selection criterion (aic/bic).
        
    Returns:
        Dictionary containing:
        - 'adjacency_matrix': Directed causal graph (lag 0).
        - 'causal_order': Inferred causal ordering.
        - 'residuals': Residuals of the VAR model.
        - 'model': The fitted VARLiNGAM object.
    """
    
    # 1. Fit VARLiNGAM
    model = lingam.VARLiNGAM(lags=lags, criterion=criterion)
    model.fit(df)
    
    # 2. Extract results
    # adjacency_matrices[0] is contemporaneous (B0)
    # adjacency_matrices[k] is lagged (Bk)
    adj_matrices = model.adjacency_matrices_
    causal_order = model.causal_order_
    
    return {
        "adjacency_matrices": adj_matrices,
        "causal_order": causal_order,
        "residuals": model.residuals_,
        "model": model
    }
