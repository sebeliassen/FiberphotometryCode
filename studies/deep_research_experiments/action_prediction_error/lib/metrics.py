import numpy as np


def calculate_derivative(trace: np.ndarray, dt: float) -> np.ndarray:
    """
    Calculate the first derivative of the trace (dF/dt).
    """
    return np.gradient(trace, dt)

def get_onset_slopes(
    traces: np.ndarray,
    window_frames: int
) -> dict:
    """
    Calculate slope metrics for a set of traces aligned to onset.
    traces: (N_trials, T_time) array.
    window_frames: Number of frames post-onset to consider for slope variance.
    
    Returns:
       dict with keys 'mean_slope_var', 'mean_slope_mag'
    """
    # traces should be sliced to the window of interest [0, window_frames]
    # But usually we want to derivatives of the whole trace first?
    # Or derivative of the slice?
    # Derivative is local, so we can compute it on the slice.

    # Assume traces are already dF/F or z-score.
    # We just want the slope over time.

    # Simple linear fit slope? Or instantaneous slope (derivative) variance?
    # The proposal says: "Variance of the DLS trace slope (dF/dt) at syllable onset"
    # This implies we compute dF/dt(t) for each trial, then take the variance across trials?
    # OR variance over time?
    # "KI mice will show higher slope variance for identical syllables"
    # This implies Variance ACROSS TRIALS (Action Prediction Error variability).

    # So:
    # 1. Compute dF/dt for each trial.
    # 2. Extract the value at t=0 (or mean over small window).
    # 3. Compute variance of these values across trials.

    # Let's do simply: value at t=0 to t=window_frames mean slope.

    # Let's take the first derivative.
    slopes = np.gradient(traces, axis=1) # along time axis

    # Slice window
    slopes_window = slopes[:, :window_frames]

    # Mean slope per trial in this window
    trial_mean_slopes = np.mean(slopes_window, axis=1)

    # Variance of these mean slopes across trials
    slope_var = np.var(trial_mean_slopes)

    # Mean magnitude
    slope_mag = np.mean(trial_mean_slopes)

    n_trials = len(traces)

    return {
        'slope_variance': np.var(slopes),
        'slope_mean': np.mean(slopes),
        'n_trials': n_trials,
        'slopes': slopes
    }

def get_residual_variance(df_trials):
    """
    Calculate variance of residuals after regressing Slope ~ History + Velocity + Accel.
    df_trials: DataFrame with columns ['slope', 'history', 'velocity', 'acceleration']
    """
    from sklearn.linear_model import LinearRegression

    # Drop NaNs
    df_clean = df_trials.dropna()

    if len(df_clean) < 10:
        # Not enough data for regression
        return {
            'residual_variance': np.var(df_trials['slope']), # Fallback to raw
            'r_squared': 0.0,
            'n_reg': len(df_clean)
        }

    X = df_clean[['history', 'velocity', 'acceleration']]
    y = df_clean['slope']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    residuals = y - predictions

    return {
        'residual_variance': np.var(residuals),
        'r_squared': model.score(X, y),
        'n_reg': len(df_clean)
    }
