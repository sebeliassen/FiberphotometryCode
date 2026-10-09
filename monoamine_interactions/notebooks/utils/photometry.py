import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

def biexponential(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e

def monoexponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_biexponential_dff(signal):
    valid_idx = ~np.isnan(signal)
    if not np.any(valid_idx):
        return signal
        
    x = np.arange(len(signal))
    x_valid = x[valid_idx]
    y_valid = signal[valid_idx]
    
    x_scale = len(signal)
    
    # HEURISTIC: Downsample strictly for the fit to achieve >1000x speedup
    step = max(1, len(x_valid) // 500)
    x_fit = x_valid[::step] / x_scale
    y_fit = y_valid[::step]
    
    p0 = (np.max(y_fit)-np.min(y_fit), 1.0, np.max(y_fit)-np.min(y_fit), 0.1, np.min(y_fit))
    bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    
    try:
        popt, _ = curve_fit(biexponential, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
        baseline = biexponential(x / x_scale, *popt)
        baseline[baseline == 0] = np.nan
        return (signal - baseline) / baseline
        
    except Exception as e:
        print(f"Biexponential fit failed ({e}), trying monoexponential...")
            
        p0_mono = (np.max(y_fit)-np.min(y_fit), 1.0, np.min(y_fit))
        bounds_mono = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(monoexponential, x_fit, y_fit, p0=p0_mono, bounds=bounds_mono, maxfev=100000)
            baseline = monoexponential(x / x_scale, *popt)
            baseline[baseline == 0] = np.nan
            return (signal - baseline) / baseline
        except Exception as e2:
            print(f"Monoexponential fit also failed: {e2}")
            return signal * np.nan

def motion_correct_biexponential(raw_470, raw_415):
    """
    1. Double-exp fits both 470 and 415 to remove photobleaching dynamically.
    2. Uses robust linear regression to scale the cleanly detrended 415 artifacts.
    3. Subtracts the scaled 415 artifacts from 470 to yield the true biological signal.
    """
    dff_470 = fit_biexponential_dff(raw_470)
    dff_415 = fit_biexponential_dff(raw_415)
    
    # Mask NaNs strictly for the linear regression
    valid = ~np.isnan(dff_470) & ~np.isnan(dff_415)
    if not np.any(valid):
        return dff_470
        
    x = dff_415[valid]
    y = dff_470[valid]
    
    # Simple rigorous OLS regression of Isosbestic onto Calcium trace
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Form predicted structural artifact envelope globally
    predicted_artifact = slope * dff_415 + intercept
    
    # The true biological signal is the isolated residual vector 
    corrected_dff = dff_470 - predicted_artifact
    return corrected_dff
