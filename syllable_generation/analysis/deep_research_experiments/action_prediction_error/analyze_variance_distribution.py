import pandas as pd
import numpy as np
import os

out_dir = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/deep_research_experiments/action_prediction_error/output"

def analyze_set(suffix):
    csv_path = os.path.join(out_dir, f"ppe_results{suffix}.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_path}: File not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Check if we have the mean_abs column
    # If using metric='mean', the 'slope_variance' column contains the mean_abs data!
    # If using metric='variance', the 'slope_mean_abs' column contains the mean_abs data.
    # To be safe, we'll check for 'slope_mean_abs' first.
    
    target_metrics = []
    if 'slope_variance_raw' in df.columns: target_metrics.append('slope_variance_raw')
    if 'slope_variance_resid' in df.columns: target_metrics.append('slope_variance_resid')
    if 'slope_mean_abs' in df.columns: target_metrics.append('slope_mean_abs')
    
    stats = []
    for m in target_metrics:
        grouped = df.groupby(['genotype', 'drug'])[m]
        summary = grouped.agg(['mean', 'var', 'std', 'count']).reset_index()
        summary.rename(columns={
            'mean': f'Mean_{m}',
            'var': f'Var_{m}',
            'std': f'Std_{m}'
        }, inplace=True)
        stats.append(summary)
        
    if not stats: return
    
    final_df = stats[0]
    for other in stats[1:]:
        final_df = pd.merge(final_df, other, on=['genotype', 'drug', 'count'])
        
    out_path = os.path.join(out_dir, f"variance_mean_dist_stats{suffix}.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Saved stats to {out_path}")

# Run for both sets
analyze_set("_no_warp")
analyze_set("_warp")
