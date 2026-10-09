"""
Kinematic Reversal Test (Double Activity Project)
==================================================
For each target syllable in the ORIGINAL orientation:
  1. Find all onset timestamps.
  2. Measure the ACTUAL physical angular velocity at those timestamps.
  3. Find what syllable the FLIPPED model assigned at those same timestamps.
  4. Measure the global mean AV of those flipped syllables.

If MoSeq learned spatial parity, a "Right Turn" original syllable should
map to a "Left Turn" flipped syllable, and vice versa.

Methodological improvement: We also directly compare the instance-level
physical AV to avoid dilution from the global syllable mean lookup.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')

pkl_path = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_double_w_sylls.pkl'
print("Loading double flip pickle...")
sessions = pd.read_pickle(pkl_path)
print(f"Loaded {len(sessions)} sessions.\n")

# --- Step 1: Compute global mean AV per syllable (original) ---
all_sylls = []
all_av = []
av_col_name = None

for s in sessions:
    df = s.all_data_df
    av_col = [c for c in df.columns if 'angular_velocity' in c]
    if not av_col:
        if 'heading' in df.columns:
            df['computed_av'] = df['heading'].diff()
            df['computed_av'] = (df['computed_av'] + np.pi) % (2 * np.pi) - np.pi
            av_col_name = 'computed_av'
        else:
            continue
    else:
        av_col_name = av_col[0]

    valid = ~df['syllable'].isna() & ~df[av_col_name].isna()
    all_sylls.append(df.loc[valid, 'syllable'].values)
    all_av.append(df.loc[valid, av_col_name].values)

s_arr = np.concatenate(all_sylls)
av_arr = np.concatenate(all_av)
df_global = pd.DataFrame({'syllable': s_arr, 'av': av_arr})
mean_av = df_global.groupby('syllable')['av'].mean()

# --- Step 2: For each target syllable, do the timestamp-matched test ---
TARGET_SYLLS = [13, 18, 20, 28, 36, 40, 41, 42]

print(f"Target Syllables: {TARGET_SYLLS}\n")
print(f"{'Syll':>4} | {'N':>6} | {'Orig Mean AV':>12} | {'Flipped Syll Mean AV':>20} | {'Instance AV at t':>16} | {'Sign Flip?':>10}")
print("-" * 85)

results_orig = []
results_mapped = []

for TARGET_SYLL in TARGET_SYLLS:
    mapped_global_avs = []      # global mean AV of the flipped syllable
    instance_avs_at_t = []      # actual physical AV at each onset timestamp

    for s in sessions:
        df = s.all_data_df
        if 'syllable_flipped' not in df.columns:
            continue

        # Find timestamps where original syllable == TARGET
        mask = df['syllable'] == TARGET_SYLL
        if not mask.any():
            continue

        # At those timestamps, get:
        # (a) the flipped syllable ID
        # (b) the actual physical angular velocity
        for idx in df.index[mask]:
            fsyll = df.at[idx, 'syllable_flipped']
            if pd.isna(fsyll):
                continue
            # (a) global mean AV of that flipped syllable
            if fsyll in mean_av:
                mapped_global_avs.append(mean_av[fsyll])
            # (b) actual physical AV at this exact timestamp
            if av_col_name and av_col_name in df.columns:
                av_val = df.at[idx, av_col_name]
                if not np.isnan(av_val):
                    instance_avs_at_t.append(av_val)

    if len(mapped_global_avs) == 0:
        continue

    orig_av = mean_av.get(TARGET_SYLL, 0)
    mapped_av = np.mean(mapped_global_avs)
    instance_av = np.mean(instance_avs_at_t) if instance_avs_at_t else float('nan')

    # Check if sign flipped
    sign_flip = "✅ YES" if (orig_av * mapped_av < 0) else "❌ NO"

    results_orig.append(orig_av)
    results_mapped.append(mapped_av)

    print(f"{TARGET_SYLL:4.0f} | {len(mapped_global_avs):6d} | {orig_av:12.4f} | {mapped_av:20.4f} | {instance_av:16.4f} | {sign_flip:>10}")

# --- Step 3: Pearson correlation ---
orig_avs = np.array(results_orig)
mapped_avs = np.array(results_mapped)

if len(orig_avs) >= 3:
    r, p = stats.pearsonr(orig_avs, mapped_avs)

    print(f"\n{'='*60}")
    print(f"PEARSON CORRELATION TEST (n={len(orig_avs)} syllables)")
    print(f"{'='*60}")
    print(f"r = {r:.4f}, p = {p:.4e}")

    if r < -0.7 and p < 0.05:
        print("RESULT: Strong inverse correlation. MoSeq learned spatial parity.")
    elif r < 0 and p < 0.05:
        print("RESULT: Significant negative correlation. Spatial parity partially learned.")
    else:
        print("RESULT: No significant inverse correlation detected.")

    # Spearman (rank-based, robust to outliers)
    rho, p_s = stats.spearmanr(orig_avs, mapped_avs)
    print("\nSpearman rank correlation (robust to outliers):")
    print(f"rho = {rho:.4f}, p = {p_s:.4e}")
else:
    print("\nNot enough data points for correlation test.")
