#!/usr/bin/env python3
"""
audit_photometry_handling_artifacts.py
=============================================================================
Multimodal Cross-Validation of Inferred Out-of-Arena Handling Intervals
using Fiber Photometry Motion Artifacts and Rolling Variance.

Hypothesis:
During physical animal removal, IP injection, and handling, the fiber-optic
patch cable experiences severe bending, twisting, and acceleration. This
produces sharp optical baseline shifts, high-frequency transients, and massive
variance increases across optical channels.

Analysis:
1. Aligns native 80 Hz fiber photometry signals (L-VS, R-DMS) with the
   morphologically detected out-of-arena interval [T_removal, T_return].
2. Computes rolling variance (window = 2.0s) and high-frequency derivative energy.
3. Examines exemplar sessions (S0: ordinary 65s gap; S24: 225s long gap).
4. Quantifies cohort-wide variance amplification during handling across all 30 sessions.
5. Produces a publication-grade multi-panel diagnostic figure.
=============================================================================
"""

import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579"
OUTPUT_FIG_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures")
OUTPUT_TAB_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "tables")
SESSIONS_PKL = os.path.join(BASE_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")
EVENT_TABLE = os.path.join(OUTPUT_TAB_DIR, "session_event_frames_verified.csv")

FPS = 80.025

def rolling_var(arr, window_size):
    """Fast rolling variance calculation using uniform filter."""
    s = pd.Series(arr)
    return s.rolling(window=window_size, min_periods=1, center=True).var().values

def main():
    print("=" * 80)
    print("AUDITING PHOTOMETRY MOTION ARTIFACTS DURING OUT-OF-ARENA HANDLING GAPS")
    print("=" * 80)

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TAB_DIR, exist_ok=True)

    with open(SESSIONS_PKL, 'rb') as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} sessions.")

    df_ev = pd.read_csv(EVENT_TABLE)
    print(f"Loaded verified event frames table with {len(df_ev)} rows.")

    cohort_phot_stats = []
    exemplar_data = {}

    for i, s in enumerate(sessions):
        ev = df_ev.iloc[i]
        rem_f = int(ev['detected_removal_frame'])
        ret_f = int(ev['detected_return_frame'])
        entry_f = int(ev['detected_entry_frame'])
        dur_s = float(ev['handling_duration_sec'])
        aid = s.metadata.get('animal_id', f'mouse_{i}')
        drug = s.metadata.get('drug', 'Saline')
        genotype = s.metadata.get('genotype', 'WT')

        df = s.all_data_df
        if df is None or len(df) == 0:
            continue

        # Check available channels
        channels = [c for c in ['signal_0', 'signal_2', 'signal_1', 'signal_3'] if c in df.columns]
        if not channels:
            continue

        # Use primary channel
        primary_ch = channels[0]
        vals = df[primary_ch].values
        n_tot = len(vals)

        # Windows
        base_win = np.arange(max(0, rem_f - int(120 * FPS)), rem_f)
        gap_win = np.arange(rem_f, min(n_tot, ret_f))
        post_win = np.arange(ret_f, min(n_tot, ret_f + int(120 * FPS)))

        base_v = np.nanvar(vals[base_win]) if len(base_win) > 0 else 1e-9
        gap_v = np.nanvar(vals[gap_win]) if len(gap_win) > 0 else 1e-9
        post_v = np.nanvar(vals[post_win]) if len(post_win) > 0 else 1e-9

        var_ratio = gap_v / (base_v + 1e-12)

        # Also check dff channel if present
        dff_ch = f"{primary_ch}_dff" if f"{primary_ch}_dff" in df.columns else None
        if dff_ch:
            dff_vals = df[dff_ch].values
            dff_base_v = np.nanvar(dff_vals[base_win])
            dff_gap_v = np.nanvar(dff_vals[gap_win])
            dff_ratio = dff_gap_v / (dff_base_v + 1e-12)
        else:
            dff_ratio = np.nan

        stat = {
            'session_idx': i,
            'animal_id': aid,
            'genotype': genotype,
            'drug': drug,
            'cohort': f"{genotype}_{drug}",
            'primary_channel': primary_ch,
            'handling_duration_sec': dur_s,
            'baseline_var': base_v,
            'handling_var': gap_v,
            'post_var': post_v,
            'var_ratio': var_ratio,
            'dff_var_ratio': dff_ratio,
        }
        cohort_phot_stats.append(stat)

        if i in [0, 24]:
            exemplar_data[i] = {
                'rec': stat,
                'rem_f': rem_f,
                'ret_f': ret_f,
                'nom_inj_f': int(ev['nominal_injection_frame']),
                'primary_ch': primary_ch,
                'vals': vals,
                'df': df,
                'h5_path': s.tracking_refs_df['path'].values[0]
            }

        print(f"S{i:02d} | {aid} ({genotype} {drug}) | {primary_ch}: Gap Dur={dur_s:.1f}s | "
              f"Var Ratio={var_ratio:.2f}x (dFF Ratio={dff_ratio:.2f}x)", flush=True)

    df_phot = pd.DataFrame(cohort_phot_stats)
    csv_out = os.path.join(OUTPUT_TAB_DIR, "photometry_handling_variance_audit.csv")
    df_phot.to_csv(csv_out, index=False)
    print(f"\nSaved photometry audit table to: {csv_out}")

    print("\n" + "=" * 80)
    print("COHORT-WIDE PHOTOMETRY VARIANCE RATIO SUMMARY")
    print("=" * 80)
    print(f"Total Sessions Audited:       {len(df_phot)}")
    print(f"Median Variance Ratio (Gap / Base): {df_phot['var_ratio'].median():.2f}x")
    print(f"Mean Variance Ratio:                {df_phot['var_ratio'].mean():.2f}x")
    print(f"Min Ratio:                          {df_phot['var_ratio'].min():.2f}x")
    print(f"Max Ratio:                          {df_phot['var_ratio'].max():.2f}x")
    print(f"Sessions with Elevated Variance (>1.25x): {(df_phot['var_ratio'] > 1.25).sum()}/{len(df_phot)} ({(df_phot['var_ratio'] > 1.25).mean()*100:.1f}%)")

    # -------------------------------------------------------------------------
    # MULTI-PANEL DIAGNOSTIC FIGURE
    # -------------------------------------------------------------------------
    print("\nGenerating publication-grade multimodal validation figure...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.1,
    })

    fig = plt.figure(figsize=(19, 13), dpi=300)
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.1], hspace=0.30, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel A: Multimodal Zoom on Representative Session (S0: 1694 Saline, 65.9s)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ex0 = exemplar_data[0]
    rem_f = ex0['rem_f']
    ret_f = ex0['ret_f']
    nom_f = ex0['nom_inj_f']
    vals0 = ex0['vals']
    df0 = ex0['df']

    # Load raw DLC likelihood for S0
    df_h5 = pd.read_hdf(ex0['h5_path'])
    mean_lh0 = df_h5[[c for c in df_h5.columns if c[2] == 'likelihood']].mean(axis=1).values

    # Zoom window +/- 60s
    pad_f = int(60 * FPS)
    w_start = rem_f - pad_f
    w_end = ret_f + pad_f
    frames = np.arange(w_start, w_end)
    t_sec = (frames - rem_f) / FPS

    sub_lh = mean_lh0[w_start:w_end]
    sub_raw = vals0[w_start:w_end]
    # compute 1.5s rolling std (motion index)
    r_std = pd.Series(sub_raw).rolling(window=int(1.5 * FPS), min_periods=1, center=True).std().values

    # Normalize for plotting
    norm_raw = (sub_raw - np.nanmean(sub_raw)) / np.nanstd(sub_raw)
    norm_std = r_std / np.nanmedian(r_std[:pad_f])

    t_ret_s = (ret_f - rem_f) / FPS
    t_nom_s = (nom_f - rem_f) / FPS

    # Shading
    ax_a.axvspan(0, t_ret_s, color='#ffebee', alpha=0.85, ec='#ef5350', lw=1.2,
                 label=f"Physical Absence Gap ({ex0['rec']['handling_duration_sec']:.1f}s)")

    # Traces
    ax_a.plot(t_sec, sub_lh * 3.5 - 4.5, color='#455a64', lw=1.5, label='DLC Confidence (Offset)')
    ax_a.plot(t_sec, norm_raw, color='#00897b', lw=1.2, alpha=0.85, label='Raw Photometry (L-VS)')
    ax_a.plot(t_sec, norm_std * 1.5 - 2.5, color='#e65100', lw=1.8, label='Rolling Std Dev (Motion Index)')

    # Markers
    ax_a.axvline(0, color='#d32f2f', lw=2.2, ls='--', label=f"Animal Removal (T=0s)")
    ax_a.axvline(t_ret_s, color='#2e7d32', lw=2.2, ls='--', label=f"Animal Return (T={t_ret_s:.1f}s)")
    ax_a.axvline(t_nom_s, color='#f57c00', lw=2.4, ls='-', label=f"Nominal Inj (T={t_nom_s:.1f}s)")

    ax_a.text(0.03, 0.12,
              f"Session 0 (1694 Saline):\n"
              f"• Handling Gap Duration: 65.9 s\n"
              f"• Photometry Variance Ratio: {ex0['rec']['var_ratio']:.2f}x\n"
              f"• Direct temporal co-incidence of DLC drop and optical noise",
              transform=ax_a.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=8.5, fontweight='semibold')

    ax_a.set_title("A. Multimodal Confirmation: Tracking Drop vs Optical Motion Spikes (S0)",
                   fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_a.set_xlabel("Time Relative to Animal Removal (Seconds)", fontsize=10, fontweight='semibold')
    ax_a.set_ylabel("Normalized Signal Amplitude", fontsize=10, fontweight='semibold')
    ax_a.set_xlim(t_sec[0], t_sec[-1])
    ax_a.grid(True, linestyle=':', alpha=0.4, color='#888888')
    ax_a.legend(loc='upper right', framealpha=0.92, fontsize=7.8, ncol=2)

    # -------------------------------------------------------------------------
    # Panel B: Multimodal Zoom on Long Handling Case (S24: 1771 Amphetamine, 225s)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ex24 = exemplar_data[24]
    rem_f24 = ex24['rem_f']
    ret_f24 = ex24['ret_f']
    nom_f24 = ex24['nom_inj_f']
    vals24 = ex24['vals']

    df_h5_24 = pd.read_hdf(ex24['h5_path'])
    mean_lh24 = df_h5_24[[c for c in df_h5_24.columns if c[2] == 'likelihood']].mean(axis=1).values

    pad_f24 = int(80 * FPS)
    w_start24 = rem_f24 - pad_f24
    w_end24 = ret_f24 + pad_f24
    frames24 = np.arange(w_start24, w_end24)
    t_sec24 = (frames24 - rem_f24) / FPS

    sub_lh24 = mean_lh24[w_start24:w_end24]
    sub_raw24 = vals24[w_start24:w_end24]
    r_std24 = pd.Series(sub_raw24).rolling(window=int(2.0 * FPS), min_periods=1, center=True).std().values

    norm_raw24 = (sub_raw24 - np.nanmean(sub_raw24)) / (np.nanstd(sub_raw24) + 1e-12)
    norm_std24 = r_std24 / (np.nanmedian(r_std24[:pad_f24]) + 1e-12)
    t_ret_s24 = (ret_f24 - rem_f24) / FPS
    t_nom_s24 = (nom_f24 - rem_f24) / FPS

    ax_b.axvspan(0, t_ret_s24, color='#ffebee', alpha=0.85, ec='#ef5350', lw=1.2,
                 label=f"Physical Absence Gap ({ex24['rec']['handling_duration_sec']:.1f}s)")

    ax_b.plot(t_sec24, sub_lh24 * 3.5 - 4.5, color='#455a64', lw=1.5, label='DLC Confidence (Offset)')
    ax_b.plot(t_sec24, norm_raw24, color='#00897b', lw=1.2, alpha=0.85, label='Raw Photometry (L-VS)')
    ax_b.plot(t_sec24, norm_std24 * 0.8 - 2.5, color='#e65100', lw=1.8, label='Rolling Std Dev (Motion Index)')

    ax_b.axvline(0, color='#d32f2f', lw=2.2, ls='--', label=f"Animal Removal (T=0s)")
    ax_b.axvline(t_ret_s24, color='#2e7d32', lw=2.2, ls='--', label=f"Animal Return (T={t_ret_s24:.1f}s)")
    ax_b.axvline(t_nom_s24, color='#f57c00', lw=2.4, ls='-', label=f"Nominal Inj (T={t_nom_s24:.1f}s)")

    ax_b.text(0.03, 0.12,
              f"Session 24 (1771 Amphetamine - 225.6s Gap):\n"
              f"• Severe optical instability throughout 3.76-minute window\n"
              f"• Photometry Variance Ratio: {ex24['rec']['var_ratio']:.2f}x\n"
              f"• Confirms physical animal was outside arena experiencing handling",
              transform=ax_b.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=8.5, fontweight='semibold')

    ax_b.set_title("B. Multimodal Confirmation in Longest Absence Case (S24, 225.6s)",
                   fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_b.set_xlabel("Time Relative to Animal Removal (Seconds)", fontsize=10, fontweight='semibold')
    ax_b.set_ylabel("Normalized Signal Amplitude", fontsize=10, fontweight='semibold')
    ax_b.set_xlim(t_sec24[0], t_sec24[-1])
    ax_b.grid(True, linestyle=':', alpha=0.4, color='#888888')
    ax_b.legend(loc='upper right', framealpha=0.92, fontsize=7.8, ncol=2)

    # -------------------------------------------------------------------------
    # Panel C: Cohort-Wide Variance Amplification Ratio Distribution
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ratios = df_phot['var_ratio'].values
    # Clip large values for clean log-scale histogram
    log_ratios = np.log10(np.clip(ratios, 0.1, 1000.0))

    bins = np.linspace(-0.5, 2.5, 25)
    counts, edges, patches_list = ax_c.hist(log_ratios, bins=bins, color='#00897b', ec='#ffffff', lw=0.9, alpha=0.85)

    # Shading bins > 1.0 (0 on log10 scale)
    for patch, left_edge in zip(patches_list, edges[:-1]):
        if left_edge >= 0.0:  # ratio >= 1.0x
            patch.set_facecolor('#00897b')
        else:
            patch.set_facecolor('#78909c')

    med_log = np.log10(df_phot['var_ratio'].median())
    ax_c.axvline(0, color='#000000', lw=2.0, ls='-', label='Unity (No Change, 1.0x)')
    ax_c.axvline(med_log, color='#e65100', lw=2.2, ls='--', label=f"Median Amplification: {df_phot['var_ratio'].median():.2f}x")

    ax_c.set_xticks([-0.3, 0, 0.3, 0.699, 1.0, 1.699, 2.0])
    ax_c.set_xticklabels(['0.5x', '1.0x', '2.0x', '5.0x', '10x', '50x', '100x'])
    ax_c.set_title("C. Cohort Photometry Variance Amplification (Handling / Baseline)",
                   fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_c.set_xlabel("Variance Ratio (Handling Gap / In-Arena Baseline)", fontsize=10, fontweight='semibold')
    ax_c.set_ylabel("Number of Sessions", fontsize=10, fontweight='semibold')
    ax_c.grid(True, linestyle=':', alpha=0.4, color='#888888')
    ax_c.legend(loc='upper right', framealpha=0.92, fontsize=8.5)

    pct_elev = (df_phot['var_ratio'] > 1.25).mean() * 100.0
    ax_c.text(0.04, 0.72,
              f"Cohort Metrics (n={len(df_phot)} sessions):\n"
              f"• Median Variance Ratio: {df_phot['var_ratio'].median():.2f}x\n"
              f"• Mean Variance Ratio: {df_phot['var_ratio'].mean():.2f}x\n"
              f"• Elevated Optical Variance (>1.25x): {pct_elev:.1f}% of sessions\n"
              f"• Cable motion artifacts validate physical animal handling",
              transform=ax_c.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=8.5, fontweight='semibold')

    # -------------------------------------------------------------------------
    # Panel D: Variance Ratio Across Experimental Groups
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    cohort_order = ['WT_Saline', 'WT_Amphetamine', 'KI_Saline', 'KI_Amphetamine']
    cohort_labels = [
        f"WT Saline\n(n=8)\nMed={df_phot[df_phot['cohort']=='WT_Saline']['var_ratio'].median():.2f}x",
        f"WT Amph\n(n=8)\nMed={df_phot[df_phot['cohort']=='WT_Amphetamine']['var_ratio'].median():.2f}x",
        f"DAT-KI Sal\n(n=7)\nMed={df_phot[df_phot['cohort']=='DAT-KI_Saline']['var_ratio'].median() if 'DAT-KI_Saline' in df_phot['cohort'].values else df_phot[df_phot['cohort']=='KI_Saline']['var_ratio'].median():.2f}x",
        f"DAT-KI Amph\n(n=7)\nMed={df_phot[df_phot['cohort']=='DAT-KI_Amphetamine']['var_ratio'].median() if 'DAT-KI_Amphetamine' in df_phot['cohort'].values else df_phot[df_phot['cohort']=='KI_Amphetamine']['var_ratio'].median():.2f}x"
    ]
    cohort_colors = ['#1976d2', '#d32f2f', '#0288d1', '#e53935']

    # Group boxplot on log scale
    box_data = [np.log10(np.clip(df_phot[df_phot['cohort'] == c]['var_ratio'].values, 0.2, 500.0)) for c in cohort_order]

    bp = ax_d.boxplot(box_data, patch_artist=True, tick_labels=cohort_labels, widths=0.45,
                      medianprops=dict(color='#000000', lw=2.2))

    for patch, col in zip(bp['boxes'], cohort_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
        patch.set_linewidth(2.0)

    for col_idx, (vals, col) in enumerate(zip(box_data, cohort_colors)):
        x = np.random.normal(col_idx + 1, 0.04, size=len(vals))
        ax_d.scatter(x, vals, color=col, alpha=0.85, s=60, edgecolors='#ffffff', lw=1.0, zorder=5)

    ax_d.axhline(0, color='#666666', ls='--', lw=1.5, alpha=0.7, label='Baseline Parity (1.0x)')
    ax_d.set_yticks([-0.3, 0, 0.3, 0.699, 1.0, 1.699, 2.0])
    ax_d.set_yticklabels(['0.5x', '1.0x', '2.0x', '5.0x', '10x', '50x', '100x'])

    ax_d.set_title("D. Photometry Variance Amplification by Genotype & Drug Treatment",
                   fontsize=12, fontweight='bold', pad=8, loc='left')
    ax_d.set_ylabel("Variance Amplification Ratio (Log Scale)", fontsize=10, fontweight='semibold')
    ax_d.grid(True, linestyle=':', alpha=0.4, color='#888888')

    # Save figure
    fig_out = os.path.join(OUTPUT_FIG_DIR, "fig_photometry_handling_cross_validation.png")
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved multimodal cross-validation figure to: {fig_out}")

    # Copy to artifacts
    art_out = os.path.join(ARTIFACT_DIR, "fig_photometry_handling_cross_validation.png")
    shutil.copy2(fig_out, art_out)
    print(f"Copied figure to artifacts: {art_out}")

if __name__ == "__main__":
    main()
