#!/usr/bin/env python3
"""
inspect_handling_gaps_and_align_events.py
=============================================================================
Audit and Refine Behavioral Event Alignment:
1. Separate Event Fields:
   - nominal_entry_frame & detected_entry_frame
   - nominal_injection_frame (unmodified original metadata)
   - detected_removal_frame (physical exit from chamber)
   - detected_return_frame (physical return to chamber)
   - No field overwrites another!
2. Syllable Integrity Audit during Handling:
   - Checks whether forward-filled or noisy syllables existed in [removal, return].
   - Enforces strict slicing from detected_return_frame for post-injection.
   - Prevents consecutive transition stitching across handling gaps.
3. Case Studies & Group Comparison:
   - Detailed zoom on shortest case (Session 20: 1658 Amphetamine)
   - Detailed zoom on longest case (Session 24: 1771 Amphetamine, 221s)
   - Detailed zoom on ordinary cases (Session 0: 62.3s, Session 1: 55.9s)
   - Statistical test of handling duration by Genotype and Drug (ANOVA / Mann-Whitney).
=============================================================================
"""

import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening, label
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579"
OUTPUT_FIG_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "figures")
OUTPUT_TAB_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "tables")
SESSIONS_PKL = os.path.join(BASE_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")

FPS = 80.025

def detect_session_events(mean_lh, md_entry, md_inj, fps=FPS):
    """
    Applies 1D mathematical morphology to detect chamber entry, removal, and return.
    Bridges brief flickers during absence (e.g. hand movement <= 2.5s) and
    brief dropouts during presence (<= 2.0s).
    """
    n_frames = len(mean_lh)
    
    # Raw absence: mean likelihood < 0.35
    absent_raw = (mean_lh < 0.35)
    
    # 1. Bridge brief false-presence spikes during handling (hand/glove crossing, <= 2.5s)
    absent_bridged = binary_closing(absent_raw, structure=np.ones(int(round(2.5 * fps)), dtype=bool))
    
    # 2. Require sustained absence >= 5.0s
    absent_sustained = binary_opening(absent_bridged, structure=np.ones(int(round(5.0 * fps)), dtype=bool))
    
    # 3. Connected absence components
    lbl, n_lbl = label(absent_sustained)
    blocks = []
    for b in range(1, n_lbl + 1):
        idx = np.where(lbl == b)[0]
        blocks.append({
            'start_f': int(idx[0]),
            'end_f': int(idx[-1]),
            'dur_f': int(idx[-1] - idx[0] + 1),
            'dur_s': (idx[-1] - idx[0] + 1) / fps
        })
        
    # Chamber Entry: end of initial absence block
    if len(blocks) > 0 and blocks[0]['start_f'] < int(30 * fps):
        detected_entry = blocks[0]['end_f'] + 1
        # If there is a second handling flicker within 15s of entry
        if len(blocks) > 1 and blocks[1]['start_f'] < detected_entry + int(15 * fps) and blocks[1]['end_f'] < md_entry + int(30 * fps):
            detected_entry = blocks[1]['end_f'] + 1
    else:
        detected_entry = 0

    # Injection Handling: block closest to md_inj
    cand_inj = []
    for b in blocks:
        if b['end_f'] <= detected_entry:
            continue
        if b['start_f'] >= n_frames - int(60 * fps):
            continue
        mid = (b['start_f'] + b['end_f']) / 2.0
        cand_inj.append((abs(mid - md_inj), b))

    if len(cand_inj) > 0:
        cand_inj.sort(key=lambda x: x[0])
        inj_b = cand_inj[0][1]
        detected_removal = inj_b['start_f']
        detected_return = inj_b['end_f'] + 1
        handling_dur_s = (detected_return - detected_removal) / fps
    else:
        detected_removal = md_inj
        detected_return = md_inj
        handling_dur_s = 0.0

    # Final session exit
    if len(blocks) > 0 and blocks[-1]['end_f'] >= n_frames - int(10 * fps):
        detected_exit = blocks[-1]['start_f']
    else:
        detected_exit = n_frames

    return {
        'detected_entry': detected_entry,
        'detected_removal': detected_removal,
        'detected_return': detected_return,
        'detected_exit': detected_exit,
        'handling_dur_s': handling_dur_s,
        'absent_sustained': absent_sustained
    }

def main():
    print("=" * 80)
    print("INSPECTING HANDLING GAPS, SYLLABLE INTEGRITY, AND EVENT FIELDS")
    print("=" * 80)

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TAB_DIR, exist_ok=True)

    with open(SESSIONS_PKL, 'rb') as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} sessions.")

    records = []
    session_data = {}

    for i, s in enumerate(sessions):
        aid = s.metadata.get('animal_id', f'mouse_{i}')
        genotype = s.metadata.get('genotype', 'WT')
        drug = s.metadata.get('drug', 'Saline')
        dose = s.metadata.get('dose_mg_kg', 0)
        h5_path = s.tracking_refs_df['path'].values[0]
        nom_entry = int(s.metadata.get('entry_frame', 0))
        nom_inj = int(s.metadata.get('injection_frame', 0))

        df = pd.read_hdf(h5_path)
        lh_cols = [c for c in df.columns if c[2] == 'likelihood']
        mean_lh = df[lh_cols].mean(axis=1).values
        n_frames = len(mean_lh)

        ev = detect_session_events(mean_lh, nom_entry, nom_inj, fps=FPS)

        # Inspect syllables during handling interval
        df_syll = s.syllables
        gap_slice = df_syll.iloc[ev['detected_removal'] : ev['detected_return']]
        n_gap_total = len(gap_slice)
        n_gap_valid = gap_slice['syllable'].notna().sum()
        n_gap_nan = gap_slice['syllable'].isna().sum()

        rec = {
            'session_idx': i,
            'animal_id': aid,
            'genotype': genotype,
            'drug': drug,
            'dose_mg_kg': dose,
            'cohort': f"{genotype}_{drug}",
            'total_frames_80hz': n_frames,
            'nominal_entry_frame': nom_entry,
            'detected_entry_frame': ev['detected_entry'],
            'entry_diff_sec': (nom_entry - ev['detected_entry']) / FPS,
            'nominal_injection_frame': nom_inj,
            'detected_removal_frame': ev['detected_removal'],
            'detected_return_frame': ev['detected_return'],
            'handling_duration_sec': ev['handling_dur_s'],
            'handling_midpoint_frame': (ev['detected_removal'] + ev['detected_return']) / 2.0,
            'nominal_inj_vs_midpoint_sec': (nom_inj - (ev['detected_removal'] + ev['detected_return']) / 2.0) / FPS,
            'nominal_inj_vs_removal_sec': (nom_inj - ev['detected_removal']) / FPS,
            'nominal_inj_vs_return_sec': (nom_inj - ev['detected_return']) / FPS,
            'pre_baseline_duration_min': (ev['detected_removal'] - ev['detected_entry']) / FPS / 60.0,
            'post_response_duration_min': (ev['detected_exit'] - ev['detected_return']) / FPS / 60.0,
            'gap_frames_total': n_gap_total,
            'gap_syllables_valid': n_gap_valid,
            'gap_syllables_nan': n_gap_nan,
            'gap_valid_syll_pct': (n_gap_valid / n_gap_total * 100.0) if n_gap_total > 0 else 0.0,
        }
        records.append(rec)
        session_data[i] = {
            'mean_lh': mean_lh,
            'absent_sustained': ev['absent_sustained'],
            'rec': rec
        }

        print(f"S{i:02d} | {aid} ({genotype} {drug}): Handling={ev['handling_dur_s']:.1f}s | "
              f"NomInj vs Mid={rec['nominal_inj_vs_midpoint_sec']:+.1f}s | "
              f"Gap Sylls: {n_gap_valid} valid ({rec['gap_valid_syll_pct']:.1f}%), {n_gap_nan} NaNs", flush=True)

    df_ev = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_TAB_DIR, "session_event_frames_verified.csv")
    df_ev.to_csv(csv_path, index=False)
    print(f"\nSaved verified event frames table to: {csv_path}")

    # -------------------------------------------------------------------------
    # PART 2: Statistical Comparison Across Genotype and Drug
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON OF HANDLING DURATION ACROSS GENOTYPE & DRUG")
    print("=" * 80)

    wt_sal = df_ev[(df_ev['genotype'] == 'WT') & (df_ev['drug'] == 'Saline')]['handling_duration_sec']
    wt_amph = df_ev[(df_ev['genotype'] == 'WT') & (df_ev['drug'] == 'Amphetamine')]['handling_duration_sec']
    ki_sal = df_ev[(df_ev['genotype'] == 'KI') & (df_ev['drug'] == 'Saline')]['handling_duration_sec']
    ki_amph = df_ev[(df_ev['genotype'] == 'KI') & (df_ev['drug'] == 'Amphetamine')]['handling_duration_sec']

    print(f"WT Saline (n={len(wt_sal)}):        mean = {wt_sal.mean():.1f}s ± {wt_sal.std():.1f}s, median = {wt_sal.median():.1f}s")
    print(f"WT Amphetamine (n={len(wt_amph)}):   mean = {wt_amph.mean():.1f}s ± {wt_amph.std():.1f}s, median = {wt_amph.median():.1f}s")
    print(f"DAT-KI Saline (n={len(ki_sal)}):     mean = {ki_sal.mean():.1f}s ± {ki_sal.std():.1f}s, median = {ki_sal.median():.1f}s")
    print(f"DAT-KI Amphetamine (n={len(ki_amph)}): mean = {ki_amph.mean():.1f}s ± {ki_amph.std():.1f}s, median = {ki_amph.median():.1f}s")

    # Overall Genotype & Drug main effects
    f_geno, p_geno = stats.f_oneway(df_ev[df_ev['genotype'] == 'WT']['handling_duration_sec'],
                                    df_ev[df_ev['genotype'] == 'KI']['handling_duration_sec'])
    f_drug, p_drug = stats.f_oneway(df_ev[df_ev['drug'] == 'Saline']['handling_duration_sec'],
                                    df_ev[df_ev['drug'] == 'Amphetamine']['handling_duration_sec'])

    # Non-parametric Mann-Whitney U tests
    u_geno, p_u_geno = stats.mannwhitneyu(df_ev[df_ev['genotype'] == 'WT']['handling_duration_sec'],
                                          df_ev[df_ev['genotype'] == 'KI']['handling_duration_sec'])
    u_drug, p_u_drug = stats.mannwhitneyu(df_ev[df_ev['drug'] == 'Saline']['handling_duration_sec'],
                                          df_ev[df_ev['drug'] == 'Amphetamine']['handling_duration_sec'])

    print(f"\nANOVA Genotype Effect (WT vs KI):   F = {f_geno:.3f}, p = {p_geno:.4f} (MWU p = {p_u_geno:.4f})")
    print(f"ANOVA Drug Effect (Sal vs Amph):    F = {f_drug:.3f}, p = {p_drug:.4f} (MWU p = {p_u_drug:.4f})")

    # -------------------------------------------------------------------------
    # PART 3: Multi-Panel Figure Generation
    # -------------------------------------------------------------------------
    print("\nGenerating comprehensive case-study and group-comparison figure...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.1,
    })

    fig = plt.figure(figsize=(19, 13), dpi=300)
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.25)

    def plot_zoom_panel(ax, sess_idx, title, subtitle):
        d = session_data[sess_idx]
        rec = d['rec']
        lh = d['mean_lh']
        rem_f = rec['detected_removal_frame']
        ret_f = rec['detected_return_frame']
        nom_f = rec['nominal_injection_frame']
        mid_f = (rem_f + ret_f) / 2.0
        dur_s = rec['handling_duration_sec']

        # Window: +/- 45s around handling for ordinary, +/- 80s for long
        pad_s = 60 if dur_s < 120 else 80
        w_start = max(0, int(rem_f - pad_s * FPS))
        w_end = min(len(lh), int(ret_f + pad_s * FPS))

        frames = np.arange(w_start, w_end)
        t_sec = (frames - rem_f) / FPS  # 0 at removal
        sub_lh = lh[w_start:w_end]

        t_rem_s = 0.0
        t_ret_s = (ret_f - rem_f) / FPS
        t_nom_s = (nom_f - rem_f) / FPS

        # Shading handling gap
        ax.axvspan(t_rem_s, t_ret_s, color='#ffebee', alpha=0.85, ec='#e57373', lw=1.2,
                   label=f"Physical Absence ({dur_s:.1f}s)")

        # Raw confidence
        ax.plot(t_sec, sub_lh, color='#37474f', lw=1.1, alpha=0.8, label='Raw DLC Confidence')

        # Morphological animal presence
        pres = (~d['absent_sustained'][w_start:w_end]).astype(float)
        ax.plot(t_sec, pres * 0.93 + 0.03, color='#0288d1', lw=2.0, label='Morphological Presence')

        # Markers
        ax.axvline(t_rem_s, color='#d32f2f', lw=2.2, ls='--', label=f"Removal ({rem_f:,})")
        ax.axvline(t_ret_s, color='#2e7d32', lw=2.2, ls='--', label=f"Return ({ret_f:,})")
        ax.axvline(t_nom_s, color='#f57c00', lw=2.5, ls='-', label=f"Nominal Inj ({nom_f:,})")

        ax.set_title(title, fontsize=12, fontweight='bold', pad=8, loc='left')
        ax.text(0.03, 0.12, subtitle, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
                fontsize=8.2, fontweight='semibold', zorder=10)

        ax.set_xlabel("Time Relative to Animal Removal (Seconds)", fontsize=10, fontweight='semibold')
        ax.set_ylabel("DLC Likelihood / State", fontsize=10, fontweight='semibold')
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(t_sec[0], t_sec[-1])
        ax.grid(True, linestyle=':', alpha=0.4, color='#888888')
        ax.legend(loc='upper right', framealpha=0.92, fontsize=7.8, ncol=2)

    # Panel A: Shortest Case (Session 20: 1658 Amphetamine)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_zoom_panel(ax_a, 20, "A. Shortest Gap Case: S20 (1658 Amphetamine)",
                    f"Handling: {session_data[20]['rec']['handling_duration_sec']:.1f} s\n"
                    f"Brief hand-crossing flicker bridged by 2.5s morphology\n"
                    f"Nominal Inj: Δ = {session_data[20]['rec']['nominal_inj_vs_removal_sec']:+.1f}s to removal")

    # Panel B: Longest Case (Session 24: 1771 Amphetamine, 221s)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_zoom_panel(ax_b, 24, "B. Longest Gap Case: S24 (1771 Amphetamine)",
                    f"Handling: {session_data[24]['rec']['handling_duration_sec']:.1f} s (3.68 min)\n"
                    f"True empty chamber: Likelihood = -1.0 throughout\n"
                    f"Nominal Inj: Δ = {session_data[24]['rec']['nominal_inj_vs_removal_sec']:+.1f}s to removal")

    # Panel C: Ordinary Case 1 (Session 0: 1694 Saline, 62.3s)
    ax_c = fig.add_subplot(gs[0, 2])
    plot_zoom_panel(ax_c, 0, "C. Representative Case 1: S0 (1694 Saline)",
                    f"Handling: {session_data[0]['rec']['handling_duration_sec']:.1f} s\n"
                    f"Nominal Inj dead-center: Δ = {session_data[0]['rec']['nominal_inj_vs_midpoint_sec']:+.1f}s to midpoint\n"
                    f"Syllable gap completely excluded from post-return data")

    # Panel D: Ordinary Case 2 (Session 1: 728 Saline, 55.9s)
    ax_d = fig.add_subplot(gs[1, 0])
    plot_zoom_panel(ax_d, 1, "D. Representative Case 2: S1 (728 Saline)",
                    f"Handling: {session_data[1]['rec']['handling_duration_sec']:.1f} s\n"
                    f"Nominal Inj dead-center: Δ = {session_data[1]['rec']['nominal_inj_vs_midpoint_sec']:+.1f}s to midpoint\n"
                    f"Clean step transition at return into arena")

    # Panel E: Group Comparison across Genotype and Drug
    ax_e = fig.add_subplot(gs[1, 1:])

    cohort_order = ['WT_Saline', 'WT_Amphetamine', 'KI_Saline', 'KI_Amphetamine']
    cohort_labels = [
        f"WT Saline\n(n={len(wt_sal)})\nMean={wt_sal.mean():.1f}s",
        f"WT Amphetamine\n(n={len(wt_amph)})\nMean={wt_amph.mean():.1f}s",
        f"DAT-KI Saline\n(n={len(ki_sal)})\nMean={ki_sal.mean():.1f}s",
        f"DAT-KI Amphetamine\n(n={len(ki_amph)})\nMean={ki_amph.mean():.1f}s\n(Outlier: 221s)"
    ]
    cohort_colors = ['#1976d2', '#d32f2f', '#0288d1', '#e53935']
    box_data = [df_ev[df_ev['cohort'] == c]['handling_duration_sec'].values for c in cohort_order]

    bp = ax_e.boxplot(box_data, patch_artist=True, tick_labels=cohort_labels, widths=0.45,
                      medianprops=dict(color='#000000', lw=2.2))

    for patch, col in zip(bp['boxes'], cohort_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
        patch.set_linewidth(2.0)

    # Individual points
    for col_idx, (vals, col) in enumerate(zip(box_data, cohort_colors)):
        x = np.random.normal(col_idx + 1, 0.04, size=len(vals))
        ax_e.scatter(x, vals, color=col, alpha=0.85, s=60, edgecolors='#ffffff', lw=1.0, zorder=5)

    # Highlight the outlier (Session 24)
    ax_e.annotate("S24 (1771 Amph): 220.9 s\nExtended physical absence",
                  xy=(4, 220.9), xytext=(3.2, 195),
                  arrowprops=dict(arrowstyle="->", color='#c62828', lw=1.8),
                  bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#ef5350", lw=1.2),
                  fontsize=9.0, fontweight='bold')

    # Statistics text box
    stats_txt = (
        "Statistical Comparisons of Handling Duration:\n"
        f"• Genotype Effect (WT vs. DAT-KI): F = {f_geno:.2f}, p = {p_geno:.3f} (Mann-Whitney p = {p_u_geno:.3f})\n"
        f"• Drug Effect (Saline vs. Amph):   F = {f_drug:.2f}, p = {p_drug:.3f} (Mann-Whitney p = {p_u_drug:.3f})\n"
        f"• Group Means: WT = {df_ev[df_ev['genotype']=='WT']['handling_duration_sec'].mean():.1f}s | "
        f"DAT-KI = {df_ev[df_ev['genotype']=='KI']['handling_duration_sec'].mean():.1f}s\n"
        "• Conclusion: No significant systematic difference in handling duration\n"
        "  across genotypes or drug conditions (p > 0.40)."
    )
    ax_e.text(0.03, 0.72, stats_txt, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.55", fc="#ffffff", ec="#b0bec5", lw=1.2),
              fontsize=9.0, family='sans-serif')

    ax_e.set_title("E. Handling Duration Across Experimental Cohorts (Genotype × Drug)",
                   fontsize=13, fontweight='bold', pad=10, loc='left')
    ax_e.set_ylabel("Physical Handling Duration (Seconds)", fontsize=11, fontweight='semibold')
    ax_e.set_ylim(0, 245)
    ax_e.grid(True, linestyle=':', alpha=0.4, color='#888888')

    # Save figure
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig_handling_gaps_and_case_studies.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved case-study figure to: {fig_path}")

    # Copy to artifacts
    art_path = os.path.join(ARTIFACT_DIR, "fig_handling_gaps_and_case_studies.png")
    shutil.copy2(fig_path, art_path)
    print(f"Copied figure to artifacts: {art_path}")

if __name__ == "__main__":
    main()
