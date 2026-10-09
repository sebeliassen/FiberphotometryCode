#!/usr/bin/env python3
"""
Render Figures for Conditional Future Dispersion Benchmark
===========================================================
Produces:
1. Fig A: Conditional Future Dispersion vs. Past Behavioral Distance (Empirical Extrapolation Curves)
2. Fig B: Cross-Session k-NN Predictive Gain vs. Parametric Model Ceilings
3. Fig C: Empirical Fanning Futures Case Studies (Matched Pasts across Different Mice -> Divergent Future Distributions)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
TABLE_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables")
os.makedirs(FIG_DIR, exist_ok=True)

csv_path = os.path.join(TABLE_DIR, "conditional_future_dispersion_results.csv")
case_study_path = os.path.join(TABLE_DIR, "fanning_futures_exemplars.pkl")

if not os.path.exists(csv_path):
    print(f"Results CSV not found yet: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded results from {csv_path}")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# -------------------------------------------------------------------------
# FIGURE 1: Dual-Panel Dispersion & Predictive Gain vs. Past Distance
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

colors = {
    "1. Spatial Context": "#7f7f7f",
    "2. Continuous Pose": "#17becf",
    "3. Current Syllable": "#9467bd",
    "4. Repertoire (Hellinger)": "#1f77b4",
    "5. Repertoire + Space (Hybrid)": "#2ca02c",
    "6. NMF K=4 (Cross-Validated)": "#ff7f0e"
}

markers = {
    "1. Spatial Context": "o",
    "2. Continuous Pose": "^",
    "3. Current Syllable": "s",
    "4. Repertoire (Hellinger)": "D",
    "5. Repertoire + Space (Hybrid)": "p",
    "6. NMF K=4 (Cross-Validated)": "X"
}

# Panel A: Within-Neighborhood Future Dispersion vs. Past Distance
for rep_name in df["Representation"].unique():
    sub = df[df["Representation"] == rep_name].sort_values("Mean_Past_Distance")
    c = colors.get(rep_name, "#333333")
    m = markers.get(rep_name, "o")
    
    ax1.plot(sub["Mean_Past_Distance"], sub["Future_Dispersion_bits"], 
             marker=m, linewidth=2.2, markersize=7, label=rep_name, color=c)
    ax1.fill_between(sub["Mean_Past_Distance"], 
                     sub["Future_Dispersion_bits"] - sub["Dispersion_CI_95"],
                     sub["Future_Dispersion_bits"] + sub["Dispersion_CI_95"],
                     color=c, alpha=0.15)

ax1.set_title("A. Conditional Future Dispersion vs. Past Behavioral Distance\n(Within-Neighborhood Fanning Among Matched Pasts)", fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel("Mean Past-State Behavioral Distance $\\bar{d}(X_i, X_j)$", fontsize=11)
ax1.set_ylabel("Conditional Future Dispersion (bits, $D_{JS}$)", fontsize=11)
ax1.legend(loc='lower right', frameon=True, fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Held-Out Predictive Gain vs. Past Distance
for rep_name in df["Representation"].unique():
    sub = df[df["Representation"] == rep_name].sort_values("Mean_Past_Distance")
    c = colors.get(rep_name, "#333333")
    m = markers.get(rep_name, "o")
    
    ax2.plot(sub["Mean_Past_Distance"], sub["Predictive_Gain_pct"], 
             marker=m, linewidth=2.2, markersize=7, label=rep_name, color=c)
    ax2.fill_between(sub["Mean_Past_Distance"], 
                     sub["Predictive_Gain_pct"] - sub["Gain_CI_95"],
                     sub["Predictive_Gain_pct"] + sub["Gain_CI_95"],
                     color=c, alpha=0.15)

# Horizontal Reference Ceilings
ax2.axhline(31.88, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Parametric BoW Ceiling (+31.9%)')
ax2.axhline(35.82, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8, label='Action-GPT Transformer (+35.8%)')

ax2.set_title("B. Cross-Session $k$-NN Predictive Gain vs. Model Baselines\n(Held-Out Anchor Future Evaluation: $D_{JS}(Y_i, \\bar{Y}_{\\mathcal{N}_i})$)", fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel("Mean Past-State Behavioral Distance $\\bar{d}(X_i, X_j)$", fontsize=11)
ax2.set_ylabel("Predictive Information Gain (% $\\Delta D_{JS}$ vs. Null)", fontsize=11)
ax2.legend(loc='lower left', frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig5_conditional_future_dispersion_extrapolation.png")
fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved Figure 1 to: {fig1_path}")

# -------------------------------------------------------------------------
# FIGURE 2: Empirical "Fanning Futures" Case Studies
# -------------------------------------------------------------------------
if os.path.exists(case_study_path):
    with open(case_study_path, "rb") as f:
        cases = pickle.load(f)
        
    case_keys = list(cases.keys())[:2] # Top 2 case studies
    if len(case_keys) > 0:
        fig = plt.figure(figsize=(14, 7), dpi=300)
        gs = gridspec.GridSpec(len(case_keys), 2, width_ratios=[1, 1.4], wspace=0.25, hspace=0.35)
        
        for row_idx, ck in enumerate(case_keys):
            c_data = cases[ck]
            
            # Left: Matched Past Repertoires (Anchor + 5 Neighbors from Different Sessions)
            ax_past = fig.add_subplot(gs[row_idx, 0])
            anchor_past = c_data["anchor_past_rep"]
            top_sylls = np.argsort(anchor_past)[-10:][::-1]
            
            x_bar = np.arange(len(top_sylls))
            width = 0.14
            
            ax_past.bar(x_bar - width*2.5, anchor_past[top_sylls], width=width*1.8, color='#1f77b4', label=f'Anchor (Sess #{c_data["anchor_sess"]})')
            
            for n_i in range(min(4, len(c_data["neighbor_past_reps"]))):
                n_rep = c_data["neighbor_past_reps"][n_i]
                n_sess = c_data["neighbor_sessions"][n_i]
                ax_past.bar(x_bar - width*0.5 + n_i*width, n_rep[top_sylls], width=width*0.9, alpha=0.7, label=f'Match {n_i+1} (Sess #{n_sess})')
                
            ax_past.set_title(f"State #{row_idx+1}: Highly Matched Past Repertoires ($W = 68.3\\text{{s}}$)\n[Top Syllables Across Separate Animals]", fontsize=10.5, fontweight='bold')
            ax_past.set_xticks(x_bar)
            ax_past.set_xticklabels([f"Syll #{s}" for s in top_sylls], rotation=45, ha='right', fontsize=8.5)
            ax_past.set_ylabel("Syllable Frequency in $W$", fontsize=9.5)
            ax_past.legend(loc='upper right', frameon=True, fontsize=7.5)
            ax_past.grid(axis='y', linestyle='--', alpha=0.4)
            
            # Right: Corresponding Future Horizons (Divergent / Fanning Distributions)
            ax_fut = fig.add_subplot(gs[row_idx, 1])
            anchor_fut = c_data["anchor_future"]
            fut_top = np.argsort(anchor_fut)[-10:][::-1]
            
            x_fut = np.arange(len(fut_top))
            ax_fut.plot(x_fut, anchor_fut[fut_top], color='#1f77b4', marker='o', linewidth=2.5, label='Anchor Future ($H=120\\text{s}$)')
            
            for n_i in range(min(4, len(c_data["neighbor_futures"]))):
                n_fut = c_data["neighbor_futures"][n_i]
                n_sess = c_data["neighbor_sessions"][n_i]
                ax_fut.plot(x_fut, n_fut[fut_top], marker='^', linewidth=1.5, alpha=0.75, linestyle='--', label=f'Matched Animal #{n_sess} Future')
                
            ax_fut.set_title(f"Fanning Future Outcomes ($H = 120\\text{{s}}$)\n[Observed Variation Conditional on Matched Past]", fontsize=10.5, fontweight='bold')
            ax_fut.set_xticks(x_fut)
            ax_fut.set_xticklabels([f"Syll #{s}" for s in fut_top], rotation=45, ha='right', fontsize=8.5)
            ax_fut.set_ylabel("Future Syllable Probability in $H$", fontsize=9.5)
            ax_fut.legend(loc='upper right', frameon=True, fontsize=7.5)
            ax_fut.grid(True, linestyle='--', alpha=0.4)
            
        fig2_path = os.path.join(FIG_DIR, "fig6_empirical_fanning_futures_case_study.png")
        fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Figure 2 to: {fig2_path}")

print("FIGURE RENDERING SCRIPT COMPLETE!")
