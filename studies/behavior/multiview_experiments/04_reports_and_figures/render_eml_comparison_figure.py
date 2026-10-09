#!/usr/bin/env python3
"""
render_eml_comparison_figure.py
=============================================================================
Generates a publication-quality comparative figure for the Tandem KPMS
multi-seed Expected Marginal Likelihood (EML) model selection evaluation.
=============================================================================
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

CSV_PATH = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/tandem_cohort_eml_model_rankings.csv")
OUT_PATH = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/tandem_cohort_eml_detailed_comparison.png")

df = pd.read_csv(CSV_PATH)

baseline_score = df.loc[df["seed"] == 42, "eml_score"].values[0]
df["delta_eml"] = df["eml_score"] - baseline_score
df["delta_eml_k"] = df["delta_eml"] / 1e5  # in hundreds of thousands

# Sort by rank ascending
df = df.sort_values("rank", ascending=True).reset_index(drop=True)

fig = plt.figure(figsize=(14, 6.8), dpi=220)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)

# --- Panel A: Delta EML relative to Baseline ---
ax1 = fig.add_subplot(gs[0])
colors = [
    "#2e7d32" if r == 1 else "#1976d2" if s != 42 else "#757575"
    for r, s in zip(df["rank"], df["seed"])
]

y_pos = np.arange(len(df))
bars = ax1.barh(
    y_pos, df["delta_eml_k"],
    xerr=df["standard_error"] / 1e5,
    color=colors, alpha=0.88, capsize=6, edgecolor="black", linewidth=0.8
)
ax1.set_yticks(y_pos)
labels = [
    f"Rank #{r}: Seed {s}\n({'WINNER' if r == 1 else 'Baseline' if s == 42 else 'Runner-up'})"
    for r, s in zip(df["rank"], df["seed"])
]
ax1.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Relative Gain: Δ Log-Likelihood vs Baseline (× 10⁵ nats)", fontsize=11, fontweight="bold")
ax1.set_title("A. Model Selection by Relative EML Gain\n(Higher is Better)", fontsize=12.5, fontweight="bold", pad=12)

# Value annotations on bars
for bar, d_val, se, r in zip(bars, df["delta_eml"], df["standard_error"], df["rank"]):
    if r == 5:
        ax1.text(0.3, bar.get_y() + bar.get_height()/2, "0 (Baseline Reference)",
                 va="center", ha="left", fontsize=9.5, fontweight="bold", color="#424242")
    else:
        x_pos = bar.get_width() + 0.35
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f"+{d_val:,.0f}\n(±{se:,.0f})",
                 va="center", ha="left", fontsize=9, fontweight="bold",
                 color="#1b5e20" if r == 1 else "#0d47a1")

ax1.set_xlim(-0.5, 14.8)
ax1.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.grid(axis="x", linestyle=":", alpha=0.6)

# --- Panel B: Absolute EML (Zoomed Focused Axis) ---
ax2 = fig.add_subplot(gs[1])
bars2 = ax2.barh(
    y_pos, df["eml_score"] / 1e6,
    xerr=df["standard_error"] / 1e6,
    color=colors, alpha=0.88, capsize=6, edgecolor="black", linewidth=0.8
)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"Seed {s}" for s in df["seed"]], fontsize=10.5, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("Expected Marginal Log-Likelihood (Millions of nats)", fontsize=11, fontweight="bold")
ax2.set_title("B. Absolute Expected Marginal Likelihood\n(Focused Axis: 269.0M – 271.4M nats)", fontsize=12.5, fontweight="bold", pad=12)
ax2.set_xlim(269.0, 271.45)
ax2.grid(axis="x", linestyle=":", alpha=0.6)

# Annotate exact score inside or at end of bars
for bar, score, err in zip(bars2, df["eml_score"], df["standard_error"]):
    val_m = score / 1e6
    err_m = err / 1e6
    x_pos = val_m + err_m + 0.05
    ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
             f"{val_m:.2f}M", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#212121")

plt.suptitle("Tandem KPMS Cohort (118 Sessions, 18M Frames): Multi-Seed EML Model Selection",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

plt.savefig(OUT_PATH, bbox_inches="tight")
plt.close()
print(f"✅ Saved publication plot to: {OUT_PATH}")
