#!/usr/bin/env python3
"""
plot_kappa_scan_results.py
--------------------------
Generates a visualization of the unconstrained multi-view kappa search convergence.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search")
JSON_PATH = EXPERIMENT_DIR / "kappa_scan_history.json"
OUT_PLOT = EXPERIMENT_DIR / "kappa_scan_curve.png"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

history = data["history"]
kappas = [h["kappa"] for h in history]
durations_frames = [h["median_frames"] for h in history]
durations_ms = [h["median_ms"] for h in history]
steps = [h["step"] for h in history]

target_frames = data["target_frames"]
target_ms = (target_frames / 30.0) * 1000.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# 1. Kappa vs Duration
ax1.plot(kappas, durations_frames, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Multi-View 23-KP")
ax1.axhline(target_frames, color="#d62728", linestyle="--", linewidth=1.5, label=f"Target ({target_frames:.1f} f / {target_ms:.0f} ms)")
ax1.set_xscale("log")
ax1.set_xlabel("Stickiness Hyperparameter $\\kappa$ (log scale)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Median Syllable Duration (frames @ 30 FPS)", fontsize=11, fontweight="bold")
ax1.set_title("Multi-View $\\kappa$ Scan Convergence", fontsize=12, fontweight="bold")
ax1.grid(True, which="both", linestyle=":", alpha=0.6)
ax1.legend(loc="lower right", frameon=True)

# Annotate points
for s, k, d in zip(steps, kappas, durations_frames):
    ax1.annotate(
        f"Step {s}\n$\\kappa$={k:.2e}\n{d:.1f} f",
        xy=(k, d),
        xytext=(0, 15 if s == 1 else -35),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#1f77b4")
    )

# 2. Step vs Duration & Elapsed Time
ax2.plot(steps, durations_ms, "s-", color="#2ca02c", linewidth=2.5, markersize=8, label="Dwell Time (ms)")
ax2.axhline(target_ms, color="#d62728", linestyle="--", linewidth=1.5, label=f"Target Dwell ({target_ms:.0f} ms)")
ax2.set_xlabel("Search Iteration Step", fontsize=11, fontweight="bold")
ax2.set_ylabel("Median Duration (ms)", fontsize=11, fontweight="bold")
ax2.set_title("Dwell Time Alignment to Ethological Target", fontsize=12, fontweight="bold")
ax2.set_xticks(steps)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig(OUT_PLOT, bbox_inches="tight")
plt.close()

print(f"Generated plot: {OUT_PLOT}")
