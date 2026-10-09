"""
Clustering Diagnostics: Before vs After log1p Transform

Loads session data, extracts bouts, and runs K-Means clustering under two regimes:
  A) StandardScaler only (current baseline)
  B) log1p → StandardScaler (proposed fix)

Outputs:
  - Statistical diagnostics printed to stdout (skewness, kurtosis, cluster balance)
  - Visual diagnostic panels saved as PNGs
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader
from analysis.language_models.lib.fusion.clustering import extract_bouts_from_session

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_KEYS = [
    "duration_frames",
    "mean_velocity",
    "mean_angular_velocity",
    "mean_acceleration",
    "mean_dist_from_center",
]
FEATURE_LABELS = [
    "Duration (frames)",
    "Mean Velocity",
    "Mean Angular Vel.",
    "Mean Acceleration",
    "Mean Dist. from Center",
]

N_CLUSTERS = 3
RANDOM_STATE = 42
# Columns to log-transform (rate-based / skewed features; NOT dist_from_center)
LOG_COLS = [0, 1, 2, 3]  # duration, vel, ang_vel, acc

# Syllables to deep-dive on (the ones whose existing plots we inspected)
SPOTLIGHT_SYLLS = [4, 41, 53]

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/rich_tokens/diagnostics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cluster_balance_metrics(labels: np.ndarray, k: int):
    """Return dict with counts, proportions, entropy, and Gini of cluster sizes."""
    counts = np.bincount(labels, minlength=k).astype(float)
    n = counts.sum()
    props = counts / n
    # Entropy (normalised 0-1)
    eps = 1e-12
    H = -np.sum(props * np.log(props + eps)) / np.log(k + eps)
    # Gini coefficient (0=perfect equality, 1=all in one cluster)
    sorted_c = np.sort(counts)
    index = np.arange(1, k + 1)
    gini = ((2 * np.sum(index * sorted_c)) / (k * np.sum(sorted_c))) - (k + 1) / k
    return {
        "counts": counts.astype(int).tolist(),
        "proportions": np.round(props, 4).tolist(),
        "entropy_norm": round(float(H), 4),
        "gini": round(float(gini), 4),
        "min_frac": round(float(props.min()), 4),
        "max_frac": round(float(props.max()), 4),
        "ratio_max_min": round(float(props.max() / (props.min() + eps)), 2),
    }


def fit_and_report(X_raw, syll_id, label, apply_log):
    """Fit K-Means, return labels, metrics, and transformed X."""
    X = X_raw.copy()
    if apply_log:
        X[:, LOG_COLS] = np.log1p(X[:, LOG_COLS])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels) if len(set(labels)) > 1 else 0.0
    bal = cluster_balance_metrics(labels, N_CLUSTERS)
    bal["silhouette"] = round(sil, 4)
    return labels, X_sc, bal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load data ---
    print("Loading AMPH (pre) + IFDN datasets...")
    sessions = []
    try:
        s_amph = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="pre")
        sessions.extend(s_amph)
        print(f"  AMPH pre-injection: {len(s_amph)} sessions")
    except Exception as e:
        print(f"  Could not load AMPH: {e}")
    try:
        s_ifdn = loader.load_dataset("ifdn", base_dir=ROOT_DIR)
        sessions.extend(s_ifdn)
        print(f"  IFDN: {len(s_ifdn)} sessions")
    except Exception as e:
        print(f"  Could not load IFDN: {e}")

    if not sessions:
        print("ERROR: No sessions loaded. Exiting.")
        return

    # --- Extract all bouts ---
    print("Extracting bouts across all sessions...")
    all_bouts = []
    for s in sessions:
        all_bouts.extend(extract_bouts_from_session(s))
    print(f"  Total bouts extracted: {len(all_bouts)}")

    syllable_bouts = defaultdict(list)
    for b in all_bouts:
        syllable_bouts[b["syllable"]].append(b)

    # ========================================================================
    # PART 1: Global Feature Distribution Diagnostics
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 1: Raw Feature Distributions (all bouts pooled)")
    print("=" * 72)
    X_all = np.array([[b[k] for k in FEATURE_KEYS] for b in all_bouts], dtype=np.float64)

    print(f"\n{'Feature':<24} {'Min':>10} {'Median':>10} {'Mean':>10} {'Max':>10} {'Skew':>8} {'Kurtosis':>10}")
    print("-" * 92)
    for i, (fk, fl) in enumerate(zip(FEATURE_KEYS, FEATURE_LABELS)):
        col = X_all[:, i]
        sk = stats.skew(col)
        ku = stats.kurtosis(col)
        print(f"{fl:<24} {col.min():>10.2f} {np.median(col):>10.2f} {col.mean():>10.2f} {col.max():>10.2f} {sk:>8.2f} {ku:>10.2f}")

    print(f"\n{'After log1p:':<24}")
    print(f"{'Feature':<24} {'Min':>10} {'Median':>10} {'Mean':>10} {'Max':>10} {'Skew':>8} {'Kurtosis':>10}")
    print("-" * 92)
    for i, (fk, fl) in enumerate(zip(FEATURE_KEYS, FEATURE_LABELS)):
        col = X_all[:, i]
        if i in LOG_COLS:
            col = np.log1p(col)
            label_suffix = " [log1p]"
        else:
            label_suffix = ""
        sk = stats.skew(col)
        ku = stats.kurtosis(col)
        print(f"{fl + label_suffix:<24} {col.min():>10.2f} {np.median(col):>10.2f} {col.mean():>10.2f} {col.max():>10.2f} {sk:>8.2f} {ku:>10.2f}")

    # --- Visual: Feature distributions (raw vs log1p) ---
    fig, axes = plt.subplots(len(FEATURE_KEYS), 2, figsize=(14, 3 * len(FEATURE_KEYS)))
    fig.suptitle("Feature Distributions: Raw vs log1p", fontsize=16, fontweight="bold", y=1.01)
    for i, (fk, fl) in enumerate(zip(FEATURE_KEYS, FEATURE_LABELS)):
        col_raw = X_all[:, i]
        axes[i, 0].hist(col_raw, bins=100, color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
        axes[i, 0].set_title(f"{fl} — Raw", fontsize=11)
        axes[i, 0].set_ylabel("Count")
        sk_raw = stats.skew(col_raw)
        axes[i, 0].text(0.97, 0.92, f"skew={sk_raw:.2f}", transform=axes[i, 0].transAxes,
                        ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        if i in LOG_COLS:
            col_log = np.log1p(col_raw)
            tag = "log1p"
        else:
            col_log = col_raw
            tag = "unchanged"
        axes[i, 1].hist(col_log, bins=100, color="#dd8452", alpha=0.85, edgecolor="white", linewidth=0.3)
        axes[i, 1].set_title(f"{fl} — {tag}", fontsize=11)
        sk_log = stats.skew(col_log)
        axes[i, 1].text(0.97, 0.92, f"skew={sk_log:.2f}", transform=axes[i, 1].transAxes,
                        ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "01_feature_distributions_raw_vs_log1p.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] {path}")

    # ========================================================================
    # PART 2: Per-Syllable Clustering Comparison (aggregate stats)
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 2: Cluster Balance Comparison — Baseline vs log1p (all syllables)")
    print("=" * 72)

    rows = []
    for syll_id in sorted(syllable_bouts.keys()):
        bouts = syllable_bouts[syll_id]
        n = len(bouts)
        if n < 10:
            continue
        X = np.array([[b[k] for k in FEATURE_KEYS] for b in bouts], dtype=np.float64)
        _, _, bal_base = fit_and_report(X, syll_id, "baseline", apply_log=False)
        _, _, bal_log = fit_and_report(X, syll_id, "log1p", apply_log=True)
        rows.append({
            "syll": syll_id,
            "n_bouts": n,
            "baseline_min_frac": bal_base["min_frac"],
            "baseline_max_frac": bal_base["max_frac"],
            "baseline_ratio": bal_base["ratio_max_min"],
            "baseline_entropy": bal_base["entropy_norm"],
            "baseline_gini": bal_base["gini"],
            "baseline_silhouette": bal_base["silhouette"],
            "log1p_min_frac": bal_log["min_frac"],
            "log1p_max_frac": bal_log["max_frac"],
            "log1p_ratio": bal_log["ratio_max_min"],
            "log1p_entropy": bal_log["entropy_norm"],
            "log1p_gini": bal_log["gini"],
            "log1p_silhouette": bal_log["silhouette"],
        })

    df_comp = pd.DataFrame(rows)
    print(f"\nSyllables compared: {len(df_comp)}")
    print(f"\n--- Aggregate Balance Metrics (mean ± std across syllables) ---")
    for metric in ["min_frac", "ratio", "entropy", "gini", "silhouette"]:
        b_col = f"baseline_{metric}"
        l_col = f"log1p_{metric}"
        b_mean, b_std = df_comp[b_col].mean(), df_comp[b_col].std()
        l_mean, l_std = df_comp[l_col].mean(), df_comp[l_col].std()
        print(f"  {metric:<14}  Baseline: {b_mean:.4f} ± {b_std:.4f}   |   log1p: {l_mean:.4f} ± {l_std:.4f}")

    # How many syllables improved?
    improved_entropy = (df_comp["log1p_entropy"] > df_comp["baseline_entropy"]).sum()
    improved_gini = (df_comp["log1p_gini"] < df_comp["baseline_gini"]).sum()
    improved_min_frac = (df_comp["log1p_min_frac"] > df_comp["baseline_min_frac"]).sum()
    total = len(df_comp)
    print(f"\n--- Improvement Counts ---")
    print(f"  Entropy improved (higher=better):    {improved_entropy}/{total} ({100*improved_entropy/total:.1f}%)")
    print(f"  Gini improved (lower=better):        {improved_gini}/{total} ({100*improved_gini/total:.1f}%)")
    print(f"  Min-fraction improved (higher=better): {improved_min_frac}/{total} ({100*improved_min_frac/total:.1f}%)")

    # Severely imbalanced syllables (min_frac < 0.05)
    severe_base = (df_comp["baseline_min_frac"] < 0.05).sum()
    severe_log = (df_comp["log1p_min_frac"] < 0.05).sum()
    print(f"\n  Severely imbalanced (min_frac < 5%):")
    print(f"    Baseline: {severe_base}/{total}   |   log1p: {severe_log}/{total}")

    # --- Visual: scatter plot entropy and gini baseline vs log1p ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cluster Balance: Baseline vs log1p (each dot = one syllable)", fontsize=14, fontweight="bold")

    for ax, metric, label, better_dir in zip(
        axes,
        ["entropy", "gini", "min_frac"],
        ["Normalised Entropy", "Gini Coefficient", "Smallest Cluster Fraction"],
        ["above", "below", "above"],
    ):
        x = df_comp[f"baseline_{metric}"]
        y = df_comp[f"log1p_{metric}"]
        ax.scatter(x, y, s=18, alpha=0.6, c="#4c72b0", edgecolor="white", linewidth=0.3)
        lims = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Baseline {label}", fontsize=11)
        ax.set_ylabel(f"log1p {label}", fontsize=11)
        ax.set_title(label, fontsize=12)
        # Shade the "better" side
        if better_dir == "above":
            ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.06, color="green")
            ax.text(0.05, 0.95, "log1p better ↑", transform=ax.transAxes, fontsize=9, color="green", va="top")
        else:
            ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.06, color="green")
            ax.text(0.95, 0.05, "log1p better ↓", transform=ax.transAxes, fontsize=9, color="green", ha="right")
        ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "02_balance_scatter_baseline_vs_log1p.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] {path}")

    # --- Visual: distribution of min_frac across syllables ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Distribution of Smallest-Cluster Fraction Across All Syllables", fontsize=14, fontweight="bold")
    for ax, col, label, color in zip(
        axes,
        ["baseline_min_frac", "log1p_min_frac"],
        ["Baseline (StandardScaler only)", "log1p + StandardScaler"],
        ["#4c72b0", "#dd8452"],
    ):
        ax.hist(df_comp[col], bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(1 / N_CLUSTERS, color="red", linestyle="--", alpha=0.7, label=f"Ideal = {1/N_CLUSTERS:.2f}")
        ax.axvline(0.05, color="black", linestyle=":", alpha=0.5, label="Severe threshold (5%)")
        ax.set_xlabel("Min cluster fraction", fontsize=11)
        ax.set_ylabel("# Syllables", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "03_min_frac_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ========================================================================
    # PART 3: Spotlight Syllable Deep-Dives (PCA scatter + box-whisker)
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 3: Spotlight Syllable Deep-Dives")
    print("=" * 72)

    for syll_id in SPOTLIGHT_SYLLS:
        bouts = syllable_bouts.get(syll_id, [])
        n = len(bouts)
        if n < 10:
            print(f"  Syllable {syll_id}: only {n} bouts, skipping.")
            continue

        X = np.array([[b[k] for k in FEATURE_KEYS] for b in bouts], dtype=np.float64)
        labels_base, X_sc_base, bal_base = fit_and_report(X, syll_id, "baseline", apply_log=False)
        labels_log, X_sc_log, bal_log = fit_and_report(X, syll_id, "log1p", apply_log=True)

        print(f"\n  Syllable {syll_id} (n={n}):")
        print(f"    Baseline  — counts={bal_base['counts']}, props={bal_base['proportions']}, "
              f"entropy={bal_base['entropy_norm']}, gini={bal_base['gini']}, sil={bal_base['silhouette']}")
        print(f"    log1p     — counts={bal_log['counts']}, props={bal_log['proportions']}, "
              f"entropy={bal_log['entropy_norm']}, gini={bal_log['gini']}, sil={bal_log['silhouette']}")

        # PCA for visualisation (on each scaled space respectively)
        pca_base = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_sc_base)
        pca_log = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_sc_log)

        # Build a 2-row × 3-col figure
        # Row 1: PCA scatter (baseline, log1p, side-by-side balance bars)
        # Row 2: Feature box-whiskers per cluster (baseline vs log1p)
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        colors = ["#7570b3", "#1b9e77", "#d95f02"]

        # --- Row 1 col 0: PCA baseline ---
        ax = fig.add_subplot(gs[0, 0])
        for ci in range(N_CLUSTERS):
            mask = labels_base == ci
            ax.scatter(pca_base[mask, 0], pca_base[mask, 1], s=10, alpha=0.5,
                       c=colors[ci], label=f"C{ci} ({mask.sum()})")
        ax.set_title(f"Baseline (StandardScaler only)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=9, loc="upper right")

        # --- Row 1 col 1: PCA log1p ---
        ax = fig.add_subplot(gs[0, 1])
        for ci in range(N_CLUSTERS):
            mask = labels_log == ci
            ax.scatter(pca_log[mask, 0], pca_log[mask, 1], s=10, alpha=0.5,
                       c=colors[ci], label=f"C{ci} ({mask.sum()})")
        ax.set_title(f"log1p + StandardScaler", fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=9, loc="upper right")

        # --- Row 1 col 2: Balance bar chart comparison ---
        ax = fig.add_subplot(gs[0, 2])
        x_pos = np.arange(N_CLUSTERS)
        width = 0.35
        ax.bar(x_pos - width / 2, bal_base["proportions"], width, label="Baseline", color="#4c72b0", alpha=0.85)
        ax.bar(x_pos + width / 2, bal_log["proportions"], width, label="log1p", color="#dd8452", alpha=0.85)
        ax.axhline(1 / N_CLUSTERS, color="red", linestyle="--", alpha=0.6, label=f"Ideal ({1/N_CLUSTERS:.2f})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Cluster {i}" for i in range(N_CLUSTERS)])
        ax.set_ylabel("Proportion")
        ax.set_title("Cluster Proportions", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

        # --- Row 2: Feature boxplots per cluster (baseline vs log1p) ---
        # We show 5 features in a compact strip
        for fi, fl in enumerate(FEATURE_LABELS):
            ax = fig.add_subplot(gs[1, 0]) if fi == 0 else fig.add_subplot(gs[1, fi % 3]) if fi < 3 else None
        # Actually, let's do it as a single long axes with grouped boxplots
        # Simpler: 5 subplots in row2 spanning 3 cols via nested gridspec
        gs_inner = gs[1, :].subgridspec(1, len(FEATURE_KEYS), wspace=0.4)
        for fi, fl in enumerate(FEATURE_LABELS):
            ax = fig.add_subplot(gs_inner[0, fi])
            raw_col = X[:, fi]
            data_base = [raw_col[labels_base == ci] for ci in range(N_CLUSTERS)]
            if fi in LOG_COLS:
                raw_log = np.log1p(raw_col)
            else:
                raw_log = raw_col
            data_log = [raw_log[labels_log == ci] for ci in range(N_CLUSTERS)]

            # Two side-by-side boxplots for each cluster
            positions_b = np.array([1, 4, 7]) - 0.5
            positions_l = np.array([1, 4, 7]) + 0.5
            bp1 = ax.boxplot(data_base, positions=positions_b, widths=0.7, patch_artist=True, showfliers=False)
            bp2 = ax.boxplot(data_log, positions=positions_l, widths=0.7, patch_artist=True, showfliers=False)
            for patch in bp1["boxes"]:
                patch.set_facecolor("#4c72b0")
                patch.set_alpha(0.7)
            for patch in bp2["boxes"]:
                patch.set_facecolor("#dd8452")
                patch.set_alpha(0.7)
            ax.set_xticks([1, 4, 7])
            ax.set_xticklabels(["C0", "C1", "C2"], fontsize=9)
            ax.set_title(fl, fontsize=9)
            if fi == 0:
                ax.set_ylabel("Value")

        fig.suptitle(f"Syllable {syll_id} — Clustering Diagnostic (n={n} bouts)", fontsize=15, fontweight="bold", y=1.01)
        path = os.path.join(OUT_DIR, f"04_syllable_{syll_id}_deep_dive.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [Saved] {path}")

    # ========================================================================
    # PART 4: Wilcoxon Signed-Rank Test (paired comparison across syllables)
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART 4: Statistical Tests (Wilcoxon signed-rank, paired across syllables)")
    print("=" * 72)
    for metric, direction in [("entropy", "higher"), ("gini", "lower"), ("min_frac", "higher"), ("silhouette", "higher")]:
        diff = df_comp[f"log1p_{metric}"] - df_comp[f"baseline_{metric}"]
        stat, p = stats.wilcoxon(diff, alternative="two-sided")
        median_diff = diff.median()
        print(f"  {metric:<14}  median Δ = {median_diff:+.4f} ({direction} is better)  "
              f"Wilcoxon W={stat:.0f}, p={p:.4e}"
              f"{'  ★' if p < 0.05 else ''}")

    print("\n" + "=" * 72)
    print("DONE. All diagnostics saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
