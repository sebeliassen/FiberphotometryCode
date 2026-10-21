#!/usr/bin/env python3
"""
karen_strategy_benchmark.py

Multi-Horizon Action Budget Forecasting Benchmark on the Karen Dataset.
Evaluates both Overall and Genotype-Separated (Monomer vs. PFF) cohorts
using RLE-collapsed discrete syllable bouts and exact AMPH Bayesian champions.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PKL = ROOT_DIR / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = ROOT_DIR / "syllable_generation" / "analysis" / "language_models" / "output" / "karen_strategy_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_BOUTS = 300
HORIZONS_BOUTS = [30, 60, 90, 120]
NUM_SYLLABLES = 100

CHAMPIONS = {
    "KMeans (K=8)": {
        "family": "KMeans",
        "k": 8,
        "svd_dim": 96,
        "norm_type": "l1",
        "laplace_smooth": 0.0357,
        "spherical": False,
        "n_init": 20
    },
    "KMeans (K=16)": {
        "family": "KMeans",
        "k": 16,
        "svd_dim": 24,
        "norm_type": "l1",
        "laplace_smooth": 0.0085,
        "spherical": False,
        "n_init": 20
    },
    "GMM (K=8)": {
        "family": "GMM",
        "k": 8,
        "svd_dim": 24,
        "norm_type": "log1p",
        "laplace_smooth": 0.002,
        "cov_type": "spherical",
        "reg_covar": 0.01
    },
    "GMM (K=16)": {
        "family": "GMM",
        "k": 16,
        "svd_dim": 32,
        "norm_type": "none",
        "laplace_smooth": 0.0867,
        "cov_type": "spherical",
        "reg_covar": 0.0115
    }
}


def compute_js_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def extract_rle_bout_samples(sessions, target_genotype=None, exclude_drug=True):
    session_bouts = []
    for s in sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        aid = str(md.get("animal_id", ""))
        cond = md.get("condition", "Unknown")

        if g not in ["PFF", "Monomer"]:
            continue
        if target_genotype is not None and g != target_genotype:
            continue
        if exclude_drug and cond == "L_dopa":
            continue

        df = s.all_data_df.dropna(subset=["syllable"])
        if len(df) < 10000:
            continue

        sylls = df["syllable"].values.astype(int)
        changes = np.concatenate(([True], sylls[1:] != sylls[:-1]))
        collapsed = sylls[changes]

        if len(collapsed) > WIN_BOUTS + max(HORIZONS_BOUTS):
            session_bouts.append({"session_id": s.session_id, "animal_id": aid, "genotype": g, "bouts": collapsed})

    samples = []
    for s in session_bouts:
        aid = s["animal_id"]
        g = s["genotype"]
        b = s["bouts"]
        N = len(b)

        for t in range(WIN_BOUTS, N - max(HORIZONS_BOUTS), 10):
            past_b = b[t - WIN_BOUTS : t]
            curr_b = b[t]

            src = past_b[:-1]
            dst = past_b[1:]
            mat = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES), dtype=np.float32)
            np.add.at(mat, (src, dst), 1.0)

            past_counts = np.bincount(past_b, minlength=NUM_SYLLABLES)
            past_dist = (past_counts + 1e-4) / np.sum(past_counts + 1e-4)

            targets = {}
            for h in HORIZONS_BOUTS:
                fut_seq = b[t + 1 : t + 1 + h]
                f_counts = np.bincount(fut_seq, minlength=NUM_SYLLABLES)
                targets[h] = (f_counts + 1e-4) / np.sum(f_counts + 1e-4)

            samples.append({
                "session_id": s["session_id"],
                "animal_id": aid,
                "genotype": g,
                "curr_syll": curr_b,
                "past_mat": mat.flatten(),
                "past_dist": past_dist,
                "targets": targets
            })

    return samples, session_bouts


def preprocess_features(X_tr_raw, X_te_raw, cfg):
    smooth = cfg["laplace_smooth"]
    norm_type = cfg["norm_type"]
    svd_dim = cfg["svd_dim"]

    mat_tr = X_tr_raw + smooth
    mat_te = X_te_raw + smooth

    if norm_type == "log1p":
        mat_tr = np.log1p(mat_tr)
        mat_te = np.log1p(mat_te)
    elif norm_type == "l1":
        mat_tr = normalize(mat_tr, norm="l1")
        mat_te = normalize(mat_te, norm="l1")
    elif norm_type == "l2":
        mat_tr = normalize(mat_tr, norm="l2")
        mat_te = normalize(mat_te, norm="l2")

    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    feats_tr = svd.fit_transform(mat_tr)
    feats_te = svd.transform(mat_te)

    if norm_type == "standard":
        scaler = StandardScaler()
        feats_tr = scaler.fit_transform(feats_tr)
        feats_te = scaler.transform(feats_te)
    elif cfg.get("spherical", False) or norm_type in ["l2", "log1p"]:
        feats_tr = normalize(feats_tr, norm="l2")
        feats_te = normalize(feats_te, norm="l2")

    return feats_tr, feats_te


def evaluate_cohort(samples, cohort_name):
    print("\n=======================================================")
    print(f"EVALUATING COHORT: {cohort_name} ({len(samples)} windows)")
    print("=======================================================")

    animals = sorted(list(set(s["animal_id"] for s in samples)))
    np.random.seed(42)
    shuffled = np.random.permutation(animals)
    n_train = max(1, int(0.8 * len(shuffled)))
    tr_animals = set(shuffled[:n_train])

    tr_samples = [s for s in samples if s["animal_id"] in tr_animals]
    te_samples = [s for s in samples if s["animal_id"] not in tr_animals]

    if len(te_samples) == 0:
        te_samples = tr_samples[:int(0.2 * len(tr_samples))]
        tr_samples = tr_samples[int(0.2 * len(tr_samples)):]

    X_tr_mat = np.array([s["past_mat"] for s in tr_samples])
    X_te_mat = np.array([s["past_mat"] for s in te_samples])
    X_tr_past = np.array([s["past_dist"] for s in tr_samples])
    X_te_past = np.array([s["past_dist"] for s in te_samples])
    c_tr = np.array([s["curr_syll"] for s in tr_samples])
    c_te = np.array([s["curr_syll"] for s in te_samples])

    # Fit Strategy Models
    fitted = {}
    for name, cfg in CHAMPIONS.items():
        feats_tr, feats_te = preprocess_features(X_tr_mat, X_te_mat, cfg)
        if cfg["family"] == "KMeans":
            km = KMeans(n_clusters=cfg["k"], n_init=cfg["n_init"], random_state=42)
            lbl_tr = km.fit_predict(feats_tr)
            lbl_te = km.predict(feats_te)
        else:
            gmm = GaussianMixture(n_components=cfg["k"], covariance_type=cfg["cov_type"], reg_covar=cfg["reg_covar"], random_state=42, n_init=3)
            lbl_tr = gmm.fit_predict(feats_tr)
            lbl_te = gmm.predict(feats_te)
        fitted[name] = {"k": cfg["k"], "lbl_tr": lbl_tr, "lbl_te": lbl_te}

    rows = []
    for h in HORIZONS_BOUTS:
        Y_tr = np.array([s["targets"][h] for s in tr_samples])
        Y_te = np.array([s["targets"][h] for s in te_samples])

        p_glob = np.mean(Y_tr, axis=0)
        p_glob /= np.sum(p_glob)
        js_blind = np.mean([compute_js_divergence(y, p_glob) for y in Y_te])

        # 1. Syllable Model
        s_means = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES))
        for s_id in range(NUM_SYLLABLES):
            mask = (c_tr == s_id)
            s_means[s_id] = np.mean(Y_tr[mask], axis=0) if np.sum(mask) > 0 else p_glob
            s_means[s_id] /= np.sum(s_means[s_id])
        js_syll = np.mean([compute_js_divergence(y, s_means[s]) for y, s in zip(Y_te, c_te)])
        gain_syll = ((js_blind - js_syll) / js_blind) * 100.0

        # 2. Ceiling
        rg = Ridge(alpha=1.0).fit(X_tr_past, Y_tr)
        pred_ceil = np.clip(rg.predict(X_te_past), 1e-4, None)
        pred_ceil /= np.sum(pred_ceil, axis=1, keepdims=True)
        js_ceil = np.mean([compute_js_divergence(y, p) for y, p in zip(Y_te, pred_ceil)])
        gain_ceil = ((js_blind - js_ceil) / js_blind) * 100.0

        row = {
            "cohort": cohort_name,
            "horizon_bouts": h,
            "js_blind": js_blind,
            "gain_syllable_pct": gain_syll,
            "pct_ceil_syllable": (gain_syll / gain_ceil) * 100.0 if gain_ceil > 0 else 0.0,
            "gain_ceiling_pct": gain_ceil
        }

        for name, m in fitted.items():
            k = m["k"]
            lbl_tr, lbl_te = m["lbl_tr"], m["lbl_te"]
            strat_means = np.zeros((k, NUM_SYLLABLES))
            for k_id in range(k):
                mask = (lbl_tr == k_id)
                strat_means[k_id] = np.mean(Y_tr[mask], axis=0) if np.sum(mask) > 0 else p_glob
                strat_means[k_id] /= np.sum(strat_means[k_id])
            js_strat = np.mean([compute_js_divergence(y, strat_means[idx]) for y, idx in zip(Y_te, lbl_te)])
            gain_strat = ((js_blind - js_strat) / js_blind) * 100.0

            key = name.lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            row[f"gain_{key}_pct"] = gain_strat
            row[f"pct_ceil_{key}"] = (gain_strat / gain_ceil) * 100.0 if gain_ceil > 0 else 0.0

        print(f"H={h:3d} bouts | Syllable: +{gain_syll:5.2f}% ({row['pct_ceil_syllable']:4.1f}% ceil) | KMeans-16: +{row['gain_kmeans_k16_pct']:5.2f}% ({row['pct_ceil_kmeans_k16']:4.1f}% ceil) | GMM-16: +{row['gain_gmm_k16_pct']:5.2f}% | Ceiling: +{gain_ceil:5.2f}%")
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Unsupervised Discrete Bout Strategy Benchmark on Karen Dataset")
    print("=" * 80)
    start_time = time.time()

    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)

    # 1. Full Combined Untreated Cohort
    all_samples, _ = extract_rle_bout_samples(sessions, target_genotype=None, exclude_drug=True)
    df_all = evaluate_cohort(all_samples, "Combined_Untreated")

    # 2. Monomer Control Cohort Alone
    mon_samples, _ = extract_rle_bout_samples(sessions, target_genotype="Monomer", exclude_drug=True)
    df_mon = evaluate_cohort(mon_samples, "Monomer_Control")

    # 3. PFF Cohort Alone
    pff_samples, _ = extract_rle_bout_samples(sessions, target_genotype="PFF", exclude_drug=True)
    df_pff = evaluate_cohort(pff_samples, "PFF_Untreated")

    df_combined = pd.concat([df_all, df_mon, df_pff], ignore_index=True)
    csv_path = OUT_DIR / "karen_strategy_horizon_results.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"\nSaved master results to: {csv_path}")

    # 4. Generate Publication Figure (3 Panels: Combined, Monomer, PFF)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    cohorts_meta = [
        ("Combined_Untreated", "A. All Untreated Sessions (N=78)", axes[0]),
        ("Monomer_Control", "B. Monomer Control Cohort (N=34)", axes[1]),
        ("PFF_Untreated", "C. PFF Disease Cohort (N=44)", axes[2])
    ]

    for c_name, title, ax in cohorts_meta:
        sub = df_combined[df_combined["cohort"] == c_name]
        h_vals = sub["horizon_bouts"].values

        ax.plot(h_vals, sub["gain_syllable_pct"], marker="o", color="#7f7f7f", linewidth=2.0, linestyle="--", label="Instant Syllable S_t (Markovian)")
        ax.plot(h_vals, sub["gain_kmeans_k8_pct"], marker="s", color="#1f77b4", linewidth=2.0, label="KMeans (K=8)")
        ax.plot(h_vals, sub["gain_kmeans_k16_pct"], marker="D", color="#2ca02c", linewidth=2.5, label="KMeans (K=16)")
        ax.plot(h_vals, sub["gain_gmm_k16_pct"], marker="v", color="#9467bd", linewidth=2.2, linestyle="-.", label="GMM (K=16)")
        ax.plot(h_vals, sub["gain_ceiling_pct"], marker="*", color="#d62728", linewidth=1.8, linestyle=":", label="100% History Ceiling")

        ax.set_xlabel("Forecast Horizon (Distinct Bouts Ahead)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Predictive Gain (% D_JS Error Reduction)", fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xticks(h_vals)
        ax.set_xticklabels([f"{h} bouts\n(~{int(h*0.35)}s)" for h in h_vals], fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax == axes[0]:
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.8, loc="upper left")

    fig.suptitle("Unsupervised Strategy Modeling Benchmark on Karen Dataset (Bout-Level Multi-Horizon Forecasts)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = OUT_DIR / "karen_non_markovian_horizon.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved publication figure to: {plot_path}")

    elapsed = time.time() - start_time
    print(f"\nAll cohorts evaluated successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
