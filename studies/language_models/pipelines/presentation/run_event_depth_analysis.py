"""
run_event_depth_analysis.py
===========================
Bout-Wise (Event-Depth) History Evaluation across 4 Cohorts.

Sweeps the preceding number of discrete behavioral bouts:
  n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50] bouts
under strict Leave-One-Animal-Out (LOAO) cross-validation.

Forecast Target: Future repertoire over h = 120s (matching standard benchmark).

Outputs:
  - Table: table_event_depth_history_curve.csv
  - Table: table_event_depth_animals.csv
  - Figure: fig_event_depth_history_curve.png (.pdf)
"""

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FPS = 80.025
NUM_SYLLS = 100
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
LAMBDA = 10.0

# Fine granularity from 0 to 50 bouts
N_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#d95f02",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#7570b3"
}

def js_divergence(P, Q):
    M = 0.5 * (P + Q)
    kl_pm = np.sum(np.where(P > 0, P * np.log2(P / np.maximum(M, 1e-12)), 0.0), axis=-1)
    kl_qm = np.sum(np.where(Q > 0, Q * np.log2(Q / np.maximum(M, 1e-12)), 0.0), axis=-1)
    return 0.5 * (kl_pm + kl_qm)

def build_bout_sessions():
    print("Loading bout cache and raw metadata...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    with open(RAW_PATH, "rb") as f:
        raw_sessions = pickle.load(f)

    sessions = []
    max_history_bouts = max(N_GRID)

    for i, bouts in enumerate(session_bouts):
        if len(bouts) < 150:
            continue
        raw_s = raw_sessions[i]
        meta = getattr(raw_s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        if cohort not in COHORT_ORDER:
            continue
        inj_f = meta.get("injection_frame", 0)
        sid = meta.get("subject_id", f"mouse_{i}")

        # Post-injection bouts only
        post_bouts = [b for b in bouts if b["start_frame"] > inj_f]
        if len(post_bouts) < max_history_bouts + 50:
            continue

        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int32)
        starts = np.array([b["start_frame"] for b in post_bouts], dtype=np.int64)
        ends = np.array([b["end_frame"] for b in post_bouts], dtype=np.int64)
        max_f = ends[-1]
        M = len(post_bouts)

        # Build cumulative syllable occurrence array over bouts: shape (M + 1, 100)
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0

        # Build continuous frame array for fast target Y extraction over h = 120s
        N_frames = max_f + 1
        cum_frame_sylls = np.zeros((N_frames, NUM_SYLLS), dtype=np.float32)
        for b in post_bouts:
            dur = b["end_frame"] - b["start_frame"]
            if dur > 0:
                cum_frame_sylls[b["start_frame"]:b["end_frame"], b["syllable"]] = 1.0
        cum_frame_sylls = np.cumsum(cum_frame_sylls, axis=0)

        # Sample anchor bout indices: start at max_history_bouts, stride by 2 bouts
        valid_anchors = []
        for anchor_idx in range(max_history_bouts, M, 2):
            t_anchor_end = ends[anchor_idx]
            if t_anchor_end + H_FRAMES < N_frames:
                valid_anchors.append(anchor_idx)

        if len(valid_anchors) < 50:
            continue

        valid_anchors = np.array(valid_anchors, dtype=np.int32)

        # Current Syllable one-hot: shape (n_anchors, 100)
        s_curr = np.zeros((len(valid_anchors), NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(len(valid_anchors)), sylls[valid_anchors]] = 1.0

        # Target Y: shape (n_anchors, 100)
        t_ends = ends[valid_anchors]
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)

        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "anchors": valid_anchors,
            "cum_events": cum_events,
            "s_curr": s_curr,
            "target_y": Y_dist,
            "n_samples": len(valid_anchors)
        })

    print(f"Built {len(sessions)} valid sessions across {len(COHORT_ORDER)} cohorts.")
    return sessions

def run_analysis():
    sessions = build_bout_sessions()

    # Store results per cohort and per animal
    summary_records = []
    animal_records = []

    print("\n" + "="*75)
    print(f"EVALUATING EVENT-DEPTH HISTORY GAIN (LOAO CROSS-VALIDATION, h = {H_SEC}s)")
    print(f"N_GRID: {N_GRID}")
    print("="*75)

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        print(f"\nProcessing Cohort: {c} (N={n_mice} mice)...")

        # 1. Evaluate M0 (Current Syllable S_t only) under LOAO
        m0_per_mouse = {}
        for test_idx in range(n_mice):
            te_s = c_sess[test_idx]
            tr_s = [c_sess[j] for j in range(n_mice) if j != test_idx]

            Y_tr = np.vstack([s["target_y"] for s in tr_s])
            Y_te = te_s["target_y"]
            p_prior = np.mean(Y_tr, axis=0, keepdims=True)
            d_prior = np.mean(js_divergence(Y_te, p_prior))

            X0_tr = np.vstack([s["s_curr"] for s in tr_s])
            X0_te = te_s["s_curr"]

            beta_0 = np.linalg.solve(X0_tr.T @ X0_tr + LAMBDA * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
            Y0_pr = np.clip(X0_te @ beta_0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
            d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
            g_m0 = (d_prior - d_m0) / d_prior * 100.0

            m0_per_mouse[test_idx] = {
                "d_prior": d_prior,
                "g_m0": g_m0
            }

        m0_mean = np.mean([m0_per_mouse[i]["g_m0"] for i in range(n_mice)])

        # 2. Evaluate each n in N_GRID
        for n_bouts in N_GRID:
            if n_bouts == 0:
                summary_records.append({
                    "cohort": c,
                    "n_bouts": 0,
                    "mean_delta_history": 0.0,
                    "sem_delta_history": 0.0,
                    "m0_gain": m0_mean,
                    "total_gain": m0_mean
                })
                continue

            # Extract past feature vector for this n: (cum_events[i] - cum_events[i-n]) / n
            for s in c_sess:
                anc = s["anchors"]
                cnt_p = s["cum_events"][anc] - s["cum_events"][anc - n_bouts]
                s["past_feat"] = cnt_p / float(n_bouts)

            mouse_deltas = []
            for test_idx in range(n_mice):
                te_s = c_sess[test_idx]
                tr_s = [c_sess[j] for j in range(n_mice) if j != test_idx]

                Y_tr = np.vstack([s["target_y"] for s in tr_s])
                Y_te = te_s["target_y"]
                d_prior = m0_per_mouse[test_idx]["d_prior"]
                g_m0 = m0_per_mouse[test_idx]["g_m0"]

                # Standardize past features on training set
                Xp_tr_raw = np.vstack([s["past_feat"] for s in tr_s])
                Xp_te_raw = te_s["past_feat"]
                mu = np.mean(Xp_tr_raw, axis=0, keepdims=True)
                sig = np.std(Xp_tr_raw, axis=0, keepdims=True) + 1e-6
                Xp_tr = (Xp_tr_raw - mu) / sig
                Xp_te = (Xp_te_raw - mu) / sig

                X_tr = np.hstack([np.vstack([s["s_curr"] for s in tr_s]), Xp_tr])
                X_te = np.hstack([te_s["s_curr"], Xp_te])

                beta_n = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(X_tr.shape[1], dtype=np.float32), X_tr.T @ Y_tr)
                Yn_pr = np.clip(X_te @ beta_n, 1e-12, None); Yn_pr /= Yn_pr.sum(axis=-1, keepdims=True)
                d_mn = np.mean(js_divergence(Y_te, Yn_pr))
                g_mn = (d_prior - d_mn) / d_prior * 100.0

                delta_h = g_mn - g_m0
                mouse_deltas.append(delta_h)

                animal_records.append({
                    "cohort": c,
                    "session_idx": te_s["session_idx"],
                    "sid": te_s["sid"],
                    "n_bouts": n_bouts,
                    "g_m0": g_m0,
                    "total_gain": g_mn,
                    "delta_history": delta_h
                })

            mean_d = float(np.mean(mouse_deltas))
            sem_d = float(np.std(mouse_deltas) / np.sqrt(len(mouse_deltas)))
            summary_records.append({
                "cohort": c,
                "n_bouts": n_bouts,
                "mean_delta_history": mean_d,
                "sem_delta_history": sem_d,
                "m0_gain": m0_mean,
                "total_gain": m0_mean + mean_d
            })
            print(f"  n = {n_bouts:2d} bouts: Delta_history = {mean_d:+6.2f}% ± {sem_d:4.2f}%")

    # Save summary and animal tables
    df_sum = pd.DataFrame(summary_records)
    df_anim = pd.DataFrame(animal_records)

    csv_sum = os.path.join(TAB_DIR, "table_event_depth_history_curve.csv")
    csv_anim = os.path.join(TAB_DIR, "table_event_depth_animals.csv")
    df_sum.to_csv(csv_sum, index=False)
    df_anim.to_csv(csv_anim, index=False)
    print(f"\nSaved tables:\n  -> {csv_sum}\n  -> {csv_anim}")

    # Compute optimal event depth n* per animal
    # n* is the n that maximizes delta_history
    peak_records = []
    for s_id in df_anim["session_idx"].unique():
        sub_a = df_anim[df_anim["session_idx"] == s_id]
        c = sub_a["cohort"].iloc[0]
        sid = sub_a["sid"].iloc[0]
        best_row = sub_a.loc[sub_a["delta_history"].idxmax()]
        n_peak = best_row["n_bouts"]
        max_gain = best_row["delta_history"]

        # Zero-crossing / breakdown point: max n where delta_history > 0
        pos_rows = sub_a[sub_a["delta_history"] > 0]
        n_breakdown = pos_rows["n_bouts"].max() if len(pos_rows) > 0 else 0

        peak_records.append({
            "session_idx": s_id,
            "sid": sid,
            "cohort": c,
            "n_peak": n_peak,
            "max_gain": max_gain,
            "n_breakdown": n_breakdown
        })
    df_peaks = pd.DataFrame(peak_records)

    print("\n--- OPTIMAL EVENT HORIZON n* PER COHORT (MEAN ± SEM) ---")
    print(df_peaks.groupby("cohort")[["n_peak", "max_gain", "n_breakdown"]].agg(["mean", "sem"]))

    # Render Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.8), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.suptitle("Behavioral Event-Depth Memory Horizon: Discrete Action Transitions (LOAO Cross-Validation)",
                 fontsize=14, fontweight="bold", y=0.99)

    # Panel A: Mean Delta_history(n) vs n bouts
    for c in COHORT_ORDER:
        sub_c = df_sum[df_sum["cohort"] == c].sort_values("n_bouts")
        n_vals = sub_c["n_bouts"].values
        means = sub_c["mean_delta_history"].values
        sems = sub_c["sem_delta_history"].values

        ax1.plot(n_vals, means, color=COLORS[c], linewidth=2.4, marker="o", markersize=5, label=c.replace("_", " "))
        ax1.fill_between(n_vals, means - sems, means + sems, color=COLORS[c], alpha=0.18)

    ax1.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, label="Current Syllable Baseline ($M_0$)")
    ax1.set_xlabel("Number of Preceding Behavioral Bouts ($n$ Events)", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Incremental Memory Gain $\\Delta_{\\text{hist}}(n)$ (% above $S_t$)", fontsize=11.5, fontweight="bold")
    ax1.set_title("A. Predictive Gain vs. Preceding Behavioral Bouts ($n = 0\\dots 50$)\n(How many discrete prior action steps improve future prediction?)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(-1, 51)
    ax1.set_xticks([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower left", fontsize=9.5, framealpha=0.92)

    # Accurate curve annotations
    ax1.annotate("KI Saline Peaks at n=20 Bouts (+6.7%)\n(Turns downward past n=25 bouts)",
                 xy=(20, 6.71), xytext=(24, 8.2),
                 arrowprops=dict(facecolor=COLORS["KI_Saline"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2, color=COLORS["KI_Saline"])

    ax1.annotate("WT Saline Peaks at n=8-10 Bouts\n(Zero-crossing at n=20 bouts)",
                 xy=(10, 0.30), xytext=(12, 2.2),
                 arrowprops=dict(facecolor=COLORS["WT_Saline"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2, color=COLORS["WT_Saline"])

    ax1.annotate("WT Amphetamine Breakdown\n(Immediate drop below S_t baseline)",
                 xy=(10, -1.86), xytext=(14, -4.5),
                 arrowprops=dict(facecolor=COLORS["WT_Amphetamine"], shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", fontsize=9.2, color=COLORS["WT_Amphetamine"])

    # Panel B: Optimal Event Horizon n* per animal
    box_data = [df_peaks[df_peaks["cohort"] == c]["n_peak"].values for c in COHORT_ORDER]
    bp = ax2.boxplot(box_data, positions=range(4), widths=0.45, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2.0), showfliers=False)
    for patch, c in zip(bp['boxes'], COHORT_ORDER):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[c])
        patch.set_linewidth(1.5)

    for i, c in enumerate(COHORT_ORDER):
        vals = df_peaks[df_peaks["cohort"] == c]["n_peak"].values
        x_jitter = np.random.normal(i, 0.07, size=len(vals))
        ax2.scatter(x_jitter, vals, color=COLORS[c], s=55, edgecolors="white", linewidth=0.8, zorder=5)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels([c.replace("_", "\n") for c in COHORT_ORDER], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Peak Useful Event Horizon ($n^*$ Bouts)", fontsize=11.5, fontweight="bold")
    ax2.set_title("B. Optimal Event Horizon $n^*$ per Animal\n($n$ yielding maximum held-out predictive gain)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig_event_depth_history_curve.png")
    pdf_path = os.path.join(OUT_DIR, "fig_event_depth_history_curve.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

    import shutil
    shutil.copy(png_path, os.path.join(ARTIFACT_DIR, "fig_event_depth_history_curve.png"))

if __name__ == "__main__":
    t0 = time.time()
    run_analysis()
    print(f"Total execution time: {time.time() - t0:.1f}s")
