#!/usr/bin/env python3
"""
run_unified_history_benchmark.py
=============================================================================
UNIFIED COORDINATE SYSTEM BENCHMARK:
Physical Clock Time vs. Event Composition vs. Dwell Duration vs. Sequence Order

Evaluates 5 competing history representations on PERFECTLY MATCHED ANCHORS
under strict within-cohort Leave-One-Animal-Out (LOAO) cross-validation:

1. Baseline M0: Current Syllable S_t (100-D one-hot)
2. Model 1 (Clock-Time Dwell, W=15s): Frame occupancy in past 15 seconds
3. Model 2 (Event Composition, n=20): Flat syllable counts in past 20 bouts (simplex)
4. Model 3 (Duration-Weighted, n=20): Dwell-time fraction across the same 20 bouts
5. Model 4 (Chronological Sequence, n=20): Older 10 bouts vs. Recent 10 bouts
6. Model 5 (Markov Order-2): S_{t-1} + S_t immediate 2-step transition context

Output Deliverables:
- Table: table_unified_history_benchmark.csv
- Table: table_unified_history_benchmark_animals.csv
- Figure: fig_unified_history_benchmark.png / .pdf
=============================================================================
"""

import os
import pickle
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------------
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
W_SEC = 15.0          # Clock time history window (seconds)
W_FRAMES = int(W_SEC * FPS) # ~1200 frames
N_BOUTS = 20          # Discrete event history window (bouts)
H_SEC = 120.0         # Future forecast horizon (seconds)
H_FRAMES = int(H_SEC * FPS) # ~9603 frames
ANCHOR_STRIDE = 2     # Stride over bouts to balance speed & sample size

COHORT_ORDER = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "WT_Amphetamine": "WT Amphetamine",
    "KI_Saline": "KI Saline",
    "KI_Amphetamine": "KI Amphetamine"
}
COLORS = {
    "WT_Saline": "#1f77b4",
    "WT_Amphetamine": "#ff7f0e",
    "KI_Saline": "#2ca02c",
    "KI_Amphetamine": "#9467bd"
}
REP_COLORS = {
    "Clock_Time_15s": "#17becf",
    "Event_Comp_20b": "#2ca02c",
    "Duration_Wt_20b": "#e377c2",
    "Chrono_Split_20b": "#9467bd",
    "Markov_Order_2": "#8c564b"
}

# ---------------------------------------------------------------------------
# UTILITIES: JS DIVERGENCE
# ---------------------------------------------------------------------------
def js_divergence(P, Q):
    P = np.clip(P, 1e-12, 1.0)
    Q = np.clip(Q, 1e-12, 1.0)
    P = P / np.sum(P, axis=-1, keepdims=True)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(np.where(P > 0, P * np.log2(P / np.maximum(M, 1e-12)), 0.0), axis=-1)
    kl_qm = np.sum(np.where(Q > 0, Q * np.log2(Q / np.maximum(M, 1e-12)), 0.0), axis=-1)
    return 0.5 * (kl_pm + kl_qm)

# ---------------------------------------------------------------------------
# BUILD PERFECTLY MATCHED SESSIONS
# ---------------------------------------------------------------------------
def build_matched_sessions():
    print("Loading bout cache and raw metadata...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    with open(RAW_PATH, "rb") as f:
        raw_sessions = pickle.load(f)

    sessions = []
    total_anchors = 0

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
        if len(post_bouts) < N_BOUTS + 50:
            continue

        sylls = np.array([b["syllable"] for b in post_bouts], dtype=np.int32)
        starts = np.array([b["start_frame"] for b in post_bouts], dtype=np.int64)
        ends = np.array([b["end_frame"] for b in post_bouts], dtype=np.int64)
        durs = ends - starts
        max_f = ends[-1]
        M = len(post_bouts)

        # 1. Cumulative Syllable Event Counts over bouts: shape (M + 1, 100)
        cum_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_events[idx_b + 1] = cum_events[idx_b]
            cum_events[idx_b + 1, sylls[idx_b]] += 1.0

        # 2. Cumulative Duration-Weighted Event Counts over bouts: shape (M + 1, 100)
        cum_dur_events = np.zeros((M + 1, NUM_SYLLS), dtype=np.float32)
        for idx_b in range(M):
            cum_dur_events[idx_b + 1] = cum_dur_events[idx_b]
            cum_dur_events[idx_b + 1, sylls[idx_b]] += float(durs[idx_b])

        # 3. Continuous frame array for physical time extraction & target Y
        N_frames = max_f + 1
        cum_frame_sylls = np.zeros((N_frames, NUM_SYLLS), dtype=np.float32)
        for b in post_bouts:
            dur = b["end_frame"] - b["start_frame"]
            if dur > 0:
                cum_frame_sylls[b["start_frame"]:b["end_frame"], b["syllable"]] = 1.0
        cum_frame_sylls = np.cumsum(cum_frame_sylls, axis=0)

        # Find MATCHED ANCHORS
        # Conditions for anchor bout k:
        # - k >= N_BOUTS (at least 20 prior bouts)
        # - starts[k] - W_FRAMES >= starts[0] (at least 15s post-injection recording)
        # - ends[k] + H_FRAMES < N_frames (at least 120s future recording)
        valid_anchors = []
        min_start = starts[0]
        for anc in range(N_BOUTS, M, ANCHOR_STRIDE):
            t_st = starts[anc]
            t_ed = ends[anc]
            if (t_st - W_FRAMES >= min_start) and (t_ed + H_FRAMES < N_frames):
                valid_anchors.append(anc)

        if len(valid_anchors) < 50:
            continue

        valid_anchors = np.array(valid_anchors, dtype=np.int32)
        K = len(valid_anchors)
        total_anchors += K

        # 1. Base Target Y: Future h=120s Syllable distribution (K, 100)
        t_ends = ends[valid_anchors]
        Y_counts = cum_frame_sylls[t_ends + H_FRAMES] - cum_frame_sylls[t_ends]
        Y_dist = Y_counts / (np.sum(Y_counts, axis=1, keepdims=True) + 1e-6)

        # 2. Feature: Current Syllable S_t (K, 100)
        s_curr = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        s_curr[np.arange(K), sylls[valid_anchors]] = 1.0

        # 3. Feature: Physical Clock-Time Dwell (W=15s) (K, 100)
        t_starts = starts[valid_anchors]
        counts_clock = cum_frame_sylls[t_starts] - cum_frame_sylls[t_starts - W_FRAMES]
        feat_clock = counts_clock / (np.sum(counts_clock, axis=1, keepdims=True) + 1e-6)

        # 4. Feature: Discrete Event Composition (n=20 bouts) (K, 100)
        counts_event = cum_events[valid_anchors] - cum_events[valid_anchors - N_BOUTS]
        feat_event = counts_event / float(N_BOUTS)

        # 5. Feature: Duration-Weighted Event Composition (n=20 bouts) (K, 100)
        counts_dur = cum_dur_events[valid_anchors] - cum_dur_events[valid_anchors - N_BOUTS]
        feat_dur = counts_dur / (np.sum(counts_dur, axis=1, keepdims=True) + 1e-6)

        # 6. Feature: Chronological Sequence (Split Halves: Older 10 vs Recent 10) (K, 200)
        counts_older = (cum_events[valid_anchors - 10] - cum_events[valid_anchors - 20]) / 10.0
        counts_recent = (cum_events[valid_anchors] - cum_events[valid_anchors - 10]) / 10.0
        feat_chrono = np.hstack([counts_older, counts_recent])

        # 7. Feature: Markov Order-2 (S_{t-1} + S_t) (K, 200)
        s_prev = np.zeros((K, NUM_SYLLS), dtype=np.float32)
        s_prev[np.arange(K), sylls[valid_anchors - 1]] = 1.0
        feat_markov2 = np.hstack([s_prev, s_curr])

        sessions.append({
            "session_idx": i,
            "sid": sid,
            "cohort": cohort,
            "n_samples": K,
            "Y_dist": Y_dist,
            "s_curr": s_curr,
            "feat_clock": feat_clock,
            "feat_event": feat_event,
            "feat_dur": feat_dur,
            "feat_chrono": feat_chrono,
            "feat_markov2": feat_markov2
        })

    print(f"Built {len(sessions)} valid matched sessions with {total_anchors} total observations.")
    return sessions

# ---------------------------------------------------------------------------
# RUN UNIFIED BENCHMARK UNDER LOAO CROSS-VALIDATION
# ---------------------------------------------------------------------------
def run_benchmark():
    sessions = build_matched_sessions()

    LAMBDAS = [1.0, 10.0, 100.0]
    MAIN_LAMBDA = 10.0

    REPRESENTATIONS = [
        "Clock_Time_15s",
        "Event_Comp_20b",
        "Duration_Wt_20b",
        "Chrono_Split_20b",
        "Markov_Order_2"
    ]

    summary_records = []
    animal_records = []

    print("\n" + "="*80)
    print("RUNNING UNIFIED BENCHMARK ON PERFECTLY MATCHED ANCHORS (LOAO-CV)")
    print("="*80)

    for c in COHORT_ORDER:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        print(f"\nCohort: {c} (N = {n_mice} mice)")

        for lam in LAMBDAS:
            # 1. Evaluate M0 (S_t only)
            m0_scores = {}
            for test_idx in range(n_mice):
                te = c_sess[test_idx]
                tr = [c_sess[j] for j in range(n_mice) if j != test_idx]

                Y_tr = np.vstack([s["Y_dist"] for s in tr])
                Y_te = te["Y_dist"]
                p_prior = np.mean(Y_tr, axis=0, keepdims=True)
                d_prior = np.mean(js_divergence(Y_te, p_prior))

                X0_tr = np.vstack([s["s_curr"] for s in tr])
                X0_te = te["s_curr"]

                beta0 = np.linalg.solve(X0_tr.T @ X0_tr + lam * np.eye(NUM_SYLLS, dtype=np.float32), X0_tr.T @ Y_tr)
                Y0_pr = np.clip(X0_te @ beta0, 1e-12, None); Y0_pr /= Y0_pr.sum(axis=-1, keepdims=True)
                d_m0 = np.mean(js_divergence(Y_te, Y0_pr))
                g_m0 = (d_prior - d_m0) / d_prior * 100.0
                m0_scores[test_idx] = {"d_prior": d_prior, "g_m0": g_m0}

            m0_mean = np.mean([m0_scores[i]["g_m0"] for i in range(n_mice)])

            # 2. Evaluate Each Representation
            for rep in REPRESENTATIONS:
                deltas = []
                total_gains = []

                for test_idx in range(n_mice):
                    te = c_sess[test_idx]
                    tr = [c_sess[j] for j in range(n_mice) if j != test_idx]

                    Y_tr = np.vstack([s["Y_dist"] for s in tr])
                    Y_te = te["Y_dist"]
                    d_prior = m0_scores[test_idx]["d_prior"]
                    g_m0 = m0_scores[test_idx]["g_m0"]

                    if rep == "Clock_Time_15s":
                        F_tr_raw = np.vstack([s["feat_clock"] for s in tr])
                        F_te_raw = te["feat_clock"]
                        base_tr = np.vstack([s["s_curr"] for s in tr])
                        base_te = te["s_curr"]
                    elif rep == "Event_Comp_20b":
                        F_tr_raw = np.vstack([s["feat_event"] for s in tr])
                        F_te_raw = te["feat_event"]
                        base_tr = np.vstack([s["s_curr"] for s in tr])
                        base_te = te["s_curr"]
                    elif rep == "Duration_Wt_20b":
                        F_tr_raw = np.vstack([s["feat_dur"] for s in tr])
                        F_te_raw = te["feat_dur"]
                        base_tr = np.vstack([s["s_curr"] for s in tr])
                        base_te = te["s_curr"]
                    elif rep == "Chrono_Split_20b":
                        F_tr_raw = np.vstack([s["feat_chrono"] for s in tr])
                        F_te_raw = te["feat_chrono"]
                        base_tr = np.vstack([s["s_curr"] for s in tr])
                        base_te = te["s_curr"]
                    elif rep == "Markov_Order_2":
                        F_tr_raw = None
                        X_tr = np.vstack([s["feat_markov2"] for s in tr])
                        X_te = te["feat_markov2"]

                    if F_tr_raw is not None:
                        mu = np.mean(F_tr_raw, axis=0, keepdims=True)
                        sig = np.std(F_tr_raw, axis=0, keepdims=True) + 1e-6
                        F_tr = (F_tr_raw - mu) / sig
                        F_te = (F_te_raw - mu) / sig
                        X_tr = np.hstack([base_tr, F_tr])
                        X_te = np.hstack([base_te, F_te])

                    d_in = X_tr.shape[1]
                    beta = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(d_in, dtype=np.float32), X_tr.T @ Y_tr)
                    Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                    d_m = np.mean(js_divergence(Y_te, Y_pr))
                    g_tot = (d_prior - d_m) / d_prior * 100.0
                    delta = g_tot - g_m0

                    deltas.append(delta)
                    total_gains.append(g_tot)

                    if lam == MAIN_LAMBDA:
                        animal_records.append({
                            "cohort": c,
                            "session_idx": te["session_idx"],
                            "sid": te["sid"],
                            "representation": rep,
                            "lambda": lam,
                            "g_m0": g_m0,
                            "total_gain": g_tot,
                            "delta_history": delta
                        })

                mean_d = float(np.mean(deltas))
                sem_d = float(stats.sem(deltas)) if len(deltas) > 1 else 0.0
                mean_tot = float(np.mean(total_gains))

                summary_records.append({
                    "cohort": c,
                    "representation": rep,
                    "lambda": lam,
                    "m0_gain": m0_mean,
                    "total_gain": mean_tot,
                    "mean_delta": mean_d,
                    "sem_delta": sem_d
                })

                if lam == MAIN_LAMBDA:
                    print(f"  [{rep:17s}] Delta_history = {mean_d:+6.2f}% ± {sem_d:4.2f}% (Total = {mean_tot:5.2f}%)")

    df_summary = pd.DataFrame(summary_records)
    df_animals = pd.DataFrame(animal_records)

    tab_sum_path = os.path.join(TAB_DIR, "table_unified_history_benchmark.csv")
    tab_ani_path = os.path.join(TAB_DIR, "table_unified_history_benchmark_animals.csv")
    df_summary.to_csv(tab_sum_path, index=False)
    df_animals.to_csv(tab_ani_path, index=False)
    print(f"\nSaved tables to:\n  -> {tab_sum_path}\n  -> {tab_ani_path}")

    # Paired stats
    print("\n" + "="*80)
    print("PAIRED STATISTICAL AUDITS PER COHORT (Wilcoxon Signed-Rank Test)")
    print("="*80)
    p_values = {}
    for c in COHORT_ORDER:
        sub = df_animals[df_animals["cohort"] == c]
        p_sub = sub.pivot(index="sid", columns="representation", values="delta_history")

        diff_event_dur = p_sub["Event_Comp_20b"] - p_sub["Duration_Wt_20b"]
        w_stat, p_event_dur = stats.wilcoxon(diff_event_dur)

        diff_event_clk = p_sub["Event_Comp_20b"] - p_sub["Clock_Time_15s"]
        w_stat2, p_event_clk = stats.wilcoxon(diff_event_clk)

        diff_chrono_event = p_sub["Chrono_Split_20b"] - p_sub["Event_Comp_20b"]
        w_stat3, p_chrono_event = stats.wilcoxon(diff_chrono_event)

        p_values[c] = {
            "p_event_dur": p_event_dur,
            "p_event_clk": p_event_clk,
            "p_chrono_event": p_chrono_event,
            "mean_diff_event_dur": float(np.mean(diff_event_dur)),
            "mean_diff_event_clk": float(np.mean(diff_event_clk)),
            "mean_diff_chrono_event": float(np.mean(diff_chrono_event))
        }

        print(f"Cohort {c:16s}:")
        print(f"  * Event vs. Duration-Wt: diff = {np.mean(diff_event_dur):+5.2f}%, p = {p_event_dur:.4f}")
        print(f"  * Event vs. Clock-Time : diff = {np.mean(diff_event_clk):+5.2f}%, p = {p_event_clk:.4f}")
        print(f"  * Chrono vs. Flat Event: diff = {np.mean(diff_chrono_event):+5.2f}%, p = {p_chrono_event:.4f}")

    # Multi-panel figure
    print("\nRendering Publication Figure: fig_unified_history_benchmark.png...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    # PANEL A: Grouped Bar Chart across 4 Cohorts
    ax1 = axes[0]
    sub_main = df_summary[df_summary["lambda"] == MAIN_LAMBDA]

    n_reps = len(REPRESENTATIONS)
    bar_width = 0.15
    rep_labels = [
        "Clock Time (15s)",
        "Event Comp (20b)",
        "Duration-Wt (20b)",
        "Chrono-Split (20b)",
        "Markov-2"
    ]

    for i, rep in enumerate(REPRESENTATIONS):
        vals = []
        errs = []
        for c in COHORT_ORDER:
            row = sub_main[(sub_main["cohort"] == c) & (sub_main["representation"] == rep)]
            vals.append(row["mean_delta"].values[0])
            errs.append(row["sem_delta"].values[0])

        x_pos = np.arange(len(COHORT_ORDER)) + (i - (n_reps - 1) / 2.0) * bar_width
        ax1.bar(x_pos, vals, bar_width, yerr=errs, capsize=3,
                label=rep_labels[i], color=REP_COLORS[rep], alpha=0.88, edgecolor="black", linewidth=0.8)

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.set_xticks(np.arange(len(COHORT_ORDER)))
    ax1.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Incremental History Gain $\\Delta_{\\text{hist}}$ (% above $S_t$)", fontsize=11.5, fontweight="bold")
    ax1.set_title("A. Matched History Representations\n(Identical Anchors & Targets, $h=120$s)", fontsize=12.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9.0, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # PANEL B: Paired Difference per Animal: Event Composition vs. Duration-Weighted
    ax2 = axes[1]
    box_data = []
    for c in COHORT_ORDER:
        sub = df_animals[df_animals["cohort"] == c]
        p_sub = sub.pivot(index="sid", columns="representation", values="delta_history")
        diff = (p_sub["Event_Comp_20b"] - p_sub["Duration_Wt_20b"]).values
        box_data.append(diff)

    bp = ax2.boxplot(box_data, positions=range(4), widths=0.45, patch_artist=True,
                     boxprops=dict(facecolor="#e0e0e0", alpha=0.6),
                     medianprops=dict(color="black", linewidth=2.0))

    for i, c in enumerate(COHORT_ORDER):
        bp["boxes"][i].set_facecolor(COLORS[c])
        bp["boxes"][i].set_alpha(0.45)
        y = box_data[i]
        x = np.random.normal(i, 0.06, size=len(y))
        ax2.scatter(x, y, color=COLORS[c], edgecolors="black", s=55, zorder=5)

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER], fontsize=11, fontweight="bold")
    ax2.set_ylabel("$\\Delta_{\\text{Event}} - \\Delta_{\\text{Duration}}$ (% Difference)", fontsize=11.5, fontweight="bold")
    ax2.set_title("B. Event Counts vs. Dwell Duration\n(>0: Action Token Driven | <0: Dwell Driven)", fontsize=12.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)

    for i, c in enumerate(COHORT_ORDER):
        p_val = p_values[c]["p_event_dur"]
        y_max = np.max(box_data[i]) + 0.3
        sig_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
        ax2.text(i, y_max, sig_str, ha="center", va="bottom", fontsize=9.2, fontweight="bold")

    # PANEL C: Regularization Robustness across lambda in [1, 10, 100]
    ax3 = axes[2]
    for c in COHORT_ORDER:
        sub_c = df_summary[(df_summary["cohort"] == c) & (df_summary["representation"] == "Event_Comp_20b")]
        lams = sub_c["lambda"].values
        means = sub_c["mean_delta"].values
        sems = sub_c["sem_delta"].values
        ax3.errorbar(range(len(lams)), means, yerr=sems, label=COHORT_LABELS[c],
                     color=COLORS[c], marker="o", linewidth=2.2, capsize=4)

    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax3.set_xticks(range(len(LAMBDAS)))
    ax3.set_xticklabels([f"$\\lambda={l}$" for l in LAMBDAS], fontsize=11, fontweight="bold")
    ax3.set_ylabel("Event History Gain $\\Delta_{\\text{Event}}$ (%)", fontsize=11.5, fontweight="bold")
    ax3.set_title("C. Regularization Invariance\n(Event Gain Robust across $\\lambda$ Grid)", fontsize=12.5, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_unified_history_benchmark.png")
    pdf_path = os.path.join(OUT_DIR, "fig_unified_history_benchmark.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "fig_unified_history_benchmark.png"))
    print(f"Saved figure to:\n  -> {fig_path}\n  -> {pdf_path}")
    print(f"Copied to artifact dir: {os.path.join(ARTIFACT_DIR, 'fig_unified_history_benchmark.png')}")

if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    print(f"\nExecution completed in {time.time() - t0:.1f}s.")
