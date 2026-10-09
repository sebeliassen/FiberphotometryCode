#!/usr/bin/env python3
"""
Module 3: 2D Memory-Timescale Surface Delta(W, h)
Maps the predictive landscape across:
  - History Window Length W in [2.0, 6.0, 15.0, 30.0, 68.3, 120.0] seconds
  - Future Forecast Horizon h in [2.0, 5.0, 15.0, 30.0, 60.0, 120.0] seconds

Evaluated strictly within-cohort under Leave-One-Animal-Out Cross-Validation.
Computes:
  1. Total Predictability Surface: Gain(W, h)
  2. Incremental History Memory Surface: Delta_history(W, h) = Score([S_t, X_past(W)]) - Score(S_t)
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)
from pipelines.presentation.render_4_cohort_figures import js_divergence

CACHE_PATH = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
RAW_PATH = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
OUT_DIR = os.path.join(LM_DIR, "output", "presentation", "figures", "cohort_comparisons")
TAB_DIR = os.path.join(LM_DIR, "output", "presentation", "tables")
ARTIFACT_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/8f712add-07ef-4c3b-ad40-66422a61a805"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FPS = 80.025
NUM_SYLLS = 100
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)

W_GRID = [2.0, 6.0, 15.0, 30.0, 60.0, 120.0]
H_GRID = [2.0, 5.0, 15.0, 30.0, 60.0, 120.0]

COHORTS = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

def build_sessions():
    print("Loading continuous bout cache and raw metadata...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    with open(RAW_PATH, "rb") as f:
        raw_sessions = pickle.load(f)

    sessions = []
    max_w_frames = int(max(W_GRID) * FPS)
    max_h_frames = int(max(H_GRID) * FPS)

    for i, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        raw_s = raw_sessions[i]
        meta = getattr(raw_s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        c = f"{gt}_{drug}"
        if c not in COHORTS: continue
        inj_f = meta.get("injection_frame", 0)

        N = bouts[-1]["end_frame"] + 1
        if N < max_w_frames + max_h_frames + 200: continue

        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        fs = np.full(N, -1, dtype=np.int32)

        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = b["syllable"]
            if 0 <= s_id < NUM_SYLLS:
                fs[sf:ef+1] = s_id
                oh_bout[sf, s_id] = 1.0

        b_starts = np.array([b["start_frame"] for b in bouts])
        next_bout_ptrs = np.full(N, N, dtype=np.int32)
        for t_idx, b_st in enumerate(b_starts):
            prev_st = b_starts[t_idx-1] if t_idx > 0 else 0
            next_bout_ptrs[prev_st:b_st] = b_st
        if len(b_starts) > 0:
            next_bout_ptrs[b_starts[-1]:] = N

        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

        # Post-injection valid anchor frames
        t_anchors = np.arange(max_w_frames, N - max_h_frames, STRIDE_FRAMES)
        t_tgt = next_bout_ptrs[t_anchors]
        valid = (t_tgt + max_h_frames < N) & (t_anchors - max_w_frames > inj_f)
        t_anchors = t_anchors[valid]
        t_tgt = t_tgt[valid]

        if len(t_anchors) == 0: continue

        # Current syllable one-hot at t_anchors
        s_curr = np.zeros((len(t_anchors), NUM_SYLLS), dtype=np.float32)
        curr_sylls = fs[t_anchors]
        valid_s = (curr_sylls >= 0) & (curr_sylls < NUM_SYLLS)
        s_curr[np.where(valid_s)[0], curr_sylls[valid_s]] = 1.0

        sessions.append({
            "sess_idx": i,
            "cohort": c,
            "t_anchors": t_anchors,
            "t_tgt": t_tgt,
            "cum_bout": cum_bout,
            "s_curr": s_curr
        })

    print(f"Loaded {len(sessions)} valid post-injection sessions across 4 cohorts.")
    return sessions

def run_2d_surface():
    print("="*75)
    print("COMPUTING 2D MEMORY-TIMESCALE SURFACES: Delta(W, h)")
    print(f"W Grid (History): {W_GRID} sec")
    print(f"h Grid (Forecast): {H_GRID} sec")
    print("="*75)

    sessions = build_sessions()

    # Results container: [cohort, W, h] -> mean_total_gain, mean_delta_hist
    records = []
    LAMBDA = 10.0

    for c in COHORTS:
        c_sess = [s for s in sessions if s["cohort"] == c]
        n_mice = len(c_sess)
        print(f"\nProcessing Cohort: {c} (N={n_mice} mice)...")

        for h_sec in H_GRID:
            h_f = int(h_sec * FPS)

            # Precompute targets for this h for each mouse
            for s in c_sess:
                cnt = s["cum_bout"][s["t_tgt"] + h_f] - s["cum_bout"][s["t_tgt"]]
                s["target_h"] = (cnt + 1e-4) / (np.sum(cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

            # Baseline M0 (S_t only for this h)
            m0_scores = []
            for test_idx in range(n_mice):
                te_s = c_sess[test_idx]
                tr_s = [c_sess[j] for j in range(n_mice) if j != test_idx]

                Y_tr = np.vstack([s["target_h"] for s in tr_s])
                Y_te = te_s["target_h"]

                p_tr = np.mean(Y_tr, axis=0, keepdims=True)
                d_prior = np.mean(js_divergence(Y_te, p_tr))

                X_s_tr = np.vstack([s["s_curr"] for s in tr_s])
                X_s_te = te_s["s_curr"]

                beta_s = np.linalg.solve(X_s_tr.T @ X_s_tr + LAMBDA * np.eye(X_s_tr.shape[1], dtype=np.float32), X_s_tr.T @ Y_tr)
                Y_pr = np.clip(X_s_te @ beta_s, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(Y_te, Y_pr))
                m0_scores.append((d_prior - d_pr) / d_prior * 100.0)
            m0_mean = np.mean(m0_scores)

            for w_sec in W_GRID:
                w_f = int(w_sec * FPS)

                # Precompute past features for this W
                for s in c_sess:
                    cnt_p = s["cum_bout"][s["t_anchors"]] - s["cum_bout"][s["t_anchors"] - w_f]
                    s["past_w"] = cnt_p / (np.sum(cnt_p, axis=1, keepdims=True) + 1e-6)

                full_scores = []
                for test_idx in range(n_mice):
                    te_s = c_sess[test_idx]
                    tr_s = [c_sess[j] for j in range(n_mice) if j != test_idx]

                    Y_tr = np.vstack([s["target_h"] for s in tr_s])
                    Y_te = te_s["target_h"]
                    p_tr = np.mean(Y_tr, axis=0, keepdims=True)
                    d_prior = np.mean(js_divergence(Y_te, p_tr))

                    # Standardize past on train
                    X_p_tr_raw = np.vstack([s["past_w"] for s in tr_s])
                    X_p_te_raw = te_s["past_w"]
                    mu = np.mean(X_p_tr_raw, axis=0, keepdims=True)
                    sig = np.std(X_p_tr_raw, axis=0, keepdims=True) + 1e-6
                    X_p_tr = (X_p_tr_raw - mu) / sig
                    X_p_te = (X_p_te_raw - mu) / sig

                    X_tr = np.hstack([np.vstack([s["s_curr"] for s in tr_s]), X_p_tr])
                    X_te = np.hstack([te_s["s_curr"], X_p_te])

                    beta = np.linalg.solve(X_tr.T @ X_tr + LAMBDA * np.eye(X_tr.shape[1], dtype=np.float32), X_tr.T @ Y_tr)
                    Y_pr = np.clip(X_te @ beta, 1e-12, None); Y_pr /= Y_pr.sum(axis=-1, keepdims=True)
                    d_pr = np.mean(js_divergence(Y_te, Y_pr))
                    full_scores.append((d_prior - d_pr) / d_prior * 100.0)

                full_mean = np.mean(full_scores)
                delta_hist = full_mean - m0_mean

                records.append({
                    "cohort": c,
                    "W_sec": w_sec,
                    "h_sec": h_sec,
                    "m0_gain": m0_mean,
                    "total_gain": full_mean,
                    "delta_history": delta_hist
                })

        print(f"  Finished {c} grid (36 points).")

    df_grid = pd.DataFrame(records)
    csv_path = os.path.join(TAB_DIR, "table_2d_memory_timescale_surface.csv")
    df_grid.to_csv(csv_path, index=False)
    print(f"\nSaved 2D surface table: {csv_path}")

    # -------------------------------------------------------------------------
    # PLOTTING 2D HEATMAP CONTOURS: 1x4 Panels for Delta_History(W, h)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.0), dpi=300, sharey=True)

    for ax, c in zip(axes, COHORTS):
        sub = df_grid[df_grid["cohort"] == c]
        pivot = sub.pivot(index="W_sec", columns="h_sec", values="delta_history")

        # Sort index and columns
        pivot = pivot.loc[W_GRID, H_GRID]

        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower",
                       interpolation="bicubic")

        # Overlay values
        for i in range(len(W_GRID)):
            for j in range(len(H_GRID)):
                val = pivot.values[i, j]
                text_col = "white" if val < np.max(pivot.values)*0.6 else "black"
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color=text_col)

        ax.set_xticks(range(len(H_GRID)))
        ax.set_xticklabels([f"{int(h)}s" for h in H_GRID], fontsize=9.5, fontweight="bold")
        ax.set_yticks(range(len(W_GRID)))
        ax.set_yticklabels([f"{int(w)}s" for w in W_GRID], fontsize=9.5, fontweight="bold")

        ax.set_xlabel("Forecast Horizon $h$ (s)", fontsize=10.5, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("History Window $W$ (s)", fontsize=10.5, fontweight="bold")
        ax.set_title(c.replace("_", " "), fontsize=11.5, fontweight="bold", pad=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta_{\\text{hist}}$ (% above $S_t$)")

    fig.suptitle("2D Memory-Timescale Surfaces: $\\Delta_{\\text{history}}(W, h) = \\text{Score}(S_t + X_{\\text{past}}) - \\text{Score}(S_t)$\n(Strict Leave-One-Animal-Out Within-Cohort Cross-Validation)",
                 fontsize=13.0, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_png = os.path.join(OUT_DIR, "fig_2d_memory_timescale_surface.png")
    fig_pdf = os.path.join(OUT_DIR, "fig_2d_memory_timescale_surface.pdf")
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"Saved: {fig_png} and {fig_pdf}")
    import shutil
    shutil.copy(fig_png, os.path.join(ARTIFACT_DIR, "fig_2d_memory_timescale_surface.png"))

if __name__ == "__main__":
    t0 = time.time()
    run_2d_surface()
    print(f"Total time: {time.time() - t0:.1f}s")
