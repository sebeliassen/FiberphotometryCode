#!/usr/bin/env python3
"""
===============================================================================
Pipeline 01 Extension: Past Context Window (W) x Future Horizon (H) 2D Grid Sweep
===============================================================================
Evaluates how past behavioral memory window size W in {30s, 60s, 120s} interacts with
future lookahead horizon H in {2s, 5s, 15s, 30s, 60s, 120s} under Session-Grouped 5-Fold CV.

Generates:
  1. 2D Heatmap of Predictive Gain vs. (W, H).
  2. Multi-Horizon Interaction Curves across different past history lengths.
  3. Structured CSV & Markdown report.
===============================================================================
"""

import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository setup
ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

OUT_DIR = os.path.join(LM_DIR, "output", "01_horizon_dynamics")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "past_window_sweep.log")

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

FPS = 80.0
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)

PAST_WINDOWS_SEC = [30.0, 60.0, 120.0]
HORIZONS_SEC = [2.0, 5.0, 15.0, 30.0, 60.0, 120.0]
HORIZONS_FRAMES = [int(h * FPS) for h in HORIZONS_SEC]
NUM_SYLLABLES = 100

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True)
    Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))


def load_continuous_session_data():
    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    log(f"Loading continuous bout cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

    sessions = []
    for sess_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        max_frame = bouts[-1]["end_frame"]
        frame_sylls = np.zeros(max_frame + 1, dtype=int)
        frame_spatial = np.zeros((max_frame + 1, 6), dtype=np.float32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLABLES - 1)
            frame_sylls[sf : ef + 1] = s_id
            vec = b.get("vector", None)
            if vec is not None and len(vec) >= 14:
                frame_spatial[sf : ef + 1, :4] = vec[:4]
                frame_spatial[sf : ef + 1, 4] = vec[10]
                frame_spatial[sf : ef + 1, 5] = vec[12]
        sessions.append({
            "session_idx": sess_idx,
            "session_id": f"sess_{sess_idx:02d}",
            "frame_sylls": frame_sylls,
            "frame_spatial": frame_spatial,
            "n_frames": len(frame_sylls)
        })
    log(f"Loaded {len(sessions)} sessions.")
    return sessions


def run_sweep():
    log("=" * 80)
    log("LAUNCHING 2D PAST CONTEXT (W) x FUTURE HORIZON (H) SWEEP")
    log("=" * 80)

    sessions = load_continuous_session_data()
    max_h_frames = max(HORIZONS_FRAMES)

    grid_results = []

    for w_sec in PAST_WINDOWS_SEC:
        w_frames = int(w_sec * FPS)
        log(f"\n--- Processing Past Window W = {w_sec:.1f}s ({w_frames} frames) ---")

        samples = []
        for s in sessions:
            sess_idx = s["session_idx"]
            sylls = s["frame_sylls"]
            spatial = s["frame_spatial"]
            N = s["n_frames"]
            if N < w_frames + max_h_frames + 200: continue

            for t in range(w_frames, N - max_h_frames, STRIDE_FRAMES):
                curr_syll = sylls[t]
                t_fut_start = t
                while t_fut_start < N and sylls[t_fut_start] == curr_syll:
                    t_fut_start += 1
                if t_fut_start >= N - max_h_frames: continue

                t_norm = t / float(N)
                time_feat = np.array([t_norm, t_norm**2], dtype=np.float32)

                past_sylls = sylls[t - w_frames : t]
                past_spatial = spatial[t - w_frames : t]

                past_counts = np.bincount(past_sylls, minlength=NUM_SYLLABLES).astype(np.float32)
                past_dist = (past_counts + 1e-4) / (np.sum(past_counts) + 1e-4 * NUM_SYLLABLES)

                src = past_sylls[:-1]; dst = past_sylls[1:]
                trans_mat = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES), dtype=np.float32)
                np.add.at(trans_mat, (src, dst), 1.0)
                row_sums = trans_mat.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
                trans_prob = (trans_mat + 1e-4) / (row_sums + 1e-4 * NUM_SYLLABLES)

                spatial_mean = np.mean(past_spatial, axis=0)
                spatial_std = np.std(past_spatial, axis=0)
                spatial_feat = np.hstack([spatial_mean, spatial_std]).astype(np.float32)

                curr_onehot = np.zeros(NUM_SYLLABLES, dtype=np.float32)
                curr_onehot[curr_syll] = 1.0

                targets = {}
                for h_sec, h_frames in zip(HORIZONS_SEC, HORIZONS_FRAMES):
                    fut_sylls = sylls[t_fut_start : t_fut_start + h_frames]
                    f_counts = np.bincount(fut_sylls, minlength=NUM_SYLLABLES).astype(np.float32)
                    targets[h_sec] = (f_counts + 1e-4) / (np.sum(f_counts) + 1e-4 * NUM_SYLLABLES)

                samples.append({
                    "session_idx": sess_idx,
                    "curr_syll": curr_syll,
                    "curr_onehot": curr_onehot,
                    "time_feat": time_feat,
                    "spatial_feat": spatial_feat,
                    "past_dist": past_dist,
                    "past_trans_flat": trans_prob.flatten(),
                    "targets": targets
                })

        log(f"Extracted {len(samples):,} valid samples for W={w_sec:.1f}s.")

        # Fit SVD + KMeans K=16
        X_trans = np.array([s["past_trans_flat"] for s in samples])
        svd = TruncatedSVD(n_components=24, random_state=42)
        latent = svd.fit_transform(X_trans)
        latent_norm = normalize(latent, norm="l2")
        strat_k16 = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(latent_norm)

        for i, s in enumerate(samples):
            s["latent_24d"] = latent[i]
            k16_oh = np.zeros(16, dtype=np.float32)
            k16_oh[strat_k16[i]] = 1.0
            s["strat_k16_oh"] = k16_oh

        groups = np.array([s["session_idx"] for s in samples])
        gkf = GroupKFold(n_splits=5)

        for h_sec in HORIZONS_SEC:
            g_markov, g_macro, g_strat16, g_bound = [], [], [], []
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(samples, groups=groups)):
                tr_samples = [samples[i] for i in tr_idx]
                te_samples = [samples[i] for i in te_idx]

                Y_tr = np.array([s["targets"][h_sec] for s in tr_samples])
                Y_te = np.array([s["targets"][h_sec] for s in te_samples])

                p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_samples), 1))
                p_glob /= np.sum(p_glob, axis=1, keepdims=True)
                js_blind = js_vec(Y_te, p_glob)

                # Standalone Markov
                c_tr = np.array([s["curr_syll"] for s in tr_samples])
                c_te = np.array([s["curr_syll"] for s in te_samples])
                sm = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES), dtype=np.float32)
                for s_id in range(NUM_SYLLABLES):
                    mask = (c_tr == s_id)
                    sm[s_id] = np.mean(Y_tr[mask], axis=0) if np.sum(mask) > 0 else p_glob[0]
                    sm[s_id] /= np.sum(sm[s_id])
                p_m = sm[c_te]
                g_markov.append(((js_blind - js_vec(Y_te, p_m)) / js_blind) * 100.0)

                # Macro (Time + Space)
                X_tr_macro = np.hstack([np.array([s["time_feat"] for s in tr_samples]), np.array([s["spatial_feat"] for s in tr_samples])])
                X_te_macro = np.hstack([np.array([s["time_feat"] for s in te_samples]), np.array([s["spatial_feat"] for s in te_samples])])
                p_macro = np.clip(Ridge(10.0).fit(X_tr_macro, Y_tr).predict(X_te_macro), 1e-4, 1.0)
                p_macro /= np.sum(p_macro, axis=1, keepdims=True)
                g_macro.append(((js_blind - js_vec(Y_te, p_macro)) / js_blind) * 100.0)

                # Strategy K=16
                X_tr_p4 = np.hstack([X_tr_macro, np.array([s["curr_onehot"] for s in tr_samples]), np.array([s["strat_k16_oh"] for s in tr_samples])])
                X_te_p4 = np.hstack([X_te_macro, np.array([s["curr_onehot"] for s in te_samples]), np.array([s["strat_k16_oh"] for s in te_samples])])
                p_p4 = np.clip(Ridge(5.0).fit(X_tr_p4, Y_tr).predict(X_te_p4), 1e-4, 1.0)
                p_p4 /= np.sum(p_p4, axis=1, keepdims=True)
                g_strat16.append(((js_blind - js_vec(Y_te, p_p4)) / js_blind) * 100.0)

                # Past Bound P_max
                X_tr_p5 = np.hstack([X_tr_macro, np.array([s["curr_onehot"] for s in tr_samples]), np.array([s["past_dist"] for s in tr_samples]), np.array([s["latent_24d"] for s in tr_samples])])
                X_te_p5 = np.hstack([X_te_macro, np.array([s["curr_onehot"] for s in te_samples]), np.array([s["past_dist"] for s in te_samples]), np.array([s["latent_24d"] for s in te_samples])])
                p_p5 = np.clip(Ridge(1.0).fit(X_tr_p5, Y_tr).predict(X_te_p5), 1e-4, 1.0)
                p_p5 /= np.sum(p_p5, axis=1, keepdims=True)
                g_bound.append(((js_blind - js_vec(Y_te, p_p5)) / js_blind) * 100.0)

            m_mark = float(np.mean(g_markov))
            m_mac = float(np.mean(g_macro))
            m_s16 = float(np.mean(g_strat16))
            m_b = float(np.mean(g_bound))
            ret = (m_s16 / m_b) * 100.0 if m_b > 0 else 0.0

            log(f"W={w_sec:5.1f}s x H={h_sec:5.1f}s | Macro: +{m_mac:5.2f}% | Strat(K=16): +{m_s16:5.2f}% | P_max: +{m_b:5.2f}% (Ret: {ret:4.1f}%)")

            grid_results.append({
                "past_window_sec": w_sec,
                "horizon_sec": h_sec,
                "gain_markov_mean": m_mark,
                "gain_macro_mean": m_mac,
                "gain_strat16_mean": m_s16,
                "gain_p_max_mean": m_b,
                "retention_k16_pct": ret
            })

    df_grid = pd.DataFrame(grid_results)

    # Save CSV
    csv_path = os.path.join(TAB_DIR, "past_window_vs_future_horizon_results.csv")
    df_grid.to_csv(csv_path, index=False)
    log(f"Saved Grid CSV to: {csv_path}")

    # Generate 2-Panel Figure
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    colors = {30.0: '#e6ab02', 60.0: '#1b9e77', 120.0: '#1f77b4'}
    markers = {30.0: 'o', 60.0: 's', 120.0: '^'}

    # Panel 1: Strategy K=16 across W and H
    for w in PAST_WINDOWS_SEC:
        sub = df_grid[df_grid["past_window_sec"] == w]
        ax1.plot(sub["horizon_sec"], sub["gain_strat16_mean"], f"{markers[w]}-", color=colors[w], lw=2.4, ms=6, label=f"Strategy $K=16$ ($W={int(w)}$s past)")
        ax1.plot(sub["horizon_sec"], sub["gain_p_max_mean"], f"{markers[w]}:", color=colors[w], lw=1.5, alpha=0.6, label=f"$\\mathcal{{P}}_{{\\text{{max}}}}$ ($W={int(w)}$s past)")

    ax1.set_xlabel('Future Lookahead Horizon $H$ (Seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predictive Gain (% $D_{\\text{JS}}$ Error Reduction)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Past Context Window ($W$) vs. Future Lookahead ($H$)', fontsize=12, fontweight='bold')
    ax1.set_xticks(HORIZONS_SEC)
    ax1.set_xticklabels([f"{int(h)}s" for h in HORIZONS_SEC])
    ax1.legend(frameon=True, fontsize=8.5, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: 2D Heatmap of Strategy K=16 Gain
    pivot_gain = df_grid.pivot(index="past_window_sec", columns="horizon_sec", values="gain_strat16_mean")
    im = ax2.imshow(pivot_gain.values, cmap="viridis", aspect="auto", origin="lower")

    ax2.set_xticks(np.arange(len(HORIZONS_SEC)))
    ax2.set_xticklabels([f"{int(h)}s" for h in HORIZONS_SEC])
    ax2.set_yticks(np.arange(len(PAST_WINDOWS_SEC)))
    ax2.set_yticklabels([f"{int(w)}s" for w in PAST_WINDOWS_SEC])

    ax2.set_xlabel('Future Horizon $H$ (Seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Past Context Window $W$ (Seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Strategy $K=16$ Predictive Landscape (% Gain)', fontsize=12, fontweight='bold')

    for i in range(len(PAST_WINDOWS_SEC)):
        for j in range(len(HORIZONS_SEC)):
            val = pivot_gain.values[i, j]
            ax2.text(j, i, f"+{val:.1f}%", ha="center", va="center", color="white" if val < 20 else "black", fontweight="bold", fontsize=9.5)

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('% Gain vs. Blind Baseline', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "past_window_vs_future_horizon_grid.png")
    fig_pdf = os.path.join(FIG_DIR, "past_window_vs_future_horizon_grid.pdf")
    fig.savefig(fig_path)
    fig.savefig(fig_pdf)
    plt.close(fig)
    log(f"Saved Grid Figure to: {fig_path}")

    log("=" * 80)
    log("PAST WINDOW SWEEP COMPLETED SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    run_sweep()
