#!/usr/bin/env python3
"""
===============================================================================
Rigorous 4-Cohort Comparison across Fig 1 and Fig 2 Presentation Representations
(WT Saline, WT Amph, KI Saline, KI Amph — Post-Injection Analysis)
===============================================================================
Enforces:
1. Strict Bout Filtering: Removes glitch/sub-syllable bouts (<= 2 frames / < 37.5 ms).
2. True Future Transitions: Eliminates current-bout continuation leakage.
3. Reference Training on WT Saline Post-Injection -> Out-of-Distribution Transfer to 4 Cohorts.
4. Within-Cohort Cross-Validation with proper regularization (alpha=100.0).
5. Bounded, physically constrained Bateman biexponential fitting for memory curves.
===============================================================================
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation", "analysis", "ulrik_kinematics"))
sys.path.insert(0, ROOT_DIR)

FPS = 80.025
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
MIN_BOUT_FRAMES = 3  # Filter out 1-2 frame glitch bouts (sub-37.5ms noise)

W_VALUES = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 68.3, 75.0, 90.0, 105.0, 120.0, 160.0, 200.0, 240.0, 300.0, 360.0]

def js_divergence(p, q):
    p = np.clip(p, 1e-12, None)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = np.clip(q, 1e-12, None)
    q = q / np.sum(q, axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log2(p / m), axis=-1) + np.sum(q * np.log2(q / m), axis=-1))

def biexp_func(w, P0, A, tau_rise, tau_decay):
    return P0 + A * (np.exp(-w / tau_decay) - np.exp(-w / tau_rise))

def main():
    print("=" * 80)
    print("RIGOROUS 4-COHORT PRESENTATION COMPARISON (FIG 1 & FIG 2)")
    print("=" * 80)

    # 1. Load Sessions & Metadata
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    session_bouts = cache["session_bouts"] if isinstance(cache, dict) and "session_bouts" in cache else cache

    cohort_names = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]

    # 2. Extract clean discrete RLE bouts per session (filtered >= 3 frames)
    print("\nExtracting filtered RLE bouts and sliding evaluation windows...")
    clean_session_data = []

    for s_idx, bouts in enumerate(session_bouts):
        s = raw_sessions[s_idx]
        meta = getattr(s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        inj_f = meta.get("injection_frame", 0)
        entry_f = meta.get("entry_frame", 0)
        sid = getattr(s, "session_id", f"sess_{s_idx}")

        # Filter bouts >= MIN_BOUT_FRAMES
        filtered_bouts = [b for b in bouts if (b["end_frame"] - b["start_frame"] + 1) >= MIN_BOUT_FRAMES]
        if len(filtered_bouts) < 100: continue

        max_frame = filtered_bouts[-1]["end_frame"]
        N = max_frame + 1

        # Discrete onset matrix (1.0 only at start frame of genuine bouts)
        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        frame_sylls = np.zeros(N, dtype=np.int32)

        for b in filtered_bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLS - 1)
            frame_sylls[sf : ef + 1] = s_id
            oh_bout[sf, s_id] = 1.0

        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

        # Extract sliding windows
        max_w_frames = int(max(W_VALUES) * FPS)

        for t in range(max_w_frames, N - H_FRAMES, STRIDE_FRAMES):
            # Target future: distinct bouts initiated between [t, t + H_FRAMES]
            y_count = cum_bout[t + H_FRAMES] - cum_bout[t]
            tot_y = np.sum(y_count)
            if tot_y < 5.0: continue  # require at least 5 future bouts
            y_dist = y_count / tot_y

            # Current syllable one-hot
            curr_s = frame_sylls[t]
            curr_oh = np.zeros(NUM_SYLLS, dtype=np.float32)
            curr_oh[curr_s] = 1.0

            # Past multi-window counts
            past_w = {}
            for w_sec in W_VALUES:
                w_fr = int(w_sec * FPS)
                past_w[w_sec] = cum_bout[t] - cum_bout[t - w_fr]

            is_post = (t >= inj_f + int(68.3 * FPS))
            is_pre = (t < inj_f and t >= entry_f + max_w_frames)
            t_min_inj = (t - inj_f) / FPS / 60.0

            clean_session_data.append({
                "session_idx": s_idx,
                "sid": sid,
                "cohort": cohort,
                "genotype": gt,
                "drug": drug,
                "t_frame": t,
                "t_min_inj": t_min_inj,
                "is_post": is_post,
                "is_pre": is_pre,
                "curr_syll_id": curr_s,
                "curr_syll_oh": curr_oh,
                "y_dist": y_dist,
                "past_w": past_w
            })

    df_win = pd.DataFrame(clean_session_data)
    df_post = df_win[df_win["is_post"]].copy()
    print(f"Total post-injection windows: {len(df_post)}")
    for c in cohort_names:
        print(f"  {c:15s}: {len(df_post[df_post['cohort'] == c])} windows ({len(df_post[df_post['cohort'] == c]['session_idx'].unique())} animals)")

    # -------------------------------------------------------------------------
    # PART 1: FIG 1 & FIG 1D — NMF MACRO-COMPONENTS (K=4) ACROSS 4 COHORTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: FIG 1 & FIG 1D — NMF MACRO-COMPONENTS (K=4) ACROSS 4 COHORTS")
    print("=" * 80)

    # Fit NMF K=4 across all 68.3s past windows
    X_68_all = np.array([w[68.3] for w in df_win["past_w"]], dtype=np.float32)
    K = 4
    nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
    nmf.fit(X_68_all)

    print("\n--- Canonical NMF Components (K=4) ---")
    for k in range(K):
        top_s = np.argsort(nmf.components_[k])[::-1][:6]
        top_w = nmf.components_[k][top_s]
        s_str = ", ".join([f"Syllable {s} ({w:.1f})" for s, w in zip(top_s, top_w)])
        print(f"  Component {k+1}: {s_str}")

    Z_all = nmf.transform(X_68_all)
    Z_probs = Z_all / np.clip(np.sum(Z_all, axis=1, keepdims=True), 1e-8, None)
    for k in range(K):
        df_win[f"z_{k+1}"] = Z_probs[:, k]

    z_clip = np.clip(Z_probs, 1e-12, None)
    df_win["z_entropy"] = -np.sum(z_clip * np.log2(z_clip), axis=1)
    df_post = df_win[df_win["is_post"]].copy()

    # Per-animal aggregation
    sess_summary = df_post.groupby(["cohort", "genotype", "drug", "session_idx"])[
        ["z_1", "z_2", "z_3", "z_4", "z_entropy"]
    ].mean().reset_index()

    print("\n--- Post-Injection NMF Component Occupancy (Mean ± SEM across Animals) ---")
    cohort_stats = {}
    for c in cohort_names:
        sub = sess_summary[sess_summary["cohort"] == c]
        cohort_stats[c] = sub
        print(f"\nCohort: {c} (N = {len(sub)} animals):")
        for k in range(K):
            m, s = sub[f"z_{k+1}"].mean() * 100, sub[f"z_{k+1}"].sem() * 100
            print(f"  Component {k+1}: {m:5.2f}% ± {s:4.2f}%")
        m_e, s_e = sub["z_entropy"].mean(), sub["z_entropy"].sem()
        print(f"  State Entropy: {m_e:5.3f} ± {s_e:4.3f} bits")

    # Statistical comparisons
    print("\n--- Statistical Hypotheses Testing (2-Sample Welch t-test & Mann-Whitney) ---")
    comps = [
        ("WT_Saline", "WT_Amphetamine", "Amphetamine Effect in WT"),
        ("KI_Saline", "KI_Amphetamine", "Amphetamine Effect in KI"),
        ("WT_Saline", "KI_Saline", "Genotype Baseline Effect (Saline)"),
        ("WT_Amphetamine", "KI_Amphetamine", "Genotype Drug Response Effect (Amph)")
    ]
    for c1, c2, title in comps:
        s1, s2 = cohort_stats[c1], cohort_stats[c2]
        print(f"\n[{title}] ({c1} vs {c2}):")
        for k in range(K):
            t_stat, p_val = ttest_ind(s1[f"z_{k+1}"], s2[f"z_{k+1}"], equal_var=False)
            u_stat, p_mw = mannwhitneyu(s1[f"z_{k+1}"], s2[f"z_{k+1}"])
            diff = (s2[f"z_{k+1}"].mean() - s1[f"z_{k+1}"].mean()) * 100
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  Component {k+1}: Delta = {diff:+5.2f} pp | t = {t_stat:5.2f}, p_t = {p_val:.4f}, p_mw = {p_mw:.4f} [{sig}]")
        t_stat, p_val = ttest_ind(s1["z_entropy"], s2["z_entropy"], equal_var=False)
        diff_e = s2["z_entropy"].mean() - s1["z_entropy"].mean()
        sig_e = "*" if p_val < 0.05 else "n.s."
        print(f"  State Entropy : Delta = {diff_e:+5.3f} bits | t = {t_stat:5.2f}, p = {p_val:.4f} [{sig_e}]")

    # -------------------------------------------------------------------------
    # PART 2: FIG 1B — NMF DIMENSIONAL RETENTION (K in [1..30])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: FIG 1B — NMF PREDICTIVE RETENTION ACROSS K IN [1..30]")
    print("=" * 80)

    K_LIST = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30]
    wtsal_sub = df_post[df_post["cohort"] == "WT_Saline"]
    X_68_wtsal = np.array([w[68.3] for w in wtsal_sub["past_w"]], dtype=np.float32)
    Y_wtsal = np.array(wtsal_sub["y_dist"].tolist(), dtype=np.float32)

    # Evaluate transfer when trained on WT Saline
    retention_results = []
    for k_val in K_LIST:
        nmf_k = NMF(n_components=k_val, init='nndsvda', random_state=42, max_iter=600)
        Z_train = nmf_k.fit_transform(X_68_wtsal)

        reg = Ridge(alpha=100.0)
        reg.fit(Z_train, Y_wtsal)

        row = {"K": k_val}
        for c in cohort_names:
            c_sub = df_post[df_post["cohort"] == c]
            X_test = np.array([w[68.3] for w in c_sub["past_w"]], dtype=np.float32)
            Y_test = np.array(c_sub["y_dist"].tolist(), dtype=np.float32)

            p0 = np.mean(Y_test, axis=0, keepdims=True)
            d_null = np.mean(js_divergence(Y_test, p0))

            Z_test = nmf_k.transform(X_test)
            Y_pred = np.clip(reg.predict(Z_test), 1e-12, None)
            Y_pred = Y_pred / np.sum(Y_pred, axis=-1, keepdims=True)
            d_pred = np.mean(js_divergence(Y_test, Y_pred))

            row[c] = (d_null - d_pred) / d_null * 100.0
        retention_results.append(row)

    df_retention = pd.DataFrame(retention_results)
    print("\nPredictive Gain (% Delta D_JS) vs NMF Components K (Trained on WT Saline):")
    print(df_retention.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 3: FIG 2 — MEMORY CAPACITY TIMESCALE CURVE (W IN [2..360s])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: FIG 2 — TIMESCALE MEMORY CURVES (W IN [2..360s])")
    print("=" * 80)

    # Feature: [Current Syllable OH (100-D) + Past Repertoire BoW(W) (100-D)] -> Future Y (100-D)

    # Case A: Trained on WT Saline Post-Injection -> Evaluated on 4 Cohorts
    mem_transfer_results = []
    for w_sec in W_VALUES:
        row = {"W_sec": w_sec}
        X_tr = np.hstack([np.array(wtsal_sub["curr_syll_oh"].tolist()), np.array([w[w_sec] for w in wtsal_sub["past_w"]])])
        d_feat = X_tr.shape[1]
        I_d = np.eye(d_feat, dtype=np.float32)
        beta_wtsal = np.linalg.solve(X_tr.T @ X_tr + 500.0 * I_d, X_tr.T @ Y_wtsal)

        for c in cohort_names:
            c_sub = df_post[df_post["cohort"] == c]
            X_te = np.hstack([np.array(c_sub["curr_syll_oh"].tolist()), np.array([w[w_sec] for w in c_sub["past_w"]])])
            Y_te = np.array(c_sub["y_dist"].tolist())

            p0 = np.mean(Y_te, axis=0, keepdims=True)
            d_null = np.mean(js_divergence(Y_te, p0))

            Y_pr = np.clip(X_te @ beta_wtsal, 1e-12, None)
            Y_pr = Y_pr / np.sum(Y_pr, axis=-1, keepdims=True)
            d_pr = np.mean(js_divergence(Y_te, Y_pr))

            row[c] = (d_null - d_pr) / d_null * 100.0
        mem_transfer_results.append(row)

    df_mem_trans = pd.DataFrame(mem_transfer_results)
    print("\n--- Memory Curve: Trained on WT Saline Reference ---")
    print(df_mem_trans.to_string(index=False))

    # Case B: Within-Cohort 5-Fold / Cross-Session CV
    mem_within_results = []
    for w_sec in W_VALUES:
        row = {"W_sec": w_sec}
        for c in cohort_names:
            c_sub = df_post[df_post["cohort"] == c]
            sess_ids = c_sub["session_idx"].unique()
            c_gains = []

            for test_s in sess_ids:
                tr = c_sub[c_sub["session_idx"] != test_s]
                te = c_sub[c_sub["session_idx"] == test_s]
                if len(tr) == 0 or len(te) == 0: continue

                X_tr = np.hstack([np.array(tr["curr_syll_oh"].tolist()), np.array([w[w_sec] for w in tr["past_w"]])])
                Y_tr = np.array(tr["y_dist"].tolist())
                X_te = np.hstack([np.array(te["curr_syll_oh"].tolist()), np.array([w[w_sec] for w in te["past_w"]])])
                Y_te = np.array(te["y_dist"].tolist())

                p0 = np.mean(Y_te, axis=0, keepdims=True)  # test set marginal baseline
                d_null = np.mean(js_divergence(Y_te, p0))

                d_feat = X_tr.shape[1]
                I_d = np.eye(d_feat, dtype=np.float32)
                beta = np.linalg.solve(X_tr.T @ X_tr + 500.0 * I_d, X_tr.T @ Y_tr)

                Y_pr = np.clip(X_te @ beta, 1e-12, None)
                Y_pr = Y_pr / np.sum(Y_pr, axis=-1, keepdims=True)
                d_pr = np.mean(js_divergence(Y_te, Y_pr))

                c_gains.append((d_null - d_pr) / d_null * 100.0)
            row[c] = np.mean(c_gains)
        mem_within_results.append(row)

    df_mem_within = pd.DataFrame(mem_within_results)
    print("\n--- Memory Curve: Within-Cohort Cross-Validation ---")
    print(df_mem_within.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 4: FIT PHYSICAL BATEMAN BIEXPONENTIAL CURVES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: BATEMAN BIEXPONENTIAL FIT PARAMETERS (WITHIN-COHORT CV)")
    print("=" * 80)

    w_arr = df_mem_within["W_sec"].values

    for c in cohort_names:
        g_arr = df_mem_within[c].values
        try:
            # Bounded parameters: P0 in [0, 40], A in [0, 40], tau_rise in [2, 100], tau_decay in [50, 3000]
            popt, _ = curve_fit(
                biexp_func, w_arr, g_arr,
                p0=[15.0, 15.0, 18.0, 600.0],
                bounds=([0.0, 0.0, 1.0, 40.0], [45.0, 45.0, 150.0, 5000.0]),
                maxfev=20000
            )
            p0_fit, a_fit, t_rise, t_decay = popt
            w_dense = np.linspace(0.1, 400.0, 2000)
            g_dense = biexp_func(w_dense, *popt)
            w_peak = w_dense[np.argmax(g_dense)]
            g_peak = np.max(g_dense)

            # Compute R^2
            res = g_arr - biexp_func(w_arr, *popt)
            ss_res = np.sum(res**2)
            ss_tot = np.sum((g_arr - np.mean(g_arr))**2)
            r2 = 1.0 - (ss_res / ss_tot)

            print(f"\nCohort: {c}")
            print(f"  P0 (Instantaneous Gain at W=0s) : {p0_fit:5.2f}%")
            print(f"  P_max (Peak Predictive Gain)     : {g_peak:5.2f}%")
            print(f"  W* (Optimal History Window)      : {w_peak:5.1f} seconds")
            print(f"  tau_rise (Information Rise Rate) : {t_rise:5.1f} s")
            print(f"  tau_decay (Memory Decay Rate)    : {t_decay:5.1f} s")
            print(f"  Fit Quality (R^2)                : {r2:5.4f}")
        except Exception as e:
            print(f"Could not fit {c}: {e}")

    # Export tables
    out_dir = os.path.join(LM_DIR, "output", "presentation", "tables")
    os.makedirs(out_dir, exist_ok=True)
    sess_summary.to_csv(os.path.join(out_dir, "cohort_4group_nmf_occupancies.csv"), index=False)
    df_retention.to_csv(os.path.join(out_dir, "cohort_4group_nmf_k_retention.csv"), index=False)
    df_mem_trans.to_csv(os.path.join(out_dir, "cohort_4group_memory_curve_transfer.csv"), index=False)
    df_mem_within.to_csv(os.path.join(out_dir, "cohort_4group_memory_curve_within.csv"), index=False)
    print(f"\n[SUCCESS] Exported all tables to {out_dir}")

if __name__ == "__main__":
    main()
