#!/usr/bin/env python3
"""
===============================================================================
4-Cohort Comparison across Fig 1 and Fig 2 Presentation Representations
(WT Saline, WT Amph, KI Saline, KI Amph — Post-Injection Analysis)
===============================================================================
Evaluates:
1. Fig 1 / 1d: NMF Macro-Components (K=4) — Occupancy, Entropy, Trajectories & Transitions
2. Fig 1b: NMF Dimensional Retention Curve across K in [1..30]
3. Fig 2: Timescale Memory Curves (W in [2..360s]) — Trained on WT Saline vs Within-Cohort
===============================================================================
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
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

def js_divergence(p, q):
    p = np.clip(p, 1e-12, None)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = np.clip(q, 1e-12, None)
    q = q / np.sum(q, axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_p_m = np.sum(p * np.log2(p / m), axis=-1)
    kl_q_m = np.sum(q * np.log2(q / m), axis=-1)
    return 0.5 * (kl_p_m + kl_q_m)

def biexp_func(w, P0, A, tau_rise, tau_decay):
    return P0 + A * (np.exp(-w / tau_decay) - np.exp(-w / tau_rise))

def main():
    print("=" * 80)
    print("LOADING SESSIONS & BOUT CACHE...")
    print("=" * 80)

    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    session_bouts = cache["session_bouts"] if isinstance(cache, dict) and "session_bouts" in cache else cache

    cohort_names = ["WT_Saline", "WT_Amphetamine", "KI_Saline", "KI_Amphetamine"]
    session_meta = []

    for i, s in enumerate(raw_sessions):
        meta = getattr(s, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        cohort = f"{gt}_{drug}"
        inj_f = meta.get("injection_frame", 0)
        entry_f = meta.get("entry_frame", 0)
        sid = getattr(s, "session_id", f"sess_{i}")
        max_f = session_bouts[i][-1]["end_frame"] if len(session_bouts[i]) > 0 else 0
        session_meta.append({
            "idx": i, "sid": sid, "genotype": gt, "drug": drug, "cohort": cohort,
            "entry_f": entry_f, "inj_f": inj_f, "max_f": max_f
        })
    df_meta = pd.DataFrame(session_meta)

    # -------------------------------------------------------------------------
    # 1. Precompute multi-window counts and future target arrays
    # -------------------------------------------------------------------------
    W_VALUES = [2.0, 5.0, 10.0, 20.0, 40.0, 68.3, 90.0, 120.0, 180.0, 240.0, 360.0]

    print("\nExtracting sliding windows across all sessions...")
    session_windows = []

    for s_idx, bouts in enumerate(session_bouts):
        if len(bouts) < 100: continue
        meta = df_meta.iloc[s_idx]
        inj_f = meta["inj_f"]
        entry_f = meta["entry_f"]
        max_frame = bouts[-1]["end_frame"]
        N = max_frame + 1

        frame_sylls = np.zeros(N, dtype=np.int32)
        for b in bouts:
            sf, ef = b["start_frame"], b["end_frame"]
            s_id = min(b["syllable"], NUM_SYLLS - 1)
            frame_sylls[sf : ef + 1] = s_id

        oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
        diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
        onsets = np.zeros(N, dtype=bool)
        onsets[0] = True; onsets[diffs] = True
        oh_bout[onsets, frame_sylls[onsets]] = 1.0
        cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

        max_w_frames = int(max(W_VALUES) * FPS)

        # We extract windows: pre-injection and post-injection
        for t in range(max_w_frames, N - H_FRAMES, STRIDE_FRAMES):
            # Target future distribution over H_FRAMES
            y_count = cum_bout[t + H_FRAMES] - cum_bout[t]
            tot_y = np.sum(y_count)
            if tot_y < 1e-4: continue
            y_dist = y_count / tot_y

            # Extract past counts for each W
            past_dict = {}
            for w_sec in W_VALUES:
                w_fr = int(w_sec * FPS)
                past_dict[w_sec] = cum_bout[t] - cum_bout[t - w_fr]

            is_post = (t >= inj_f + int(68.3 * FPS))
            is_pre = (t < inj_f and t >= entry_f + max_w_frames)
            t_min_from_inj = (t - inj_f) / FPS / 60.0

            session_windows.append({
                "session_idx": s_idx,
                "sid": meta["sid"],
                "cohort": meta["cohort"],
                "genotype": meta["genotype"],
                "drug": meta["drug"],
                "t_frame": t,
                "t_min_from_inj": t_min_from_inj,
                "is_post": is_post,
                "is_pre": is_pre,
                "y_dist": y_dist,
                "past_dict": past_dict
            })

    df_win = pd.DataFrame(session_windows)
    print(f"Total evaluation windows extracted: {len(df_win)}")
    print("Post-injection windows per cohort:")
    print(df_win[df_win["is_post"]]["cohort"].value_counts())

    # -------------------------------------------------------------------------
    # PART 1: FIG 1 / 1D — NMF COMPONENT INTERPRETATION & GROUP TRAJECTORIES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: FIG 1 & FIG 1D — NMF MACRO-COMPONENTS (K=4) ACROSS 4 COHORTS")
    print("=" * 80)

    # Fit NMF K=4 on 68.3s past counts across reference WT Saline or All
    # Let's fit on WT_Saline (reference baseline) and all windows to get canonical components
    X_68_all = np.array([w["past_dict"][68.3] for w in session_windows], dtype=np.float32)
    K = 4
    nmf = NMF(n_components=K, init='nndsvda', random_state=42, max_iter=800)
    nmf.fit(X_68_all)

    # Top syllable loadings per component
    print("\n--- Canonical NMF Components (K=4) Syllable Loadings ---")
    for k in range(K):
        top_sylls = np.argsort(nmf.components_[k])[::-1][:6]
        top_weights = nmf.components_[k][top_sylls]
        syll_str = ", ".join([f"Syllable {s} ({w:.2f})" for s, w in zip(top_sylls, top_weights)])
        print(f"  Component {k+1}: {syll_str}")

    # Compute Z_t for all windows
    Z_all = nmf.transform(X_68_all)
    Z_probs = Z_all / np.clip(np.sum(Z_all, axis=1, keepdims=True), 1e-8, None)

    for k in range(K):
        df_win[f"z_{k+1}"] = Z_probs[:, k]

    # Behavioral Entropy per window: H(z) = -sum(z * log2(z))
    z_clipped = np.clip(Z_probs, 1e-12, None)
    df_win["z_entropy"] = -np.sum(z_clipped * np.log2(z_clipped), axis=1)

    # Filter post-injection windows
    df_post = df_win[df_win["is_post"]].copy()

    # Session-level aggregations
    sess_summary = df_post.groupby(["cohort", "genotype", "drug", "session_idx"])[
        ["z_1", "z_2", "z_3", "z_4", "z_entropy"]
    ].mean().reset_index()

    print("\n--- Post-Injection NMF Component Occupancy (Mean ± SEM across Sessions) ---")
    cohort_stats = {}
    for cohort in cohort_names:
        sub = sess_summary[sess_summary["cohort"] == cohort]
        n_s = len(sub)
        print(f"\nCohort: {cohort} (N = {n_s} animals):")
        for k in range(K):
            m, s = sub[f"z_{k+1}"].mean(), sub[f"z_{k+1}"].sem()
            print(f"  Component {k+1}: {m*100:5.2f}% ± {s*100:4.2f}%")
        m_ent, s_ent = sub["z_entropy"].mean(), sub["z_entropy"].sem()
        print(f"  State Entropy: {m_ent:5.3f} ± {s_ent:4.3f} bits")
        cohort_stats[cohort] = sub

    # Statistical tests: WT Sal vs WT Amph, KI Sal vs KI Amph, WT Sal vs KI Sal, WT Amph vs KI Amph
    print("\n--- Statistical Comparisons (Mann-Whitney U / Welch t-test) ---")
    comps = [
        ("WT_Saline", "WT_Amphetamine", "Effect of Amph in WT"),
        ("KI_Saline", "KI_Amphetamine", "Effect of Amph in KI"),
        ("WT_Saline", "KI_Saline", "Genotype Baseline Difference (Saline)"),
        ("WT_Amphetamine", "KI_Amphetamine", "Genotype Drug Response Difference (Amph)")
    ]
    for c1, c2, title in comps:
        s1 = cohort_stats[c1]
        s2 = cohort_stats[c2]
        print(f"\n[Comparison: {title}] ({c1} vs {c2}):")
        for k in range(K):
            stat, p = ttest_ind(s1[f"z_{k+1}"], s2[f"z_{k+1}"], equal_var=False)
            diff = (s2[f"z_{k+1}"].mean() - s1[f"z_{k+1}"].mean()) * 100
            print(f"  Component {k+1}: Delta = {diff:+5.2f} pp, t = {stat:5.2f}, p = {p:.4f} {'*' if p < 0.05 else ''}")
        stat, p = ttest_ind(s1["z_entropy"], s2["z_entropy"], equal_var=False)
        diff_ent = s2["z_entropy"].mean() - s1["z_entropy"].mean()
        print(f"  State Entropy : Delta = {diff_ent:+5.3f} bits, t = {stat:5.2f}, p = {p:.4f} {'*' if p < 0.05 else ''}")

    # -------------------------------------------------------------------------
    # PART 2: FIG 1B — NMF PREDICTIVE UTILITY CURVE ACROSS K IN [1..30]
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: FIG 1B — NMF PREDICTIVE UTILITY VS COMPONENT COUNT K ACROSS COHORTS")
    print("=" * 80)

    K_LIST = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30]

    # We train Ridge model on WT_Saline post-injection to predict future Y
    wt_sal_mask = (df_win["cohort"] == "WT_Saline") & df_win["is_post"]
    wtsal_sub = df_win[wt_sal_mask]
    X_68_train = np.array([w[68.3] for w in wtsal_sub["past_dict"]], dtype=np.float32)
    Y_train = np.array(wtsal_sub["y_dist"].tolist(), dtype=np.float32)

    # Global null on WT Saline
    null_p0_wtsal = np.mean(Y_train, axis=0, keepdims=True)

    nmf_k_results = []

    for k_val in K_LIST:
        nmf_k = NMF(n_components=k_val, init='nndsvda', random_state=42, max_iter=600)
        Z_train_k = nmf_k.fit_transform(X_68_train)

        reg = Ridge(alpha=10.0)
        reg.fit(Z_train_k, Y_train)

        row = {"K": k_val}
        for cohort in cohort_names:
            c_sub = df_win[(df_win["cohort"] == cohort) & df_win["is_post"]]
            X_68_test = np.array([w[68.3] for w in c_sub["past_dict"]], dtype=np.float32)
            Y_test = np.array(c_sub["y_dist"].tolist(), dtype=np.float32)

            # Cohort-specific marginal baseline
            p0_c = np.mean(Y_test, axis=0, keepdims=True)
            d_null = np.mean(js_divergence(Y_test, p0_c))

            Z_test_k = nmf_k.transform(X_68_test)
            Y_pred = np.clip(reg.predict(Z_test_k), 1e-12, None)
            Y_pred = Y_pred / np.sum(Y_pred, axis=-1, keepdims=True)
            d_pred = np.mean(js_divergence(Y_test, Y_pred))

            gain = (d_null - d_pred) / d_null * 100.0
            row[f"gain_{cohort}"] = gain

        nmf_k_results.append(row)

    df_nmf_k = pd.DataFrame(nmf_k_results)
    print("\nPredictive Gain (% Delta D_JS) vs NMF Components (Trained on WT Saline):")
    print(df_nmf_k.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 3: FIG 2 — HISTORY TIMESCALE MEMORY CURVES (W IN [2..360s])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: FIG 2 — HISTORY TIMESCALE MEMORY CURVES ACROSS 4 COHORTS")
    print("=" * 80)

    # Evaluate across W_VALUES
    # Model 1: Trained on WT Saline Post-Injection -> Evaluated on all 4 cohorts
    # Model 2: Within-Cohort 5-Fold Cross-Validation on Post-Injection

    print("\n--- Case A: Trained on WT Saline (Reference Model Transfer) ---")
    memory_results_transfer = []

    for w_sec in W_VALUES:
        X_w_train = np.array([w[w_sec] for w in wtsal_sub["past_dict"]], dtype=np.float32)
        reg = Ridge(alpha=10.0)
        reg.fit(X_w_train, Y_train)

        row = {"W_sec": w_sec}
        for cohort in cohort_names:
            c_sub = df_win[(df_win["cohort"] == cohort) & df_win["is_post"]]
            X_w_test = np.array([w[w_sec] for w in c_sub["past_dict"]], dtype=np.float32)
            Y_test = np.array(c_sub["y_dist"].tolist(), dtype=np.float32)

            p0_c = np.mean(Y_test, axis=0, keepdims=True)
            d_null = np.mean(js_divergence(Y_test, p0_c))

            Y_pred = np.clip(reg.predict(X_w_test), 1e-12, None)
            Y_pred = Y_pred / np.sum(Y_pred, axis=-1, keepdims=True)
            d_pred = np.mean(js_divergence(Y_test, Y_pred))

            gain = (d_null - d_pred) / d_null * 100.0
            row[f"gain_{cohort}"] = gain

        memory_results_transfer.append(row)

    df_mem_transfer = pd.DataFrame(memory_results_transfer)
    print(df_mem_transfer.to_string(index=False))

    print("\n--- Case B: Within-Cohort Cross-Validation (Optimal Within-Group Fit) ---")
    memory_results_within = []

    for w_sec in W_VALUES:
        row = {"W_sec": w_sec}
        for cohort in cohort_names:
            c_sub = df_win[(df_win["cohort"] == cohort) & df_win["is_post"]]
            sess_ids = c_sub["session_idx"].unique()

            # Leave-one-session-out / cross-session CV
            gains = []
            for test_s in sess_ids:
                train_sub = c_sub[c_sub["session_idx"] != test_s]
                test_sub = c_sub[c_sub["session_idx"] == test_s]
                if len(train_sub) == 0 or len(test_sub) == 0: continue

                X_tr = np.array([w[w_sec] for w in train_sub["past_dict"]], dtype=np.float32)
                Y_tr = np.array(train_sub["y_dist"].tolist(), dtype=np.float32)
                X_te = np.array([w[w_sec] for w in test_sub["past_dict"]], dtype=np.float32)
                Y_te = np.array(test_sub["y_dist"].tolist(), dtype=np.float32)

                p0_tr = np.mean(Y_tr, axis=0, keepdims=True)  # train marginal
                d_null = np.mean(js_divergence(Y_te, p0_tr))

                reg = Ridge(alpha=10.0)
                reg.fit(X_tr, Y_tr)
                Y_pred = np.clip(reg.predict(X_te), 1e-12, None)
                Y_pred = Y_pred / np.sum(Y_pred, axis=-1, keepdims=True)
                d_pred = np.mean(js_divergence(Y_te, Y_pred))

                gain = (d_null - d_pred) / d_null * 100.0
                gains.append(gain)

            row[f"gain_{cohort}"] = np.mean(gains)
        memory_results_within.append(row)

    df_mem_within = pd.DataFrame(memory_results_within)
    print(df_mem_within.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 4: FIT BIEXPONENTIAL CURVES PER COHORT
    # -------------------------------------------------------------------------
    print("\n--- Biexponential Fit Parameters per Cohort (Within-Group Curve) ---")
    w_arr = df_mem_within["W_sec"].values

    for cohort in cohort_names:
        g_arr = df_mem_within[f"gain_{cohort}"].values
        try:
            popt, _ = curve_fit(biexp_func, w_arr, g_arr, p0=[10.0, 15.0, 15.0, 600.0], maxfev=20000)
            p0_fit, a_fit, t_rise, t_decay = popt

            # Find peak
            w_dense = np.linspace(0.1, 400.0, 2000)
            g_dense = biexp_func(w_dense, *popt)
            w_peak = w_dense[np.argmax(g_dense)]
            g_peak = np.max(g_dense)

            print(f"\nCohort: {cohort}")
            print(f"  P0 (Instantaneous Gain)  : {p0_fit:5.2f}%")
            print(f"  P_max (Peak Gain)        : {g_peak:5.2f}%")
            print(f"  W* (Optimal History Window): {w_peak:5.1f} seconds")
            print(f"  tau_rise (Rise timescale): {t_rise:5.1f} s")
            print(f"  tau_decay (Decay timescale): {t_decay:5.1f} s")
        except Exception as e:
            print(f"  Could not fit biexponential for {cohort}: {e}")

    # Save all output tables
    out_table_dir = os.path.join(LM_DIR, "output", "presentation", "tables")
    os.makedirs(out_table_dir, exist_ok=True)
    sess_summary.to_csv(os.path.join(out_table_dir, "cohort_4group_nmf_occupancies.csv"), index=False)
    df_nmf_k.to_csv(os.path.join(out_table_dir, "cohort_4group_nmf_k_retention.csv"), index=False)
    df_mem_transfer.to_csv(os.path.join(out_table_dir, "cohort_4group_memory_curve_transfer.csv"), index=False)
    df_mem_within.to_csv(os.path.join(out_table_dir, "cohort_4group_memory_curve_within.csv"), index=False)
    print(f"\n[SUCCESS] Saved all 4-cohort summary tables to {out_table_dir}")

if __name__ == "__main__":
    main()
