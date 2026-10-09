"""
Confined Hyperparameter Sweep: η² vs Decodability within well-conditioned manifolds only.

Only sweeps n_neighbors ∈ {25, 30, 35} (stable manifold regime) with dense
min_cluster_size sampling to get N≥20 data points for a proper correlation test.
"""

import os
import sys
import time
import pickle
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.models.strategies import StrategyBERTopic
from analysis.language_models.lib.evaluation.kinematic_encoding import (
    compute_eta_squared, evaluate_session_kinematic_encoding
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass

def evaluate_decodability(X_features, y_labels):
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_features[train_idx], y_labels[train_idx])
        accs.append(accuracy_score(y_labels[test_idx], clf.predict(X_features[test_idx])))
    return np.mean(accs)

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "confined_sweep_eta2.log")
    
    def log_msg(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    
    log_msg("=== CONFINED MANIFOLD SWEEP: eta2 vs Decodability (well-conditioned only) ===")
    send_notification("🚀 Starting confined manifold η² sweep")
    
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_windows_per_session = dataset.n_windows_per_session
    
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)
    
    session_id_list = [meta_tuple[6] for meta_tuple in dataset.session_window_meta]
    session_dfs = {}
    for s in raw_sessions:
        exact_id = s.session_id
        for s_id in session_id_list:
            if s_id.startswith(exact_id) or exact_id.startswith(s_id):
                session_dfs[s_id] = s.all_data_df
                break
    
    # Confined grid: only well-conditioned manifold topologies
    nn_vals = [25, 30, 35]
    mcs_vals = [100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
    
    csv_path = os.path.join(out_dir, "confined_sweep_eta2_results.csv")
    existing_df = None
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            log_msg(f"Loaded existing checkpoint with {len(existing_df)} completed runs from {csv_path}")
        except Exception:
            existing_df = None

    # Also load dense audit csv for fallback caching if available
    dense_audit_path = os.path.join(out_dir, "dense_sweep_audit.csv")
    dense_audit_df = None
    if os.path.exists(dense_audit_path):
        try:
            dense_audit_df = pd.read_csv(dense_audit_path)
            log_msg(f"Loaded dense_sweep_audit.csv with {len(dense_audit_df)} historical entries")
        except Exception:
            dense_audit_df = None

    results = []
    if existing_df is not None:
        results = existing_df.to_dict('records')

    completed_keys = {(int(r["n_neighbors"]), int(r["min_cluster_size"])) for r in results}
    n_total = len(nn_vals) * len(mcs_vals)
    run_i = 0
    
    for nn in nn_vals:
        for mcs in mcs_vals:
            run_i += 1
            if (nn, mcs) in completed_keys:
                log_msg(f"  [{run_i}/{n_total}] nn={nn} mcs={mcs:3d} -> SKIPPED (already in checkpoint)")
                continue

            # Check if present in dense_sweep_audit.csv
            if dense_audit_df is not None:
                match = dense_audit_df[(dense_audit_df["nn"] == nn) & (dense_audit_df["mcs"] == mcs)]
                if not match.empty:
                    row = match.iloc[0]
                    rec = {
                        "n_neighbors": int(nn), "min_cluster_size": int(mcs),
                        "active_K": int(row["K"]),
                        "decodability_acc_pct": float(row["acc"]),
                        "eta2_speed": float(row["eta2_speed"]),
                        "eta2_dist_center": 0.0,
                        "eta2_angular_vel": 0.0,
                        "eta2_cadence": 0.0,
                    }
                    results.append(rec)
                    completed_keys.add((nn, mcs))
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    log_msg(f"  [{run_i}/{n_total}] nn={nn} mcs={mcs:3d} -> REUSED from dense_sweep_audit (K={int(row['K'])}, Acc={float(row['acc']):5.2f}%)")
                    continue

            try:
                model = StrategyBERTopic(
                    vectorizer_type="transition", n_neighbors=nn,
                    min_cluster_size=mcs, random_state=42
                )
                model.fit(all_win_inputs)
                strat_ids, _ = model.transform(all_win_inputs)
                n_topics = model.n_topics
                
                # Session proportions for decodability
                props = np.zeros((len(labels), max(1, n_topics)))
                frame_strat_dict = {}
                win_offset = 0
                for s_idx, s_id in enumerate(session_id_list):
                    n_win = n_windows_per_session[s_idx]
                    w_ids = strat_ids[win_offset: win_offset + n_win]
                    win_offset += n_win
                    counts = pd.Series(w_ids).value_counts(normalize=True)
                    for t_id, prop in counts.items():
                        if 0 <= t_id < n_topics:
                            props[s_idx, t_id] = prop
                    frame_strat_dict[s_id] = np.repeat(w_ids, 800)
                
                acc = evaluate_decodability(props, np.array(labels))
                
                # Full kinematic encoding
                kin = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)
                
                rec = {
                    "n_neighbors": int(nn), "min_cluster_size": int(mcs),
                    "active_K": int(n_topics),
                    "decodability_acc_pct": float(acc * 100),
                    "eta2_speed": float(kin["eta2_velocity"]),
                    "eta2_dist_center": float(kin["eta2_dist_center"]),
                    "eta2_angular_vel": float(kin["eta2_angular_vel"]),
                    "eta2_cadence": float(kin["eta2_cadence"]),
                }
                results.append(rec)
                completed_keys.add((nn, mcs))
                pd.DataFrame(results).to_csv(csv_path, index=False)
                
                log_msg(f"  [{run_i}/{n_total}] nn={nn} mcs={mcs:3d} -> K={n_topics:2d} Acc={acc*100:5.2f}% eta2_speed={kin['eta2_velocity']:.4f}")
                
            except Exception as e:
                log_msg(f"  [{run_i}/{n_total}] nn={nn} mcs={mcs:3d} -> FAILED: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Correlations
    acc_arr = df["decodability_acc_pct"].values
    speed_arr = df["eta2_speed"].values
    k_arr = df["active_K"].values.astype(float)
    
    r_raw, p_raw = pearsonr(acc_arr, speed_arr)
    rho_raw, _ = spearmanr(acc_arr, speed_arr)
    
    r_ak, _ = pearsonr(acc_arr, k_arr)
    r_ek, _ = pearsonr(speed_arr, k_arr)
    partial_r = (r_raw - r_ak * r_ek) / np.sqrt((1 - r_ak**2) * (1 - r_ek**2))
    
    # Per-nn correlations
    log_msg(f"\n{'='*70}")
    log_msg(f"CONFINED SWEEP RESULTS (N={len(df)}, nn ∈ {{25, 30, 35}} only)")
    log_msg(f"{'='*70}")
    log_msg(f"  Overall r(Acc, eta2_speed):            {r_raw:+.3f} (p={p_raw:.6f})")
    log_msg(f"  Overall Spearman rho:                  {rho_raw:+.3f}")
    log_msg(f"  Partial r(Acc, eta2_speed | K):         {partial_r:+.3f}")
    log_msg(f"  r(K, Acc):                             {r_ak:+.3f}")
    log_msg(f"  r(K, eta2_speed):                      {r_ek:+.3f}")
    
    for nn in nn_vals:
        sub = df[df["n_neighbors"] == nn]
        if len(sub) >= 3:
            r_nn, p_nn = pearsonr(sub["decodability_acc_pct"].values, sub["eta2_speed"].values)
            log_msg(f"  nn={nn} slice (N={len(sub)}): r={r_nn:+.3f} (p={p_nn:.4f})")
    
    log_msg(f"{'='*70}")
    log_msg(f"  CSV saved to: {csv_path}")
    
    summary = (
        f"✅ Confined sweep done (N={len(df)})\n"
        f"r(Acc, eta2_speed) = {r_raw:+.3f} (p={p_raw:.4f})\n"
        f"Partial r | K = {partial_r:+.3f}"
    )
    send_notification(summary)
    log_msg("=== CONFINED MANIFOLD SWEEP COMPLETE ===")

if __name__ == "__main__":
    main()
