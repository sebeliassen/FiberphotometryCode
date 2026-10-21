"""
Clean Confined Manifold Sweep: η² vs Decodability (Fresh Execution on Filtered Dataset)
Located at: syllable_generation/analysis/language_models/pipelines/14_run_clean_confined_eta2_sweep.py

Performs a fresh, clean hyperparameter sweep across well-conditioned manifolds (nn ∈ {25, 30, 35})
using the clean dataset cache (prep_amph_postural_gmm_k3_transition.pkl) with all pre-entry empty arena frames removed.
Calculates Pearson & Spearman correlations between cohort decodability and kinematic explained variance (η²).
"""

import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.data.dataset import SessionDataset
from studies.behavior.language_models.lib.models.strategies import StrategyBERTopic

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass

def evaluate_decodability(X_features: np.ndarray, y_labels: np.ndarray) -> float:
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_features[train_idx], y_labels[train_idx])
        accs.append(accuracy_score(y_labels[test_idx], clf.predict(X_features[test_idx])))
    return float(np.mean(accs))

def compute_clean_eta2(model, dataset, raw_sessions):
    all_win_inputs = dataset.get_window_inputs()
    session_window_meta = dataset.session_window_meta

    strat_ids, _ = model.transform(all_win_inputs)
    n_topics = model.n_topics

    if n_topics <= 1:
        return 0.0

    session_lookup = {s.session_id: s for s in raw_sessions}

    vel_by_strat = {t: [] for t in range(n_topics)}
    win_offset = 0

    for meta in session_window_meta:
        n_win = meta[0]
        abs_start_frame = meta[1]
        sid = meta[3]

        session = session_lookup.get(sid) or session_lookup.get(sid.replace("_pre", "").replace("_post", ""))
        if session is None:
            win_offset += n_win
            continue

        df = session.all_data_df
        vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))

        w_ids = strat_ids[win_offset : win_offset + n_win]
        win_offset += n_win

        for w_i, s_id in enumerate(w_ids):
            if 0 <= s_id < n_topics:
                # 128s window = 10240 frames @ 80Hz
                sf = abs_start_frame + int(w_i * 800)
                ef = min(len(df), sf + 10240)
                w_vel = vel[sf:ef]
                valid_vel = w_vel[~np.isnan(w_vel)]
                if len(valid_vel) > 0:
                    vel_by_strat[s_id].append(np.mean(valid_vel))

    # Aggregate mean velocity per strategy
    strat_means = []
    strat_counts = []
    all_means = []

    for t_id in range(n_topics):
        vals = vel_by_strat[t_id]
        if len(vals) > 0:
            strat_means.append(np.mean(vals))
            strat_counts.append(len(vals))
            all_means.extend(vals)

    if len(all_means) == 0 or len(strat_means) <= 1:
        return 0.0

    grand_mean = np.mean(all_means)
    ss_total = np.sum((np.array(all_means) - grand_mean) ** 2)
    ss_between = np.sum([n * (m - grand_mean) ** 2 for m, n in zip(strat_means, strat_counts)])

    return float(ss_between / (ss_total + 1e-12))

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    log_file = os.path.join(out_dir, "logs/clean_confined_eta2_sweep.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_msg(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {msg}\n")

    log_msg("=== FRESH CLEAN CONFINED MANIFOLD SWEEP (Filtering Empty Arena Frames) ===")
    send_notification("🚀 Starting clean confined manifold η² sweep")

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

    # Well-conditioned manifold grid
    nn_vals = [25, 30, 35]
    mcs_vals = [50, 60, 70, 80, 100, 120, 150, 200, 250, 300]

    results = []
    total_runs = len(nn_vals) * len(mcs_vals)
    run_i = 0

    for nn in nn_vals:
        for mcs in mcs_vals:
            run_i += 1
            log_msg(f"[{run_i}/{total_runs}] Running Clean Model: nn={nn}, mcs={mcs}...")

            try:
                model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=nn, min_cluster_size=mcs, random_state=42)
                model.fit(all_win_inputs)
                strat_ids, _ = model.transform(all_win_inputs)
                n_topics = model.n_topics

                if n_topics <= 1:
                    log_msg(f"  nn={nn} mcs={mcs} -> K={n_topics} (Single cluster/degenerate)")
                    continue

                props = np.zeros((len(labels), max(1, n_topics)))
                win_offset = 0
                for s_idx, n_win in enumerate(n_windows_per_session):
                    w_ids = strat_ids[win_offset : win_offset + n_win]
                    win_offset += n_win
                    counts = pd.Series(w_ids).value_counts(normalize=True)
                    for t_id, prop in counts.items():
                        if 0 <= t_id < n_topics:
                            props[s_idx, t_id] = prop

                acc = evaluate_decodability(props, np.array(labels))
                eta2_speed = compute_clean_eta2(model, dataset, raw_sessions)

                log_msg(f"  nn={nn} mcs={mcs} -> K={n_topics:2d} | Cohort Acc={acc*100:.2f}% | η²_speed={eta2_speed:.4f}")

                results.append({
                    "nn": nn,
                    "mcs": mcs,
                    "K": n_topics,
                    "accuracy": acc,
                    "eta2_speed": eta2_speed
                })
            except Exception as e:
                log_msg(f"  Error running nn={nn} mcs={mcs}: {e}")

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(out_dir, "clean_confined_eta2_sweep_results.csv")
    df_res.to_csv(csv_out, index=False)

    # Statistical Correlation Report
    log_msg("\n" + "=" * 70)
    log_msg("CLEAN CONFINED MANIFOLD SWEEP STATISTICAL REPORT")
    log_msg("=" * 70)

    if len(df_res) > 3:
        r_val, p_val = pearsonr(df_res["accuracy"], df_res["eta2_speed"])
        rho_val, _ = spearmanr(df_res["accuracy"], df_res["eta2_speed"])
        log_msg(f"  Overall Pearson r(Acc, η²_speed):  {r_val:+.3f} (p = {p_val:.6f})")
        log_msg(f"  Overall Spearman rho:              {rho_val:+.3f}")

        for nn in nn_vals:
            sub = df_res[df_res["nn"] == nn]
            if len(sub) > 3:
                r_sub, p_sub = pearsonr(sub["accuracy"], sub["eta2_speed"])
                log_msg(f"  nn={nn} slice (N={len(sub)}): r = {r_sub:+.3f} (p = {p_sub:.4f})")

    send_notification(f"✅ Clean manifold η² sweep complete! r = {r_val:+.3f}")
    log_msg("=" * 70)

if __name__ == "__main__":
    main()
