"""
GPU-Accelerated 1000-Trial Bayesian Hyperparameter Sweep (Optuna TPE)
Located at: syllable_generation/analysis/language_models/pipelines/16_run_bayesian_hyperparameter_sweep.py

1. Fits GPU StrategyBERTopic (cuml UMAP + cuml HDBSCAN on RTX 4090) on 14,870 clean active-mouse windows.
2. Slices temporal strategy proportions into Pre-Injection Baseline (t < injection_frame) and Post-Injection Epoch (t >= injection_frame).
3. Evaluates 5-fold Stratified Cross-Validated Random Forest classification for:
   - Post-Injection 4-Way Cohort Accuracy (WT_Saline, WT_AMPH, KI_Saline, KI_AMPH)
   - Post-Injection Drug Accuracy (Saline vs AMPH)
   - Post-Injection Genotype Accuracy (WT vs DAT-KI)
   - Baseline Genotype Accuracy (Pre-injection WT vs DAT-KI)
   - Kinematic Explained Variance (η²_speed)
4. Target Objective Function (Weighted highest on 4-Way Cohort separation):
   Target Score = 2.0 * acc_4way_post + 1.0 * acc_drug_post + 1.0 * acc_gt_post
5. Sweeps across 4 core manifold parameters:
   - n_neighbors: [5, 150]
   - min_cluster_size: [5, 500]
   - min_samples: [1, 50]
   - n_components: [2, 30]
6. Sends mobile ntfy notifications at start, every 100 trials, and completion.
"""

import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Detect GPU acceleration via RAPIDS cuML
HAS_GPU = False
try:
    import cuml
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
    from cuml.manifold import UMAP as GPU_UMAP
    HAS_GPU = True
except ImportError:
    from sklearn.cluster import HDBSCAN as CPU_HDBSCAN

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg: str):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except Exception:
        pass

def eval_classifier(X_features: np.ndarray, y_labels: np.ndarray) -> float:
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    if len(unique_classes) <= 1 or np.min(counts) < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=min(5, np.min(counts)), shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_features[train_idx], y_labels[train_idx])
        accs.append(accuracy_score(y_labels[test_idx], clf.predict(X_features[test_idx])))
    return float(np.mean(accs))

def precompute_window_velocities(session_window_meta, raw_sessions) -> np.ndarray:
    session_lookup = {s.session_id: s for s in raw_sessions}
    win_vels = []
    for meta in session_window_meta:
        n_win, abs_start_frame, _, sid = meta[0], meta[1], meta[2], meta[3]
        session = session_lookup.get(sid)
        if session is None:
            win_vels.extend([0.0] * n_win)
            continue
        df = session.all_data_df
        vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))
        for w_i in range(n_win):
            sf = abs_start_frame + int(w_i * 800)
            ef = min(len(df), sf + 10240)
            w_vel = vel[sf:ef]
            valid_vel = w_vel[~np.isnan(w_vel)]
            win_vels.append(float(np.mean(valid_vel)) if len(valid_vel) > 0 else 0.0)
    return np.array(win_vels, dtype=np.float32)

def compute_fast_eta2(strat_ids: np.ndarray, win_vels: np.ndarray, n_topics: int) -> float:
    if n_topics <= 1:
        return 0.0
    valid_mask = (strat_ids >= 0) & (strat_ids < n_topics)
    if not np.any(valid_mask):
        return 0.0
    valid_ids = strat_ids[valid_mask]
    valid_vels = win_vels[valid_mask]

    grand_mean = np.mean(valid_vels)
    ss_total = np.sum((valid_vels - grand_mean) ** 2)
    if ss_total < 1e-9:
        return 0.0

    ss_between = 0.0
    for t_id in range(n_topics):
        t_mask = (valid_ids == t_id)
        n_t = np.sum(t_mask)
        if n_t > 0:
            m_t = np.mean(valid_vels[t_mask])
            ss_between += n_t * ((m_t - grand_mean) ** 2)

    return float(ss_between / (ss_total + 1e-12))

def fit_gpu_bertopic(X_vectors: np.ndarray, n_neighbors: int, min_cluster_size: int, min_samples: int, n_components: int, random_state: int = 42):
    # 1. TruncatedSVD feature pre-reduction if high-dimensional
    if X_vectors.shape[1] > 100:
        svd_dim = min(50, X_vectors.shape[1] - 1)
        svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
        X_proc = svd.fit_transform(X_vectors)
    else:
        X_proc = X_vectors

    # 2. UMAP Projection
    if HAS_GPU:
        u_model = GPU_UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='euclidean', random_state=random_state)
        umap_emb = u_model.fit_transform(X_proc)
        if hasattr(umap_emb, 'get'):
            umap_emb = umap_emb.get()
    else:
        import umap as cpu_umap
        u_model = cpu_umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='euclidean', random_state=random_state)
        umap_emb = u_model.fit_transform(X_proc)

    # 3. HDBSCAN Clustering
    if HAS_GPU:
        h_model = GPU_HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        raw_labels = h_model.fit_predict(umap_emb)
        if hasattr(raw_labels, 'get'):
            raw_labels = raw_labels.get()
    else:
        h_model = CPU_HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
        raw_labels = h_model.fit_predict(umap_emb)

    raw_labels = np.asarray(raw_labels, dtype=int)

    # 4. Resolve Outliers (-1) to nearest UMAP centroid
    unique_clusters = [c for c in set(raw_labels) if c != -1]
    if len(unique_clusters) == 0:
        return np.zeros(len(X_vectors), dtype=int), 0

    centroids = np.array([umap_emb[raw_labels == c].mean(axis=0) for c in unique_clusters])
    resolved_ids = raw_labels.copy()
    outlier_mask = (raw_labels == -1)

    if np.any(outlier_mask):
        outlier_coords = umap_emb[outlier_mask]
        dists = np.linalg.norm(outlier_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        closest_indices = np.argmin(dists, axis=1)
        resolved_ids[outlier_mask] = np.array(unique_clusters)[closest_indices]

    # Remap cluster IDs to continuous 0..K-1
    label_map = {old_c: new_idx for new_idx, old_c in enumerate(sorted(set(resolved_ids)))}
    mapped_ids = np.array([label_map[c] for c in resolved_ids], dtype=int)
    K = len(label_map)
    return mapped_ids, K

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    log_file = os.path.join(out_dir, "logs/bayesian_sweep_1000.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_msg(msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {msg}\n")
            f.flush()

    log_msg("=== 1000-TRIAL BAYESIAN SWEEP (GPU-Accelerated, Post-Injection Sliced, 4-Way Weighted) ===")
    log_msg(f"GPU Acceleration Active: {HAS_GPU} (cuML RAPIDS)")
    send_notification(f"🚀 Starting 1000-Trial GPU Bayesian Sweep (RAPIDS={HAS_GPU})")

    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = np.array(dataset.get_window_inputs(), dtype=np.float32)
    session_window_meta = dataset.session_window_meta

    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)
    session_lookup = {s.session_id: s for s in raw_sessions}

    # Pre-build session window slicing indices and pre/post injection masks
    session_slices = []
    labels_4way = []
    labels_gt = []
    labels_drug = []

    win_offset = 0
    for meta in session_window_meta:
        n_win, abs_start_frame, abs_end_frame, sid = meta[0], meta[1], meta[2], meta[3]
        cohort_4way = meta[5]
        gt, drug = cohort_4way.split("_")[0], cohort_4way.split("_")[1]

        labels_4way.append(cohort_4way)
        labels_gt.append(gt)
        labels_drug.append(drug)

        s_obj = session_lookup.get(sid)
        inj_frame = getattr(s_obj, "metadata", {}).get("injection_frame", 0)

        win_range = range(win_offset, win_offset + n_win)
        win_offset += n_win

        pre_indices = []
        post_indices = []
        for local_i, global_i in enumerate(win_range):
            sf = abs_start_frame + local_i * 800
            ef = sf + 10240
            cf = (sf + ef) // 2
            if cf >= inj_frame:
                post_indices.append(global_i)
            else:
                pre_indices.append(global_i)

        session_slices.append({
            "sid": sid,
            "all_indices": np.array(list(win_range)),
            "pre_indices": np.array(pre_indices),
            "post_indices": np.array(post_indices),
        })

    labels_4way = np.array(labels_4way)
    labels_gt = np.array(labels_gt)
    labels_drug = np.array(labels_drug)

    log_msg(f"Loaded dataset: {len(labels_4way)} sessions, {len(all_win_inputs)} total clean windows.")
    log_msg("Pre-computing window velocities...")
    win_vels = precompute_window_velocities(session_window_meta, raw_sessions)
    log_msg("Pre-computation complete.")

    results = []

    def objective(trial: optuna.Trial) -> float:
        # Sample across 4 core manifold parameters (Expanded Search Ranges)
        nn = trial.suggest_int("n_neighbors", 5, 150)
        mcs = trial.suggest_int("min_cluster_size", 5, 500)
        ms = trial.suggest_int("min_samples", 1, 50)
        nc = trial.suggest_int("n_components", 2, 30)

        t0 = time.time()
        # Fit StrategyBERTopic on ALL 14,870 clean windows
        strat_ids, K = fit_gpu_bertopic(all_win_inputs, n_neighbors=nn, min_cluster_size=mcs, min_samples=ms, n_components=nc, random_state=42)

        if K <= 1:
            return -1.0

        n_sessions = len(session_slices)
        props_full = np.zeros((n_sessions, K))
        props_post = np.zeros((n_sessions, K))
        props_pre = np.zeros((n_sessions, K))

        for s_idx, s_info in enumerate(session_slices):
            # Full session
            w_full = strat_ids[s_info["all_indices"]]
            if len(w_full) > 0:
                counts = pd.Series(w_full).value_counts(normalize=True)
                for t_id, prop in counts.items():
                    props_full[s_idx, t_id] = prop

            # Post-injection epoch
            w_post = strat_ids[s_info["post_indices"]]
            if len(w_post) > 0:
                counts = pd.Series(w_post).value_counts(normalize=True)
                for t_id, prop in counts.items():
                    props_post[s_idx, t_id] = prop

            # Pre-injection baseline
            w_pre = strat_ids[s_info["pre_indices"]]
            if len(w_pre) > 0:
                counts = pd.Series(w_pre).value_counts(normalize=True)
                for t_id, prop in counts.items():
                    props_pre[s_idx, t_id] = prop

        # Post-Injection Epoch Group Separation
        acc_4way_post = eval_classifier(props_post, labels_4way)
        acc_drug_post = eval_classifier(props_post, labels_drug)
        acc_gt_post = eval_classifier(props_post, labels_gt)

        # Baseline Epoch Genotype Accuracy
        acc_gt_pre = eval_classifier(props_pre, labels_gt)

        # Kinematic Explained Variance
        eta2_speed = compute_fast_eta2(strat_ids, win_vels, K)
        elapsed = time.time() - t0

        # Weighted Target Score: 4-Way Cohort Weighted Highest (2.0x weight)
        target_score = 2.0 * acc_4way_post + 1.0 * acc_drug_post + 1.0 * acc_gt_post

        trial_data = {
            "trial": trial.number + 1,
            "nn": nn,
            "mcs": mcs,
            "ms": ms,
            "nc": nc,
            "K": K,
            "acc_4way_post": acc_4way_post,
            "acc_drug_post": acc_drug_post,
            "acc_gt_post": acc_gt_post,
            "acc_gt_pre": acc_gt_pre,
            "eta2_speed": eta2_speed,
            "target_score": target_score,
            "elapsed_sec": elapsed
        }
        results.append(trial_data)

        # Periodically flush results CSV
        csv_out = os.path.join(out_dir, "bayesian_sweep_1000_results.csv")
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_out, index=False)

        trial_num = trial.number + 1
        log_msg(f"  [{trial_num:4d}/1000] nn={nn:3d} mcs={mcs:3d} ms={ms:2d} nc={nc:2d} -> K={K:3d} | 4-Way={acc_4way_post*100:5.1f}% | Drug={acc_drug_post*100:5.1f}% | GT={acc_gt_post*100:5.1f}% | Pre-GT={acc_gt_pre*100:5.1f}% | Target={target_score:.3f} ({elapsed:.1f}s)")

        # Send ntfy notification every 100 trials
        if trial_num % 100 == 0:
            best_idx = np.argmax([r["target_score"] for r in results])
            b = results[best_idx]
            send_notification(f"📊 Trial #{trial_num}/1000 | Best 4Way={b['acc_4way_post']*100:.1f}%, Drug={b['acc_drug_post']*100:.1f}%, GT={b['acc_gt_post']*100:.1f}% (K={b['K']})")

        return target_score

    db_path = os.path.join(out_dir, "optuna_study_1000.db")
    storage_name = f"sqlite:///{db_path}"

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="strategy_bertopic_1000_sweep",
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )

    n_remaining = max(0, 1000 - len(study.trials))
    log_msg(f"Study currently has {len(study.trials)} trials. Running {n_remaining} remaining trials (Total = 1000)...")

    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining)

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(out_dir, "bayesian_sweep_1000_results.csv")
    df_res.to_csv(csv_out, index=False)

    log_msg("\n" + "=" * 80)
    log_msg("1000-TRIAL BAYESIAN SWEEP COMPLETED")
    log_msg("=" * 80)
    best_row = df_res.loc[df_res["target_score"].idxmax()]
    log_msg(f"Best Trial #{best_row['trial']}:")
    log_msg(f"  nn={best_row['nn']}, mcs={best_row['mcs']}, ms={best_row['ms']}, nc={best_row['nc']}")
    log_msg(f"  K = {best_row['K']}")
    log_msg(f"  Post 4-Way Accuracy  = {best_row['acc_4way_post']*100:.1f}%")
    log_msg(f"  Post Drug Accuracy   = {best_row['acc_drug_post']*100:.1f}%")
    log_msg(f"  Post Genotype Acc    = {best_row['acc_gt_post']*100:.1f}%")
    log_msg(f"  Pre Genotype Acc     = {best_row['acc_gt_pre']*100:.1f}%")
    log_msg(f"  Target Score         = {best_row['target_score']:.4f}")

    send_notification(f"🏆 1000-Trial GPU Sweep Complete! Best K={best_row['K']}, 4Way={best_row['acc_4way_post']*100:.1f}%, Drug={best_row['acc_drug_post']*100:.1f}%, GT={best_row['acc_gt_post']*100:.1f}%")
    log_msg("=" * 80)

if __name__ == "__main__":
    main()
