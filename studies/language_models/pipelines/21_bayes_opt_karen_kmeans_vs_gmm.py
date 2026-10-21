#!/usr/bin/env python3
"""
21_bayes_opt_karen_kmeans_vs_gmm.py

Bayesian Hyperparameter Optimization for Strategy Models on the Karen Dataset.
- Evaluates KMeans vs GMM across K in [8, 12, 16] using RLE discrete bouts.
- Primary Objective: Out-of-sample future 60-bout action budget forecast gain (% JS Divergence reduction).
- Secondary Validation: Future 30/90/120-bout gains and Monomer vs PFF Genotype Classification accuracy.
- Sends live progress and alerts to NTFY (https://ntfy.sh/woody_videos_seb).
- Includes background RAM Memory Watchdog (auto-safeguard at 90%).
"""

import json
import os
import pickle
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import optuna
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PKL = ROOT_DIR / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = ROOT_DIR / "syllable_generation" / "analysis" / "language_models" / "output" / "karen_strategy_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NTFY_URL = "https://ntfy.sh/woody_videos_seb"
NUM_SYLLABLES = 100
WIN_BOUTS = 300
N_TRIALS = 150


def send_notification(msg: str, title: str = "Karen Bayesian Opt"):
    try:
        clean_title = title.encode('ascii', 'ignore').decode('ascii')
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode('utf-8'),
            headers={"Title": clean_title, "Priority": "default"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
        print(f"[NTFY Notification Sent] {clean_title}", flush=True)
    except Exception as e:
        print(f"[NTFY Notification Warning] {e}", flush=True)


def start_memory_watchdog(max_mem_pct: float = 90.0):
    def watchdog_loop():
        while True:
            mem_pct = psutil.virtual_memory().percent
            if mem_pct >= max_mem_pct:
                err_msg = f"🚨 CRITICAL WATCHDOG ALERT: System RAM at {mem_pct:.1f}% exceeded {max_mem_pct}%. Process terminated safely to prevent OOM."
                print(err_msg, flush=True)
                send_notification(err_msg, title="🚨 OOM Safeguard Alert")
                os._exit(1)
            time.sleep(1.0)
    t = threading.Thread(target=watchdog_loop, daemon=True)
    t.start()
    print(f"✅ Memory Watchdog active (Hard limit: {max_mem_pct}% RAM).", flush=True)


def compute_js_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def eval_rf(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) <= 1 or np.min(counts) < 2:
        return 0.0
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], rf.predict(X[te])))
    return float(np.mean(accs))


print("=" * 80)
print("Initializing Bayesian Hyperparameter Optimization for Karen Dataset")
print("=" * 80)
start_memory_watchdog(max_mem_pct=90.0)

print(f"Loading pickled dataset from: {DATA_PKL}")
with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

print("Extracting Run-Length Encoded discrete syllable bouts...")
session_bouts = []
for s in sessions:
    md = getattr(s, "metadata", {})
    g = md.get("genotype", "Unknown")
    aid = str(md.get("animal_id", ""))
    cond = md.get("condition", "Unknown")

    if g not in ["PFF", "Monomer"] or cond == "L_dopa":
        continue
    df = s.all_data_df.dropna(subset=["syllable"])
    if len(df) < 10000:
        continue
    sylls = df["syllable"].values.astype(int)
    changes = np.concatenate(([True], sylls[1:] != sylls[:-1]))
    collapsed = sylls[changes]

    if len(collapsed) > WIN_BOUTS + 130:
        session_bouts.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "genotype": g,
            "bouts": collapsed
        })

print(f"Extracted {len(session_bouts)} clean untreated sessions with RLE bouts.")

# Extract sliding windows of 300 past bouts with multi-horizon targets
HORIZONS = [30, 60, 90, 120]
samples = []
for s_idx, s in enumerate(session_bouts):
    aid = s["animal_id"]
    g = s["genotype"]
    b = s["bouts"]
    N = len(b)

    for t in range(WIN_BOUTS, N - 120, 10):
        past_b = b[t - WIN_BOUTS : t]
        curr_b = b[t]

        src = past_b[:-1]
        dst = past_b[1:]
        mat = np.zeros((NUM_SYLLABLES, NUM_SYLLABLES), dtype=np.float32)
        np.add.at(mat, (src, dst), 1.0)

        targets = {}
        for h in HORIZONS:
            fut_seq = b[t + 1 : t + 1 + h]
            f_counts = np.bincount(fut_seq, minlength=NUM_SYLLABLES)
            targets[h] = (f_counts + 1e-4) / np.sum(f_counts + 1e-4)

        samples.append({
            "session_idx": s_idx,
            "session_id": s["session_id"],
            "animal_id": aid,
            "genotype": g,
            "curr_syll": curr_b,
            "past_mat": mat.flatten(),
            "targets": targets
        })

print(f"Extracted {len(samples)} valid bout-level evaluation windows.")

# Split by Animal ID (80% train, 20% test)
all_animals = sorted(list(set(s["animal_id"] for s in samples)))
np.random.seed(42)
shuffled_animals = np.random.permutation(all_animals)
n_train_animals = max(1, int(0.8 * len(shuffled_animals)))
train_animals = set(shuffled_animals[:n_train_animals])
test_animals = set(shuffled_animals[n_train_animals:])

train_indices = np.array([i for i, s in enumerate(samples) if s["animal_id"] in train_animals])
test_indices = np.array([i for i, s in enumerate(samples) if s["animal_id"] in test_animals])

print(f"Animal Split: {len(train_animals)} Train ({len(train_indices)} windows) | {len(test_animals)} Held-Out Test ({len(test_indices)} windows)")

X_trans_raw = np.array([s["past_mat"] for s in samples], dtype=np.float32)
X_train_raw = X_trans_raw[train_indices]
X_test_raw = X_trans_raw[test_indices]

# Precompute Ground Truth Targets
Y_targets = {
    h: np.array([s["targets"][h] for s in samples], dtype=np.float32)
    for h in HORIZONS
}

# Blind Global Baselines on Test Set
p_global_targets = {
    h: np.mean(Y_targets[h][train_indices], axis=0) / np.sum(np.mean(Y_targets[h][train_indices], axis=0))
    for h in HORIZONS
}
js_blind_targets = {
    h: float(np.mean([compute_js_divergence(y, p_global_targets[h]) for y in Y_targets[h][test_indices]]))
    for h in HORIZONS
}
print(f"Blind Global Baseline (H=60 bouts): {js_blind_targets[60]:.4f} bits")

# Session Genotype Mapping for Phenotype Probe
session_genotypes = np.array([1 if s["genotype"] == "PFF" else 0 for s in session_bouts])


def objective(trial: optuna.Trial) -> float:
    # 1. Model Choice & Hyperparameters
    model_family = trial.suggest_categorical("model_family", ["KMeans", "GMM"])
    k_clusters = trial.suggest_categorical("k_clusters", [8, 12, 16])

    svd_dim = trial.suggest_categorical("svd_dim", [16, 24, 32, 48, 64, 96])
    norm_type = trial.suggest_categorical("norm_type", ["l2", "l1", "standard", "log1p", "none"])
    laplace_smooth = trial.suggest_float("laplace_smooth", 1e-5, 1.0, log=True)

    # 2. Preprocessing
    mat_tr = X_train_raw + laplace_smooth
    mat_te = X_test_raw + laplace_smooth

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
    elif norm_type in ["l2", "log1p"]:
        feats_tr = normalize(feats_tr, norm="l2")
        feats_te = normalize(feats_te, norm="l2")

    # 3. Fit Model
    if model_family == "KMeans":
        spherical = trial.suggest_categorical("spherical", [False, True])
        init_method = trial.suggest_categorical("init", ["k-means++", "random"])
        n_init = trial.suggest_int("n_init", 5, 20)
        max_iter = trial.suggest_int("max_iter", 100, 300)

        if spherical:
            feats_tr = normalize(feats_tr, norm="l2")
            feats_te = normalize(feats_te, norm="l2")

        km = KMeans(n_clusters=k_clusters, init=init_method, n_init=n_init, max_iter=max_iter, random_state=42)
        lbl_tr = km.fit_predict(feats_tr)
        lbl_te = km.predict(feats_te)
    else: # GMM
        cov_type = trial.suggest_categorical("covariance_type", ["diag", "full", "spherical", "tied"])
        reg_covar = trial.suggest_float("reg_covar", 1e-5, 1e-1, log=True)
        init_params = trial.suggest_categorical("init_params", ["kmeans", "random", "k-means++"])
        gmm_n_init = trial.suggest_int("gmm_n_init", 1, 3)
        gmm_max_iter = trial.suggest_int("gmm_max_iter", 50, 200)

        gmm = GaussianMixture(
            n_components=k_clusters,
            covariance_type=cov_type,
            reg_covar=reg_covar,
            init_params=init_params,
            n_init=gmm_n_init,
            max_iter=gmm_max_iter,
            random_state=42
        )
        lbl_tr = gmm.fit_predict(feats_tr)
        lbl_te = gmm.predict(feats_te)

    # 4. Evaluate Future Action Budget Forecast (H=60 bouts Primary Objective)
    gains = {}
    for h in HORIZONS:
        Y_tr = Y_targets[h][train_indices]
        Y_te = Y_targets[h][test_indices]

        strat_means = np.zeros((k_clusters, NUM_SYLLABLES), dtype=np.float32)
        for k_id in range(k_clusters):
            mask = (lbl_tr == k_id)
            if np.sum(mask) > 0:
                m_vec = np.mean(Y_tr[mask], axis=0)
                strat_means[k_id] = m_vec / np.sum(m_vec)
            else:
                strat_means[k_id] = p_global_targets[h]

        preds_te = strat_means[lbl_te]
        js_val = float(np.mean([compute_js_divergence(y, p) for y, p in zip(Y_te, preds_te)]))
        gains[h] = float((js_blind_targets[h] - js_val) / js_blind_targets[h] * 100.0)

    gain_60b = gains[60]

    # 5. Evaluate Secondary Genotype Classification (Monomer vs PFF)
    all_labels = np.zeros(len(samples), dtype=int)
    all_labels[train_indices] = lbl_tr
    all_labels[test_indices] = lbl_te

    session_strat_occupancy = np.zeros((len(session_bouts), k_clusters), dtype=np.float32)
    for s_i in range(len(session_bouts)):
        s_win_mask = np.array([s["session_idx"] == s_i for s in samples])
        if np.sum(s_win_mask) > 0:
            counts = np.bincount(all_labels[s_win_mask], minlength=k_clusters)
            session_strat_occupancy[s_i] = counts / np.sum(counts)

    acc_genotype = eval_rf(session_strat_occupancy, session_genotypes) * 100.0

    # Store attributes
    trial.set_user_attr("gain_30b", gains[30])
    trial.set_user_attr("gain_60b", gains[60])
    trial.set_user_attr("gain_90b", gains[90])
    trial.set_user_attr("gain_120b", gains[120])
    trial.set_user_attr("acc_genotype", acc_genotype)

    return gain_60b


def main():
    start_time = time.time()
    start_msg = (
        f"🚀 Karen Bayesian Hyperparameter Optimization Started!\n"
        f"🔬 Search: {N_TRIALS} Trials across KMeans & GMM (K=8, 12, 16)\n"
        f"📊 Objective: Out-of-sample 60-bout action budget forecast gain (% JS reduction)\n"
        f"🛡️ Memory Watchdog: 90% Hard RAM Limit active"
    )
    print(start_msg, flush=True)
    send_notification(start_msg, title="🚀 Karen Optuna Search Started")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    def callback(study, trial):
        t_num = trial.number
        if t_num > 0 and (t_num % 25 == 0 or trial.value == study.best_value):
            b_val = study.best_value
            b_params = study.best_params
            fam = b_params.get("model_family", "N/A")
            k = b_params.get("k_clusters", "N/A")
            svd = b_params.get("svd_dim", "N/A")
            norm = b_params.get("norm_type", "N/A")

            update_msg = (
                f"📊 Trial {t_num}/{N_TRIALS} Complete\n"
                f"🏆 Best 60-Bout Gain: +{b_val:.2f}%\n"
                f"Top Model: {fam} (K={k}, SVD={svd}, Norm={norm})"
            )
            print(update_msg, flush=True)
            if t_num % 50 == 0 or trial.value == study.best_value:
                send_notification(update_msg, title="📊 Karen Optuna Progress")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], n_jobs=1)

    # Save Full Study Results
    df_trials = study.trials_dataframe()
    csv_path = OUT_DIR / "bayes_opt_karen_results.csv"
    df_trials.to_csv(csv_path, index=False)
    print(f"\nSaved full trials CSV to: {csv_path}")

    # Save Best Parameters
    best_t = study.best_trial
    champion_data = {
        "best_trial_number": best_t.number,
        "best_gain_60b": best_t.value,
        "best_params": best_t.params,
        "secondary_metrics": best_t.user_attrs
    }
    json_path = OUT_DIR / "karen_champion_configs.json"
    with open(json_path, "w") as f:
        json.dump(champion_data, f, indent=2)
    print(f"Saved champion configs to: {json_path}")

    elapsed = time.time() - start_time
    summary_msg = (
        f"🏁 Karen Bayesian Optimization Complete in {elapsed/60:.1f} min!\n"
        f"🏆 Champion: {best_t.params['model_family']} (K={best_t.params['k_clusters']})\n"
        f"⭐ Best 60-Bout Gain: +{best_t.value:.2f}%\n"
        f"SVD Dim: {best_t.params['svd_dim']} | Norm: {best_t.params['norm_type']} | Smooth: {best_t.params['laplace_smooth']:.4f}\n"
        f"🧬 Monomer vs PFF Genotype Acc: {best_t.user_attrs.get('acc_genotype', 0.0):.1f}%"
    )
    print("\n" + "=" * 80)
    print(summary_msg)
    print("=" * 80)
    send_notification(summary_msg, title="🏁 Karen Optimization Finished")


if __name__ == "__main__":
    main()
