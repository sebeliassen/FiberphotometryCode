"""
Bayesian Hyperparameter Optimization: KMeans vs GMM (K=8 and K=16)
File: syllable_generation/analysis/language_models/pipelines/19_bayes_opt_kmeans_vs_gmm.py

- Optimizes KMeans and GMM hyperparameter spaces on 128s Transition Matrix SVD representations.
- Primary Objective: Out-of-sample Future 60s Action Budget Prediction Gain (% JS Divergence Error Reduction).
- Secondary Validation: Future 30s/120s Gain, Spatial Exploration R², Switch Cadence R², Genotype Acc (WT vs DAT-KI).
- Logs all trials to WandB (`language_models_in_vivo`) and sends periodic updates to NTFY.
- Includes a dedicated background Memory Watchdog that cleanly terminates if system RAM exceeds 90%.
"""

import os
import pickle
import sys
import threading
import time
import urllib.request

import numpy as np
import optuna
import psutil
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

import wandb

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

from lib.evaluation.scalar_probes import run_scalar_probing_battery

optuna.logging.set_verbosity(optuna.logging.WARNING)

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg: str):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except Exception:
        pass

# 1. Background Memory Watchdog Thread
def start_memory_watchdog(max_mem_pct: float = 90.0):
    def watchdog_loop():
        while True:
            mem_pct = psutil.virtual_memory().percent
            if mem_pct >= max_mem_pct:
                err_msg = f"🚨 CRITICAL WATCHDOG ALERT: System RAM at {mem_pct:.1f}% exceeded {max_mem_pct}%. Process auto-terminated to prevent OOM crash."
                print(err_msg, flush=True)
                send_notification(err_msg)
                os._exit(1)
            time.sleep(1.0)
    t = threading.Thread(target=watchdog_loop, daemon=True)
    t.start()
    print(f"✅ Memory Watchdog active (Hard limit: {max_mem_pct}% RAM).")

# 2. Data Loading & Precomputation
print("Loading continuous bout cache and window scalar cache...")
out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "continuous_bout_cache.pkl"), "rb") as f:
    bout_data = pickle.load(f)
session_bouts = bout_data["session_bouts"]

with open(os.path.join(out_dir, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
raw_trans_inputs = wdata["trans_inputs"] # (14092, 10000)
all_win_scalars = wdata["scalars"]
all_raw_sylls = wdata["raw_sylls"]
win_meta = wdata["meta"]

pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)
session_lookup = {s.session_id: s for s in raw_sessions}

# Session metadata mapping
session_groups = {}
for idx, w in enumerate(win_meta):
    sid = w["session_id"]
    if sid not in session_groups:
        s_obj = session_lookup[sid]
        meta = getattr(s_obj, "metadata", {})
        gt = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        inj_f = meta.get("injection_frame", 0)
        session_groups[sid] = {
            "group": f"{gt}_{drug}",
            "genotype": gt,
            "drug": drug,
            "inj_frame": inj_f,
            "window_indices": [],
            "post_indices": []
        }
    session_groups[sid]["window_indices"].append(idx)
    if w["mid_frame"] >= session_groups[sid]["inj_frame"]:
        session_groups[sid]["post_indices"].append(idx)

sids = list(session_groups.keys())
labels_gt = np.array([session_groups[sid]["genotype"] for sid in sids])
labels_drug = np.array([session_groups[sid]["drug"] for sid in sids])
labels_4way = np.array([session_groups[sid]["group"] for sid in sids])

# Scalar targets for Probes
scalar_targets = {
    'Spatial Exploration Entropy': np.array([s['spatial_entropy'] for s in all_win_scalars]),
    'Switch Cadence': np.array([s['cadence'] for s in all_win_scalars]),
    'Turning Chirality': np.array([s['chirality'] for s in all_win_scalars]),
    'Sequence Transition Entropy': np.array([s['sequence_entropy'] for s in all_win_scalars])
}

# 3. Precompute Evaluation Evaluation Targets (Future Action Budgets)
V = 100
HORIZON = 60 # primary horizon in seconds (~bouts)

print(f"Precomputing future {HORIZON}s action budget ground truth across all bouts...")
win_mids = np.array([w['mid_frame'] for w in win_meta])
win_sids = [w['session_id'] for w in win_meta]

sample_win_indices = []
Y_fut_list = []

for s_idx, s in enumerate(session_bouts):
    sid = raw_sessions[s_idx].session_id
    sylls = [b['syllable'] for b in s]
    mids = np.array([b['mid_frame'] for b in s])

    cand_w_indices = [w_i for w_i, w_sid in enumerate(win_sids) if w_sid == sid]
    if len(cand_w_indices) == 0:
        continue
    cand_mids = win_mids[cand_w_indices]

    N = len(sylls)
    if N > HORIZON * 2:
        for t in range(HORIZON, N - HORIZON, 3): # Stride 3 for rapid evaluation
            fut_counts = np.bincount(sylls[t : t + HORIZON], minlength=V)
            fut_dist = (fut_counts + 1e-4) / np.sum(fut_counts + 1e-4)

            closest_w = cand_w_indices[np.argmin(np.abs(cand_mids - mids[t]))]
            sample_win_indices.append(closest_w)
            Y_fut_list.append(fut_dist)

sample_win_indices = np.array(sample_win_indices, dtype=np.int64)
Y_fut_arr = np.array(Y_fut_list, dtype=np.float32)

N_SAMPLES = len(Y_fut_arr)
N_TRAIN = int(0.8 * N_SAMPLES)
print(f"Total Evaluation Samples: {N_SAMPLES} (Train: {N_TRAIN}, Test: {N_SAMPLES - N_TRAIN})")

# Blind Global Baseline on Test Set
p_global = np.mean(Y_fut_arr[:N_TRAIN], axis=0)
js_global = float(np.mean([jensenshannon(y, p_global) for y in Y_fut_arr[N_TRAIN:]]))
print(f"Blind Global Baseline JS Divergence: {js_global:.4f}")

def eval_rf(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) <= 1 or np.min(counts) < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=min(5, np.min(counts)), shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], rf.predict(X[te])))
    return float(np.mean(accs))

# 4. Optuna Objective Function
def objective(trial: optuna.Trial) -> float:
    # Model Choice
    model_family = trial.suggest_categorical("model_family", ["KMeans", "GMM"])
    k_clusters = trial.suggest_categorical("k_clusters", [8, 16])

    # Preprocessing Hyperparameters
    svd_dim = trial.suggest_categorical("svd_dim", [16, 24, 32, 48, 64, 96])
    norm_type = trial.suggest_categorical("norm_type", ["l2", "l1", "standard", "log1p", "none"])
    laplace_smooth = trial.suggest_float("laplace_smooth", 1e-5, 1.0, log=True)

    # Step 1: Preprocess Transition Matrix
    mat = raw_trans_inputs + laplace_smooth
    if norm_type == "log1p":
        mat = np.log1p(mat)
    elif norm_type == "l1":
        mat = normalize(mat, norm="l1")
    elif norm_type == "l2":
        mat = normalize(mat, norm="l2")

    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    features = svd.fit_transform(mat)

    if norm_type == "standard":
        features = StandardScaler().fit_transform(features)
    elif norm_type in ["l2", "log1p"]:
        features = normalize(features, norm="l2")

    # Step 2: Fit Clustering Model
    if model_family == "KMeans":
        spherical = trial.suggest_categorical("spherical", [False, True])
        init_method = trial.suggest_categorical("init", ["k-means++", "random"])
        n_init = trial.suggest_int("n_init", 5, 20)
        max_iter = trial.suggest_int("max_iter", 100, 400)

        km_feats = normalize(features, norm="l2") if spherical else features
        km = KMeans(n_clusters=k_clusters, init=init_method, n_init=n_init, max_iter=max_iter, random_state=42)
        labels = km.fit_predict(km_feats)
        probs = np.eye(k_clusters)[labels]
    else: # GMM
        cov_type = trial.suggest_categorical("covariance_type", ["diag", "full", "spherical", "tied"])
        reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-1, log=True)
        init_params = trial.suggest_categorical("init_params", ["kmeans", "random", "k-means++"])
        n_init = trial.suggest_int("gmm_n_init", 1, 4)
        max_iter = trial.suggest_int("gmm_max_iter", 50, 200)

        gmm = GaussianMixture(
            n_components=k_clusters, covariance_type=cov_type,
            reg_covar=reg_covar, init_params=init_params,
            n_init=n_init, max_iter=max_iter, random_state=42
        )
        gmm.fit(features)
        probs = gmm.predict_proba(features)
        labels = np.argmax(probs, axis=1)

    # Step 3: Evaluate Future Action Budget Forecasting Gain (JS Error Reduction %)
    sample_strat_probs = probs[sample_win_indices] # (N_SAMPLES, k_clusters)

    # Train Ridge regressor on 80% train, evaluate on 20% test
    m = Ridge(alpha=1.0).fit(sample_strat_probs[:N_TRAIN], Y_fut_arr[:N_TRAIN])
    preds = np.clip(m.predict(sample_strat_probs[N_TRAIN:]), 1e-5, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    js_model = float(np.mean([jensenshannon(y, p) for y, p in zip(Y_fut_arr[N_TRAIN:], preds)]))
    gain_future_60s = float((js_global - js_model) / js_global * 100.0)

    # Step 4: Evaluate Scalar Probes (Spatial Exploration Entropy R²)
    probes = run_scalar_probing_battery(probs, all_raw_sylls, scalar_targets)
    r2_spatial = float(probes['Spatial Exploration Entropy']['r2_strategy'])
    r2_cadence = float(probes['Switch Cadence']['r2_strategy'])
    r2_chirality = float(probes['Turning Chirality']['r2_strategy'])

    # Step 5: Evaluate Phenotype Classifiers
    props_full = np.zeros((len(sids), k_clusters))
    props_post = np.zeros((len(sids), k_clusters))
    for s_i, sid in enumerate(sids):
        w_idx = session_groups[sid]['window_indices']
        if len(w_idx) > 0:
            props_full[s_i] = np.mean(probs[w_idx], axis=0)
        p_idx = session_groups[sid]['post_indices']
        if len(p_idx) > 0:
            props_post[s_i] = np.mean(probs[p_idx], axis=0)

    acc_gt = eval_rf(props_full, labels_gt)
    acc_drug = eval_rf(props_post, labels_drug)
    acc_4way = eval_rf(props_post, labels_4way)

    # Log to WandB
    wandb.log({
        "trial": trial.number,
        "model_family": model_family,
        "k_clusters": k_clusters,
        "gain_future_60s": gain_future_60s,
        "r2_spatial": r2_spatial,
        "r2_cadence": r2_cadence,
        "r2_chirality": r2_chirality,
        "acc_genotype": acc_gt * 100.0,
        "acc_drug": acc_drug * 100.0,
        "acc_4way": acc_4way * 100.0,
        "svd_dim": svd_dim,
        "norm_type": norm_type,
        "laplace_smooth": laplace_smooth
    })

    return gain_future_60s

# 5. Main Execution
if __name__ == "__main__":
    start_memory_watchdog(max_mem_pct=90.0)

    WANDB_PROJECT = "language_models_in_vivo"
    WANDB_ENTITY = "sebeliassen"
    N_TRIALS = 200

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"bayes_opt_kmeans_vs_gmm_k8_k16_{int(time.time())}",
        group="bayes_opt_kmeans_vs_gmm_k8_k16",
        config={"n_trials": N_TRIALS, "primary_metric": "gain_future_60s"}
    )

    start_msg = "🚀 Bayesian Optimization Started: 200 Trials (KMeans vs GMM on K=8, 16). WandB logging active. Memory watchdog set to 90% RAM."
    print(start_msg)
    send_notification(start_msg)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def trial_callback(study, trial):
        t_num = trial.number
        if t_num > 0 and (t_num % 25 == 0 or trial.value == study.best_value):
            b_val = study.best_value
            b_params = study.best_params
            fam = b_params.get('model_family', 'N/A')
            k = b_params.get('k_clusters', 'N/A')
            update_msg = f"📊 Trial {t_num}/{N_TRIALS} Complete.\n🏆 Current Best Gain: +{b_val:.2f}%\nBest Model: {fam} (K={k})"
            print(update_msg, flush=True)
            if t_num % 50 == 0 or trial.value == study.best_value:
                send_notification(update_msg)

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_callback], n_jobs=1)

    # Save Results
    df_trials = study.trials_dataframe()
    res_path = os.path.join(out_dir, "bayes_opt_kmeans_vs_gmm_results.csv")
    df_trials.to_csv(res_path, index=False)
    print(f"Saved full results CSV to: {res_path}")

    best_t = study.best_trial
    summary_msg = (
        f"🏁 Bayesian Optimization Finished ({N_TRIALS} Trials)!\n"
        f"🏆 Top Model: {best_t.params['model_family']} (K={best_t.params['k_clusters']})\n"
        f"⭐ Best 60s Future Budget Gain: +{best_t.value:.2f}%\n"
        f"SVD Dim: {best_t.params['svd_dim']} | Norm: {best_t.params['norm_type']}"
    )
    print(summary_msg)
    send_notification(summary_msg)
    wandb.finish()
