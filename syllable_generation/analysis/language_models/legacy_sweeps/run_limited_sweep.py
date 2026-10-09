import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gc
import multiprocessing
import pickle
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

# Resolve root path and add to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg: str):
    try:
        urllib.request.urlopen(NTFY_URL, data=msg.encode('utf-8'))
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")

_SESSIONS_CACHE = None

def get_sessions(dataset, base_dir):
    global _SESSIONS_CACHE
    if _SESSIONS_CACHE is None:
        _SESSIONS_CACHE = loader.load_dataset(dataset, base_dir=base_dir)
    return _SESSIONS_CACHE

def get_cached_prep(dataset, f_mode, c_model, k, vec, sessions, cache_dir, base_dir):
    from analysis.language_models.lib.evaluation import group_analysis
    from analysis.language_models.lib.fusion.clustering import KinematicSyllableClustered

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"prep_{dataset}_{f_mode}_{c_model}_k{k}_{vec}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as rf:
                return pickle.load(rf)
        except Exception:
            pass

    # Fit clusterer
    clusterer = KinematicSyllableClustered(
        n_clusters=k,
        min_bouts=10,
        use_gpu=False,
        feature_mode=f_mode,
        cluster_model=c_model
    )
    clusterer.fit(sessions)
    all_win_inputs, session_meta = group_analysis.extract_window_inputs(sessions, clusterer, vec)

    # Save cache atomically
    tmp_path = cache_file + ".tmp"
    try:
        with open(tmp_path, "wb") as wf:
            pickle.dump((clusterer, all_win_inputs, session_meta), wf)
        os.replace(tmp_path, cache_file)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return clusterer, all_win_inputs, session_meta

def process_config_task(args_tuple):
    cfg, run_idx, total_configs, dataset, base_dir, runs_dir = args_tuple

    # Prevent JAX preallocation in child processes
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["PARALLEL_CPU"] = "1" # Forces Random Forest in children to use n_jobs=1

    # Imports inside worker to avoid pre-initializing JAX/CUDA in parent
    import sys
    if base_dir not in sys.path:
        sys.path.append(base_dir)
        sys.path.append(os.path.join(base_dir, "syllable_generation"))

    from analysis.language_models.lib.evaluation import group_analysis
    from analysis.language_models.lib.models.strategies import StrategyBERTopic

    f_mode = cfg["feature_mode"]
    c_model = cfg["cluster_model"]
    k = cfg["n_clusters"]
    vec = cfg["vectorizer"]
    n_neigh = cfg["n_neighbors"]
    mcs = cfg["min_cluster_size"]

    run_name = f"{f_mode}_{c_model}_k{k}_{vec}_n{n_neigh}_mcs{mcs}"
    run_txt_path = os.path.join(runs_dir, f"{run_name}.txt")

    # Check if already completed
    if os.path.exists(run_txt_path):
        try:
            with open(run_txt_path, "r") as rf:
                lines = rf.readlines()
                acc = float(lines[0].split(":")[1].strip())
            return {"status": "skipped", "cfg": cfg, "accuracy": acc, "run_name": run_name, "run_idx": run_idx}
        except Exception:
            pass

    try:
        sessions = get_sessions(dataset, base_dir)
        cache_dir = os.path.join(os.path.dirname(runs_dir), "cache")
        clusterer, all_win_inputs, session_meta = get_cached_prep(dataset, f_mode, c_model, k, vec, sessions, cache_dir, base_dir)

        # Fit StrategyBERTopic
        vocab_size = clusterer.next_token_id
        strategy_model = StrategyBERTopic(
            vectorizer_type=vec,
            n_components=5,
            n_neighbors=n_neigh,
            min_cluster_size=mcs,
            min_samples=10,
            random_state=42,
            vocab_size=vocab_size
        )
        strategy_model.fit(all_win_inputs)


        # Evaluate
        rf_summary_table = group_analysis.predict_experimental_groups(sessions, clusterer, strategy_model)

        accuracy = 0.0
        if isinstance(rf_summary_table, str):
            for line in rf_summary_table.split("\n"):
                if "BERTopic Strategies" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 3:
                        try:
                            accuracy = float(parts[2])
                        except (ValueError, IndexError):
                            accuracy = 0.0
                    break

        # Save progress txt to enable resumption
        with open(run_txt_path, "w") as wf:
            wf.write(f"accuracy: {accuracy}\n")
            wf.write(f"table:\n{rf_summary_table}\n")

        return {"status": "success", "cfg": cfg, "accuracy": accuracy, "run_name": run_name, "run_idx": run_idx}

    except Exception as run_err:
        # Record failed run text so sweep does not get stuck in a loop
        try:
            with open(run_txt_path, "w") as wf:
                wf.write("accuracy: 0.0\n")
                wf.write(f"error: {run_err}\n")
        except Exception:
            pass
        return {"status": "failed", "cfg": cfg, "error": str(run_err), "run_name": run_name, "run_idx": run_idx}
    finally:
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Targeted Behavioral Strategy Pipeline Sweep")
    parser.add_argument("--dataset", type=str, default="amph", help="Dataset to load")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for parallel execution")
    args = parser.parse_args()

    # 1. Load Sessions
    print(f"\n*** Loading {args.dataset.upper()} dataset... ***")
    sessions = loader.load_dataset(args.dataset, base_dir=ROOT_DIR)
    print(f"Loaded {len(sessions)} sessions.")

    # Define targeted hyperparameter grid search options based on top performers
    feature_modes = ["postural", "hybrid"]
    cluster_models = ["gmm"]
    vectorizers = ["transition", "soft_count", "transition_base"]
    n_clusters_opts = [2, 3, 4]
    n_neighbors_opts = [30, 45]
    min_cluster_size_opts = [50, 75, 100]

    # Setup output paths
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    report_path = os.path.join(out_dir, "limited_sweep_results.md")

    # Build list of all configurations to run
    configs = []
    for f_mode in feature_modes:
        for c_model in cluster_models:
            for vec in vectorizers:
                for k in n_clusters_opts:
                    for n_neigh in n_neighbors_opts:
                        for mcs in min_cluster_size_opts:
                            configs.append({
                                "feature_mode": f_mode,
                                "cluster_model": c_model,
                                "n_clusters": k,
                                "vectorizer": vec,
                                "n_neighbors": n_neigh,
                                "min_cluster_size": mcs
                            })

    total_configs = len(configs)
    print(f"Total configurations generated for targeted sweep: {total_configs}")

    # Pre-filter completed runs to avoid scheduling them
    completed_runs = []
    pending_configs = []

    for run_idx, cfg in enumerate(configs):
        f_mode = cfg["feature_mode"]
        c_model = cfg["cluster_model"]
        k = cfg["n_clusters"]
        vec = cfg["vectorizer"]
        n_neigh = cfg["n_neighbors"]
        mcs = cfg["min_cluster_size"]

        run_name = f"{f_mode}_{c_model}_k{k}_{vec}_n{n_neigh}_mcs{mcs}"
        run_txt_path = os.path.join(runs_dir, f"{run_name}.txt")

        if os.path.exists(run_txt_path):
            try:
                with open(run_txt_path, "r") as rf:
                    lines = rf.readlines()
                    acc = float(lines[0].split(":")[1].strip())
                    completed_runs.append({**cfg, "accuracy": acc})
                print(f"[{run_idx+1}/{total_configs}] Skipping completed run: {run_name} (Accuracy: {acc:.4f})")
                continue
            except Exception:
                pass
        pending_configs.append((cfg, run_idx))

    print(f"Pending configurations to run: {len(pending_configs)}")
    send_notification(f"Starting Targeted Strategy Sweep. Total: {total_configs}, Pending: {len(pending_configs)}. Workers: {args.workers}")

    if completed_runs:
        write_markdown_report(report_path, completed_runs)

    if not pending_configs:
        print("\n*** All configurations already completed! ***")
        send_notification("Targeted strategy sweep finished! All configurations already completed.")
        return

    print(f"Running CPU-bound configurations in parallel using ProcessPoolExecutor ({args.workers} workers max, memory-guarded)...")
    tasks = [(cfg, r_idx, total_configs, args.dataset, ROOT_DIR, runs_dir) for cfg, r_idx in pending_configs]

    def process_result(res):
        r_name = res["run_name"]
        r_idx = res["run_idx"]
        if res["status"] == "success":
            acc = res["accuracy"]
            completed_runs.append({**res["cfg"], "accuracy": acc})
            print(f"[{r_idx+1}/{total_configs}] Success: {r_name} (Accuracy: {acc:.4f})")
            send_notification(f"Run {r_idx+1}/{total_configs} completed. Accuracy: {acc:.2%} | Config: {r_name}")
            write_markdown_report(report_path, completed_runs)
        elif res["status"] == "failed":
            err_msg = res.get("error", "Unknown error")
            print(f"[{r_idx+1}/{total_configs}] Failed: {r_name} | Error: {err_msg}")
            send_notification(f"ERROR in run {r_idx+1}/{total_configs} ({r_name}): {err_msg}")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending_futures = {executor.submit(process_config_task, t): t for t in tasks}
        for fut in as_completed(pending_futures):
            try:
                res = fut.result()
                process_result(res)
            except Exception as exc:
                print(f"Worker task raised exception: {exc}")
            gc.collect()

    print("\n*** Targeted Hyperparameter Sweep Completed! ***")
    send_notification("Targeted strategy sweep finished successfully! Check limited_sweep_results.md for rankings.")

def write_markdown_report(path: str, runs: List[Dict]):
    sorted_runs = sorted(runs, key=lambda x: x["accuracy"], reverse=True)
    rows = []
    rows.append("# Targeted Behavioral Strategy Pipeline - Hyperparameter Sweep Results")
    rows.append(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rows.append(f"Total evaluated configurations: {len(runs)}")
    rows.append("\n| Rank | Feature Mode | Clusterer Model | k Substates | Vectorizer | UMAP n_neighbors | HDBSCAN min_cluster_size | Classifier Accuracy |")
    rows.append("|---|---|---|---|---|---|---|---|")

    for idx, r in enumerate(sorted_runs):
        rows.append(
            f"| {idx+1} | `{r['feature_mode']}` | `{r['cluster_model']}` | {r['n_clusters']} | `{r['vectorizer']}` | {r['n_neighbors']} | {r['min_cluster_size']} | **{r['accuracy']:.2%}** |"
        )

    with open(path, "w") as f:
        f.write("\n".join(rows))

if __name__ == "__main__":
    main()
