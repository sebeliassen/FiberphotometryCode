import os
import sys
import argparse
import time
import pickle
import urllib.request
import numpy as np
import pandas as pd
from typing import List, Dict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        
    from analysis.language_models.lib.fusion.clustering import KinematicSyllableClustered
    from analysis.language_models.lib.models.strategies import StrategyBERTopic
    from analysis.language_models.lib.evaluation import group_analysis
    
    f_mode = cfg["feature_mode"]
    c_model = cfg["cluster_model"]
    vec = cfg["vectorizer"]
    n_neigh = cfg["n_neighbors"]
    mcs = cfg["min_cluster_size"]
    
    run_name = f"{f_mode}_{c_model}_{vec}_n{n_neigh}_mcs{mcs}"
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
        
        # Fit clusterer
        clusterer = KinematicSyllableClustered(
            n_clusters=3,
            min_bouts=10,
            use_gpu=False,
            feature_mode=f_mode,
            cluster_model=c_model
        )
        clusterer.fit(sessions)
        
        # Extract window inputs (internally cached inside group_analysis.py)
        all_win_inputs, session_meta = group_analysis.extract_window_inputs(sessions, clusterer, vec)
        
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
        for line in rf_summary_table.split("\n"):
            if "BERTopic Strategies" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                accuracy = float(parts[2])
                break
                
        # Save progress txt to enable resumption
        with open(run_txt_path, "w") as wf:
            wf.write(f"accuracy: {accuracy}\n")
            wf.write(f"table:\n{rf_summary_table}\n")
            
        return {"status": "success", "cfg": cfg, "accuracy": accuracy, "run_name": run_name, "run_idx": run_idx}
        
    except Exception as run_err:
        return {"status": "failed", "cfg": cfg, "error": str(run_err), "run_name": run_name, "run_idx": run_idx}

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Behavioral Strategy Pipeline Sweep")
    parser.add_argument("--dataset", type=str, default="amph", help="Dataset to load")
    parser.add_argument("--parallel", action="store_true", help="Run CPU configs in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for parallel execution")
    args = parser.parse_args()
    
    # 1. Load Sessions
    print(f"\n*** Loading {args.dataset.upper()} dataset... ***")
    sessions = loader.load_dataset(args.dataset, base_dir=ROOT_DIR)
    print(f"Loaded {len(sessions)} sessions.")
    
    # Define hyperparameter grid search options
    feature_modes = ["kinematic", "postural", "hybrid"]
    cluster_models = ["kmeans", "gmm"]
    vectorizers = ["count", "transition", "transition_base", "soft_count", "moseq_soft_count"]
    n_neighbors_opts = [15, 30]
    min_cluster_size_opts = [100, 200]
    
    # Setup output paths
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "hyperparameter_sweep_results.md")
    
    # Caches to avoid duplicate fit calls (used in sequential mode)
    fitted_clusterers = {}
    cached_window_inputs = {}
    
    # Build list of all configurations to run
    configs = []
    for f_mode in feature_modes:
        for c_model in cluster_models:
            for vec in vectorizers:
                if vec == "soft_count" and c_model != "gmm":
                    continue
                if vec == "moseq_soft_count" and (f_mode != "kinematic" or c_model != "kmeans"):
                    continue
                    
                for n_neigh in n_neighbors_opts:
                    for mcs in min_cluster_size_opts:
                        configs.append({
                            "feature_mode": f_mode,
                            "cluster_model": c_model,
                            "vectorizer": vec,
                            "n_neighbors": n_neigh,
                            "min_cluster_size": mcs
                        })
                        
    # Run all non-marginal based configs first to yield results immediately
    configs = [c for c in configs if c["vectorizer"] != "moseq_soft_count"] + \
              [c for c in configs if c["vectorizer"] == "moseq_soft_count"]
                        
    total_configs = len(configs)
    print(f"Total configurations generated for sweep: {total_configs}")
    
    # Pre-filter completed runs to avoid scheduling them
    completed_runs = []
    pending_configs = []
    
    for run_idx, cfg in enumerate(configs):
        f_mode = cfg["feature_mode"]
        c_model = cfg["cluster_model"]
        vec = cfg["vectorizer"]
        n_neigh = cfg["n_neighbors"]
        mcs = cfg["min_cluster_size"]
        
        run_name = f"{f_mode}_{c_model}_{vec}_n{n_neigh}_mcs{mcs}"
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
    send_notification(f"Starting Hyperparameter Sweep. Total: {total_configs}, Pending: {len(pending_configs)}. Parallel: {args.parallel}")
    
    # Relaunching sweep report with currently completed
    if completed_runs:
        write_markdown_report(report_path, completed_runs)
        
    if not pending_configs:
        print("\n*** All configurations already completed! ***")
        send_notification("Hyperparameter sweep finished! All configurations already completed.")
        return

    # Split configs into marginal-dependent (slow) and non-marginal (fast)
    fast_pending = [item for item in pending_configs if item[0]["vectorizer"] != "moseq_soft_count"]
    slow_pending = [item for item in pending_configs if item[0]["vectorizer"] == "moseq_soft_count"]

    if args.parallel:
        print(f"Running FAST CPU-bound configurations in parallel using ProcessPoolExecutor ({args.workers} workers)...")
        
        # 1. First run fast configs in parallel
        if fast_pending:
            tasks = [(cfg, r_idx, total_configs, args.dataset, ROOT_DIR, runs_dir) for cfg, r_idx in fast_pending]
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = {executor.submit(process_config_task, t): t for t in tasks}
                for fut in as_completed(futures):
                    res = fut.result()
                    r_name = res["run_name"]
                    r_idx = res["run_idx"]
                    if res["status"] == "success":
                        acc = res["accuracy"]
                        completed_runs.append({**res["cfg"], "accuracy": acc})
                        print(f"[{r_idx+1}/{total_configs}] Success: {r_name} (Accuracy: {acc:.4f})")
                        send_notification(f"Run {r_idx+1}/{total_configs} completed. Accuracy: {acc:.2%} | Config: {r_name}")
                        write_markdown_report(report_path, completed_runs)
                    elif res["status"] == "failed":
                        err_msg = res["error"]
                        print(f"[{r_idx+1}/{total_configs}] Failed: {r_name} | Error: {err_msg}")
                        send_notification(f"ERROR in run {r_idx+1}/{total_configs} ({r_name}): {err_msg}")
                        
        # 2. Run slow (marginal based) configs sequentially to utilize the GPU and avoid OOM thrashing
        if slow_pending:
            print("\nRunning SLOW GPU-bound configurations sequentially to avoid VRAM exhaustion...")
            for cfg, r_idx in slow_pending:
                task_arg = (cfg, r_idx, total_configs, args.dataset, ROOT_DIR, runs_dir)
                res = process_config_task(task_arg)
                r_name = res["run_name"]
                if res["status"] == "success":
                    acc = res["accuracy"]
                    completed_runs.append({**res["cfg"], "accuracy": acc})
                    print(f"[{r_idx+1}/{total_configs}] Success: {r_name} (Accuracy: {acc:.4f})")
                    send_notification(f"Run {r_idx+1}/{total_configs} completed. Accuracy: {acc:.2%} | Config: {r_name}")
                    write_markdown_report(report_path, completed_runs)
                elif res["status"] == "failed":
                    err_msg = res["error"]
                    print(f"[{r_idx+1}/{total_configs}] Failed: {r_name} | Error: {err_msg}")
                    send_notification(f"ERROR in run {r_idx+1}/{total_configs} ({r_name}): {err_msg}")
    else:
        # Sequential execution mode
        from analysis.language_models.lib.fusion.clustering import KinematicSyllableClustered
        from analysis.language_models.lib.models.strategies import StrategyBERTopic
        from analysis.language_models.lib.evaluation import group_analysis
        
        print("Running all configurations sequentially...")
        for cfg, r_idx in pending_configs:
            f_mode = cfg["feature_mode"]
            c_model = cfg["cluster_model"]
            vec = cfg["vectorizer"]
            n_neigh = cfg["n_neighbors"]
            mcs = cfg["min_cluster_size"]
            run_name = f"{f_mode}_{c_model}_{vec}_n{n_neigh}_mcs{mcs}"
            run_txt_path = os.path.join(runs_dir, f"{run_name}.txt")
            
            print(f"\n==================================================")
            print(f"[{r_idx+1}/{total_configs}] Running: {run_name}")
            print(f"==================================================")
            
            try:
                # Step 1: Clusterer Caching
                clusterer_key = (f_mode, c_model)
                if clusterer_key not in fitted_clusterers:
                    print(f"Fitting KinematicSyllableClustered (feature_mode={f_mode}, cluster_model={c_model})...")
                    clusterer = KinematicSyllableClustered(
                        n_clusters=3,
                        min_bouts=10,
                        use_gpu=False,
                        feature_mode=f_mode,
                        cluster_model=c_model
                    )
                    clusterer.fit(sessions)
                    fitted_clusterers[clusterer_key] = clusterer
                else:
                    clusterer = fitted_clusterers[clusterer_key]
                    
                # Step 2: Window Slicing Caching
                window_key = (f_mode, c_model, vec)
                if window_key not in cached_window_inputs:
                    print(f"Extracting sliding window inputs for vectorizer: {vec}...")
                    all_win_inputs, session_meta = group_analysis.extract_window_inputs(sessions, clusterer, vec)
                    cached_window_inputs[window_key] = (all_win_inputs, session_meta)
                else:
                    all_win_inputs, session_meta = cached_window_inputs[window_key]
                    
                # Step 3: Fit StrategyBERTopic
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
                
                # Step 4: Evaluate Predictability via stratified RF
                rf_summary_table = group_analysis.predict_experimental_groups(sessions, clusterer, strategy_model)
                
                accuracy = 0.0
                for line in rf_summary_table.split("\n"):
                    if "BERTopic Strategies" in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        accuracy = float(parts[2])
                        break
                        
                print(f"Run completed. Accuracy: {accuracy:.4f}")
                
                # Save progress txt to enable resumption
                with open(run_txt_path, "w") as wf:
                    wf.write(f"accuracy: {accuracy}\n")
                    wf.write(f"table:\n{rf_summary_table}\n")
                    
                cfg_w_acc = {**cfg, "accuracy": accuracy}
                completed_runs.append(cfg_w_acc)
                
                # Send ntfy notification
                send_notification(f"Run {r_idx+1}/{total_configs} completed. Accuracy: {accuracy:.2%} | Config: {run_name}")
                
                # Incrementally write markdown report
                write_markdown_report(report_path, completed_runs)
                
            except Exception as run_err:
                print(f"Error in configuration {run_name}: {run_err}")
                send_notification(f"ERROR in run {r_idx+1}/{total_configs} ({run_name}): {run_err}")
                
    print("\n*** Hyperparameter Sweep Completed! ***")
    send_notification("Hyperparameter sweep finished successfully! Check hyperparameter_sweep_results.md for the final rankings.")

def write_markdown_report(path: str, runs: List[Dict]):
    # Sort runs by accuracy descending
    sorted_runs = sorted(runs, key=lambda x: x["accuracy"], reverse=True)
    
    rows = []
    rows.append("# Unsupervised Strategy Pipeline - Hyperparameter Sweep Results")
    rows.append(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rows.append(f"Total evaluated configurations: {len(runs)}")
    rows.append("\n| Rank | Feature Mode | Clusterer Model | Vectorizer | UMAP n_neighbors | HDBSCAN min_cluster_size | Classifier Accuracy |")
    rows.append("|---|---|---|---|---|---|---|")
    
    for idx, r in enumerate(sorted_runs):
        rows.append(
            f"| {idx+1} | `{r['feature_mode']}` | `{r['cluster_model']}` | `{r['vectorizer']}` | {r['n_neighbors']} | {r['min_cluster_size']} | **{r['accuracy']:.2%}** |"
        )
        
    with open(path, "w") as f:
        f.write("\n".join(rows))

if __name__ == "__main__":
    main()
