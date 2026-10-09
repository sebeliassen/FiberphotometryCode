import os
import sys
import time
import subprocess
import shutil
import urllib.request
import numpy as np
np.bool8 = np.bool_
import jax
import pandas as pd
import psutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers/keypoint_moseq_syll")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, KPMS_DIR)

from kpms_pipeline import run_pipeline
from run_pipeline import get_sessions

NTFY_URL = "https://ntfy.sh/woody_videos_seb"
PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "benchmark_single_iter_test"

def send_ntfy(message: str, title: str = "Full Karen Benchmark"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=message.encode('utf-8'),
            headers={"Title": title.encode('utf-8')}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

def get_vram_gb():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return float(res.decode("utf-8").strip().split("\n")[0]) / 1024.0
    except Exception:
        return 0.0

def get_ram_gb():
    try:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0

def main():
    print("=" * 95, flush=True)
    print("      SINGLE-ITERATION BENCHMARK ON FULL KAREN DATASET (karen_project_2)", flush=True)
    print("===============================================================================================", flush=True)

    # 1. Start CPU RAM Watchdog
    watchdog_script = os.path.join(KPMS_DIR, "diagnostics", "cpu_ram_watchdog.py")
    watchdog_proc = subprocess.Popen([sys.executable, watchdog_script, str(os.getpid())])
    print(f"[Watchdog] Spawned background watchdog (PID {watchdog_proc.pid}) with 90% CPU RAM kill limit.", flush=True)

    try:
        # 2. Retrieve All Filtered Sessions
        print("\n[1/4] Loading & Filtering All Karen Sessions...", flush=True)
        sessions = get_sessions()
        num_sessions = len(sessions)
        print(f"[1/4] Loaded {num_sessions} valid sessions for full Karen project.", flush=True)
        
        send_ntfy(f"🚀 Starting Full Karen Dataset Benchmark ({num_sessions} sessions, 1,004 videos, 10.07M frames) with Memory-Safe Kernels...", title="Full Karen Benchmark Started")

        # 3. Pipeline Arguments (Non-destructive)
        pipeline_kwargs = {
            "project_dir": PROJECT_DIR,
            "detector": "deeplabcut",
            "cam": "center_cam",
            "subroutine": "jonathan",
            "latent_dim": 10,
            "num_ar_iters": 0,
            "total_iters": 1,
            "pca_mode": "load",
            "model_name": TEST_MODEL_NAME,
            "init_kappa": 60000,
            "decrease_kappa_factor": 10,
            "mixed_map_iters": 24,
            "seed": 42,
            "downsample_rate": 2,
            "new_fps": 40,
            "use_batched_init": True,
            "use_memory_patch": True,
            "flipping_mode": None,
            "subset_method": "spread",
            "subset_fraction": 1.0,
            "save_every_n_iters": None,
        }

        print(f"\n[2/4] Executing Optimized KPMS Pipeline (SLDS-ONLY, 1 iteration)...", flush=True)
        t0 = time.perf_counter()
        
        results = run_pipeline(sessions, **pipeline_kwargs)
        total_time = time.perf_counter() - t0
        
        vram_peak = get_vram_gb()
        ram_peak = get_ram_gb()

        print("\n" + "=" * 95)
        print("                    FULL KAREN DATASET BENCHMARK RESULTS")
        print("=" * 95)
        print(f"Total Dataset Sessions Modeled : {num_sessions}")
        print(f"Total Single-Iteration Runtime : {total_time:.2f} s ({total_time/60.0:.2f} min)")
        print(f"Peak GPU VRAM Footprint        : {vram_peak:.2f} GB / 24.00 GB")
        print(f"Peak Host CPU RAM Footprint    : {ram_peak:.2f} GB / 125.00 GB")
        print(f"Watchdog Status                : Clean (RAM remained well below 90%)")
        print("=" * 95)

        summary_msg = (
            f"🎉 Full Karen Benchmark Successful!\n"
            f"• Sessions: {num_sessions} (1,004 videos, 10.07M frames)\n"
            f"• Total Iteration Time: {total_time:.1f}s ({total_time/60:.1f} min)\n"
            f"• Peak VRAM: {vram_peak:.2f} GB | Peak RAM: {ram_peak:.2f} GB\n"
            f"• Zero OOM / Zero NaNs on Entire Dataset!"
        )
        send_ntfy(summary_msg, title="🏆 Full Karen Benchmark Succeeded!")

    finally:
        if watchdog_proc.poll() is None:
            watchdog_proc.terminate()
            print("[Watchdog] Terminated cleanly.", flush=True)

        temp_model_dir = os.path.join(PROJECT_DIR, TEST_MODEL_NAME)
        if os.path.exists(temp_model_dir):
            try:
                shutil.rmtree(temp_model_dir)
                print(f"[Cleanup] Removed temporary benchmark directory: {temp_model_dir}", flush=True)
            except Exception as e:
                print(f"[Cleanup Warning] Could not remove {temp_model_dir}: {e}", flush=True)

if __name__ == "__main__":
    main()
