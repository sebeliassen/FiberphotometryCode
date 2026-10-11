#!/usr/bin/env python
"""
Sweep batch sizes to find the maximum that avoids GPU OOM.

Architecture:
  - The ORCHESTRATOR (parent) has NO JAX/CUDA imports at all.
  - Each TRIAL subprocess is fully self-contained:
    imports JAX, loads the checkpoint, applies patches for Z, S, and X,
    and runs 1 fit iteration.
"""
import os
import subprocess
import sys
import time
import urllib.request

NTFY_URL = "https://ntfy.sh/woody_videos_seb"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def ntfy(title, msg):
    try:
        req = urllib.request.Request(
            NTFY_URL, data=msg.encode("utf-8"), headers={"Title": title}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[ntfy failed] {e}")

# =============================================================================
# TRIAL SCRIPT TEMPLATE
# =============================================================================
TRIAL_SCRIPT_TEMPLATE = r'''
import os, sys, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
if "{mem_fraction}" != "1.0":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "{mem_fraction}"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, "{script_dir}")

import keypoint_moseq as kpms
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.arhmm import gibbs as arhmm_gibbs
from jax_moseq.models.keypoint_slds import gibbs as slds_gibbs
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error

print(f"JAX devices: {{jax.devices()}}")

BATCH_SIZE = {discrete_bs}
IS_TEST = {is_test_flag}

# -----------------------------------------------------------------------------
# 1. Patch resample_discrete_stateseqs (Z - Discrete States)
# -----------------------------------------------------------------------------
_orig_z = arhmm_gibbs.resample_discrete_stateseqs

def _patched_z(seed, x, mask, Ab, Q, pi, **kw):
    N = x.shape[0]
    bs = N if (BATCH_SIZE <= 0 or BATCH_SIZE >= N) else BATCH_SIZE
    parts = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        z = _orig_z(jr.fold_in(seed, i), x[i:end], mask[i:end], Ab, Q, pi, **kw)
        z.block_until_ready()
        parts.append(z)
    return jnp.concatenate(parts, axis=0)

arhmm_gibbs.resample_discrete_stateseqs = _patched_z


# -----------------------------------------------------------------------------
# 2. Patch resample_scales (S - Local Noise Scales)
# -----------------------------------------------------------------------------
SCALES_BS = min(BATCH_SIZE, 50) if BATCH_SIZE > 0 else -1

@jax.jit
def _scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

def _patched_s(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kw):
    N = Y.shape[0]
    bs = N if (SCALES_BS <= 0 or SCALES_BS >= N) else SCALES_BS
    parts = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_seed = jr.fold_in(seed, i)
        s0_b = s_0[i:end] if (hasattr(s_0, "shape") and len(s_0.shape) >= 3 and s_0.shape[0] == N) else s_0
        s = _scales_kernel(batch_seed, Y[i:end], x[i:end], v[i:end], h[i:end], Cd, sigmasq, nu_s, s0_b)
        s.block_until_ready()
        parts.append(s)
    return jnp.concatenate(parts, axis=0)

slds_gibbs.resample_scales = _patched_s


# -----------------------------------------------------------------------------
# 3. Patch resample_continuous_stateseqs (X - Continuous Trajectories / Kalman)
# -----------------------------------------------------------------------------
_orig_x = slds_gibbs.resample_continuous_stateseqs

def _patched_x(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, **kw):
    N = Y.shape[0]
    bs = N if (BATCH_SIZE <= 0 or BATCH_SIZE >= N) else BATCH_SIZE
    parts = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_seed = jr.fold_in(seed, i)
        x_part = _orig_x(
            batch_seed,
            Y[i:end],
            mask[i:end],
            v[i:end],
            h[i:end],
            s[i:end],
            z[i:end],
            Cd,
            sigmasq,
            Ab,
            Q,
            **kw
        )
        x_part.block_until_ready()
        parts.append(x_part)
    return jnp.concatenate(parts, axis=0)

slds_gibbs.resample_continuous_stateseqs = _patched_x

print(f"[Patches Active] Z-batch={{BATCH_SIZE}}, S-batch={{SCALES_BS}}, X-batch={{BATCH_SIZE}}")

# --- Load checkpoint ---
project_path = "{project_path}"
model_name = "{model_name}"

print(f"Loading checkpoint: {{project_path}}/{{model_name}}")
model, data, metadata, current_iter = kpms.load_checkpoint(project_path, model_name, iteration=1)
model = kpms.update_hypparams(model, kappa=1e4)

if IS_TEST:
    print("[Test Mode] Slicing loaded checkpoint to first 10 sequences.")
    N_orig = data["Y"].shape[0]
    N_test = 10
    data = {{k: (v[:N_test] if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == N_orig else v) for k, v in data.items()}}
    model["states"] = {{k: (v[:N_test] if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == N_orig else v) for k, v in model["states"].items()}}
    for k, v in model.items():
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == N_orig:
            model[k] = v[:N_test]

N = data["Y"].shape[0]
T = data["Y"].shape[1]
print(f"Data: N={{N}} sequences, T={{T}} frames")

# --- Run 1 full SLDS iteration ---
print("Running 1 full SLDS iteration...")
t0 = time.time()
model = kpms.fit_model(
    model, data, metadata, project_path, model_name,
    ar_only=False, start_iter=current_iter, num_iters=current_iter+1,
    parallel_message_passing=False, generate_progress_plots=False
)[0]
elapsed = time.time() - t0
print(f"SUCCESS in {{elapsed:.1f}}s")
'''

def run_orchestrator(is_test):
    prefix = "[TEST]" if is_test else "[PROD]"
    mem_frac = "1.0"

    project_path = os.path.join(SCRIPT_DIR, "karen_project_2")

    model_dirs = sorted([
        d for d in os.listdir(project_path)
        if os.path.isfile(os.path.join(project_path, d, "checkpoint.h5"))
    ])
    if not model_dirs:
        msg = "No checkpoint found in karen_project_2/"
        print(msg)
        ntfy(f"{prefix} Error", msg)
        return

    model_name = model_dirs[-1]
    print(f"Using checkpoint: {project_path}/{model_name}")

    if is_test:
        batch_sizes = [4, 2]
    else:
        batch_sizes = [-1, 1000, 800, 600, 400, 200, 100, 50]

    ntfy(f"{prefix} Sweep Started",
         f"Batch sizes: {batch_sizes}\nCheckpoint: {model_name}")

    trial_path = os.path.join(SCRIPT_DIR, "_oom_trial.py")
    results = []

    for bs in batch_sizes:
        bs_label = "UNBATCHED" if bs == -1 else str(bs)
        print(f"\n{'='*60}")
        print(f"TRIAL: batch_size={bs_label}")
        print(f"{'='*60}")

        script_content = TRIAL_SCRIPT_TEMPLATE.format(
            mem_fraction=mem_frac,
            script_dir=SCRIPT_DIR,
            discrete_bs=bs,
            project_path=project_path,
            model_name=model_name,
            is_test_flag=str(is_test)
        )
        with open(trial_path, "w") as f:
            f.write(script_content)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", trial_path],
            capture_output=True, text=True,
            timeout=600
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            msg = f"SUCCESS bs={bs_label} in {elapsed:.1f}s"
            print(msg)
            print(f"STDOUT:\n{proc.stdout[-500:]}")
            ntfy(f"{prefix} SUCCESS bs={bs_label}", msg)
            results.append((bs, "OK", elapsed))
            break
        else:
            is_oom = "RESOURCE_EXHAUSTED" in proc.stderr or "out of memory" in proc.stderr.lower()
            tag = "OOM" if is_oom else "CRASH"

            err_lines = [l for l in proc.stderr.splitlines()
                         if "RESOURCE_EXHAUSTED" in l or "Error" in l or "TypeError" in l]
            err_summary = err_lines[0][:200] if err_lines else "Unknown"

            msg = f"FAIL ({tag}) bs={bs_label} after {elapsed:.1f}s\n{err_summary}"
            print(msg)
            print(f"STDERR:\n{proc.stderr}")
            ntfy(f"{prefix} {tag} bs={bs_label}", msg)
            results.append((bs, tag, elapsed))

    if os.path.exists(trial_path):
        os.remove(trial_path)

    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    for bs, status, elapsed in results:
        bs_label = "UNBATCHED" if bs == -1 else str(bs)
        print(f"  bs={bs_label:>10s}  {status:>5s}  {elapsed:6.1f}s")

    successes = [r for r in results if r[1] == "OK"]
    if successes:
        best = successes[0][0]
        best_label = "UNBATCHED" if best == -1 else str(best)
        final_msg = f"Max working batch size: {best_label}"
    else:
        final_msg = "All batch sizes failed."

    print(f"\n{final_msg}")
    ntfy(f"{prefix} Sweep Done", final_msg + "\n" + "\n".join(
        f"  bs={'UNBATCHED' if bs==-1 else bs}: {status} ({elapsed:.0f}s)"
        for bs, status, elapsed in results
    ))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    if not args.test and not args.prod:
        parser.print_help()
        sys.exit(1)

    run_orchestrator(is_test=args.test)
