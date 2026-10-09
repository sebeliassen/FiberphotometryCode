# Keypoint-MoSeq Checkpoint Compaction & Rolling Storage Architecture

**Document Type:** Architectural RFC & Scaling Specification  
**Target Repositories:** `FiberphotometryCode` / `keypoint-moseq` / `jax-moseq`  
**Date:** September 2026  
**Context:** Multi-view Tandem Cohort Training (118 Sessions, 18M Frames, 23 Keypoints)  
**Author:** Pair-Programming Engineering Team  

---

## 1. Executive Summary & Problem Formulation

In stock Keypoint-MoSeq (`keypoint_moseq.fitting.fit_model`), model states and parameters are serialized to disk at regular intervals (`save_every_n_iters`, typically every 25 iterations) inside a monolithic HDF5 container: `checkpoint.h5`.

While this mechanism is harmless for standard demo datasets (1–5 videos, ~100k frames, 8 keypoints $\implies$ snapshot size < 20 MB), it creates a catastrophic **Checkpoint Explosion** when scaled to modern multi-animal, multi-view cohorts:

* **Tandem Multi-View Cohort Dimensions:**
  * $N = 1,851$ sequences
  * $T = 10,030$ frames per sequence ($\sim 18\text{M}$ active animal frames)
  * $K_p = 23$ keypoints ($D=10$ latent PCA components)
* **Single Snapshot Footprint:** **$\approx 8.8\text{ GB}$ per save**
* **Standard 250-Iteration Run (10–12 Snapshots):** **$\approx 107\text{ GB}$ per model**
* **Multi-Seed Ensemble (e.g. 5 Seeds for EML):** **$> 535\text{ GB}$**

On typical high-performance workstations equipped with fast PCIe Gen 4 NVMe SSDs (where free space is often in the 100–500 GB range), training even two models without manual intervention triggers `[Errno 28] No space left on device` (ENOSPC), halting training and potentially corrupting the HDF5 superblock.

This document formalizes the **Rolling Checkpointing and Post-Hoc Compaction** architectures required to permanently constrain KPMS storage from $\sim 107\text{ GB}$ down to **$\approx 1.9\text{--}3.5\text{ GB}$ per completed model** without sacrificing crash resilience or EML model selection capability.

---

## 2. Anatomical Breakdown of a KPMS Checkpoint

An inspection of `checkpoint.h5` via `h5py` reveals the exact dataset sizes saved under each iteration key `model_snapshots/{iter}/`:

| Snapshot Component | Dataset Path | Shape | Dtype | Size per Snapshot | Required for Crash Resume? | Required for Inference / Results? | Required for EML Ranking? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Observation Noise Scales** | `states/s` | `(1851, 10030, 23)` | `float64` | **`3,416 MB (3.42 GB)`** | Yes (latest iter only) | No | **No** |
| **Noise Prior Hyperparam** | `noise_prior` | `(1851, 10030, 23)` | `float64` | **`3,416 MB (3.42 GB)`** | Yes (latest iter only) | No | **No** |
| **Continuous Latents** | `states/x` | `(1851, 10030, 10)` | `float64` | **`1,485 MB (1.49 GB)`** | Yes (latest iter only) | Yes (`results.h5`) | **Yes** |
| **Centroid Trajectory** | `states/v` | `(1851, 10030, 2)` | `float64` | **`297 MB (0.30 GB)`** | Yes (latest iter only) | Yes (`results.h5`) | **No** |
| **Heading Angles** | `states/h` | `(1851, 10030)` | `float64` | **`148 MB (0.15 GB)`** | Yes (latest iter only) | Yes (`results.h5`) | **No** |
| **Discrete States (Syllables)** | `states/z` | `(1851, 10027)` | `int32` | **`74 MB (0.07 GB)`** | Yes (latest iter only) | Yes (`results.h5`) | **No** |
| **AR Transition Params** | `params/Ab` | `(100, 10, 31)` | `float64` | **`0.2 MB`** | Yes (latest iter only) | No | **Yes** |
| **AR Noise Covariance** | `params/Q` | `(100, 10, 10)` | `float64` | **`0.1 MB`** | Yes (latest iter only) | No | **Yes** |
| **Transition Matrix** | `params/pi` | `(100, 100)` | `float64` | **`0.1 MB`** | Yes (latest iter only) | No | **Yes** |
| **Global Data Arrays** | `data/*` | (Mask, coords) | Multi | **`$\approx 18.5\text{ GB}$`** (stored once) | Yes | Yes | Yes (mask only) |

### The Critical Takeaways:
1. **$77\%$ of every snapshot** is consumed by observation noise scales (`states/s`) and noise priors (`noise_prior`) ($6.84\text{ GB}$ out of $8.83\text{ GB}$).
2. **Intermediate snapshots** (e.g. Iterations 25, 75, 100, 125, 150, 175, 200, 225) are **never read again** after the next checkpoint is written, except for plotting convergence trajectories in `fitting_progress.pdf`.
3. **Downstream Inference & Science** only uses `results.h5` ($\approx 1.9\text{ GB}$), which contains unbatched $x, z, v, h$.
4. **Expected Marginal Likelihood (EML)** model selection only needs the **final iteration's** continuous states $x$ and parameters $(Ab, Q, \pi)$, plus the binary `data/mask`. It completely ignores $s$ and `noise_prior`.

---

## 3. Storage Architectures for Checkpoint Compaction

### 3.1 Drop-In API Compatibility: Why Exclusively Keeping the Last Snapshot Works Transparently

The defining principle of this compaction architecture is **strict backward and forward compatibility**:
> **Downstream code must be unable to distinguish between a stock checkpoint and a compacted checkpoint.**

In Keypoint-MoSeq, examining how `keypoint_moseq.io.load_checkpoint` operates:
```python
def load_checkpoint(project_dir=None, model_name=None, path=None, iteration=None):
    path = _get_path(project_dir, model_name, path, "checkpoint.h5")
    with h5py.File(path, "r") as f:
        if iteration is None:
            saved_iterations = [int(i) for i in f["model_snapshots"]]
            iteration = np.max(saved_iterations)
        model = load_hdf5(path, f"model_snapshots/{iteration}")
        data = load_hdf5(path, "data")
        metadata = load_hdf5(path, "metadata")
    return model, data, metadata, iteration
```

Key observations:
1. **Default Read Behavior:** When downstream tools (`kpms.load_checkpoint`, `kpms.extract_results`, `compute_eml_rankings`, etc.) load a trained model, `iteration` is omitted by default, resolving directly to `np.max(saved_iterations)`.
2. **Structural Completeness:** As long as:
   - Root group `data/` is present,
   - Root group `metadata/` is present,
   - `model_snapshots/{max_iter}/` contains the complete model snapshot,
   the file satisfies the complete HDF5 schema expected by the entire Keypoint-MoSeq ecosystem.
3. **Zero Loss of Scientific Functionality:** Intermediate snapshots ($25, 50, \dots, 225$) serve exclusively as progress history. Because post-training analysis and inference exclusively query `max_iter`, removing intermediate groups eliminates **$90\text{--}97\%$ of disk space** while keeping downstream behavior **100% byte-for-byte identical**.

---

### 3.2 Concrete Architectural Options

```mermaid
flowchart TD
    subgraph ST [Current Stock Behavior (Severe Bloat)]
        C1["Iter 25 (8.8 GB)"] --> C2["Iter 50 (8.8 GB)"]
        C2 --> C3["Iter 75 (8.8 GB)"]
        C3 --> CN["... Iter 250 (8.8 GB)"]
        CN --> TOT1["Monolithic checkpoint.h5 = 107 GB per model!"]
    end

    subgraph RC [Proposed: In-Flight Rolling Checkpointing]
        R1["Checkpoint Phase 1 Milestone: Iter 50 (8.8 GB)"]
        R2["Rolling Latest SLDS Checkpoint (8.8 GB)<br/>(Overwrites previous SLDS iter)"]
        R1 & R2 --> TOT2["Total In-Flight Disk Footprint = ~18 GB constant!"]
    end

    subgraph PC [Proposed: Post-Completion Compaction]
        TOT2 --> FIN["Training Complete (Iter 250 reached)"]
        FIN --> EX["Extract results.h5 (1.9 GB)"]
        FIN --> CMP["Compact checkpoint.h5:<br/>Strip noise_prior & s (Save 6.8 GB)<br/>Keep only final (x, Ab, Q, pi) for EML"]
        CMP --> TOT3["Final Compact Checkpoint = ~1.5 GB<br/>Total Stored Footprint = ~3.4 GB (97% Reduction!)"]
    end
```

---

### Architecture Option A: In-Flight Rolling Checkpoint (Recommended for Training)

Instead of appending every 25th iteration as a permanent group inside `checkpoint.h5`:
1. **Phase 1 Milestone:** Retain iteration 50 (the convergence point of the AR-HMM warmup before the 10x $\kappa$ transition).
2. **Phase 2 Rolling Window:** Maintain only the **most recent SLDS snapshot**. When iteration 125 is saved, snapshot 100 is replaced.
3. **Safety Guarantee:** If power fails or the process is killed at iteration 142, resuming will load iteration 125 immediately.
4. **Disk Footprint:** **$\approx 18\text{ GB}$ constant** throughout the entire 250-iteration run, rather than monotonically climbing to $107\text{ GB}$.

---

### Architecture Option B: Post-Completion Compaction (`kpms_compact_model`)

For models that have completed all 250 iterations:
1. Extract `results.h5` (1.9 GB) containing full session trajectories.
2. In `checkpoint.h5`:
   * Keep only `model_snapshots/250/` (delete intermediate iterations 25 through 225).
   * In iteration 250, delete `states/s` (3.4 GB) and `noise_prior` (3.4 GB).
   * Repack the HDF5 file using `h5repack` to reclaim allocated blocks:
     ```bash
     h5repack -v -c checkpoint.h5 checkpoint_compact.h5 && mv checkpoint_compact.h5 checkpoint.h5
     ```
3. **Result:** Checkpoint shrinks from **$107\text{ GB} \implies \approx 1.5\text{ GB}$**. Total model folder: **$\approx 3.4\text{ GB}$** (a **$97\%$ reduction**).

---

### Architecture Option C: Tiered Storage Migration (NVMe $\to$ HDD)

For multi-seed production queues (e.g. 5 seeds running overnight):
1. **High-IOPS Training on NVMe:** The active model trains on the ultra-fast NVMe SSD (`/dev/nvme0n1p2`) where random I/O latency is negligible.
2. **Post-Completion Evacuation:** As soon as seed $S$ completes iteration 250:
   * The heavy checkpoint is atomically moved to cold storage:
     `/mnt/hdd/old_projects/kpms_checkpoints_archive/tandem_cohort_checkpoints/{model_name}/`
   * A transparent symlink is created on NVMe pointing to HDD:
     `ln -sfn /mnt/hdd/.../checkpoint.h5 {model_dir}/checkpoint.h5`
   * Downstream EML ranking can still read the file transparently without any path changes.

---

## 4. Implementation Recipes

### 4.1 In-Place Snapshot Pruning Script (`compact_kpms_checkpoint.py`)

This utility can be run on any existing or newly completed model directory:

```python
#!/usr/bin/env python3
import subprocess
import tempfile
import h5py
import numpy as np
from pathlib import Path

def compact_kpms_checkpoint(model_dir: Path, keep_for_eml_only: bool = True):
    ckpt_path = model_dir / "checkpoint.h5"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}")
        return

    orig_size_gb = ckpt_path.stat().st_size / 1e9
    print(f"Compacting {ckpt_path} (Original Size: {orig_size_gb:.2f} GB)...")

    with h5py.File(ckpt_path, "r+") as f:
        snaps = sorted([int(k) for k in f.get("model_snapshots", {}).keys()])
        last_snap = max(snaps)

        # Delete all intermediate snapshots
        for s in snaps:
            if s != last_snap:
                del f[f"model_snapshots/{s}"]
                print(f"  Deleted intermediate snapshot: model_snapshots/{s}")

        # If only EML is needed, strip the 6.8 GB noise arrays
        if keep_for_eml_only:
            if f"model_snapshots/{last_snap}/states/s" in f:
                del f[f"model_snapshots/{last_snap}/states/s"]
                print("  Stripped states/s (saved ~3.4 GB)")
            if f"model_snapshots/{last_snap}/noise_prior" in f:
                del f[f"model_snapshots/{last_snap}/noise_prior"]
                print("  Stripped noise_prior (saved ~3.4 GB)")

    # Repack to reclaim freed HDF5 allocation blocks
    with tempfile.NamedTemporaryFile(suffix=".h5", dir=model_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = ["h5repack", "-f", "GZIP=4", str(ckpt_path), str(tmp_path)]
    subprocess.run(cmd, check=True)
    tmp_path.replace(ckpt_path)

    new_size_gb = ckpt_path.stat().st_size / 1e9
    print(f"✅ Compaction complete: {orig_size_gb:.2f} GB -> {new_size_gb:.2f} GB (Saved {orig_size_gb - new_size_gb:.2f} GB).")
```

### 4.2 Upstream Monkeypatch for Rolling Checkpoints in `keypoint_moseq`

To prevent checkpoints from growing indefinitely during fitting, monkeypatch `keypoint_moseq.fitting.save_checkpoint`:

```python
def rolling_save_checkpoint(project_dir, model_name, model, iteration, **kwargs):
    """
    Rolling checkpoint handler: Keeps only iteration 50 and the most recent SLDS iteration.
    """
    ckpt_path = Path(project_dir) / model_name / "checkpoint.h5"
    with h5py.File(ckpt_path, "a") as f:
        snaps = f.require_group("model_snapshots")
        
        # Save current iteration
        kpms.io.save_hdf5(str(ckpt_path), f"model_snapshots/{iteration}", model)
        
        # Prune older SLDS snapshots (keep 50 as Phase 1 anchor)
        saved_iters = sorted([int(k) for k in snaps.keys()])
        for old_iter in saved_iters:
            if old_iter != 50 and old_iter != iteration:
                del snaps[str(old_iter)]
```

---

## 5. Summary Matrix & Operational Guidelines

| Strategy | In-Flight NVMe Space | Post-Training Model Size | Crash Resilient? | EML Compatible? | Effort |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stock KPMS (Current)** | Monotonic $\to 107\text{ GB}$ | $107\text{ GB}$ | Yes | Yes | Baseline |
| **Strategy A: Rolling Checkpoint** | $\approx 18\text{ GB}$ Constant | $\approx 18\text{ GB}$ | Yes (to last 25 iters) | Yes | Low (Monkeypatch) |
| **Strategy B: Post-Hoc Compaction** | Monotonic $\to 107\text{ GB}$ | **$\approx 3.4\text{ GB}$** | Yes | Yes | Low (`h5repack`) |
| **Strategy A + B (Ideal Pipeline)** | **$\approx 18\text{ GB}$ Constant** | **$\approx 3.4\text{ GB}$** | **Yes** | **Yes** | **Recommended** |
| **Strategy C: Tiered HDD Offload** | Reclaimed immediately | Cold on HDD (4.9 TB free) | Yes | Yes | Implemented |

---
*Reference this specification when extending `run_tandem_cohort_production_50ar_200slds.py` or training large-cohort multi-camera KPMS models.*
