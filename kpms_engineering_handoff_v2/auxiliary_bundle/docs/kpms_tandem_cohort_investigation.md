# KPMS Tandem Full Cohort Training: Investigation, Diagnosis & Non-Trivialities

## 1. Executive Context & Objective

We are preparing to train a **production Keypoint-MoSeq (KPMS)** model (50 AR + 200 SLDS iterations) on the **full tandem cohort** of recordings across Top and Bottom cameras:
* **Dataset Scope:** 118 physical tandem recording sessions across 4 behavioral arenas (`tl`, `bl`, `br`, `tr`), totaling **24,111,516 frames** with 23 fused 2D keypoints (combining dorsal and ventral views with median filtering, velocity-based outlier rejection, and coordinate transformation).
* **Target Hardware:** Single NVIDIA GeForce RTX 4090 (24,564 MiB VRAM).
* **Target Environment:** `/home/fsp585/miniforge3/envs/kpms_gpu_cuml` (contains CUDA-enabled JAX, cuML, PyTorch, and Keypoint-MoSeq).

Before committing to a multi-day 50+200 iteration production job, we initiated a test run (2 AR + 2 SLDS) on the full dataset to verify that batch sizing, GPU memory allocations, and disk I/O operate cleanly.

During initialization of the full cohort, a GPU Out-Of-Memory (`RESOURCE_EXHAUSTED`) exception was triggered.

---

## 2. Root Cause Analysis: The 6.8 GB JAX PCA Allocation Crash

### The Observed Error
```text
jax._src.source_info_util.JaxStackTraceBeforeHash: jax.errors.SimplifiedTraceback: 
For simplicity, JAX has removed its internal frames from the traceback...
...
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 6832115040 bytes.
Buffer assignment:
  ...
  size 6832115040 (6.36GiB)
```

### Where It Occurs
In the standard KPMS pipeline script (`run_tandem_cohort_production_50ar_200slds.py`), the initialization sequence is:
```python
# data is loaded and formatted:
# data['Y'].shape = (1851, 10030, 23, 2)   # 1,851 sequence segments of length ~10k
pca = kpms.fit_pca(**data, **config)
```

### The Mechanism Inside `keypoint_moseq`
Examining `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/lib/python3.10/site-packages/jax_moseq/models/keypoint_slds/alignment.py`:
```python
def fit_pca(Y, mask, anterior_idxs=None, posterior_idxs=None, conf=None, ...):
    Y_flat = preprocess_for_pca(
        Y, anterior_idxs, posterior_idxs, conf, conf_threshold, fix_heading, verbose
    )[0]
    ...
    return utils.fit_pca(Y_flat, pca_mask, PCA_fitting_num_frames, verbose)
```

And inside `preprocess_for_pca`:
```python
def preprocess_for_pca(Y, anterior_idxs, posterior_idxs, conf=None, ...):
    if conf is not None:
        Y = interpolate(Y, outliers).block_until_ready()
    if fix_heading: ...
    else:
        Y_aligned, v, h = align_egocentric(Y, anterior_idxs, posterior_idxs)
    Gamma_inv = center_embedding(k).T
    Y_embedded = Gamma_inv @ Y_aligned
    Y_flat = Y_embedded.reshape(*dims, (k - 1) * d)
    return Y_flat, v, h
```

### The Math of the Failure
1. **Full Tensor Size:**
   $$\text{Total Elements} = 1{,}851 \text{ seqs} \times 10{,}030 \text{ frames} \times 23 \text{ keypoints} \times 2 \text{ dims} = 854{,}011{,}980 \text{ float64 elements}$$
   $$854{,}011{,}980 \times 8 \text{ bytes} \approx 6{,}832{,}095{,}840 \text{ bytes} = 6.36 \text{ GiB}$$
2. **Intermediate Allocations on GPU:**
   When `preprocess_for_pca` executes on JAX GPU:
   * It takes the monolithic `Y` array ($6.36 \text{ GiB}$).
   * In `interpolate(Y, outliers)`, it creates boolean masks and temporary copies.
   * In `align_egocentric(Y)`, it computes centroids $v$, heading angles $h$, rotation matrices $R(h)$, and transforms `Y` into `Y_aligned` (another $6.36 \text{ GiB}$).
   * In `Gamma_inv @ Y_aligned`, it projects into centered keypoint coordinates (another $6.08 \text{ GiB}$).
   * **Cumulative peak memory requirement:** $> 20\text{--}24 \text{ GiB}$ in transient JAX buffers.
3. **The JAX VRAM Preallocation Clash:**
   By default, JAX attempts to pre-allocate $75\%\text{--}90\%$ of total GPU VRAM upon startup. If any other process or prior allocation exists, or when multiple large intermediate arrays are allocated simultaneously in one XLA graph, JAX fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 6832115040 bytes`.

---

## 3. Key Non-Trivialities & Discoveries

### Non-Triviality 1: PCA Does NOT Need 18 Million Frames
* Look closely at `utils.fit_pca`:
  ```python
  def fit_pca(Y_flat, pca_mask, PCA_fitting_num_frames=1000000, verbose=False):
      # It randomly samples PCA_fitting_num_frames (default 10,000 to 1,000,000)
      # to compute the top 10 principal components!
  ```
* In reality, fitting the static 10-dimensional posture subspace ($D=10$ PCA) needs at most **$10{,}000$ to $50{,}000$ representative frames** sampled across the cohort.
* Passing all $18{,}000{,}000$ frames through `preprocess_for_pca` on GPU is purely wasteful: it forces JAX to perform full-dataset coordinate rotation and center embedding on the GPU for millions of frames that will immediately be discarded by `sklearn.decomposition.PCA`!

### Non-Triviality 2: Why SLDS Initialization Didn't Fail First
* `batched_init_model_by_video(data, pca, config)` was carefully written by the authors to process session sequences in chunks of `batch_size = 20` and stream state to CPU RAM.
* BUT `kpms.fit_pca` is an unbatched monolithic call. Because previous single-session or small-project datasets only had 200,000–500,000 frames (where $Y$ is $< 200$ MB), this monolithic GPU dispatch never crashed before. On a massive 118-session production cohort (18M frames), it blows up instantly.

### Non-Triviality 3: NumPy 2.x / `np.bool8` Incompatibility
* The environment `/home/fsp585/miniforge3/envs/kpms_gpu_cuml` has a modern NumPy version where `np.bool8` was deprecated and removed in favor of `np.bool_`.
* When `keypoint_moseq` imports `bokeh.core.property.primitive`, Bokeh attempts to reference `np.bool8`, throwing:
  `AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?`
* **Critical Rule:** Any script importing `keypoint_moseq` in this environment must patch NumPy before import:
  ```python
  import numpy as np
  if not hasattr(np, 'bool8'):
      np.bool8 = np.bool_
  import keypoint_moseq as kpms
  ```

### Non-Triviality 4: Empty Chamber Auto-Exclusion & `auto_mask.npz`
* 6 chambers in the full 118-session cohort had 0% animal presence (e.g. empty chamber recording setups or animals not placed):
  * `2025-12-04T11_41_50_bottom_left_tandem_23kp`
  * `2025-12-04T11_41_50_bottom_right_tandem_23kp`
  * `2025-12-04T11_41_50_top_left_tandem_23kp`
  * `2025-12-04T11_41_50_top_right_tandem_23kp`
  * `2025-12-04T12_53_45_bottom_right_tandem_23kp`
  * `2025-12-04T12_53_45_top_right_tandem_23kp`
* **Behavior:** `presence_mask_from_conf` and `filter_with_masks` automatically detect that these sessions have 0 valid frames, prune them, and save valid frame indices into `video_frame_indexes.npz`.
* **Symlink Resolution:** Downstream pipelines (e.g. `enrich.py` or legacy cleaning notebooks) look for `auto_mask.npz`. Creating a symlink `auto_mask.npz -> video_frame_indexes.npz` guarantees 100% downstream compatibility without any manual edits.

### Non-Triviality 5: GPU Batching during AR & SLDS Training
* In `recursive_batching.py`:
  * JAX uses dynamic batch configs: `discrete_bs` (for HMM transitions) and `continuous_chunk_size` (for kalman filtering / SLDS updates).
  * `rb.precompute_optimal_batch_config(data, verbose=True)` determines the largest power-of-2 chunk size that fits into RTX 4090 VRAM.
  * For 23 keypoints ($K=23$) and $D=10$, standard batch sizes are typically `continuous_chunk_size = 1000` to `2000`, which consume $\sim 4\text{--}8$ GB of VRAM during model iteration steps.

---

## 4. The Clean, Mathematically Sound Fix

Instead of passing the entire $1{,}851$-sequence dataset directly into `kpms.fit_pca`:
1. **Subsample Representative Sequences for PCA:**
   * Select a stratified subset of sequences (e.g. 10 to 20 active sessions, or sampling evenly across all sessions) totaling $\sim 50{,}000\text{--}100{,}000$ frames.
   * Or pass `subset_data = {k: v[:20] for k, v in data.items()}` into `kpms.fit_pca(..., PCA_fitting_num_frames=50000)`.
   * Size of $Y_{\text{subset}}$: $20 \times 10{,}030 \times 23 \times 2 \times 8 \text{ bytes} \approx 73 \text{ MB}$ (instead of $6.8 \text{ GB}$!).
   * This executes `preprocess_for_pca` on GPU in $< 0.2$ seconds with negligible memory footprint.
2. **Reuse Existing Pre-Fit PCA If Available:**
   * The PCA basis represents the egocentric posture space spanned by the 23 keypoints.
   * If `pca.p` was already computed for this exact 23-keypoint configuration, it can be loaded directly via `joblib.load('pca.p')` or `kpms.load_pca(project_dir)`, ensuring exact alignment across runs.

---

## 5. Verification & Execution Roadmap

1. **Step 1: Test Script (`test_2ar_2slds_cohort_full.py`)**
   * Apply `data_subset_for_pca` fix.
   * Run 2 AR + 2 SLDS on all 112 active sessions.
   * Log GPU memory usage via `nvidia-smi` to establish actual VRAM baseline during SLDS sampling.
2. **Step 2: Inspect Convergence & State Saving**
   * Verify that checkpoints `model_state.p` and `results.p` write cleanly.
3. **Step 3: Launch Full 50 AR + 200 SLDS Production**
   * Launch detached via `nohup` with `PYTHONUNBUFFERED=1`.
   * Real-time iteration updates sent to `https://ntfy.sh/woody_videos_seb`.
   * Background idle watchdog monitor active to alert upon completion.
