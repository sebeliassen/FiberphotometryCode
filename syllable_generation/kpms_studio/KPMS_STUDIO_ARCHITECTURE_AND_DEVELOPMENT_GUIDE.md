# KPMS Studio: Architecture, Scientific Workflow & Development Guide
**State-of-the-Art Keypoint-MoSeq Project Studio, Calibration & Ethological Labeler**  
*Environment:* `kpms_gpu_cuml` | *GPU Acceleration:* NVIDIA GeForce RTX 4090 | *Location:* `syllable_generation/kpms_studio`

---

## 1. Executive Summary & Purpose

**KPMS Studio** is an interactive, unified GUI and API platform for Keypoint-MoSeq (KPMS). It replaces ad-hoc terminal scripts, rigid organism assumptions, and manual config editing with a streamlined, full-lifecycle scientific workflow.

### Core Philosophy
1. **100% Native KPMS Under the Hood:** Use official Keypoint-MoSeq library routines (`kpms.fit_pca`, `kpms.load_pca`, `kpms.setup_project`, `kpms.estimate_sigmasq_loc`, `kpms.reindex_syllables_in_checkpoint`, `kpms.generate_grid_movies`, `kpms.plot_similarity_dendrogram`) rather than reimplementing algorithms.
2. **Organism Agnostic:** Complete eradication of rodent-specific assumptions (no fixed 8/16/27-KP badges or hardcoded mouse stick figures). The studio dynamically parses and animates keypoints and skeletons for squids, rodents, flies, worms, or primates.
3. **Rigorous Frame Alignment:** Disallow fractional downsampling (e.g. 1.67x) that causes frame index drift; strictly enforce integer temporal strides ($1\times, 2\times, 3\times, 4\times\dots$).
4. **Focused Complexity:** Maintain clean, single-responsibility tabs with rich, dark-mode aesthetics rather than overloading any single interface with 50 disparate controls.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Web Browser (Client)                          │
│  - Vanilla ES6 JavaScript (app.js) with reactive canvas rendering     │
│  - Modern Glassmorphism CSS Design System (app.css)                   │
│  - 6 Sequential Workflow Stage Views (index.html)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST / JSON & Media Streaming
┌───────────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend (server.py)                         │
│  - Bound to 0.0.0.0:8000 (accessible over SSH / LAN / Mac browser)    │
│  - NumPy 2.x Compatibility Layer (np.bool8 shim)                       │
│  - JAX Memory Governor (XLA_PYTHON_CLIENT_PREALLOCATE=false)           │
└───────┬──────────────┬────────────────┬────────────────┬───────────────┘
        │              │                │                │
┌───────▼──────┐┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼──────────────┐
│  Data & Video ││ Keypoints &  │ │ PCA & Latent │ │ Model Fitting &    │
│  Inspection   ││ Anatomy      │ │ Dimension    │ │ Checkpoint Tools   │
│  (OpenCV/DLC) ││ (Dynamic KP) │ │ (kpms.PCA)   │ │ (Gibbs / AR-HMM)   │
└──────────────┘└──────────────┘ └──────────────┘ └─────────────────────┘
```

---

## 3. The 6-Stage Scientific Pipeline

The studio layout is organized into 6 distinct, sequential workflow stages matching the official Keypoint-MoSeq methodology:

### Stage 1: Videos & DLC Alignment (`#sessions`)
- **OpenCV Native FPS Detection:** Reads raw behavioral video container headers via `cv2.VideoCapture` to detect exact native frame rates (e.g., 60.0 FPS, 80.0 FPS), resolution, and recording counts.
- **Strict Integer Downsampling:** Prohibits fractional downsampling factors (e.g. 1.67x). Resulting model FPS is computed as $\lfloor \text{native\_fps} / k \rfloor$ where $k \in \{1, 2, 3, 4, 5, 6\}$.
- **Outlier Keypoint Removal:** Exposes `kpms.outlier_removal` parameters (`outlier_scale_factor: 6.0`) to filter out large tracking jumps based on distance to the animal's medoid prior to PCA and modeling.
- **Superset Filename Auto-Matching:** Automatically pairs video files (`Arena1234_..._bottom_left.avi`) to DLC tracking outputs (`Arena1234_..._bottom_leftDLC_HrnetW32_..._snapshot_020.h5`) by testing if the DLC stem is a superset of the video stem.
- **Correspondence Export:** Generates and saves `video_mapping.csv` for downstream grid movie generation (`kpms.generate_grid_movies`).

### Stage 2: Keypoints & Anatomy (`#anatomy`)
- **Organism-Agnostic Anatomy:** Reads bodyparts and skeleton topology directly from DLC tracking headers or `config.yml`.
- **Dynamic Modeling Subset (`use_bodyparts`):** Allows selecting any subset of keypoints with 1-click batch actions (`Select All`, `Clear All`, `Invert Selection`).
- **Heading Orientation Anchors:** Dynamic dropdown selectors for Anterior (rostral/head) and Posterior (caudal/tail) keypoints, used by KPMS to orient animal forward heading and perform rotational centering.
- **Skeleton Connectivity Visualizer:** Renders active connection edges defined in `config.skeleton`.

### Stage 3: PCA & Latent Dimension Studio (`#pca`)
- **Native PCA Fitting:** Executes `kpms.fit_pca(**data, **config)` on aligned keypoint coordinates and saves the model to `{project_dir}/pca.p`.
- **Interactive Dual-Axis Scree Plot:**
  - Individual variance explained per PC (vertical bars).
  - Cumulative variance explained (emerald curve).
  - Gold dashed reference line at 90% variance threshold.
- **KPMS Official Heuristic Calculator:** Evaluates dimensions needed to explain $\ge 90\%$ variance, applying the rule: $\min(\text{dims}_{\ge 90\%}, 10)$.
- **Live Latent Dimension Slider:** Allows real-time inspection of cumulative variance at any dimension with a 1-click "Apply Recommended" button.
- **PC Eigen-Pose Gallery (`kpms.plot_pcs`):** Renders the spatial perturbation of each PC relative to the mean pose, converted from `pcs-xy.pdf` to crisp PNG via `pdftoppm`.

### Stage 4: Kappa (κ) Scout & Model Fitting (`#calibration`)
- **Target Timescale Slider:** Controls target median syllable duration in frames (8 to 28 frames) and milliseconds.
- **Log-Space Bisection Search:** Performs fast bisection search on a small subset (e.g., 5% of frames) to find optimal $\kappa_{\text{scout}}$.
- **Analytical Dataset Upscaling:** Automatically scales kappa to full cohort size using the validated empirical relation:
  $$\kappa_{\text{prod}} = \kappa_{\text{scout}} \cdot \left(\frac{N_{\text{total}}}{N_{\text{subset}}}\right)^{0.90}$$
- **Hyperparameter Estimation:** Integrates `kpms.estimate_sigmasq_loc` to set the centroid movement prior from empirical frame-to-frame movement.
- **Gibbs Model Fitting Runner:** Configures AR-HMM iterations (default 50) and Full Model iterations (default 500) with progress tracking.
- **Checkpoint Tools:** 1-click execution of `kpms.reindex_syllables_in_checkpoint` (sorting syllables so #00 is most frequent) and `kpms.extract_results` (writing `results.h5` and CSV exports).

### Stage 5: Discovered Syllables & Ethological Labeling (`#syllables`)
- **Clean Syllable Repertoire Table:** Syllables are labeled purely by raw IDs (`#00`, `#01`, `#02`...) with clean, blank label inputs—avoiding intrusive automated naming.
- **Kinematic Statistics:** Displays frequency %, total frame counts, median durations, centroid speed (px/s), and angular yaw velocity (deg/s).
- **Interactive Sorting:** Instant client-side sorting by usage %, velocity, and duration.
- **Exemplar Modal:** Displays real grid movie clips or animates the active organism skeleton across time.
- **CSV Export:** Exports `syllable_labels.csv` and frame-by-frame mapping `syllables_video_frame_correspondence.csv`.

### Stage 6: KPMS Visualizations & Analysis (`#visualizations`)
- **Syllable Grid Movies (`kpms.generate_grid_movies`):** Embedded HTML5 video player with syllable selector to play generated `.mp4` grid movies.
- **Pose Trajectory Plots (`kpms.generate_trajectory_plots`):** Gallery displaying `all_trajectories.gif` and individual syllable trajectory GIFs/PDFs.
- **Similarity Dendrogram (`kpms.plot_similarity_dendrogram`):** Displays hierarchical clustering of syllable median trajectories (`similarity_dendrogram.png`).

---

## 4. Development Non-Trivialities & Technical Gotchas

During development, several critical non-trivial technical challenges were identified and resolved:

### 1. NumPy 2.x Compatibility Shim (`np.bool8`)
* **Problem:** In NumPy 2.0+, `np.bool8` was deprecated and removed in favor of `np.bool_`. When importing `keypoint_moseq`, legacy transitive dependencies (such as `bokeh.core.property.primitive`) fail with:
  ```
  AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?
  ```
* **Solution:** A compatibility shim must be applied before importing `keypoint_moseq`, `bokeh`, `panel`, or `jax`:
  ```python
  import numpy as np
  if not hasattr(np, "bool8"):
      np.bool8 = np.bool_
  import keypoint_moseq as kpms
  ```

### 2. JAX Aggressive GPU VRAM Pre-Allocation
* **Problem:** Keypoint-MoSeq relies on JAX for GPU acceleration. By default, JAX pre-allocates **75% to 90% of total GPU memory** upon initialization. On an NVIDIA GeForce RTX 4090 (24 GB VRAM), the server process immediately reserved **18,542 MiB (18.5 GB)**, starving other GPU jobs such as running DeepLabCut analysis.
* **Solution:** Configure JAX memory allocation behavior before importing JAX or Keypoint-MoSeq:
  ```python
  import os
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"  # Limit to 15% VRAM
  ```
  Or for lightweight browsing/labeling without GPU fitting, launch with CPU fallback:
  ```bash
  CUDA_VISIBLE_DEVICES="" python server.py
  ```

### 3. Native KPMS Config Hierarchy & `latent_dim` Placement
* **Problem:** In standard Keypoint-MoSeq `config.yml`, `latent_dim` does not reside at the top-level root; it is nested inside the `ar_hypparams` subdictionary:
  ```yaml
  ar_hypparams:
    latent_dim: 10
    nlags: 3
    K_0_scale: 10.0
  ```
  Modifying a root-level `latent_dim` key directly using generic YAML dumpers causes KPMS model initialization to ignore the update.
* **Solution:** Always invoke native `kpms.update_config`:
  ```python
  kpms.update_config(str(project_path), latent_dim=selected_dim)
  ```
  KPMS's internal `generate_config` automatically maps `latent_dim` to `ar_hypparams['latent_dim']`.

### 4. Strictly Integer Downsampling vs. Fractional Downsampling
* **Problem:** Downsampling behavioral video by fractional ratios (e.g. $60\text{ FPS} / 1.67 \approx 36\text{ FPS}$) results in fractional time steps ($1.67, 3.33, 5.0\dots$) that cannot map to discrete video frame indices ($0, 1, 2\dots$), breaking video cropping and syllable grid movie alignment.
* **Solution:** Restrict downsampling factors strictly to integers:
  $$k \in \{1, 2, 3, 4, 5, 6\}$$
  $$\text{Model FPS} = \left\lfloor \frac{\text{Native FPS}}{k} \right\rfloor$$
  Every $k$-th video frame corresponds exactly to frame index $i \times k$.

### 5. DeepLabCut Filename Superset Matching
* **Problem:** DeepLabCut tracking exports append extensive metadata tags to video base names (e.g., `Arena1234_CenterTopCamVid_2024-12-13T12_20_06_bottom_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020.h5`), making exact string equality comparisons with raw video files fail.
* **Solution:** Use prefix superset matching:
  ```python
  video_stem = Path(video_file).stem
  matched_dlc = None
  for dlc_file in dlc_files:
      if dlc_file.name.startswith(video_stem):
          matched_dlc = dlc_file.name
          break
  ```
  This resolves 100% of standard DLC naming schemes while still offering manual override in the UI.

### 6. Headless Matplotlib & Vector PDF Conversion (`pdftoppm`)
* **Problem:** Native KPMS plotting routines (`plot_pcs`, `plot_scree`, `plot_similarity_dendrogram`) save output as vector `.pdf` files. Web browsers cannot display multi-page or vector PDFs directly inside `<img>` tags, and calling `plt.show()` in a background server triggers non-interactive GUI backend warnings or X11 crashes.
* **Solution:**
  1. Force the non-interactive Agg backend:
     ```python
     import matplotlib
     matplotlib.use("Agg")
     ```
  2. Use the fast Linux system utility `pdftoppm` to convert output PDFs to high-resolution PNGs on demand:
     ```bash
     pdftoppm -png -r 150 pcs-xy.pdf pcs-xy
     ```
     The resulting `pcs-xy-1.png` is served immediately to the frontend.

### 7. High-DPI Canvas Scaling & Layout Stretching
* **Problem:** On modern high-DPI (Retina) screens, HTML `<canvas>` elements appear blurred if rendered at 1x resolution, or stretched if CSS dimensions and canvas pixel buffer dimensions fall out of sync.
* **Solution:** Dynamically scale canvas buffer by `window.devicePixelRatio` and enforce fixed CSS dimensions on the container:
  ```javascript
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  ```
  Paired with CSS:
  ```css
  .canvas-wrapper canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  ```

### 8. Remote Headless SSH Server Binding
* **Problem:** Binding FastAPI to `localhost` or `127.0.0.1` prevents users connected via local network or SSH tunneling on client machines (e.g. MacBook connecting to workstation) from reaching the service.
* **Solution:** Bind FastAPI server to `0.0.0.0:8000`:
  ```bash
  uvicorn server:app --host 0.0.0.0 --port 8000
  ```
  Accessible via browser on `http://<workstation-ip>:8000` or through an SSH tunnel:
  ```bash
  ssh -L 8000:localhost:8000 user@<workstation-ip>
  ```

---

## 5. Summary of GPU Processes & Current Status

* **DeepLabCut Production Analysis (`envs/dlc3_0_1`):**
  - **PID:** `326706` (wrapped in `/usr/bin/systemd-inhibit`)
  - **Script:** `/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/run_production_analysis.py`
  - **Status:** **RUNNING & ACTIVELY PROCESSING** (consuming 98% GPU compute, ~363W power).
  - **Progress:** 84% complete (~186,000 / 221,505 frames at ~75 it/s; ~7–8 minutes remaining).
  - **Safety:** **Completely untouched and isolated.**

* **KPMS Studio Backend (`envs/kpms_gpu_cuml`):**
  - **Previous State:** PID `456110` held ~18.5 GB VRAM due to default JAX pre-allocation.
  - **Current State:** **Cleanly terminated.** Total GPU memory dropped back to 1,250 MiB. Full 18.5 GB VRAM has been reclaimed.
  - **Future Launch Safeguard:** Added `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` to both `server.py` and `launch_gui.py`. Future runs will only consume a tiny fraction of VRAM.

---

## 6. How to Relaunch KPMS Studio Later

When you are ready to resume work with KPMS Studio:

1. **Quick Launch via Terminal:**
   ```bash
   cd /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/kpms_studio
   ./launch.sh
   ```
   Or explicitly using the conda environment:
   ```bash
   /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. **Accessing the Interface:**
   - On the workstation: Open `http://localhost:8000`
   - Over Local Network / SSH: Open `http://<workstation-ip>:8000` or forward port 8000 (`ssh -L 8000:localhost:8000 user@host`).

