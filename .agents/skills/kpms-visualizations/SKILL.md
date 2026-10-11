---
name: kpms-visualizations
description: >-
  Quick reference and automated workflows for generating Keypoint-MoSeq (KPMS)
  visualizations, specifically syllable grid movies (with/without density sampling,
  with/without keypoint overlays) and trajectory plots, from either KPMS project
  checkpoints/results or session pickle files (*.pkl).
---

# Keypoint-MoSeq (KPMS) Visualizations Skill

This skill provides verified, copy-pasteable workflows to generate **Syllable Grid Movies** and **Trajectory Plots** from Keypoint-MoSeq data.

---

## 1. Environment & Execution Guidelines

Rendering hundreds of video crops across 30+ sessions requires OpenCV/FFmpeg and CPU memory.
To prevent GPU Out-Of-Memory (OOM) errors and CUDA compiler conflicts (`ptxas`), **always force CPU execution** for movie/plot generation scripts before importing JAX or KPMS:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import keypoint_moseq as kpms
```

Preferred conda environment:
`/home/fsp585/miniforge3/envs/keypoint_moseq/bin/python`

---

## 2. Generating Grid Movies with Built-In KPMS Library

The primary function is `kpms.generate_grid_movies(...)`.

### Function Signature & Key Arguments
```python
kpms.generate_grid_movies(
    results=results,                  # Dict mapping recording names to {'syllable': ...}
    project_dir=str(project_dir),     # Path to KPMS project root
    model_name=model_name,            # Name of the model directory (e.g. 'amph_prod_k120k_run_1')
    output_dir=output_dir,            # Target directory for output mp4 files
    video_paths=video_paths,          # Dict mapping recording keys -> absolute video paths (.avi)
    coordinates=coordinates,          # Dict mapping recording keys -> ndarray (T, K, 2 or 3)
    centroids=centroids,              # Dict mapping recording keys -> ndarray (T, 2)
    headings=headings,                # Dict mapping recording keys -> ndarray (T,)
    video_frame_indexes=video_frame_indexes, # Dict mapping recording keys -> frame indices
    fps=40,                           # Frame rate of model (or new_fps from config)
    rows=4, cols=6,                   # Grid dimensions (default: 4x6 = 24 instances)
    min_frequency=0.005,              # Syllables below this frequency are skipped
    min_duration=3,                   # Minimum duration in frames
    overlay_keypoints=False,          # Set to False to exclude keypoint dots/skeletons
    sampling_options=sampling_options,# {'mode': 'random'} vs {'mode': 'density'}
    **config_for_movies               # Bodyparts, use_bodyparts, etc.
)
```

---

## 3. Density Sampling vs. Random Sampling Modes

KPMS provides two instance selection modes via `sampling_options`:

### Mode A: Without Density Sampling (Random Sampling)
Instances are chosen randomly (without replacement) from all qualifying syllable bouts:
```python
sampling_options = {
    "mode": "random"
}
output_dir = project_dir / model_name / "grid_movies_random"
```

### Mode B: With Density Sampling
For each syllable, an exemplar instance is identified that maximizes the syllable-to-background density ratio in PCA trajectory space, and its nearest neighbors are sampled:
```python
sampling_options = {
    "mode": "density",
    "pca_samples": 50000,
    "pca_dim": 4,
    "n_neighbors": 50
}
output_dir = project_dir / model_name / "grid_movies_density"
```

---

## 4. End-to-End Workflow: From KPMS Project Checkpoint

When you have a project directory (e.g. `amph_project`) containing `model_name` with `checkpoint.h5` and `results.h5`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import keypoint_moseq as kpms
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project")
MODEL_NAME = "amph_prod_k120k_run_1"
VIDEO_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/amph_vids_ssd")

# 1. Load config
config = kpms.load_config(str(PROJECT_DIR))
if 'use_bodyparts' in config:
    config['bodyparts'] = config['use_bodyparts']

# 2. Load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)

# 3. Extract results
results = kpms.extract_results(model, metadata, str(PROJECT_DIR), MODEL_NAME, save_results=False)

# 4. Estimate coordinates, centroids, headings
Y_est = estimate_coordinates(
    jnp.array(model['states']['x']),
    jnp.array(model['states']['v']),
    jnp.array(model['states']['h']),
    jnp.array(model['params']['Cd'])
)
coordinates = unbatch(Y_est, *metadata)
centroids, headings = kpms.get_centroids_headings(coordinates, **config)

# 5. Load video frame indexes & apply 2x upsampling for 80Hz raw camera video
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
with np.load(vfi_path) as z:
    raw_vfi = {k: z[k] for k in z.files}

# CRITICAL: 40Hz models on 80Hz camera videos require 2x VFI upsampling!
# Raw VFI is in 40Hz model units. If raw VFI is used directly, videos will seek to half-time (empty arena).
upsample_rate = 2
upsampled_vfi = {}
for k, v in raw_vfi.items():
    v_rep = np.repeat(v * upsample_rate, upsample_rate)
    offsets = np.tile(np.arange(upsample_rate), len(v))
    upsampled_vfi[k] = (v_rep + offsets).astype(int)
video_frame_indexes = upsampled_vfi

# Upsample syllables, coordinates, centroids, headings by 2x
results = {k: {**v, 'syllable': np.repeat(v['syllable'], upsample_rate)} for k, v in results.items()}
coordinates = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in coordinates.items()}
centroids = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in centroids.items()}
headings = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in headings.items()}
target_fps = int(config.get('fps', 40) * upsample_rate)

# 6. Map recording keys to video files
video_paths = {}
for key in results.keys():
    # Strip DLC suffix to find base video filename
    base_stem = key.split("DLC_")[0]
    avi_path = VIDEO_DIR / f"{base_stem}.avi"
    if avi_path.exists():
        video_paths[key] = str(avi_path)

# Filter dictionaries to only common mapped keys
common_keys = list(video_paths.keys())
results = {k: results[k] for k in common_keys}
coordinates = {k: coordinates[k] for k in common_keys}
centroids = {k: centroids[k] for k in common_keys}
headings = {k: headings[k] for k in common_keys}
video_frame_indexes = {k: video_frame_indexes[k] for k in common_keys}

# Clean config to avoid keyword conflicts
config_clean = config.copy()
config_clean.pop('fps', None)
config_clean.pop('overlay_keypoints', None)

# 7A. Run WITHOUT density sampling (Random)
out_random = PROJECT_DIR / MODEL_NAME / "grid_movies_random"
out_random.mkdir(parents=True, exist_ok=True)
kpms.generate_grid_movies(
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(out_random),
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=target_fps,
    overlay_keypoints=False,
    sampling_options={"mode": "random"},
    **config_clean
)

# 7B. Run WITH density sampling
out_density = PROJECT_DIR / MODEL_NAME / "grid_movies_density"
out_density.mkdir(parents=True, exist_ok=True)
kpms.generate_grid_movies(
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(out_density),
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=target_fps,
    overlay_keypoints=False,
    sampling_options={"mode": "density", "pca_samples": 50000, "pca_dim": 4, "n_neighbors": 50},
    **config_clean
)
```

---

## 5. End-to-End Workflow: From Session Pickle (`*.pkl`)

When working directly from master analysis pickles like `jonathan_amph_sessions_v2_w_sylls.pkl`:

```python
import pickle
from pathlib import Path

pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl"
with open(pkl_path, "rb") as f:
    sessions = pickle.load(f)

# Build results dictionary directly from session objects
results = {}
video_paths = {}
for s in sessions:
    key = s.results_dict_key
    sylls = s.syllables["syllable"].dropna().astype(int).values
    results[key] = {"syllable": sylls}
    
    # Video path lookup
    if hasattr(s, 'video_refs_df') and not s.video_refs_df.empty:
        orig_video = s.video_refs_df.iloc[0]['path']
        # Check SSD fast cache first
        ssd_candidate = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/amph_vids_ssd") / Path(orig_video).name
        video_paths[key] = str(ssd_candidate if ssd_candidate.exists() else orig_video)
```

---

## 6. Trajectory Plots Generation

To visualize the physical 2D/3D movement paths of keypoints for each syllable:

```python
out_traj_dir = PROJECT_DIR / MODEL_NAME / "trajectory_plots"
out_traj_dir.mkdir(parents=True, exist_ok=True)

# Using built-in kpms plot function
kpms.plot_syllable_trajectories(
    coordinates=coordinates,
    results=results,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=str(out_traj_dir),
    pre=5,
    post=15,
    use_dims=[0, 1], # 2D top-down view
    **config_clean
)
```

---

## 7. Troubleshooting & Gotchas

1. **`AttributeError: module 'numpy' has no attribute 'bool8'`**:
   - Cause: Running under Python/NumPy 2.x where `np.bool8` is deprecated.
   - Fix: Use `/home/fsp585/miniforge3/envs/keypoint_moseq/bin/python`.
2. **`Failed to launch ptxas` or CUDA OOM during video generation**:
   - Cause: JAX trying to compile kernels on GPU while FFmpeg holds video buffers.
   - Fix: Always set `os.environ["CUDA_VISIBLE_DEVICES"] = ""` and `os.environ["JAX_PLATFORMS"] = "cpu"`.
3. **`fps` conflict error in `generate_grid_movies`**:
   - Cause: `fps` defined both explicitly as argument and inside `**config`.
   - Fix: Remove `config.pop('fps', None)` before unpacking.
4. **Key Mismatch Between `results` and `video_frame_indexes`**:
   - Cause: Subset of sessions modeled or filtered (e.g. flipped sessions).
   - Fix: Always intersect keys: `common_keys = set(results.keys()) & set(video_frame_indexes.keys()) & set(video_paths.keys())`.
5. **VFI Index Scale Verification (Native Camera vs Model Index)**:
   - Always check `np.load(vfi_path)[k].max()`:
     * If `max <= 230,000` (40Hz model units), apply 2x upsampling (`v * 2` and interleave offsets).
     * If `max > 400,000` (e.g. in `amph_prod_k150k_run_12`), the VFI ALREADY stores native 80Hz camera indices with step 2! Do NOT multiply by 2, or you will get `ValueError: frame number requested outside video bounds`.
6. **Matplotlib 3.8+ `tostring_rgb` AttributeError in Trajectory Plots**:
   - Cause: `FigureCanvasAgg.tostring_rgb()` was removed in Matplotlib 3.8+.
   - Fix: Monkeypatch `viz.rasterize_figure`:
     ```python
     def modern_rasterize_figure(fig):
         fig.canvas.draw()
         if hasattr(fig.canvas, "buffer_rgba"):
             return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
         elif hasattr(fig.canvas, "tostring_rgb"):
             w, h = fig.canvas.get_width_height()
             return np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3))
         return np.asarray(fig.canvas.print_to_buffer()[0])[:, :, :3]
     viz.rasterize_figure = modern_rasterize_figure
     ```
7. **Scoping Grid Movies & Trajectory Plots to Top N Syllables**:
   - In built-in KPMS, patch `get_syllable_instances` to filter keys:
     ```python
     orig_gsi = util.get_syllable_instances
     def filtered_gsi(*args, **kwargs):
         return {k: v for k, v in orig_gsi(*args, **kwargs).items() if 0 <= k < 30}
     viz.get_syllable_instances = filtered_gsi
     util.get_syllable_instances = filtered_gsi
     ```
8. **Automated Video Synchronization Validation via Centroid Contrast**:
   - Because mice are dark on a light arena floor, verify spatial-temporal sync by checking that the cropped centroid is significantly darker than the floor:
     $$\Delta I = \text{Median}(I_{\text{floor}}) - \text{Mean}(I_{\text{mouse\_centroid}}) > 20\text{ px},\quad p < 10^{-5}$$
