# Keypoint-MoSeq (KPMS) Visualizations: Grid Movies & Trajectories Guide

**Authoritative reference for generating Syllable Grid Movies and Trajectory Visualizations across datasets (Karen, AMPH, etc.).**

---

## 1. Golden Rules for Future Agents & Chatbots

> [!IMPORTANT]
> 1. **Do NOT Reinvent the Wheel**: Never write manual 300+ line tile-rendering loops, custom bounding-box warps, or bespoke KDE density samplers. Keypoint-MoSeq provides `kpms.generate_grid_movies()` which handles orientation alignment, centroid centering, tile concatenation, and both uniform and density sampling natively.
> 2. **CRITICAL: 2x VFI Upsampling (40Hz Model vs. 80Hz Camera)**:
>    - Camera videos are typically recorded at **80 Hz** (or 60/80 Hz).
>    - KPMS models are trained on downsampled data at **40 Hz** (`fps: 40`).
>    - The saved `video_frame_indexes.npz` files contain indices in **40 Hz units** (e.g., Session 1 starts around frame `24,961`).
>    - **If raw VFI is used directly without 2x upsampling, OpenCV seeks to half-elapsed-time ($t \approx 312\text{ s}$)** — before the mouse is even placed into the arena! The resulting videos will display empty floor tiles.
>    - **Multiplying VFI by 2** ($24,961 \times 2 = 49,922$) lands exactly at the moment the animal is placed and tracks the mouse dead-center.
> 3. **Fast-Seek Monkeypatch**: Always patch `OpenCVReader.read` to sequentially step across small frame gaps ($\le 30$) rather than calling `cap.set(POS_FRAMES)` on every frame. This gives a 5x–10x speedup on large `.avi` video files.
> 4. **CPU Execution**: Always force `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS="cpu"` before importing JAX or KPMS to avoid GPU OOM and `ptxas` compiler conflicts.

---

## 2. Canonical Production Script: Grid Movies (Random & Density Modes)

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 1. Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates
import keypoint_moseq as kpms
from vidio.read import OpenCVReader

# 2. Apply fast-seek monkeypatch to OpenCVReader
_original_read = OpenCVReader.read
def _fast_read(self, framenum):
    if isinstance(framenum, slice):
        return _original_read(self, framenum)
    gap = framenum - self.fnum
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret:
                break
        self.fnum = framenum
    return _original_read(self, framenum)
OpenCVReader.read = _fast_read

# 3. Load checkpoint, config, results
PROJECT_DIR = Path("/path/to/project_dir")
MODEL_NAME = "model_name"
VIDEO_DIR = Path("/path/to/raw_videos")

config = kpms.load_config(str(PROJECT_DIR))
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

# 4. Reconstruct continuous coordinates, centroids, headings
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
centroids = {k: v["centroid"] for k, v in results.items()}
headings = {k: v["heading"] for k, v in results.items()}

# 5. Map video paths
video_paths = {}
for k in results.keys():
    stem = k.split("DLC_")[0]
    cand = VIDEO_DIR / f"{stem}.avi"
    if cand.exists():
        video_paths[k] = str(cand)

# 6. Apply 2x VFI Upsampling (40Hz Model -> 80Hz Camera Video Sync)
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
with np.load(vfi_path) as z:
    raw_vfi = {k: z[k] for k in z.files}

upsample_rate = 2
upsampled_vfi = {}
for k, v in raw_vfi.items():
    v_scaled = v * upsample_rate
    N = len(v_scaled)
    v_rep = np.repeat(v_scaled, upsample_rate)
    offsets = np.tile(np.arange(upsample_rate), N)
    upsampled_vfi[k] = (v_rep + offsets).astype(int)
video_frame_indexes = upsampled_vfi

# Upsample syllables and spatial coordinates to match upsampled timeline
results_upsampled = {
    k: {**v, "syllable": np.repeat(v["syllable"], upsample_rate)}
    for k, v in results.items()
}
coordinates = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in coordinates.items()}
centroids = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in centroids.items()}
headings = {k: np.repeat(arr, upsample_rate, axis=0) for k, arr in headings.items()}

target_fps = int(config.get("fps", 40) * upsample_rate)  # 80 fps

# Clean config
cfg = config.copy()
cfg.pop("fps", None)
cfg.pop("overlay_keypoints", None)

# 7A. Generate WITHOUT density sampling (Random uniform instance sampling)
out_random = str(PROJECT_DIR / MODEL_NAME / "grid_movies_without_density")
kpms.generate_grid_movies(
    results=results_upsampled,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=out_random,
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=target_fps,
    overlay_keypoints=False,
    sampling_options={"mode": "random"},
    **cfg
)

# 7B. Generate WITH density sampling (Modal/PCA-density exemplar sampling)
out_density = str(PROJECT_DIR / MODEL_NAME / "grid_movies_with_density")
kpms.generate_grid_movies(
    results=results_upsampled,
    project_dir=str(PROJECT_DIR),
    model_name=MODEL_NAME,
    output_dir=out_density,
    video_paths=video_paths,
    coordinates=coordinates,
    centroids=centroids,
    headings=headings,
    video_frame_indexes=video_frame_indexes,
    fps=target_fps,
    overlay_keypoints=False,
    sampling_options={"mode": "density", "pca_samples": 50000, "pca_dim": 4, "n_neighbors": 50},
    **cfg
)
```

---

## 3. Density vs. Random Sampling

| Mode | Sampling Options | Visual Outcome | Best Used For |
| :--- | :--- | :--- | :--- |
| **Random** | `{"mode": "random"}` | Uniformly samples qualifying instances across sessions | Checking syllable diversity, tail/paw variability, and edge cases |
| **Density** | `{"mode": "density", "pca_samples": 50000, "pca_dim": 4, "n_neighbors": 50}` | Finds the highest density mode in PCA trajectory space and takes its nearest neighbors | Generating canonical exemplars for figures, ethograms, and naming |

---

## 4. Troubleshooting Checklist

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **Video shows empty floor; dot floats in air** | VFI was used raw in 40Hz model units on an 80Hz video file | Apply 2x VFI upsampling (`v_scaled = v * 2`) and upsample arrays |
| **Mouse facing backwards (180° rotation)** | Inverted `anterior_bodyparts` (e.g. `tail_base`) in `config.yml` | Model is self-consistent internally; display-side flip angle by $\pi$ radians if visual forward-facing is desired |
| **Slow video extraction (>30s/movie)** | Seeking unindexed AVI frames | Ensure the `OpenCVReader.read` fast-seek monkeypatch is applied |
| **GPU OOM / JAX ptxas crash** | JAX allocating GPU memory for CPU-bound video task | Set `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS="cpu"` at very top of script |
