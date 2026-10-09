# Tandem Multi-View Visualizations Guide: Grid Movies & Trajectories

**Authoritative reference for generating Dual-View (TopCam Dorsal + BotCam Ventral) Syllable Grid Movies and Trajectory Plots in Keypoint-MoSeq.**

---

## 1. Executive Overview & Core Visual Principles

Multi-view Keypoint-MoSeq incorporates 23 keypoints (17 TopCam dorsal + 6 BotCam ventral) captured simultaneously by two cameras with distinct optical perspectives:
1. **TopCam (Dorsal View):** Mounted high above the arena. Animals appear smaller in raw pixels (spine length $\approx 80\text{--}90\text{ px}$).
2. **BotCam (Ventral View):** Mounted directly below the transparent arena floor. Animals appear $\approx 2\times$ larger in raw pixels (spine length $\approx 170\text{--}200\text{ px}$).

To create publication-grade visualizations, two critical technical hurdles must be solved:
- **Tandem Grid Movies:** Overcoming optical magnification disparity so both views match in physical scale, alignment, and tile resolution.
- **Trajectory Plots:** Overcoming spatial footprint disparity so fine paw kinematics are not squished or clipped by large dorsal spine bounding boxes.

---

## 2. Tandem Grid Movies: Dual-Camera Synchronization & Sizing

### 2.1 The Optical Scale Mismatch
TopCam and BotCam have different focal lengths and camera distances:
$$\text{median\_scale} = \frac{\text{median}(\text{length}_{\text{top}})}{\text{median}(\text{length}_{\text{bot}})} \approx 0.48\text{--}0.52$$
- If identical pixel crop windows were extracted (e.g., $120 \times 120\text{ px}$ from both cameras), the BotCam crop would show only a zoomed-in patch of the mouse's belly, while TopCam would capture the entire animal and surrounding arena.

### 2.2 The Two-Step Sizing Pipeline
To render both views at **identical physical magnification and identical tile resolution**:

1. **Scale-Aware Raw Cropping ($W_{\text{raw}}$):**
   Define the physical field of view in TopCam units ($W_{\text{top}} \approx 120\text{ px}$, capturing $\approx 12 \times 12\text{ cm}$). Compute the BotCam crop window inversely:
   $$W_{\text{bot}} = \frac{W_{\text{top}}}{\text{median\_scale}} \approx \frac{120\text{ px}}{0.50} = 240\text{ px}$$
   Both crops now cover the identical physical area centered on the mouse.

2. **Uniform Tile Bilinear Resizing (`cv2.resize`):**
   Both crops are resized to the exact same display tile dimension (e.g., $160 \times 160\text{ px}$):
   ```python
   top_tile = cv2.resize(top_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
   bot_tile = cv2.resize(bot_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
   ```

3. **Composite Tandem Layout (`[TopCam | BotCam]`):**
   Combine both tiles side-by-side inside each instance cell:
   ```python
   tandem_tile = np.hstack([top_tile, bot_tile])  # 160 x 320 px
   ```
   Arranging these into a $3 \times 4$ or $4 \times 4$ grid produces a unified movie where every syllable instance displays dorsal posture and ventral paw kinematics evolving in 1:1 temporal lockstep.

### 2.3 Egocentric Heading Alignment
In both views, the mouse is rotated egocentrically to face the same standard direction (e.g., facing "up"):
- **TopCam:** Rotate by $-\theta_{\text{top}}$ around the dorsal centroid.
- **BotCam:** Rotate by $-\theta_{\text{bot}}$ around the ventral centroid.
Because BotCam keypoints were already rotationally aligned via $R_{\text{bot\_inv}}$ during dataset fusion, the animal faces identically forward in both side-by-side tiles.

### 2.4 Video Sync & Native 60 FPS (No 2x Upsampling)
> [!IMPORTANT]
> - Unlike legacy 40 Hz models trained on 80 Hz videos (which required 2x VFI upsampling), **this multi-view pipeline operates at native 60.0 FPS**.
> - Video frame indices (`video_frame_indexes`) map **1:1** to camera frames (`frame_idx = vfi[t]`). Do **NOT** multiply by 2.

---

## 3. Trajectory Plots: View-Specific Axis Limits

### 3.1 Why Global Axis Limits Fail
In built-in Keypoint-MoSeq, `kpms.viz.get_limits(coordinates)` sets a single bounding box based on the 1st and 99th percentiles of all keypoints.
- The 17 dorsal bodyparts extend from the snout tip along the vertebral column to the tail tip ($\approx 10\text{ cm}$).
- The 4 ventral paws are clustered within a compact lateral footprint ($\approx 2.5\text{--}3.5\text{ cm}$).
- **The Problem:** If a shared axis limit is used, the ventral paws appear as tiny, indistinguishable dots, obscuring stride frequency, gait patterns, and grooming strokes. If tightly cropped to paws, the dorsal spine is clipped.

### 3.2 View-Specific Independent Limits
Compute **independent bounding boxes** for each camera view:

```python
# Separate dorsal (17) and ventral (6) keypoint indices
dorsal_idxs = [config["bodyparts"].index(bp) for bp in TOP_USE_BODYPARTS]
ventral_idxs = [config["bodyparts"].index(bp) for bp in BOT_USE_BODYPARTS]

# Compute view-specific limits with appropriate padding
lims_dorsal = kpms.viz.get_limits(
    coordinates[:, :, dorsal_idxs], 
    pctl=1, 
    left=0.15, right=0.15, top=0.15, bottom=0.15
)
lims_ventral = kpms.viz.get_limits(
    coordinates[:, :, ventral_idxs], 
    pctl=1, 
    left=0.20, right=0.20, top=0.20, bottom=0.20
)
```

### 3.3 Synchronized Two-Panel Figure Layout
Each syllable is rendered as a side-by-side (or top-bottom) 2-panel figure:
- **Panel A (Dorsal Spine & Head):** Scaled via `lims_dorsal` to display spine curvature, neck angle, and heading orientation.
- **Panel B (Ventral Paws):** Scaled via `lims_ventral` to auto-zoom on paw excursions, stance/swing phases, and bilateral symmetry.

```python
fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(10, 5))

# TopCam Dorsal Trajectory
kpms.viz.plot_trajectories(
    [f"Syllable {s} (Dorsal)"], [traj[:, dorsal_idxs]], lims=lims_dorsal, 
    edges=DORSAL_EDGES, ax=ax_top
)

# BotCam Ventral Trajectory
kpms.viz.plot_trajectories(
    [f"Syllable {s} (Ventral)"], [traj[:, ventral_idxs]], lims=lims_ventral, 
    edges=VENTRAL_EDGES, ax=ax_bot
)
```

---

## 4. Production Model Reference & Top Syllable Catalog

Visualizations should be generated from the trained full production model:

* **Model Directory:** [`syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project/fullmodel_multiview_23kp_k6e12`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project/fullmodel_multiview_23kp_k6e12)
* **Training Specs:** 200 iterations (50 AR @ $\kappa=6.0\times 10^{12}$ + 150 SLDS @ $\kappa=6.0\times 10^{11}$).
* **Dataset Scope:** 8 full sessions ($1,243,240$ valid frames @ 60.0 FPS).
* **Median Dwell:** **$18.00\text{ frames}$ ($300.0\text{ ms}$)**.
* **Active Syllables:** 100 syllables identified.

### Top Syllables for Visual Validation:
| Syllable Index | Frame Count | Usage % | Primary Behavior / Kinematics |
| :---: | :---: | :---: | :--- |
| **52** | $164,887$ | $13.26\%$ | High-frequency baseline locomotion / forward walk |
| **15** | $113,286$ | $9.11\%$ | Sustained forward exploratory locomotion |
| **79** | $104,660$ | $8.42\%$ | Low-velocity head scanning / investigative pause |
| **4** | $103,537$ | $8.33\%$ | Active sniffing / moderate pace turning |
| **55** | $78,172$ | $6.29\%$ | Rear-supported rearing / wall climbing |
| **26** | $50,348$ | $4.05\%$ | Rhythmic bilateral facial/paw grooming |
| **6** | $44,904$ | $3.61\%$ | Sharp lateral pivot / course adjustment |

---

## 5. Canonical Implementation Script

Below is the verified CPU-safe production script template to generate both tandem grid movies and 2-panel trajectory plots:

```python
#!/usr/bin/env python3
"""
render_tandem_visualizations.py
-------------------------------
Generates tandem dual-view grid movies and 2-panel trajectory plots
using scale-aware cropping and view-specific axis limits.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keypoint_moseq as kpms
from vidio.read import OpenCVReader

# --- 1. Fast-Seek Patch for OpenCVReader ---
_orig_read = OpenCVReader.read
def _fast_read(self, framenum):
    if isinstance(framenum, slice):
        return _orig_read(self, framenum)
    gap = framenum - self.fnum
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret: break
        self.fnum = framenum
    return _orig_read(self, framenum)
OpenCVReader.read = _fast_read

# --- 2. Paths & Model Loading ---
PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project")
MODEL_NAME = "fullmodel_multiview_23kp_k6e12"
OUT_DIR = PROJECT_DIR / MODEL_NAME / "tandem_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

config = kpms.load_config(str(PROJECT_DIR))
model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

print(f"Loaded {MODEL_NAME} with {len(results)} sessions.")
```
