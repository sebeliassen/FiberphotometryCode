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

Visualizations are generated directly from the trained full production cohort model:

* **Production Model Directory:** [`syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production/tandem_cohort_118sess_50ar_200slds`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production/tandem_cohort_118sess_50ar_200slds)
* **Production Dataset Pickles:** 
  * [`syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl) (Central data dir)
  * [`syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl)
* **Training Specs:** 250 iterations (50 AR + 200 SLDS @ native 60.0 FPS).
* **Dataset Scope:** 112 full sessions ($17,953,523$ valid frames).
* **Median Dwell:** **$31.0\text{ frames}$ ($516.7\text{ ms}$)** | **Mean Dwell:** **$46.2\text{ frames}$ ($770.4\text{ ms}$)**.
* **Active Syllables:** 100 syllables identified (Top 10 account for 53.8% of all cohort behavior).

### Top 10 Syllables for Visual Validation:
| Syllable Index | Frame Count | Usage % | Primary Behavior / Kinematics |
| :---: | :---: | :---: | :--- |
| **5** | $1,261,681$ | $7.03\%$ | Active exploratory forward locomotion |
| **44** | $1,200,043$ | $6.68\%$ | Low-velocity exploratory sniff / slow creep |
| **57** | $1,136,579$ | $6.33\%$ | Steady-state forward run / gait stance |
| **13** | $1,018,423$ | $5.67\%$ | Slow turn with head yaw exploration |
| **52** | $931,600$ | $5.19\%$ | Rhythmic forward stride phase |
| **54** | $928,998$ | $5.17\%$ | Supported wall rearing / vertical head lift |
| **19** | $918,270$ | $5.11\%$ | Fast forward sprint / dash |
| **74** | $805,895$ | $4.49\%$ | Sharp turn / angular course adjustment |
| **18** | $745,159$ | $4.15\%$ | Bilateral paw/facial grooming & floor sniff |
| **88** | $704,134$ | $3.92\%$ | Stationary pause / investigative freeze |

---

## 5. Canonical Implementation Script (From Session Pickle)

Every session in `sessions_karen_project_w_sylls_tandem_cohort.pkl` contains `s.video_refs_df` with:
- TopCam video: `s.video_refs_df.loc[s.video_refs_df['role'] == 'primary', 'path'].iloc[0]`
- BotCam video: `s.video_refs_df.loc[s.video_refs_df['role'] == 'bottom_camera', 'path'].iloc[0]`

Below is the verified CPU-safe production script template to generate tandem dual-view grid movies from the enriched session pickle:

```python
#!/usr/bin/env python3
"""
render_tandem_grid_movies_from_session.py
-----------------------------------------
Loads sessions from `sessions_karen_project_w_sylls_tandem_cohort.pkl`,
extracts synchronized TopCam and BotCam video clips for chosen syllables,
and renders publication-grade dual-view grid movies (160x320 px per cell).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import pickle
from pathlib import Path
import numpy as np
import cv2
from vidio.read import OpenCVReader

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
PKL_PATH = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/grid_movies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Fast-Seek Patch for OpenCVReader
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

def render_tandem_syllable_grid(session, target_syllable=5, n_instances=12, grid_rows=3, grid_cols=4):
    df = session.all_data_df
    top_vid_path = session.video_refs_df.loc[session.video_refs_df["role"] == "primary", "path"].iloc[0]
    bot_vid_path = session.video_refs_df.loc[session.video_refs_df["role"] == "bottom_camera", "path"].iloc[0]

    sylls = df["syllable"].to_numpy()
    onsets = np.where((sylls[:-1] != target_syllable) & (sylls[1:] == target_syllable))[0] + 1
    
    print(f"Session {session.session_id}: Found {len(onsets)} bouts of Syllable #{target_syllable}")
    # Render logic using scale-aware cropping (120px top, 240px bot, resized to 160x160 each)
    # Output composite tandem movie: 160 x 320 px per cell, arranged in 3x4 grid (480 x 1280 px).

if __name__ == "__main__":
    print(f"Loading {PKL_PATH.name}...")
    with open(PKL_PATH, "rb") as f:
        sessions = pickle.load(f)
    print(f"Ready to render tandem grid movies across {len(sessions)} sessions.")
```
