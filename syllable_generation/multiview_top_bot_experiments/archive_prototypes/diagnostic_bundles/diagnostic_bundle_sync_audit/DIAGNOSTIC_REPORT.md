# Comprehensive Diagnostic Report: Multi-View Synchronization, Video Rendering & Training Provenance

**Date:** October 1, 2026  
**Audience:** Experimental Neuroscientists, Computer Vision Engineers, KPMS Developers  
**Scope:** Tandem Multi-View Keypoint-MoSeq (Champion Model Seed 101 / Seed 303)  
**Package:** `diagnostic_bundle_sync_audit.zip`

---

## 1. Executive Summary & Verification of External Review

An independent technical audit of `tandem_multiview_diagnostic_package.zip` identified fundamental mathematical and engineering bugs in the dual-camera video grid movie rendering pipeline, as well as critical subtleties regarding hardware TTL synchronization.

Following an exhaustive inspection of the underlying codebase, hardware timestamp tables, and model training artifacts, **we confirm that every observation raised by the reviewer is 100% mathematically and empirically accurate**:

1. **The Bottom-View Flip Order Inversion (The 180° Discrepancy):** The video renderer rotates the raw BotCam image by raw heading to face right, and *then* applies `cv2.flip(tile_bot, 1)`. Flipping horizontally after rotation causes the mouse to face **left** ($180^\circ$ opposite to TopCam). In contrast, the fusion pipeline reflects coordinates *before* computing heading, keeping the snout facing **right**.
2. **Searchsorted Jitter & Dropped Frame Vulnerability:** The active synchronization engine computes a single scalar integer offset per session using `np.searchsorted`. Because `np.searchsorted` computes the insertion index (ceil behavior when $b > t$), sub-millisecond jitter (as small as $32\,\mu\text{s}$) can jump the offset by an entire frame ($16.7\text{ ms}$). Furthermore, a static scalar offset cannot correct intra-session dropped frames.
3. **Silent Failure Modes:** Missing sessions silently receive an offset of 0, and out-of-boundary frames are clamped to the first/last frame without warnings, producing frozen video tiles.
4. **Tautological Synchronization Metric:** Checking heading agreement on `fused_h5` is circular because ventral keypoints were projected into dorsal space using dorsal heading during fusion.
5. **CRITICAL FINDING — Training Provenance is Sound:** Seed 101 **WAS trained on TTL-aligned tracking data**. Direct binary inspection of `fused_h5_full_cohort/2026-05-16T10_41_38_top_left_tandem_23kp.h5` proves that frames 0 to 66 are padded with `NaN` / $0.0$ confidence, and ventral tracking commences precisely at frame 67. The model was trained *after* this regeneration. The timing bugs reside in the video rendering pipeline and the scalar offset assumptions, not in the core model training inputs.

---

## 2. Mathematical Breakdown of the Video vs. Keypoint Discrepancy

### 2.1 What the Fusion Engine Does (`core/fusion_engine.py`)
In the tracking fusion engine:
1. BotCam raw coordinates $(x_{\text{raw}}, y_{\text{raw}})$ are reflected horizontally **first**:
   $$x' = W_{\text{arena}} - x_{\text{raw}}, \quad y' = y_{\text{raw}}$$
2. Bilateral swap is applied (left paws $\leftrightarrow$ right paws).
3. Egocentric heading $\theta_{\text{bot}}$ is computed in the reflected coordinate system:
   $$\theta_{\text{bot}} = \text{atan2}(y'_{\text{nose}} - y'_{\text{tail}}, x'_{\text{nose}} - x'_{\text{tail}})$$
4. Coordinates are rotated into the animal's body frame by $-\theta_{\text{bot}}$:
   $$\mathbf{v}_{\text{ego}} = \mathbf{R}(-\theta_{\text{bot}}) (\mathbf{p}' - \mathbf{c}')$$
   **Result:** In egocentric coordinates, the snout points along $+X$ (**Right**), and the animal's left side is on the left ($+Y$).

### 2.2 What the Video Renderer Did (`render_all_seed101_grid_movies.py`)
In the grid movie generator:
1. Heading was computed on **raw unreflected** BotCam coordinates:
   $$h_{\text{raw}} = \text{atan2}(y_{\text{nose}} - y_{\text{tail}}, x_{\text{nose}} - x_{\text{tail}})$$
2. Affine matrix $M_{\text{bot}}$ was constructed to rotate the raw video frame by $-h_{\text{raw}}$:
   $$\text{crop\_bot} = \text{cv2.warpAffine}(\text{raw\_frame}, M_{\text{bot}}, \dots)$$
   **At this intermediate step, the snout is rotated to point Right.**
3. Then, a post-warp horizontal flip was applied:
   $$\text{tile\_bot} = \text{cv2.flip}(\text{tile\_bot}, 1)$$
   **Result:** The horizontal axis was inverted *after* heading alignment, flipping the snout from Right to **Left**!
   TopCam snout faces **Right**; BotCam snout faces **Left**.

### 2.3 The Required Mathematical Equivalence
To make the video crop match the fused keypoint coordinate frame identically:
- The bottom video must be reflected horizontally **before** heading rotation, OR
- The affine transform matrix must incorporate horizontal reflection before applying $-h$:
  $$M_{\text{corrected}} = M(c_{\text{reflected}}, h_{\text{reflected}}) \cdot \begin{bmatrix} -1 & 0 & W-1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- When transformed correctly, both TopCam and BotCam snouts point **Right**, and lateral body axes align with perfect chirality.

---

## 3. Investigation of Timing & TTL Hardware Synchronization

### 3.1 Searchsorted vs. Nearest-Neighbor Offsets
In `core/ttl_sync.py`:
```python
bot0_in_top = int(np.searchsorted(t_fp, b_fp[0]))
top0_in_bot = int(np.searchsorted(b_fp, t_fp[0]))
```
- `np.searchsorted(a, v, side='left')` returns the insertion index $i$ such that $a[i-1] < v \le a[i]$.
- If BotCam frame 0 timestamp is $T_{\text{bot}}[0] = 3398.473504\text{ s}$ and TopCam frame 67 is $T_{\text{top}}[67] = 3398.473472\text{ s}$ ($32\,\mu\text{s}$ earlier):
  Because $T_{\text{bot}}[0] > T_{\text{top}}[67]$, `np.searchsorted` returns **68**, even though frame 67 is only $32\,\mu\text{s}$ away and frame 68 is $16,634\,\mu\text{s}$ away!
- **Resolution:** The true nearest-neighbor frame is:
  $$\Delta_{\text{TTL}} = \text{argmin}_k |T_{\text{top}}[k] - T_{\text{bot}}[0]|$$

### 3.2 Static Scalar Offset vs. Continuous Timestamp Indexing
- The pipeline currently assumes a constant frame offset $\Delta_{\text{TTL}}$ across the entire session.
- If BotCam drops frames mid-recording (e.g. USB bus congestion or camera driver stalls), the true frame mapping shifts dynamically over time.
- The diagnostic bundle includes the complete, raw timestamp CSVs for both cameras so that true continuous nearest-neighbor mapping can be benchmarked against the static scalar offset.

---

## 4. Verification of Model Training Provenance (Seed 101)

The most important question is whether Champion Seed 101 was trained on synchronized data.

| Component / Artifact | File Path | Timestamp (CEST) | Invariant Status |
| :--- | :--- | :--- | :--- |
| **TTL Cohort Cache** | `core/ttl_cohort_offsets.json` | 2026-09-30 15:40:46 | Generated with hardware offsets. |
| **Fused Cohort H5** | `fused_h5_full_cohort/*.h5` | 2026-09-30 16:08:25 | Generated with TTL offsets applied. |
| **Video Frame Indexes** | `kpms_.../video_frame_indexes.npz` | 2026-09-30 16:56:39 | Built from TTL-aligned fused H5s. |
| **Preprocessed Tracking** | `kpms_.../preprocessed_data.pkl` | 2026-09-30 16:57:18 | Cached for model training. |
| **Champion Seed 101 Model** | `kpms_..._seed101/results.h5` | 2026-10-01 08:48:51 | Trained on preprocessed data. |

### Direct Binary Confirmation on Session `2026-05-16T10_41_38_top_left`:
- In `fused_h5_full_cohort/2026-05-16T10_41_38_top_left_tandem_23kp.h5`:
  - Frames `0` through `66`: `left_front_paw_x = NaN`, `likelihood = 0.0` (Exact 67-frame pre-startup padding).
  - Frame `67`: First valid ventral coordinates (`left_front_paw_x = -41.62`, `likelihood = 0.99`).
- **Conclusion:** **Seed 101 was trained on TTL-aligned tracking data.** The learned syllable dynamics and AR-SLDS latent states incorporate synchronized ventral and dorsal kinematics.

---

## 5. Diagnostic Package Contents (`diagnostic_bundle_sync_audit.zip`)

```text
diagnostic_bundle_sync_audit/
├── DIAGNOSTIC_REPORT.md                          (This comprehensive document)
├── grid_movies/
│   ├── syllable_01_tandem_grid_seed303.mp4       (Reference grid movie with Row 1 Col 3 & Row 3 Col 3)
│   └── syllable_01_tandem_grid_density_seed101.mp4 (Seed 101 density grid movie)
├── timing_and_offsets/
│   └── ttl_cohort_offsets.json                   (Cohort-wide calculated startup offsets & residuals)
├── training_provenance/
│   ├── seed101_config.yml                        (Keypoint-MoSeq configuration for Seed 101)
│   └── training_provenance_verification.json     (Detailed artifact timestamp and padding proof)
│
├── good_example_synchronous/                     (Session: 2025-12-03T14_06_45_bottom_right)
│   ├── bout_metadata.json                        (Syllable 01, Row 1 Col 3, dur 55f, offset = 0)
│   ├── video_clips/
│   │   ├── topcam_raw_frames_138868_to_138943.mp4
│   │   └── botcam_ttl_aligned_frames_138868_to_138943.mp4
│   ├── dlc_tracking/
│   │   ├── topcam_dlc_slice.csv                  (Raw TopCam DLC rows for clip)
│   │   └── botcam_dlc_ttl_aligned_slice.csv      (Raw BotCam DLC rows for clip)
│   ├── model_slices/
│   │   ├── vfi_slice.csv                         (Video Frame Index mapping for bout)
│   │   └── model_centroid_heading_syllable.csv   (Model latent centroid, heading, and syllable IDs)
│   └── timestamps/
│       ├── TopCam_Timestamps.csv.gz              (Full session TopCam FP3002 timestamps, 195k frames)
│       └── BotCam_Timestamps.csv.gz              (Full session BotCam FP3002 timestamps, 195k frames)
│
└── bad_example_startup_latency/                  (Session: 2026-05-16T10_41_38_top_left)
    ├── bout_metadata.json                        (Syllable 01, Row 3 Col 3, dur 43f, offset = +67)
    ├── video_clips/
    │   ├── topcam_raw_frames_85385_to_85448.mp4  (Raw TopCam video clip around bout)
    │   ├── botcam_ttl_aligned_frames_85318_to_85381.mp4 (BotCam video with -67 frame TTL alignment)
    │   └── botcam_unaligned_naive_frames_85385_to_85448.mp4 (BotCam naive 1-to-1 unaligned clip)
    ├── dlc_tracking/
    │   ├── topcam_dlc_slice.csv                  (Raw TopCam DLC rows for clip)
    │   └── botcam_dlc_ttl_aligned_slice.csv      (Raw BotCam DLC rows with -67 frame shift)
    ├── model_slices/
    │   ├── vfi_slice.csv                         (Video Frame Index mapping for bout)
    │   └── model_centroid_heading_syllable.csv   (Model latent centroid, heading, and syllable IDs)
    └── timestamps/
        ├── TopCam_Timestamps.csv.gz              (Full session TopCam FP3002 timestamps, 198k frames)
        └── BotCam_Timestamps.csv.gz              (Full session BotCam FP3002 timestamps, 198k frames)
```
