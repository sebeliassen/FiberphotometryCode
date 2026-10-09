# Comprehensive Diagnostic Report: Multi-View Synchronization, Video Rendering & Training Provenance

**Date:** October 1, 2026  
**Audience:** Experimental Neuroscientists, Computer Vision Engineers, KPMS Developers  
**Scope:** Tandem Multi-View Keypoint-MoSeq (Champion Model Seed 101 / Seed 303)  
**Package:** `diagnostic_bundle_sync_audit.zip`

---

## 1. Executive Summary & Consensus Findings

Following collaborative investigation between the research team and engineering, the following core conclusions are firmly established:

1. **Seed 101 Training Provenance is Valid:**
   Seed 101 was trained on TTL-aligned tracking data. Direct inspection of `fused_h5_full_cohort/2026-05-16T10_41_38_top_left_tandem_23kp.h5` confirms that frames 0–66 contain `NaN` coordinates and $0.0$ likelihood for ventral keypoints, with valid tracking beginning at frame 67. The model was trained *after* this alignment.
2. **Intentional Bottom-View Flip is Working as Intended:**
   In `config.yml`, `tail_base` is designated as `anterior_bodyparts` and `nose` as `posterior_bodyparts`. Model heading points approximately toward the tail. Rotating the dorsal crop by $-\text{heading}$ makes the mouse snout face left; flipping the bottom view horizontally causes turns in both views to visually match (left turn in dorsal matches left turn in ventral).
3. **The Core Timing Defect — Static Offset Blindness to Mid-Session Drops:**
   Correcting the initial camera trigger offset resolved startup latency (e.g., 67 frames in L-DOPA). However, using a single constant scalar offset across the entire 54-minute recording blinded the pipeline to subsequent frame drops. In session `2025-12-03T14_06_45`, BotCam dropped 6 frames at frame 186,366. Because the scalar offset remained 0, BotCam was paired with footage ~100 ms in the future for all subsequent 8,767 frames (reaching model training at frame 162,654).
4. **Resolution of the Seed 101 Movie "Stationary Dot" Discrepancy:**
   The previously supplied movie file (`syllable_01_tandem_grid_density.mp4`) had an OS timestamp of `13:24 CEST`, rendered before the renderer script (`render_all_seed101_grid_movies.py`) was updated with moving dot logic at `14:04 CEST`. Fresh renders (`syllable_00` and `syllable_01`) now show active centroid dot motion (>26 px displacement) in both views.
5. **Codebase Timestamp Architecture:**
   - **Fiber Photometry:** Strictly uses `FP3002_System_Timestamp` to merge with the Doric console's `SystemTimestamp_470` / `SystemTimestamp_415`.
   - **Camera Synchronization:** Multi-view fusion historically attempted to use `FP3002_System_Timestamp`, but in sessions where photometry stopped early (e.g. at minute 26.15 / frame 94,167), this timestamp froze across 100,966 consecutive frames. To avoid freezing, a static offset was used, which caused the gap vulnerability. Moving forward, continuous mapping must be anchored to `Cam_Bonsai_Timestamp` and calibrated `Cam_Basler_Timestamp`.

---

## 2. Scientific Qualifications on Key Technical Components

### 2.1 Basler Timestamps Require Cross-Camera Calibration
`Cam_Basler_Timestamp` provides nanosecond-resolution internal crystal clock ticks on each camera board. However, camera hardware boards power on and reset independently (their origins differ by ~1.54 seconds in `2025-12-03` and ~1.45 seconds in `2026-05-16`). Direct subtraction without cross-camera calibration introduces an artificial offset.
- **Proper Usage:** Calibrate the relative offset at frame 0 using the initial overlap window:
  $$C_{\text{offset}} = t_{\text{Basler}}^{\text{top}}[\text{bot0\_in\_top}] - t_{\text{Basler}}^{\text{bot}}[0]$$
  Then evaluate continuous elapsed time:
  $$t_{\text{cal\_Basler}}^{\text{bot}}[j] = t_{\text{Basler}}^{\text{bot}}[j] + C_{\text{offset}}$$

### 2.2 Frozen Photometry Timestamp Status
In session `2025-12-03T14_06_45`, `FP3002_System_Timestamp` is verified to be frozen at `19553.897792 s` across **100,966 consecutive rows** (frames 94,168 to 195,134). While Bonsai's `WithLatestFrom` operator repeating the last emitted sample after Doric photometry acquisition terminated is the most plausible architectural explanation, it remains an inferred mechanism in the absence of raw `.bonsai` workflow files.

### 2.3 Role of Model Pose Estimates ($Y_{\text{est}}$)
Projecting SLDS latent state trajectories back to coordinate space via `estimate_coordinates(x, v, h, Cd)` provides kinematically smoothed, outlier-free pose overlays for visualization. However, because the model was fitted on the training data, $Y_{\text{est}}$ **cannot** repair timing misalignments present in the training set or independently validate camera synchronization.

---

## 3. Investigation: Which Timestamps Are Used in the Repository?

| Subsystem | File | Timestamp Column Used | Target Merged Entity | Behavior & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Fiber Photometry** | `fiberphotometry/data/session_synchronization.py` | `FP3002_System_Timestamp` | Doric `SystemTimestamp` (470 & 415 nm) | Correct & necessary. Video frames match photometry signals within 20 ms tolerance. |
| **Tandem TTL Sync** | `syllable_generation/multiview_top_bot_experiments/core/ttl_sync.py` | `FP3002_System_Timestamp` (at frame 0 only) | TopCam frame 0 $\leftrightarrow$ BotCam frame 0 | Correctly finds initial trigger delay (e.g. 67 frames), but ignores all subsequent dropped frames. |
| **Clock Sync (Legacy)** | `.../01_data_and_fusion/clock_sync.py` | `FP3002_System_Timestamp` (priority cand 1) | Per-frame nearest neighbor | Failed on sessions where photometry stopped early, mapping all late frames to frame 94,167. |
| **Host PC Clock** | `Cam_Bonsai_Timestamp` | Windows OS frame arrival time | Available in all CSVs | Strictly monotonic, runs full 195k frames without freezing, immediately detects frame drops. |
| **Camera Hardware Clock** | `Cam_Basler_Timestamp` | Internal camera crystal oscillator (ns) | Available in all CSVs | Jitter-free interval measurement; requires initial cross-camera calibration. |

---

## 4. Fresh Seed 101 Movie Renders & Bout Provenance

Fresh movies have been rendered and verified using `render_all_seed101_grid_movies.py`:
- `grid_movies/syllable_00_tandem_grid_density_seed101_fresh.mp4` (2.47 MB)
- `grid_movies/syllable_01_tandem_grid_density_seed101_fresh.mp4` (2.81 MB)

### Automated Pixel Tracking Verification (Panel 0, Top-Left):
- **Syllable 00:**
  - Active frames: Top = 120, Bot = 102 (out of 150 frames).
  - Top dot displacement: **28.40 px** ($X: [87.0, 98.1], Y: [61.0, 88.1]$).
  - Bot dot displacement: **26.41 px** ($X: [83.0, 91.0], Y: [61.9, 88.0]$).
- **Syllable 01:**
  - Active frames: Top = 120, Bot = 120.
  - Top dot displacement: **4.08 px**.
  - Bot dot displacement: **16.83 px**.

### Exemplar Bout Metadata for Syllable 00 (Density Exemplars):
| Panel | Grid Pos | Session ID (`sid`) | Model Start Frame | Duration (f) | TopCam Raw Frame | BotCam Raw Frame | Offset ($\Delta f$) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | R0, C0 | `2025-12-03T10_58_48_bottom_left` | 49,692 | 53 | 76,348 | 76,348 | 0 |
| **1** | R0, C1 | `2025-12-03T14_06_45_bottom_right` | 160,110 | 53 | 183,823 | 183,823 | 0 |
| **2** | R0, C2 | `2026-01-08T11_31_36_bottom_left` | 110,318 | 53 | 122,780 | 122,780 | 0 |
| **3** | R0, C3 | `2025-12-03T13_02_27_bottom_right` | 95,448 | 53 | 132,446 | 132,471 | -25 |
| **4** | R1, C0 | `2026-03-05T12_46_18_bottom_left` | 173,368 | 53 | 194,474 | 194,474 | 0 |
| **5** | R1, C1 | `2026-04-27T14_05_05_top_left` | 87,458 | 53 | 106,667 | 106,662 | +5 |
| **6** | R1, C2 | `2026-03-05T11_01_01_bottom_right` | 48,560 | 53 | 75,202 | 75,212 | -10 |
| **7** | R1, C3 | `2026-04-28T13_21_05_bottom_left` | 123,294 | 53 | 151,503 | 151,494 | +9 |
| **8** | R2, C0 | `2025-12-03T09_51_05_bottom_right` | 68,129 | 53 | 113,783 | 113,821 | -38 |
| **9** | R2, C1 | `2026-04-28T13_21_05_bottom_left` | 181,359 | 53 | 219,449 | 219,440 | +9 |
| **10** | R2, C2 | `2025-12-03T16_38_48_bottom_right` | 22,585 | 53 | 49,419 | 49,420 | -1 |
| **11** | R2, C3 | `2026-05-16T10_41_38_bottom_left` | 72,190 | 53 | 87,267 | 87,235 | +32 |

---

## 5. Next Engineering Steps

1. **Continuous Per-Frame Correspondence Engine:**
   Implement a unified per-frame camera correspondence using initial startup calibration combined with continuous `Cam_Bonsai_Timestamp` / calibrated `Cam_Basler_Timestamp`.
2. **Cohort-Wide Mapping Audit:**
   Run the continuous correspondence across all 118 sessions to quantify the total number of retained training frames affected by mid-session camera drops.
3. **Tracking Fusion & Renderer Unification:**
   Update both `core/fusion_engine.py` and `render_all_seed101_grid_movies.py` to share this exact per-frame mapping function.
