# Multi-View Keypoint-MoSeq (TopCam + BotCam) Experiment Plan & Session Inventory

**Folder:** `syllable_generation/multiview_top_bot_experiments`  
**Date:** September 24, 2026  
**Objective:** Leverage completed bottom camera (BotCam) ventral pose predictions in conjunction with overhead (TopCam) dorsal tracking from `sessions_karen_project_w_sylls.pkl` to discover fine-grained locomotion, gait, and paw-kinematic syllables in Keypoint-MoSeq (ARHMM-SLDS).

---

## 1. Inventory of Available TopCam / BotCam Session Pairs

From our production inference run (`/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922`), **18 full sessions** are completely analyzed, validated, and stored on disk, with **1 additional session (Video 19)** completing imminently.

All 18 completed pairs belong to the **`16w_PI` (16 Weeks Post-Injection)** longitudinal cohort, encompassing **6 full experimental trials across all 3 active arena chambers**:

| Trial | BotCam Arena File (`predictions/`) | Corresponding TopCam Session Key (`sessions_*.pkl`) | Chamber Quadrant | Status |
| :---: | :--- | :--- | :---: | :---: |
| **T1** | `16w_PI_T1_6_2_4_050326_active_Arena1_BotCamVid_2026-03-05T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-05T11_01_01_top_left` | Arena 1 | **Done (Verified)** |
| **T1** | `16w_PI_T1_6_2_4_050326_active_Arena2_BotCamVid_2026-03-05T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-05T11_01_01_bottom_left` | Arena 2 | **Done (Verified)** |
| **T1** | `16w_PI_T1_6_2_4_050326_active_Arena3_BotCamVid_2026-03-05T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-05T11_01_01_bottom_right` | Arena 3 | **Done (Verified)** |
| **T2** | `16w_PI_T2_..._Arena1_BotCamVid_2026-03-05T12_21_21` | `Arena1234_CenterTopCamVid_2026-03-05T12_21_21_top_left` | Arena 1 | **Done (Verified)** |
| **T2** | `16w_PI_T2_..._Arena2_BotCamVid_2026-03-05T12_21_21` | `Arena1234_CenterTopCamVid_2026-03-05T12_21_21_bottom_left` | Arena 2 | **Done (Verified)** |
| **T2** | `16w_PI_T2_..._Arena3_BotCamVid_2026-03-05T12_21_21` | `Arena1234_CenterTopCamVid_2026-03-05T12_21_21_bottom_right` | Arena 3 | **Done (Verified)** |
| **T3** | `16w_PI_T3_..._Arena1_BotCamVid_2026-03-05T13_41_41` | `Arena1234_CenterTopCamVid_2026-03-05T13_41_41_top_left` | Arena 1 | **Done (Verified)** |
| **T3** | `16w_PI_T3_..._Arena2_BotCamVid_2026-03-05T13_41_41` | `Arena1234_CenterTopCamVid_2026-03-05T13_41_41_bottom_left` | Arena 2 | **Done (Verified)** |
| **T3** | `16w_PI_T3_..._Arena3_BotCamVid_2026-03-05T13_41_41` | `Arena1234_CenterTopCamVid_2026-03-05T13_41_41_bottom_right` | Arena 3 | **Done (Verified)** |
| **T4** | `16w_PI_T4_..._Arena1_BotCamVid_2026-03-06T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-06T11_01_01_top_left` | Arena 1 | **Done (Verified)** |
| **T4** | `16w_PI_T4_..._Arena2_BotCamVid_2026-03-06T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-06T11_01_01_bottom_left` | Arena 2 | **Done (Verified)** |
| **T4** | `16w_PI_T4_..._Arena3_BotCamVid_2026-03-06T11_01_01` | `Arena1234_CenterTopCamVid_2026-03-06T11_01_01_bottom_right` | Arena 3 | **Done (Verified)** |
| **T5** | `16w_PI_T5_..._Arena1_BotCamVid_2026-03-06T12_30_30` | `Arena1234_CenterTopCamVid_2026-03-06T12_30_30_top_left` | Arena 1 | **Done (Verified)** |
| **T5** | `16w_PI_T5_..._Arena2_BotCamVid_2026-03-06T12_30_30` | `Arena1234_CenterTopCamVid_2026-03-06T12_30_30_bottom_left` | Arena 2 | **Done (Verified)** |
| **T5** | `16w_PI_T5_..._Arena3_BotCamVid_2026-03-06T12_30_30` | `Arena1234_CenterTopCamVid_2026-03-06T12_30_30_bottom_right` | Arena 3 | **Done (Verified)** |
| **T6** | `16w_PI_T6_16_20_23_060326_active_Arena1_BotCamVid_2026-03-06T14_44_36` | `Arena1234_CenterTopCamVid_2026-03-06T14_44_36_top_left` | Arena 1 | **Done (Verified)** |
| **T6** | `16w_PI_T6_16_20_23_060326_active_Arena2_BotCamVid_2026-03-06T14_44_36` | `Arena1234_CenterTopCamVid_2026-03-06T14_44_36_bottom_left` | Arena 2 | **Done (Verified)** |
| **T6** | `16w_PI_T6_16_20_23_060326_active_Arena3_BotCamVid_2026-03-06T14_44_36` | `Arena1234_CenterTopCamVid_2026-03-06T14_44_36_bottom_right` | Arena 3 | **Done (Verified)** |
| *T7* | `16w_PI_T7_21_22_060326_active_Arena1_BotCamVid_2026-03-06T16_15_25` | `Arena1234_CenterTopCamVid_2026-03-06T16_15_25_top_left` | Arena 1 | *In-Flight (Video 19)* |

> [!NOTE]
> In all pairs, **Arena 4 (`top_right`)** is intentionally empty (only 3 photometry channels exist on this rig). The 18 completed pairs represent **5.3 million frames** of dual-synchronized behavioral video data.

---

## 2. Leveraging `sessions_karen_project_w_sylls.pkl`

The session pickle `sessions_karen_project_w_sylls.pkl` (13.68 GB, 110 sessions) contains rich, pre-aligned data structures that will directly accelerate this experiment:

1. **`session.session_id` & `session.results_dict_key`**:
   - Maps 1-to-1 with the timestamp strings and arena quadrant suffixes (e.g. `..._2026-03-06T14_44_36_bottom_left`).
2. **`session.all_data_df`**:
   - Stores the master synchronized timeline, including fiber photometry (`combined_filtered`), baseline TopCam speeds, angular velocities, and existing dorsal syllables.
3. **`session.video_refs_df`**:
   - Stores the primary TopCam AVI path, exact container frame counts, and frame rates (60.0 FPS).
4. **`session.syllables`**:
   - Stores the baseline TopCam-only syllable sequence (`final_karen_model`), allowing a direct before-and-after comparison against our multi-view model!

---

## 3. Mathematical Alignment & Fusion Architecture

### 3.1 The Geometry Challenge
- **TopCam**: Camera looking down at chamber ($972 \times 972$ arena quadrant). Tracks dorsal landmarks (head, spine, tail-base). Paws are occluded.
- **BotCam**: Camera looking up through the glass floor ($972 \times 972$ arena viewport). Tracks ventral landmarks: `nose_tip`, `left_front_paw`, `right_front_paw`, `left_hind_paw`, `right_hind_paw`, `tail_base`.
- Both are 60.0 FPS and share identical timestamps.

### 3.2 Egocentric Body-Centered Transformation
Because Keypoint-MoSeq models body-centered posture $\bar{Y}_t = R(-\theta_t)(Y_t - c_t)$, we align BotCam paws into TopCam space via the shared rostral-caudal body axis:

```
[TopCam Dorsal Tracking]                             [BotCam Ventral Tracking]
  - Centroid: c_top                                    - Centroid: c_bot
  - Heading: θ_top (tail_base -> nose)                 - Heading: θ_bot (tail_base -> nose_tip)
           │                                                    │
           ▼                                                    ▼
  Dorsal Egocentric Frame                              Ventral Egocentric Frame
  Y_ego_top = R(-θ_top) * (Y_top - c_top)              Y_ego_bot = R(-θ_bot) * (Y_bot - c_bot)
           │                                                    │
           └─────────────────────┬──────────────────────────────┘
                                 │
                     [ Scale Matching & Concat ]
                                 │
                                 ▼
                     Unified 23-Keypoint Pose
                     Y_ego_fused = [Y_ego_top; Y_ego_bot]
                                 │
                                 ▼
                     [ Re-project to Arena Frame ]
                     Y_virtual = c_top + R(+θ_top) * Y_ego_fused
```

$$\bar{Y}_t^{\text{bot\_ego}} = R(-\theta_t^{\text{bot}}) \left(Y_t^{\text{bot}} - c_t^{\text{bot}}\right) \times s$$
where $s = \frac{\text{median}(\|Y^{\text{top}}_{\text{nose}} - Y^{\text{top}}_{\text{tail}}\|)}{\text{median}(\|Y^{\text{bot}}_{\text{nose}} - Y^{\text{bot}}_{\text{tail}}\|)}$ ensures identical anatomical scale.

Translocating paws into the virtual TopCam frame:
$$Y_t^{\text{paws\_virtual}} = c_t^{\text{top}} + R(+\theta_t^{\text{top}}) \bar{Y}_t^{\text{bot\_ego}}$$

Now, all 21/23 keypoints (17 dorsal + 4 ventral paws) reside in a single coordinate system, ready for standard Keypoint-MoSeq without requiring custom C++/JAX engine patches!

---

## 4. Hardware Clock Synchronization Architecture

As confirmed by empirical tests across continuous 50+ minute recordings in `src_mono_inter/jonathan_juvenile_intruder/csvs/`, the system shares a common master timer:
* `photometry_data_combined_*.csv`: logs `SystemTimestamp`
* `CenterTopCam_AllTimestampsCombined*.csv`: logs `FP3002_System_Timestamp`
* `BotCam1_AllTimestampsCombined*.csv`: logs `FP3002_System_Timestamp`

Across trials, $N_{\text{frames\_top}}$, $N_{\text{frames\_bot}}$, and $N_{\text{phot}}$ match to within **0 to 1 frame** ($\le 12.5$ ms edge jitter over 250,000+ frames), with **0.000 ms microsecond timestamp offset**.

### Dual-Layer Synchronization Protocol:
1. **Clock-Hardened Alignment (`clock_sync.py`):**
   - Implements `np.searchsorted` nearest-neighbor lookup on `FP3002_System_Timestamp` (identical to `loaders/jonathan/utils/photometry.py:attach_frame_idx_column`).
   - If either camera drops a frame, it is mapped to `NaN` at that index rather than shifting subsequent frames by -1, preventing temporal phase drift.
2. **Mother Anchor Locking:**
   - The timeline length is anchored to `s.video_refs_df.iloc[0]['n_frames']` (TopCam).
   - BotCam predictions are attached as `role="ventral"` at row 1 of `video_refs_df` and `tracking_refs_df`.

---

## 5. Work Breakdown & Experimental Phases

### Phase 1: Code Backups & Baseline Isolation
* **Code Snapshot:**
  - `syllable_generation/loaders/jonathan/steps/fiberphotometry.py` $\to$ `fiberphotometry_baseline_backup.py`
  - `syllable_generation/loaders/jonathan/steps/video.py` $\to$ `video_baseline_backup.py`
  - `syllable_generation/loaders/jonathan/steps/tracking.py` $\to$ `tracking_baseline_backup.py`
  - `syllable_generation/analysis/replications/datta_utils.py` $\to$ `datta_utils_baseline_backup.py`
* **Data Isolation:**
  - Gold-standard `sessions_karen_project_w_sylls.pkl` (13.68 GB) remains untouched.
  - Multi-view output writes strictly to `multiview_top_bot_experiments/sessions_multiview_pilot.pkl`.

### Phase 2: Clock-Hardened Pairing & Session Enrichment
* **Matching & Pairing (`01_match_pairs.py`):**
  - Cross-references the 18 completed BotCam runs with `sessions_karen_project_w_sylls.pkl`.
  - Verifies $\Delta(\text{Top} - \text{Bot}) \le 1$ frame.
* **Non-Breaking Enrichment (`enrich_sessions_with_botcam.py`):**
  - Appends BotCam video to `s.video_refs_df` (`role="ventral"`) and tracking to `s.tracking_refs_df` (`role="ventral"`).
  - Pins TopCam at Row 0 (`role="primary"`), preserving 100% backwards compatibility for legacy tools.

### Phase 3: Clock-Synchronized Egocentric Pose Fusion (`02_fuse_dorsal_ventral.py`)
* Synchronizes BotCam and TopCam tracking frames via `FP3002_System_Timestamp`.
* Normalizes scale $s = \text{median}(\|Y^{\text{top}}_{\text{body}}\|)\,/\,\text{median}(\|Y^{\text{bot}}_{\text{body}}\|)$.
* Projects egocentric ventral paws into TopCam virtual arena space.
* Exports unified 21-keypoint HDF5 files (`fused_h5_predictions/`).

### Phase 4: CPU-Only Pilot KPMS Fitting on a Small Sliver
* **GPU Isolation Protocol (Safety for Active RTX 4090 Runner):**
  - `export CUDA_VISIBLE_DEVICES=""`
  - `export JAX_PLATFORMS="cpu"`
  - `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
* **Data Sliver Selection:**
  - Select 2 representative sessions (e.g. `T1_Arena1` and `T1_Arena2`).
  - Slice a test window of 5,000–10,000 active frames.
* **Sanity Checks & Shape Validation (`03_test_kpms_shapes_cpu.py`):**
  - **Data Loader Check:** Verify keypoint tensors shape `(N, 21, 2)` and confidence shape `(N, 21)`.
  - **PCA Fitting:** Fit PCA ($D=10$) on CPU; verify eigenvalues and eigenposes (dorsal spine + paw phase).
  - **AR-HMM Initialization:** Run `kpms.init_model` on CPU.
  - **Fast Execution Check:** Run 2 test iterations to confirm likelihood calculation, state transition matrix, and `results.h5` extraction without OOM or shape errors.

### Phase 5: Timeline Weaving & Parity Verification
* Weave model output back onto 80Hz mother timeline using Ground Truth VFI (`sampling.py:build_aligned_dfs`).
* Apply 2-frame forward-fill (`ffill()`) from 40Hz to 80Hz.
* Validate with `check_nframes_match.py` and `verify_weaved_coverage.py`:
  - 100% cardinality match: `len(all_data_df) == n_frames == len(photometry)`.
* Visual overlay verification with `videogen` rendering dual TopCam/BotCam panels with tracking overlays.
