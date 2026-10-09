# Dual-Video (TopCam + BotCam) Session Architecture & Non-Breaking Fusion Plan

**Author:** Antigravity AI  
**Context:** `syllable_generation/` pipeline audit & multi-camera extension  
**Target:** Enable `SessionRecord` objects to house both dorsal (TopCam) and ventral (BotCam) synchronized videos and tracking data without disrupting existing single-video workflows.

---

## 1. The Lifecycle: How `sessions_karen_project_w_sylls_k240k.pkl` Comes to Be

To design a non-breaking multi-camera extension, we audited the exact file-by-file journey through the codebase:

```
[Raw TopCam AVI & DLC H5]  +  [Photometry CSVs]
             │
             ▼
1. Discovery & Grouping (`syllable_generation/loaders/jonathan/steps/`)
   - `tracking.py`: Parses H5 filename -> extracts `ts` and `chamber` -> `session_id = f"{ts}_{chamber}"`
   - `video.py`: Probes video metadata (fps, n_frames) -> maps video to `session_id` -> attaches `s.video_refs_df`
   - `fiberphotometry.py`: Time-locks photometry ticks to camera frames -> computes dF/F
             │
             ▼
2. Session Record Instantiation (`syllable_generation/datamodel.py`)
   - Creates `SessionRecord` object with `tracking_refs_df`, `video_refs_df`, `photometry`
             │
             ▼
3. Metadata Enrichment (`syllable_generation/loaders/build_sessions_unified.py`)
   - Ingests genotype (WT vs. KI), drug treatment (Saline vs. L-DOPA/Amph), trial numbers
   - Serializes out: `syllable_generation/data/sessions_karen_project_dirty.pkl` (110 sessions)
             │
             ▼
4. KPMS Syllable Modeling (`syllable_generation/syllabizers/keypoint_moseq_syll/`)
   - `run_pipeline.py` loads `sessions_karen_project_dirty.pkl`
   - Dispatches tracking paths via `utils/paths.py` (`_subroutine_jonathan`)
   - Fits PCA and AR-SLDS in `karen_project_2/final_karen_model_k240k/`
   - Saves: `checkpoint.h5` and `results.h5`
             │
             ▼
5. Syllable Weaving & Final Attachment (`syllable_generation/analysis/`)
   - `generate_k240k_results_and_pkl.py` (or `attach_karen_final_model.py`)
   - Maps `results.h5` keys to `session.session_id`
   - Weaves 40Hz model syllables back onto 80Hz/60Hz timelines (`s.syllables`, `s.all_data_df`)
   - Produces final master artifact:
     `syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl`
```

---

## 2. Codebase Audit: The Good News About `SessionRecord`

Our inspection of [`syllable_generation/datamodel.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/datamodel.py) reveals that the session architecture **was already designed to hold multiple videos and tracking files**:

1. **`s.video_refs_df` is already a DataFrame (not a single string path)**:
   In `syllable_generation/loaders/jonathan/steps/video.py`:
   ```python
   index: dict[str, list[dict]] = {}
   for p in gather_video_paths(video_dir):
       ...
       index.setdefault(info["session_id"], []).append(row)
   
   for s in sessions:
       rows = index.get(s.session_id, [])
       if rows:
           s.video_refs_df = pd.DataFrame(sorted(rows, key=sort_video_row_key))
   ```
   `index[session_id]` is already an appendable list!
2. **`s.tracking_refs_df` is already a DataFrame**:
   It holds tracking paths keyed by `camset` and `role`.
3. **`s.tracking` is a Dictionary**:
   `s.tracking: Dict[str, pd.DataFrame]` can house multiple pose tables (e.g. `s.tracking["topcam"]`, `s.tracking["botcam"]`, `s.tracking["fused_multiview"]`).

---

## 3. The Minimal Changes Needed to Ingest BotCam Videos

Currently, BotCam files are ignored during standard loading because `camset_to_slot` and `camera_token_map` in `loaders/jonathan/config.py` do not know how to parse them.

### Change A: Update `JonathanConfig` (`loaders/jonathan/config.py`)
Add `botcamvid` to the camera slot and token mapping dictionaries:

```python
# In JonathanConfig:
camset_to_slot = {
    "centertopcamvid": "center_cam",
    "lefttopcamvid":   "left_cam",
    "righttopcamvid":  "right_cam",
    "botcamvid":       "bottom_cam",   # <--- NEW
}

camera_token_map = {
    "lefttopcamvid":  {"upper": "top_left",  "lower": "bottom_left"},
    "righttopcamvid": {"upper": "top_right", "lower": "bottom_right"},
    "botcamvid": {                      # <--- NEW (Arena hardware mapping)
        "arena1": "top_left",
        "arena2": "bottom_left",
        "arena3": "bottom_right",
        "arena4": "top_right",
    }
}
```

### Change B: Update `choose_role` (`loaders/jonathan/utils/files.py`)
```python
def choose_role(cam_slot):
    if cam_slot == "center_cam":
        return "primary"
    elif cam_slot == "bottom_cam":
        return "ventral"                # <--- NEW
    else:
        return "aux"
```

### What Happens Automatically:
When `attach_video_refs` runs on a folder containing both TopCam and BotCam:
- Both videos parse to the **exact same `session_id`** (e.g. `2026-03-05T11_01_01_top_left`).
- `s.video_refs_df` naturally receives **two rows**:
  - **Row 0**: `path`: `.../CenterTopCamVid_...avi`, `role`: `"primary"`, `cam_slot`: `"center_cam"`
  - **Row 1**: `path`: `.../BotCamVid_...avi`, `role`: `"ventral"`, `cam_slot`: `"bottom_cam"`
- `sort_video_row_key` ensures `role == "primary"` is **strictly Row 0**.

---

## 4. The Concat / Fusion System: How It Intervenes

Once a session has two synchronized videos and tracking files attached, we implement an automatic **Fusion Hook**:

### Detection Rule
```python
def is_multiview_session(s: SessionRecord) -> bool:
    if s.video_refs_df is None or s.video_refs_df.empty:
        return False
    roles = set(s.video_refs_df["role"].unique())
    return "primary" in roles and "ventral" in roles
```

### The Concat Action
If `is_multiview_session(s)` is `True`:
1. **Verification**: Check frame counts (`n_frames_primary == n_frames_ventral`). Both are 60.0 FPS.
2. **Egocentric Transform**:
   - Extract dorsal body heading $\theta_{\text{top}}$ and centroid $c_{\text{top}}$ from TopCam tracking.
   - Extract ventral paws from BotCam tracking and transform them into the canonical body frame:
     $$paws_{\text{ego}} = R(-\theta_{\text{bot}}) (paws_{\text{bot}} - c_{\text{bot}}) \times \text{scale}$$
   - Translocate paws into the TopCam virtual arena frame:
     $$paws_{\text{virtual}} = c_{\text{top}} + R(+\theta_{\text{top}}) paws_{\text{ego}}$$
3. **Storage in Session**:
   - The unified 23-keypoint table is saved directly into `s.tracking["multiview"]` (or a fused H5 file path appended to `s.tracking_refs_df` with `role: "fused"`).

### Dispatch in KPMS (`syllabizers/keypoint_moseq_syll/utils/paths.py`)
In `ref_pairs_from_sessions`:
```python
def _subroutine_jonathan(sessions, *, cam="center_cam"):
    # If cam="multiview", look for role="fused" or generate on-the-fly.
    # If cam="center_cam", look for role="primary" (100% legacy behavior).
```

---

## 5. Mathematical & Modeling Nuances: Heading, Anchors, PCA & Reference Frames

### 5.1 Dual-Role Anchors: Alignment vs. KPMS Heading
There are two distinct uses of the nose and tail-base keypoints:
1. **Camera Alignment (Fusion)**: Our fusion script uses $c_{\text{bot}}$ and $\theta_{\text{bot}}$ to rotate the ventral paws onto the dorsal body axis.
2. **KPMS Heading Initialization**: KPMS uses `anterior_bodyparts` and `posterior_bodyparts` configured in `config.yaml` to initialize a single shared heading $\theta_t$. During full-model fitting, this heading can be updated by the continuous latent model.
   - **Recommended Configuration**: Initialize KPMS heading exclusively using TopCam anchors (`anterior_bodyparts: ["nose"]`, `posterior_bodyparts: ["tail_base"]`), because TopCam defines the master arena coordinate frame.
   - **Transformed Ventral Anchors**: The transformed BotCam nose and tail base (`bot_nose_tip`, `bot_tail_base`) can still be included as observed keypoints in the 23-point model. However, because their axis was already rotated onto the TopCam axis by construction, they do *not* provide independent confirmation of heading.
   - **Anchor Accuracy Vulnerability**: If the BotCam nose detection glitches, the fusion rotation $R(-\theta_{\text{bot}})$ will rotate the entire ventral pose incorrectly—even when every paw was tracked with high precision.

### 5.2 Homologous Centroids Preserve Paw Excursions
KPMS naturally removes global translation and heading. However, removing independent transforms per view requires care:
- If we centered BotCam on the mean position of the paws, we would remove genuine forward/backward and lateral shifts of the paws relative to the torso.
- By defining homologous midpoints:
  $$c_{\text{top}} = \frac{1}{2}(\text{nose}_{\text{top}} + \text{tail}_{\text{top}}), \quad c_{\text{bot}} = \frac{1}{2}(\text{nose}_{\text{bot}} + \text{tail}_{\text{bot}})$$
  both coordinate systems are pegged to the same anatomical rostral-caudal midpoint. Reprojecting via $c_{\text{top}} + R(+\theta_{\text{top}}) paws_{\text{ego}}$ preserves true paw excursions relative to the body center.

### 5.3 PCA Variance Selection & Paw Dynamics
- **Keypoint Count vs. Information**: Having 17 dorsal points and 6 ventral points does *not* imply dorsal dominance. Highly correlated dorsal spine points collapse into few principal components (empirically, PC01–PC02 capture 78.9% variance, representing spine posture).
- **Variance Selection Limitation**: Standard PCA selects directions solely by total variance, not per-view balance. Whitening rescales retained components but cannot resurrect information dropped outside the retained $D=10$ subspace.
- **Empirical Check (Benchmark C)**: In our empirical analysis, paw variance was large enough that PC03–PC10 were 91–97% dominated by ventral paws (stride and gait).

---

## 6. Backwards Compatibility Proof: Why Old Stuff Won't Break

We audited every legacy script in the repo to guarantee single-camera sessions are unaffected:

| Component | How It Accesses Session Video / Tracking | Multi-View Impact |
| :--- | :--- | :--- |
| **`video_refs_df.iloc[0]`** | Legacy scripts grab row 0 of `video_refs_df`. | **Zero impact**: `sort_video_row_key` guarantees `role == "primary"` is always row 0. |
| **`role == "primary"`** | Filtering by `df["role"] == "primary"` grabs TopCam. | **Zero impact**: The primary row is identical. |
| **Single-Video Sessions** (AMPH, etc.) | Sessions with only one camera have `len(video_refs_df) == 1`. | **Zero impact**: `is_multiview_session` evaluates to `False`; no fusion step runs. |
| **Pickle Serialization** | `SessionRecord` uses standard dataclass fields (`video_refs_df`, `tracking`). | **Zero impact**: Python `pickle` deserializes old and new objects with 100% binary compatibility. |

---

## 7. Validation Protocol, Confidence Calibration & "Stuff to Try"

> [!TIP]
> The items below represent recommended diagnostic checks and experimental extensions to validate and refine the multi-view pipeline.

### 7.1 Diagnostic Audit: Per-View Reconstruction & Excursion Checks ("Stuff to Try")
- [ ] **Reconstruction Excursion Test**: Reconstruct the aligned 23-keypoint poses using the retained $D=10$ principal components ($\hat{Y} = \bar{x} C^T$). Inspect the paws separately: verify that stance/swing cycles, step frequencies, and grooming excursions cleanly survive reconstruction.
- [ ] **Per-View Variance Explained**: Compare per-view reconstruction error relative to each view's raw movement variance on held-out sessions:
  $$\text{Error Ratio}_{\text{view}} = \frac{\sum_{k \in \text{view}} \|Y_k - \hat{Y}_k\|^2}{\sum_{k \in \text{view}} \text{Var}(Y_k)}$$
  If the ventral paw error ratio is significantly worse than dorsal, consider increasing latent dimension $D$ (e.g. $D=12$ or $D=14$) or testing variance-standardized PCA.

### 7.2 Paired-Frame Annotation Audit (100–200 Frames) ("Stuff to Try")
- [ ] **Manual Ground-Truth Sample**: Annotate 100–200 paired frames across diverse behaviors (locomotion, rearing, grooming, still) and confidence levels, specifically including borderline/failure cases. Include TopCam and BotCam nose/tail anchors plus all 4 ventral paws.
- [ ] **Common-Unit Error Metrics**: Compare errors in common physical units using the scale factor $s$: a BotCam positional error of $e$ pixels corresponds to $s \cdot e$ in TopCam pixels.
- [ ] **Isolate Fusion Error vs. DLC Error**:
  - Compare manual annotations through manual anchor fusion vs. predicted coordinates through predicted anchor fusion. This isolates whether tracking errors or heading rotation errors dominate.

### 7.3 Confidence Calibration & Noise Prior Modeling ("Stuff to Try")
- [ ] **Single vs. Dual Calibration Curve**: KPMS calibration (`keypoint_moseq.calibration`) fits a single global polynomial/affine relationship between DLC confidence and coordinate variance $\sigma_i^2$.
  - Evaluate whether one confidence-to-error curve adequately fits both the TopCam and BotCam models.
  - If they diverge, evaluate a preprocessing lookup table mapping each camera's raw likelihood to an expected standard error in common pixels before passing to KPMS.
- [ ] **Correlated Anchor Uncertainty Propagation**:
  - If BotCam nose or tail confidence drops below a threshold (e.g. $p < 0.6$), downweight or flag the confidence of the entire ventral paw block for that frame to prevent noisy rotation matrices from corrupting otherwise well-tracked paws.

---

## 8. Implementation Action Plan

1. **Step 1 (`loaders/jonathan/config.py`)**: Add `botcamvid` token mapping so both videos attach to `s.video_refs_df`.
2. **Step 2 (`multiview_top_bot_experiments/session_enricher.py`)**: A non-destructive script that takes existing `sessions_karen_project_w_sylls.pkl`, checks which sessions have completed BotCam predictions from our production run, and attaches the second video row (`role: "ventral"`) and fused tracking table.
3. **Step 3 (Validation)**: Run a test assertion confirming that for single-camera sessions, `s.video_refs_df` remains 1 row, while for paired sessions, it has 2 rows and the fused pose coordinates match.
