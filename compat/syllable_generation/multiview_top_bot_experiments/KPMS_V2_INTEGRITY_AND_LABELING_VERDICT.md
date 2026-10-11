# Comprehensive Production Audit & Verdict: KPMS Multiview Input Integrity, Density Selection, and Labeling

**Investigation Target:** Keypoint-MoSeq (KPMS) Tandem Multiview Architecture `21kp_static` (17 dorsal + 4 bottom paws)  
**Cohort & Model:** `cohort_static_21kp_v2` / `model_21kp_static_50ar_300slds_seed101`  
**Execution Environment:** `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python` (KPMS v0.6.7, JAX v0.3.3)  
**Date of Audit:** October 7, 2026  
**Audited Artifacts Directory:** `syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/visualizations_cohort_static_21kp_v2_seed101_v4/`

---

## 1. Executive Verdict & Four-Quadrant Taxonomy

This audit rigorously evaluates all hypothesized pipeline defects against actual workstation production data (118 fused sessions, 23.86 million frames, and the trained seed101 model). We distinguish findings across four categories: **Confirmed Defects**, **Verified Repairs**, **Measured Limitations**, and **Disproven Hypotheses**.

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│             CONFIRMED DEFECTS                │               VERIFIED REPAIRS               │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ 1. Selector/Movie Context Mismatch (Tile 4)  │ 1. 180-frame movie context eligibility gate  │
│ 2. Unrotated Bottom-Cam Paw Centroid fallback│ 2. Parity-preserving homogeneous composition  │
│ 3. Trajectory y-axis inverted & time-warped  │ 3. Image-coordinate (y-down) native-time MP4 │
│ 4. Missing Fused DLC silent QC bypass        │ 4. Fail-closed require_fused_qc assertion    │
│ 5. Stale gallery cache & sidecar mutation    │ 5. SHA-256 multi-input cryptographic sidecar │
│ 6. In-place reindexing syllable key swap     │ 6. Hardened io_utils preserving exact keys   │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│            MEASURED LIMITATIONS              │        DISPROVEN / UNTESTED HYPOTHESES       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ 1. Paw tracking confidence bottleneck (QC)   │ 1. Rampant ingestion fragmentation mechanism │
│ 2. State-specific dorsal loss (Syllable 36)  │    (Hypothesis: >50% lost; Reality: 0.248%)  │
│ 3. 4-PC projection vs Local point RMS gap    │ 2. Model anterior-posterior "corruption"     │
│ 4. Fixed 180° inverted heading initialization│    (Reality: Consistent 178.4° convention,   │
│ 5. Photometry DAQ vs Video clock boundaries  │    circ_var 0.0042, not time-varying flips)  │
│                                              │ 3. Chatbot speed thresholds / 14 macro groups│
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

### Scope Impact Matrix
- **Rendering:** Repairs 1, 2, 3, 5 eliminate tile blanking, incorrect bottom-camera rotations, y-axis inversions, and false frame extensions.
- **Selection:** Repair 1 & 4 guarantee that every selected bout has valid video bounds and verified DLC tracking.
- **Training Inputs:** Audits prove that production training segments used `MIN_SEGMENT_FRAMES = 600` (10.0s @ 60 FPS), retaining 79.33% of raw frames (18.93M frames) across 338 segments. Frame loss due to runs < 600 frames is only 0.248%. Retraining is NOT required.
- **Photometry Timing:** Acquisition timestamps and segment boundaries must be preserved. Video frame interpolation (VFI) cannot be cross-referenced with analog DAQ timestamps without the master synchronization clock offset.

---

## 2. Production Input Fingerprints & Test Results

### Cryptographic Fingerprints
The canonical production model inputs were hashed and verified:
- **`results.h5`**: `31c30efadab668e5d26078d3f62627412ced32522cb01f0858c290c96e53c09c`
- **`config.yml`**: `40dd8df272f54f2de0ac61247825ccd64ba8a1b818bae833c91f0f0383622aab`
- **`video_frame_indexes.npz`**: `00bdab5a0ed36018b39a9dc1290b066a8692c79562a289dcf26a8ecc2504c4c9`
- **`preprocessed_data.pkl`**: `92c1a05bf59b70193c92ff2c30fa3fa3b1414e90a46692d0bf592a650742b1f5`

### Test Suite Execution
30 unit tests were executed against the active repository implementation in the KPMS environment (`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`):
```bash
python -m unittest tests/test_v2_visualization_repairs.py tests/test_v4_repairs.py tests/test_io_utils_hardening.py
Ran 30 tests in 1.536s - OK
```
Additionally, all 7 KPMS checkpoint compaction and transparent monkeypatching tests passed:
```bash
python -m unittest tests/test_kpms_checkpoint_compaction.py
Ran 7 tests in 32.054s - OK
```

---

## 3. Detailed Identity Probe Results (Stage 2)

Probed target syllables: **`[1, 67, 30, 70, 92, 36, 42]`** across production fused HDF5 files and video recordings.

```
┌───────────┬─────────┬──────────────┬──────────────┬─────────────┬─────────────┬──────────────┬──────────────┐
│ Syllable  │  Tier   │  Qualifying  │  QC Eligible │  Paw Rej %  │ Dorsal Rej% │ Video Seek Δ │ Max Coord Err│
├───────────┼─────────┼──────────────┼──────────────┼─────────────┼─────────────┼──────────────┼──────────────┤
│ 1         │ Tier 1  │    12,744    │    1,193     │    68.4%    │    20.0%    │    0 px      │   0.000 px   │
│ 67        │ Tier 1  │     1,452    │      248     │    51.0%    │    30.3%    │    0 px      │   0.000 px   │
│ 30        │ Tier 3  │     5,654    │        4     │    91.2%    │     7.7%    │    0 px      │   0.000 px   │
│ 70        │ Tier 2  │     1,748    │       48     │    55.7%    │    31.9%    │    0 px      │   0.000 px   │
│ 92        │ Tier 3  │       825    │       11     │    65.7%    │    23.6%    │    0 px      │   0.000 px   │
│ 36        │ Tier 0  │       330    │        0     │     5.8%    │    93.9%    │    0 px      │   0.000 px   │
│ 42        │ Tier 1  │    17,621    │    8,948     │    48.8%    │     1.0%    │    0 px      │   0.000 px   │
└───────────┴─────────┴──────────────┴──────────────┴─────────────┴─────────────┴──────────────┴──────────────┘
```

### Key Proofs
1. **Raw-to-Fused Projection Identity:** In 100% of tested instances, the recorded affine transform $p_{\text{top}} = A [W - q_{\text{bot}.x}, q_{\text{bot}.y}]^T + b$ matches the fused coordinates in `results.h5` with **$0.000$ px error** and **$0.000$ likelihood error**.
2. **Video Decoding:** Direct frame seeking (`cap.set(POS_FRAMES, f)`) matches nearby sequential decoding with bit-for-bit frame identity (0 diff frames, 0 px error).
3. **Movie & Trajectory ID Agreement:** Nonblank movie tiles, superimposed trajectory traces, and side-by-side trajectory traces match the unified selection manifest in 100% of cases (`movie IDs == traj IDs == manifest IDs`).

---

## 4. Cohort Ingestion Retention Audit (Stage 3)

The hypothesis that `split_dual_view_segments` with `MIN_SEGMENT_FRAMES = 180` caused rampant data loss across the cohort was thoroughly tested across all 118 sessions:

- **Total Raw Frames:** 23,861,208 frames (110.47 hours @ 60 FPS).
- **Supported Raw Frames (at least one valid paw):** 18,977,003 frames (79.53%).
- **Production Segments in `results.h5`:** Exactly 338 segments.
- **Production Retained Frames:** 18,929,934 frames (79.33% of raw frames).
- **Frames Lost to Minimum Segment Length (600 frames = 10.0s):** Only **47,069 frames (0.248% of supported frames)**.
- **Finite-Y Divergence:** 0 frames across the entire dataset.

**Verdict:** The hypothesized fragmentation mechanism is non-existent in production. Retraining with bounded gap bridging would recover less than 0.25% additional data and is not scientifically urgent.

---

## 5. QC Policy Comparison: Strict vs. Forepaw-Limited (Stage 4)

We compared the canonical reference policy (`strict_all_paws`: dorsal $\ge 0.8$, paw $\ge 0.8$, gap $\le 0.1\text{s}$) against the opt-in experimental policy (`dorsal_and_hind_supported_forepaw_limited`: dorsal $\ge 0.8$, hindpaws $\ge 0.8$, relaxed forepaws):

```
┌──────────┬──────────────────────┬──────────────────────────┬────────────────────────┐
│ Syllable │ Strict Eligible Pool │ Forepaw-Limited Eligible │ Scientific Mechanism   │
├──────────┼──────────────────────┼──────────────────────────┼────────────────────────┤
│ 1        │        1,193         │          7,164           │ Front paw loss (54%)   │
│ 67       │          248         │            814           │ Front paw loss (51%)   │
│ 30       │            4         │          4,432           │ Rescued: paws lifted?  │
│ 70       │           48         │            620           │ Upgraded Tier 2 -> 1   │
│ 92       │           11         │            222           │ Upgraded Tier 3 -> 1   │
│ 36       │            0         │              7           │ Dorsal loss dominates  │
│ 42       │        8,948         │         14,851           │ Abundant locomotion    │
└──────────┴──────────────────────┴──────────────────────────┴────────────────────────┘
```

### Critical Scientific Constraints
- **Preserve Canonical Gallery:** All canonical labeling galleries and manifests retain the strict 4-paw policy. The forepaw-limited policy is saved as a separate experimental record (`qc_policy_comparison.json`).
- **Low Forepaw Confidence is NOT Proof of Rearing/Grooming:** While Syllable 30 displays a dramatic rescue (4 -> 4,432 bouts) due to front paws dropping below confidence 0.5, low confidence cannot be certified as ground contact or lifting without ethologist visual review. Unsupported keypoints remain NaN in rendering and distance metrics.

---

## 6. Kinematics, Heading, and Density Diagnostics (Stage 5)

### Heading Orientation
- Probing raw torso vector (tail_base -> head_midpoint/neck) against model heading revealed a **consistent 180° inverted convention** (mean residual: $178.4^\circ$, circular variance: $0.0042$, 99.81% of frames within $30^\circ$ of $180^\circ$, 0.0% near $0^\circ$).
- This confirms a constant coordinate definition chosen during PCA/training initialization, not time-varying sign flips or tracking glitches. Existing results must NOT be inverted or flipped.

### Kinematic Estimators (Explicit 6-frame / 100 ms Denominator)
- Median filtered centroid ($[-3, +3]$ stencil) with $\Delta t = 0.100\text{s}$ at 60 FPS:
  - **Syllable 67:** Median speed $10.9\text{ px/s}$, angular vel $0.49\text{ rad/s}$, chord $28.2\text{ px}$ (Stationary/Pause).
  - **Syllable 1:** Median speed $16.1\text{ px/s}$, angular vel $0.64\text{ rad/s}$, chord $32.3\text{ px}$ (Low locomotion).
  - **Syllable 30:** Median speed $19.3\text{ px/s}$, angular vel $0.55\text{ rad/s}$, chord $40.1\text{ px}$ (Elongated body posture).
  - **Syllable 36:** Median speed $62.0\text{ px/s}$, angular vel $1.57\text{ rad/s}$, chord $38.4\text{ px}$ (Sharp turn).
  - **Syllable 42:** Median speed $110.3\text{ px/s}$, angular vel $0.69\text{ rad/s}$, chord $49.6\text{ px}$ (High-speed forward locomotion).

### Density Selection vs. Upstream KPMS
- Upstream `keypoint_moseq.viz.sample_instances` (v0.6.7) crashes if $N_{\text{eligible}} < G$, mutates coordinates via linear interpolation, and draws $G$ random samples from the 50 neighbors without guaranteeing the exemplar is tile 0.
- Our installed selector guarantees the exemplar at tile 0, sorts neighbors by nearest distance, flags rare-state shortages with explicit blank tiles, and preserves raw DLC missingness.

---

## 7. Artifacts & Generated Files

All deliverables reside on the workstation:
- **Canonical v4 Visualizations:** `04_reports_and_figures/visualizations_cohort_static_21kp_v2_seed101_v4/`
  - Movies: `grid_movies_density/syllable_XX_tandem_grid_density.mp4`
  - Trajectories: `trajectories/superimposed/` and `trajectories/side_by_side/`
  - Gallery HTML: `syllable_labeling_gallery_cohort_static_21kp_v2_seed101.html`
  - Selection Manifest: `unified_syllable_selection_manifest.json`
- **Probe Reports:**
  - `active_similarity_probe.json`
  - `qc_policy_comparison.json`
  - `kinematics_and_heading_probe.json`
  - `audits/ingestion_retention_audit.json`
  - `bound_model_annotations.json`
- **Hardened Checkpoint Engine:** `core/io_utils.py` (tested by `tests/test_io_utils_hardening.py`)
- **Compact Package Zip:** `v4_kpms_integrity_and_labeling_deliverable.zip` (excludes heavy MP4s and checkpoints).
