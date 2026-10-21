# Preflight Audit: Retained Dual-View Coverage & Safeguard Report

**Date & Time:** 2026-10-01 17:24:00  
**Target Project:** `kpms_tandem_cohort_continuous_calibrated`  
**Execution Environment:** `kpms_gpu_cuml`  

---

## 1. Targeted Artifact Safeguard Results

### A. January Screw Failure (`2026-01-09T12_19_47_top_left`)
* **Glitch Window:** Frames 94720–94750 (DLC TopCam snapped to arena screw, nose–tail length = 216 px > 180 px bound).
* **TopCam Anchor Plausibility Gating:** **PASSED** (5 frames gated to $p = 0.000$, coordinates set to `NaN`).
* **Anchor Confidence Bleed Prevention:** Untrusted anchor coordinates are completely suppressed; phantom paw projections into arena perimeter prevented.

### B. December Real Six-Frame Basler Gap (`2025-12-03T14_06_45_bottom_right`)
* **Gap Window:** Frames 186367–186372 (BotCam dropped exactly 6 frames during recording).
* **Continuous Rate-Aware Regression:** **PASSED** (Mapping accurately encodes 6 consecutive `-1` dropped markers: `[186366, -1, -1, -1, -1, -1, -1, 186367]`).
* **Post-Gap Synchronization:** Post-gap frames resume monotonically without cumulative drift across 195,134 frames (Frame 186373 $	o$ 186367).

### C. March Prolonged Missing Ventral Data & Abort (`2026-03-15T09_51_36_bottom_left`)
* **Prolonged Missing Run Exclusion:** **PASSED** (Initial 40,999 missing-ventral frames excluded; first retained segment starts at frame 40999).
* **Sequence Splitting:** Formed 49 independent continuous dual-view segments (23,292 genuine dual-view frames).
* **Artificial Temporal Join Prevention:** Zero frames concatenated across the 41k-frame gap; each segment carries independent VFI ground-truth mappings.

---

## 2. Retained Dual-View Coverage by Session and Condition

| Condition | Session ID | Arena | Raw Frames | Sync Valid | Segments | Retained Frames | Coverage (%) | Excluded Frames |
|:---|:---|:---:|---:|:---:|---:|---:|---:|---:|
| 8_weeks_PI | `2026-01-09T12_19_47_top_left` | Arena1 | 188,426 | PASS | 215 | 137,799 | **73.13%** | 50,627 |
| 3_week_PI | `2025-12-03T14_06_45_bottom_right` | Arena3 | 195,134 | PASS | 201 | 154,091 | **78.97%** | 41,043 |
| 16w_PI | `2026-03-15T09_51_36_bottom_left` | Arena2 | 73,249 | PASS | 49 | 23,292 | **31.8%** | 49,957 |
| 3_week_PI | `2025-12-03T09_51_05_bottom_left` | Arena2 | 219,515 | PASS | 439 | 135,808 | **61.87%** | 83,707 |
| 3_week_PI | `2025-12-03T13_02_27_top_left` | Arena1 | 208,852 | PASS | 378 | 135,724 | **64.99%** | 73,128 |
| 8_weeks_PI | `2026-01-09T09_41_59_bottom_left` | Arena2 | 222,982 | PASS | 233 | 158,489 | **71.08%** | 64,493 |
| 16w_PI | `2026-03-15T09_51_36_bottom_right` | Arena3 | 111,212 | PASS | 78 | 51,979 | **46.74%** | 59,233 |
| Pre_injury | `2026-04-28T13_21_05_bottom_left` | Arena2 | 248,172 | PASS | 296 | 171,761 | **69.21%** | 76,411 |
| Pre_injury | `2026-04-28T14_58_25_top_left` | Arena1 | 204,684 | PASS | 243 | 154,568 | **75.52%** | 50,116 |
| Pre_injury | `2026-04-28T16_38_02_bottom_right` | Arena3 | 248,685 | PASS | 212 | 70,128 | **28.2%** | 178,557 |

### Condition-Level Aggregates

| Condition | Sessions | Total Raw Frames | Retained Dual-View Frames | Retained Coverage (%) | Excluded Frames |
|:---|---:|---:|---:|---:|---:|
| **16w_PI** | 2.0 | 184,461.0 | 75,271.0 | **40.8%** | 109,190.0 |
| **3_week_PI** | 3.0 | 623,501.0 | 425,623.0 | **68.3%** | 197,878.0 |
| **8_weeks_PI** | 2.0 | 411,408.0 | 296,288.0 | **72.0%** | 115,120.0 |
| **Pre_injury** | 3.0 | 701,541.0 | 396,457.0 | **56.5%** | 305,084.0 |
