# Seed 404 Multi-View Gallery: Comprehensive Tile-by-Tile Audit Report

**Model Identity:** `tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073`  
**Checkpoint Iteration:** 250 (EML Tournament Rank 1 Champion, EML = $225,001,430.22$)  
**Results File:** `results.h5` (SHA-256 prefix: `4d9b1b45df15d471`)  
**Audit Timestamp:** 2026-10-05 13:55:36  
**Total Video Tiles Audited:** 2400 (100 Syllables × 2 Sampling Modes × 12 Instances)  

---

## 1. Executive Summary & Verification Matrix

| Verification Metric | Target Standard | Audited Result | Status |
| :--- | :---: | :---: | :---: |
| **Syllable Label Integrity** | 100% of bouts match advertised syllable | **2400 / 2400 (100.00%)** | ✅ PERFECT |
| **Onset Marker Alignment** | Movie frame 10 strictly matches `vfi[model_onset]` | **2400 / 2400 (100.00%)** | ✅ PERFECT |
| **Boundary Segment Invariance** | Segment-end padding maintains 0-frame offset | **23 boundary bouts tested: 0 offsets** | ✅ RESOLVED |
| **BotCam Crystal Sync** | Continuous sub-ms crystal clock correspondence | **100% mapped via `top_to_bot_frame_continuous`** | ✅ VERIFIED |
| **Clean Video Stream Coverage** | Tiles with zero dropped/unmapped frames | **2400 / 2400 (100.0%)** | ✅ VERIFIED |

---

## 2. Key Findings

1. **Zero Label Contamination:**
   Every single one of the 2400 video tiles across all 100 syllables was verified against `results.h5`. In 100% of cases, `results.h5[sid]["syllable"][start:end] == syllable_id`. There are **zero stale bouts** or cross-syllable contaminations in the gallery.

2. **Onset Marker Mathematical Invariance:**
   Across all 2400 tiles, including all 23 segment boundary cases, `movie_onset_top_frame` exactly equals `onset_top_raw_frame`. The red dwell indicator and white tracking dot turn on at **exactly frame 10 (onset)** without any lag or lead.

3. **BotCam Stream Completeness:**
   Out of 2400 tiles, 2400 (100.0%) possess 100% continuous BotCam footage with zero dropped frames. The remaining 0 tiles with sporadic dropped frames now render explicit `[DROPPED]` warnings and suppress false DLC tracking dots.

---

## 3. Sample Audit Table (First 15 Tiles)

| Syllable | Mode | Tile | Session ID | Model Interval | Duration | Label Match | Marker Aligned | BotCam Onset Valid | Dropped Bot Frames |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 00 | density | 0 | `2025-12-03T09_51_05_bott...` | [7929, 7982] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 1 | `2025-12-03T09_51_05_bott...` | [25686, 25739] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 2 | `2026-05-16T14_47_09_bott...` | [16008, 16061] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 3 | `2026-05-16T14_47_09_bott...` | [25296, 25349] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 4 | `2026-03-15T11_48_09_bott...` | [26130, 26183] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 5 | `2026-05-17T15_13_17_top_...` | [477, 530] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 6 | `2026-05-17T10_05_10_bott...` | [24072, 24125] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 7 | `2025-12-03T16_38_48_top_...` | [343, 396] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 8 | `2026-04-28T14_58_25_bott...` | [3056, 3109] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 9 | `2026-04-28T14_58_25_bott...` | [5103, 5156] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 10 | `2026-04-28T14_58_25_top_...` | [3402, 3455] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | density | 11 | `2026-03-05T12_46_18_bott...` | [28099, 28152] | 53 frames | ✅ | ✅ | ✅ | 0 |
| 00 | random | 0 | `2025-12-03T09_51_05_bott...` | [201, 231] | 30 frames | ✅ | ✅ | ✅ | 0 |
| 00 | random | 1 | `2025-12-03T15_26_16_bott...` | [1861, 2004] | 143 frames | ✅ | ✅ | ✅ | 0 |
| 00 | random | 2 | `2025-12-04T10_27_49_top_...` | [9792, 9837] | 45 frames | ✅ | ✅ | ✅ | 0 |

---
*Full tile-by-tile records for all 2400 tiles are available in [`seed404_grid_movies_tile_by_tile_audit.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/seed404_grid_movies_tile_by_tile_audit.csv).*
