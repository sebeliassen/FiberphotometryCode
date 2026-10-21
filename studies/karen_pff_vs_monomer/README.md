# Karen Project Data & Video Directory Guide

The historical session-discovery policy is owned by
[`session_policy/config.py`](session_policy/config.py). It retains
`VALID_FRAME_RANGE = (0, 999999)` and an independent default `JonathanConfig`
instance, recovered from the archived original source. It uses the standard
filename and chamber rules; the monoamine/juvenile relaxed regex and frame range
are separate cohort policies. `analysis/tracking.py` imports this Karen policy
explicitly and retains its timestamp-level grouping and empty-animal filtering.
Filename-only synthetic tests cover those discovery rules without reading video
or H5 contents. The unified builder's global configuration changes, direct old
study symlink identity and real artifact reconstruction remain separate gates.
Provenance and exact check limits are recorded in
[`project_tracking_policies_20261021.json`](../../registry/checkpoints/project_tracking_policies_20261021.json).

## Overview & Directory Topology
This directory (`karen_project/`) contains the workspace files for Karen's longitudinal Parkinson's disease / L-dopa mouse cohort studies (covering 5 experimental waves: `3_week_PI`, `8_weeks_PI`, `16w_PI`, `24_weeks_PI`, and `L_dopa`).

```
karen_project/
├── video_files/          # TopCam dorsal videos (split quadrants: top_left, bottom_left, etc.)
│                         # Used directly by Keypoint-MoSeq and DLC dorsal models
├── h5_files/             # DeepLabCut dorsal keypoint tracking outputs (.h5)
├── botcam_video_files/   # Isolated symlinks to raw upward-facing bottom camera videos (.avi)
└── README.md             # This reference guide
```

> [!IMPORTANT]
> **Keep `botcam_video_files/` isolated from `video_files/`!**
> Keypoint-MoSeq (`karen_project_2`) expects only dorsal TopCam videos in `video_files/`. Mixing BotCam videos into `video_files/` will cause Keypoint-MoSeq to index bottom cameras as dorsal animals.

---

## 1. Bottom Camera (BotCam) Storage on `/mnt/hdd`

Raw bottom camera videos (`*BotCamVid*.avi`) are stored on the local high-capacity hard disk at `/mnt/hdd/karen_project/`:

| Wave Cohort | HDD Directory | Total BotCam Videos | Notes |
| :--- | :--- | :--- | :--- |
| **`16w_PI`** | `/mnt/hdd/karen_project/T*` | 33 | Stored directly in root trial directories (`T1_...` through `T11_...`) |
| **`3_week_PI`** | `/mnt/hdd/karen_project/3_week_PI/` | 40 | Rsynced from `/mnt/neuropharm` |
| **`8_weeks_PI`** | `/mnt/hdd/karen_project/8_weeks_PI/` | 24 | Rsynced from `/mnt/neuropharm` |
| **`24_weeks_PI`** | `/mnt/hdd/karen_project/24_weeks_PI/` | 33 | Rsynced from `/mnt/neuropharm` |
| **`L_dopa`** | `/mnt/hdd/karen_project/L_dopa/` | 28 | Rsynced from `/mnt/neuropharm` |
| **Total** | | **158 videos** | ~400+ GB total storage |

Video properties:
- Resolution: $972 \times 972$
- Frame Rate: **60.0 FPS** (exact 1:1 time-lock with TopCam, no downsampling mismatch)

---

## 2. Spatial Arena Orientation & Mapping

The 4 physical chambers in the bottom-camera recording rig map directly to the TopCam overhead quadrants via a standard counter-clockwise loop:

```
+-----------------------------------+-----------------------------------+
|              Arena 1              |              Arena 4              |
|        TopCam: `top_left`         |        TopCam: `top_right`        |
|      (Cross-corr r = +0.295)      |       (ALWAYS EMPTY CHAMBER)      |
+-----------------------------------+-----------------------------------+
|              Arena 2              |              Arena 3              |
|       TopCam: `bottom_left`       |       TopCam: `bottom_right`      |
|      (Cross-corr r = +0.279)      |      (Cross-corr r = +0.127)      |
+-----------------------------------+-----------------------------------+
```

### Empirical Validation
- **Permutation Test**: Evaluated all 24 spatial permutations across TopCam and BotCam motion deltas.
  - Rank 1 permutation (`Arena 1 -> top_left`, `Arena 2 -> bottom_left`, `Arena 3 -> bottom_right`): **Score = +0.700**.
  - Rank 2 runner-up: Score = +0.328 (Rank 1 is >2x higher than any other configuration).
- **Arena 4 Status**: The fiber photometry hardware has 3 patch cords. There are never 4 mice recorded at once. Across all experimental waves, Arena 4 sits at flat optical sensor noise floor (mean motion delta $0.20$, variance $0.00$).

---

## 3. Workspace Symlink Directory (`botcam_video_files/`)

All 158 videos are symlinked in `karen_project/botcam_video_files/`.

Each symlink is formatted as:
```
{wave}_{trial_folder}_{status}_{original_arena_filename}.avi
```

Examples:
- `16w_PI_T1_6.2.4_05.03.26_active_Arena1_BotCamVid_2026-03-05T11_01_01.avi`
- `3_week_PI_T10_396_04122025_empty_chamber_Arena4_BotCamVid_2025-12-04T12_56_44.avi`

The `status` tag (`active` vs `empty_chamber`) is automatically cross-referenced against TopCam filenames in `karen_project/video_files/`.

---

## 4. Scripts & Automation Tools

The following utilities are in `scripts/`:

1. **`scripts/rsync_karen_botcam.py`**:
   - Transfers missing BotCam files from the network share `/mnt/neuropharm` to `/mnt/hdd/karen_project/`.
2. **`scripts/link_karen_botcam.py`**:
   - High-speed symlinker creating formatted links in `botcam_video_files/`.
   - Run dry-run: `python3 scripts/link_karen_botcam.py`
   - Run execution: `python3 scripts/link_karen_botcam.py --run`
   - Only link active chambers: `python3 scripts/link_karen_botcam.py --run --filter-empty`
3. **`scripts/verify_botcam_orientation.py`**:
   - Verification tool calculating concurrent frame motion deltas, cross-correlations, and 24-permutation tests across TopCam and BotCam videos.
