# Karen Project: Bottom Camera (BotCam) Reference & Orientation Guide

## 1. Executive Summary
This document summarizes the storage, spatial orientation, and linking architecture for Karen's upward-facing bottom camera recordings (`*BotCamVid*.avi`) across 5 longitudinal cohorts (`3_week_PI`, `8_weeks_PI`, `16w_PI`, `24_weeks_PI`, `L_dopa`).

---

## 2. File Storage Locations

- **Raw Data on HDD (`/mnt/hdd/karen_project/`)**:
  - `16w_PI`: 33 videos located in trial folders `/mnt/hdd/karen_project/T*`
  - `3_week_PI`: 40 videos in `/mnt/hdd/karen_project/3_week_PI/`
  - `8_weeks_PI`: 24 videos in `/mnt/hdd/karen_project/8_weeks_PI/`
  - `24_weeks_PI`: 33 videos in `/mnt/hdd/karen_project/24_weeks_PI/`
  - `L_dopa`: 28 videos in `/mnt/hdd/karen_project/L_dopa/`
  - **Total**: **158 BotCam videos** (~400+ GB)
- **Isolated Workspace Symlinks**:
  - Location: [`karen_project/botcam_video_files/`](file:///home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files/)
  - Tagged naming: `{wave}_{trial}_{status}_{original_name}.avi`
  - Example: `16w_PI_T1_6.2.4_05.03.26_active_Arena1_BotCamVid_2026-03-05T11_01_01.avi`

> [!IMPORTANT]
> BotCam symlinks are kept strictly isolated in `botcam_video_files/` and NOT placed into `karen_project/video_files/`. Keypoint-MoSeq uses `video_files/` for dorsal TopCam models; mixing angles would disrupt video key index lookups.

---

## 3. Spatial Orientation & Chamber Mapping

Through concurrent frame delta motion variance analysis and full 24-permutation testing against overhead TopCam videos, the hardware bottom-camera arena numbers were confirmed to map directly to the TopCam quadrants without rotation or mirroring:

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

### Key Validation Findings
1. **Frame Rate Synchronization**: Both TopCam and BotCam run at **60.0 FPS** (972×972 pixels). There is zero frame rate mismatch or downsampling divergence.
2. **Permutation Test**: The identity mapping scored **+0.700** across active mice, compared to +0.328 for the second-best permutation.
3. **Arena 4 Status**: The fiber photometry recording setup supports 3 simultaneous optical patch cords. Arena 4 is never occupied by a mouse and records flat sensor noise (delta $0.20$, variance $0.00$).

---

## 4. Pipeline Scripts

All utility scripts are located in [`scripts/`](file:///home/fsp585/Desktop/FiberphotometryCode/scripts/):

- [`scripts/rsync_karen_botcam.py`](file:///home/fsp585/Desktop/FiberphotometryCode/scripts/rsync_karen_botcam.py): Transfers missing BotCam files from `/mnt/neuropharm` to `/mnt/hdd/karen_project/`.
- [`scripts/link_karen_botcam.py`](file:///home/fsp585/Desktop/FiberphotometryCode/scripts/link_karen_botcam.py): Scans `/mnt/hdd` and creates annotated symlinks in `karen_project/botcam_video_files/`. Supports `--filter-empty` to skip empty chambers.
- [`scripts/verify_botcam_orientation.py`](file:///home/fsp585/Desktop/FiberphotometryCode/scripts/verify_botcam_orientation.py): Validates temporal correlation and spatial orientation via frame differencing.
