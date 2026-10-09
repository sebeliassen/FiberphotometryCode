# Syllable ID & Bout Provenance: Seed 101 vs. Seed 303

## 1. Resolution of the Syllable 1 vs. Syllable 0 Discrepancy

- **Observation:** In the previous audit package, the bouts at TopCam raw video frame `138878` (Session `2025-12-03T14_06_45_bottom_right`) and `85395` (Session `2026-05-16T10_41_38_top_left`) were identified as **Syllable 01**.
- **Reviewer Finding:** In the exported Seed 101 `results.h5`, both of these bouts are **Syllable 00**, not Syllable 01.
- **Cause:**
  - Those two bouts were extracted from [`subgrid_video_correspondences.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/tandem_multiview_diagnostic_package/docs/subgrid_video_correspondences.md), which documented the **Champion Seed 303** grid movies (`syllable_01_tandem_grid.mp4` in `visualizations_seed303/grid_movies/`).
  - In Seed 303:
    - `2025-12-03T14_06_45_bottom_right` at vfi `138878` (model frame `115896`) is **Syllable 1**.
    - `2026-05-16T10_41_38_top_left` at vfi `85395` (model frame `78367`) is **Syllable 1**.
  - In Seed 101:
    - The AR-HMM Gibbs sampler assigned a different discrete state index: that exact behavioral state is labeled **Syllable 0** in Seed 101!
    - AR-SLDS syllable labels are permutation-invariant across random seeds.

---

## 2. Renderer Version & Stationary Dots

- The grid movie `syllable_01_tandem_grid_seed303.mp4` was rendered on **September 30, 2026** by `render_seed303_grid_movies.py`.
- That legacy script hardcoded `cv2.circle(tile_top, (80, 80), ...)` and had no circle call for `tile_bot`.
- The updated dynamic tracking renderer ([`render_all_seed101_grid_movies.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/render_all_seed101_grid_movies.py)) was executed on **October 1 at 13:24:45 CEST** via `run_all_seed101_visualizations_nohup.sh`:
  ```bash
  /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -u render_all_seed101_grid_movies.py --mode density --overwrite
  ```
- Output: `visualizations_seed101/grid_movies_density/syllable_01_tandem_grid_density.mp4`.

---

## 3. Actual 12 Bouts in Supplied Seed 101 Density Movie (`syllable_01_tandem_grid_density.mp4`)

| Tile Position | Session ID | Start (Model) | End (Model) | Duration | TopCam Onset Vid Frame |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Row 1, Col 1** | `2025-12-03T10_58_48_bottom_right` | 95967 | 96007 | 40 | 142630 |
| **Row 1, Col 2** | `2025-12-03T10_58_48_bottom_right` | 165629 | 165669 | 40 | 213272 |
| **Row 1, Col 3** | `2025-12-03T10_58_48_bottom_right` | 162935 | 162975 | 40 | 210578 |
| **Row 1, Col 4** | `2025-12-03T10_58_48_top_left` | 14192 | 14232 | 40 | 26776 |
| **Row 2, Col 1** | `2025-12-03T10_58_48_top_left` | 48620 | 48660 | 40 | 67257 |
| **Row 2, Col 2** | `2025-12-03T13_02_27_bottom_left` | 68303 | 68343 | 40 | 89381 |
| **Row 2, Col 3** | `2025-12-03T13_02_27_bottom_left` | 82781 | 82821 | 40 | 103913 |
| **Row 2, Col 4** | `2025-12-03T13_02_27_bottom_left` | 99841 | 99881 | 40 | 121286 |
| **Row 3, Col 1** | `2025-12-03T13_02_27_bottom_left` | 130081 | 130121 | 40 | 151671 |
| **Row 3, Col 2** | `2025-12-03T13_02_27_top_left` | 39070 | 39110 | 40 | 50350 |
| **Row 3, Col 3** | `2025-12-03T13_02_27_top_left` | 28949 | 28989 | 40 | 40089 |
| **Row 3, Col 4** | `2025-12-03T14_06_45_bottom_left` | 34773 | 34813 | 40 | 47971 |
