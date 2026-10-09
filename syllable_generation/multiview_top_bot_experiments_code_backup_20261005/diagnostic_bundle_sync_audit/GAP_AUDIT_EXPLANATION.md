# Six-Frame Gap Audit: BotCam Frame 186366 & Impact on Model Training

## 1. Frame Gap Discovery

In session `2025-12-03T14_06_45_bottom_right` (Arena 3):
- Between BotCam row `186366` and row `186367`, the hardware Basler clock jumps:
  - Row 186366: `21,076,225,567,380 ns`
  - Row 186367: `21,076,342,231,248 ns`
  - Difference: $\Delta t = 116.663,868\text{ ms} = 7 \times 16.666\text{ ms}$
- This corresponds to a burst drop of **exactly 6 camera frames** (100.0 ms) by the USB bus / camera driver.

---

## 2. Did this Misalignment Reach Model Training?

**YES, for the tail 4.49% of this session (8,767 frames).**

### Mechanism:
1. `fusion_engine.py` computed an initial startup offset of `0` frames for this session based on frame 0.
2. In the fusion loop, it paired TopCam frame $t$ with BotCam frame $t - 0 = t$.
3. Prior to frame 186366, TopCam and BotCam were in sub-frame hardware lock ($\Delta t < 2.33\text{ ms}$).
4. At frame 186366, BotCam dropped 6 frames.
5. From frame 186367 through frame 195134, BotCam row $t$ actually corresponds to physical time of TopCam frame $t + 6$.
6. Because `video_frame_indexes.npz` retained high-confidence tracking from these frames (model frames `162654` to `167417`), these slightly shifted frames ($100\text{ ms}$ lag in ventral coordinates) were fed into the AR-SLDS model during training.

### Cohort Context:
- The 8,767 affected frames represent **4.49%** of session `2025-12-03T14_06_45_bottom_right`, and **0.036%** of the full 24-million frame cohort.
- Inspection of coordinates shows the animal was stationary/resting near the arena corner during this tail segment (`top_nose_x \approx 95.13\text{ px}`).
- Accompanying data file: [`gap_audit_frame_186366/fused_preprocessed_gap_export_frames_186355_186375.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/diagnostic_bundle_sync_audit/gap_audit_frame_186366/fused_preprocessed_gap_export_frames_186355_186375.csv) contains the exact per-frame Basler and Bonsai timestamps, fused coordinates, VFI flags, and syllable predictions across this transition.
