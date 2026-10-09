# Bonsai Multi-Stream Timestamp Wiring & Doric FP3002 Stall Analysis

## 1. Why `FP3002_System_Timestamp` Stops Advancing at Frame 94,167

In session `2025-12-03T14_06_45_bottom_right` (and simultaneously across CenterTopCam, BotCam1, BotCam2, and BotCam3), the column `FP3002_System_Timestamp` stops advancing at **Frame 94,167** and remains frozen at **`19553.897792`** for all subsequent 100,961 frames.

### Root Cause: Bonsai Reactive Architecture (`WithLatestFrom`)
1. **Doric Photometry Window:** The FP3002 photometry acquisition session was programmed or manually stopped after **26.15 minutes** ($94,167 / 60.0 = 1,569.45\text{ s}$).
2. **Video Recording Continued:** The Basler video cameras continued recording behavior for the full 54.2-minute session (195,128 frames).
3. **Reactive Stream Sampling:** In Bonsai-Rx, incoming video frames are tagged with photometry timestamps via a reactive operator like `videoStream.WithLatestFrom(photometryStream, (vid, phot) => ...)`.
4. **Behavior on Stream Termination:** When the Doric hardware driver ceased emitting new photometry packets at $t = 19553.897792\text{ s}$, the `WithLatestFrom` operator held the **last cached sample** in memory and assigned it to every incoming video frame until camera acquisition stopped.
5. **Verification Across Cameras:**
   - `CenterTopCam`: Stops at frame 94,167 ($t = 19553.897792\text{ s}$)
   - `BotCam3`: Stops at frame 94,167 ($t = 19553.897792\text{ s}$)
   Both files freeze at the exact same frame and microsecond value, confirming stream termination rather than USB packet loss.

---

## 2. Reliable Ground Truth: `Cam_Basler_Timestamp`

While `FP3002_System_Timestamp` is frozen after frame 94,167, the **`Cam_Basler_Timestamp`** column (hardware crystal clock on the Basler camera sensor, reported in nanoseconds) continues running with nanosecond precision across all 195,128 frames:
- Nominal period: $16,666,667\text{ ns}$ ($60.00\text{ Hz}$).
- This column provides the definitive, un-corrupted record of inter-frame intervals throughout the entire recording.
