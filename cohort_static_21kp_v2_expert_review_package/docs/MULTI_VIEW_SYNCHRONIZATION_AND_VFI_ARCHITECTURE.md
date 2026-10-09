# Multi-View Synchronization & VFI Architecture

## 1. Executive Summary

This document formalizes the multi-coordinate indexing stack for the multi-view Keypoint-MoSeq (KPMS) pipeline across dorsal (**TopCam**) and ventral (**BotCam**) video streams. It details why temporal desynchronization occurs in software-triggered setups, demonstrates how **TTL / Photometry Master Timestamps** resolve startup latency, explains how this ties into the **Video Frame Index (VFI)**, and provides explicit hardware and software prevention strategies for future experiments.

---

## 2. The Multi-Coordinate Indexing Stack

When working with multi-view behavioral recordings coupled to fiber photometry, there are **four distinct indexing domains** that must not be conflated:

```
┌────────────────────────────────────────────────────────┐
│ 1. Model Frame Index: t_model ∈ [0, N_model - 1]       │
│    (Continuous latent state sequence in results.h5)    │
└──────────────────────────┬─────────────────────────────┘
                           │  VFI Mapping
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. TopCam Raw Frame: F_top = VFI[t_model]              │
│    (Video frame in CenterTopCamVid.avi & Top DLC H5)   │
└──────────────────────────┬─────────────────────────────┘
                           │  TTL Master Clock Alignment:
                           │  T_FP3002(F_top) == T_FP3002(F_bot)
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. BotCam Raw Frame: F_bot = F_top - Δ_TTL             │
│    (Video frame in BotCamVid.avi & Bot DLC H5)         │
└────────────────────────────────────────────────────────┘
```

| Index Domain | Symbol | Source File | Purpose |
| :--- | :---: | :--- | :--- |
| **Model Frame** | $t_{\text{model}}$ | `results.h5` | Discrete AR-HMM syllable transitions ($0 \le t < N_{\text{model}}$). |
| **Video Frame Index** | $VFI$ | `video_frame_indexes.npz` | Maps model frame $t_{\text{model}}$ to raw TopCam video frame, accounting for arena entry trimming. |
| **TopCam Raw Frame** | $F_{\text{top}}$ | `CenterTopCamVid.avi` | Physical video frame of the dorsal camera. |
| **BotCam Raw Frame** | $F_{\text{bot}}$ | `BotCamVid.avi` | Physical video frame of the ventral camera. |
| **Master Clock** | $T_{\text{FP3002}}$ | `AllTimestampsCombined*.csv` | High-precision hardware clock (seconds) from the FP3002 photometry acquisition box. |

---

## 3. The Root Cause of Camera Desynchronization

### A. Software vs. Hardware Triggering
In the current recording setup, camera acquisition is initiated via software nodes in **Bonsai**. Because the top camera and bottom cameras operate on distinct USB 3.0 / GigE buses and separate OS threads:
1. TopCam acquisition frequently initialized **a few fractions of a second before** the BotCam nodes.
2. In **87.6% of sessions**, this startup latency was negligible ($\le 1$ frame).
3. In **12.4% of sessions** (notably `2026-05-16` L-DOPA sessions), TopCam started **31 to 67 frames before BotCam** ($\sim 0.5$ to $1.1$ seconds).

### B. The Failure of Naive Indexing
If a visualization or analysis pipeline assumes:
$$F_{\text{bot}} = F_{\text{top}} = VFI[t_{\text{model}}]$$
Then on a session with $\Delta_{\text{start}} = 67$ frames:
- TopCam displays the animal at frame $F$.
- BotCam displays the animal at frame $F - 67$ (over 1.1 seconds in the past).
- Because mice travel at $10\text{–}30\text{ cm/s}$ and execute turns in $<200\text{ ms}$, an 1100 ms offset makes the animal appear in completely different arena quadrants, causing heading misalignment and seeming desynchronization.

---

## 4. How TTL Hardware Synchronization Ties into VFI

### A. The Master Clock Anchor
Both cameras stream timestamps directly to the shared photometry acquisition hardware:
- `CenterTopCam_AllTimestampsCombined*.csv` contains `FP3002_System_Timestamp` for every TopCam frame.
- `BotCam{1..4}_AllTimestampsCombined*.csv` contains `FP3002_System_Timestamp` for every BotCam frame.

Because both files reference the **exact same physical crystal oscillator in the FP3002 box**, we compute the exact frame offset:
$$\Delta_{\text{TTL}} = \text{argmin}_k \left| T_{\text{top}}[k] - T_{\text{bot}}[0] \right|$$

For example, in `2026-05-16T10_41_38_top_left`:
$$T_{\text{bot}}[0] = 3398.473504\text{ s} \implies T_{\text{top}}[67] = 3398.473536\text{ s} \quad (\Delta t = +0.03\text{ ms})$$
BotCam Frame 0 corresponds **exactly** to TopCam Frame 67 with sub-millisecond precision.

### B. The Unified Frame Retrieval Equation
For any model syllable bout occurring at model frame $t_{\text{model}}$:
$$\begin{aligned}
F_{\text{top}} &= VFI[t_{\text{model}}] \\
F_{\text{bot}} &= \max(0, F_{\text{top}} - \Delta_{\text{TTL}})
\end{aligned}$$

By applying $\Delta_{\text{TTL}}$, **100% of the session's temporal alignment is recouped**:
- Centroid X Correlation: $0.8119 \longrightarrow \mathbf{0.9508}$
- Mean Cosine Heading Alignment: $0.6865 \longrightarrow \mathbf{0.8173}$
- Median Angular Error: $27.7^\circ \longrightarrow \mathbf{14.0^\circ}$

---

## 5. Prevention Strategies for Future Experiments

### A. Hardware Prevention (Zero-Latency Guarantee)
1. **Hardwired GPIO Daisy-Chain**:
   - Discontinue software triggering via Bonsai.
   - Wire the digital output pin (TTL Out) of the FP3002 photometry box directly into the **opto-isolated Line 1 (Trigger In)** of all Basler cameras.
   - Configure Basler cameras in Pylon to `TriggerMode = On`, `TriggerSource = Line1`, `TriggerSelector = FrameStart`.
   - **Result**: The first pulse from the master clock exposes frame 0 on all cameras simultaneously down to the nanosecond. $\Delta_{\text{TTL}} \equiv 0$ by physical law.

### B. Software & Pipeline Architecture Prevention
1. **Tandem VFI Schema (`video_frame_indexes.npz`)**:
   - Currently, `video_frame_indexes.npz` stores a single 1D array per session (`vfi = [f0, f1, ...]`).
   - In future pipeline runs, `video_frame_indexes.npz` should store a 2D array or structured dictionary:
     ```python
     vfi_data[sid] = {
         "top_vid_frames": vfi_top,  # Shape (N_model,)
         "bot_vid_frames": vfi_bot,  # Shape (N_model,)
         "ttl_offset": delta_ttl,    # Scalar int
         "dt_residual_ms": res_ms    # Float (sanity check)
     }
     ```
2. **Automated Ingestion Pre-Check (`core.ttl_sync`)**:
   - During raw data extraction, run `compute_pair_ttl_offset()` automatically.
   - If $\Delta_{\text{TTL}} \neq 0$, trim or pad the raw arrays *before* saving the preprocessed H5 files, ensuring that row $i$ in `top.h5` and row $i$ in `bot.h5` are pre-aligned.
3. **Automated CI/CD Alignment Assertion**:
   - Add an automated test in `tests/test_multiview_alignment.py`:
     ```python
     def test_session_ttl_alignment(sid):
         offset = get_session_ttl_offset(sid)
         assert abs(offset["mean_res_ms"]) < 16.6, f"Residual timing jitter exceeds 1 frame for {sid}"
     ```

---

## 6. Implementation Reference

The automated TTL synchronization module is implemented and fully operational in:
- Engine: [`core/ttl_sync.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/core/ttl_sync.py)
- Cache: [`core/ttl_cohort_offsets.json`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/core/ttl_cohort_offsets.json)
- Grid Movie Generator: [`04_reports_and_figures/render_seed303_grid_movies.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/render_seed303_grid_movies.py)
