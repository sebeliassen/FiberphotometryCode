# VFI Synchronization Debugging Report

## The Problem
We observed a persistent issue where kinematic verification snapshots (specifically 2x2 grid movies) were rendering tracking bounding boxes out-of-sync with the underlying video. Tracking was often drawn over empty frames or systematically misaligned segments of the videos.

The synchronization broke non-uniformly across sessions. For instance, Session 0 was specifically problematic and appeared to need a hardcoded `/ 2` scaling fix applied to its embedded `frame_index`. However, applying this global fix locally resolved Session 0 but broke Sessions 5, 10, and 15, indicating a non-uniform mapping logic was being injected into the dataset prior to generation.

## The Investigation
We constructed a standalone diagnostic script (`test_sync.py`) to test 4 different mathematical index mapping methods across Sessions 0, 5, 10, and 15 natively exactly at the 25% tracking progress mark (to guarantee active mouse visibility).

The visual test revealed:
- In the existing buggy data state, Session 0 required Method 1 (Direct VFI Lookup mapping from phot_idx * 2) to display correctly.
- Conversely, Sessions 5, 10, and 15 required Method 3 (1:1 mapping mapping directly from the embedded `frame_index`).

This split behavior strongly implied that the data preprocessing pipeline was inconsistently mutating or scaling the timeline for certain sessions before the `frame_index` column was baked into the `jonathan_amph_sessions_double_w_sylls.pkl` file.

## The Root Cause
A codebase search evaluated files like `legacy_support.py` and `attach_kpms.py` that handled merging KPMS results and behavioral data into the unified session objects. The culprit was a naive array-length heuristic:

```python
# Naive Heuristic (Removed)
if vfi_max < (expected_video_length / scale_factor * 0.75):
    # Falsely detected as legacy 40Hz downsampled VFI
    vfi = vfi * scale_factor
```

Since our new preprocessing pipeline trims the tracking timeline aggressively to only include "active" frames (removing rest states), the resulting Keypoint MoSeq `.npz` VFI arrays are physically subsetted. This naturally causes their true array lengths and maximum indices to drop to roughly `~50%` of the raw video frame length. 

Consequently, the attachment script heuristically compared these naturally shortened, Ground-Truth `VFI` arrays to the expected (full) video lengths, falsely identified them as "legacy 40Hz downsampled" runs, and erroneously multiplied their indices by `x2`.

## The Solution
We explicitly disabled and detached the array-length heuristic from both `legacy_support.py` and `attach_kpms.py`. The updated script now strictly trusts the derived Ground Truth mapped VFIs identically as they are delivered from the preprocessing output without ad-hoc scaling logic.

After rebuilding the main `jonathan_amph_sessions_double_w_sylls.pkl` file identically for all recordings, a re-run of the 4-method diagnostic across the entire battery verified that standard 1:1 mapping (using `v_idx = int(row['frame_index'])`) now perfectly orchestrates true video frame synchrony universally across every session, without any custom ID routing.
