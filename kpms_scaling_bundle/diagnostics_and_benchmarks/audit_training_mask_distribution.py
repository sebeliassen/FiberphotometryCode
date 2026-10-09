import os

import h5py
import numpy as np

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
CHECKPOINT_PATH = os.path.join(KPMS_DIR, "amph_project", "amph_prod_k150k_run_01", "checkpoint.h5")

if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(KPMS_DIR, "amph_project", "amph_benchmark_test_timing", "checkpoint.h5")

def audit_dataset_mask(h5_path):
    print("=" * 80, flush=True)
    print("  AMPH DATASET MASK & MISSINGNESS DISTRIBUTION AUDIT", flush=True)
    print("=" * 80, flush=True)
    print(f"Loading mask from: {h5_path}", flush=True)

    with h5py.File(h5_path, "r") as f:
        mask = np.array(f["data"]["mask"]).astype(bool)  # shape: (N, T); stored as float64

    N, T = mask.shape
    total_frames = N * T
    valid_frames = int(np.sum(mask))
    masked_frames = total_frames - valid_frames

    print("\n[1] Overall Sequence Dimensions:", flush=True)
    print(f"  * Total Sequences (N):          {N}", flush=True)
    print(f"  * Padded Max Time (T):           {T} frames", flush=True)
    print(f"  * Total Frame Slots (N x T):     {total_frames:,}", flush=True)
    print(f"  * Valid Active Frames:           {valid_frames:,} ({100.0 * valid_frames / total_frames:.2f}%)", flush=True)
    print(f"  * Total Masked Slots:            {masked_frames:,} ({100.0 * masked_frames / total_frames:.2f}%)", flush=True)

    # Classify masked frames: Leading padding, Trailing padding, or Internal Gaps
    leading_padding_count = 0
    trailing_padding_count = 0
    internal_gap_count = 0
    internal_gap_lengths = []
    seqs_with_internal_gaps = 0

    for i in range(N):
        seq = mask[i]
        valid_indices = np.where(seq)[0]
        if len(valid_indices) == 0:
            trailing_padding_count += T
            continue

        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        leading_padding_count += first_valid
        trailing_padding_count += (T - 1 - last_valid)

        # Examine within active window [first_valid : last_valid + 1]
        active_slice = seq[first_valid : last_valid + 1]
        num_invalid_inside = int(np.sum(~active_slice))

        if num_invalid_inside > 0:
            seqs_with_internal_gaps += 1
            internal_gap_count += num_invalid_inside

            cur_len = 0
            for val in active_slice:
                if not val:
                    cur_len += 1
                else:
                    if cur_len > 0:
                        internal_gap_lengths.append(cur_len)
                        cur_len = 0
            if cur_len > 0:
                internal_gap_lengths.append(cur_len)

    print("\n[2] Breakdown of Masked Frames:", flush=True)
    print(f"  * Leading Padding (before start):    {leading_padding_count:,} ({100.0 * leading_padding_count / max(1, masked_frames):.2f}% of masked)", flush=True)
    print(f"  * Trailing Padding (after end):     {trailing_padding_count:,} ({100.0 * trailing_padding_count / max(1, masked_frames):.2f}% of masked)", flush=True)
    print(f"  * Internal Behavioral Gaps:         {internal_gap_count:,} ({100.0 * internal_gap_count / max(1, masked_frames):.2f}% of masked)", flush=True)
    print(f"  * Sequences with Internal Gaps:     {seqs_with_internal_gaps} / {N} ({100.0 * seqs_with_internal_gaps / N:.2f}%)", flush=True)

    if internal_gap_lengths:
        print("\n[3] Internal Gap Duration Statistics (frames at 40 Hz):", flush=True)
        print(f"  * Total Gap Events:                 {len(internal_gap_lengths)}", flush=True)
        print(f"  * Mean Gap Length:                  {np.mean(internal_gap_lengths):.2f} frames ({np.mean(internal_gap_lengths)/40.0 * 1000:.1f} ms)", flush=True)
        print(f"  * Median Gap Length:                {np.median(internal_gap_lengths):.1f} frames ({np.median(internal_gap_lengths)/40.0 * 1000:.1f} ms)", flush=True)
        print(f"  * Max Gap Length:                   {np.max(internal_gap_lengths)} frames ({np.max(internal_gap_lengths)/40.0:.2f} s)", flush=True)
    else:
        print("\n[3] Internal Gap Duration Statistics:", flush=True)
        print("  * ZERO internal gaps found! 100% of masked frames are array boundary padding.", flush=True)

    print("\n" + "=" * 80, flush=True)

if __name__ == "__main__":
    audit_dataset_mask(CHECKPOINT_PATH)
