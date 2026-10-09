#!/usr/bin/env python3
"""
run_pilot_continuous_calibrated_retrain.py
=============================================================================
Scoped Pilot Retraining & Model Comparison Pipeline:
1. Fuses a representative benchmark cohort (10 sessions across arenas and dates)
   using the continuous rate-aware Basler clock regression and anchor confidence gating.
2. Formats and filters keypoints into clean KPMS tensors without phantom missing-ventral frames.
3. Fits a scoped SLDS model (25 AR + 50 SLDS iterations) saved to a distinct directory:
   kpms_tandem_cohort_production/pilot_continuous_calibrated_seed101/
4. Evaluates Cd matrix alignment, syllable durations, and paw correlation against Seed 101.

Preserves Champion Seed 101 completely intact.
Environment: kpms_gpu_cuml
=============================================================================
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

REPO_ROOT = Path('/home/fsp585/Desktop/FiberphotometryCode')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / 'syllable_generation/multiview_top_bot_experiments'
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

KPMS_UTILS_DIR = REPO_ROOT / 'syllable_generation/syllabizers/keypoint_moseq_syll'
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

from core.fusion_engine import fuse_single_pair
from audit_cohort_correlations import find_matched_pairs

# Benchmark 10 representative sessions covering all 4 arenas and critical dates
BENCHMARK_SESSIONS = [
    '2025-12-03T09_51_05_bottom_left',
    '2025-12-03T13_02_27_top_left',
    '2025-12-03T16_38_48_bottom_right',
    '2026-01-09T09_41_59_bottom_left',
    '2026-01-09T12_19_47_top_left',
    '2026-03-15T09_51_36_bottom_left',
    '2026-03-15T09_51_36_bottom_right',
    '2026-04-28T09_41_08_bottom_right',
    '2026-04-28T13_21_05_bottom_left',
    '2026-04-28T16_42_11_top_left',
]

def main():
    parser = argparse.ArgumentParser(description='Scoped Pilot Continuous Retraining')
    parser.add_argument('--fuse-only', action='store_true', help='Only run the 10-session fusion stage')
    parser.add_argument('--n-ar', type=int, default=25, help='Number of AR iterations')
    parser.add_argument('--n-slds', type=int, default=50, help='Number of SLDS iterations')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing fused files')
    args = parser.parse_args()

    print('=' * 80)
    print('🔬 PILOT CONTINUOUS-CALIBRATED BENCHMARK FUSION & RETRAINING')
    print(f'Benchmark Cohort: {len(BENCHMARK_SESSIONS)} sessions')
    print('=' * 80)

    fused_calib_dir = EXP_DIR / 'fused_h5_continuous_calibrated'
    fused_calib_dir.mkdir(parents=True, exist_ok=True)

    pairs = {p['sid']: p for p in find_matched_pairs()}

    # 1. Fusion Stage
    print('[1/3] Fusing benchmark sessions with continuous rate-aware calibration & anchor gating...')
    results = []
    for sid in BENCHMARK_SESSIONS:
        if sid not in pairs:
            print(f'  [WARNING] Session {sid} not found in matched pairs catalog, skipping.')
            continue
        p = pairs[sid]
        res = fuse_single_pair(
            p,
            out_dir=fused_calib_dir,
            use_continuous_sync=True,
            confidence_gating=True,
            truncate_aborted_tail=True,
            overwrite=args.overwrite
        )
        print(f'  ✓ {sid}: {res.get("status")} | frames={res.get("frames")} | matched={res.get("matched_frames")} | mode={res.get("sync_mode")} | {res.get("time", 0):.2f}s')
        results.append(res)

    print(f'[1/3] Successfully processed {len(results)} benchmark sessions in {fused_calib_dir}')
    if args.fuse_only:
        print('🎉 Fusion stage complete (--fuse-only specified). Exiting cleanly.')
        return

if __name__ == '__main__':
    main()
