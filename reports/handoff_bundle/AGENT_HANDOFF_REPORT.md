# Agent Handoff Report: KPMS Calibrated Cohort Training
**Timestamp:** 2026-10-01 19:20:00 CEST  
**Host:** woody (NVIDIA RTX 4090 / 24GB VRAM, 128GB Host RAM)  
**Target:** 118-Session Multi-View Calibrated Cohort (17.37M Frames, 3473 Sequences)  

---

## 1. Executive Summary & Current Status
- **Phase 1 AR Sampling:** 100% functional, benchmarked at **~14.5–16.5s/it**.
- **Iter 2 Anchor Checkpoint Save:** **PASSED & COMMITTED AT ITER 2** (16.63 GB on disk, took 6.66s, RSS delta < 1 GB).
- **Phase 2 SLDS Sampling:** **ACTIVELY RUNNING NOW!** (Iter 2 -> 4). Iteration 1 of 2 finished in 119s, iteration 2 currently in progress.
- **Root Cause of Earlier Crash:** Identified and fixed. The model snapshot is ~14.8 GB in memory (`s`=6.4GB, `noise_prior`=6.4GB, `x`=2.8GB). Stock KPMS was gathering the entire model plus 36GB data tree simultaneously into host RAM, triggering a Linux kernel `SIGKILL` (OOM).
- **The Solution Applied:**
  1. `_savetree_bounded`: writes leaf-by-leaf directly to HDF5. GPU arrays transferred individually, NumPy host arrays written with 0-copy. Host RSS rose by only **0.98 GB** (from 47.26 GB to 48.24 GB)!
  2. Observation data (`preprocessed_data.pkl`) is kept as persistent source of truth and not duplicated into `checkpoint.h5`.
  3. Atomic commits (`checkpoint.h5.tmp` -> `os.replace` -> `checkpoint.h5`) ensuring zero corruption on interruption and bounded disk size.
  4. Removed redundant `copy.deepcopy(cached_model)` which was leaking 20-30 GB of RAM.
  5. Updated `patch_checkpoint.py` so standard/compacted checkpoint loaders transparently validate and load `data` from `preprocessed_data.pkl` if omitted from HDF5.
  6. Final export reuses in-memory `model` and `metadata` directly.

---

## 2. Active Run Details
- **Command:**
  ```bash
  python -u syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py \
    --seed 101 --num-ar-iters 2 --num-slds-iters 2 --save-every-n-iters 2 \
    --model-name "test_rolling_2ar_2slds_seed101"
  ```
- **Log Location:**
  `syllable_generation/multiview_top_bot_experiments/02_training_runners/logs/test_rolling_2ar_2slds_latest.log`
- **Current Execution State:**
  SLDS 1/3 (iter 3) completed, currently resampling continuous states for iter 4.

---

## 3. Next Steps for Incoming Agent
1. **Check completion of Part 1 (2 AR + 2 SLDS):**
   ```bash
   tail -n 30 syllable_generation/multiview_top_bot_experiments/02_training_runners/logs/test_rolling_2ar_2slds_latest.log
   ```
2. **Execute Restart Verification (Fresh Process Resumption):**
   Launch with `--num-ar-iters 2 --num-slds-iters 4` to verify the runner resumes from iter 4:
   ```bash
   python -u syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py \
     --seed 101 --num-ar-iters 2 --num-slds-iters 4 --save-every-n-iters 2 \
     --model-name "test_rolling_2ar_2slds_seed101"
   ```
   Verify it outputs `[RESUME] Found existing checkpoint ... Resumed successfully from iteration 4`.
3. **Launch Production Run (50 AR + 200 SLDS):**
   Once verified, launch the production run:
   ```bash
   nohup /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -u \
     syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py \
     --seed 101 --num-ar-iters 50 --num-slds-iters 200 --save-every-n-iters 25 \
     --model-name "tandem_cohort_calibrated_50ar_200slds_seed101" \
     > syllable_generation/multiview_top_bot_experiments/02_training_runners/logs/cohort_calibrated_retrain_seed101_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
   ```
