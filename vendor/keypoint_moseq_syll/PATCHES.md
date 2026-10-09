# Keypoint-MoSeq Fork Modifications & Patches

**Location**: `vendor/keypoint_moseq_syll/`  
**Upstream Project**: [Keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) (`keypoint_moseq`)  
**Status**: Vendored & Frozen  
**Canonical Production Successor**: [`src/fpcode/kpms/`](../../src/fpcode/kpms/)  

---

## 1. Overview of Modifications

This vendored directory represents the historically patched Keypoint-MoSeq fork utilized for multi-view and single-view behavioral decomposition across the *Karen* and *Amph* cohorts.

The key customizations applied over upstream `keypoint_moseq` include:

### 1.1. Checkpoint Compaction & Optimization (`utils/compact_checkpoint.py`, `utils/patch_checkpoint.py`)
- **Compaction Engine**: Prunes intermediate JAX/HDF5 optimizer states and redundant iteration snapshots, reducing model checkpoint size from ~180 MB to ~12 MB without loss of final parameters.
- **Bit-for-Bit Parity**: Preserves exact syllable emissions, AR parameters, and latent state trajectories.
- **Production Migration**: Formally consolidated into [`fpcode.kpms.checkpoints`](../../src/fpcode/kpms/checkpoints.py).

### 1.2. Batch Resampling & Runtime Monkeypatches (`utils/resample_batching.py`)
- **Memory Scaling**: Dynamic VRAM allocation and batch sizing based on GPU memory headroom.
- **Video Frame Interpolation (VFI) Masking**: Handles non-uniform temporal cadences and camera dropouts.
- **Production Migration**: Formally consolidated into [`fpcode.kpms.runtime`](../../src/fpcode/kpms/runtime.py).

### 1.3. Trimming & Velocity Filters (`utils/trimming.py`)
- **Outlier Filtering**: Speed- and acceleration-gated thresholding to suppress camera occlusions and tracking jitters.
- **Production Migration**: Formally consolidated into [`fpcode.kpms.data`](../../src/fpcode/kpms/data.py).

### 1.4. Invariant Orientation & Flipping (`utils/flipping.py`, `utils/optimal_flip_list.py`)
- **Bilateral Invariance**: Symmetrization of animal posture to ensure left/right turn parity in syllable modeling.

### 1.5. Production Orchestration & Diagnostics (`diagnostics/`, `kpms_pipeline.py`)
- Granular profiling of GPU Dirichlet-HDP transition sampling.
- Model comparison and multi-seed Expected Marginal Likelihood (EML) calculation engines.
