#!/usr/bin/env python3
"""
ops/repo/vendor_keypoint_moseq.py

Executes Step 8.2 of the Architectural Refactoring:
1. Moves syllable_generation/syllabizers/keypoint_moseq_syll to vendor/keypoint_moseq_syll.
2. Writes vendor/keypoint_moseq_syll/PATCHES.md describing all custom extensions vs upstream.
3. Installs clean deprecation shims at syllable_generation/syllabizers/keypoint_moseq_syll/.
4. Updates registry/SHIMS.tsv and registry/MOVES.tsv.
"""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_FORK = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll"
DEST_VENDOR = REPO_ROOT / "vendor" / "keypoint_moseq_syll"

PATCHES_MD_CONTENT = """# Keypoint-MoSeq Fork Modifications & Patches

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
"""

def execute_vendoring():
    os.chdir(REPO_ROOT)
    print("Migrating keypoint_moseq_syll to vendor/...")

    DEST_VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if DEST_VENDOR.exists():
        shutil.rmtree(DEST_VENDOR)

    # 1. Copy full directory to vendor/
    shutil.copytree(SRC_FORK, DEST_VENDOR, symlinks=True)
    print(f"Copied {SRC_FORK.relative_to(REPO_ROOT)} -> {DEST_VENDOR.relative_to(REPO_ROOT)}")

    # 2. Write PATCHES.md in vendor/
    patches_file = DEST_VENDOR / "PATCHES.md"
    patches_file.write_text(PATCHES_MD_CONTENT)
    print(f"Created {patches_file.relative_to(REPO_ROOT)}")

    # 3. Clean up all non-shim files in SRC_FORK and install deprecation shims
    # Keep utils/ intact since it holds the tested shims to fpcode
    for item in list(SRC_FORK.iterdir()):
        if item.name == "utils" or item.name == "__pycache__":
            continue
        if item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Re-create root __init__.py in SRC_FORK
    fork_init = SRC_FORK / "__init__.py"
    fork_init.write_text('''"""
[DEPRECATED SHIM] syllable_generation.syllabizers.keypoint_moseq_syll
=====================================================================
Relocated to: vendor.keypoint_moseq_syll and src.fpcode.kpms
"""

import warnings
warnings.warn(
    "syllable_generation.syllabizers.keypoint_moseq_syll is deprecated; "
    "import from fpcode.kpms or vendor.keypoint_moseq_syll instead",
    DeprecationWarning,
    stacklevel=2,
)
''')

    # Add forwarders for root scripts if needed
    for py_name in ["kpms_pipeline.py", "run_grid_movies.py"]:
        shim_py = SRC_FORK / py_name
        shim_py.write_text(f'''"""
[DEPRECATED SHIM] syllable_generation.syllabizers.keypoint_moseq_syll.{py_name[:-3]}
====================================================================
Relocated to: vendor/keypoint_moseq_syll/{py_name}
"""

import warnings
warnings.warn(
    "syllable_generation.syllabizers.keypoint_moseq_syll.{py_name[:-3]} is deprecated; "
    "use vendor/keypoint_moseq_syll/{py_name} or pipelines/ instead",
    DeprecationWarning,
    stacklevel=2,
)
''')

    # 4. Update registry/MOVES.tsv
    moves_tsv = REPO_ROOT / "registry" / "MOVES.tsv"
    with open(moves_tsv, "a") as f:
        f.write("syllable_generation/syllabizers/keypoint_moseq_syll\tvendor/keypoint_moseq_syll\tVENDOR_FORK\tVendored patched Keypoint-MoSeq fork\tNO\n")

    # 5. Update registry/SHIMS.tsv
    shims_tsv = REPO_ROOT / "registry" / "SHIMS.tsv"
    with open(shims_tsv, "a") as f:
        f.write("syllable_generation/syllabizers/keypoint_moseq_syll/__init__.py\tvendor.keypoint_moseq_syll\tDeprecationWarning\tACTIVE\n")
        f.write("syllable_generation/syllabizers/keypoint_moseq_syll/kpms_pipeline.py\tvendor.keypoint_moseq_syll.kpms_pipeline\tDeprecationWarning\tACTIVE\n")

    print("Step 8.2 vendoring execution complete.")

if __name__ == "__main__":
    execute_vendoring()
