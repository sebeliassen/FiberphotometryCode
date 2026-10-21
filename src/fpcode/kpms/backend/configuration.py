"""KPMS configuration; numerical algorithms preserved from the baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def configure_jax_runtime(
    preallocate: bool = False,
    allocator: str = "cuda_malloc_async",
    mem_fraction: Optional[float] = 0.75,
    force_cpu: bool = False,
) -> None:
    """Configures environment variables for JAX / XLA memory management before JAX initializes."""
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        if allocator:
            os.environ["TF_GPU_ALLOCATOR"] = allocator
        if not preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if mem_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)


@dataclass
class BatchConfig:
    discrete_bs: int = 10
    scales_bs: int = 20
    heading_bs: int = 20
    gpu_loc_bs: int = 20
    continuous_chunk_size: int = 50  # Right-sized from 300 to 50 (reduces banded buffer from 9.6GB to 1.6GB)
    use_numba_hmm: bool = True       # 15.6x faster parallel CPU FFBS for discrete states (z)
    use_banded_continuous: bool = True # True: Numba Banded Cholesky (58x); False: Upstream JAX Kalman smoother
    banded_solver: str = "numba"     # "numba" (in-place parallel) or "lapack" (dpbtrf + dtbsv)


CONFIG = BatchConfig()


BATCH_SIZE = CONFIG.scales_bs


SCALES_BS = CONFIG.scales_bs


HEADING_BS = CONFIG.heading_bs


GPU_LOC_BS = CONFIG.gpu_loc_bs


DISCRETE_BS = CONFIG.discrete_bs


def set_batch_sizes(scale_factor=0.5, **overrides):
    """Dynamically adjust all batch sizes (e.g. upon OOM backoff)."""
    if overrides:
        validated = {}
        for key, value in overrides.items():
            if not hasattr(CONFIG, key):
                raise ValueError(f"Unknown BatchConfig parameter: '{key}'")
            if key in ("use_banded_continuous", "use_numba_hmm"):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a boolean")
                validated[key] = value
            elif key == "banded_solver":
                if value not in ("numba", "lapack"):
                    raise ValueError("banded_solver must be numba or lapack")
                validated[key] = value
            else:
                converted = int(value)
                if isinstance(value, bool) or converted != value or converted <= 0:
                    raise ValueError(f"Batch size '{key}' must be a positive integer")
                validated[key] = converted
        for key, value in validated.items():
            setattr(CONFIG, key, value)
    else:
        if not 0 < scale_factor <= 1:
            raise ValueError("scale_factor must be in (0, 1]")
        CONFIG.discrete_bs = max(1, int(CONFIG.discrete_bs * scale_factor))
        CONFIG.scales_bs = max(1, int(CONFIG.scales_bs * scale_factor))
        CONFIG.heading_bs = max(1, int(CONFIG.heading_bs * scale_factor))
        CONFIG.gpu_loc_bs = max(1, int(CONFIG.gpu_loc_bs * scale_factor))
        CONFIG.continuous_chunk_size = max(1, int(CONFIG.continuous_chunk_size * scale_factor))
    print(f"[BatchConfig] Updated batch configuration: {CONFIG}")


HEADING_BS = 20
