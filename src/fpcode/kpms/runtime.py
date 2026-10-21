"""Compatibility facade. Implementations live in focused modules.

Historical imports and object identities remain available; optional backends load
only when a backend attribute is requested.
"""

from importlib import import_module

_EXPORTS = {'hmm_posterior_sample': 'fpcode.kpms.backend.kernels.discrete', 'configure_jax_runtime': 'fpcode.kpms.backend.configuration', 'BatchConfig': 'fpcode.kpms.backend.configuration', 'CONFIG': 'fpcode.kpms.backend.configuration', 'BATCH_SIZE': 'fpcode.kpms.backend.configuration', 'SCALES_BS': 'fpcode.kpms.backend.configuration', 'HEADING_BS': 'fpcode.kpms.backend.configuration', 'GPU_LOC_BS': 'fpcode.kpms.backend.configuration', 'DISCRETE_BS': 'fpcode.kpms.backend.configuration', 'precompute_optimal_batch_config': 'fpcode.kpms.backend.sizing', 'set_batch_sizes': 'fpcode.kpms.backend.configuration', 'KPMSStateCache': 'fpcode.kpms.backend.state', '_orig_z': 'fpcode.kpms.backend.kernels.discrete', '_jit_compute_discrete_emissions': 'fpcode.kpms.backend.kernels.discrete', '_numba_sample_categorical': 'fpcode.kpms.backend.kernels.discrete', '_numba_ffbs_single_sequence': 'fpcode.kpms.backend.kernels.discrete', '_numba_ffbs_all_parallel': 'fpcode.kpms.backend.kernels.discrete', '_jit_resample_discrete_batch': 'fpcode.kpms.backend.kernels.discrete', 'fast_resample_discrete_stateseqs': 'fpcode.kpms.backend.kernels.discrete', '_orig_s': 'fpcode.kpms.backend.kernels.scales_heading', '_orig_h': 'fpcode.kpms.backend.kernels.scales_heading', '_resample_scales_kernel': 'fpcode.kpms.backend.kernels.scales_heading', 'batched_resample_scales': 'fpcode.kpms.backend.kernels.scales_heading', '_resample_heading_kernel': 'fpcode.kpms.backend.kernels.scales_heading', 'batched_resample_heading': 'fpcode.kpms.backend.kernels.scales_heading', '_orig_ar': 'fpcode.kpms.backend.kernels.autoregression', '_zero_alloc_accumulate_ar_stats': 'fpcode.kpms.backend.kernels.autoregression', '_numba_accumulate_ar_stats': 'fpcode.kpms.backend.kernels.autoregression', '_jit_solve_mniw_posteriors': 'fpcode.kpms.backend.kernels.autoregression', '_numpy_get_lags': 'fpcode.kpms.backend.kernels.autoregression', '_numpy_pad_affine': 'fpcode.kpms.backend.kernels.autoregression', 'vectorized_resample_ar_params': 'fpcode.kpms.backend.kernels.autoregression', '_numba_banded_cholesky_inplace': 'fpcode.kpms.backend.kernels.continuous', '_numba_banded_forward_solve': 'fpcode.kpms.backend.kernels.continuous', '_numba_banded_forward_solve_inplace': 'fpcode.kpms.backend.kernels.continuous', '_numba_banded_backward_solve': 'fpcode.kpms.backend.kernels.continuous', '_numba_assemble_and_solve_all_parallel': 'fpcode.kpms.backend.kernels.continuous', '_numba_fused_rigid_transform': 'fpcode.kpms.backend.kernels.continuous', '_orig_x': 'fpcode.kpms.backend.kernels.continuous', 'pure_numba_resample_continuous_stateseqs': 'fpcode.kpms.backend.kernels.continuous', '_orig_v': 'fpcode.kpms.backend.kernels.location', '_jit_resample_v_batch': 'fpcode.kpms.backend.kernels.location', 'batched_gpu_resample_location': 'fpcode.kpms.backend.kernels.location', '_ORIGINAL_FUNCTIONS': 'fpcode.kpms.backend.patching', 'fast_consolidated_check_for_nans': 'fpcode.kpms.backend.residency', 'safe_device_put_as_scalar': 'fpcode.kpms.backend.residency', 'apply_optimizations': 'fpcode.kpms.backend.patching', 'restore_original_kernels': 'fpcode.kpms.backend.patching', 'kpms_optimizations': 'fpcode.kpms.backend.patching'}

__all__ = [name for name in _EXPORTS if not name.startswith("_")]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
