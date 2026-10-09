"""KPMS patching; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from fpcode.kpms.backend.configuration import CONFIG
from fpcode.kpms.backend.kernels.autoregression import vectorized_resample_ar_params
from fpcode.kpms.backend.kernels.continuous import pure_numba_resample_continuous_stateseqs
from fpcode.kpms.backend.kernels.discrete import fast_resample_discrete_stateseqs
from fpcode.kpms.backend.kernels.location import batched_gpu_resample_location
from fpcode.kpms.backend.kernels.scales_heading import batched_resample_heading
from fpcode.kpms.backend.kernels.scales_heading import batched_resample_scales
from fpcode.kpms.backend.residency import fast_consolidated_check_for_nans
from fpcode.kpms.backend.residency import safe_device_put_as_scalar
from fpcode.kpms.backend.state import KPMSStateCache
from jax_moseq.models import keypoint_slds
import contextlib

from .patch_registry import PatchSet, temporary_patches
import jax_moseq.models.arhmm as arhmm_model
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs


_ORIGINAL_FUNCTIONS = {
    "arhmm_resample_discrete": arhmm_gibbs.resample_discrete_stateseqs,
    "arhmm_resample_ar": arhmm_gibbs.resample_ar_params,
    "kp_slds_resample_scales": slds_gibbs.resample_scales,
    "kp_slds_resample_heading": slds_gibbs.resample_heading,
    "kp_slds_resample_x": slds_gibbs.resample_continuous_stateseqs,
    "kp_slds_resample_loc": slds_gibbs.resample_location,
}


def _optimization_bindings():
    continuous = (pure_numba_resample_continuous_stateseqs if CONFIG.use_banded_continuous
                  else _ORIGINAL_FUNCTIONS["kp_slds_resample_x"])
    bindings = []
    for module in (arhmm_gibbs, arhmm_model):
        bindings.extend([
            (module, "resample_discrete_stateseqs", fast_resample_discrete_stateseqs),
            (module, "resample_ar_params", vectorized_resample_ar_params),
        ])
    for module in (slds_gibbs, keypoint_slds):
        bindings.extend([
            (module, "resample_scales", batched_resample_scales),
            (module, "resample_heading", batched_resample_heading),
            (module, "resample_continuous_stateseqs", continuous),
            (module, "resample_location", batched_gpu_resample_location),
        ])
    # Optional helpers have separate originals in each namespace. Restoring a
    # fitting function into utils used to lose utils' own implementation.
    try:
        import keypoint_moseq.fitting as fitting
        import jax_moseq.utils.utils as utils
    except ImportError:
        pass
    else:
        bindings.extend([
            (fitting, "check_for_nans", fast_consolidated_check_for_nans),
            (fitting, "device_put_as_scalar", safe_device_put_as_scalar),
            (utils, "device_put_as_scalar", safe_device_put_as_scalar),
        ])
    return bindings


_ACTIVE_PATCHES = None


def apply_optimizations():
    """Install explicit backend patches once, retaining each original target."""
    global _ACTIVE_PATCHES
    if _ACTIVE_PATCHES is None:
        _ACTIVE_PATCHES = PatchSet(_optimization_bindings()).apply()


def restore_original_kernels():
    """Restore the exact functions present before apply_optimizations()."""
    global _ACTIVE_PATCHES
    patches, _ACTIVE_PATCHES = _ACTIVE_PATCHES, None
    try:
        if patches is not None:
            patches.restore()
    finally:
        KPMSStateCache.clear()


@contextlib.contextmanager
def kpms_optimizations():
    """Temporarily patch the backend; nested scopes restore their entry state."""
    try:
        with temporary_patches(_optimization_bindings()):
            yield
    finally:
        KPMSStateCache.clear()
