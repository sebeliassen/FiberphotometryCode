"""
core.eval_artifacts
===================
Read-only helpers for model evaluation (EML) that guarantee source checkpoints
are never modified and that derived artifacts cannot be silently reused across
a different checkpoint iteration, label mapping, or dataset.

Design rules
  * Source ``checkpoint.h5`` files are opened ``"r"`` only.
  * Syllable re-indexing is a *pure in-memory permutation* computed here and
    recorded (old->new map + hash) in the derived artifact's manifest.
  * The permutation uses a **stable** sort (ties broken by original index), so
    applying it twice is a no-op. (The previous ``argsort(freqs)[::-1]`` reversed
    the order of tied states on every call, so repeated runs kept relabeling.)
  * Models are only comparable (EML) if they share dataset identity: same
    recording keys/order, segment bounds (=> mask), array shapes, iteration.

Depends only on numpy + h5py (no JAX / keypoint_moseq import).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


class IncompatibleModelsError(ValueError):
    """Raised when candidate models cannot be compared on a common dataset."""


# --------------------------------------------------------------------------- #
# Label permutation (pure)
# --------------------------------------------------------------------------- #
def stable_frequency_order(freqs: Sequence[float]) -> np.ndarray:
    """Return ``index`` such that new label ``i`` is old state ``index[i]``.

    Sorted by descending frequency; ties keep ascending original index, so the
    result is the identity on an already-sorted labeling (idempotent).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    return np.lexsort((np.arange(len(freqs)), -freqs)).astype(np.int64)


def invert_permutation(index: Sequence[int]) -> np.ndarray:
    """``inv[old_state] = new_label``."""
    index = np.asarray(index, dtype=np.int64)
    inv = np.empty_like(index)
    inv[index] = np.arange(len(index), dtype=np.int64)
    return inv


def is_permutation(index: Sequence[int]) -> bool:
    index = np.asarray(index)
    return bool(index.ndim == 1 and np.array_equal(np.sort(index), np.arange(len(index))))


def mapping_hash(index: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()


def relabel_states(z: np.ndarray, inv_index: Sequence[int]) -> np.ndarray:
    """Map old state ids to new labels (``inv_index[z]``) without touching ``z``."""
    return np.asarray(inv_index)[z]


def old_to_new_map(index: Sequence[int]) -> Dict[int, int]:
    inv = invert_permutation(index)
    return {int(o): int(n) for o, n in enumerate(inv)}


# --------------------------------------------------------------------------- #
# Dataset identity
# --------------------------------------------------------------------------- #
def reconstruct_mask(bounds: np.ndarray, seg_len: int) -> np.ndarray:
    mask = np.zeros((len(bounds), seg_len), dtype=bool)
    for i, (s, e) in enumerate(np.asarray(bounds)):
        mask[i, : int(e - s)] = True
    return mask


def _decode(x: Any) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def metadata_hash(keys: Iterable[Any], bounds: np.ndarray) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(_decode(k).encode())
        h.update(b"\0")
    h.update(np.ascontiguousarray(np.asarray(bounds, dtype=np.int64)).tobytes())
    return h.hexdigest()


def read_checkpoint_identity(ckpt_path: PathLike, iteration: Optional[int] = None) -> Dict[str, Any]:
    """Read-only identity of a checkpoint's dataset and latest (or given) snapshot."""
    import h5py  # local import keeps module import cheap

    ckpt_path = Path(ckpt_path)
    st = ckpt_path.stat()
    with h5py.File(ckpt_path, "r") as f:
        iterations = sorted(int(i) for i in f["model_snapshots"])
        it = int(iteration) if iteration is not None else iterations[-1]
        snap = f[f"model_snapshots/{it}"]
        x_shape = tuple(int(v) for v in snap["states/x"].shape)
        z_shape = tuple(int(v) for v in snap["states/z"].shape)
        num_states = int(snap["params/pi"].shape[0])
        keys = f["metadata/arr0"][()]
        bounds = f["metadata/arr1"][()]
    seg_len = x_shape[1]
    mask_frames = int(sum(int(e - s) for s, e in np.asarray(bounds)))
    return {
        "path": str(ckpt_path),
        "iteration": it,
        "iterations": iterations,
        "num_states": num_states,
        "x_shape": x_shape,
        "z_shape": z_shape,
        "nlags": x_shape[1] - z_shape[1],
        "n_seq": x_shape[0],
        "seg_len": seg_len,
        "latent_dim": x_shape[2],
        "n_recordings": len(set(_decode(k) for k in keys)),
        "active_frames": mask_frames,
        "metadata_hash": metadata_hash(keys, bounds),
        "ckpt_size": int(st.st_size),
        "ckpt_mtime_ns": int(st.st_mtime_ns),
    }


# --------------------------------------------------------------------------- #
# Compatibility validation
# --------------------------------------------------------------------------- #
_MUST_MATCH = (
    ("metadata_hash", "recording keys / segment order / bounds (dataset + mask)"),
    ("x_shape", "latent state array shape (segments x length x latent dim)"),
    ("z_shape", "syllable state array shape"),
    ("nlags", "AR lag count"),
    ("latent_dim", "latent dimension (preprocessing / PCA identity)"),
    ("active_frames", "number of active frames"),
)


def validate_model_compatibility(
    identities: Dict[str, Dict[str, Any]],
    require_same_iteration: bool = True,
) -> None:
    """Raise ``IncompatibleModelsError`` listing every mismatch among candidates."""
    if len(identities) < 2:
        raise IncompatibleModelsError("need at least 2 models to compare")
    names = list(identities)
    ref_name, ref = names[0], identities[names[0]]
    problems: List[str] = []
    fields = list(_MUST_MATCH)
    if require_same_iteration:
        fields.append(("iteration", "evaluated checkpoint iteration"))
    for field, desc in fields:
        for n in names[1:]:
            a, b = ref[field], identities[n][field]
            if a != b:
                shown_a = a[:12] if isinstance(a, str) else a
                shown_b = b[:12] if isinstance(b, str) else b
                problems.append(f"{desc}: {ref_name}={shown_a} != {n}={shown_b}")
    if problems:
        raise IncompatibleModelsError("Candidate models are not comparable:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# Immutability guard
# --------------------------------------------------------------------------- #
def stat_signature(paths: Iterable[PathLike]) -> Dict[str, Tuple[int, int]]:
    out = {}
    for p in paths:
        st = Path(p).stat()
        out[str(p)] = (int(st.st_size), int(st.st_mtime_ns))
    return out


def assert_unchanged(before: Dict[str, Tuple[int, int]], after: Dict[str, Tuple[int, int]]) -> None:
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        raise RuntimeError(
            "Source checkpoint(s) were modified during evaluation (must be read-only): " + ", ".join(changed)
        )
