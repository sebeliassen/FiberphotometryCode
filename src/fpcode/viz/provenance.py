"""Visualization provenance; historical V2 behavior preserved."""

from __future__ import annotations

from fpcode.viz.hashing import HashCache
from fpcode.viz.hashing import NumpyEncoder
from fpcode.viz.hashing import hash_files
from fpcode.viz.hashing import sha256_file
from fpcode.viz.hashing import sha256_json
from fpcode.viz.legacy_defaults import CALIBRATION_FILES
from fpcode.viz.legacy_defaults import CONFIG_PATH
from fpcode.viz.legacy_defaults import PREPROCESSED_PATH
from fpcode.viz.legacy_defaults import RESULTS_PATH
from fpcode.viz.legacy_defaults import SCHEMA_VERSION
from fpcode.viz.legacy_defaults import VFI_PATH
from fpcode.viz.legacy_io import media_file_identity
from fpcode.viz.selection.exemplars import selection_ids
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
import hashlib
import json
import os


def source_input_hashes(cache) -> Dict[str, str]:
    return {
        "results_sha256": cache.get(RESULTS_PATH),
        "config_sha256": cache.get(CONFIG_PATH),
        "vfi_sha256": cache.get(VFI_PATH),
        "preprocessed_sha256": cache.get(PREPROCESSED_PATH),
    }


def verify_manifest_inputs(manifest: Dict[str, Any], cache) -> None:
    """Reject selection provenance tied to different model/coordinate arrays."""
    expected = manifest.get("input_hashes")
    if not expected or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Regenerate a schema-v4 selection manifest before rendering")
    if not manifest.get("qc_config", {}).get("applied", False):
        raise ValueError("QC was bypassed; this debug manifest cannot certify a production gallery")
    current = source_input_hashes(cache)
    changed = [k for k, v in current.items() if expected.get(k) != v]
    if changed:
        raise ValueError("Selection input mismatch: " + ", ".join(changed))


def selection_provenance_matches(
    sidecar: Path, manifest: Dict[str, Any], syllable: int, code_files: Sequence[Path] = ()
) -> bool:
    """Gallery validation checks actual IDs/status, rather than sidecar existence."""
    try:
        rec = json.loads(Path(sidecar).read_text())
        key = rec["cache_key"]
        if (
            key.get("schema_version") != SCHEMA_VERSION
            or key.get("manifest_syllable_sha256") != manifest_syllable_hash(manifest, syllable)
            or key.get("manifest_global_sha256") != manifest_global_hash(manifest)
        ):
            return False
        if code_files and key.get("code_sha256") != hash_files(code_files):
            return False
        if code_files and (
            key.get("selected_source_file_identity") != media_file_identity(manifest, syllable)
            or key.get("calibration_sha256") != hash_files(CALIBRATION_FILES)
        ):
            return False
        wanted = selection_ids(manifest, syllable)
        if not wanted:
            return False
        if "rendered_tiles" in rec:
            tiles = rec["rendered_tiles"]
            if any(t.get("flagged_invalid") for t in tiles):
                return False
            actual = [t.get("bout_id") for t in tiles if not t.get("is_blank")]
        else:
            actual = rec["rendered_bouts"]
        return actual == wanted and len(actual) == len(set(actual))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest_syllable_hash(manifest: Dict[str, Any], syllable: int) -> str:
    return sha256_json(manifest["syllables"][str(int(syllable))])


def manifest_global_hash(manifest: Dict[str, Any]) -> str:
    return sha256_json({k: v for k, v in manifest.items() if k != "syllables"})


def make_cache_key(
    kind: str,
    syllable: int,
    manifest: Dict[str, Any],
    manifest_path: Path,
    hash_cache: HashCache,
    code_files: Iterable[Path],
    render_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Everything that must be identical for an existing output to be reused.  The
    selection for THIS syllable, the global selection settings, the model
    results, relevant code/config and the calibration/correspondence inputs are
    all hashed.  (A change to some other syllable's selection does not
    invalidate this syllable; the whole-manifest hash is still recorded in the
    sidecar for traceability.)
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "syllable": int(syllable),
        "manifest_syllable_sha256": manifest_syllable_hash(manifest, syllable),
        "manifest_global_sha256": manifest_global_hash(manifest),
        "results_sha256": hash_cache.get(RESULTS_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "vfi_sha256": hash_cache.get(VFI_PATH),
        "preprocessed_sha256": hash_cache.get(PREPROCESSED_PATH),
        "selected_source_file_identity": media_file_identity(manifest, syllable),
        "code_sha256": hash_files(code_files),
        "calibration_sha256": hash_files(CALIBRATION_FILES),
        "render_settings": render_settings,
    }


def sidecar_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".provenance.json")


def write_sidecar(
    output_paths: Sequence[Path],
    key: Dict[str, Any],
    manifest_path: Path,
    hash_cache: HashCache,
    payload: Dict[str, Any],
    sidecar_target: Optional[Path] = None,
) -> Path:
    """Atomically write the provenance sidecar next to the (first) output."""
    output_paths = [Path(p) for p in output_paths]
    target = Path(sidecar_target) if sidecar_target else sidecar_path(output_paths[0])
    record = {
        "cache_key": key,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "outputs": {p.name: {"size": p.stat().st_size} for p in output_paths if p.exists()},
        **payload,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, cls=NumpyEncoder))
    os.replace(tmp, target)
    return target


def cache_decision(
    output_paths: Sequence[Path], key: Dict[str, Any], sidecar_target: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Reuse an output only if (a) all files exist with the recorded sizes, and
    (b) its sidecar cache key equals the current key.  Returns (reuse, reason);
    an existing large file alone never suppresses re-rendering.
    """
    output_paths = [Path(p) for p in output_paths]
    for p in output_paths:
        if not p.exists():
            return False, f"missing output {p.name}"
    side = Path(sidecar_target) if sidecar_target else sidecar_path(output_paths[0])
    if not side.exists():
        return False, "no provenance sidecar (legacy or unverified output)"
    try:
        rec = json.loads(side.read_text())
    except Exception as exc:
        return False, f"unreadable sidecar ({exc})"
    for p in output_paths:
        recorded = rec.get("outputs", {}).get(p.name)
        if recorded is None or recorded.get("size") != p.stat().st_size:
            return False, f"output {p.name} differs from recorded size"
    old = rec.get("cache_key", {})
    if old == key:
        return True, "cache key matches"
    changed = sorted(k for k in set(old) | set(key) if old.get(k) != key.get(k))
    return False, "cache key changed: " + ", ".join(changed)


def kpms_provenance() -> Dict[str, Any]:
    """Record installed package versions and a hash of the exact KPMS source used."""
    import importlib.metadata as md
    import inspect

    import keypoint_moseq.util as kutil

    funcs = ["sample_instances", "get_instance_trajectories", "filter_centroids_headings", "filter_angle"]
    sources = {f: inspect.getsource(getattr(kutil, f)) for f in funcs}

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    import numpy
    import scipy
    import sklearn

    return {
        "keypoint_moseq": ver("keypoint_moseq"),
        "jax_moseq": ver("jax_moseq"),
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "util_module_path": kutil.__file__,
        "function_source_sha256": {f: hashlib.sha256(s.encode()).hexdigest() for f, s in sources.items()},
    }
