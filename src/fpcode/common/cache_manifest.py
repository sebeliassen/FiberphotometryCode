"""
core.cache_manifest
===================
Verifiable caches and atomic publication.

Provides:
  * ``atomic_path(target)``   - context manager yielding a unique temp path next
                                to ``target``; published with ``os.replace`` only
                                if the ``with`` block succeeds. Partial/failed
                                writes never appear at ``target``.
  * ``fingerprint_file``      - cheap identity (size + mtime_ns); optional content
                                hash only for small files. Huge unchanged files are
                                never re-hashed on each invocation.
  * ``source_fingerprint``    - sha256 over implementation source files, so a code
                                change invalidates caches that depend on it.
  * ``write_manifest`` / ``verify_manifest`` - sidecar ``<artifact>.manifest.json``
                                recording inputs, configuration, code version and
                                dependency versions. Old artifacts without a
                                manifest are reported as UNVERIFIED (never deleted).
  * ``verify_artifact_set``   - validates several related artifacts as one
                                consistent set (same ``set_id`` / config hash).

Pure stdlib: safe to import without JAX / h5py / pandas.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SUFFIX = ".manifest.json"
SMALL_FILE_HASH_LIMIT = 16 * 1024 * 1024  # content-hash only files <= 16 MiB

PathLike = Union[str, Path]

# Verification status strings
OK = "OK"
UNVERIFIED_NO_MANIFEST = "UNVERIFIED_NO_MANIFEST"
MISSING_ARTIFACT = "MISSING_ARTIFACT"
STALE = "STALE"
CORRUPT_MANIFEST = "CORRUPT_MANIFEST"


# --------------------------------------------------------------------------- #
# Atomic publication
# --------------------------------------------------------------------------- #
def manifest_path_for(artifact: PathLike) -> Path:
    artifact = Path(artifact)
    return artifact.with_name(artifact.name + MANIFEST_SUFFIX)


def _tmp_name(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextlib.contextmanager
def atomic_path(target: PathLike):
    """Yield a unique temp path beside ``target``; atomically publish on success.

    The temp file keeps a non-matching name (leading dot, ``.tmp`` suffix) so
    globbing for ``*.h5`` / ``*.pkl`` never picks up incomplete writes. On any
    exception the temp file is removed and ``target`` is left untouched.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(target)
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"atomic_path: writer did not create {tmp}")
        _fsync_path(tmp)
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def atomic_write_text(target: PathLike, text: str) -> None:
    with atomic_path(target) as tmp:
        Path(tmp).write_text(text)


def atomic_write_json(target: PathLike, obj: Any) -> None:
    atomic_write_text(target, json.dumps(obj, indent=2, sort_keys=True, default=str))


def cleanup_stale_temps(directory: PathLike, older_than_s: float = 3600.0) -> List[Path]:
    """Remove orphaned ``.*.tmp`` files left by killed writers (older than N seconds)."""
    removed = []
    now = time.time()
    for p in Path(directory).glob(".*.tmp"):
        try:
            if now - p.stat().st_mtime > older_than_s:
                p.unlink()
                removed.append(p)
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint_file(path: PathLike, hash_small: bool = True) -> Dict[str, Any]:
    """Cheap identity of an input file: path, size, mtime_ns (+ sha256 if small).

    Large files are identified by stat only, so repeated invocations do not
    re-read multi-GB inputs.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    fp: Dict[str, Any] = {
        "path": str(p),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if hash_small and st.st_size <= SMALL_FILE_HASH_LIMIT:
        fp["sha256"] = _sha256_file(p)
    return fp


def source_fingerprint(paths: Iterable[PathLike]) -> str:
    """sha256 over the bytes of implementation source files (order-independent)."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode())
        # Missing implementation files are configuration errors, not stable
        # fingerprints. Silently hashing '<missing>' hid stale paths after moves.
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def config_hash(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def dependency_versions(modules: Sequence[str] = ("numpy", "pandas", "h5py", "jax")) -> Dict[str, str]:
    """Versions of already-imported dependencies (never imports them itself)."""
    versions = {"python": platform.python_version()}
    for name in modules:
        mod = sys.modules.get(name)
        if mod is not None:
            versions[name] = str(getattr(mod, "__version__", "unknown"))
    return versions


def _inputs_match(recorded: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Optional[str]:
    """Return None if inputs match, else a human-readable reason."""
    rec = {d["path"]: d for d in recorded}
    cur = {d["path"]: d for d in current}
    if set(rec) != set(cur):
        return f"input set changed: +{sorted(set(cur) - set(rec))} -{sorted(set(rec) - set(cur))}"
    for path, r in rec.items():
        c = cur[path]
        if r.get("exists") != c.get("exists"):
            return f"input existence changed: {path}"
        if not c.get("exists"):
            continue
        if "sha256" in r and "sha256" in c:
            if r["sha256"] != c["sha256"]:
                return f"input content changed: {path}"
        elif (r.get("size"), r.get("mtime_ns")) != (c.get("size"), c.get("mtime_ns")):
            return f"input size/mtime changed: {path}"
    return None


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def write_manifest(
    artifact: PathLike,
    *,
    kind: str,
    inputs: Sequence[PathLike] = (),
    config: Optional[Dict[str, Any]] = None,
    code_files: Sequence[PathLike] = (),
    set_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``<artifact>.manifest.json`` atomically *after* the artifact is published."""
    artifact = Path(artifact)
    if not artifact.exists():
        raise FileNotFoundError(f"Cannot write manifest; artifact missing: {artifact}")
    st = artifact.stat()
    config = dict(config or {})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "created_unix": time.time(),
        "artifact": {"name": artifact.name, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)},
        "inputs": [fingerprint_file(p) for p in inputs],
        "config": config,
        "config_hash": config_hash(config),
        "code_hash": source_fingerprint(code_files) if code_files else None,
        "code_files": [str(p) for p in code_files],
        "dependencies": dependency_versions(),
        "set_id": set_id,
        "extra": extra or {},
    }
    atomic_write_json(manifest_path_for(artifact), manifest)
    return manifest


def read_manifest(artifact: PathLike) -> Optional[Dict[str, Any]]:
    mp = manifest_path_for(artifact)
    if not mp.exists():
        return None
    try:
        value = json.loads(mp.read_text())
        return value if isinstance(value, dict) else {"_corrupt": True}
    except (OSError, json.JSONDecodeError):
        return {"_corrupt": True}


def _manifest_shape_valid(manifest: Dict[str, Any]) -> bool:
    artifact = manifest.get("artifact")
    inputs = manifest.get("inputs")
    return (
        isinstance(artifact, dict)
        and isinstance(artifact.get("name"), str)
        and isinstance(artifact.get("size"), int)
        and isinstance(artifact.get("mtime_ns"), int)
        and isinstance(manifest.get("config_hash"), str)
        and isinstance(inputs, list)
        and all(isinstance(item, dict) and isinstance(item.get("path"), str)
                and isinstance(item.get("exists"), bool) for item in inputs)
        and len({item["path"] for item in inputs}) == len(inputs)
    )


def verify_manifest(
    artifact: PathLike,
    *,
    kind: Optional[str] = None,
    inputs: Optional[Sequence[PathLike]] = None,
    config: Optional[Dict[str, Any]] = None,
    code_files: Sequence[PathLike] = (),
    set_id: Optional[str] = None,
    check_inputs: bool = True,
) -> Tuple[str, List[str]]:
    """Return ``(status, reasons)``. Status is one of the module-level constants.

    Anything other than ``OK`` means the cache must not be trusted. Old
    artifacts with no manifest are ``UNVERIFIED_NO_MANIFEST`` (kept on disk).
    """
    artifact = Path(artifact)
    if not artifact.exists():
        return MISSING_ARTIFACT, [f"{artifact} does not exist"]

    m = read_manifest(artifact)
    if m is None:
        return UNVERIFIED_NO_MANIFEST, ["no manifest sidecar; artifact provenance unknown"]
    if (m.get("_corrupt") or m.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or not _manifest_shape_valid(m)):
        return CORRUPT_MANIFEST, ["manifest unreadable or unsupported schema_version"]

    reasons: List[str] = []
    st = artifact.stat()
    rec = m.get("artifact", {})
    if (rec.get("name") != artifact.name or rec.get("size") != int(st.st_size)
            or rec.get("mtime_ns") != int(st.st_mtime_ns)):
        reasons.append("artifact size/mtime differs from manifest (modified or truncated after publication)")
    if kind is not None and m.get("kind") != kind:
        reasons.append(f"kind mismatch: manifest={m.get('kind')} expected={kind}")
    if config is not None and m.get("config_hash") != config_hash(dict(config)):
        reasons.append("configuration differs from the one used to build this artifact")
    if code_files:
        if m.get("code_hash") != source_fingerprint(code_files):
            reasons.append("implementation source changed since artifact was built")
    if set_id is not None and m.get("set_id") != set_id:
        reasons.append(f"set_id mismatch: manifest={m.get('set_id')} expected={set_id}")
    # Omitted inputs skip input comparison; an explicit empty sequence means
    # there are now no inputs and must detect removal of the last dependency.
    if check_inputs and inputs is not None:
        why = _inputs_match(m.get("inputs", []), [fingerprint_file(p) for p in inputs])
        if why:
            reasons.append(why)

    return (STALE, reasons) if reasons else (OK, [])


def verify_artifact_set(
    artifacts: Dict[str, PathLike],
    *,
    set_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Tuple[str, List[str]]]]:
    """Validate related artifacts (e.g. fused H5, PCA, init, results) as one set.

    The set is OK only if every member is OK, shares the same ``config_hash``
    and (if given) the same ``set_id``.
    """
    per: Dict[str, Tuple[str, List[str]]] = {}
    hashes = set()
    for name, path in artifacts.items():
        status, reasons = verify_manifest(path, set_id=set_id, config=config, check_inputs=False)
        per[name] = (status, reasons)
        m = read_manifest(path)
        if m and _manifest_shape_valid(m):
            hashes.add(m.get("config_hash"))
    overall = OK
    for status, _ in per.values():
        if status != OK:
            overall = status
            break
    if overall == OK and len(hashes) > 1:
        overall = STALE
        for name in per:
            per[name] = (STALE, per[name][1] + ["artifact set mixes different config hashes"])
    return overall, per
