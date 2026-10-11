"""Visualization hashing; historical V2 behavior preserved."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
import hashlib
import json
import numpy as np

from fpcode.common.cache_manifest import atomic_write_text


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        return _json_default(o)


class HashCache:
    """
    sha256 of large files (e.g. results.h5), memoized by (size, mtime_ns) in a
    small JSON file so each renderer does not re-read hundreds of MB.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        try:
            self._d = json.loads(self.cache_path.read_text())
            if not isinstance(self._d, dict):
                self._d = {}
        except Exception:
            self._d = {}

    def get(self, path: Path) -> str:
        path = Path(path)
        st = path.stat()
        key = str(path.resolve())
        rec = self._d.get(key)
        if rec and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
            return rec["sha256"]
        digest = sha256_file(path)
        self._d[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.cache_path, json.dumps(self._d, indent=1))
        return digest


def hash_files(paths: Iterable[Path]) -> Dict[str, str]:
    paths = [Path(path) for path in paths]
    out = {}
    for p in paths:
        p = Path(p)
        out[p.name] = sha256_file(p) if p.exists() else "MISSING"
    # Old renderers fingerprint their facade. Include its actual implementations
    # so edits to extracted selection/geometry code invalidate old output keys.
    if any(p.name in ("viz_v2_common.py", "v2.py") for p in paths):
        package_dir = Path(__file__).resolve().parent
        for implementation in sorted(package_dir.rglob("*.py")):
            out["fpcode/viz/" + implementation.relative_to(package_dir).as_posix()] = sha256_file(implementation)
    return out


def write_json_atomic(path: Path, obj: Any, indent: Optional[int] = 2) -> None:
    """Write JSON via a temp file + ``os.replace`` so readers never see a partial file."""
    atomic_write_text(path, json.dumps(obj, indent=indent, cls=NumpyEncoder))
