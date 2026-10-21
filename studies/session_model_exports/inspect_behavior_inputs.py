"""Inspect explicitly selected local session inputs without modifying them.

Pickles are decoded only with --execute. This is a provenance/schema audit, not
a preprocessing pipeline or certification of scientific signal quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return _json_value(value.item())
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", None))}
    return str(value)


def inspect_session(session):
    """Return original identity, metadata, table schema and signal support."""
    import numpy as np
    import pandas as pd

    meta = getattr(session, "metadata", {})
    row = {
        "session_id": session.session_id,
        "project": getattr(session, "project", None),
        "metadata": _json_value(meta),
        "results_dict_key": getattr(session, "results_dict_key", None),
        "video_refs": [_json_value(vars(v)) for v in getattr(session, "video_refs", [])],
        "tables": {},
    }
    tables = {"all_data_df": getattr(session, "all_data_df", None)}
    for attr in ("photometry", "syllables", "tracking"):
        obj = getattr(session, attr, None)
        if isinstance(obj, dict):
            tables.update({f"{attr}:{key}": value for key, value in obj.items()})
        else:
            tables[attr] = obj
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            continue
        desc = {"rows": len(table), "columns": [str(c) for c in table.columns],
                "index_name": table.index.name, "index_dtype": str(table.index.dtype),
                "index_unique": bool(table.index.is_unique),
                "index_increasing": bool(table.index.is_monotonic_increasing)}
        if len(table):
            desc["index_bounds"] = _json_value([table.index[0], table.index[-1]])
        if "syllable" in table and isinstance(table["syllable"], pd.Series):
            labels = table["syllable"]
            desc["labels"] = {"count": int(labels.notna().sum()),
                              "types": _json_value(sorted(labels.dropna().unique().tolist()))}
        desc["signals"] = {}
        for col in table.columns:
            if not isinstance(col, str) or not col.startswith("signal_"):
                continue
            series = table[col]
            if not isinstance(series, pd.Series) or not pd.api.types.is_numeric_dtype(series):
                continue
            a = series.to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(a)
            ix = np.flatnonzero(valid)
            desc["signals"][col] = {
                "finite_rows": int(valid.sum()), "total_rows": len(a),
                "first_finite_index": _json_value(table.index[ix[0]]) if len(ix) else None,
                "last_finite_index": _json_value(table.index[ix[-1]]) if len(ix) else None,
            }
        if "frame_index" in table and isinstance(table["frame_index"], pd.Series):
            desc["frame_index_equals_index"] = bool(np.array_equal(
                table["frame_index"].to_numpy(), table.index.to_numpy()))
        row["tables"][name] = desc
    return row


def export_label_tables(session, output):
    """Export label/index arrays for independent model binding checks only.

    Index coordinates retain their original values. They are not certified as
    native video frames by this export and must be verified against source VFI.
    """
    import numpy as np
    import pandas as pd

    tables = {"all_data_df": getattr(session, "all_data_df", None)}
    syllables = getattr(session, "syllables", None)
    if isinstance(syllables, dict):
        tables.update({f"syllables_{i}": t for i, t in enumerate(syllables.values())})
    else:
        tables["syllables"] = syllables
    paths = {}
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame) or "syllable" not in table:
            continue
        if not pd.api.types.is_integer_dtype(table.index.dtype) or not table.index.is_unique:
            continue
        series = table["syllable"]
        if not isinstance(series, pd.Series):
            continue
        valid = series.notna().to_numpy()
        labels = series.to_numpy(dtype=float, na_value=np.nan)[valid]
        if not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
            raise ValueError("label values must be finite integers on supported rows")
        frames = table.index.to_numpy(dtype=np.int64)[valid]
        key = hashlib.sha256(str(session.session_id).encode()).hexdigest()[:16]
        path = output / f"{key}_{name}.npz"
        arrays = {"table_index": frames, "labels": labels.astype(np.int64)}
        if "frame_index" in table:
            arrays["frame_index"] = table["frame_index"].to_numpy()[valid]
        for column in ("velocity_px_s", "angular_velocity", "centroid_x", "centroid_y", "heading",
                       *(f"latent_state_{i}" for i in range(10))):
            if column in table and isinstance(table[column], pd.Series):
                arrays[column] = table[column].to_numpy(dtype=float, na_value=np.nan)[valid]
        np.savez_compressed(path, **arrays)
        paths[name] = str(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram-cap-mib", type=int, default=24576)
    parser.add_argument("--export-labels", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output.resolve()
    if args.ram_cap_mib < 256:
        parser.error("RAM cap must be at least 256 MiB")
    if not source.is_file():
        parser.error("selected input does not exist")
    if output == source or source.is_relative_to(output):
        parser.error("output must be a separate directory, not an input ancestor")
    if not args.execute:
        print(json.dumps({"input": str(source), "bytes": source.stat().st_size,
                          "output": str(output), "execute": False}))
        return 0
    limit = args.ram_cap_mib * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.monotonic()
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    # Import the established local compatibility shim; no backend imports.
    from fpcode.common.file_io import safe_load_pickle
    sessions = safe_load_pickle(source)
    if not isinstance(sessions, (tuple, list)):
        raise ValueError("selected artifact must contain a session list")
    ids = [s.session_id for s in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session identities")
    rows = [inspect_session(s) for s in sessions]
    output.mkdir(parents=True, exist_ok=True)
    if args.export_labels:
        labels_dir = output / "labels"
        labels_dir.mkdir(exist_ok=True)
        for session, row in zip(sessions, rows):
            row["label_exports"] = export_label_tables(session, labels_dir)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("input changed during audit")
    backend_imports = sorted(m for m in sys.modules if m.split(".")[0] in {"jax", "torch", "keypoint_moseq"})
    if backend_imports:
        raise RuntimeError("unexpected modelling backend import during metadata audit")
    report = {"input": str(source), "sha256": digest.hexdigest(), "bytes": after.st_size,
              "session_count": len(rows), "sessions": rows,
              "resource_settings": {"address_space_cap_mib": args.ram_cap_mib,
                                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                    "elapsed_s": time.monotonic() - started},
              "backend_imports": backend_imports,
              "limits": ["signal support does not certify preprocessing or biological validity",
                         "table indices require independent VFI verification",
                         "original session metadata is reported without patching"]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ("input", "session_count", "resource_settings")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
