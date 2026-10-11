"""Read explicitly named pose files and identify selected media inputs."""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fpcode.viz.selection.bouts import base_session_id


def load_fused_tracks(path: Path, bodyparts: Sequence[str]):
    """Read original observations in the caller's requested landmark order."""
    import pandas as pd

    df = pd.read_hdf(path)
    scorer = df.columns.levels[0][0]
    x = np.column_stack([df[(scorer, bp, "x")].to_numpy(dtype=np.float32) for bp in bodyparts])
    y = np.column_stack([df[(scorer, bp, "y")].to_numpy(dtype=np.float32) for bp in bodyparts])
    conf = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy(dtype=np.float32) for bp in bodyparts])
    return np.stack([x, y], axis=-1), conf


def media_file_identity(manifest, syllable: int, *, fused_artifacts: Mapping[str, Path]):
    """Record selected media and supplied fused files by path, size and mtime."""
    paths = set()
    for bout in manifest["syllables"][str(syllable)]["bouts"]:
        if bout.get("is_blank"):
            continue
        paths.update(str(bout[key]) for key in ("top_vid", "bot_vid", "top_h5", "bot_h5"))
        base = base_session_id(bout["segment_name"])
        if base in fused_artifacts:
            fused = Path(fused_artifacts[base])
            paths.update((str(fused), str(fused.with_name(fused.name + ".manifest.json"))))
    result = {}
    for name in sorted(paths):
        stat = Path(name).stat()
        result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result
