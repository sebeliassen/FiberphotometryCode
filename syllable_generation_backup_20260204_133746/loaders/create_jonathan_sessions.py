# scripts/build_sessions.py
"""
Build and cache 'sessions' for a dataset.
- Edit the constants below.
- If OUTPUT_SESSIONS_PKL exists and OVERWRITE=False, it will be loaded and returned.
- Otherwise, sessions are built from paths and saved to OUTPUT_SESSIONS_PKL.

Requires:
  - syllable_generation.dialects.jonathan.context.BuildContext
  - syllable_generation.dialects.jonathan.build.build
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
from typing import Optional

# ========= constants (edit these) =========
DATA_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full")
H5_DIR    = DATA_ROOT / "h5_files"
VIDEO_DIR = DATA_ROOT / "video_files"
CSV_ROOT  = DATA_ROOT / "Jonathan_IFxDN_csvs"


# If True, skips exclusions (manual/auto masks) and saves as '..._dirty.pkl'
FORCE_DIRTY = True

PROJECT_NAME = "jonathan_full"

# Where to save the sessions pickle
OUTPUT_SESSIONS_PKL = Path(__file__).resolve().parent.parent / "data" / f"jonathan_ifdn_sessions_{'dirty' if FORCE_DIRTY else 'sanitized'}.pkl"

# If False and OUTPUT_SESSIONS_PKL exists, load instead of rebuild
OVERWRITE = True
# ==========================================


def _validate_paths(*pairs: tuple[Path, str]) -> None:
    for p, label in pairs:
        if p is None:
            continue
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")


def build_sessions(
    *,
    data_root: Path | str,
    h5_dir: Path | str,
    video_dir: Path | str,
    csv_root: Path | str,
    output_pickle: Path | str,
    project: str = PROJECT_NAME,
    overwrite: bool = OVERWRITE,
    force_dirty: bool = FORCE_DIRTY,
):
    """Build or load sessions and persist them to `output_pickle`."""
    # Imports here to keep the script self-contained and import-time fast
    from syllable_generation.loaders.jonathan.context import BuildContext
    from syllable_generation.loaders.jonathan.build import build
    from syllable_generation.loaders.jonathan.steps import tracking, video, fiberphotometry
    from syllable_generation.projects.jonathan_openfield_example import enrich
    
    data_root = Path(data_root).resolve()
    h5_dir = Path(h5_dir).resolve()
    video_dir = Path(video_dir).resolve()
    csv_root = Path(csv_root).resolve()
    out_path = Path(output_pickle).resolve()

    # Fast path: reuse existing pickle
    if out_path.exists() and not overwrite:
        print(f"[sessions] Loading cached sessions from: {out_path}")
        with out_path.open("rb") as f:
            return pickle.load(f)

    # Validate inputs
    _validate_paths(
        (data_root, "DATA_ROOT"),
        (h5_dir, "H5_DIR"),
        (video_dir, "VIDEO_DIR"),
        (csv_root, "CSV_ROOT"),
    )

    # Custom enrich steps to pass the dirty/sanitized flag
    def _custom_enrich_pre(ctx, sessions):
        # points to .../projects/jonathan_openfield_example/overrides
        overrides_dir = Path(enrich.__file__).resolve().parent / "overrides"
        return enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=force_dirty)

    def _custom_enrich_post(ctx, sessions):
        # points to .../projects/jonathan_openfield_example/overrides
        overrides_dir = Path(enrich.__file__).parent / "overrides"
        return enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=force_dirty)

    CUSTOM_STEPS = [
        tracking.discover_sessions,
        video.attach_video_refs,
        fiberphotometry.attach_photometry_refs,
        _custom_enrich_pre,
        fiberphotometry.realize_photometry_columns,
        _custom_enrich_post,
        fiberphotometry.split_photometry,
        fiberphotometry.compute_dff_zscore,
    ]

    # Build via dialect
    print(f"[sessions] Building sessions (FORCE_DIRTY={force_dirty})...")
    ctx = BuildContext(
        root=data_root,
        h5_dir=h5_dir,
        video_dir=video_dir,
        csv_root=csv_root,
        project=project,
    )

    sessions = build(ctx, steps=CUSTOM_STEPS)

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[sessions] Saved {len(sessions)} sessions → {out_path}")

    return sessions


def main():
    print("== Build/Load SessionsV2 ==")
    print("DATA_ROOT:", DATA_ROOT)
    print("H5_DIR exists:", H5_DIR.exists())
    print("VIDEO_DIR exists:", VIDEO_DIR.exists())
    print("CSV_ROOT exists:", CSV_ROOT.exists())

    sessions = build_sessions(
        data_root=DATA_ROOT,
        h5_dir=H5_DIR,
        video_dir=VIDEO_DIR,
        csv_root=CSV_ROOT,
        output_pickle=OUTPUT_SESSIONS_PKL,
        project=PROJECT_NAME,
        overwrite=OVERWRITE,
        force_dirty=FORCE_DIRTY,
    )

    print(f"\nSessions count: {len(sessions)}")
    print("Saved to:", OUTPUT_SESSIONS_PKL.resolve())
    if sessions:
        # Try a quick peek
        try:
            print("\nFirst session summary:")
            print(sessions[0].summary())
        except Exception:
            tdf = getattr(sessions[0], "tracking_refs_df", None)
            vdf = getattr(sessions[0], "video_refs_df", None)
            print("tracking_refs_df:", getattr(tdf, "shape", None))
            print("video_refs_df:", getattr(vdf, "shape", None))


if __name__ == "__main__":
    main()
