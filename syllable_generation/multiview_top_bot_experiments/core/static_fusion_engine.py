"""
core.static_fusion_engine
=========================
Occam's Razor Static Affine Tracking Fusion Engine:
1. TopCam coordinates for 17 dorsal body parts preserved verbatim.
2. BotCam ventral keypoints transformed into virtual TopCam arena space via a
   frozen, calibrated static affine map per arena (r >= 0.9993, residual ~3px).
3. Continuous sub-millisecond Basler crystal synchronization preserved.
4. NO frame-by-frame rotation, NO interpolation, NO anchor-confidence coupling.
   Raw DeepLabCut detection likelihoods are preserved verbatim.
5. Supports both '21kp' (17 dorsal + 4 paws) and '23kp' (17 dorsal + 4 paws + 2 bot anchors).
6. Full atomic publishing (.tmp -> rename) with cryptographic sidecar manifests.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .schemas import (
    TOP_USE_BODYPARTS,
    DEFAULT_ARENA_WIDTH,
    PathConfig,
)
from .camera_correspondence import get_session_mapping, get_session_calibration_details
from .cache_manifest import (
    OK as _CACHE_OK,
    UNVERIFIED_NO_MANIFEST as _CACHE_UNVERIFIED,
    atomic_path,
    verify_manifest,
    write_manifest,
)

_CORE_DIR = Path(__file__).resolve().parent
STATIC_TRANSFORMS_PATH = _CORE_DIR / "static_arena_transforms.json"
EPOCH_TRANSFORMS_PATH = _CORE_DIR / "epoch_arena_transforms.json"

# Load pre-calibrated static transforms
with open(STATIC_TRANSFORMS_PATH, "r") as f:
    STATIC_ARENA_TRANSFORMS = json.load(f)

if EPOCH_TRANSFORMS_PATH.exists():
    with open(EPOCH_TRANSFORMS_PATH, "r") as f:
        EPOCH_ARENA_TRANSFORMS = json.load(f)
else:
    EPOCH_ARENA_TRANSFORMS = None


def get_session_epoch(ts_str: str) -> str:
    """Documented epoch assignment based on experimental recording schedule."""
    if ts_str < "2026-04-01":
        return "epoch1_winter_spring_2026"
    return "epoch2_spring_post_shift_2026"


_FUSION_CODE_FILES = [
    _CORE_DIR / "static_fusion_engine.py",
    _CORE_DIR / "static_arena_transforms.json",
    _CORE_DIR / "epoch_arena_transforms.json",
    _CORE_DIR / "camera_correspondence.py",
    _CORE_DIR / "schemas.py",
]

PAW_MAP = {
    # Original BotCam name: (Mapped TopCam name, Swap Partner for horizontal reflection)
    "left_front_paw": ("left_front_paw", "right_front_paw"),
    "right_front_paw": ("right_front_paw", "left_front_paw"),
    "left_hind_paw": ("left_hind_paw", "right_hind_paw"),
    "right_hind_paw": ("right_hind_paw", "left_hind_paw"),
}

ANCHOR_MAP = {
    "nose_tip": "bot_nose_tip",
    "tail_base": "bot_tail_base",
}

def fuse_single_pair_static(
    pair_info: Dict[str, Any],
    out_dir: Path,
    keypoint_mode: str = "21kp",
    arena_width: float = DEFAULT_ARENA_WIDTH,
    truncate_aborted_tail: bool = True,
    use_epochs: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Fuses a matched TopCam and BotCam session using static per-arena affine projection.

    Parameters
    ----------
    pair_info : dict
        Session metadata containing 'sid', 'arena', 'top_h5', 'bot_h5'.
    out_dir : Path
        Destination directory for the fused HDF5 file.
    keypoint_mode : str, default='21kp'
        Either '21kp' (17 dorsal + 4 paws) or '23kp' (17 dorsal + 4 paws + 2 anchors).
    arena_width : float, default=972.0
        Native width of BotCam sensor for horizontal mirror reflection.
    truncate_aborted_tail : bool, default=True
        Whether to truncate trailing frames if BotCam acquisition aborted early.
    use_epochs : bool, default=False
        Whether to route transforms through epoch_arena_transforms.json based on acquisition epoch.
    overwrite : bool, default=False
        Whether to overwrite existing files.
    """
    if keypoint_mode not in ("21kp", "23kp"):
        raise ValueError(f"Invalid keypoint_mode '{keypoint_mode}'. Must be '21kp' or '23kp'.")

    sid = pair_info["sid"]
    arena = pair_info.get("arena", "Arena1")
    top_h5_path = Path(pair_info["top_h5"])
    bot_h5_path = Path(pair_info["bot_h5"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5_path = out_dir / f"{sid}_tandem_{keypoint_mode}.h5"

    ts = pair_info.get("ts", sid[:19])
    if use_epochs and EPOCH_ARENA_TRANSFORMS is not None:
        epoch = get_session_epoch(ts)
        transform_info = EPOCH_ARENA_TRANSFORMS.get(epoch, {}).get(arena)
        if transform_info is None:
            transform_info = STATIC_ARENA_TRANSFORMS.get(arena, STATIC_ARENA_TRANSFORMS.get("Arena1"))
            used_transforms_path = STATIC_TRANSFORMS_PATH
            resolved_epoch = "fallback_static"
        else:
            used_transforms_path = EPOCH_TRANSFORMS_PATH
            resolved_epoch = epoch
    else:
        transform_info = STATIC_ARENA_TRANSFORMS.get(arena, STATIC_ARENA_TRANSFORMS.get("Arena1"))
        used_transforms_path = STATIC_TRANSFORMS_PATH
        resolved_epoch = "static_global"

    A = np.array(transform_info["matrix_2x2"], dtype=np.float32)
    b = np.array(transform_info["translation_2d"], dtype=np.float32)

    manifest_config = {
        "keypoint_mode": keypoint_mode,
        "arena": arena,
        "arena_width": float(arena_width),
        "use_epochs": bool(use_epochs),
        "epoch": resolved_epoch,
        "affine_matrix": A.tolist(),
        "affine_translation": b.tolist(),
        "truncate_aborted_tail": bool(truncate_aborted_tail),
    }
    manifest_inputs = [top_h5_path, bot_h5_path, used_transforms_path]

    t0 = time.time()
    if not overwrite and out_h5_path.exists() and out_h5_path.stat().st_size > 500_000:
        cache_status, _ = verify_manifest(
            out_h5_path,
            kind=f"fused_static_{keypoint_mode}",
            inputs=manifest_inputs,
            config=manifest_config,
            code_files=_FUSION_CODE_FILES,
        )
        if cache_status in (_CACHE_OK, _CACHE_UNVERIFIED):
            return {
                "sid": sid,
                "status": "CACHED",
                "keypoint_mode": keypoint_mode,
                "path": str(out_h5_path),
                "time": 0.0,
            }

    try:
        df_top = pd.read_hdf(top_h5_path)
        df_bot = pd.read_hdf(bot_h5_path)
        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        N_top = len(df_top)
        N_bot = len(df_bot)

        # Continuous Basler crystal correspondence mapping
        mapping = get_session_mapping(sid, strict=True)
        if mapping is None:
            raise ValueError(f"Strict Basler crystal mapping unavailable for {sid}.")

        calibration = get_session_calibration_details(sid)

        N_map = min(N_top, len(mapping))
        mapping = mapping[:N_map]

        if truncate_aborted_tail:
            valid_mapped = np.where((mapping >= 0) & (mapping < N_bot))[0]
            if len(valid_mapped) > 0 and valid_mapped.max() < N_top - 500:
                N_top = int(valid_mapped.max() + 1)
                df_top = df_top.iloc[:N_top]
                mapping = mapping[:N_top]

        top_idx = np.where((mapping >= 0) & (mapping < N_bot))[0]
        bot_idx = mapping[top_idx]

        if len(top_idx) == 0:
            raise ValueError(f"No temporal overlap for {sid}")

        fused_dict = {}
        fused_scorer = "fused_multiview"

        # 1. TopCam keypoints preserved verbatim
        for bp in TOP_USE_BODYPARTS:
            fused_dict[(fused_scorer, bp, "x")] = df_top[(top_scorer, bp, "x")].iloc[:N_top].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "y")] = df_top[(top_scorer, bp, "y")].iloc[:N_top].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "likelihood")] = df_top[(top_scorer, bp, "likelihood")].iloc[:N_top].to_numpy(dtype=np.float32)

        # 2. Ventral Paws: horizontal mirror reflection + bilateral swap + static affine
        for target_bp, (_, orig_swap_bp) in PAW_MAP.items():
            raw_bx = df_bot[(bot_scorer, orig_swap_bp, "x")].iloc[bot_idx].to_numpy(dtype=np.float32)
            raw_by = df_bot[(bot_scorer, orig_swap_bp, "y")].iloc[bot_idx].to_numpy(dtype=np.float32)
            raw_bp = df_bot[(bot_scorer, orig_swap_bp, "likelihood")].iloc[bot_idx].to_numpy(dtype=np.float32)

            # Mirror reflection
            refl_bx = arena_width - raw_bx

            # Static affine projection into virtual TopCam arena space
            # [x_top, y_top] = [refl_bx, raw_by] @ A.T + b
            pts_bot = np.column_stack([refl_bx, raw_by])
            pts_top = pts_bot @ A.T + b

            arr_x = np.full(N_top, np.nan, dtype=np.float32)
            arr_y = np.full(N_top, np.nan, dtype=np.float32)
            arr_like = np.zeros(N_top, dtype=np.float32)

            arr_x[top_idx] = pts_top[:, 0]
            arr_y[top_idx] = pts_top[:, 1]
            arr_like[top_idx] = raw_bp

            fused_dict[(fused_scorer, target_bp, "x")] = arr_x
            fused_dict[(fused_scorer, target_bp, "y")] = arr_y
            fused_dict[(fused_scorer, target_bp, "likelihood")] = arr_like

        # 3. Optional Bottom Anchors for 23kp mode
        if keypoint_mode == "23kp":
            for orig_bp, target_bp in ANCHOR_MAP.items():
                raw_bx = df_bot[(bot_scorer, orig_bp, "x")].iloc[bot_idx].to_numpy(dtype=np.float32)
                raw_by = df_bot[(bot_scorer, orig_bp, "y")].iloc[bot_idx].to_numpy(dtype=np.float32)
                raw_bp = df_bot[(bot_scorer, orig_bp, "likelihood")].iloc[bot_idx].to_numpy(dtype=np.float32)

                refl_bx = arena_width - raw_bx
                pts_bot = np.column_stack([refl_bx, raw_by])
                pts_top = pts_bot @ A.T + b

                arr_x = np.full(N_top, np.nan, dtype=np.float32)
                arr_y = np.full(N_top, np.nan, dtype=np.float32)
                arr_like = np.zeros(N_top, dtype=np.float32)

                arr_x[top_idx] = pts_top[:, 0]
                arr_y[top_idx] = pts_top[:, 1]
                arr_like[top_idx] = raw_bp

                fused_dict[(fused_scorer, target_bp, "x")] = arr_x
                fused_dict[(fused_scorer, target_bp, "y")] = arr_y
                fused_dict[(fused_scorer, target_bp, "likelihood")] = arr_like

        columns = pd.MultiIndex.from_tuples(list(fused_dict.keys()), names=["scorer", "bodyparts", "coords"])
        df_fused = pd.DataFrame(fused_dict, index=np.arange(N_top))
        df_fused.columns = columns

        # Atomic publication
        with atomic_path(out_h5_path) as tmp_h5:
            df_fused.to_hdf(str(tmp_h5), key="df_with_missing", mode="w", format="fixed")

        write_manifest(
            out_h5_path,
            kind=f"fused_static_{keypoint_mode}",
            inputs=manifest_inputs,
            config=manifest_config,
            code_files=_FUSION_CODE_FILES,
            extra={
                "sid": sid,
                "arena": arena,
                "keypoint_mode": keypoint_mode,
                "frames": int(N_top),
                "matched_frames": int(len(top_idx)),
                "calibration_status": (calibration or {}).get("status"),
            },
        )

        return {
            "sid": sid,
            "status": "FUSED",
            "keypoint_mode": keypoint_mode,
            "frames": N_top,
            "matched_frames": len(top_idx),
            "time": time.time() - t0,
            "path": str(out_h5_path),
        }

    except Exception as e:
        return {
            "sid": sid,
            "status": "ERROR",
            "keypoint_mode": keypoint_mode,
            "error": str(e),
            "time": time.time() - t0,
        }
