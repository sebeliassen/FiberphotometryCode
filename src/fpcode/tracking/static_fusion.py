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

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..common.cache_manifest import (
    OK as _CACHE_OK,
)
from ..common.cache_manifest import (
    UNVERIFIED_NO_MANIFEST as _CACHE_UNVERIFIED,
)
from ..common.cache_manifest import (
    atomic_path,
    verify_manifest,
    write_manifest,
)
from ..common.schemas import (
    DEFAULT_ARENA_WIDTH,
    TOP_USE_BODYPARTS,
)
from .camera_correspondence import CameraMappingContext, get_session_calibration_details, get_session_mapping, mapping_fingerprint
from .clock_sync import get_session_ttl_offset
from .geometry import project_reflected_points
from fpcode.adapters.multiview_compatibility import (
    STATIC_TRANSFORMS_PATH, EPOCH_TRANSFORMS_PATH, LazyJsonMapping, legacy_session_epoch,
)

_CORE_DIR = Path(__file__).resolve().parent
STATIC_ARENA_TRANSFORMS = LazyJsonMapping(STATIC_TRANSFORMS_PATH)
EPOCH_ARENA_TRANSFORMS = LazyJsonMapping(EPOCH_TRANSFORMS_PATH, missing_ok=True)


def get_session_epoch(ts_str: str) -> str:
    """Documented epoch assignment based on experimental recording schedule."""
    return legacy_session_epoch(ts_str)


_FUSION_CODE_FILES = [
    _CORE_DIR / "static_fusion.py",
    _CORE_DIR / "geometry.py",
    _CORE_DIR.parent / "adapters" / "multiview_compatibility.py",
    _CORE_DIR / "camera_correspondence.py",
    _CORE_DIR / "clock_sync.py",
    _CORE_DIR.parent / "common" / "schemas.py",
    _CORE_DIR.parent / "contracts" / "pose.py",
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
    unmanifested_policy: str = "trust",
    *,
    transform_info=None,
    transform_path=None,
    transform_source_path=None,
    epoch=None,
    mapping_context=None,
    ttl_cache=None,
    ttl_cache_path=None,
    offset_info=None,
    camera_mapping=None,
    calibration_details=None,
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
    transform_info / transform_path : optional
        Explicit affine calibration, as resolved metadata or a JSON resource.
        An explicit epoch JSON requires an explicit ``epoch``. The historical
        no-argument resource/epoch policy remains a compatibility facade.
    transform_source_path : optional
        Source dependency for an already resolved ``transform_info``; it is
        fingerprinted in the manifest without loading calibration again.
    mapping_context / ttl_cache / ttl_cache_path / offset_info : optional
        Explicit timestamp lookup, scoped to those inputs.
    camera_mapping / calibration_details : optional
        Supplied integer correspondence with validated crystal provenance.
    """
    if keypoint_mode not in ("21kp", "23kp"):
        raise ValueError(f"Invalid keypoint_mode '{keypoint_mode}'. Must be '21kp' or '23kp'.")
    if unmanifested_policy not in ("trust", "rebuild"):
        raise ValueError("unmanifested_policy must be 'trust' or 'rebuild'")

    sid = pair_info["sid"]
    arena = pair_info.get("arena", "Arena1")
    top_h5_path = Path(pair_info["top_h5"])
    bot_h5_path = Path(pair_info["bot_h5"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5_path = out_dir / f"{sid}_tandem_{keypoint_mode}.h5"

    ts = pair_info.get("ts", sid[:19])
    if transform_info is not None and transform_path is not None:
        raise ValueError("Provide transform_info or transform_path, not both")
    if transform_source_path is not None and transform_info is None:
        raise ValueError("transform_source_path requires supplied transform_info")
    if transform_path is not None:
        used_transforms_path = Path(transform_path)
        with used_transforms_path.open() as handle:
            transforms = json.load(handle)
        if use_epochs:
            if epoch is None:
                raise ValueError("Explicit epoch transforms require an explicit epoch")
            transforms = transforms[epoch]
        transform_info = transforms if "matrix_2x2" in transforms else transforms[arena]
        resolved_epoch = epoch or "explicit_static"
    elif transform_info is not None:
        used_transforms_path = Path(transform_source_path) if transform_source_path is not None else None
        resolved_epoch = epoch or "explicit_static"
    elif use_epochs and EPOCH_ARENA_TRANSFORMS:
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
    if A.shape != (2, 2) or b.shape != (2,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Affine transform requires a finite 2x2 matrix and two-element translation")

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
    if mapping_context is not None and any(value is not None for value in (ttl_cache, ttl_cache_path, offset_info)):
        raise ValueError("Provide a mapping context or TTL inputs, not both")
    if sum(value is not None for value in (ttl_cache, ttl_cache_path, offset_info)) > 1:
        raise ValueError("Provide one TTL cache, cache path, or offset record")
    if mapping_context is None and any(value is not None for value in (ttl_cache, ttl_cache_path, offset_info)):
        mapping_context = (CameraMappingContext(offset_lookup=lambda session: offset_info) if offset_info is not None
                           else CameraMappingContext(ttl_cache=ttl_cache, ttl_cache_path=ttl_cache_path))
    if mapping_context is not None:
        manifest_config["camera_max_jitter_ms"] = float(mapping_context.max_jitter_ms)
    if camera_mapping is not None:
        camera_mapping = np.asarray(camera_mapping)
        if camera_mapping.ndim != 1 or not np.issubdtype(camera_mapping.dtype, np.integer):
            raise ValueError("camera_mapping must be a one-dimensional integer array")
        manifest_config["camera_mapping_sha256"] = mapping_fingerprint(camera_mapping)
        manifest_config["calibration_details"] = dict(calibration_details or {})
    manifest_inputs = [top_h5_path, bot_h5_path]
    if used_transforms_path is not None:
        manifest_inputs.append(used_transforms_path)
    if camera_mapping is None or mapping_context is not None:
        try:
            info = (mapping_context.offset_info(sid) if mapping_context is not None else get_session_ttl_offset(sid))
            manifest_inputs.extend(Path(path) for path in (info.get("top_csv"), info.get("bot_csv")) if path)
        except Exception:
            if mapping_context is not None:
                raise

    t0 = time.time()
    if not overwrite and out_h5_path.exists() and out_h5_path.stat().st_size > 500_000:
        cache_status, _ = verify_manifest(
            out_h5_path,
            kind=f"fused_static_{keypoint_mode}",
            inputs=manifest_inputs,
            config=manifest_config,
            code_files=_FUSION_CODE_FILES,
        )
        if cache_status == _CACHE_OK or (cache_status == _CACHE_UNVERIFIED and unmanifested_policy == "trust"):
            return {
                "sid": sid,
                "status": "CACHED",
                "cache_verification": cache_status,
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
        if camera_mapping is not None:
            mapping, calibration = camera_mapping, calibration_details
            if not (calibration or {}).get("is_valid_crystal", False):
                raise ValueError(f"Session {sid} supplied mapping is not valid crystal calibration")
        elif mapping_context is not None:
            mapping, calibration = mapping_context.resolve(sid, strict=True)
        else:
            mapping = get_session_mapping(sid, strict=True)
            calibration = get_session_calibration_details(sid)
        if mapping is None:
            raise ValueError(f"Strict Basler crystal mapping unavailable for {sid}.")

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
            fused_dict[(fused_scorer, bp, "likelihood")] = (
                df_top[(top_scorer, bp, "likelihood")].iloc[:N_top].to_numpy(dtype=np.float32)
            )

        # 2. Ventral Paws: horizontal mirror reflection + bilateral swap + static affine
        for target_bp, (_, orig_swap_bp) in PAW_MAP.items():
            raw_bx = df_bot[(bot_scorer, orig_swap_bp, "x")].iloc[bot_idx].to_numpy(dtype=np.float32)
            raw_by = df_bot[(bot_scorer, orig_swap_bp, "y")].iloc[bot_idx].to_numpy(dtype=np.float32)
            raw_bp = df_bot[(bot_scorer, orig_swap_bp, "likelihood")].iloc[bot_idx].to_numpy(dtype=np.float32)

            pts_top = project_reflected_points(np.column_stack([raw_bx, raw_by]), A, b,
                                               width=arena_width, dtype=np.float32)

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

                pts_top = project_reflected_points(np.column_stack([raw_bx, raw_by]), A, b,
                                                   width=arena_width, dtype=np.float32)

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


fuse_multiview_static_21kp = fuse_single_pair_static
