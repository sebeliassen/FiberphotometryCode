import pandas as pd

# --- core helpers ---

def _collect_cam_paths_from_dfs(dfs):
    """Return {'center_cam': [...], 'side_cam': [...]} from a list of refs DFs."""
    df = pd.concat(dfs, ignore_index=True)

    if "cam_slot" in df.columns:
        center_mask = df["cam_slot"].eq("center_cam")
        side_mask   = df["cam_slot"].isin(["left_cam", "right_cam"])
    else:
        key = df["camera_key"].astype(str)
        center_mask = key.str.startswith("center_cam:")
        side_mask   = key.str.startswith(("left_cam:", "right_cam:"))

    return {
        "center_cam": df.loc[center_mask, "path"].tolist(),
        "side_cam":   df.loc[side_mask,   "path"].tolist(),
    }

# --- subroutines ---

def _subroutine_jonathan(sessions, *, cam="center_cam"):
    """
    sessions: iterable with .tracking_refs_df and .video_refs_df
    returns: (keypoint_paths, video_paths) for requested cam
    """
    trk_dfs = [s.tracking_refs_df for s in sessions]
    vid_dfs = [s.video_refs_df    for s in sessions]

    trk = _collect_cam_paths_from_dfs(trk_dfs)[cam]
    vid = _collect_cam_paths_from_dfs(vid_dfs)[cam]

    if len(trk) != len(vid):
        raise ValueError(f"Tracking vs video count mismatch: {len(trk)} != {len(vid)} for cam={cam}")
    return trk, vid

def _subroutine_soeren(sessions, *, cam="center_cam"):
    # Placeholder for your alternative path-building strategy
    raise NotImplementedError("subroutine 'soeren' not implemented yet")

# --- public wrapper ---

def ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam"):
    """
    Dispatch to a path-extraction subroutine and return (keypoint_paths, video_paths).

    subroutine: one of {'jonathan', 'soeren', ...}
    cam: 'center_cam' or 'side_cam'
    """
    registry = {
        "jonathan": _subroutine_jonathan,
        "soeren":   _subroutine_soeren,
    }
    if subroutine not in registry:
        raise ValueError(f"Unknown subroutine {subroutine!r}. Available: {sorted(registry)}")
    return registry[subroutine](sessions, cam=cam)
