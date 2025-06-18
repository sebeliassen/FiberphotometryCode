# oft_session_loading.py

from pathlib import Path
import re
from typing import Callable
import warnings
import pandas as pd

from fiberphotometry.data.session_loading import populate_session
from fiberphotometry.config import (
    OFT_H5_GLOB,

    OFT_TRACKDATA_GLOB,
    OFT_ALLTIMES_GLOB,
    OFT_BONSAITS_GLOB,

    OFT_CAMERA_TIMES_REGEX,
    OFT_H5_EXACT_CORNER_REGEX,
    OFT_H5_UPDOWN_REGEX,

    OFT_VIDEO_GLOB,
    OFT_VIDEO_EXACT_CORNER_REGEX,
    OFT_VIDEO_UPDOWN_REGEX,

    CHAMBER_CAMERAS
)


def _find_camera_timestamp_files(trial_dir: Path) -> list[Path]:
    """
    Return all files in trial_dir that match either:
      - '*_TrackData*.csv'
      - '*_AllTimestampsCombined*.csv'
      - '*_BonsaiTimestamp_AfterStart*.csv'
    """
    return (
        list(trial_dir.glob(OFT_TRACKDATA_GLOB))
        + list(trial_dir.glob(OFT_ALLTIMES_GLOB))
        + list(trial_dir.glob(OFT_BONSAITS_GLOB))
    )


def _parse_camera_timestamp_filename(csv_path: Path) -> tuple[str, str]:
    """
    Given a filename like:
      - CenterTopCam_TrackData2024-12-13T12_20_06.csv
      - LeftTopCam_AllTimestampsCombined2024-12-13T12_20_06.csv
      - RightTopCam_BonsaiTimestamp_AfterStart2024-12-13T12_20_06.csv

    Return (camera_name, timestamp_str) as ("CenterTopCam", "2024-12-13T12_20_06"), etc.
    """
    fname = csv_path.name
    m = re.match(OFT_CAMERA_TIMES_REGEX, fname)
    if not m:
        raise ValueError(f"Cannot parse camera/timestamp from '{fname}'")
    return m.group(1), m.group(2)


def _populate_oft_sync_dataframes(session) -> None:
    """
    Finds, loads, and attaches the correct camera timestamp CSVs for an OFT session
    directly into the session.dfs DataContainer.
    """
    trial_dir = Path(session.trial_dir)
    # This helper correctly finds all potential timestamp files
    all_ts_files = _find_camera_timestamp_files(trial_dir)
    
    # We are primarily interested in the 'AllTimestampsCombined' files for syncing
    sync_csv_paths = [p for p in all_ts_files if 'AllTimestampsCombined' in p.name]

    if not sync_csv_paths:
        warnings.warn(f"No 'AllTimestampsCombined' files found for session {session.trial_id}")
        return

    # Get the list of cameras that are valid for this session's chamber position
    allowed_cameras = CHAMBER_CAMERAS.get(session.chamber_position)
    if not allowed_cameras:
        warnings.warn(f"No camera mapping found in CHAMBER_CAMERAS for position: {session.chamber_position}")
        return

    # 1. Sort the paths to ensure a consistent order (e.g., sync_0 is always the same camera)
    sync_csv_paths.sort()
    
    # 2. Initialize a counter for the generic key
    sync_counter = 0

    for path in sync_csv_paths:
        try:
            cam_name, _ = _parse_camera_timestamp_filename(path)
            if cam_name in allowed_cameras:
                df = pd.read_csv(path)
                
                # 3. Add a new column to the DataFrame to store the camera name
                df['camera_name'] = cam_name
                
                # 4. Use the generic 'sync_n' key to add the data
                session.dfs.add_data(f'sync_{sync_counter}', df)
                
                print(f"Loaded sync data for session {session.trial_id}/{session.chamber_id} from: {path.name} as sync_{sync_counter}")
                
                # 5. Increment the counter for the next file
                sync_counter += 1

        except ValueError:
            continue
        except Exception as e:
            warnings.warn(f"Could not load or process sync file {path.name}: {e}")


def _process_media(
    session,
    root: Path,
    fpath_pattern: str,
    exact_corner_regex: str,
    updown_regex: str,
    prefix: str,
    column_name: str = "path",
    load_fn: Callable[[Path], pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generic camera-timestamp -> file matcher.
    - session: your Session object
    - root: Path to h5_root or video_root
    - fpath_pattern: glob for files (*.h5 or *.mp4)
    - exact_corner_regex: to pull out “top_left”, etc.
    - updown_regex: to pull out “upper”/“lower”
    - prefix: “tracking” vs. “video”
    - column_name: “h5_path” vs. “path”
    - load_fn: e.g. pd.read_hdf or None (for path-only)
    """
    # 1) timestamps
    csvs = _find_camera_timestamp_files(Path(session.trial_dir))
    if not csvs:
        return {}
    camera_ts = []
    for csvf in csvs:
        try:
            cam, ts = _parse_camera_timestamp_filename(csvf)
        except ValueError:
            continue
        camera_ts.append((cam, ts))
    allowed = CHAMBER_CAMERAS[session.chamber_position]
    camera_ts = [(c, t) for c, t in camera_ts if c in allowed]
    if not camera_ts:
        return {}

    # 2) glob
    all_files = list(root.glob(fpath_pattern))
    if not all_files:
        return {}
    # 3) match substrings
    matched = []
    for cam, ts in camera_ts:
        tag = f"{cam}Vid_{ts}"
        for p in all_files:
            if tag in p.name:
                matched.append((p, cam))

    # 4) corner-map & filter
    final = []
    for p, cam in matched:
        nm = p.name
        m = re.search(exact_corner_regex, nm, flags=re.IGNORECASE)
        if m:
            pos = m.group(1).lower()
        else:
            m2 = re.search(updown_regex, nm, flags=re.IGNORECASE)
            if not m2:
                continue
            ud = m2.group(1).lower()
            side = "left" if cam=="LeftTopCam" else "right"
            pos = f"{'top' if ud=='upper' else 'bottom'}_{side}"
        if pos == session.chamber_position:
            final.append(p)

    if not final:
        return {}

    # 5) dedupe & sort
    unique = {str(p): p for p in final}.values()
    unique = sorted(unique, key=lambda p: p.name)

    # 6) build output dict
    out = {}
    for i, p in enumerate(unique):
        key = f"{prefix}_{i}"
        
        # This part finds the camera name (e.g., "LeftTopCam") associated with the file path `p`
        cam = None
        for m_p, m_cam in matched:
            if m_p == p:
                cam = m_cam
                break

        # This 'if' block is executed for tracking and video files when data isn't fully loaded
        if load_fn is None:
            # --- THIS IS THE CRITICAL CHANGE ---

            # OLD version (only has one column):
            # out[key] = pd.DataFrame({column_name: [str(p)]})
            
            # NEW, CORRECTED version (has two columns):
            out[key] = pd.DataFrame({column_name: [str(p)], 'camera_name': [cam]})

        else:
            # This part handles loading the full H5 file, which is not what you're doing initially
            try:
                df = load_fn(p)
                df.attrs['camera_name'] = cam
                out[key] = df
            except Exception:
                warnings.warn(f"couldn’t load {p.name}")
    return out



def _process_tracking(fpath_pattern: str, session, h5_root: Path, load_h5: bool=False) -> dict:
    return _process_media(
        session=session,
        root=h5_root,
        fpath_pattern=fpath_pattern,
        exact_corner_regex=OFT_H5_EXACT_CORNER_REGEX,
        updown_regex=OFT_H5_UPDOWN_REGEX,
        prefix="tracking",
        column_name="h5_path",
        load_fn=(pd.read_hdf if load_h5 else None),
    )


def _process_videos(fpath_pattern: str, session, video_root: Path) -> dict:
    return _process_media(
        session=session,
        root=video_root,
        fpath_pattern=fpath_pattern,
        exact_corner_regex=OFT_VIDEO_EXACT_CORNER_REGEX,
        updown_regex=OFT_VIDEO_UPDOWN_REGEX,
        prefix="video",
        column_name="path",
        load_fn=None,
    )


def populate_oft_session(session, h5_root: str=None, video_root: str=None, load_h5: bool=False):
    """
    Populates a session object with photometry, sync, tracking, and video data.
    """
    # This first call populates the basic photometry data ('phot_415', 'phot_470')
    populate_session(session)
    
    # This new call finds and loads the corresponding sync CSVs into session.dfs
    _populate_oft_sync_dataframes(session)

    if h5_root:
        tracks = _process_tracking(
            fpath_pattern=OFT_H5_GLOB,
            session=session,
            h5_root=Path(h5_root),
            load_h5=load_h5,
        )
        for k, df in tracks.items():
            session.dfs.add_data(k, df)

    if video_root:
        videos = _process_videos(
            fpath_pattern=OFT_VIDEO_GLOB,
            session=session,
            video_root=Path(video_root),
        )
        for k, df in videos.items():
            session.dfs.add_data(k, df)


def populate_oft_containers(sessions, h5_root: str=None, video_root: str=None, load_h5: bool=False):
    """
    Runs the OFT population logic for a list of session objects.
    """
    for session in sessions:
        if session.session_type != 'oft':
            continue
        populate_oft_session(session, h5_root=h5_root, video_root=video_root, load_h5=load_h5)