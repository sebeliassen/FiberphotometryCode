# session.py
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple # Added Tuple
import config as main_config

@dataclass
class Session:
    # ... (previous attributes) ...
    trial_folder: str; mouse_id: int; timestamp: str; arena: str
    left_region: str; right_region: str; genotype: str
    behavior_video_path: Optional[Path] = None
    pose_3d_video_path: Optional[Path] = None
    syllable_video_path: Optional[Path] = None
    photometry_csv_path: Optional[Path] = None
    center_timestamp_csv_path: Optional[Path] = None
    photometry_df: pd.DataFrame = field(default_factory=pd.DataFrame) # Raw loaded+merged
    frame_range_original: Optional[Tuple[int, int]] = None
    is_valid: bool = True

    # --- Add field for processed data ---
    filtered_photometry_segment_df: Optional[pd.DataFrame] = None

    # Derived properties
    @property
    def base_video_key(self) -> str:
        from config import CENTER_SUFFIX_MAP
        suffix = CENTER_SUFFIX_MAP.get(self.arena, "unknown")
        return f"Arena1234_CenterTopCamVid_{self.timestamp}_{suffix}"

    @property
    def output_video_filename(self) -> str:
        mode_suffix = "_dopamine_only" if main_config.DOPAMINE_ONLY_MODE else "_composite"
        preview_suffix = "_preview" if main_config.PREVIEW_MODE else ""
        return f"{self.base_video_key}{mode_suffix}{preview_suffix}.mp4"

    @property
    def num_output_frames(self) -> Optional[int]:
        """Calculates the number of frames based on mode and preview settings, using TARGET_FPS."""
        if not self.frame_range_original: return None
        start, end = self.frame_range_original
        total_original_frames = end - start + 1
        segment_duration_seconds = total_original_frames / main_config.BEHAVIOR_VIDEO_FPS
        # Always use TARGET_FPS for calculating output frames now
        total_output_frames_possible = int(segment_duration_seconds * main_config.TARGET_FPS)

        if not main_config.PREVIEW_MODE:
            return max(1, total_output_frames_possible) if total_output_frames_possible > 0 else 0
        else:
            preview_frames = int(main_config.PREVIEW_DURATION_SECONDS * main_config.TARGET_FPS)
            return max(1, min(preview_frames, total_output_frames_possible)) if total_output_frames_possible > 0 else 0