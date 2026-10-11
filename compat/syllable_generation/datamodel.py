from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Import pandas only for static type checking
if TYPE_CHECKING:
    import pandas as pd

# Soft dependency typing (so importing this module doesn't force pandas at import-time)
try:
    import pandas as pd
    DataFrame = pd.DataFrame
    Series = pd.Series
except Exception:  # pragma: no cover
    DataFrame = Any
    Series = Any


@dataclass(frozen=True)
class VideoRef:
    """
    Lightweight descriptor for a video (no frames loaded).
    """
    path: str
    role: str = "primary"          # e.g., 'primary', 'aux_left', 'aux_right'
    fps: Optional[float] = None
    n_frames: Optional[int] = None
    t0_ns: Optional[int] = None    # absolute start time if known (else relative 0)
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionRecord:
    """
    One session’s data with standardized, DataFrame-based modalities.

    Core tidy tables (each is a single, canonical DataFrame when present):
      - tracking_df       : per-frame pose (combine multiple cameras upstream if needed)
      - photometry_df     : long-form (t_ns, channel, value)
      - events_df         : long-form (t_ns and/or frame_ix + event info)

    Reference tables (DataFrames pointing to heavy assets/slices):
      - video_refs_df       : paths + roles/slices for video files
      - tracking_refs_df    : DLC H5/CSV paths (+ per-camera tokens)
      - photometry_refs_df  : combined photometry paths (+ slice windows)
      - event_refs_df       : event sources (+ slice windows)

    Timelines:
      - timelines["primary"] may hold a frame_ix -> t_ns mapping (Series/DataFrame).

    Extras:
      - extras[name] : any additional DataFrame or small object you want to attach.

    Column rules live in validate.py.
    """

    # Identity / context
    session_id: str
    project: Optional[str] = None
    session_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Core tidy tables
    tracking: Dict[str, "pd.DataFrame"] = field(default_factory=dict)
    photometry: Dict[str, "pd.DataFrame"] = field(default_factory=dict)
    events: Dict[str, "pd.DataFrame"] = field(default_factory=dict)
    syllables: Dict[str, "pd.DataFrame"] = field(default_factory=dict)

    # Reference tables (DFs with paths/slices/roles)
    video_refs_df: Optional["pd.DataFrame"] = None
    tracking_refs_df: Optional["pd.DataFrame"] = None
    photometry_refs_df: Optional["pd.DataFrame"] = None
    event_refs_df: Optional["pd.DataFrame"] = None

    # Optional explicit video objects (handy for simple cases)
    video_refs: List[VideoRef] = field(default_factory=list)

    # Timelines (e.g., {"primary": Series frame_ix -> t_ns})
    timelines: Dict[str, Any] = field(default_factory=dict)

    # Extras (free-form; may include additional DataFrames)
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---- minimal API ----

    def require_tracking(self) -> None:
        """
        tracking considered present if:
        - any DataFrame is present in self.tracking, OR
        - tracking_refs_df exists.
        """
        has_any_df = any(hasattr(v, "shape") for v in self.tracking.values())
        if not (has_any_df or self.tracking_refs_df is not None):
            raise ValueError("tracking is required but missing (no tracking tables and no tracking_refs_df).")


    def set_timeline(self, name: str, mapping: Any) -> None:
        self.timelines[name] = mapping

    def get_timeline(self, name: str) -> Optional[Any]:
        return self.timelines.get(name)

    def add_video_ref(self, ref: VideoRef) -> None:
        self.video_refs.append(ref)

    def primary_video(self) -> Optional[VideoRef]:
        for v in self.video_refs:
            if v.role == "primary":
                return v
        return self.video_refs[0] if self.video_refs else None

    def primary_fps(self) -> Optional[float]:
        v = self.primary_video()
        return v.fps if v else None

    def add_extra(self, name: str, obj: Any) -> None:
        self.extras[name] = obj

    def get_table_map(self) -> Dict[str, "pd.DataFrame"]:
        out: Dict[str, "pd.DataFrame"] = {}
        for k, df in self.tracking.items():
            if hasattr(df, "shape"): out[f"tracking:{k}"] = df
        for k, df in self.photometry.items():
            if hasattr(df, "shape"): out[f"photometry:{k}"] = df
        for k, df in self.events.items():
            if hasattr(df, "shape"): out[f"events:{k}"] = df
        for k, df in self.syllables.items():
            if hasattr(df, "shape"): out[f"syllables:{k}"] = df
        out.update({f"extra:{k}": v for k, v in self.extras.items() if hasattr(v, "shape")})
        return out

    def get_refs_map(self) -> Dict[str, "pd.DataFrame"]:
        """Dict view of present reference tables."""
        out: Dict[str, "pd.DataFrame"] = {}
        if self.video_refs_df is not None:      out["videos"] = self.video_refs_df
        if self.tracking_refs_df is not None:   out["tracking_refs"] = self.tracking_refs_df
        if self.photometry_refs_df is not None: out["photometry_refs"] = self.photometry_refs_df
        if self.event_refs_df is not None:      out["event_refs"] = self.event_refs_df
        return out

    def summary(self) -> dict:
        def _shape(x): return list(x.shape) if hasattr(x, "shape") else None
        def _dict_shapes(d): return {k: _shape(v) for k, v in d.items() if hasattr(v, "shape")}
        return {
            "session_id": self.session_id,
            "project": self.project,
            "session_type": self.session_type,
            "tables": self.get_table_map(),
            "refs": {k: _shape(v) for k, v in self.get_refs_map().items()},
            "modalities": {
                "tracking":   _dict_shapes(self.tracking),
                "photometry": _dict_shapes(self.photometry),
                "events":     _dict_shapes(self.events),
                "syllables":  _dict_shapes(self.syllables),         # <-- NEW
            },
            "n_video_refs": len(self.video_refs),
            "timelines": list(self.timelines.keys()),
            "meta_keys": sorted(self.metadata.keys()),
        }


