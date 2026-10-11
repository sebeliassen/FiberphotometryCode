# loaders/jonathan/build.py
# NEW: import project enrich and locate its overrides dir
from pathlib import Path

from fpcode.contracts.session_record import SessionRecord
from syllable_generation.projects.jonathan_openfield_example import enrich as project_enrich

from .context import BuildContext
from .steps import fiberphotometry, tracking, video


def _project_overrides_dir() -> Path:
    # points to .../projects/jonathan_openfield_example/overrides
    return Path(project_enrich.__file__).resolve().parent / "overrides"

def _enrich_pre(ctx: BuildContext, sessions: list[SessionRecord]) -> list[SessionRecord]:
    return project_enrich.enrich(ctx, sessions, overrides_dir=_project_overrides_dir())

def _enrich_post(ctx: BuildContext, sessions: list[SessionRecord]) -> list[SessionRecord]:
    # run again after effective_indices are known to populate fiber_regions_by_index
    return project_enrich.enrich(ctx, sessions, overrides_dir=_project_overrides_dir())

DEFAULT_STEPS = (
    tracking.discover_sessions,
    video.attach_video_refs,
    fiberphotometry.attach_photometry_refs,
    _enrich_pre,                         # <- seeds signal_0 / signal_1
    fiberphotometry.realize_photometry_columns,
    _enrich_post,                        # <- now fills fiber_regions_by_index
    fiberphotometry.split_photometry,
    fiberphotometry.compute_dff_zscore,
)

def build(ctx: BuildContext, steps=None) -> list[SessionRecord]:
    steps = list(steps or DEFAULT_STEPS)
    sessions: list[SessionRecord] = []
    for step in steps:
        sessions = step(ctx, sessions)
    return sessions
