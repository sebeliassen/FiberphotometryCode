from __future__ import annotations
from typing import List
from syllable_generation.datamodel import SessionRecord
from ..context import BuildContext

def attach_events(ctx: BuildContext, sessions: List[SessionRecord]) -> List[SessionRecord]:
    # placeholder no-op
    return sessions
