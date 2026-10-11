from __future__ import annotations
from typing import List
from fpcode.contracts.session_record import SessionRecord
from ..context import BuildContext

def attach_events(ctx: BuildContext, sessions: List[SessionRecord]) -> List[SessionRecord]:
    # placeholder no-op
    return sessions
