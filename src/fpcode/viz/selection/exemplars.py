"""Visualization selection / exemplars; historical V2 behavior preserved."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple
import math


def assert_unique_bout_ids(bouts: Iterable[Dict[str, Any]], context: str = "") -> None:
    ids = [b["bout_id"] for b in bouts if not b.get("is_blank", False)]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise AssertionError(f"Duplicate bout IDs in final selection{(' (' + context + ')') if context else ''}: {dup}")


def selection_ids(manifest: Dict[str, Any], syllable: int) -> List[str]:
    """Ordered non-blank bout IDs of the finalized manifest (the only source of truth)."""
    entry = manifest["syllables"][str(int(syllable))]
    return [b["bout_id"] for b in entry["bouts"] if not b.get("is_blank", False)]


def resolve_display_bouts(
    manifest_bouts: Sequence[Dict[str, Any]],
    target_validator: Callable[[Dict[str, Any]], Tuple[bool, str]],
    log: Callable[[str], None] = lambda msg: None,
) -> List[Dict[str, Any]]:
    """
    Turn the finalized manifest list into display tiles WITHOUT choosing
    different executions.  A bout whose target execution unexpectedly fails
    validation becomes an explicitly flagged blank that keeps the original
    bout ID; it is never replaced by another bout, and no flag (exemplar,
    medoid) can transfer to anything else.
    """
    tiles: List[Dict[str, Any]] = []
    for pos, b in enumerate(manifest_bouts):
        if b.get("is_blank", False):
            tile = dict(b)
            tile["tile_index"] = pos
            tile["render_status"] = "blank_in_manifest"
            tiles.append(tile)
            continue
        ok, reason = target_validator(b)
        if ok:
            tile = dict(b)
            tile["tile_index"] = pos
            tile["is_blank"] = False
            tile["flagged_invalid"] = False
            tile["render_status"] = "rendered"
            tiles.append(tile)
        else:
            log(f"  [FLAGGED BLANK] tile {pos} bout {b['bout_id']} failed target validation: {reason}")
            tiles.append(
                {
                    "is_blank": True,
                    "flagged_invalid": True,
                    "bout_id": b["bout_id"],
                    "invalid_reason": reason,
                    "is_anchor": bool(b.get("is_anchor", False)),
                    "anchor_type": b.get("anchor_type"),
                    "tile_index": pos,
                    "render_status": "flagged_invalid",
                }
            )
    return tiles


def assert_rendered_equals_manifest(tiles: Sequence[Dict[str, Any]], manifest_bouts: Sequence[Dict[str, Any]]) -> None:
    """The displayed ID sequence must equal the finalized manifest sequence, in order."""
    got = [t.get("bout_id") for t in tiles if t.get("render_status") in ("rendered", "flagged_invalid")]
    want = [b["bout_id"] for b in manifest_bouts if not b.get("is_blank", False)]
    if got != want:
        raise AssertionError(f"Rendered IDs differ from finalized manifest.\n  rendered: {got}\n  manifest: {want}")
    if len(set(got)) != len(got):
        raise AssertionError(f"Duplicate rendered bout IDs: {got}")
    assert_unique_bout_ids(manifest_bouts, "manifest")


def grid_layout(G: int) -> Tuple[int, int]:
    """(rows, cols) for ``G`` displayed slots.  12 -> 3x4, 24 -> 4x6 (the KPMS default)."""
    known = {12: (3, 4), 24: (4, 6)}
    if G in known:
        return known[G]
    cols = int(math.ceil(math.sqrt(G * 4.0 / 3.0)))
    rows = int(math.ceil(G / cols))
    return rows, cols
