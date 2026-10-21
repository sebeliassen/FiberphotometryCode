"""Independent boundary expectations for shared behavioural-unit records."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fpcode.contracts import FrameSpan, SessionKey, UnitOccurrence, UnitStream, UnitType


def _stream(**changes):
    values = {
        "session": SessionKey("recording", subject="mouse"),
        "level": "episode",
        "model_id": "frozen-model-v1",
        "clock": "native-video",
        "fps": 60,
        "valid_intervals": (FrameSpan(0, 60), FrameSpan(120, 180)),
        "catalog": (UnitType("A", "left turn then forward movement"), UnitType("B")),
        "occurrences": (UnitOccurrence("event-1", "A", FrameSpan(5, 20)),),
    }
    values.update(changes)
    return UnitStream(**values)


@pytest.mark.parametrize("start,stop", [(-1, 4), (4, 4), (5, 4), (0.5, 4), (0, 4.0), (True, 4)])
def test_frame_spans_require_nonempty_native_integer_intervals(start, stop):
    with pytest.raises(ValueError):
        FrameSpan(start, stop)


def test_frame_spans_accept_numpy_integers_and_are_immutable():
    span = FrameSpan(np.int64(0), np.int64(60))
    assert (span.start, span.stop) == (0, 60)
    assert type(span.start) is int
    with pytest.raises(FrozenInstanceError):
        span.stop = 120


def test_stream_owns_tuple_inputs_and_preserves_overlapping_detector_calls():
    valid, catalog = [FrameSpan(0, 60)], [UnitType("A")]
    calls = [UnitOccurrence("second", "A", FrameSpan(10, 30)), UnitOccurrence("first", "A", FrameSpan(0, 20))]
    stream = _stream(valid_intervals=valid, catalog=catalog, occurrences=calls)
    valid.clear()
    catalog.clear()
    calls.clear()
    assert stream.valid_intervals == (FrameSpan(0, 60),)
    assert len(stream.catalog) == 1
    assert len(stream.occurrences) == 2
    assert [event.occurrence_id for event in stream.occurrences] == ["second", "first"]
    assert (stream.occurrences[0].interval.start, stream.occurrences[1].interval.stop) == (10, 20)
    with pytest.raises(FrozenInstanceError):
        stream.clock = "another-clock"


@pytest.mark.parametrize("spans", [
    (FrameSpan(10, 30), FrameSpan(0, 10)),
    (FrameSpan(0, 30), FrameSpan(20, 40)),
])
def test_validity_rejects_out_of_order_or_overlapping_spans(spans):
    with pytest.raises(ValueError, match="ordered and nonoverlapping"):
        _stream(valid_intervals=spans, occurrences=())


@pytest.mark.parametrize("span", [FrameSpan(55, 125), FrameSpan(60, 120), FrameSpan(180, 181)])
def test_occurrences_cannot_cross_or_occupy_unobserved_time(span):
    with pytest.raises(ValueError, match="one valid interval"):
        _stream(occurrences=(UnitOccurrence("event", "A", span),))


def test_half_open_endpoints_allow_exactly_valid_spans_without_bridging_boundaries():
    calls = (UnitOccurrence("first", "A", FrameSpan(0, 60)), UnitOccurrence("last", "B", FrameSpan(120, 180)))
    stream = _stream(occurrences=calls)
    assert stream.occurrences == calls
    with pytest.raises(ValueError, match="one valid interval"):
        _stream(valid_intervals=(FrameSpan(0, 30), FrameSpan(30, 60)),
                occurrences=(UnitOccurrence("boundary", "A", FrameSpan(20, 40)),))


def test_empty_detection_stream_and_empty_observation_stream_are_distinct_valid_records():
    no_calls = _stream(occurrences=())
    no_observations = _stream(valid_intervals=(), catalog=(), occurrences=())
    assert no_calls.valid_intervals and no_calls.catalog
    assert not no_observations.valid_intervals and not no_observations.catalog
    with pytest.raises(ValueError, match="one valid interval"):
        _stream(valid_intervals=())


def test_catalogue_and_occurrence_identity_are_checked_independently_of_descriptions():
    with pytest.raises(ValueError, match="type identities must be unique"):
        _stream(catalog=(UnitType("A", "one"), UnitType("A", "another")))
    with pytest.raises(ValueError, match="occurrence identities must be unique"):
        _stream(occurrences=(UnitOccurrence("same", "A", FrameSpan(0, 2)),
                             UnitOccurrence("same", "B", FrameSpan(2, 4))))
    with pytest.raises(ValueError, match="reference the stream catalogue"):
        _stream(occurrences=(UnitOccurrence("unknown", "C", FrameSpan(0, 2)),))


@pytest.mark.parametrize("changes", [
    {"session": "recording"}, {"level": "L3"}, {"model_id": " "}, {"clock": ""},
    {"fps": 0}, {"fps": -60}, {"fps": np.inf}, {"fps": np.nan}, {"fps": True},
])
def test_stream_requires_explicit_identity_abstraction_clock_and_fps(changes):
    with pytest.raises(ValueError):
        _stream(**changes)


def test_abstraction_and_detector_identity_do_not_change_occurrence_shape():
    stream = _stream()
    alternative = replace(stream, level="assembly", model_id="another-method")
    assert alternative.occurrences == stream.occurrences
    assert alternative.session == stream.session
    assert alternative.fps == 60.0


@pytest.mark.parametrize("score,kind", [
    (np.nan, "model_score"), (np.inf, "model_score"), (True, "probability"),
    (0.5, None), (None, "posterior"), (0.5, "confidence"), (-0.1, "probability"), (1.1, "posterior"),
])
def test_score_semantics_are_explicit_and_do_not_invent_confidence(score, kind):
    with pytest.raises(ValueError):
        UnitOccurrence("event", "A", FrameSpan(0, 2), score=score, score_kind=kind)


def test_detector_scores_can_be_unbounded_while_probabilities_cannot():
    for score in (-100.0, 100.0):
        event = UnitOccurrence("event", "A", FrameSpan(0, 2), score=score, score_kind="model_score")
        assert event.score == score
    for score in (0.0, 1.0):
        event = UnitOccurrence("event", "A", FrameSpan(0, 2), score=score, score_kind="posterior")
        assert event.score_kind == "posterior"


@pytest.mark.parametrize("record", [
    lambda: UnitType(""), lambda: UnitType("A", None),
    lambda: UnitOccurrence(" ", "A", FrameSpan(0, 2)),
    lambda: UnitOccurrence("event", "A", (0, 2)),
])
def test_nested_records_reject_ambiguous_identity_and_shape(record):
    with pytest.raises(ValueError):
        record()
