"""Native timing and group-blind example-selection expectations."""

import numpy as np
import pytest

from studies.sequence_viz.render_generator_screening import choose_examples, sample_indexes


def test_playback_uses_native_time_and_keeps_half_open_boundaries():
    # An 80Hz source with a60fps container must retain a one-second call.
    clock = np.arange(401)/80
    indexes, relative = sample_indexes(clock, 80, 160, fps=20, context=.5)
    assert len(indexes)==40
    np.testing.assert_array_equal(indexes, np.arange(40, 200, 4))
    np.testing.assert_allclose(relative, np.arange(40)/20-.5)
    assert np.sum((indexes>=80)&(indexes<160))==20
    # Timestamp duplicates are allowed; lookup skips the earlier co-timed frame.
    indexes, _ = sample_indexes([0., .1, .1, .2, .3, .4], 2, 4, fps=10, context=0)
    np.testing.assert_array_equal(indexes, [2,3])
    with pytest.raises(ValueError, match="endpoint"):
        sample_indexes(clock, 390, 401)


def test_examples_preserve_ordinary_calls_and_animal_diversity_without_scores():
    def call(token, mouse, recording, start, duration, parents):
        return dict(type_id=token, subject=mouse, recording=recording, start_frame=start,
                    stop_frame=start+int(duration*10), duration_s=duration, parent_counts=parents,
                    fps=10, score=1000 if duration==20 else 0)
    calls=[call("a","1","r1",10,2,{"X":2}),call("a","1","r1",50,20,{"X":2}),
           call("a","2","r2",10,2.2,{"X":2}),call("a","1","r1",300,2.1,{"X":2}),
           call("a","1","r1",700,1.9,{"X":2}),
           call("b","1","r1",100,2.1,{"X":1,"Y":1}),call("b","1","r1",400,2.1,{"Z":2})]
    result=next(r for r in choose_examples(calls) if r["type_id"]=="a")
    assert result["panels"][0]["duration_s"]==2.1
    assert result["panels"][1]["subject"]=="2"
    assert result["second_different_animal"]
    assert result["panels"][2]["start_frame"]==100
    assert result["panels"][2]["type_id"]=="b"
    singleton=choose_examples([calls[0]])[0]
    assert singleton["panels"][1:] == [None,None]


@pytest.mark.parametrize("start,length", [(61, 169), (17003, 229), (200001, 731)])
def test_exact_sixty_hz_native_frames_have_no_float_floor_repeats_or_phantom_tail(start, length):
    clock = np.arange(start + length + 100) / 60
    indexes, relative = sample_indexes(clock, start, start + length, fps=60, context=.5)
    np.testing.assert_array_equal(indexes, np.arange(start - 30, start + length + 30))
    np.testing.assert_allclose(relative, np.arange(length + 60) / 60 - .5, atol=1e-12)


def test_irregular_clock_keeps_genuine_hold_before_future_frame():
    indexes, _ = sample_indexes([0., .101, .201, .303, .401, .502], 1, 4, fps=10, context=0)
    np.testing.assert_array_equal(indexes, [1, 2, 2])


@pytest.mark.parametrize("fps,context", [(0, .5), (float("inf"), .5), (60, -.1)])
def test_invalid_playback_settings_are_rejected(fps, context):
    with pytest.raises(ValueError, match="fps"):
        sample_indexes([0, 1, 2], 0, 1, fps=fps, context=context)
