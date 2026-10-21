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
