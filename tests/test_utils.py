import pytest
import numpy as np
from types import SimpleNamespace

from fiberphotometry.utils import (
    find_start_end_idxs,
    find_session_by_trial_mouse_id,
    count_session_events,
    concat_dicts,
    mouse_br_events_count,
)


def test_find_start_end_idxs():
    # For 'hit' with fps=20: start=-2.5*20+200=150, end=5*20+200=300
    start, end = find_start_end_idxs('hit', fps=20)
    assert start == 150
    assert end == 300


def test_find_session_by_trial_mouse_id():
    class Dummy:
        def __init__(self, trial, mouse):
            self.trial_id = f"T{trial}_foo"
            self.mouse_id = str(mouse)

    sessions = [Dummy(1, 1), Dummy(2, 2)]
    assert find_session_by_trial_mouse_id(sessions, 1, 1) is sessions[0]
    assert find_session_by_trial_mouse_id(sessions, 3, 3) is None
    with pytest.raises(TypeError):
        find_session_by_trial_mouse_id(sessions, '1', 1)
    with pytest.raises(TypeError):
        find_session_by_trial_mouse_id(sessions, 1, '1')


def test_count_session_events():
    class Dummy:
        def __init__(self, data):
            self.event_idxs_container = SimpleNamespace(data=data)

    sessions = [Dummy({'hit': [1, 2]}), Dummy({'hit': [3], 'miss': []})]
    assert count_session_events(sessions, 'hit') == 3
    assert count_session_events(sessions, 'miss') == 0


def test_concat_dicts():
    dicts = [{'a': [1, 2]}, {'a': [3], 'b': [4]}]
    res = concat_dicts(dicts)
    assert isinstance(res['a'], np.ndarray)
    assert np.array_equal(res['a'], np.array([1, 2, 3]))
    assert isinstance(res['b'], np.ndarray)
    assert np.array_equal(res['b'], np.array([4]))


def test_mouse_br_events_count():
    class DummySession:
        def __init__(self, info):
            self.signal_info = info

    class DummyMouse:
        def __init__(self, sessions):
            self.sessions = sessions

    s1 = DummySession({('LH', 'hit'): {'signal_matrix': np.zeros((2, 3))}})
    s2 = DummySession({('LH', 'hit'): {'signal_matrix': np.zeros((5, 1))}})
    s3 = DummySession({('LH', 'miss'): {'signal_matrix': np.zeros((10, 1))}})
    mouse = DummyMouse([s1, s2, s3])
    assert mouse_br_events_count(mouse, 'LH', 'hit') == 7
    assert mouse_br_events_count(mouse, 'LH', 'miss') == 10
    # no matching key
    assert mouse_br_events_count(mouse, 'LH', 'unknown') == 0