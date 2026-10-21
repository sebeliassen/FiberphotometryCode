"""Native-video preview clock/seek checks without empirical video files."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fpcode.viz.event_clips import read_native_frame


def test_native_frame_preview_rejects_rate_bounds_and_wrong_seek(monkeypatch):
    class Capture:
        def __init__(self, path):
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            return {1: rate[0], 2: 20, 3: self.position}[prop]

        def set(self, prop, position):
            self.position = position + offset[0]
            return True

        def read(self):
            image = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, image

        def release(self):
            self.released = True

    captures, rate, offset = [], [60.], [0]
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=Capture,
        CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_FRAMES=3))
    image = read_native_frame('selected.avi', frame_index=5, fps=60.)
    assert np.all(image == 5)
    assert captures[-1].released
    rate[0] = 30.
    with pytest.raises(ValueError, match='rates'):
        read_native_frame('selected.avi', frame_index=5, fps=60.)
    rate[0] = 60.
    with pytest.raises(ValueError, match='outside'):
        read_native_frame('selected.avi', frame_index=20, fps=60.)
    offset[0] = 1
    with pytest.raises(ValueError, match='position'):
        read_native_frame('selected.avi', frame_index=5, fps=60.)
    assert all(capture.released for capture in captures)


def test_event_clip_rejects_wrong_native_seek_before_writing(monkeypatch, tmp_path):
    from fpcode.viz.event_clips import render_event_clip

    class Capture:
        def __init__(self, path):
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            return {1: 60., 2: 100, 3: self.position}[prop]

        def set(self, prop, position):
            self.position = position + 1
            return True

        def read(self):
            self.position += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    captures = []
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=Capture,
        CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_FRAMES=3))
    with pytest.raises(ValueError, match='position'):
        render_event_clip('selected.avi', tmp_path / 'clip.mp4', start_frame=5,
                          end_frame_exclusive=10, fps=60., caption='review',
                          pre_seconds=0, post_seconds=0, active_label='REVIEW')
    assert captures[0].released
    assert not (tmp_path / 'clip.mp4').exists()
