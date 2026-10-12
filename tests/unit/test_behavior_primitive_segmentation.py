"""Independent numerical tests for CentroidPrimitiveSegmenter."""

import numpy as np
import pytest

from fpcode.behavior.primitives.segmentation import CentroidPrimitiveSegmenter
from fpcode.contracts import (
    FrameSpan,
    SessionKey,
    SignalMetadata,
    SignalSeries,
    TimeAxis,
)

SESSION = SessionKey("rec_seg", "mouse_seg")


def _make_features(data, valid=None, names=("f1", "f2")):
    t = TimeAxis(np.arange(len(data)), unit="frame", clock="video")
    meta = tuple(SignalMetadata(name=n) for n in names)
    return SignalSeries(time=t, values=data, signals=meta, valid=valid)


def test_separable_synthetic_features_fit_and_assign():
    # Two well-separated clusters: near (0, 0) and near (10, 10)
    c1 = np.array([[0.1, 0.2], [0.0, -0.1], [-0.1, 0.0], [0.2, 0.1]])
    c2 = np.array([[10.1, 9.9], [10.0, 10.1], [9.9, 10.0], [10.2, 10.2]])
    data = np.vstack([c1, c2])
    feats = _make_features(data)

    model = CentroidPrimitiveSegmenter.fit(feats, k=2, seed=42)
    assert model.k == 2
    assert len(model.catalog) == 2

    labels = model.predict_labels(feats)
    # First 4 must share one label, last 4 must share another label
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_held_out_outlier_does_not_alter_frozen_model():
    # Training data
    train_data = np.array([[1.0, 1.0], [1.1, 0.9], [5.0, 5.0], [5.1, 4.9]])
    train_feats = _make_features(train_data)
    model = CentroidPrimitiveSegmenter.fit(train_feats, k=2, seed=42)

    centers_before = model.centers.copy()
    mean_before = model.scaler_mean.copy()

    # Held out data with extreme outlier (1000, 1000)
    test_data = np.array([[1.0, 1.0], [1000.0, 1000.0]])
    test_feats = _make_features(test_data)
    test_labels = model.predict_labels(test_feats)

    # Centroids and scaling parameters in model remain exactly unchanged
    assert np.array_equal(model.centers, centers_before)
    assert np.array_equal(model.scaler_mean, mean_before)
    assert test_labels[0] != -1


def test_schema_mismatch_rejection():
    train_data = np.array([[1.0, 1.0], [2.0, 2.0]])
    train_feats = _make_features(train_data, names=("speed", "heading"))
    model = CentroidPrimitiveSegmenter.fit(train_feats, k=2, seed=42)

    # Missing heading
    bad_feats = _make_features(train_data, names=("speed", "other"))
    with pytest.raises(ValueError, match="required feature heading missing"):
        model.predict_labels(bad_feats)


def test_gaps_and_unsupported_rows_receive_negative_one_and_break_runs():
    # 5 frames: [0, 1, 2, 3, 4]. Frame 2 is invalid!
    data = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.1, 0.1], [0.0, 0.1]])
    valid = np.array([[True, True], [True, True], [False, True], [True, True], [True, True]])
    feats = _make_features(data, valid=valid)

    model = CentroidPrimitiveSegmenter.fit(feats, k=2, seed=42)
    labels = model.predict_labels(feats)
    assert labels[2] == -1

    stream = model.segment_stream(feats, session=SESSION, fps=10.0)
    # The invalid frame 2 must split the stream into two valid intervals [0, 2) and [3, 5)
    assert stream.valid_intervals == (FrameSpan(0, 2), FrameSpan(3, 5))


def test_physical_time_mapping_requires_origin_for_seconds_ms_ns():
    from dataclasses import replace
    times = np.array([10.,10.1,11.])
    raw = SignalSeries(TimeAxis(np.arange(3),'frame','video'),np.ones(3),(SignalMetadata('f','px/s'),))
    model = CentroidPrimitiveSegmenter.fit(raw,k=1)
    for unit, scale in (('s',1.),('ms',1000.),('ns',1e9)):
        series = replace(raw,time=TimeAxis(times*scale,unit,'video'))
        with pytest.raises(ValueError,match='explicit native mapping'):
            model.segment_stream(series,session=SESSION,fps=10.)
        expected = (FrameSpan(500,502),FrameSpan(510,511))
        mapped = model.segment_stream(series,session=SESSION,fps=10.,physical_time_origin_s=10.,native_frame_origin=500)
        explicit = model.segment_stream(series,session=SESSION,fps=10.,native_frames=np.array([500,501,510]))
        assert mapped.valid_intervals == explicit.valid_intervals == expected
    shifted = replace(raw,time=TimeAxis(times+.01,'s','video'))
    with pytest.raises(ValueError,match='fractional native'):
        model.segment_stream(shifted,session=SESSION,fps=10.,physical_time_origin_s=10.)


def test_primitive_budget_failure_and_constant_one_channel_schema():
    from fpcode.contracts import SignalMetadata, SignalSeries
    constant = SignalSeries(TimeAxis(np.arange(4),'frame','video'),np.ones(4),(SignalMetadata('f','px/s'),))
    model = CentroidPrimitiveSegmenter.fit(constant,k=2,batch_size=1)
    assert np.isfinite(model.centers).all() and np.all(model.scaler_std > 0)
    assert np.all(model.predict_labels(constant) >= 0)
    with pytest.raises(ValueError,match='memory budget'):
        CentroidPrimitiveSegmenter.fit(constant,k=1,memory_budget_bytes=1)
    with pytest.raises(ValueError,match='memory budget'):
        model.predict_labels(constant,memory_budget_bytes=1)
    with pytest.raises(ValueError,match='insufficient'):
        CentroidPrimitiveSegmenter.fit(constant,k=3,max_samples=2)
