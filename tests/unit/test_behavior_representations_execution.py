"""Independent numerical tests for continuous features and execution analysis."""

import numpy as np
import pytest

from fpcode.behavior.analysis.execution import (
    aggregate_execution_metrics,
    measure_occurrence_execution,
    measure_phase_pacing,
)
from fpcode.behavior.observations.representations import (
    compute_interval_trajectory_metrics,
    extract_trajectory_features,
)
from fpcode.contracts import (
    BehaviorResult,
    CompletionEvidence,
    FrameSpan,
    PhaseRecord,
    PoseSeries,
    SessionKey,
    TimeAxis,
    UnitOccurrence,
    UnitStream,
    UnitType,
)

SESSION = SessionKey("rec_test", "mouse_test")


def test_arena_distances_and_downstream_units_preserve_native_support():
    from fpcode.behavior.observations.representations import extract_associated_trajectory_features

    centers = np.array([[0., 0.], [5., 5.], [2., 3.], [12., 5.], [4., 4.]])
    coords = np.stack([centers + [0.5, 0.], centers - [0.5, 0.]], axis=1)
    confidence = np.ones((5, 2))
    confidence[-1, 0] = 0.
    time = TimeAxis(np.array([100, 101, 105, 106, 107]), 'frame', 'video')
    pose = PoseSeries(time, coords, confidence, ('nose', 'tail_base'), 'cm', 'arena')
    association = extract_associated_trajectory_features(
        pose, session=SESSION, fps=10., confidence_threshold=.5,
        arena_bounds=np.array([0., 10., 0., 10.]),
    )
    features = association.series
    names = [signal.name for signal in features.signals]
    columns = [names.index(name) for name in (
        'distance_nearest_corner', 'distance_nearest_edge', 'distance_center',
    )]
    np.testing.assert_allclose(features.values[:3, columns], [
        [0., 0., np.sqrt(50.)], [np.sqrt(50.), 5., 0.], [np.sqrt(13.), 2., np.sqrt(13.)],
    ])
    assert np.all(features.valid[:3, columns])
    assert not np.any(features.valid[3:, columns])
    assert np.all(np.isnan(features.values[3:, columns]))
    assert all(features.signals[col].unit == 'cm' for col in columns)
    assert features.time is time
    assert 'arena_bounds' in association.provenance
    # Same frame features can measure primitive, coarse-resolution, or episode
    # intervals; they need not be inserted into the detector's input labels.
    for level in ('primitive', 'episode', 'assembly'):
        stream = UnitStream(SESSION, level, 'frozen', 'video', 10., (FrameSpan(100, 108),),
                            (UnitType('A'),), (UnitOccurrence('a', 'A', FrameSpan(100, 108)),))
        table = measure_occurrence_execution(stream, association, feature_name='distance_nearest_edge')
        assert table.mean_feature.iloc[0] == pytest.approx(7. / 3.)
        assert table.exposure_s.iloc[0] == pytest.approx(.3)
        assert table.supported_samples.iloc[0] == 3


@pytest.mark.parametrize('bounds', [(0, 0, 0, 10), (10, 0, 0, 10),
                                    (0, 10, 0, np.nan), (0, 10, 0)])
def test_arena_geometry_requires_explicit_finite_positive_extents(bounds):
    pose = PoseSeries(TimeAxis(np.array([0]), 'frame', 'video'),
                      np.array([[[1., 1.], [0., 1.]]]), np.ones((1, 2)),
                      ('nose', 'tail_base'), 'px', 'arena')
    with pytest.raises(ValueError, match='arena_bounds'):
        extract_trajectory_features(pose, fps=10., arena_bounds=bounds)


def test_nose_corner_distance_uses_its_own_support_and_robust_bounds_reject_outliers():
    from fpcode.behavior.observations.representations import robust_position_bounds

    pose = PoseSeries(TimeAxis(np.array([100, 101]), 'frame', 'video'),
                      np.array([[[0., 0.], [4., 4.]], [[0., 0.], [4., 4.]]]),
                      np.array([[1., 1.], [1., .1]]), ('nose', 'tail_base'), 'px', 'arena')
    features = extract_trajectory_features(pose, fps=10., confidence_threshold=.8,
                                           arena_bounds=(0., 10., 0., 10.), arena_landmarks=('nose',),
                                           arena_corner_features=True)
    names = [signal.name for signal in features.signals]
    midpoint = names.index('distance_nearest_corner')
    nose = names.index('nose_distance_nearest_corner')
    assert features.values[0, midpoint] == pytest.approx(np.sqrt(8.))
    assert features.values[0, nose] == 0.
    assert not features.valid[1, midpoint]
    assert features.valid[1, nose]
    assert features.values[0, names.index('nose_distance_top_left_corner')] == 0.
    assert features.values[0, names.index('nose_distance_bottom_right_corner')] == pytest.approx(np.sqrt(200.))
    positions = np.vstack([np.zeros((100, 2)), np.full((100, 2), 10.), [[-999., -999.], [999., 999.]]])
    assert robust_position_bounds(positions, quantiles=(.01, .99)) == (0., 10., 0., 10.)
    with pytest.raises(ValueError, match='extent'):
        robust_position_bounds(np.ones((5, 2)))
    with pytest.raises(ValueError, match='quantiles'):
        robust_position_bounds(positions, quantiles=(.99, .01))


def test_landmark_centroid_keeps_missing_paws_missing_and_owns_its_support():
    from fpcode.behavior.observations.representations import append_landmark_centroid

    coords = np.tile(np.array([[0., 0.], [4., 0.], [0., 4.], [4., 4.]]), (3, 1, 1))
    confidence = np.array([[1., 1., 1., 1.], [.9, .8, .1, .1], [1., .1, .1, .1]])
    pose = PoseSeries(TimeAxis(np.arange(3), 'frame', 'video'), coords, confidence,
                      ('lf', 'rf', 'lh', 'rh'), 'px', 'topcam_registered')
    augmented = append_landmark_centroid(pose, bodyparts=pose.bodyparts, name='paw_center',
                                         confidence_threshold=.8, minimum_count=2)
    np.testing.assert_allclose(augmented.coordinates[:2, -1], [[2., 2.], [2., 0.]])
    assert augmented.confidence[:2, -1].tolist() == [1., .8]
    assert not augmented.valid[2, -1]
    assert np.isnan(augmented.coordinates[2, -1]).all()
    assert pose.bodyparts == ('lf', 'rf', 'lh', 'rh')


def test_speed_analytical_translation_and_stationary():
    # 5 frames at 10 FPS: x advances 2 px per frame -> speed = 20 px/s
    coords = np.zeros((5, 2, 2))  # (T, K, 2)
    for t in range(5):
        coords[t, 0] = [t * 2.0, 0.0]  # nose
        coords[t, 1] = [t * 2.0 - 1.0, 0.0]  # tail_base
    time = TimeAxis(np.arange(5), unit="frame", clock="video")
    conf = np.ones((5, 2))
    pose = PoseSeries(
        time=time,
        coordinates=coords,
        confidence=conf,
        bodyparts=("nose", "tail_base"),
        unit="px",
        coordinate_frame="arena",
    )
    sig = extract_trajectory_features(pose, fps=10.0)
    names = [s.name for s in sig.signals]
    speed_idx = names.index("speed")

    # Frame 0 has no prior frame -> valid = False
    assert not sig.valid[0, speed_idx]
    # Frames 1..4 -> valid = True, speed = 20 px/s
    assert np.all(sig.valid[1:, speed_idx])
    assert np.allclose(sig.values[1:, speed_idx], 20.0)


def test_dropped_frame_and_nonzero_origin_support():
    # Frames: [100, 101, 105, 106] (gap between 101 and 105)
    t_frames = np.array([100, 101, 105, 106])
    coords = np.zeros((4, 2, 2))
    for i, t in enumerate(t_frames):
        coords[i, 0] = [t * 1.0, 0.0]
        coords[i, 1] = [t * 1.0 - 1.0, 0.0]
    time = TimeAxis(t_frames, unit="frame", clock="video")
    conf = np.ones((4, 2))
    pose = PoseSeries(
        time=time,
        coordinates=coords,
        confidence=conf,
        bodyparts=("nose", "tail_base"),
        unit="px",
        coordinate_frame="arena",
    )
    sig = extract_trajectory_features(pose, fps=10.0)
    names = [s.name for s in sig.signals]
    speed_idx = names.index("speed")

    # Frame 0 (origin 100) -> invalid
    assert not sig.valid[0, speed_idx]
    # Frame 1 (101) contiguous -> valid
    assert sig.valid[1, speed_idx]
    # Frame 2 (105, gap of 4) -> invalid
    assert not sig.valid[2, speed_idx]
    # Frame 3 (106) contiguous to 105 -> valid
    assert sig.valid[3, speed_idx]


def test_heading_wrap_circular_yaw():
    # Mouse turns slightly across +/- pi:
    # At t=0, angle is +3.10 rad (177.6 deg)
    # At t=1, angle is -3.10 rad (-177.6 deg)
    # Physical turn is +0.0828 rad (or circular diff ~ +0.08 rad), NOT -6.20 rad jump!
    t_frames = np.array([0, 1])
    time = TimeAxis(t_frames, unit="frame", clock="video")
    coords = np.zeros((2, 2, 2))
    # Tail at origin
    coords[:, 1] = [0.0, 0.0]
    # Nose at angle
    coords[0, 0] = [np.cos(3.10), np.sin(3.10)]
    coords[1, 0] = [np.cos(-3.10), np.sin(-3.10)]
    conf = np.ones((2, 2))
    pose = PoseSeries(
        time=time,
        coordinates=coords,
        confidence=conf,
        bodyparts=("nose", "tail_base"),
        unit="px",
        coordinate_frame="arena",
    )
    sig = extract_trajectory_features(pose, fps=10.0)
    names = [s.name for s in sig.signals]
    yaw_idx = names.index("angular_velocity")

    # Frame 1 yaw rate
    assert sig.valid[1, yaw_idx]
    # Angular velocity = circular_diff * fps
    # circular diff is (-3.10 - 3.10 + pi) % 2pi - pi = (-6.20 + 3.14159...) % 2pi ...
    # which is approx +0.0831 * 10 = +0.831 rad/s
    assert np.abs(sig.values[1, yaw_idx]) < 2.0  # NOT ~62.0 rad/s!


def test_path_length_versus_net_displacement():
    # Out and back: (0,0) -> (10,0) -> (0,0)
    # Path length = 20, net displacement = 0
    t_frames = np.array([0, 1, 2])
    time = TimeAxis(t_frames, unit="frame", clock="video")
    coords = np.zeros((3, 2, 2))
    coords[0, 0] = [0.0, 0.0]
    coords[0, 1] = [-1.0, 0.0]
    coords[1, 0] = [10.0, 0.0]
    coords[1, 1] = [9.0, 0.0]
    coords[2, 0] = [0.0, 0.0]
    coords[2, 1] = [-1.0, 0.0]
    pose = PoseSeries(time=time, coordinates=coords, confidence=np.ones((3, 2)), bodyparts=("nose", "tail_base"), unit="px", coordinate_frame="arena")
    sig = extract_trajectory_features(pose, fps=10.0)
    metrics = compute_interval_trajectory_metrics(sig, 0, 3)

    assert metrics["path_length"] == pytest.approx(20.0)
    assert metrics["net_displacement"] == pytest.approx(0.0)


def test_execution_metrics_and_aggregation():
    # Stream with two occurrences
    stream = UnitStream(
        session=SESSION,
        level="episode",
        model_id="test_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 50),),
        catalog=(UnitType("turn"), UnitType("run")),
        occurrences=(
            UnitOccurrence("occ_1", "turn", FrameSpan(0, 10)),
            UnitOccurrence("occ_2", "run", FrameSpan(10, 30)),
        ),
    )
    # Pose with constant speed
    coords = np.zeros((50, 2, 2))
    for t in range(50):
        coords[t, 0] = [t * 2.0, 0.0]
        coords[t, 1] = [t * 2.0 - 1.0, 0.0]
    pose = PoseSeries(
        time=TimeAxis(np.arange(50), unit="frame", clock="video"),
        coordinates=coords,
        confidence=np.ones((50, 2)),
        bodyparts=("nose", "tail_base"),
        unit="px",
        coordinate_frame="arena",
    )
    features = extract_trajectory_features(pose, fps=10.0)

    completion = (
        CompletionEvidence("occ_1", status="completed", final_phase_reached=True),
        CompletionEvidence("occ_2", observed_exit=False, status="censored", final_phase_reached=False, right_censored=True),
    )
    phases = (
        PhaseRecord("occ_1", "entry", 0, FrameSpan(0, 4)),
        PhaseRecord("occ_1", "exit", 1, FrameSpan(4, 10)),
    )
    result = BehaviorResult(stream=stream, completion=completion, phases=phases)

    df = measure_occurrence_execution(result, features, feature_name="speed")
    assert len(df) == 2
    assert df.loc[0, "duration_s"] == pytest.approx(1.0)
    assert df.loc[1, "duration_s"] == pytest.approx(2.0)
    assert df.loc[0, "completion_status"] == "completed"
    assert df.loc[1, "completion_status"] == "censored"
    assert df.loc[0, "phase_count"] == 2
    assert df.loc[1, "phase_count"] == 0

    agg = aggregate_execution_metrics(df)
    assert len(agg) == 2
    turn_row = agg[agg["type_id"] == "turn"].iloc[0]
    assert turn_row["count"] == 1
    assert turn_row["completion_rate"] == 1.0

    run_row = agg[agg["type_id"] == "run"].iloc[0]
    assert np.isnan(run_row["completion_rate"])
    assert run_row["censored_count"] == 1

    # Phase pacing
    pacing = measure_phase_pacing(result)
    assert len(pacing) == 2
    assert pacing.loc[0, "phase_id"] == "entry"
    assert pacing.loc[0, "fraction_of_occurrence"] == pytest.approx(0.4)
    assert pacing.loc[1, "fraction_of_occurrence"] == pytest.approx(0.6)


def test_native_interval_never_guesses_row_indices_and_profiles_preserve_gaps():
    from fpcode.behavior.analysis.execution import measure_onset_profiles, measure_phase_profiles
    from fpcode.contracts import SignalMetadata, SignalSeries
    small_origin = SignalSeries(TimeAxis(np.arange(1, 5), 'frame', 'video'),
                                np.array([[10.], [20.], [30.], [40.]]), (SignalMetadata('speed', 'px/s'),))
    assert compute_interval_trajectory_metrics(small_origin, 1, 3)['mean_speed'] == pytest.approx(15.)
    features = SignalSeries(TimeAxis(np.array([0, 1, 4, 5]), 'frame', 'video'),
                            np.array([[1.], [3.], [5.], [7.]]), (SignalMetadata('speed', 'px/s'),))
    stream = UnitStream(SESSION, 'episode', 'profiles', 'video', 10., (FrameSpan(0, 6),),
                        (UnitType('A'),), (UnitOccurrence('a', 'A', FrameSpan(0, 6)),))
    result = BehaviorResult(stream, phases=(PhaseRecord('a', 'p', 0, FrameSpan(0, 6)),))
    phase = measure_phase_profiles(result, features, n_bins=3)
    onset = measure_onset_profiles(stream, features, after_s=.6, bin_s=.2)
    for table in (phase, onset):
        assert table.mean_feature.iloc[0] == pytest.approx(2.)
        assert np.isnan(table.mean_feature.iloc[1])
        assert table.mean_feature.iloc[2] == pytest.approx(6.)
        assert table.exposure_s.tolist() == pytest.approx([.2, 0., .2])
    assert phase.phase_duration_s.tolist() == pytest.approx([.6]*3)
    with pytest.raises(ValueError, match='validated phases'):
        measure_phase_profiles(BehaviorResult(stream), features)


def test_explicit_seconds_mapping_and_weighted_exposure():
    from fpcode.contracts import SignalMetadata, SignalSeries
    from fpcode.contracts.behavior_results import FeatureAssociation
    series = SignalSeries(TimeAxis(np.array([10., 10.1]), 's', 'video'),
                          np.array([[10.], [30.]]), (SignalMetadata('speed', 'px/s'),))
    stream = UnitStream(SESSION, 'primitive', 'weighted', 'video', 10., (FrameSpan(100, 102),),
                        (UnitType('A'),), (UnitOccurrence('a', 'A', FrameSpan(100, 102)),))
    with pytest.raises(ValueError, match='explicit native'):
        measure_occurrence_execution(stream, series)
    association = FeatureAssociation(series, SESSION, 'arena', native_frames=np.array([100, 101]),
                                     sample_exposure_s=np.array([.1, .05]))
    table = measure_occurrence_execution(stream, association)
    assert table.mean_feature.iloc[0] == pytest.approx(2.5/.15)
    assert table.exposure_s.iloc[0] == pytest.approx(.15)
    assert aggregate_execution_metrics(table).pooled_feature_mean.iloc[0] == pytest.approx(2.5/.15)
    assert table.peak_latency_frames.iloc[0] == 1


def test_onset_profile_cannot_cross_touching_validity_boundary():
    from fpcode.behavior.analysis.execution import measure_onset_profiles
    from fpcode.contracts import SignalMetadata, SignalSeries
    features = SignalSeries(TimeAxis(np.arange(4), 'frame', 'video'),
                            np.array([[1.], [1.], [99.], [99.]]), (SignalMetadata('speed'),))
    stream = UnitStream(SESSION, 'episode', 'boundary', 'video', 10.,
                        (FrameSpan(0, 2), FrameSpan(2, 4)), (UnitType('A'),),
                        (UnitOccurrence('a', 'A', FrameSpan(0, 2)),))
    profile = measure_onset_profiles(stream, features, after_s=.4, bin_s=.2)
    assert profile.mean_feature.iloc[0] == 1.
    assert profile.exposure_s.iloc[1] == 0.
    assert np.isnan(profile.mean_feature.iloc[1])


def test_profile_budget_and_exact_native_coordinate_limit():
    from fpcode.behavior.analysis.execution import measure_onset_profiles
    from fpcode.contracts import SignalMetadata, SignalSeries
    origin = 2**53+1
    stream = UnitStream(SESSION,'episode','large','video',10.,(FrameSpan(origin,origin+2),),
                        (UnitType('A'),),(UnitOccurrence('a','A',FrameSpan(origin,origin+2)),))
    feature = SignalSeries(TimeAxis(np.array([origin,origin+1],dtype=np.int64),'frame','video'),
                           np.ones((2,1)),(SignalMetadata('speed'),))
    with pytest.raises(ValueError,match='floating integer'):
        measure_onset_profiles(stream,feature)
    with pytest.raises(ValueError,match='positive integer'):
        measure_onset_profiles(stream,feature,max_rows=True)
