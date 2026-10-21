"""Independent geometry, timing, medoid and syllable-boundary expectations."""

import numpy as np
import pytest

from studies.sequence_viz.recreate_generator_videos import (
    execution_descriptors,
    onset_matrix,
    rank_executions,
    rank_exemplar_neighbors,
    select_grid_views,
    select_views,
    syllable_segments,
)


def test_camera_centres_and_rotates_once_without_following_later_movement():
    transform = onset_matrix([100, 50], np.pi / 2, 200, output_size=400)
    np.testing.assert_allclose(transform @ [100, 50, 1], [200, 200], atol=1e-12)
    np.testing.assert_allclose(transform @ [100, 60, 1], [220, 200], atol=1e-12)
    # A later displacement stays a displacement; it is not re-centred.
    np.testing.assert_allclose(transform @ [100, 80, 1], [260, 200], atol=1e-12)


def test_descriptor_preserves_motion_but_removes_initial_position_and_reports_gaps():
    frames = np.arange(100)
    track = dict(
        frames=frames,
        centroid=np.column_stack([frames, np.zeros(100)]),
        heading=np.zeros(100),
        latent=np.ones((100, 10)),
    )
    calls = [dict(start_frame=10, stop_frame=30), dict(start_frame=50, stop_frame=70)]
    clock = frames / 10
    scales = dict(latent=np.ones(10), path=1)
    profiles, valid, duration = execution_descriptors(track, clock, calls, scales, samples=10)
    np.testing.assert_allclose(profiles[0], profiles[1], atol=1e-12)
    np.testing.assert_allclose(duration, [2, 2])
    assert valid.all()
    track["centroid"][50:70, 0] = 50
    moved, _, _ = execution_descriptors(track, clock, calls, scales, samples=10)
    assert np.linalg.norm(moved[0] - moved[1]) > 1
    # The missing sampled native frame remains unsupported rather than borrowing a neighbour.
    keep = frames != 20
    missing = {k: v[keep] for k, v in track.items()}
    assert not execution_descriptors(missing, clock, calls, scales, samples=10)[1][0]


def test_analytical_animal_balanced_medoid_matches_explicit_pair_distances():
    values = np.array([[0.0], [1.0], [3.0], [10.0]])
    distances, _ = rank_executions(values, ["a", "a", "b", "c"])
    weights = np.array([1 / 6, 1 / 6, 1 / 3, 1 / 3])
    explicit = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2) @ weights
    np.testing.assert_allclose(distances - distances[0], explicit - explicit[0], atol=1e-12)
    assert np.argmin(distances) == np.argmin(explicit) == 2
    _, percentile = rank_executions(np.array([[0.0], [0.0], [2.0]]), ["a", "b", "c"])
    np.testing.assert_allclose(percentile, [75.0, 75.0, 0.0])


def test_seeded_views_are_unique_and_keep_actual_percentiles():
    calls = [dict(recording="r", subject=str(i % 3), start_frame=i * 10) for i in range(11)]
    distance, percentile = np.arange(11, dtype=float), np.arange(100, -1, -10, dtype=float)
    result = select_views(calls, distance, percentile, seed=42)
    assert [e["role"] for e in result] == ["exemplar", "p90", "p50", "p10", "random"]
    assert [e["start_frame"] for e in result[:4]] == [0, 10, 50, 90]
    assert len({e["start_frame"] for e in result}) == 5
    assert result == select_views(calls, distance, percentile, seed=42)
    assert result[1]["actual_percentile"] == 90


def test_segments_are_actual_rle_with_physical_time_and_no_gap_filling():
    track = dict(frames=np.arange(8), labels=np.array([1, 1, 2, 2, 2, 3, 3, 3]))
    clock = np.arange(9) * 0.2
    segments = syllable_segments(track, clock, 1, 7)
    assert [s["syllable_id"] for s in segments] == [1, 2, 3]
    np.testing.assert_allclose([s["duration_s"] for s in segments], [0.2, 0.6, 0.4])
    missing = {k: v[v != 3] if k == "frames" else v[np.arange(8) != 3] for k, v in track.items()}
    with pytest.raises(ValueError, match="missing"):
        syllable_segments(missing, clock, 1, 7)


def test_grid_ranks_distance_to_actual_exemplar_and_random_includes_ranked_choices():
    # Weighted centre lies at 4.5; actual medoid is 3. Distances must be to 3.
    values = np.array([[0.0], [1.0], [3.0], [10.0]])
    distance, percentile, exemplar = rank_exemplar_neighbors(values, ["a", "a", "b", "c"])
    np.testing.assert_allclose(distance, [9, 4, 0, 49])
    assert exemplar == 2
    calls = [dict(recording="r", subject=str(i), start_frame=i) for i in range(4)]
    result = select_grid_views(calls, distance, percentile, seed=0, exemplar=exemplar)
    assert [e["role"] for e in result] == ["exemplar", "p80", "random"]
    assert result[0]["start_frame"] == 2
    assert result[1]["start_frame"] == 1
    # A separate uniform full-pool RNG expectation, including repeated ranked choices.
    expected = int(np.random.default_rng(0).integers(4))
    assert result[-1]["start_frame"] == expected
    draws = [select_grid_views(calls, distance, percentile, seed, exemplar)[-1] for seed in range(30)]
    assert {e["start_frame"] for e in draws} == {0, 1, 2, 3}
    assert any(e["repeats_roles"] for e in draws)


def test_grid_preserves_each_execution_frame_and_blanks_only_finished_panels(tmp_path):
    import shutil

    cv2 = pytest.importorskip("cv2")
    if not shutil.which("ffmpeg"):
        pytest.skip("synthetic grid encoding requires ffmpeg")
    from studies.sequence_viz.render_generator_review_grids import render_grid

    counts, clips = [2, 4, 3], []
    for panel, count in enumerate(counts):
        filename = f"source{panel}.mp4"
        writer = cv2.VideoWriter(str(tmp_path / filename), cv2.VideoWriter_fourcc(*"mp4v"), 60, (720, 960))
        assert writer.isOpened()
        for index in range(count):
            frame = np.zeros((960, 720, 3), np.uint8)
            frame[:, :, panel] = 100 + 20 * index
            writer.write(frame)
        writer.release()
        clips.append(
            dict(
                filename=filename, frames=count, sha256="synthetic", example=dict(role=str(panel), selection_mode="p80")
            )
        )
    result = render_grid(dict(stem="fixture"), clips, tmp_path, selection_mode="p80")
    assert result["frames"] == 4
    capture = cv2.VideoCapture(str(tmp_path / result["filename"]))
    try:
        for index in range(4):
            ok, image = capture.read()
            assert ok
            for panel, count in enumerate(counts):
                expected = np.full(3, 18) if index >= count else np.eye(3, dtype=int)[panel] * (100 + 20 * index)
                np.testing.assert_allclose(image[10, panel * 720 + 10], expected, atol=12)
        assert not capture.read()[0]
    finally:
        capture.release()


def test_separate_movie_sets_share_mode_stay_near_target_and_random_has_full_support():
    from studies.sequence_viz.render_generator_review_grids import select_movie_sets

    calls = [dict(recording="r", subject=str(i % 3), start_frame=i) for i in range(101)]
    distance, percentile = np.arange(101, dtype=float), np.arange(100, -1, -1, dtype=float)
    sets = select_movie_sets(calls, distance, percentile, seed=42, exemplar=0)
    assert {k: [e["start_frame"] for e in v] for k, v in sets.items() if k != "random"} == {
        "p100": [0, 1, 2],
        "p80": [20, 19, 21],
    }
    assert [e["start_frame"] for e in sets["random"]] == np.random.default_rng(42).choice(
        101, 3, replace=False
    ).tolist()
    for mode, es in sets.items():
        assert len(es) == len({e["start_frame"] for e in es}) == 3
        assert {e["selection_mode"] for e in es} == {mode}
    assert {
        e["start_frame"]
        for seed in range(200)
        for e in select_movie_sets(calls, distance, percentile, seed, 0)["random"]
    } == set(range(101))
    # No forced distant-animal selection: three closest top examples still supplied.
    single_mouse = [dict(c, subject="one") for c in calls]
    assert [e["start_frame"] for e in select_movie_sets(single_mouse, distance, percentile, 42, 0)["p100"]] == [0, 1, 2]


def test_encoder_rejects_mixed_selection_modes_before_writing_movie(tmp_path):
    from studies.sequence_viz.render_generator_review_grids import render_grid

    mixed = [dict(example=dict(selection_mode=m)) for m in ("p100", "p80", "random")]
    with pytest.raises(ValueError, match="mixed selection"):
        render_grid(dict(stem="family"), mixed, tmp_path, selection_mode="p80")
    assert list(tmp_path.iterdir()) == []
