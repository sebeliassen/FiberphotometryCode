"""Small independent landmark/file contexts; no cohort data or video backend."""

import __future__

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpcode.viz import hashing
from fpcode.viz.context import BottomPoseLandmarks, LandmarkGroups, SelectionInputs
from fpcode.viz.hashing import HashCache
from fpcode.viz.media import load_fused_tracks, media_file_identity
from fpcode.viz.pose_geometry import compute_robust_bot_pose, fusion_geometry
from fpcode.viz.provenance import (
    make_cache_key,
    selection_provenance_matches,
    source_input_hashes,
    verify_manifest_inputs,
)
from fpcode.viz.selection.density import choose_tier
from fpcode.viz.selection.eligibility import summarize_window_qc

ROOT = Path(__file__).resolve().parents[2]


def test_quality_groups_follow_each_contexts_column_order_and_names():
    supported = np.array([[True, True, False, True], [True, False, True, True]])
    first = LandmarkGroups((0,), (1, 2), ("fore", "hind"), (1,), (2,))
    second = LandmarkGroups((2,), (0, 3), ("left", "right"), (0,), (3,))
    a = summarize_window_qc(supported, groups=first)
    b = summarize_window_qc(supported, groups=second)
    assert a["min_dorsal_valid_fraction"] == 1.
    assert a["min_paw_valid_fraction"] == .5
    assert a["per_paw_valid_fraction"] == {"fore": .5, "hind": .5}
    assert a["per_paw_max_gap_frames"] == {"fore": 1, "hind": 1}
    assert b["min_dorsal_valid_fraction"] == .5
    assert b["min_paw_valid_fraction"] == 1.
    assert b["per_paw_valid_fraction"] == {"left": 1., "right": 1.}
    assert b["per_paw_max_gap_frames"] == {"left": 0, "right": 0}
    # Neither context rewrites the source mask or inherits the other's groups.
    np.testing.assert_array_equal(supported, [[True, True, False, True], [True, False, True, True]])


def test_bottom_pose_uses_supplied_names_and_fallback_location():
    columns = pd.MultiIndex.from_product([["tracker"], ["snout_a", "tail_a", "snout_b", "tail_b"],
                                         ["x", "y", "likelihood"]])
    frame = pd.DataFrame([[80., 20., .9, 20., 20., .9, 200., 100., .9, 200., 60., .9]], columns=columns)
    first = BottomPoseLandmarks("snout_a", "tail_a", (), (), (1., 2.))
    second = BottomPoseLandmarks("snout_b", "tail_b", (), (), (30., 40.))
    assert compute_robust_bot_pose(frame, "tracker", 0, landmarks=first) == ((50., 20.), 0., True)
    centroid, heading, valid = compute_robust_bot_pose(frame, "tracker", 0, landmarks=second)
    assert centroid == (200., 80.) and valid
    assert heading == pytest.approx(np.pi / 2)
    missing = frame * np.nan
    centroid, heading, valid = compute_robust_bot_pose(missing, "tracker", 0, landmarks=second,
                                                     top_heading_fallback=0.)
    assert centroid == (30., 40.) and not valid
    assert heading == pytest.approx(np.pi)


def test_density_tier_uses_explicit_sampler_neighbor_count():
    assert choose_tier(6, 4, n_neighbors=5) == "tier_1_density"
    assert choose_tier(6, 4, n_neighbors=10) == "tier_2_compact_neighborhood"
    assert choose_tier(50, 12) == "tier_2_compact_neighborhood"
    assert choose_tier(51, 12) == "tier_1_density"


def test_track_reader_uses_file_and_requested_landmark_order(tmp_path):
    columns = pd.MultiIndex.from_product([["tracker"], ["point_b", "point_a"], ["x", "y", "likelihood"]])
    frame = pd.DataFrame([[9., 8., .2, 1., 2., .9], [np.nan, 7., 0., 3., 4., .5]], columns=columns)
    path = tmp_path / "custom_pose.h5"
    frame.to_hdf(path, key="df")
    xy, confidence = load_fused_tracks(path, ("point_a", "point_b"))
    np.testing.assert_array_equal(xy, [[[1., 2.], [9., 8.]], [[3., 4.], [np.nan, 7.]]])
    np.testing.assert_array_equal(confidence, np.array([[.9, .2], [.5, 0.]], dtype=np.float32))
    assert xy.dtype == confidence.dtype == np.float32


def test_production_selector_loads_its_explicit_tracking_file_without_model_imports(tmp_path):
    source = ROOT / "studies/multiview_experiments/04_reports_and_figures/select_cohort_static_21kp_v2_syllable_manifest.py"
    tree = ast.parse(source.read_text())
    loader = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "get_fused")
    namespace = {"fused_cache": {}, "FUSED_DIR": tmp_path, "load_fused_tracks": load_fused_tracks,
                 "cohort_config": {"use_bodyparts": ["custom_landmark"]}}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    columns = pd.MultiIndex.from_product([["tracker"], ["custom_landmark"], ["x", "y", "likelihood"]])
    pd.DataFrame([[4., 5., .6]], columns=columns).to_hdf(tmp_path / "trial_tandem_21kp.h5", key="df")
    xy, confidence = namespace["get_fused"]("trial")
    np.testing.assert_array_equal(xy, [[[4., 5.]]])
    np.testing.assert_array_equal(confidence, np.array([[.6]], dtype=np.float32))
    assert namespace["get_fused"]("trial")[0] is xy
    with pytest.raises(RuntimeError, match="refusing to bypass QC") as error:
        namespace["get_fused"]("missing")
    assert isinstance(error.value.__cause__, FileNotFoundError)


def _context(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    paths = {}
    for kind in ("results", "config", "video_frame_indexes", "preprocessed", "calibration"):
        paths[kind] = directory / kind
        paths[kind].write_bytes(f"{name}:{kind}".encode())
    fused = directory / "custom_tracking.h5"
    fused.write_bytes(name.encode())
    fused.with_name(fused.name + ".manifest.json").write_text(name)
    bout = {"bout_id": "session_seg2:3", "segment_name": "session_seg2", "is_blank": False}
    for kind in ("top_vid", "bot_vid", "top_h5", "bot_h5"):
        path = directory / kind
        path.write_bytes(f"{name}:{kind}".encode())
        bout[kind] = str(path)
    inputs = SelectionInputs(**{key: paths[key] for key in (
        "results", "config", "video_frame_indexes", "preprocessed")},
        calibration_files=(paths["calibration"],), fused_artifacts={"session": fused})
    manifest = {"schema_version": 4, "qc_config": {"applied": True}, "syllables": {"1": {"bouts": [bout]}}}
    return inputs, manifest


def test_provenance_keeps_two_projects_files_and_calibration_independent(tmp_path):
    cache = HashCache(tmp_path / "hash_cache.json")
    first, manifest_a = _context(tmp_path, "first")
    second, manifest_b = _context(tmp_path, "second")
    for inputs, manifest, name in ((first, manifest_a, "first"), (second, manifest_b, "second")):
        hashes = source_input_hashes(cache, inputs=inputs)
        assert hashes["results_sha256"] == hashlib.sha256(f"{name}:results".encode()).hexdigest()
        manifest["input_hashes"] = hashes
        verify_manifest_inputs(manifest, cache, inputs=inputs)
        code = tmp_path / "renderer.py"
        code.write_text("# small renderer\n")
        key = make_cache_key("grid", 1, manifest, tmp_path / "manifest.json", cache, [code], {}, inputs=inputs)
        assert key["calibration_sha256"] == {
            "calibration": hashlib.sha256(f"{name}:calibration".encode()).hexdigest(),
        }
        identity = media_file_identity(manifest, 1, fused_artifacts=inputs.fused_artifacts)
        assert len(identity) == 6
        assert all(Path(path).parent == inputs.results.parent for path in identity)
        assert key["selected_source_file_identity"] == identity
        sidecar = inputs.results.parent / "sidecar.json"
        sidecar.write_text(json.dumps({"cache_key": key, "rendered_bouts": ["session_seg2:3"]}))
        assert selection_provenance_matches(sidecar, manifest, 1, [code], inputs=inputs)
        inputs.calibration_files[0].write_text("changed")
        assert not selection_provenance_matches(sidecar, manifest, 1, [code], inputs=inputs)
    with pytest.raises(ValueError, match="Selection input mismatch"):
        verify_manifest_inputs(manifest_a, cache, inputs=second)
    with pytest.raises(TypeError):
        first.fused_artifacts["another"] = tmp_path / "other.h5"


@pytest.mark.parametrize("changed_implementation", [
    "context.py", "media.py", "legacy_api.py", "legacy_defaults.py", "selection/eligibility.py",
])
def test_facade_fingerprint_tracks_actual_helpers_and_invalidates_provenance(
    tmp_path, monkeypatch, changed_implementation,
):
    # Simulate installed implementation files without changing any repository
    # source. The caller supplies its actual V2 facade, as production studies do.
    package = tmp_path / "installed_viz"
    (package / "selection").mkdir(parents=True)
    implementations = ("v2.py", "hashing.py", "context.py", "media.py", "pose_geometry.py",
                       "provenance.py", "legacy_api.py", "legacy_defaults.py", "selection/eligibility.py")
    for relative in implementations:
        (package / relative).write_text("# original " + relative)
    monkeypatch.setattr(hashing, "__file__", str(package / "hashing.py"))
    inputs, manifest = _context(tmp_path, "project")
    cache = HashCache(tmp_path / "hash_cache.json")
    code_files = [package / "v2.py"]
    key = make_cache_key("grid", 1, manifest, tmp_path / "manifest.json", cache, code_files, {}, inputs=inputs)
    assert {"fpcode/viz/" + name for name in implementations} <= key["code_sha256"].keys()
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps({"cache_key": key, "rendered_bouts": ["session_seg2:3"]}))
    assert selection_provenance_matches(sidecar, manifest, 1, code_files, inputs=inputs)
    (package / changed_implementation).write_text("# changed algorithm or context")
    changed_key = make_cache_key("grid", 1, manifest, tmp_path / "manifest.json", cache, code_files, {}, inputs=inputs)
    assert changed_key["code_sha256"] != key["code_sha256"]
    assert not selection_provenance_matches(sidecar, manifest, 1, code_files, inputs=inputs)


def test_geometry_reads_the_supplied_artifacts_recorded_transform(tmp_path):
    for name, shift in (("first", [2., 3.]), ("second", [20., 30.])):
        artifact = tmp_path / (name + ".h5")
        artifact.write_bytes(b"artifact")
        stat = artifact.stat()
        record = {"kind": "fused_static_21kp", "artifact": {
            "name": artifact.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, "config": {
                "affine_matrix": [[1., 0.], [0., 1.]], "affine_translation": shift,
                "arena_width": 100., "arena": name, "epoch": "recorded"}}
        manifest = artifact.with_name(artifact.name + ".manifest.json")
        manifest.write_text(json.dumps(record))
        geometry = fusion_geometry(artifact)
        np.testing.assert_array_equal(geometry["b"], shift)
        assert geometry["arena"] == name and geometry["epoch"] == "recorded"


def test_generic_visualization_imports_and_use_do_not_discover_a_cohort():
    # Fresh process prevents the historical tests' V2 imports from masking an
    # accidental dependency on cohort defaults or source-media discovery.
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import importlib.abc
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {{'fpcode.viz.legacy_defaults', 'fpcode.viz.legacy_io', 'fpcode.viz.legacy_api'}}:
            raise AssertionError('generic visualization reached ' + fullname)
sys.meta_path.insert(0, NoLegacy())
import numpy as np
from pathlib import Path
def no_discovery(*args, **kwargs):
    raise AssertionError('generic visualization attempted filesystem discovery')
Path.glob = Path.rglob = no_discovery
from fpcode.viz.context import LandmarkGroups
from fpcode.viz.pose_geometry import egocentric_window
from fpcode.viz.provenance import manifest_global_hash
from fpcode.viz.selection.eligibility import summarize_window_qc
from fpcode.viz.selection.density import choose_tier
assert summarize_window_qc(np.ones((2, 2), bool), groups=LandmarkGroups((0,), (1,), ('paw',)))['min_paw_valid_fraction'] == 1.
assert choose_tier(51, 12) == 'tier_1_density'
assert egocentric_window(np.zeros((2, 2, 2)), [0., 0.], 0.).shape == (2, 2, 2)
assert len(manifest_global_hash({{'schema_version': 4}})) == 64
assert not {{'jax', 'keypoint_moseq', 'fpcode.tracking.fusion'}} & sys.modules.keys()
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
