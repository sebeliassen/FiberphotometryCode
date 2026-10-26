"""Exercise real compatibility paths without the study test search paths."""

import os
import subprocess
import sys
from pathlib import Path

from ops.repo import update_shims

ROOT = Path(__file__).resolve().parents[2]


def test_compat_camera_and_fusion_imports_resolve_in_a_fresh_interpreter(tmp_path):
    script = f"""
import importlib
import sys
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT)!r}, {str(ROOT / 'compat')!r}]
for legacy, canonical, attribute in [
    ('core.fusion_engine', 'fpcode.tracking.fusion', 'fuse_single_pair'),
    ('core.static_fusion_engine', 'fpcode.tracking.static_fusion', 'fuse_single_pair_static'),
    ('core.camera_correspondence', 'fpcode.tracking.camera_correspondence', 'get_session_mapping'),
    ('core.ttl_sync', 'fpcode.tracking.clock_sync', 'top_to_bot_frame'),
]:
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new, (legacy, old.__file__)
    assert callable(getattr(old, attribute))
assert not {{'jax', 'keypoint_moseq', 'torch', 'numba', 'cv2'}} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_shim_regeneration_preserves_independently_registered_paths(tmp_path, monkeypatch):
    registry = tmp_path / "registry/SHIMS.tsv"
    registry.parent.mkdir()
    retained = "studies/custom.py\tfpcode.photometry.event_windows\tforwarder\tACTIVE\tkeep this note"
    registry.write_text("shim_path\ttarget_module\ttype\tstatus\tnotes\n" + retained + "\n"
                        "core/fusion_engine.py\tobsolete.module\tmodule_proxy\tACTIVE\told\n")
    monkeypatch.setattr(update_shims, "REPO_ROOT", tmp_path)
    update_shims.write_shims_tsv()
    lines = registry.read_text().splitlines()
    assert retained in lines
    assert len(lines) == len(update_shims.SHIMS_MAP) + 2
    targets = {parts[0]: parts[1] for line in lines[1:] if (parts := line.split("\t"))}
    assert targets["compat/core/fusion_engine.py"] == "fpcode.tracking.fusion"
    assert targets["compat/core/static_fusion_engine.py"] == "fpcode.tracking.static_fusion"
    assert targets["compat/core/camera_correspondence.py"] == "fpcode.tracking.camera_correspondence"
    assert not any(path.startswith("core/") for path in targets)


def test_pipeline_aliases_keep_canonical_identity_in_both_import_orders(tmp_path):
    for legacy_first in (True, False):
        script = f"""
import importlib
import sys
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT)!r}, {str(ROOT / 'compat')!r}]
for verb in ('train', 'evaluate', 'fuse', 'render', 'export'):
    legacy, canonical = 'pipelines.' + verb, 'fpcode.cli.' + verb
    first, second = (legacy, canonical) if {legacy_first!r} else (canonical, legacy)
    one = importlib.import_module(first)
    two = importlib.import_module(second)
    assert one is two, (first, second)
    import runpy
    sys.argv = [str({str(ROOT / 'compat/pipelines')!r} + '/' + verb + '.py'), '--help']
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as result:
        assert result.code == 0
    else:
        raise AssertionError('physical compatibility help did not exit')
assert not {{'jax', 'jaxlib', 'keypoint_moseq', 'torch'}} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr


def test_historical_record_names_reconstruct_canonical_classes_without_pandas(tmp_path):
    for legacy_first in (True, False):
        script = f"""
import importlib
import importlib.abc
import pickle
import sys
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'compat')!r}]
class NoPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pandas':
            raise AssertionError('record compatibility import initialized pandas')
sys.meta_path.insert(0, NoPandas())
legacy_name = 'syllable_generation.datamodel'
canonical_name = 'fpcode.contracts.session_record'
first, second = (legacy_name, canonical_name) if {legacy_first!r} else (canonical_name, legacy_name)
one, two = importlib.import_module(first), importlib.import_module(second)
assert one is two
for name in ('SessionRecord', 'VideoRef'):
    historical_class = pickle.loads(('c' + legacy_name + '\\n' + name + '\\n.').encode())
    assert historical_class is getattr(two, name)
record = two.SessionRecord('sample', video_refs=[two.VideoRef('example.avi', fps=30.)])
historical_payload = pickle.dumps(record, protocol=0).replace(canonical_name.encode(), legacy_name.encode())
loaded = pickle.loads(historical_payload)
assert type(loaded) is two.SessionRecord
assert type(loaded.video_refs[0]) is two.VideoRef
assert loaded.session_id == 'sample' and loaded.primary_fps() == 30.
assert 'pandas' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr


def test_documented_checkout_paths_expose_legacy_aliases_outside_repo(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in (
        ROOT / "src", ROOT, ROOT / "compat", ROOT / "vendor/keypoint_moseq_syll"))
    for name in ("FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    script = f"""
import importlib.util
import sys
import warnings
from pathlib import Path
warnings.simplefilter('ignore', DeprecationWarning)
root = Path({str(ROOT)!r})
for name in ('syllable_generation', 'fiberphotometry'):
    spec = importlib.util.find_spec(name)
    assert spec is not None and Path(spec.origin) == root / 'compat' / name / '__init__.py'
core_spec = importlib.util.find_spec('core')
assert Path(core_spec.origin) == root / 'compat/core/__init__.py'
pipeline_spec = importlib.util.find_spec('pipelines')
assert root / 'compat/pipelines' in map(Path, pipeline_spec.submodule_search_locations)
from fpcode.photometry.alignment import build_aligned_moseq_phot as canonical
from syllable_generation.analysis.replications.datta_utils import build_aligned_moseq_phot as legacy
assert legacy is canonical and legacy([], []) == []
assert Path(sys.modules[canonical.__module__].__file__) == root / 'src/fpcode/compat/photometry/frame_merge.py'
assert not {{'jax', 'keypoint_moseq', 'torch', 'numba', 'cv2'}} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and not list(tmp_path.iterdir())
