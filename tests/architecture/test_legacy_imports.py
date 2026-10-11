"""Exercise real compatibility paths without the study test search paths."""

from pathlib import Path
import subprocess
import sys

from ops.repo import update_shims

ROOT = Path(__file__).resolve().parents[2]


def test_root_camera_and_fusion_imports_resolve_in_a_fresh_interpreter(tmp_path):
    script = f"""
import importlib
import sys
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT)!r}]
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
    assert targets["core/fusion_engine.py"] == "fpcode.tracking.fusion"
    assert targets["core/static_fusion_engine.py"] == "fpcode.tracking.static_fusion"
    assert targets["core/camera_correspondence.py"] == "fpcode.tracking.camera_correspondence"
