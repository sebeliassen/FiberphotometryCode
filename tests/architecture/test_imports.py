import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_lightweight_imports_do_not_load_model_backends(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import fpcode
from fpcode.kpms.runtime import BatchConfig, CONFIG, configure_jax_runtime
import fpcode.kpms.data
import fpcode.kpms.init_model
import fpcode.kpms.training
import fpcode.kpms.export
import fpcode.common.run_record
import fpcode.cli.train
import fpcode.cli.evaluate
import fpcode.cli.fuse
import fpcode.cli.render
assert BatchConfig().discrete_bs == CONFIG.discrete_bs
assert not {{'jax', 'keypoint_moseq', 'numba', 'torch', 'cv2'}} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_tracking_primitives_do_not_load_cohort_configuration_or_fusion(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import fpcode.tracking.geometry
import fpcode.tracking.quality
assert not {{'pandas', 'h5py', 'scipy', 'jax', 'keypoint_moseq',
    'fpcode.tracking.fusion', 'fpcode.tracking.static_fusion',
    'fpcode.tracking.camera_correspondence', 'fpcode.tracking.pair_matching'}} & set(sys.modules)
from fpcode.tracking import fuse_multiview_session
from fpcode.tracking.fusion import fuse_single_pair
assert fuse_multiview_session is fuse_single_pair
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
