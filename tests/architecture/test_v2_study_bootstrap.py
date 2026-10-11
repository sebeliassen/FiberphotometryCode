"""Canonical V2 study entry points import without data lookup or GPU setup."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/behavior/multiview_experiments/04_reports_and_figures"
CALLERS = (
    "select_cohort_static_21kp_v2_syllable_manifest.py",
    "render_all_cohort_static_21kp_v2_seed101_grid_movies.py",
    "render_all_cohort_static_21kp_v2_seed101_random_grid_movies.py",
    "render_all_cohort_static_21kp_v2_seed101_trajectories.py",
    "build_cohort_static_21kp_v2_seed101_labeling_gallery.py",
    "render_syllable_12_random_movie.py",
    "probe_qc_policies_comparison.py",
    "probe_heading_and_kinematics.py",
    "probe_avi_tracking_pairings.py",
    "probe_active_similarity.py",
    "generate_experimental_selection_variants.py",
)


def test_canonical_v2_family_imports_use_existing_data_layout_and_actual_code_paths(tmp_path):
    # A different declared root makes accidental __file__-depth paths visible.
    # The root contains no data: imports must not inspect study inputs or write.
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import importlib.abc
import importlib.util
import runpy
from pathlib import Path
import matplotlib.pyplot  # Initialize its isolated temporary cache before guarding study code.
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax', 'jax_moseq', 'keypoint_moseq'}}:
            raise AssertionError('study bootstrap initialized ' + fullname)
sys.meta_path.insert(0, NoModel())
def no_discovery_or_output(*args, **kwargs):
    raise AssertionError('study bootstrap discovered data or wrote an output')
Path.glob = Path.rglob = Path.mkdir = Path.write_text = Path.write_bytes = no_discovery_or_output
study = Path({str(STUDY)!r})
data_root = Path({str(tmp_path)!r}) / 'syllable_generation/multiview_top_bot_experiments'
for index, name in enumerate({CALLERS!r}):
    spec = importlib.util.spec_from_file_location('v2_bootstrap_' + str(index), study / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'EXP_DIR'):
        assert module.EXP_DIR == data_root, (name, module.EXP_DIR)
        assert module.SCRIPT_DIR == study, (name, module.SCRIPT_DIR)
    if hasattr(module, 'RESULTS_PATH'):
        assert module.RESULTS_PATH == data_root / 'cohort_static_21kp_v2/model_21kp_static_50ar_300slds_seed101/results.h5'
    if hasattr(module, 'VFI_PATH'):
        assert module.VFI_PATH == data_root / 'cohort_static_21kp_v2/video_frame_indexes.npz'
    if hasattr(module, 'DEFAULT_VIZ_DIR'):
        assert module.DEFAULT_VIZ_DIR == data_root / '04_reports_and_figures/visualizations_cohort_static_21kp_v2_seed101_v4'
    if hasattr(module, 'vc'):
        assert module.vc.__name__ == 'fpcode.viz.v2'
        assert Path(module.vc.__file__).is_file()
    if name not in {{'render_syllable_12_random_movie.py', 'probe_avi_tracking_pairings.py',
                    'generate_experimental_selection_variants.py'}}:
        sys.argv = [name, '--help']
        try:
            runpy.run_path(str(study / name), run_name='__main__')
        except SystemExit as error:
            assert error.code == 0, (name, error.code)
        else:
            raise AssertionError(name + ' did not stop after help')
assert not {{'jax', 'jax_moseq', 'keypoint_moseq', 'viz_v2_common'}} & set(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
        env={**os.environ, "FIBERPHOTOMETRY_ROOT": str(tmp_path), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
