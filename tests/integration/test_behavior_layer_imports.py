"""The structural framework is importable without fitting or optional backends."""

import subprocess
import sys
from pathlib import Path


def test_all_layer_namespaces_and_placeholders_import_without_backends_or_outputs(tmp_path):
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = "\n".join([
        "import importlib, pathlib, sys",
        f"sys.path.insert(0, {str(source_root)!r})",
        "before = set(pathlib.Path('.').iterdir())",
        "names = {",
        "    'observations.representations': ('extract_trajectory_features', 'compute_interval_trajectory_metrics'),",
        "    'primitives.segmentation': ('CentroidPrimitiveSegmenter',),",
        "    'episodes.staged': ('StagedEpisodeModel',),",
        "    'episodes.grammar': ('GrammarEpisodeDetector',),",
        "    'assemblies.grammar': ('GrammarAssemblyComposer',),",
        "    'context.models': ('EventContextHMM',),",
        "    'analysis.execution': ('measure_occurrence_execution', 'aggregate_execution_metrics', 'measure_phase_pacing'),",
        "}",
        "for name, expected_symbols in names.items():",
        "    module = importlib.import_module('fpcode.behavior.' + name)",
        "    assert module.IMPLEMENTATION_STATUS == 'implemented'",
        "    for sym in expected_symbols:",
        "        assert hasattr(module, sym)",
        "for name in ('unit_adapters', 'compatibility', 'analysis.unit_views', 'episodes.seqnmf'):",
        "    importlib.import_module('fpcode.behavior.' + name)",
        "assert not {'jax', 'jaxlib', 'keypoint_moseq', 'torch', 'studies'} & set(sys.modules)",
        "assert before == set(pathlib.Path('.').iterdir())",
    ])
    subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True)
