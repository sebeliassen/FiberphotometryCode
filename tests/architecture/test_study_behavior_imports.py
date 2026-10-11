"""Pure moved study calculations import without discovery or output creation."""

from pathlib import Path
import subprocess
import sys
import textwrap
import pytest


def test_study_behavior_imports_work_in_fresh_interpreter_without_eager_config():
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(f"""
        import sys
        sys.path[:0] = [{str(root)!r}, {str(root / 'src')!r}]
        from studies.behavior.kinematics_ulrik.lib import aggregation, networks, micro_kinematics, motifs
        from studies.behavior.replications_datta.datta.lib import snippets, features
        from studies.behavior.replications_datta.datta.panel_1f import run_panel_1f_unified as unified
        from syllable_generation.analysis.replications.datta.lib import snippets as old_snippets
        from syllable_generation.analysis.replications.datta.lib import features as old_features
        from syllable_generation.analysis.ulrik_kinematics.lib import aggregation as old_aggregation
        assert old_snippets is snippets and old_features is features
        assert old_aggregation is aggregation
        assert callable(networks.build_transition_matrix) and callable(micro_kinematics.extract_bouts)
        assert callable(motifs.extract_ngrams) and callable(unified.detect_movement_initiations)
        assert str(unified.repo_root) == {str(root)!r}
        sys.argv = ['run_panel_1f_unified', '--help']
        try:
            unified.main()
        except SystemExit as result:
            assert result.code == 0
        else:
            raise AssertionError('--help did not exit')
        assert not any(name.endswith('datta.lib.config') for name in sys.modules)
        assert not any(name.startswith(('jax', 'keypoint_moseq', 'matplotlib', 'seaborn', 'networkx'))
                       for name in sys.modules)
        print('Pure study imports and historical identities passed')
    """)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pure study imports and historical identities passed" in result.stdout


@pytest.mark.parametrize("split_first", [False, True])
def test_historical_metrics_split_exports_keep_identity_without_import_cycles(split_first):
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(f"""
        import importlib
        import sys
        sys.path[:0] = [{str(root)!r}, {str(root / 'src')!r}, {str(root / 'syllable_generation')!r}]
        canonical_prefix = 'studies.behavior.kinematics_ulrik.lib.'
        historical_name = 'syllable_generation.analysis.ulrik_kinematics.lib.metrics'
        if {split_first!r}:
            aggregation = importlib.import_module(canonical_prefix + 'aggregation')
            labels = importlib.import_module(canonical_prefix + 'label_mapping')
            assert historical_name not in sys.modules
        metrics = importlib.import_module(historical_name)
        bare_metrics = importlib.import_module('analysis.ulrik_kinematics.lib.metrics')
        if not {split_first!r}:
            assert canonical_prefix + 'aggregation' not in sys.modules
            assert canonical_prefix + 'label_mapping' not in sys.modules
        aggregation = importlib.import_module(canonical_prefix + 'aggregation')
        labels = importlib.import_module(canonical_prefix + 'label_mapping')
        for name in ['get_group_transition_matrix', 'get_group_usage']:
            assert getattr(metrics, name) is getattr(aggregation, name)
            assert getattr(bare_metrics, name) is getattr(aggregation, name)
        for name in ['load_quantitative_map', 'get_quantitative_group_usage', 'load_manual_labels',
                     'get_manual_group_usage', 'get_manual_group_transition_matrix']:
            assert getattr(metrics, name) is getattr(labels, name)
            assert getattr(bare_metrics, name) is getattr(labels, name)
        for name in ['calculate_kinematics', 'get_syllable_instances', 'get_syllable_usage',
                     'calculate_entropy_metrics', 'calculate_duration_entropy', 'analyze_session',
                     'calculate_advanced_stats', 'get_fingerprint_data']:
            assert getattr(metrics, name).__module__ == historical_name
        assert len(metrics.__all__) == 15 and set(metrics.__all__) <= set(dir(metrics))
        try:
            metrics.unknown_api
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown attributes must remain absent')
        assert not any(name.startswith(('jax', 'keypoint_moseq', 'matplotlib', 'seaborn', 'networkx'))
                       for name in sys.modules)
        print('Metrics facade identity and import order passed')
    """)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Metrics facade identity and import order passed" in result.stdout
