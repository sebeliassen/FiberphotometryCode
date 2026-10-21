"""Unit test for synthetic behavior layers comparison workflow."""


from fpcode.workflows.behavior_layers import (
    run_synthetic_behavior_comparison,
)


def test_frozen_empirical_workflow_preserves_features_boundaries_and_never_fits(monkeypatch):
    import numpy as np
    import pytest

    from fpcode.behavior.episodes.seqnmf import SeqNMF
    from fpcode.behavior.unit_adapters import from_frame_labels
    from fpcode.contracts import SessionKey, SignalMetadata, SignalSeries, TimeAxis, UnitType
    from fpcode.contracts.behavior_results import FeatureAssociation
    from fpcode.workflows.behavior_layers import compare_frozen_episode_methods

    def forbidden_fit(*args, **kwargs):
        raise AssertionError('held-out workflow must not fit')

    monkeypatch.setattr(SeqNMF, 'fit', forbidden_fit)
    session = SessionKey('native_recording', 'mouse')
    frames = np.array([100, 101, 102, 103, 110, 111, 112, 113])
    stream = from_frame_labels(frames, np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                               session=session, model_id='kpms', clock='video', fps=10.,
                               catalog=(UnitType('0'), UnitType('1')))
    series = SignalSeries(TimeAxis(frames, 'frame', 'video'), np.arange(8.)[:, None],
                           (SignalMetadata('distance_center', 'px'),))
    association = FeatureAssociation(series, session, 'arena')
    templates = np.array([[[1., 0.]], [[0., 1.]]], dtype=np.float32)
    before = templates.copy()
    output = compare_frozen_episode_methods(
        stream, association, exact_motifs={'cycle': ('0', '1')},
        seqnmf_templates=templates, seqnmf_type_ids=('0', '1'), seqnmf_thresholds=np.array([.1]),
        seqnmf_penalty=0.,
    )
    for method in ('exact', 'seqnmf'):
        result = output['results'][method]
        assert result.features is association
        assert result.stream.session == session
        assert result.stream.valid_intervals == stream.valid_intervals
        assert [(o.interval.start, o.interval.stop) for o in result.occurrences] == [(100, 104), (110, 114)]
    np.testing.assert_array_equal(templates, before)
    assert output['results']['primitives'].stream is stream
    assert output['seqnmf_events'].template_error.max() < 1e-6
    with pytest.raises(ValueError, match='vocabulary'):
        compare_frozen_episode_methods(stream, association, exact_motifs={},
            seqnmf_templates=templates, seqnmf_type_ids=('1', '0'), seqnmf_thresholds=np.array([.1]))
    with pytest.raises(ValueError, match='budget'):
        compare_frozen_episode_methods(stream, association, exact_motifs={},
            seqnmf_templates=templates, seqnmf_type_ids=('0', '1'), seqnmf_thresholds=np.array([.1]),
            memory_budget_bytes=1)
    foreign = FeatureAssociation(series, SessionKey('other', 'mouse'), 'arena')
    with pytest.raises(ValueError, match='recording'):
        compare_frozen_episode_methods(stream, foreign, exact_motifs={},
            seqnmf_templates=templates, seqnmf_type_ids=('0', '1'), seqnmf_thresholds=np.array([.1]))


def test_synthetic_workflow_executes_and_compares_all_layers():
    res = run_synthetic_behavior_comparison(seed=42, include_context=True)
    assert "primitives" in res
    assert "staged_result" in res
    assert "grammar_result" in res
    assert "exact_stream" in res
    assert "seqnmf_stream" in res
    assert "context_result" in res
    assert "execution_metrics" in res
    assert "organisation_metrics" in res

    # Check that execution metrics table has expected keys
    keys = {"primitives", "primitives_coarse", "staged", "grammar", "exact", "seqnmf", "assembly"}
    assert set(res["execution_metrics"].keys()) == keys

    # Context off option
    res_no_ctx = run_synthetic_behavior_comparison(seed=42, include_context=False)
    assert res_no_ctx["context_result"] is None


def test_resolution_changes_units_but_preserves_features_and_common_consumers():
    import numpy as np

    from fpcode.behavior.analysis.execution import measure_occurrence_execution
    from fpcode.behavior.primitives.resolutions import remap_states
    from fpcode.behavior.unit_adapters import from_frame_labels
    from fpcode.contracts import SessionKey, SignalMetadata, SignalSeries, TimeAxis, UnitType
    from fpcode.contracts.behavior_results import FeatureAssociation
    from fpcode.workflows.behavior_layers import compare_frozen_episode_methods

    frames = np.arange(100, 110)
    raw = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0])
    session = SessionKey('recording', 'animal')
    features = FeatureAssociation(SignalSeries(TimeAxis(frames, 'frame', 'video'), np.arange(10.)[:, None],
        (SignalMetadata('speed', 'px/s'),)), session, 'topcam')
    outputs = []
    for mapping, motif in (([0, 1, 2], ('0', '1', '2')), ([0, 0, 1], ('0', '1'))):
        labels = remap_states(raw, mapping)
        catalog = tuple(UnitType(str(i)) for i in range(max(mapping) + 1))
        primitives = from_frame_labels(frames, labels, session=session, model_id=str(mapping),
            clock='video', fps=10., catalog=catalog)
        template = np.zeros((len(catalog), 1, len(motif)), dtype=np.float32)
        for lag, token in enumerate(motif):
            template[int(token), 0, lag] = 1.
        output = compare_frozen_episode_methods(primitives, features, exact_motifs={'anchor': motif},
            seqnmf_templates=template, seqnmf_type_ids=tuple(u.type_id for u in catalog),
            seqnmf_thresholds=np.array([.1]), seqnmf_penalty=0.)
        assert output['results']['exact'].features is features
        outputs.append(output)
    assert len(outputs[0]['results']['primitives'].occurrences) == 5
    assert len(outputs[1]['results']['primitives'].occurrences) == 3
    first, second = [measure_occurrence_execution(output['results']['exact']) for output in outputs]
    assert first.columns.tolist() == second.columns.tolist()
    assert first.mean_feature.iloc[0] == second.mean_feature.iloc[0] == 2.5
    assert outputs[0]['results']['primitives'].valid_intervals == outputs[1]['results']['primitives'].valid_intervals


def test_context_is_optional_and_does_not_refit_lower_evidence_or_composition():
    import numpy as np
    on = run_synthetic_behavior_comparison(seed=42,include_context=True)
    off = run_synthetic_behavior_comparison(seed=42,include_context=False)
    np.testing.assert_array_equal(on['features'].values,off['features'].values)
    np.testing.assert_array_equal(on['features'].valid,off['features'].valid)
    for name in ('primitives','primitives_coarse','staged_result','grammar_result','exact_stream','seqnmf_stream','assembly_result'):
        assert on[name].occurrences == off[name].occurrences
        assert on[name].catalog == off[name].catalog
        assert on[name].valid_intervals == off[name].valid_intervals
        assert getattr(on[name],'derivations',None) == getattr(off[name],'derivations',None)
        assert getattr(on[name],'phases',None) == getattr(off[name],'phases',None)
        assert len(on[name].occurrences) > 0
    assert all(len(d.child_references) == 2 for d in on['assembly_result'].derivations)
    assert on['phase_profiles'].exposure_s.sum() > 0
    lower = {o.occurrence_id: o for o in on['primitives'].occurrences}
    for d in on['grammar_result'].derivations:
        for ref in d.child_references:
            assert ref.session == on['primitives'].session
            assert ref.model_id == on['primitives'].model_id
            assert ref.occurrence_id in lower
            child = lower[ref.occurrence_id]
            assert d.interval.start <= child.interval.start < child.interval.stop <= d.interval.stop
    assert on['primitives'].catalog != on['primitives_coarse'].catalog
