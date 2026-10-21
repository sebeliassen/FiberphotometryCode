"""Study-policy acceptance tests; no empirical files are loaded."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.run_layer_pilot import configured_routines, select_pilot_sessions


def test_robust_geometry_reads_training_positions_only(tmp_path, monkeypatch):
    from studies.karen_pff_vs_monomer.analysis import run_layer_pilot as pilot

    sessions = pd.DataFrame({'session_id': ['train_m', 'train_p', 'held'],
        'animal_id': ['m', 'p', 'h'], 'group': ['Monomer', 'PFF', 'Monomer'], 'condition': ['3w_PI'] * 3})
    splits = {'train': ['m', 'p'], 'validation': [], 'test': ['h']}
    selected = sessions[sessions.animal_id.eq('h')]
    cache = tmp_path / 'frame_cache'
    cache.mkdir()
    for recording in ('train_m', 'train_p'):
        np.savez(cache / f'{recording}.npz', frames=np.arange(150), labels=np.zeros(150, dtype=int))
    columns = pd.MultiIndex.from_product([['tracker'], ['nose'], ['x', 'y', 'likelihood']])
    table = pd.DataFrame(np.column_stack([np.linspace(1., 9., 150), np.linspace(2., 8., 150), np.ones(150)]), columns=columns)
    read = []

    def read_training_only(path):
        assert 'held' not in str(path)
        read.append(str(path))
        return table

    monkeypatch.setattr(pilot, '_geometry_descriptor', lambda meta, args: ('camera_setup', (0., 10., 0., 10.)))
    monkeypatch.setattr(pilot, '_fingerprint', lambda path: {'path': str(path)})
    monkeypatch.setattr(pilot.pd, 'read_hdf', read_training_only)
    geometry, keys = pilot.training_geometry_proxies(sessions, selected, splits,
        SimpleNamespace(atlas=tmp_path, fused_dir=tmp_path))
    assert len(read) == 2
    assert keys == {'held': 'camera_setup'}
    assert {row['split'] for row in geometry['camera_setup']['reference_recordings']} == {'train'}
    # Two identical 150-point records duplicate each order statistic. The
    # lower percentile interpolates .495 of the first step in the pooled sample.
    dx, dy = .495 * 8 / 149, .495 * 6 / 149
    assert geometry['camera_setup']['robust_bounds'] == pytest.approx((1 + dx, 9 - dx, 2 + dy, 8 - dy))


def test_configured_routines_respect_unresolved_flags_and_distinct_inventories():
    catalog = pd.DataFrame({'method': ['exact'] * 4, 'chain_id': ['dump', 'a', 'reversed_a', 'b'],
                            'tokens': ['0 1 0', '1 2 1', '2 1 2', '2 3 2']})
    annotations = pd.DataFrame({'syllable_id': [0, 1, 2, 3], 'category': ['S0', 'S1', 'S2', 'S3'],
                                'description': ['unknown', 'one', 'two', 'three'],
                                'unresolved': [True, False, False, False]})
    routines, _ = configured_routines(catalog, annotations)
    assert routines == {'a': ('1', '2', '1'), 'b': ('2', '3', '2')}


def test_pilot_selection_is_held_out_by_animal_and_independent_of_effect_columns():
    rows = []
    for group in ('Monomer', 'PFF'):
        for animal in range(3):
            for condition in ('3w_PI', '16w_PI', 'L_dopa'):
                rows.append({'session_id': f'{group}_{animal}_{condition}', 'animal_id': f'{group}_{animal}',
                             'group': group, 'condition': condition, 'behaviour_value': animal})
    sessions = pd.DataFrame(rows)
    splits = {'train': ['Monomer_0', 'PFF_0'], 'validation': [],
              'test': ['Monomer_1', 'Monomer_2', 'PFF_1', 'PFF_2']}
    original = select_pilot_sessions(sessions, splits)
    assert len(original) == 12
    assert original['split'].eq('test').all()
    assert original.groupby('group').animal_id.nunique().to_dict() == {'Monomer': 2, 'PFF': 2}
    changed = sessions.sample(frac=1., random_state=1).copy()
    changed['behaviour_value'] = -1000.
    assert select_pilot_sessions(changed, splits).session_id.tolist() == original.session_id.tolist()
    with pytest.raises(ValueError, match='positive'):
        select_pilot_sessions(sessions, splits, animals_per_group=0)


def test_missing_wave_is_explicit_rather_than_silently_dropping_a_recording():
    sessions = pd.DataFrame([{'session_id': f'{group}_pre', 'animal_id': group,
                             'group': group, 'condition': '3w_PI'} for group in ('Monomer', 'PFF')])
    with pytest.raises(ValueError, match='lacks pilot wave'):
        select_pilot_sessions(sessions, {'train': [], 'validation': [], 'test': ['Monomer', 'PFF']}, animals_per_group=1)
