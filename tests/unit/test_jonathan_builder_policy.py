"""Cold builder aliases and independently expected synthetic IFdN source policy."""

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.contracts.session_record import SessionRecord
from studies.oft_baseline.ifdn_policy import enrich as policy

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.session_building.jonathan.build'
LEGACY = 'syllable_generation.loaders.jonathan.build'
ENRICH = 'studies.oft_baseline.ifdn_policy.enrich'
OLD_ENRICH = 'syllable_generation.projects.jonathan_openfield_example.enrich'


def test_preserved_source_policy_matches_pinned_originals():
    # Original policy blobs from 969c893b8b6ba7ddd42af178f36aff3406f84cea.
    expected = (
        ('animals.csv', 432, '5c47448b92171f4e10afd504dc340fa0f33a2588d05de556a76b3cf36f059c84'),
        ('animals_old.csv', 423, '545436bb9f59a4be257d0fcfd72de845b7da8dc115847f32a73ae97c7ade542e'),
        ('exclusions.csv', 333, '4f8bf74e0aa1c8bc0464d3089632711f22fbfcf00c12ac9285f4aa3f61b1fa9b'),
        ('experiment.csv', 0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        ('reviewed_sessions.txt', 588, '16947411fb86dec2105fe8470b72cbfb4aa24f3b3d9a8fc998c5bd86ffa1a8d5'),
        ('sessions.csv', 0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        ('time_overlays.csv', 0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    )
    for name, size, sha256 in expected:
        canonical = ROOT/'studies/oft_baseline/ifdn_policy/overrides'/name
        historical = ROOT/'compat/syllable_generation/projects/jonathan_openfield_example/overrides'/name
        payload = canonical.read_bytes()
        assert payload == historical.read_bytes()
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == sha256


def child(script, tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    for name in ('PYTHONPATH', 'FIBERPHOTOMETRY_ROOT', 'FP_DATA_ROOT'):
        environment.pop(name, None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT/'src'), str(ROOT), str(ROOT/'compat')]!r}\n"
    result = subprocess.run([sys.executable, '-I', '-B', '-c', setup+script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
    return result.stdout


@pytest.mark.parametrize('legacy_first', [False, True])
def test_builder_and_policy_leaf_identity_in_fresh_cold_processes(legacy_first, tmp_path):
    assert child(f'''
import importlib
import importlib.abc
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter('ignore', DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','config','enrich','cv2','joblib','sklearn'}} or fullname.startswith('studies.behavior'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
def forbidden(*args, **kwargs):
    raise AssertionError('import attempted data loading')
pd.read_csv = pd.read_hdf = np.load = pickle.load = forbidden
paths, handlers = list(sys.path), list(logging.root.handlers)
for canonical, legacy in (({ENRICH!r},{OLD_ENRICH!r}),({CANONICAL!r},{LEGACY!r})):
    first, second = (legacy,canonical) if {legacy_first!r} else (canonical,legacy)
    one, two = importlib.import_module(first), importlib.import_module(second)
    assert one is two
    attribute = 'build' if canonical=={CANONICAL!r} else 'enrich'
    assert getattr(one,attribute) is getattr(two,attribute)
    assert getattr(one,attribute).__module__==canonical
from studies.session_building.jonathan.build import build
assert build is importlib.import_module({CANONICAL!r}).build
assert importlib.import_module({CANONICAL!r}).project_enrich is importlib.import_module({ENRICH!r})
assert sys.path==paths and logging.root.handlers==handlers
''', tmp_path) == ''


@pytest.mark.parametrize('relative', [
    'compat/syllable_generation/loaders/jonathan/build.py',
    'compat/syllable_generation/projects/jonathan_openfield_example/enrich.py',
])
def test_physical_proxies_execute_without_building_or_loading(relative, tmp_path):
    assert child(f'''
import importlib
import runpy
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
builder=importlib.import_module({CANONICAL!r})
policy=importlib.import_module({ENRICH!r})
def forbidden(*args, **kwargs):
    raise AssertionError('physical proxy executed workflow')
builder.build=policy.enrich=policy.load_animal_map=forbidden
runpy.run_path({str(ROOT)!r}+'/'+{relative!r},run_name='__main__')
''', tmp_path) == ''


def test_default_eight_stages_keep_order_and_both_enrichment_boundaries(monkeypatch):
    module = importlib.import_module(CANONICAL)
    assert module.DEFAULT_STEPS == (
        module.tracking.discover_sessions,
        module.video.attach_video_refs,
        module.fiberphotometry.attach_photometry_refs,
        module._enrich_pre,
        module.fiberphotometry.realize_photometry_columns,
        module._enrich_post,
        module.fiberphotometry.split_photometry,
        module.fiberphotometry.compute_dff_zscore,
    )
    selected = ROOT/'studies/oft_baseline/ifdn_policy/overrides'
    assert module._project_overrides_dir() == selected
    context, seen = object(), []

    def step(name):
        def controlled(ctx, sessions):
            assert ctx is context and sessions == seen
            seen.append(name)
            return seen.copy()
        return controlled

    def controlled_enrichment(ctx, sessions, *, overrides_dir):
        assert overrides_dir == selected
        return step('enrich_pre' if len(seen)==3 else 'enrich_post')(ctx, sessions)

    monkeypatch.setattr(module.project_enrich, 'enrich', controlled_enrichment)
    steps = [step('discover'), step('video'), step('photometry_refs'), module._enrich_pre,
             step('realize'), module._enrich_post, step('split'), step('compute')]
    monkeypatch.setattr(module, 'DEFAULT_STEPS', tuple(steps))
    result = module.build(context)
    assert result == ['discover','video','photometry_refs','enrich_pre',
                      'realize','enrich_post','split','compute']
    assert result is not seen


def test_explicit_steps_forward_objects_and_propagate_errors_without_defaults(monkeypatch):
    module = importlib.import_module(CANONICAL)
    context, output = object(), object()

    def forbidden(*args, **kwargs):
        raise AssertionError('explicit steps ran defaults')

    monkeypatch.setattr(module, 'DEFAULT_STEPS', (forbidden,))

    def first(ctx, sessions):
        assert ctx is context and sessions == []
        return output

    def second(ctx, sessions):
        assert ctx is context and sessions is output
        raise ValueError('fixture step failed')

    assert module.build(context, steps=[first]) is output
    with pytest.raises(ValueError, match='fixture step failed'):
        module.build(context, steps=[first,second])


def write_map(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('ID,Left Hemisphere,Right Hemisphere,Geno\nfixture,'+label+',R-NAc,het\n')


@pytest.mark.parametrize('selected', ['policy','csv_root','nested','missing'])
def test_synthetic_animal_map_precedence_and_missing_input_policy(selected, tmp_path, monkeypatch):
    policy_file = tmp_path/'policy/enrich.py'
    root = tmp_path/'csv'
    direct, nested = root/'animals.csv', root/'Jonathan_IFxDN_csvs/animals.csv'
    candidate = policy_file.parent/'overrides/animals.csv'
    monkeypatch.setattr(policy, '__file__', str(policy_file))
    if selected in ('policy','csv_root','nested'):
        write_map(nested, 'L-NESTED')
    if selected in ('policy','csv_root'):
        write_map(direct, 'L-DIRECT')
    if selected == 'policy':
        write_map(candidate, 'L-POLICY')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.csv')}
    actual = policy.load_animal_map(SimpleNamespace(csv_root=root))
    labels = {'policy':'L-POLICY','csv_root':'L-DIRECT','nested':'L-NESTED'}
    expected = {} if selected=='missing' else {
        'fixture': {'L':labels[selected],'R':'R-NAc','Geno':'het'}}
    assert actual == expected
    assert {p: p.read_bytes() for p in tmp_path.rglob('*.csv')} == before


def test_synthetic_mapping_fuzzy_headers_and_optional_csv_root(tmp_path, monkeypatch):
    selected = tmp_path/'overrides/animals.csv'
    selected.parent.mkdir()
    selected.write_text('ID,left-hemisphere,Right Hemisphere,Geno\n fixture , L-DLS , , het \n,ignored,ignored,ignored\n')
    monkeypatch.setattr(policy, '__file__', str(tmp_path/'enrich.py'))
    assert policy.load_animal_map(SimpleNamespace(csv_root=None)) == {
        'fixture': {'L':'L-DLS','R':'','Geno':'het'}}


@pytest.mark.parametrize('skip_exclusions', [False, True])
@pytest.mark.parametrize('separate_overrides', [False, True])
def test_owned_csv_npz_union_keeps_native_indices_and_unrelated_columns(skip_exclusions, separate_overrides, tmp_path, monkeypatch):
    owned = tmp_path/'policy/overrides'
    write_map(owned/'animals.csv', 'L-DLS')
    exclusions = tmp_path/'selected_exclusions' if separate_overrides else owned
    if separate_overrides:
        write_map(exclusions/'animals.csv', 'L-WRONG')
    (exclusions/'exclusions.csv').write_text(
        'ID,Region,Intervals\n'
        'fixture,DLS-L,idx:1-1;ts:30-40\n'
        'other,DLS-L,idx:0-4\n'
        'fixture,unrelated,idx:0-4\n'
        'fixture,DLS-L,bad-interval;\n')
    np.savez(exclusions/'auto_mask.npz', fixture_session=np.array([0,1,3,4]))
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(policy, '__file__', str(owned.parent/'enrich.py'))
    frame = pd.DataFrame({
        'frame_idx': [0,1,2,3,4], 'SystemTimestamp': [0.,10.,20.,30.,40.],
        'signal_0': [10.,11.,12.,13.,14.], 'signal_0_wallace': [20.,21.,22.,23.,24.],
        'signal_1': [30.,31.,32.,33.,34.], 'signal_2': [40.,41.,42.,43.,44.],
        'unrelated': [50.,51.,52.,53.,54.],
    }, index=[100,101,102,103,104])
    original = frame.copy(deep=True)
    record = SessionRecord(session_id='fixture_session', project='fixture',
                           metadata={'animal_id':'fixture', 'signal_meta': {
                               'signal_0': {'brain_reg':'DLS','hemisphere':'L'},
                               'signal_1': {'brain_reg':'DLS','hemisphere':'L'},
                               'signal_2': {'brain_reg':'NAc','hemisphere':'R'},
                           }}, photometry={'combined_filtered': frame})
    sessions = [record]
    context = SimpleNamespace(csv_root=None, cfg=SimpleNamespace())
    assert policy.enrich(context, sessions, overrides_dir=exclusions,
                         skip_exclusions=skip_exclusions) is sessions
    assert record.metadata['genotype'] == 'HET'
    assert [record.metadata[f'signal_{i}'] for i in range(4)] == ['L-DLS','L-DLS','R-NAc','R-NAc']
    expected = original.copy(deep=True)
    if not skip_exclusions:
        expected.loc[[102], ['signal_2']] = np.nan
        expected.loc[[101,102,103,104], ['signal_0','signal_0_wallace','signal_1']] = np.nan
    pd.testing.assert_frame_equal(record.photometry['combined_filtered'], expected)
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
