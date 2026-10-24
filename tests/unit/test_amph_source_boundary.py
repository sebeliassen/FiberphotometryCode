"""Pinned AMPH source, independent metadata expectations and blocked execution."""

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from studies.oft_amph import enrich as policy

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.oft_amph.enrich'
LEGACY = 'syllable_generation.projects.jonathan_openfield_amph.enrich'
SAFE = ('_val','load_animal_map','load_sessions_map','load_exclusions_map',
        'load_auto_mask_data','_get_signal_label','get_region_to_cols_map')
GATED = ('apply_auto_masks','apply_manual_exclusions','enrich')
POLICY_SHA = {
    'animals.csv': (432,'5c47448b92171f4e10afd504dc340fa0f33a2588d05de556a76b3cf36f059c84'),
    'exclusions.csv': (1197,'2c53ca321202278050ed16cb144de55c0215c2a99c6d4b4a01d4f8956ac5a545'),
    'exclusions_test_ifdn.csv': (139,'20d04b6df49dc3be72a760d43ac0b03fbc842e3ce9a791e67e87170670fa9acc'),
    'reviewed_sessions.txt': (938,'81fb840556fae5e975c9983dce0aa3dc2f5cf4160ef69496e897945439755952'),
    'sessions.csv': (1851,'65af682cdfefe302f798196ece7769522c5d5dbf7121d77fb0714a74266cb388'),
}


def test_exact_inert_source_and_retained_tables_match_independent_pinned_hashes():
    frozen = ROOT/'studies/oft_amph/historical/enrich_969c893.py.txt'
    payload = frozen.read_bytes()
    assert len(payload) == 12768 and not payload.endswith(b'\n')
    assert hashlib.sha256(payload).hexdigest() == '4fcb471c6dba2269b0017cf6e6adccb5f47cd677a2f901fbe65c20d2ff326814'
    assert not list(frozen.parent.glob('*.py'))
    original = {node.name:node for node in ast.parse(payload).body
                if isinstance(node,ast.FunctionDef)}
    active = {node.name:node for node in ast.parse(Path(policy.__file__).read_text()).body
              if isinstance(node,ast.FunctionDef)}
    assert set(original) == set(SAFE+GATED)
    for name in SAFE:
        assert ast.dump(original[name]) == ast.dump(active[name]), name
    # Preserve evidence of the incomplete original without ever executing it.
    assert not any(isinstance(node,ast.Return) for node in ast.walk(original['enrich']))
    for name in GATED:
        assert ast.dump(original[name].args) == ast.dump(active[name].args)
        assert len(active[name].body) == 1 and isinstance(active[name].body[0],ast.Raise)
    for name,(length,sha) in POLICY_SHA.items():
        current = (ROOT/'studies/oft_amph/overrides'/name).read_bytes()
        retained = (ROOT/'compat/syllable_generation/projects/jonathan_openfield_amph/overrides'/name).read_bytes()
        assert current == retained and len(current) == length
        assert hashlib.sha256(current).hexdigest() == sha


def write_animals(path, label):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('ID,left-hemisphere,Right Hemisphere,Geno\n fixture , '+label+' , R-DMS , het \n,ignored,ignored,ignored\n')


@pytest.mark.parametrize('selected',['owner','csv_root','missing'])
def test_animal_metadata_uses_owned_priority_and_exact_fallback(selected, tmp_path, monkeypatch):
    owner = tmp_path/'amph_owner'
    fallback = tmp_path/'selected_csv_root'
    owner_csv = owner/'overrides/animals.csv'
    fallback_csv = fallback/'animals.csv'
    monkeypatch.setattr(policy,'__file__',str(owner/'enrich.py'))
    if selected != 'missing':
        write_animals(fallback_csv,'L-FALLBACK')
    if selected == 'owner':
        write_animals(owner_csv,'L-OWNER')
    before = {p:p.read_bytes() for p in tmp_path.rglob('*.csv')}
    result = policy.load_animal_map(SimpleNamespace(csv_root=fallback))
    expected = {} if selected=='missing' else {'fixture':{
        'L':'L-OWNER' if selected=='owner' else 'L-FALLBACK','R':'R-DMS','Geno':'het'}}
    assert result == expected
    assert {p:p.read_bytes() for p in tmp_path.rglob('*.csv')} == before


def test_existing_invalid_owned_animal_file_does_not_select_another_source(tmp_path, monkeypatch):
    owned = tmp_path/'amph/overrides/animals.csv'
    owned.parent.mkdir(parents=True)
    owned.touch()
    fallback = tmp_path/'other'
    write_animals(fallback/'animals.csv','L-OTHER')
    monkeypatch.setattr(policy,'__file__',str(owned.parent.parent/'enrich.py'))
    with pytest.raises(pd.errors.EmptyDataError):
        policy.load_animal_map(SimpleNamespace(csv_root=fallback))


def test_optional_csv_root_and_fuzzy_value_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(policy,'__file__',str(tmp_path/'enrich.py'))
    assert policy.load_animal_map(SimpleNamespace(csv_root=None)) == {}
    row = pd.Series({'left-hemisphere':' L-DLS ','Geno':' het '})
    assert policy._val(row,'Left Hemisphere') == 'L-DLS'
    assert policy._val(row,'Geno') == 'het'
    assert policy._val(row,'missing') == ''


def test_session_metadata_keeps_strings_delay_exclusion_and_ignores_blank_ids(tmp_path):
    selected = tmp_path/'sessions.csv'
    selected.write_text(
        'SessionID,AnimalID,Drug,Dose_mg_kg,InjectionDelay_sec,Exclude\n'
        ' session-a , fixture , Amphetamine , 3.0 , 900 ,TRUE\n'
        'session-b,other,Saline,0,,False\n'
        ',ignored,ignored,ignored,ignored,True\n')
    original = selected.read_bytes()
    assert policy.load_sessions_map(tmp_path) == {
        'session-a':{'animal_id':'fixture','drug':'Amphetamine','dose':'3.0',
                     'inj_delay':'900','exclude':True},
        'session-b':{'animal_id':'other','drug':'Saline','dose':'0',
                     'inj_delay':'','exclude':False},
    }
    assert selected.read_bytes() == original


def test_missing_session_columns_keep_original_defaults(tmp_path):
    (tmp_path/'sessions.csv').write_text('SessionID\nowned-session\n')
    assert policy.load_sessions_map(tmp_path) == {'owned-session':{
        'animal_id':'','drug':'','dose':'','inj_delay':'1200','exclude':False}}


@pytest.mark.parametrize('kind',['missing','empty','header_only'])
def test_missing_or_empty_session_and_exclusion_maps_keep_original_fallbacks(kind, tmp_path):
    if kind != 'missing':
        (tmp_path/'sessions.csv').write_text('SessionID,AnimalID\n' if kind=='header_only' else '')
        (tmp_path/'exclusions.csv').write_text('SessionID,ID,Region,Intervals\n' if kind=='header_only' else '')
    assert policy.load_sessions_map(tmp_path) == {}
    assert policy.load_exclusions_map(tmp_path) == []


def test_session_and_exclusion_read_errors_keep_distinct_historical_policies(tmp_path, monkeypatch):
    (tmp_path/'sessions.csv').touch()
    (tmp_path/'exclusions.csv').touch()
    def malformed(*args,**kwargs):
        raise ValueError('synthetic malformed table')
    monkeypatch.setattr(policy.pd,'read_csv',malformed)
    assert policy.load_sessions_map(tmp_path) == {}
    with pytest.raises(ValueError,match='synthetic malformed table'):
        policy.load_exclusions_map(tmp_path)


def test_session_aware_exclusions_preserve_regions_modes_and_float_endpoints(tmp_path):
    selected = tmp_path/'exclusions.csv'
    selected.write_text(
        'SessionID,Session,ID,Region,Intervals\n'
        'session-a,,fixture,L-DLS,idx:2-4;ts:10.5-20.5;30-32;bad\n'
        ',legacy-session,,R-DMS,1-3\n'
        ',,animal-only,L-DLS,5-6\n'
        ',,,L-DLS,0-2\n'
        'skip,,,L-DLS,invalid\n'
        'skip,,,,0-2\n')
    original = selected.read_bytes()
    assert policy.load_exclusions_map(tmp_path) == [
        {'sid':'session-a','aid':'fixture','region':'L-DLS',
         'intervals':[('idx',2.,4.),('ts',10.5,20.5),('idx',30.,32.)]},
        {'sid':'legacy-session','aid':'','region':'R-DMS','intervals':[('idx',1.,3.)]},
        {'sid':'','aid':'animal-only','region':'L-DLS','intervals':[('idx',5.,6.)]},
    ]
    assert selected.read_bytes() == original


@pytest.mark.parametrize('values',[[],[1,3,299999],[400000,500000]])
def test_owned_npz_loader_returns_unchanged_coordinates_without_applying_masks(values, tmp_path):
    path = tmp_path/'auto_mask.npz'
    expected = np.array(values,dtype=np.int64)
    np.savez(path,owned_session=expected)
    original = path.read_bytes()
    loaded = policy.load_auto_mask_data(tmp_path)
    try:
        assert loaded.files == ['owned_session']
        np.testing.assert_array_equal(loaded['owned_session'],expected)
        assert loaded['owned_session'].dtype == np.int64
    finally:
        loaded.close()
    assert path.read_bytes() == original


@pytest.mark.parametrize('invalid',[False,True])
def test_missing_or_invalid_owned_npz_keeps_none_fallback(invalid, tmp_path):
    if invalid:
        (tmp_path/'auto_mask.npz').write_bytes(b'not a valid npz fixture')
    assert policy.load_auto_mask_data(tmp_path) is None


def test_hemisphere_first_labels_and_logical_signal_columns_are_amph_specific():
    frame = pd.DataFrame(columns=['frame_idx','G4','signal_0','signal_0_trace',
                                  'signal_01','signal_1','signal_2','signal_3','signal_3_ransac'])
    session = SimpleNamespace(metadata={
        'signal_meta':{'signal_0':{'brain_reg':'DLS','hemisphere':'L'},
                       'signal_1':{'brain_reg':'DLS','hemisphere':'L'},
                       'signal_2':{'brain_reg':'DMS'}},
        'signal_2':'unqualified','signal_3':'R-DMS'},
        photometry={'combined_filtered':frame})
    assert policy._get_signal_label(session,'signal_0') == 'L-DLS'
    assert policy._get_signal_label(session,'signal_2') is None
    assert policy._get_signal_label(session,'signal_3') == 'R-DMS'
    assert policy.get_region_to_cols_map(session) == {
        'L-DLS':['signal_0','signal_0_trace','signal_1'],
        'R-DMS':['signal_3','signal_3_ransac'],
    }
    assert policy.get_region_to_cols_map(SimpleNamespace(photometry={})) == {}


@pytest.mark.parametrize('name,nonempty,skip_exclusions',[
    ('enrich',False,False),('enrich',True,False),('enrich',False,True),('enrich',True,True),
    ('apply_auto_masks',False,None),('apply_auto_masks',True,None),
    ('apply_manual_exclusions',False,None),('apply_manual_exclusions',True,None),
])
def test_all_execution_apis_reject_before_data_reads_or_context_session_access(name, nonempty, skip_exclusions, monkeypatch):
    class Untouched:
        def __getattribute__(self, attribute):
            raise AssertionError('execution gate inspected context/session: '+attribute)
        def __setattr__(self, attribute, value):
            raise AssertionError('execution gate mutated context/session: '+attribute)
    def forbidden(*args,**kwargs):
        raise AssertionError('execution gate attempted a data read')
    monkeypatch.setattr(policy.pd,'read_csv',forbidden)
    monkeypatch.setattr(policy.np,'load',forbidden)
    for helper in SAFE:
        monkeypatch.setattr(policy,helper,forbidden)
    ctx,session = Untouched(),Untouched()
    assert issubclass(policy.AMPHPolicyError,RuntimeError)
    with pytest.raises(policy.AMPHPolicyError,match='(?i)(incomplete|clock)'):
        if name == 'enrich':
            policy.enrich(ctx,[session] if nonempty else [],overrides_dir=Untouched(),
                          skip_exclusions=skip_exclusions)
        elif name == 'apply_auto_masks':
            policy.apply_auto_masks(ctx,session,Untouched() if nonempty else None)
        else:
            policy.apply_manual_exclusions(ctx,session,[Untouched()] if nonempty else [],
                                            auto_mask_data=Untouched())


def child(script, tmp_path):
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',
                       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for name in ('PYTHONPATH','FIBERPHOTOMETRY_ROOT','FP_DATA_ROOT'):
        environment.pop(name,None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT/'src'),str(ROOT),str(ROOT/'compat')]!r}\n"
    guard = '''
import importlib
import importlib.abc
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter('ignore',DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','cv2','regex','config','enrich'} or fullname.startswith('studies.behavior'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
def forbidden(*args,**kwargs):
    raise AssertionError('cold import attempted data reading or execution')
pd.read_csv = pd.read_hdf = pd.read_pickle = np.load = pickle.load = pickle.loads = forbidden
paths,handlers = list(sys.path),list(logging.root.handlers)
'''
    result = subprocess.run([sys.executable,'-I','-B','-c',setup+guard+script],cwd=tmp_path,
                            env=environment,capture_output=True,text=True,timeout=20)
    assert result.returncode == 0,result.stderr
    assert result.stdout == '' and not list(tmp_path.iterdir())


@pytest.mark.parametrize('legacy_first',[False,True])
def test_cold_aliases_share_the_amph_owner_error_and_functions_not_ifdn(legacy_first, tmp_path):
    child(f'''
first,second = ({LEGACY!r},{CANONICAL!r}) if {legacy_first!r} else ({CANONICAL!r},{LEGACY!r})
one,two = importlib.import_module(first),importlib.import_module(second)
assert one is two
for name in {SAFE+GATED!r}:
    assert getattr(one,name) is getattr(two,name)
    assert getattr(one,name).__module__ == {CANONICAL!r}
assert one.AMPHPolicyError is two.AMPHPolicyError
assert one.AMPHPolicyError.__module__ == {CANONICAL!r}
assert issubclass(one.AMPHPolicyError,RuntimeError)
ifdn = importlib.import_module('studies.oft_baseline.ifdn_policy.enrich')
assert one is not ifdn and one.enrich is not ifdn.enrich
assert one._get_signal_label is not ifdn._get_signal_label
from syllable_generation.projects.jonathan_openfield_amph import enrich
assert enrich is one
assert not any(name.startswith('studies.oft_amph.historical') for name in sys.modules)
assert sys.path == paths and logging.root.handlers == handlers
''',tmp_path)


def test_physical_legacy_proxy_has_no_main_or_empirical_execution(tmp_path):
    child(f'''
import runpy
module = importlib.import_module({CANONICAL!r})
assert not hasattr(module,'main')
for name in {SAFE+GATED!r}:
    setattr(module,name,forbidden)
runpy.run_path({str(ROOT/'compat/syllable_generation/projects/jonathan_openfield_amph/enrich.py')!r},run_name='__main__')
assert sys.path == paths and logging.root.handlers == handlers
assert not any(name.startswith('studies.oft_amph.historical') for name in sys.modules)
''',tmp_path)
