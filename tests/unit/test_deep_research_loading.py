"""Controlled loader policy, trusted synthetic pickle discovery and cold imports."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from studies.deep_research_experiments.utils import loaders

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.deep_research_experiments.utils'
LEGACY = 'syllable_generation.analysis.deep_research_experiments.utils'


def child(script, tmp_path, *, mirror=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    for name in ('PYTHONPATH', 'FIBERPHOTOMETRY_ROOT', 'FP_DATA_ROOT'):
        environment.pop(name, None)
    if mirror is not None:
        environment['FP_DATA_ROOT'] = str(mirror)
    code = f"import sys\nsys.path[:0] = {[str(ROOT/'src'), str(ROOT), str(ROOT/'compat')]!r}\n" + script
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize('project', ['IFDN', 'aMPh'])
@pytest.mark.parametrize('apply_filter', [False, True])
def test_loading_retains_stage_order_object_flow_and_five_minute_policy(project, apply_filter, monkeypatch):
    seen = []
    loaded, enriched, filtered, post = [object() for _ in range(4)]
    selected = Path('/explicit/trusted/session.pkl')

    def base(*, project, pkl_path):
        seen.append(('load', project, pkl_path))
        return loaded

    def enrich(value):
        assert value is loaded
        seen.append(('enrich',))
        return enriched

    def signal_filter(value):
        assert value is enriched
        seen.append(('filter',))
        return filtered

    def post_injection(value, *, delay_mins):
        assert value is (filtered if apply_filter else enriched)
        seen.append(('post', delay_mins))
        return post

    monkeypatch.setattr(loaders.base_loader, 'load_sessions', base)
    monkeypatch.setattr(loaders.base_processing, 'enrich_sessions', enrich)
    monkeypatch.setattr(loaders.base_processing, 'apply_filter_to_sessions', signal_filter)
    monkeypatch.setattr(loaders.base_processing, 'filter_post_injection', post_injection)
    result = loaders.load_sessions(project, apply_filter, pkl_path=selected)
    expected = [('load', project.lower(), selected), ('enrich',)]
    if apply_filter:
        expected.append(('filter',))
    if project.lower() == 'amph':
        expected.append(('post', 5.0))
    assert seen == expected
    assert result is (post if project.lower() == 'amph' else filtered if apply_filter else enriched)


@pytest.mark.parametrize('legacy_first', [False, True])
def test_fresh_imports_keep_leaf_identity_without_loading_or_global_setup(legacy_first, tmp_path):
    script = f'''
import importlib
import importlib.abc
import logging
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','loader','processing'}}:
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
base = importlib.import_module('studies.replications_datta.panel_1z.lib.loader')
def forbidden(*args, **kwargs):
    raise AssertionError('import attempted cohort loading')
base.load_sessions = forbidden
paths = list(sys.path)
handlers = list(logging.root.handlers)
for name in ('loaders','check_project_attr','summarize_regions'):
    canonical, legacy = {CANONICAL!r}+'.'+name, {LEGACY!r}+'.'+name
    first, second = (legacy, canonical) if {legacy_first!r} else (canonical, legacy)
    one, two = importlib.import_module(first), importlib.import_module(second)
    assert one is two
    attr = 'load_sessions' if name=='loaders' else 'main'
    assert getattr(one,attr) is getattr(two,attr)
    assert getattr(one,attr).__module__==canonical
    namespace={{}}
    exec('from '+{LEGACY!r}+' import '+name,namespace)
    assert namespace[name] is one
importlib.import_module('studies.deep_research_experiments.action_prediction_error.run_ppe_analysis')
assert sys.path == paths and logging.root.handlers == handlers
'''
    assert child(script, tmp_path) == ''
    assert not list(tmp_path.iterdir())


def write_sessions(path, label):
    valid = SimpleNamespace(session_id=label, all_data_df=pd.DataFrame({'value': [1., 2.]}),
                            metadata={'signal_meta': {'s': {'brain_reg': 'l_dls', 'hemisphere': 'LEFT'}}})
    empty = SimpleNamespace(session_id='empty', all_data_df=pd.DataFrame(), metadata={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps([valid, empty]))


@pytest.mark.parametrize('fallback', [False, True])
def test_mirror_discovery_uses_historical_priority_latest_and_metadata_policy(fallback, tmp_path):
    mirror = tmp_path/'mirror'
    data = mirror/'data'
    generic = data/'sessions_all_data.pkl'
    write_sessions(generic, 'generic')
    os.utime(generic, (3000, 3000))
    if not fallback:
        for tick in (1000, 2000):
            path = data/f'jonathan_amph_sessions_{tick}_w_sylls.pkl'
            write_sessions(path, str(tick))
            os.utime(path, (tick, tick))
    wrong = data/'jonathan_ifdn_sessions_new_w_sylls.pkl'
    write_sessions(wrong, 'wrong-project')
    os.utime(wrong, (4000, 4000))
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    outside = tmp_path/'elsewhere'
    outside.mkdir()
    output = child('''
import json
from studies.replications_datta.panel_1z.lib.loader import load_sessions
sessions = load_sessions(project='amph')
assert len(sessions)==1
s = sessions[0]
assert s.metadata['signal_meta']['s']=={'brain_reg':'L-DLS','hemisphere':'L'}
assert s.all_data_df.value.tolist()==[1.,2.]
print(json.dumps(s.session_id))
''', outside, mirror=mirror)
    assert json.loads(output) == ('generic' if fallback else '2000')
    assert before == {p.name: p.read_bytes() for p in data.iterdir()}
    assert not list(outside.iterdir())


def test_explicit_pickle_wins_and_missing_explicit_never_falls_back(tmp_path):
    selected = tmp_path/'selected.pkl'
    write_sessions(selected, 'explicit')
    before = selected.read_bytes()
    mirror = tmp_path/'mirror'
    write_sessions(mirror/'data/sessions_all_data.pkl', 'fallback-must-not-load')
    outside = tmp_path/'elsewhere'
    outside.mkdir()
    script = f'''
from studies.replications_datta.panel_1z.lib.loader import load_sessions
assert load_sessions(project='amph',pkl_path={str(selected)!r})[0].session_id=='explicit'
try:
    load_sessions(project='amph',pkl_path={str(tmp_path/'missing.pkl')!r})
except FileNotFoundError:
    pass
else:
    raise AssertionError('missing explicit input must not fall back')
'''
    assert child(script, outside, mirror=mirror) == ''
    assert selected.read_bytes() == before and not list(outside.iterdir())


@pytest.mark.parametrize('name', ['check_project_attr', 'summarize_regions'])
@pytest.mark.parametrize('legacy_file', [False, True])
def test_diagnostic_module_and_physical_proxy_execute_only_when_requested(name, legacy_file, tmp_path):
    target = ROOT/'compat/syllable_generation/analysis/deep_research_experiments/utils'/f'{name}.py'
    script = '''
import runpy
import warnings
from types import SimpleNamespace
warnings.simplefilter('ignore', DeprecationWarning)
from studies.deep_research_experiments.utils import loaders
calls=[]
def fake(project, *, apply_filter):
    assert apply_filter is False
    calls.append(project)
    return [SimpleNamespace(session_id='fixture',project=project,metadata={'drug':'Saline','signal_meta':{'s':{'brain_reg':'l-dls'}}})]
loaders.load_sessions=fake
'''
    if legacy_file:
        script += f"runpy.run_path({str(target)!r},run_name='__main__')\n"
    else:
        script += f"runpy.run_module({(CANONICAL+'.'+name)!r},run_name='__main__')\n"
    script += "assert calls==['ifdn','amph']\n"
    output = child(script, tmp_path)
    assert 'IFDN' in output and 'AMPH' in output
    if name == 'check_project_attr':
        assert 'session_id: fixture' in output and "drug metadata: 'Saline'" in output
    else:
        assert output.count("('L-DLS',): 1 sessions") == 2
    assert not list(tmp_path.iterdir())


def test_analysis_help_stays_cold(tmp_path):
    output = child('''
import runpy
from studies.replications_datta.panel_1z.lib import loader

def forbidden(*args, **kwargs):
    raise AssertionError('help attempted cohort loading')
loader.load_sessions = forbidden
sys.argv=['run_ppe_analysis','--help']
try:
    runpy.run_module('studies.deep_research_experiments.action_prediction_error.run_ppe_analysis',run_name='__main__')
except SystemExit as exc:
    assert exc.code==0
else:
    raise AssertionError('argparse help should exit')
assert not {'jax','jaxlib','keypoint_moseq','torch'} & set(sys.modules)
''', tmp_path)
    assert 'usage:' in output and '--help' in output
    assert not list(tmp_path.iterdir())
