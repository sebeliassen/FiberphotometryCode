"""Stable lazy callable alias alongside the preserved historical build-name quirk."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.session_building.jonathan'
LEGACY = 'syllable_generation.loaders.jonathan'


def child(script, tmp_path):
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',
                       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for name in ('PYTHONPATH','FIBERPHOTOMETRY_ROOT','FP_DATA_ROOT'):
        environment.pop(name,None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT/'src'),str(ROOT),str(ROOT/'compat')]!r}\n"
    guard = '''
import builtins
import importlib
import importlib.abc
import inspect
import io
import logging
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.simplefilter('ignore',DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','cv2','joblib','sklearn','config','enrich'} or fullname.startswith('studies.behavior'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
def forbidden(*args,**kwargs):
    raise AssertionError('package import or controlled call attempted data I/O')
# These saved pickle callables are used only by the explicitly owned function-
# reference test. The public readers/writers remain blocked throughout imports.
owned_dumps,owned_loads = pickle.dumps,pickle.loads
pd.read_csv = pd.read_hdf = pd.read_pickle = np.load = np.fromfile = forbidden
pd.DataFrame.to_csv = pd.DataFrame.to_pickle = pd.DataFrame.to_hdf = forbidden
np.save = np.savez = np.savez_compressed = forbidden
pickle.dump = pickle.dumps = pickle.load = pickle.loads = forbidden
real_builtin_open,real_io_open = builtins.open,io.open
def guarded_open(original):
    def opening(file,mode='r',*args,**kwargs):
        if any(flag in mode for flag in ('w','a','x','+')):
            raise AssertionError('unexpected file write: '+str(file))
        return original(file,mode,*args,**kwargs)
    return opening
builtins.open,io.open = guarded_open(real_builtin_open),guarded_open(real_io_open)
Path.mkdir = Path.write_text = Path.write_bytes = forbidden
paths,handlers = list(sys.path),list(logging.root.handlers)
'''
    result = subprocess.run([sys.executable,'-I','-B','-c',setup+guard+script],cwd=tmp_path,
                            env=environment,capture_output=True,text=True,timeout=25)
    assert result.returncode == 0,result.stderr
    assert result.stdout == '' and not list(tmp_path.iterdir())


@pytest.mark.parametrize('order',[
    'canonical_alias_first',
    'legacy_alias_first',
    'canonical_leaf_first',
    'legacy_leaf_first',
    'old_callable_then_legacy_leaf',
    'canonical_wildcard',
    'legacy_wildcard',
])
def test_exact_alias_across_supported_package_and_leaf_orders(order, tmp_path):
    child(f'''
canonical_name,legacy_name = {CANONICAL!r},{LEGACY!r}
order = {order!r}
if order in ('canonical_alias_first','legacy_alias_first'):
    package_name = legacy_name if order=='legacy_alias_first' else canonical_name
    package = importlib.import_module(package_name)
    config = importlib.import_module(package_name+'.config')
    assert canonical_name+'.build' not in sys.modules
    assert canonical_name+'.build_sessions' not in sys.modules
    assert 'build_sessions' not in vars(importlib.import_module(canonical_name))
    assert {{'BuildContext','build','build_sessions'}} <= set(dir(package))
    assert canonical_name+'.build' not in sys.modules
    alias = package.build_sessions
elif order in ('canonical_leaf_first','legacy_leaf_first'):
    leaf_name = (legacy_name if order=='legacy_leaf_first' else canonical_name)+'.build'
    leaf = importlib.import_module(leaf_name)
    package = importlib.import_module(legacy_name if order=='legacy_leaf_first' else canonical_name)
    assert 'build_sessions' not in vars(importlib.import_module(canonical_name))
    alias = package.build_sessions
elif order=='old_callable_then_legacy_leaf':
    canonical = importlib.import_module(canonical_name)
    old_callable = canonical.build
    assert callable(old_callable) and 'build_sessions' not in vars(canonical)
    leaf = importlib.import_module(legacy_name+'.build')
    legacy = importlib.import_module(legacy_name)
    assert canonical.build is old_callable
    assert legacy.build is leaf
    alias = legacy.build_sessions
else:
    package_name = canonical_name if order=='canonical_wildcard' else legacy_name
    namespace = {{}}
    exec('from '+package_name+' import *',namespace)
    assert {{'BuildContext','build','build_sessions'}} <= set(namespace)
    alias = namespace['build_sessions']
    assert namespace['build'] is alias

canonical = importlib.import_module(canonical_name)
leaf = importlib.import_module(canonical_name+'.build')
assert alias is leaf.build and canonical.build_sessions is alias
assert vars(canonical)['build_sessions'] is alias
assert alias.__module__ == canonical_name+'.build'
assert alias.__name__ == alias.__qualname__ == 'build'
signature = inspect.signature(alias)
assert list(signature.parameters) == ['ctx','steps']
assert signature.parameters['ctx'].default is inspect.Parameter.empty
assert signature.parameters['ctx'].annotation is canonical.BuildContext
assert signature.parameters['steps'].default is None
assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
assert signature.return_annotation == list[leaf.SessionRecord]
assert signature == inspect.signature(leaf.build)
if order in ('canonical_alias_first','legacy_alias_first','canonical_leaf_first','legacy_leaf_first'):
    # Requesting the new name caches only that callable; normal child-module
    # binding of the old name stays a module in these import histories.
    assert canonical.build is leaf
else:
    # Historical callable-first/wildcard behavior is deliberately preserved.
    # This characterization does not assert that the old name is stable.
    assert canonical.build is alias
legacy = importlib.import_module(legacy_name)
assert legacy.build_sessions is alias
assert legacy.BuildContext is canonical.BuildContext
assert importlib.import_module(legacy_name+'.context').BuildContext is canonical.BuildContext
assert importlib.import_module(legacy_name+'.build') is leaf
assert list(legacy.__path__) == [{str(ROOT/'compat/syllable_generation/loaders/jonathan')!r}]
assert 'build_sessions' not in vars(legacy)
assert canonical_name+'.build_sessions' not in sys.modules
assert sys.path == paths and logging.root.handlers == handlers
''',tmp_path)


def test_real_alias_runs_only_explicit_steps_and_preserves_context_flow_errors(tmp_path):
    child(f'''
from types import SimpleNamespace
package = importlib.import_module({CANONICAL!r})
legacy = importlib.import_module({LEGACY!r})
assert {CANONICAL!r}+'.build' not in sys.modules
alias = package.build_sessions
assert legacy.build_sessions is alias
cfg = SimpleNamespace(marker='owned-policy')
ctx = package.BuildContext(root=Path('/owned/no-input-root'),
                           h5_dir=Path('/owned/no-tracking'),cfg=cfg,project='fixture')
seen = []
first_output,final_output = [object()],[object(),object()]
def first(actual, sessions):
    assert actual is ctx and actual.cfg is cfg and sessions == []
    seen.append('first')
    return first_output
def second(actual, sessions):
    assert actual is ctx and sessions is first_output
    seen.append('second')
    return final_output
assert alias(ctx,steps=(first,second)) is final_output
assert seen == ['first','second']
seen.clear()
def failed(actual, sessions):
    assert actual is ctx and sessions is first_output
    seen.append('failed')
    raise ValueError('owned controlled-step failure')
try:
    legacy.build_sessions(ctx,steps=[first,failed,forbidden])
except ValueError as exc:
    assert str(exc)=='owned controlled-step failure'
else:
    raise AssertionError('step failure did not propagate')
assert seen == ['first','failed']
assert ctx.cfg is cfg and vars(cfg)=={{'marker':'owned-policy'}}
assert sys.path == paths and logging.root.handlers == handlers
''',tmp_path)


@pytest.mark.parametrize('entrypoint',[CANONICAL,LEGACY])
def test_failed_lazy_import_leaves_no_alias_cache_and_can_retry(entrypoint, tmp_path):
    child(f'''
from dataclasses import asdict
canonical_name = {CANONICAL!r}
package = importlib.import_module({entrypoint!r})
canonical = importlib.import_module(canonical_name)
config = importlib.import_module(canonical_name+'.config')
context_type,config_object = canonical.BuildContext,config.JCFG
fields = asdict(config_object)
assert canonical_name+'.build' not in sys.modules
assert {{'build','build_sessions'}} <= set(dir(package))
assert 'build_sessions' not in vars(canonical)
class FailOnce(importlib.abc.MetaPathFinder):
    enabled = True
    attempts = 0
    def find_spec(self,fullname,path=None,target=None):
        if self.enabled and fullname==canonical_name+'.build':
            self.attempts += 1
            raise ImportError('owned lazy builder dependency failure')
failure = FailOnce()
sys.meta_path.insert(0,failure)
try:
    package.build_sessions
except ImportError as exc:
    assert str(exc)=='owned lazy builder dependency failure'
else:
    raise AssertionError('lazy import did not fail')
assert failure.attempts == 1
assert 'build_sessions' not in vars(canonical)
assert 'build_sessions' not in vars(package)
assert canonical_name+'.build' not in sys.modules
assert canonical.BuildContext is context_type and config.JCFG is config_object
assert asdict(config_object)==fields
assert sys.path == paths and logging.root.handlers == handlers
failure.enabled = False
alias = package.build_sessions
leaf = importlib.import_module(canonical_name+'.build')
assert alias is leaf.build and vars(canonical)['build_sessions'] is alias
assert canonical.BuildContext is context_type and config.JCFG is config_object
assert asdict(config_object)==fields and failure.attempts==1
assert sys.path == paths and logging.root.handlers == handlers
''',tmp_path)


def test_test_created_function_reference_resolves_the_canonical_leaf_alias(tmp_path):
    child(f'''
canonical = importlib.import_module({CANONICAL!r})
legacy = importlib.import_module({LEGACY!r})
alias = canonical.build_sessions
# This is an entirely test-created reference to one known Python function,
# carried only in memory. No historical pickle or arbitrary artifact is opened.
payload = owned_dumps(alias,protocol=0)
assert payload == ('c'+{CANONICAL!r}+'.build\\nbuild\\np0\\n.').encode()
assert owned_loads(payload) is alias
assert legacy.build_sessions is alias
assert sys.path == paths and logging.root.handlers == handlers
''',tmp_path)
