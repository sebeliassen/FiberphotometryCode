"""The latest historical VFI profile preserves coordinates and returns copies."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from studies.session_model_exports.utils.legacy_support import migrate_legacy_vfi

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.session_model_exports.utils.legacy_support'
LEGACY = 'syllable_generation.analysis.utils.legacy_support'


@pytest.mark.parametrize('dtype',['int16','int32','int64','float32','float64'])
def test_strided_absolute_indices_keep_values_dtype_and_independent_storage(dtype):
    backing=np.array([0,99,2,99,10,99],dtype=dtype)
    source=backing[::2]
    source.flags.writeable=False
    original=source.copy()
    inputs={'recording':source}
    output=migrate_legacy_vfi(inputs,{'recording':1000000})
    assert list(output)==['recording'] and output is not inputs
    np.testing.assert_array_equal(output['recording'],np.array([0,2,10],dtype=dtype))
    assert output['recording'].dtype==source.dtype
    assert output['recording'] is not source
    assert not np.shares_memory(output['recording'],backing)
    output['recording'][0]=7
    np.testing.assert_array_equal(source,original)


@pytest.mark.parametrize('factor',[0.5,2.,100.])
def test_scale_argument_and_large_expected_length_never_infer_or_rescale_clock(factor):
    source=np.array([0,1,4,9],dtype=np.int64)
    output=migrate_legacy_vfi({'short_trimmed_recording':source},
                              {'short_trimmed_recording':1000000},scale_factor=factor)
    np.testing.assert_array_equal(output['short_trimmed_recording'],[0,1,4,9])
    assert output['short_trimmed_recording'].dtype==np.int64


@pytest.mark.parametrize('verbose',[False,True])
def test_empty_and_unmatched_recordings_keep_copies_and_historical_log(verbose,capsys):
    empty=np.array([],dtype=np.int32)
    values=np.array([2,8],dtype=np.int32)
    output=migrate_legacy_vfi({'empty':empty,'unmatched':values},{},verbose=verbose)
    assert list(output)==['empty','unmatched']
    for name,source in [('empty',empty),('unmatched',values)]:
        assert output[name] is not source
        np.testing.assert_array_equal(output[name],source)
        assert output[name].dtype==source.dtype
    assert capsys.readouterr().out==('[Legacy Support] No legacy VFI detected.\n' if verbose else '')


def child(script,tmp_path):
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    for name in ('PYTHONPATH','FIBERPHOTOMETRY_ROOT','FP_DATA_ROOT'):
        environment.pop(name,None)
    setup=f"import sys\nsys.path[:0] = {[str(ROOT/'src'),str(ROOT),str(ROOT/'compat')]!r}\n"
    result=subprocess.run([sys.executable,'-I','-B','-c',setup+script],cwd=tmp_path,
                           env=environment,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout=='' and not list(tmp_path.iterdir())


@pytest.mark.parametrize('legacy_first',[False,True])
def test_fresh_alias_identity_and_cold_import_without_backends(legacy_first,tmp_path):
    child(f'''
import importlib
import importlib.abc
import logging
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml'}}:
            raise AssertionError('unexpected backend import: '+fullname)
sys.meta_path.insert(0,Block())
paths,handlers=list(sys.path),list(logging.root.handlers)
first,second=({LEGACY!r},{CANONICAL!r}) if {legacy_first!r} else ({CANONICAL!r},{LEGACY!r})
one,two=importlib.import_module(first),importlib.import_module(second)
assert one is two and one.migrate_legacy_vfi is two.migrate_legacy_vfi
assert one.migrate_legacy_vfi.__module__=={CANONICAL!r}
from syllable_generation.analysis.utils import legacy_support
assert legacy_support is one
assert sys.path==paths and logging.root.handlers==handlers
''',tmp_path)


def test_physical_proxy_executes_without_loading_or_writing_artifacts(tmp_path):
    path=ROOT/'compat/syllable_generation/analysis/utils/legacy_support.py'
    child(f'''
import runpy
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
runpy.run_path({str(path)!r},run_name='__main__')
assert not {{'jax','jaxlib','keypoint_moseq','torch'}} & set(sys.modules)
''',tmp_path)
