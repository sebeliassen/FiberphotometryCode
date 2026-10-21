"""Historical utility routing and independent synthetic numerical policies."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from studies.session_model_exports.utils import frame_map_placeholder, sampling, syll_results

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.session_model_exports.utils'
LEGACY = 'syllable_generation.analysis.utils'


def child(script, tmp_path):
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    for name in ('PYTHONPATH','FIBERPHOTOMETRY_ROOT','FP_DATA_ROOT'):
        environment.pop(name,None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT/'src'),str(ROOT),str(ROOT/'compat')]!r}\n"
    result = subprocess.run([sys.executable,'-I','-B','-c',setup+script],cwd=tmp_path,
                            env=environment,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('legacy_first',[False,True])
def test_cold_utility_leaf_identity_in_both_orders(legacy_first,tmp_path):
    child(f'''
import importlib
import importlib.abc
import logging
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml'}} or fullname.startswith('studies.behavior'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
paths,handlers=list(sys.path),list(logging.root.handlers)
for leaf in ('sampling','syll_results','frame_map_placeholder'):
    canonical,legacy={CANONICAL!r}+'.'+leaf,{LEGACY!r}+'.'+leaf
    first,second=(legacy,canonical) if {legacy_first!r} else (canonical,legacy)
    one,two=importlib.import_module(first),importlib.import_module(second)
    assert one is two
    namespace={{}}
    exec('from '+{LEGACY!r}+' import '+leaf,namespace)
    assert namespace[leaf] is one
assert sys.path==paths and logging.root.handlers==handlers
''',tmp_path)


@pytest.mark.parametrize('leaf',['sampling','syll_results','frame_map_placeholder'])
def test_physical_utility_proxies_remain_inert(leaf,tmp_path):
    path=ROOT/'compat/syllable_generation/analysis/utils'/(leaf+'.py')
    child(f'''
import runpy
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
runpy.run_path({str(path)!r},run_name='__main__')
assert not {{'jax','jaxlib','keypoint_moseq','torch'}} & set(sys.modules)
''',tmp_path)


def test_sparse_densification_preserves_leading_gaps_internal_fill_and_tail_mask():
    frame=pd.DataFrame({'name':['fixture']*3,'frame_index':[100,103,107],
                        'syllable':[1,2,3],'velocity':[10.,20.,30.]})
    original=frame.copy(deep=True)
    session=SimpleNamespace(results_dict_key='fixture',
                            video_refs_df=pd.DataFrame({'role':['primary'],'n_frames':[10]}),
                            photometry={'combined_filtered':np.zeros(8)})
    aligned,=sampling.build_aligned_dfs([session],frame,{'fixture':np.array([2,4,6])})
    assert aligned.frame_index.tolist()==list(range(10))
    pd.testing.assert_series_equal(aligned.syllable,pd.Series([None,None,1,1,2,2,3,None,None,None],dtype='Int64',name='syllable'))
    np.testing.assert_allclose(aligned.velocity,[np.nan,np.nan,10.,10.,20.,20.,30.,np.nan,np.nan,np.nan],rtol=0,atol=0,equal_nan=True)
    pd.testing.assert_frame_equal(frame,original)


def test_densification_refuses_missing_session_result_mapping():
    with pytest.raises(ValueError,match='Missing data'):
        sampling.build_aligned_dfs([SimpleNamespace(results_dict_key='missing')],pd.DataFrame({'name':['other']}),{})


@pytest.mark.parametrize('strict',[False,True])
def test_matching_preserves_trial_filter_flipped_pairs_and_unmatched_policy(strict):
    sessions=[SimpleNamespace(session_id='animalA',metadata={'trial_id':'T1'}),
              SimpleNamespace(session_id='animalB',metadata={'trial_id':'T2'})]
    keys=['T1_model_animalA','T1_model_animalA_flipped','T0_other_animalB','unknown']
    mapping=syll_results.build_result_to_session_map(sessions,keys,strict=strict)
    assert mapping=={keys[0]:0,keys[1]:0}
    assert sessions[0].results_dict_key==keys[1]
    assert sessions[1].results_dict_key==('' if strict else None)


def test_matching_retains_ambiguous_session_error():
    sessions=[SimpleNamespace(session_id='animal',metadata={}),SimpleNamespace(session_id='animalA',metadata={})]
    with pytest.raises(ValueError,match='ambiguous: 2'):
        syll_results.build_result_to_session_map(sessions,['model_animalA'])


def test_matching_retains_strict_nonflipped_duplicate_error():
    with pytest.raises(ValueError,match='matched by multiple result keys'):
        syll_results.build_result_to_session_map([SimpleNamespace(session_id='animalA',metadata={})],['one_animalA','two_animalA'])


@pytest.mark.parametrize('mode,expected',[('identity',[0,1,2]),('spread',[0,4,7])])
def test_placeholder_maps_are_explicit_and_separately_named(mode,expected,tmp_path):
    mapping=frame_map_placeholder.build_placeholder_frame_index_map({'fixture':{'syllable':np.array([1,2,3])}},mode=mode,full_lengths={'fixture':8})
    np.testing.assert_array_equal(mapping['fixture'],expected)
    path=frame_map_placeholder.save_placeholder_frame_index_map(mapping,tmp_path)
    assert Path(path).name=='frame_index_map_placeholder.npz'
    assert frame_map_placeholder.is_placeholder_frame_index_map(path)
    assert not frame_map_placeholder.is_placeholder_frame_index_map('video_frame_indexes.npz')
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['fixture'],expected)


@pytest.mark.parametrize('saved_results',[False,True])
def test_result_loading_branch_and_historical_extraction_write_request_are_explicit(saved_results,tmp_path,monkeypatch):
    model_dir=tmp_path/'project/model'
    model_dir.mkdir(parents=True)
    results={'fixture':{'syllable':np.array([1,2])}}
    calls=[]

    def load_results(project,model):
        calls.append(('load',project,model))
        return results

    def load_checkpoint(project,model):
        calls.append(('checkpoint',project,model))
        return 'model','data','metadata',42

    def extract(model,metadata,project,name,*,save_results):
        calls.append(('extract',model,metadata,project,name,save_results))
        return results

    monkeypatch.setitem(sys.modules,'keypoint_moseq',SimpleNamespace(load_results=load_results,load_checkpoint=load_checkpoint,extract_results=extract))
    monkeypatch.setenv('JAX_PLATFORMS','fixture-sentinel')
    if saved_results:
        (model_dir/'results.h5').touch()
        np.savez(model_dir/'video_frame_indexes.npz',fixture=np.array([2.,4.]))
    before={p.name:p.read_bytes() for p in model_dir.iterdir()}
    loaded,indexes=syll_results.load_kpms_results_and_vfi(model_dir,use_cpu=saved_results)
    assert loaded is results
    project=str(model_dir.parent)
    assert calls==([('load',project,'model')] if saved_results else
                   [('checkpoint',project,'model'),('extract','model','metadata',project,'model',True)])
    np.testing.assert_array_equal(indexes['fixture'],[2,4] if saved_results else [0,1])
    assert indexes['fixture'].dtype.kind=='i'
    assert os.environ['JAX_PLATFORMS']==('cpu' if saved_results else 'fixture-sentinel')
    assert {p.name:p.read_bytes() for p in model_dir.iterdir()}==before


def test_actual_length_policy_uses_first_row_and_rejects_missing_metadata():
    sessions=[SimpleNamespace(session_id='fixture',video_refs_df=pd.DataFrame({'n_frames':[100,200]}))]
    assert syll_results.compute_actual_lengths_from_sessions(sessions,{'model':0})=={'model':100}
    sessions[0].video_refs_df=pd.DataFrame()
    with pytest.raises(ValueError,match='missing video_refs_df'):
        syll_results.compute_actual_lengths_from_sessions(sessions,{'model':0})
