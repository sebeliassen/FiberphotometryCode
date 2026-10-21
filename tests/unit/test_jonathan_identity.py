"""Canonical contracts, parsers and narrowly scoped historical reconstruction."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.session_building.jonathan.config import JCFG
from studies.session_building.jonathan.steps.tracking import discover_sessions
from studies.session_building.jonathan.utils.files import parse_h5_path, parse_video_path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'studies.session_building.jonathan'
LEGACY = 'syllable_generation.loaders.jonathan'


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
def test_cold_contracts_and_all_nonbuilder_leaf_proxies_keep_identity(legacy_first, tmp_path):
    assert child(f'''
import importlib
import importlib.abc
import logging
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','config','cv2'}} or fullname.startswith(('studies.behavior','syllable_generation.projects')):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
paths, handlers = list(sys.path), list(logging.root.handlers)
for suffix in ('config','context','utils.files','utils.time','utils.photometry',
               'steps.tracking','steps.video','steps.events','steps.fiberphotometry'):
    canonical, legacy = {CANONICAL!r}+'.'+suffix, {LEGACY!r}+'.'+suffix
    first, second = (legacy,canonical) if {legacy_first!r} else (canonical,legacy)
    one, two = importlib.import_module(first), importlib.import_module(second)
    assert one is two, suffix
    for name in ('JonathanConfig','JCFG','BuildContext','ParsedH5','parse_video_path','parse_h5_path','discover_sessions'):
        if hasattr(one,name):
            assert getattr(one,name) is getattr(two,name), (suffix,name)
for parent in ({CANONICAL!r},{LEGACY!r}):
    package = importlib.import_module(parent)
    assert package.BuildContext is importlib.import_module({CANONICAL!r}+'.context').BuildContext
    assert 'build' in dir(package)
assert {CANONICAL!r}+'.build' not in sys.modules
assert sys.path==paths and logging.root.handlers==handlers
''', tmp_path) == ''


@pytest.mark.parametrize('suffix,class_name', [('config','JonathanConfig'),
                                              ('context','BuildContext'),
                                              ('steps.tracking','ParsedH5')])
@pytest.mark.parametrize('legacy_first', [False, True])
def test_test_created_historical_class_references_reconstruct_canonical_types(suffix, class_name, legacy_first, tmp_path):
    assert child(f'''
import importlib
import pickle
import warnings
from pathlib import Path
warnings.simplefilter('ignore',DeprecationWarning)
canonical,legacy={CANONICAL!r}+'.'+{suffix!r},{LEGACY!r}+'.'+{suffix!r}
first,second=(legacy,canonical) if {legacy_first!r} else (canonical,legacy)
one,two=importlib.import_module(first),importlib.import_module(second)
cls=getattr(one,{class_name!r})
assert cls is getattr(two,{class_name!r}) and cls.__module__==canonical
if {class_name!r}=='JonathanConfig':
    original=cls(tolerance_seconds=1.25)
elif {class_name!r}=='BuildContext':
    original=cls(root=Path('/fixture/root'),h5_dir=Path('/fixture/tracks'),project='fixture')
else:
    original=cls(Path('/fixture/track.h5'),'Arena1and2','LeftTopCamVid','2026-01-20T15_07_45','upper','top_left','2026-01-20T15_07_45_top_left','left_cam')
payload=pickle.dumps(original,protocol=0)
canonical_reference=('c'+canonical+'\\n'+{class_name!r}+'\\n').encode()
historical_reference=('c'+legacy+'\\n'+{class_name!r}+'\\n').encode()
assert payload.count(canonical_reference)==1
# This deliberately manufactured pickle is wholly owned by this test. No real
# historical artifact is opened; the assertion covers only this named class/state.
restored=pickle.loads(payload.replace(canonical_reference,historical_reference))
assert type(restored) is cls
assert restored==original
assert canonical_reference in pickle.dumps(restored,protocol=0)
''', tmp_path) == ''


@pytest.mark.parametrize('camset,token,chamber,slot,index', [
    ('LeftTopCamVid','upper','top_left','left_cam',0),
    ('LeftTopCamVid','lower','bottom_left','left_cam',1),
    ('RightTopCamVid','lower','bottom_right','right_cam',2),
    ('RightTopCamVid','upper','top_right','right_cam',3),
    ('CenterTopCamVid','bottom_left','bottom_left','center_cam',1),
])
def test_filename_parsers_retain_explicit_chamber_slot_index_and_timestamp(camset, token, chamber, slot, index):
    stamp = '2026-01-20T15_07_45'
    prefix = f'T1_4.8.e.e_Arena1and2_{camset}_{stamp}_{token}'
    for kind, parser, tail in [('video',parse_video_path,'.avi'),('tracking',parse_h5_path,'DLC_fixture.h5')]:
        path = Path('/fixture')/(prefix+tail)
        result = parser(path, JCFG)
        expected = dict(arena='Arena1and2',camset=camset,ts=stamp,token=token,chamber=chamber,
                        cam_slot=slot,session_id=f'{stamp}_{chamber}',path=path)
        if kind == 'video':
            expected['ext'] = 'avi'
        assert result == expected
    assert JCFG.chamber_index(chamber) == index
    assert JCFG.default_effective_indices(index) == list(range(4*index, 4*index+4))


def test_invalid_names_remain_unparsed_and_global_defaults_stay_explicit():
    assert parse_video_path(Path('/fixture/unrelated.mp4'), JCFG) is None
    assert parse_h5_path(Path('/fixture/unrelated.h5'), JCFG) is None
    assert JCFG.combine_freqs == ('470',)
    assert JCFG.tolerance_seconds == 2.0
    assert JCFG.led_state_map == {1: ('iso','415'), 2: ('G','470'), 4: ('R','560')}


def test_discovery_uses_only_synthetic_names_and_preserves_independent_expected_rows(tmp_path):
    stamp = '2026-01-20T15_07_45'
    names = [f'Arena1and2_LeftTopCamVid_{stamp}_upperDLC_{suffix}.h5' for suffix in ('z','a')]
    names += [f'Arena3and4_RightTopCamVid_{stamp}_lowerDLC_fixture.h5', 'invalid.h5']
    for name in names:
        (tmp_path/name).touch()
    sessions = discover_sessions(SimpleNamespace(h5_dir=tmp_path,project='fixture'), [])
    assert [s.session_id for s in sessions] == [f'{stamp}_bottom_right',f'{stamp}_top_left']
    for session, chamber, slot, camset, arena, index, expected_names in [
        (sessions[0],'bottom_right','right_cam','RightTopCamVid','Arena3and4',2,[names[2]]),
        (sessions[1],'top_left','left_cam','LeftTopCamVid','Arena1and2',0,sorted(names[:2])),
    ]:
        assert session.project == 'fixture' and session.session_type == 'openfield'
        assert session.metadata == dict(timestamp=stamp,chamber=chamber,camsets_present=[camset],
                                        cam_slots_present=[slot],arenas_present=[arena],chamber_index=index)
        rows = session.tracking_refs_df.to_dict('records')
        assert [Path(row.pop('path')).name for row in rows] == expected_names
        assert rows == [dict(path_type='tracking',kind='dlc_h5',role='upper' if index==0 else 'lower',
                             camset=camset,cam_slot=slot,arena=arena,chamber=chamber,ts=stamp,
                             camera_key=f'{slot}:{chamber}',chamber_index=index)]*len(expected_names)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)
    assert all(p.stat().st_size==0 for p in tmp_path.iterdir())


def test_juvenile_policy_uses_canonical_class_without_mutating_base_singleton():
    from studies.juvenile_intruder import config
    from studies.session_building.jonathan.config import JonathanConfig

    assert type(config.JCFG) is JonathanConfig and config.JCFG is not JCFG
    assert config.VALID_FRAME_RANGE == (50000,250000)
    parsed = parse_video_path(Path('T1_4.8.e.e_Arena1_LeftTopCamVid_2026-01-20T15_07_45.avi'),config.JCFG)
    assert parsed['chamber']=='top_left' and parsed['ts']=='2026-01-20T15_07_45'
    assert JCFG.token_to_chamber('LeftTopCamVid','lower')=='bottom_left'
    assert JCFG.token_to_chamber('unknown','token')=='token'


@pytest.mark.parametrize('suffix', ['config','context','steps/tracking'])
def test_physical_contract_proxies_execute_without_cohort_work(suffix, tmp_path):
    path = ROOT/'compat/syllable_generation/loaders/jonathan'/(suffix+'.py')
    assert child(f'''
import runpy
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
runpy.run_path({str(path)!r},run_name='__main__')
assert {CANONICAL!r}+'.build' not in sys.modules
''', tmp_path) == ''
