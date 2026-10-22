"""Audited project ownership, assembled pipeline flow and temporary I/O boundaries."""

import importlib
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    'jonathan_full': ('studies.oft_baseline.ifdn_policy.enrich', None,
                      'studies/oft_baseline/ifdn_policy/overrides', 'h5'),
    'jonathan_openfield_example': ('studies.oft_baseline.ifdn_policy.enrich', None,
                                   'studies/oft_baseline/ifdn_policy/overrides', 'h5'),
    'jonathan_oft_baseline': ('studies.oft_baseline.enrich', 'studies.oft_baseline.config',
                              'studies/oft_baseline/overrides', 'hybrid'),
    'jonathan_juvenile_intruder': ('studies.juvenile_intruder.enrich', 'studies.juvenile_intruder.config',
                                  'studies/juvenile_intruder/overrides', 'hybrid'),
    'karen_project': ('studies.karen_pff_vs_monomer.analysis.enrich',
                      'studies.karen_pff_vs_monomer.session_policy.config',
                      'studies/karen_pff_vs_monomer/session_policy/overrides', 'hybrid'),
}
EXPECTED_FLOW = ['discover', 'video', 'photometry_refs', 'enrich_pre',
                 'realize', 'enrich_post', 'split', 'compute']


@pytest.fixture
def owners():
    """Preload policy/context/stages so stale imported singleton bugs are exposed."""
    return SimpleNamespace(
        driver=importlib.import_module('studies.session_building.build_sessions_unified'),
        builder=importlib.import_module('studies.session_building.jonathan.build'),
        config=importlib.import_module('studies.session_building.jonathan.config'),
        context=importlib.import_module('studies.session_building.jonathan.context'),
        tracking=importlib.import_module('studies.session_building.jonathan.steps.tracking'),
        video=importlib.import_module('studies.session_building.jonathan.steps.video'),
        photometry=importlib.import_module('studies.session_building.jonathan.steps.fiberphotometry'),
        baseline_tracking=importlib.import_module('studies.monoamine_interactions.steps.tracking'),
        karen_tracking=importlib.import_module('studies.karen_pff_vs_monomer.analysis.tracking'),
        enrichments={name:importlib.import_module(spec[0]) for name,spec in PROJECTS.items()},
        policies={spec[1]:importlib.import_module(spec[1]) for spec in PROJECTS.values() if spec[1]},
    )


@pytest.fixture
def pipeline(owners, monkeypatch, tmp_path):
    """Execute the actual assembled list with test-owned discovery/processing only."""
    harness = SimpleNamespace(owners=owners, calls=[], active=None, failure=None, current=None)

    def event(label, ctx, sessions):
        current = harness.current
        assert ctx is current['ctx'] and sessions is current['previous']
        assert current['flow'] == EXPECTED_FLOW[:len(current['flow'])]
        if current['original_output'] is None:
            assert not harness.active['output'].exists()
        else:
            assert harness.active['output'].read_bytes() == current['original_output']
        if label == harness.failure:
            raise RuntimeError('injected '+label)
        current['flow'].append(label)
        if label == 'discover':
            result = [SimpleNamespace(metadata={'animal_id':728}),
                      SimpleNamespace(metadata={'animal_id':'other'})]
        else:
            result = list(sessions)
        current['previous'] = result
        return result

    def stage(label):
        def controlled(ctx, sessions):
            return event(label, ctx, sessions)
        return controlled

    for module,name,label in [
        (owners.tracking,'discover_sessions','discover'),
        (owners.baseline_tracking,'discover_sessions_hybrid','discover'),
        (owners.karen_tracking,'discover_sessions_hybrid','discover'),
        (owners.video,'attach_video_refs','video'),
        (owners.photometry,'attach_photometry_refs','photometry_refs'),
        (owners.photometry,'realize_photometry_columns','realize'),
        (owners.photometry,'split_photometry','split'),
        (owners.photometry,'compute_dff_zscore','compute'),
    ]:
        monkeypatch.setattr(module,name,stage(label))

    def enrichment(module_name):
        def controlled(ctx, sessions, *, overrides_dir, skip_exclusions):
            spec = PROJECTS[harness.active['project']]
            assert module_name == spec[0]
            assert overrides_dir == ROOT/spec[2]
            assert skip_exclusions is harness.active['force_dirty']
            label = 'enrich_pre' if len(harness.current['flow'])==3 else 'enrich_post'
            return event(label,ctx,sessions)
        return controlled

    for module in set(owners.enrichments.values()):
        monkeypatch.setattr(module,'enrich',enrichment(module.__name__))

    def controlled_build(ctx, *, steps):
        selected = harness.active
        spec = PROJECTS[selected['project']]
        assert ctx.project == selected['project']
        assert ctx.root == selected['inputs'].resolve()
        assert ctx.h5_dir == (selected['inputs']/'h5_files').resolve()
        assert ctx.video_dir == (selected['inputs']/'video_files').resolve()
        assert ctx.csv_root == (selected['inputs']/'csvs').resolve()
        if spec[1] is None:
            assert type(ctx.cfg) is owners.config.JonathanConfig
            assert asdict(ctx.cfg) == asdict(owners.config.JonathanConfig())
            assert all(ctx.cfg is not module.JCFG for module in owners.policies.values())
            assert ctx.cfg is not owners.config.JCFG
        else:
            assert ctx.cfg is owners.policies[spec[1]].JCFG

        mode = selected['mode'] or spec[3]
        if mode == 'h5':
            discover = owners.tracking.discover_sessions
        elif ctx.project == 'karen_project':
            discover = owners.karen_tracking.discover_sessions_hybrid
        else:
            discover = owners.baseline_tracking.discover_sessions_hybrid
        has_flip_hook = spec[1] and hasattr(owners.policies[spec[1]],'OPTIMAL_FLIP_IDS')
        assert len(steps) == (9 if has_flip_hook else 8)
        assert steps[:3] == [discover,owners.video.attach_video_refs,
                             owners.photometry.attach_photometry_refs]
        assert callable(steps[3]) and callable(steps[5])
        assert steps[4] is owners.photometry.realize_photometry_columns
        assert steps[-2:] == [owners.photometry.split_photometry,
                              owners.photometry.compute_dff_zscore]
        output = selected['output']
        current = dict(ctx=ctx,steps=steps,flow=[],previous=[],
                       original_output=output.read_bytes() if output.exists() else None)
        harness.current = current
        harness.calls.append(current)
        sessions = current['previous']
        for step in steps:
            sessions = step(ctx,sessions)
        assert current['flow'] == EXPECTED_FLOW
        current['result'] = sessions
        return sessions

    monkeypatch.setattr(owners.builder,'build',controlled_build)

    def call(project, *, mode=None, force_dirty=None, output=None, **kwargs):
        output = output or tmp_path/f'output_{len(harness.calls)}/sessions.pkl'
        inputs = tmp_path/'inputs'
        harness.active = dict(project=project,mode=mode,
                              force_dirty=True if force_dirty is None else force_dirty,
                              output=output,inputs=inputs)
        if force_dirty is not None:
            kwargs['force_dirty'] = force_dirty
        return owners.driver.build_sessions(
            data_root=inputs,h5_dir=inputs/'h5_files',video_dir=inputs/'video_files',
            csv_root=inputs/'csvs',output_pickle=output,project=project,
            discovery_mode=mode,**kwargs)

    harness.call = call
    return harness


@pytest.mark.parametrize('project',list(PROJECTS))
@pytest.mark.parametrize('mode',[None,'h5','hybrid'])
@pytest.mark.parametrize('force_dirty',[False,True])
def test_audited_projects_keep_owned_policy_overrides_discovery_and_eight_stages(project, mode, force_dirty, pipeline):
    result = pipeline.call(project,mode=mode,force_dirty=force_dirty)
    assert result is pipeline.calls[-1]['result']
    # Only this test-created output is unpickled; no historical artifact is read.
    output = pipeline.active['output']
    assert pickle.loads(output.read_bytes()) == result
    assert pipeline.calls[-1]['flow'] == EXPECTED_FLOW


def state_snapshot(owners):
    modules = [owners.config,owners.context,owners.tracking,owners.video,
               owners.photometry,owners.baseline_tracking,owners.karen_tracking]
    modules.extend(owners.policies.values())
    values = [(module,module.JCFG,asdict(module.JCFG)) for module in modules]
    foreign = {}
    for name in ('config','enrich','tracking','steps','steps.tracking'):
        module = sys.modules.get(name)
        foreign[name] = (name in sys.modules,module,dict(vars(module)) if module else None)
    return values,list(sys.path),list(logging.root.handlers),foreign


def assert_state_unchanged(owners, snapshot):
    values,paths,handlers,foreign = snapshot
    for module,original,fields in values:
        assert module.JCFG is original and asdict(module.JCFG)==fields
    assert sys.path == paths and logging.root.handlers == handlers
    for name,(present,module,namespace) in foreign.items():
        assert (name in sys.modules) is present
        assert sys.modules.get(name) is module
        if module is not None:
            assert vars(module) == namespace


@pytest.fixture
def foreign_modules(monkeypatch, tmp_path):
    def forbidden(name):
        raise AssertionError('foreign module attribute accessed: '+name)
    foreign = {}
    for name in ('config','enrich','tracking','steps','steps.tracking'):
        module = ModuleType(name)
        module.sentinel = object()
        module.__getattr__ = forbidden
        monkeypatch.setitem(sys.modules,name,module)
        foreign[name] = (module,dict(vars(module)))
    for name in ('config','enrich','tracking','steps'):
        (tmp_path/(name+'.py')).write_text("raise AssertionError('foreign bare module imported')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return foreign


@pytest.mark.parametrize('reverse',[False,True])
def test_sequential_cohorts_and_fresh_ifdn_configs_do_not_leak_globals(pipeline, foreign_modules, reverse):
    owners = pipeline.owners
    before = state_snapshot(owners)
    projects = ['jonathan_full','jonathan_oft_baseline','jonathan_juvenile_intruder','karen_project']
    if reverse:
        projects.reverse()
    projects.extend(['jonathan_openfield_example','jonathan_full'])
    for project in projects:
        pipeline.call(project)
        assert_state_unchanged(owners,before)
    configs = [item['ctx'].cfg for item in pipeline.calls
               if PROJECTS[item['ctx'].project][1] is None]
    assert len({id(cfg) for cfg in configs}) == len(configs)
    for module,namespace in foreign_modules.values():
        assert vars(module) == namespace


@pytest.mark.parametrize('failure',EXPECTED_FLOW)
def test_each_pipeline_failure_preserves_globals_foreign_modules_and_unwritten_output(pipeline, foreign_modules, failure):
    before = state_snapshot(pipeline.owners)
    pipeline.failure = failure
    with pytest.raises(RuntimeError,match='injected '+failure):
        pipeline.call('karen_project',force_dirty=False)
    assert not pipeline.active['output'].parent.exists()
    assert_state_unchanged(pipeline.owners,before)
    pipeline.failure = None
    pipeline.call('jonathan_oft_baseline')
    assert_state_unchanged(pipeline.owners,before)


@pytest.mark.parametrize('failed_module',[
    'studies.karen_pff_vs_monomer.analysis.enrich',
    'studies.karen_pff_vs_monomer.session_policy.config',
])
def test_project_dependency_failure_propagates_without_fallback_or_output(pipeline, foreign_modules, monkeypatch, failed_module):
    driver = pipeline.owners.driver
    original = importlib.import_module
    before = state_snapshot(pipeline.owners)
    def controlled(name, *args, **kwargs):
        if name == failed_module:
            raise ImportError('fixture project dependency failure')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(driver.importlib,'import_module',controlled)
    with pytest.raises(ImportError,match='fixture project dependency failure'):
        pipeline.call('karen_project')
    assert not pipeline.calls and not pipeline.active['output'].parent.exists()
    assert_state_unchanged(pipeline.owners,before)


@pytest.mark.parametrize('project',['unknown','jonathan_openfield_amph','jonathan_flipped','Jonathan_full'])
def test_unknown_or_unaudited_project_fails_before_building(pipeline, project):
    before = state_snapshot(pipeline.owners)
    with pytest.raises(ValueError):
        pipeline.call(project)
    assert not pipeline.calls and not pipeline.active['output'].parent.exists()
    assert_state_unchanged(pipeline.owners,before)


def test_invalid_api_discovery_mode_fails_before_stages_or_output(pipeline):
    before = state_snapshot(pipeline.owners)
    with pytest.raises(ValueError):
        pipeline.call('jonathan_full',mode='unrecognized')
    assert not pipeline.calls and not pipeline.active['output'].parent.exists()
    assert_state_unchanged(pipeline.owners,before)


def test_test_created_cache_short_circuits_unknown_project_and_preserves_bytes(pipeline, tmp_path):
    output = tmp_path/'owned-cache.pkl'
    expected = [SimpleNamespace(metadata={'fixture':True})]
    payload = pickle.dumps(expected,protocol=pickle.HIGHEST_PROTOCOL)
    output.write_bytes(payload)
    before = state_snapshot(pipeline.owners)
    assert pipeline.call('unknown',output=output,overwrite=False) == expected
    assert not pipeline.calls and output.read_bytes() == payload
    assert_state_unchanged(pipeline.owners,before)


def test_api_overwrite_default_rebuilds_owned_cache_only_after_pipeline_success(pipeline, tmp_path):
    output = tmp_path/'owned-cache.pkl'
    payload = pickle.dumps(['old fixture'],protocol=pickle.HIGHEST_PROTOCOL)
    output.write_bytes(payload)
    result = pipeline.call('jonathan_full',output=output)
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]['original_output'] == payload
    assert output.read_bytes() != payload and pickle.loads(output.read_bytes()) == result


@pytest.mark.parametrize('boundary',['directory','serialization'])
def test_output_failure_after_controlled_build_preserves_global_policy(pipeline, foreign_modules, monkeypatch, tmp_path, boundary):
    before = state_snapshot(pipeline.owners)
    if boundary == 'directory':
        parent = tmp_path/'occupied-parent'
        parent.write_bytes(b'owned fixture')
        output = parent/'session.pkl'
        expected = OSError
    else:
        output = tmp_path/'selected-output/session.pkl'
        def failed_dump(*args,**kwargs):
            raise RuntimeError('fixture serialization failure')
        monkeypatch.setattr(pipeline.owners.driver.pickle,'dump',failed_dump)
        expected = RuntimeError
    with pytest.raises(expected):
        pipeline.call('jonathan_oft_baseline',output=output)
    assert len(pipeline.calls) == 1 and pipeline.calls[0]['flow'] == EXPECTED_FLOW
    assert_state_unchanged(pipeline.owners,before)
    if boundary == 'directory':
        assert parent.read_bytes() == b'owned fixture'
    else:
        # Historical serialization opens the selected file after a successful
        # build. This test claims global-state isolation, not atomic file writes.
        assert output.read_bytes() == b''


def test_existing_config_flip_hook_is_between_second_enrichment_and_split(pipeline, monkeypatch):
    config = pipeline.owners.policies['studies.oft_baseline.config']
    monkeypatch.setattr(config,'OPTIMAL_FLIP_IDS',[728],raising=False)
    result = pipeline.call('jonathan_oft_baseline')
    assert len(pipeline.calls[0]['steps']) == 9
    assert [session.metadata['is_flipped'] for session in result] == [True,False]
    assert pipeline.calls[0]['flow'] == EXPECTED_FLOW


@pytest.mark.parametrize('csv_choice',['data_root','legacy','csvs','explicit'])
@pytest.mark.parametrize('explicit_flags',[False,True])
def test_cli_preserves_csv_priority_path_arguments_and_distinct_defaults(owners, monkeypatch, tmp_path, csv_choice, explicit_flags):
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path/'dataset'
    data_root.mkdir()
    if csv_choice in ('legacy','csvs','explicit'):
        (data_root/'Jonathan_IFxDN_csvs').mkdir()
    if csv_choice in ('csvs','explicit'):
        (data_root/'csvs').mkdir()
    expected_csv = {'data_root':data_root,'legacy':data_root/'Jonathan_IFxDN_csvs',
                    'csvs':data_root/'csvs','explicit':tmp_path/'alternate'}[csv_choice]
    calls = []
    monkeypatch.setattr(owners.driver,'build_sessions',lambda **kwargs:calls.append(kwargs))
    arguments = ['unified-builder','--project','karen_project','--data-dir','dataset',
                 '--output','temporary/result.pkl']
    if csv_choice == 'explicit':
        arguments.extend(['--csv-root','alternate'])
    if explicit_flags:
        arguments.extend(['--dirty','--overwrite','--discovery-mode','h5'])
    monkeypatch.setattr(sys,'argv',arguments)
    owners.driver.main()
    assert calls == [dict(data_root=data_root,h5_dir=data_root/'h5_files',
                         video_dir=data_root/'video_files',csv_root=expected_csv,
                         output_pickle='temporary/result.pkl',project='karen_project',
                         overwrite=explicit_flags,force_dirty=explicit_flags,
                         discovery_mode='h5' if explicit_flags else None)]
    assert not (tmp_path/'temporary').exists()
