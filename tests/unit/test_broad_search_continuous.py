"""Continuous batches, duplicate suppression and diagnostic provenance."""

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from studies.sequence_viz.broad_search_tracking import CandidateTracker, result_metrics
from studies.sequence_viz.run_broad_search import (
    SearchLedger,
    _batch_recipes,
    _recipe_signature,
    atomic_json,
    continuation_jobs,
    execute_search,
)
from tests.unit.test_broad_search_scheduler import _fixture, _initialize


def _evaluate_until_next_batch(state, job):
    if job.get("batch_index", 0) > 0:
        (Path(state["output"])/"STOP").touch()
        time.sleep(.3)
    return dict(status="SUCCESS", value=job["base_recipe"])


def _new_tiny_batch(plan, batch):
    return [dict(job_id=f"batch{batch}", base_recipe=100+batch, cohort="fixture", fold=0,
                 replicate=0, batch_index=batch, recipe=dict(method="fixture", creator_seed=7, knob=100+batch))]


def test_fresh_balanced_batches_and_expanded_fitting_budgets():
    plan=dict(seed=20261013, recipes_per_cohort=30, replicates=2, folds=3, cohorts=["karen", "jonathan"])
    first=continuation_jobs(plan, 1)
    assert first == continuation_jobs(plan, 1)
    assert 180 <= len(first) <= 252
    assert not {j["job_id"] for j in first} & {j["job_id"] for j in continuation_jobs(plan, 2)}
    recipes=_batch_recipes(20261013, 120)
    assert {r["iterations"] for r in recipes if r["method"]=='hsmm'} <= {8,16,32,64}
    assert max(r["iterations"] for r in recipes if r["method"]=='hsmm') > 8
    assert all(j['replicate']==0 for j in first if j['recipe']['method'] in {'exact','tolerant','grammar'})
    assert all(j['replicate']==0 for j in first if j['recipe']['method']=='hsmm' and j['recipe']['initialization']=='configured')


def test_equivalent_inactive_settings_are_not_new_recipes():
    first=dict(method='exact', route='kpms', seed=1, bundle='spatial',
               active_parameters=['route','seed'], base_id='one')
    second=dict(first, seed=999, bundle='movement', base_id='two')
    assert _recipe_signature(first)==_recipe_signature(second)


def test_continuous_stop_resume_retains_generated_batch_and_terminal_results(tmp_path):
    plan,_,_,output=_fixture(tmp_path)
    plan.update(continuous=True)
    result=execute_search(plan, evaluator=_evaluate_until_next_batch, initializer=_initialize,
                          batch_generator=_new_tiny_batch)
    assert result['state']=='STOPPED'
    assert result['counts']=={'STOPPED':1,'SUCCESS':3}
    assert json.loads((output/'status.json').read_text())['total_jobs'] is None
    ledger=SearchLedger(output,plan,resume=True)
    assert ledger.next_batch()==2
    assert len(ledger.pending())==1
    assert ledger.pending()[0]['batch_index']==1
    assert ledger.connection.execute("SELECT COUNT(*) FROM jobs WHERE attempts=1 AND status='SUCCESS'").fetchone()[0]==3
    # Equivalent request in a new batch is skipped but its empty batch record persists.
    repeated=[dict(ledger.pending()[0],job_id='duplicate',base_recipe=999)]
    assert ledger.append_batch(2,repeated)==[]
    ledger.close()


def test_allocation_cap_returns_memoryerror_instead_of_large_allocation():
    script='''from studies.sequence_viz.run_broad_search import set_worker_allocation_cap
set_worker_allocation_cap(0.125)
try:
    bytearray(1024**3)
except MemoryError:
    print('allocation refused')
else:
    raise AssertionError('allocation cap failed')
'''
    # -S avoids the lab site's unrelated startup imports; PYTHONPATH provides source.
    result=subprocess.run([sys.executable,'-S','-B','-c',script],capture_output=True,text=True,check=True)
    assert 'allocation refused' in result.stdout


def _result(calls=10,within=.9):
    return dict(status='SUCCESS',summary=dict(selected_calls=calls,coverage_fraction=.5,held_animals=5),
        quality=dict(types=[dict(occurrences=calls,known_animals=4,mean_within_sequence_agreement=within,
                                 mean_other_type_sequence_agreement=.3)],
                     catalogues=[dict(observed_types=2,singleton_types=0)]),conditions={})


def test_best_tracking_preserves_config_artifact_missingness_and_seed_pair(tmp_path):
    con=sqlite3.connect(':memory:')
    tracker=CandidateTracker(con,tmp_path,atomic_json)
    job=dict(job_id='id1',cohort='karen',recipe=dict(method='hsmm',route='kpms',phase_count=3))
    tracker.add(job,_result(),tmp_path/'original.json')
    tracker.add_pair(dict(reference=job,alternative=dict(job_id='missing'),comparison=dict(matched_type_temporal_f1=.8)))
    tracker.export(force=True)
    saved=json.loads((tmp_path/'best_candidates.json').read_text())
    assert saved['tracked_results']==1
    assert saved['candidates'][0]['job']==job
    assert saved['candidates'][0]['artifact']==str(tmp_path/'original.json')
    assert saved['candidates'][0]['seed_stability']==.8
    assert result_metrics(dict(summary={},quality={},conditions={}))['agreement'] is None
    assert 'semantic winner' in saved['policy']
