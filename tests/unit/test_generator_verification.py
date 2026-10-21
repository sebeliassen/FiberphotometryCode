"""Independent expectations for native support, matching and mouse-level pooling."""
import gzip
import json
import sqlite3

import numpy as np
import pytest

from studies.sequence_viz.export_wt_verification_signals import audit_clock
from studies.sequence_viz.measure_wt_generator_da import bout_signal_means, matched_difference
from studies.sequence_viz.summarize_generator_verification import (
    correspondences,
    physical_from_moments,
    stratified_group_contrast,
    summarize,
)
from studies.sequence_viz.verification_readouts import recording_readouts


def _record():
    starts = np.arange(12)*10
    values = np.column_stack([np.ones(12),np.r_[np.full(6,4.),np.ones(6)],np.zeros((12,2))])
    return dict(cohort="synthetic",subject="m",recording="r",fps=10.,start_frames=starts,stop_frames=starts+10,
        parent_ids=np.array(["A"]*9+["B"]*3),source_vocabulary=["A","B"],values=values,
        feature_names=["duration_s","mean_speed_px_s","centroid_x_px","centroid_y_px"])


def _prediction():
    return dict(valid_intervals=[[0,120]],diagnostics={},consumed_features=["mean_speed_px_s"],selected_calls=[
        dict(type_id="routine_0",start_frame=0,stop_frame=30),
        dict(type_id="routine_0",start_frame=30,stop_frame=60)])


def test_matched_intra_known_label_coverage_and_no_opportunity_is_missing():
    result = recording_readouts(_record(),_prediction(),{},["routine_0","routine_1"],
        {"A":"annotation:groom","B":"unmapped:B"},dict(location=np.zeros(4)),null_repeats=0)
    def value(family,feature,token="__all__"):
        return next(r for r in result["observations"] if r["family"]==family and r["feature"]==feature and r["type_id"]==token)
    intra = value("intra","mean_speed_px_s_inside_minus_outside")
    assert intra["value"] == 3. and intra["n_support"] == 6 and intra["detector_consumed"]
    assert value("annotation_coverage","annotation:groom")["value"] == pytest.approx(6/9)
    assert not any(r["feature"]=="unmapped:B" for r in result["observations"] if r["family"].startswith("annotation"))
    assert value("inter","g10_self_excess")["value"] == 0.
    assert value("inter","g10_return2_probability")["value"] is None
    assert value("recruitment","calls_per_min","routine_1")["value"] == 0.
    assert value("execution","mean_call_duration_s","routine_1")["value"] is None


def test_missing_native_gap_is_not_an_inter_switch_or_opportunity():
    rec, prediction = _record(),_prediction()
    rec["start_frames"][3:] += 100
    rec["stop_frames"][3:] += 100
    prediction["valid_intervals"] = [[0,30],[130,220]]
    prediction["selected_calls"][1].update(start_frame=130,stop_frame=160)
    result = recording_readouts(rec,prediction,{},["routine_0"],{},dict(location=np.zeros(4)),null_repeats=0)
    counts = next(r["counts"] for r in result["inter"] if r["gap_seconds"]==10.)
    assert counts["calls"] == 2 and counts["pair_opportunities"] == 0
    assert counts["right_censored_runs"] == counts["left_censored_runs"] == 2


def test_fold_adjustment_recovers_group_effect_despite_fold_offset():
    result = stratified_group_contrast([0,0,4,100,104,104],["WT","WT","KI","WT","KI","KI"],
        [0,0,0,1,1,1],first="WT",second="KI",resamples=99)
    assert result["difference"] == 4. and result["supported_folds"] == 2
    assert result["ci_low"] == result["ci_high"] == 4.
    assert result["n_first"] == result["n_second"] == 3


def test_clock_audit_uses_stored_times_and_does_not_fit_away_an_offset():
    frames,valid,report = audit_clock([0,4,7],[.001,.055,.19],np.arange(10)*.01)
    np.testing.assert_array_equal(frames,[0,4,7])
    np.testing.assert_array_equal(valid,[True,True,False])
    assert report["eligible_samples"]==2
    assert report["maximum_absolute_error_s"] == pytest.approx(.12)
    with pytest.raises(ValueError,match="camera clock"):
        audit_clock([0,1],[0.,.01],[.02,0.])


def test_da_means_require_native_and_timestamp_membership_and_keep_missing_samples():
    record = dict(start_frames=np.array([0,10]),stop_frames=np.array([10,20]))
    frames = np.array([0,3,6,10,13,16])
    signals = dict(native_camera_time_s=np.arange(21)*.1,native_photo_frames=frames,
        photo_time_s=frames*.1,alignment_valid=np.ones(6,bool),trace=np.array([1.,2.,3.,10.,20.,30.]))
    means,counts,duration = bout_signal_means(record,signals,"trace")
    np.testing.assert_array_equal(means,[2.,20.])
    np.testing.assert_array_equal(counts,[3,3])
    np.testing.assert_allclose(duration,[1.,1.])
    signals["photo_time_s"][2] = 1.
    means,counts,_ = bout_signal_means(record,signals,"trace")
    assert np.isnan(means[0]) and counts[0]==2 and means[1]==20.
    signals["trace"][3] = np.nan
    assert np.isnan(bout_signal_means(record,signals,"trace")[0][1])


def test_da_context_contrast_removes_stratum_offsets_and_requires_both_sides():
    values = [2.,2.,2.,0.,0.,0.,12.,12.,12.,10.,10.,10.]
    inside = np.tile(np.array([True]*3+[False]*3),2)
    result = matched_difference(values,inside,~inside,[0]*6+[1]*6)
    assert result==dict(difference=2.,inside_bouts=6,outside_bouts=6,strata=2)
    assert matched_difference(values,np.zeros(12,bool),np.ones(12,bool),[0]*12)["difference"] is None


def test_crosswalk_matches_permuted_ids_but_rejects_ambiguous_type():
    job = dict(panel_id="P",fold=0,replicate=0,source_job_id="ref")
    ref = dict(model_key="ref",vocabulary=["A","B"],prototypes={"0":[1.,0.],"1":[0.,1.]},training_calls={"0":5,"1":5})
    other = dict(model_key="other",vocabulary=["A","B"],prototypes={"a":[0.,1.],"b":[1.,0.],"c":[.5,.5]},
                 training_calls={"a":5,"b":5,"c":5})
    mapping,audit = correspondences([(job,ref),(dict(job,fold=1,source_job_id="other"),other)])
    assert mapping["ref"] == {"0":"ref:0","1":"ref:1"}
    assert mapping["other"] == {"a":"ref:1","b":"ref:0"}
    assert any(r["local_type"]=="c" and not r["accepted"] for r in audit)


def test_streamed_physical_moments_match_explicit_cross_animal_pairs():
    values = np.array([[0.,1.],[2.,3.],[1.,4.],[3.,2.],[10.,5.],[12.,7.],[11.,8.],[13.,6.]])
    labels = np.array(["a"]*4+["b"]*4)
    subjects = np.array(["1","1","2","2"]*2)
    groups = {}
    for token in ("a","b"):
        for subject in ("1","2"):
            part = values[(labels==token)&(subjects==subject)]
            groups[(token,subject)] = dict(calls=len(part),count=np.full(2,len(part)),sum=part.sum(axis=0),sum_squared=(part**2).sum(axis=0))
    result = physical_from_moments(groups,[0,1])
    # There are two within-type independent-animal pair sets and symmetric between comparisons.
    within = np.mean([np.mean((a-b)**2) for token in ("a","b") for a in values[(labels==token)&(subjects=="1")]
                      for b in values[(labels==token)&(subjects=="2")]])
    between = np.mean([np.mean((a-b)**2) for a,t,s in zip(values,labels,subjects) for b,u,v in zip(values,labels,subjects)
                       if t!=u and s!=v])
    np.testing.assert_allclose(result["within"],within,rtol=1e-13)
    np.testing.assert_allclose(result["between"],between,rtol=1e-13)
    assert physical_from_moments(groups,[0,1],omit_subject="1")["separation"] is None


def test_summary_uses_real_drug_names_and_seeds_do_not_inflate_animals(tmp_path):
    from studies.sequence_viz.run_broad_search import digest

    (tmp_path/"jobs").mkdir()
    (tmp_path/"recordings").mkdir()
    (tmp_path/"run_manifest.json").write_text(json.dumps(dict(panel_hash="fixed",null_repeats=0)))
    db = sqlite3.connect(tmp_path/"trials.sqlite")
    db.execute("CREATE TABLE jobs(artifact TEXT,status TEXT,ordinal INTEGER)")
    ordinal = 0
    for seed in (0,1):
        anchor_job = dict(kind="training_anchor",panel_id="P",fold=0,replicate=seed,source_job_id=str(seed))
        anchor_result = dict(model_key=str(seed),vocabulary=["A"],prototypes={"r":[1.]},training_calls={"r":5})
        path = tmp_path/"jobs"/f"anchor{seed}.json"
        path.write_text(json.dumps(dict(job=anchor_job,result=anchor_result)))
        db.execute("INSERT INTO jobs VALUES(?,'SUCCESS',?)",(str(path),ordinal))
        ordinal += 1
        for mouse in range(6):
            group = "WT" if mouse<3 else "KI"
            for drug in ("Saline","Amphetamine"):
                metadata = dict(genotype=group,drug=drug)
                identity = dict(panel_id="P",cohort="jonathan",fold=0,replicate=seed,model_key=str(seed),
                    subject=str(mouse),recording=f"{mouse}_{drug}",metadata=metadata)
                observations = []
                for epoch in ("pre_injection_15min","post_injection_2to17min"):
                    value = 1.+(4 if group=="KI" else 0)+seed*10+(2 if drug=="Amphetamine" and epoch.startswith("post") else 0)
                    observations.append(dict(epoch=epoch,family="recruitment",feature="calls_per_min",type_id="__all__",
                        parent_id="",value=value,denominator=900,n_support=10,detector_consumed=False))
                compressed = tmp_path/"recordings"/f"{seed}_{mouse}_{drug}.gz"
                with gzip.open(compressed,"wt") as handle:
                    json.dump(dict(identity=identity,observations=observations),handle)
                path = tmp_path/"jobs"/f"{seed}_{mouse}_{drug}.json"
                path.write_text(json.dumps(dict(job=dict(kind="recording"),result=dict(recording_artifact=str(compressed),
                    sha256=digest(compressed),selected_calls=10))))
                db.execute("INSERT INTO jobs VALUES(?,'SUCCESS',?)",(str(path),ordinal))
                ordinal += 1
    db.commit()
    db.close()
    summary = summarize(tmp_path,resamples=99)
    import pandas as pd

    contrasts = pd.read_csv(tmp_path/"phenotype_contrasts.csv")
    between = contrasts[contrasts.comparison=="between_group"]
    assert (between.n_first==3).all() and (between.n_second==3).all()
    assert (between.difference==4.).all()
    paired = contrasts[contrasts.comparison=="within_mouse_paired_drug_change"]
    np.testing.assert_allclose(paired.mean_change,2.)
    assert summary["recording_evaluations"] == 24
