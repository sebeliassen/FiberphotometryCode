"""Supporting WT associations from frozen calls and already processed DA samples.

Native and actual-timestamp membership must agree. Same-source/time/occupancy/
movement/duration matching is observational; this is not a detector truth test or
a temporal-shuffle inference. Animals, not bouts, fibres or seeds, are the units.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import resource
from collections import defaultdict
from pathlib import Path


def bout_signal_means(record, signals, column, *, minimum_samples=3, minimum_fraction=.5):
    """Original LED-sample means, with agreeing native/time bounds and no fills."""
    import numpy as np

    starts,stops = record["start_frames"],record["stop_frames"]
    camera = signals["native_camera_time_s"]
    frames,times = signals["native_photo_frames"],signals["photo_time_s"]
    bound = stops < len(camera)  # No unobserved exclusive clock endpoint is extrapolated.
    duration = np.full(len(starts),np.nan)
    duration[bound] = camera[stops[bound]]-camera[starts[bound]]
    slots = np.searchsorted(starts,frames,side="right")-1
    safe = np.maximum(slots,0)
    valid = signals["alignment_valid"] & (slots>=0) & (frames<stops[safe]) & bound[safe]
    valid &= np.isfinite(signals[column])
    where = np.flatnonzero(valid)
    valid[where] &= (times[where]>=camera[starts[slots[where]]]) & (times[where]<camera[stops[slots[where]]])
    counts = np.bincount(slots[valid],minlength=len(starts))
    sums = np.bincount(slots[valid],weights=signals[column][valid],minlength=len(starts))
    means = np.divide(sums,counts,out=np.full(len(starts),np.nan),where=counts>0)
    interval = float(np.median(np.diff(times)))
    fraction = np.divide(counts,duration/interval,out=np.zeros(len(starts)),where=duration>0)
    means[(counts<minimum_samples)|(fraction<minimum_fraction)] = np.nan
    return means,counts,duration


def matched_difference(values,inside,outside,strata, *, minimum_bouts=3):
    """Signed, inside-count-weighted stratum mean difference on common support."""
    import numpy as np

    values,strata = np.asarray(values,float),np.asarray(strata,int)
    finite = np.isfinite(values)
    own,other = np.asarray(inside,bool)&finite,np.asarray(outside,bool)&finite
    size = max(1,int(strata.max())+1)
    ni,no = np.bincount(strata[own],minlength=size),np.bincount(strata[other],minlength=size)
    si,so = np.bincount(strata[own],weights=values[own],minlength=size),np.bincount(strata[other],weights=values[other],minlength=size)
    eligible = (ni>=minimum_bouts)&(no>=minimum_bouts)
    if not eligible.any():
        return dict(difference=None,inside_bouts=0,outside_bouts=0,strata=0)
    differences = si[eligible]/ni[eligible]-so[eligible]/no[eligible]
    return dict(difference=float(np.average(differences,weights=ni[eligible])),inside_bouts=int(ni[eligible].sum()),
        outside_bouts=int(no[eligible].sum()),strata=int(eligible.sum()))


def measure(*, verification, signals, output):
    import numpy as np
    import pandas as pd

    from fpcode.common.cache_manifest import atomic_write_json, atomic_write_text
    from fpcode.statistics.behavior_comparisons import adjust_fdr, paired_contrast
    from studies.sequence_viz.assess_broad_search import training_context
    from studies.sequence_viz.broad_search_inputs import _load_npz, load_job_records
    from studies.sequence_viz.run_broad_search import digest
    from studies.sequence_viz.verification_readouts import epochs

    verification,output = Path(verification),Path(output)
    signal_manifest = json.loads(Path(signals).read_text())
    if json.loads((verification/"aggregation_status.json").read_text())["state"]!="COMPLETED":
        raise ValueError("completed behaviour scorecards required before DA readout binding")
    if output.exists():
        raise ValueError("choose a new explicit DA observation output")
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    plan = json.loads((verification/"run_manifest.json").read_text())
    manifest = json.loads(Path(plan["manifest"]).read_text())
    cohort = next(c for c in manifest["cohorts"] if c["cohort_id"]=="jonathan")
    sigrows = {r["recording"]:r for r in signal_manifest["records"]}
    native,signal_data,contexts,means_cache = {},{},{},{}
    mappings = defaultdict(dict)
    with (verification/"catalogue_correspondence.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["accepted"]=="True":
                mappings[row["model_key"]][row["local_type"]] = "ref:"+row["reference_type"]
    cases = []
    for path in sorted((verification/"jobs").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload["job"]["kind"]=="recording" and payload["result"]["cohort"]=="jonathan" and payload["result"]["recording"] in sigrows:
            cases.append(payload)
    observations,unlocated = [],set()
    for payload in cases:
        job,result = payload["job"],payload["result"]
        recname,fold = result["recording"],job["fold"]
        if fold not in contexts:
            contexts[fold] = training_context(load_job_records(manifest,"jonathan",fold,per_animal=600,held_per_animal=180,window_seed=73))
        context = contexts[fold]
        if recname not in native:
            row = next(r for r in cohort["records"] if r["recording"]==recname)
            if digest(row["npz_path"])!=row["sha256"]:
                raise ValueError("frozen native bout input changed")
            data = _load_npz(row["npz_path"],max_bytes=64*1024**2)
            native[recname] = dict(data,cohort="jonathan",recording=recname,subject=row["subject"],fps=cohort["fps"])
            if digest(sigrows[recname]["npz_path"])!=sigrows[recname]["sha256"]:
                raise ValueError("WT signal component changed")
            signal_data[recname] = _load_npz(sigrows[recname]["npz_path"],max_bytes=64*1024**2)
        record,stream = native[recname],signal_data[recname]
        if digest(result["recording_artifact"])!=result["sha256"]:
            raise ValueError("behaviour call component changed")
        with gzip.open(result["recording_artifact"],"rt") as handle:
            predicted = json.load(handle)["prediction"]
        starts,stops,values = record["start_frames"],record["stop_frames"],record["values"]
        names = list(record["feature_names"])
        valid = np.asarray(predicted["valid_intervals"],int).reshape(-1,2)
        pos = np.searchsorted(valid[:,0],starts,side="right")-1
        available = (pos>=0)&(stops<=valid[np.maximum(pos,0),1]) if len(valid) else np.zeros(len(starts),bool)
        camera = stream["native_camera_time_s"]
        clock_known = stops<len(camera)
        time_bin = np.full(len(starts),-1)
        time_bin[clock_known] = ((camera[starts[clock_known]]-camera[0])/300).astype(int)
        x,y = names.index("centroid_x_px"),names.index("centroid_y_px")
        space = (values[:,x]>=context["location"][x]).astype(int)*2+(values[:,y]>=context["location"][y]).astype(int)
        bins = []
        for feature in ("mean_speed_px_s","duration_s"):
            j = names.index(feature)
            bins.append(np.digitize(values[:,j],context["location"][j]+np.array([-.5,.5])*context["scale"][j]))
        stratum = pd.factorize(pd.MultiIndex.from_arrays([record["parent_ids"],time_bin,space,*bins]))[0]
        covariates_finite = np.isfinite(values[:,[x,y,names.index("mean_speed_px_s"),names.index("duration_s")]]).all(axis=1)
        for epoch,lo,hi in epochs(record,result["metadata"]):
            eligible = available & clock_known & (starts>=lo)&(stops<=hi)&covariates_finite
            membership = np.full(len(starts),"",dtype=object)
            for call in predicted["selected_calls"]:
                if lo<=call["start_frame"]<call["stop_frame"]<=hi:
                    a,b = np.searchsorted(starts,call["start_frame"]),np.searchsorted(stops,call["stop_frame"],side="right")
                    membership[a:b] = call["type_id"]
            for info in sigrows[recname]["signals"]:
                column = info["column"]
                if not column.endswith(("_dff","_zscore","_zscore_datta")):
                    continue
                region,hemisphere = info.get("brain_reg"),info.get("hemisphere")
                if not region or not hemisphere:
                    unlocated.add((recname,info["base_signal"]))
                    continue
                cachekey = (recname,column)
                if cachekey not in means_cache:
                    means_cache[cachekey] = bout_signal_means(record,stream,column)
                da,counts,duration = means_cache[cachekey]
                outside = eligible & (membership=="")
                for token in ["__all__"]+sorted(set(membership)-{""}):
                    inside = eligible & ((membership!="") if token=="__all__" else (membership==token))
                    match = matched_difference(da,inside,outside,stratum)
                    own,other = da[inside & np.isfinite(da)],da[outside & np.isfinite(da)]
                    total = float(own.mean()-other.mean()) if len(own)>=3 and len(other)>=3 else None
                    scope = "__all__" if token=="__all__" else mappings[result["model_key"]].get(token,result["model_key"]+":"+token)
                    common = dict(panel_id=result["panel_id"],fold=fold,replicate=job["replicate"],subject=result["subject"],
                        recording=recname,drug=result["metadata"]["drug"],epoch=epoch,region=region,hemisphere=hemisphere,
                        base_signal=info["base_signal"],column=column,trace_suffix=column[len(info["base_signal"]):],
                        scope=scope,local_type=token,model_key=result["model_key"],
                        inside_bouts=int(len(own)),outside_bouts=int(len(other)),matched_inside_bouts=match["inside_bouts"],
                        matched_outside_bouts=match["outside_bouts"],matched_strata=match["strata"])
                    observations += [dict(common,metric="total_inside_minus_outside",value=total),
                                     dict(common,metric="matched_inside_minus_outside",value=match["difference"])]
    output.mkdir(parents=True)
    table = pd.DataFrame(observations)
    table.to_csv(output/"recording_readouts.csv.gz",index=False,compression="gzip")
    keys = ["panel_id","fold","subject","drug","epoch","region","hemisphere","trace_suffix","scope","metric"]
    mice = table.groupby(keys,observed=True,as_index=False).agg(value=("value","mean"),seeds=("replicate","nunique"),
        inside_bouts=("inside_bouts","mean"),matched_inside_bouts=("matched_inside_bouts","mean"))
    # Pool same-region fibres/hemispheres within mouse, preserving separate-side sensitivity too.
    pooled = mice.groupby([k for k in keys if k!="hemisphere"],observed=True,as_index=False).agg(
        value=("value","mean"),seeds=("seeds","max"),inside_bouts=("inside_bouts","mean"),matched_inside_bouts=("matched_inside_bouts","mean"))
    pooled["hemisphere"] = "pooled_within_mouse"
    mice = pd.concat([mice,pooled],ignore_index=True)
    mice.to_csv(output/"mouse_readouts.csv.gz",index=False,compression="gzip")
    contrasts = []
    group = ["panel_id","region","hemisphere","trace_suffix","scope","metric"]
    for ordinal,(key,part) in enumerate(mice.groupby(group,observed=True)):
        base = dict(zip(group,key))
        for (drug,epoch),section in part.groupby(["drug","epoch"],observed=True):
            contrasts.append(dict(base,comparison="WT_association",drug=drug,epoch=epoch,
                **paired_contrast(section.value.to_numpy(float),seed=101+ordinal)))
        pivot = part.pivot_table(index="subject",columns=["drug","epoch"],values="value",observed=True)
        for epoch in part.epoch.unique():
            before,after = ("Saline",epoch),("Amphetamine",epoch)
            if before in pivot and after in pivot:
                contrasts.append(dict(base,comparison="WT_paired_drug_association_change",drug="Amphetamine minus Saline",epoch=epoch,
                    **paired_contrast((pivot[after]-pivot[before]).dropna().to_numpy(float),seed=101+ordinal)))
        pre,post = "pre_injection_15min","post_injection_2to17min"
        for drug in ("Saline","Amphetamine"):
            if (drug,pre) in pivot and (drug,post) in pivot:
                contrasts.append(dict(base,comparison="WT_pre_post_association_change",drug=drug,epoch="post minus pre",
                    **paired_contrast((pivot[(drug,post)]-pivot[(drug,pre)]).dropna().to_numpy(float),seed=101+ordinal)))
        if all((drug,epoch) in pivot for drug in ("Saline","Amphetamine") for epoch in (pre,post)):
            change = (pivot[("Amphetamine",post)]-pivot[("Amphetamine",pre)])-(pivot[("Saline",post)]-pivot[("Saline",pre)])
            contrasts.append(dict(base,comparison="WT_paired_drug_difference_in_change",drug="Amphetamine minus Saline",
                epoch="difference in pre-post change",**paired_contrast(change.dropna().to_numpy(float),seed=101+ordinal)))
    family = defaultdict(list)
    for i,row in enumerate(contrasts):
        family[(row["comparison"],row["drug"],row["epoch"],row["metric"])].append(i)
    for indices in family.values():
        q = adjust_fdr([contrasts[i]["p_permutation"] for i in indices])
        for i,value in zip(indices,q):
            contrasts[i]["q_all_settings_regions_suffixes"] = float(value) if np.isfinite(value) else None
    from studies.sequence_viz.summarize_generator_verification import _write_csv

    _write_csv(output/"contrasts.csv",contrasts)
    report = dict(recording_evaluations=len(cases),mice=sorted({r["subject"] for r in signal_manifest["records"]}),
        observation_rows=len(table),mouse_rows=len(mice),contrast_rows=len(contrasts),
        unlocated_traces=sorted(unlocated),source_sha256=signal_manifest["source_sha256"],
        resource_settings=dict(address_space_cap_gib=4,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
        policy="Primary stored_zscore_datta; dff/global_zscore sensitivity; native/time membership agrees; >=3LEDsamples and50% actual-clock coverage per bout",
        limitations=["WT only, anatomically labelled fibres only; region-specificN smaller than8.",
            "Source/time/occupancy/movement/duration matching is observational; no causal DA or semantic truth claim.",
            "No temporal-shuffle or held-animal incremental predictive model is included in this supporting readout.",
            "Centered stored Datta processing uses future samples; no pre-onset causal interpretation.",
            "Condition order/date is fixed; detector/type and suffix selection on these data remains exploratory."])
    atomic_write_json(output/"summary.json",report)
    atomic_write_text(output/"report.md",f"""# WT DA supporting readouts

Applied existing calls to {len(cases)} held WT recording evaluations from the
11fixed Jonathan settings. Eight mice/16source recordings; region-specific
animal counts remain in `contrasts.csv`. Two traces lack stored region/hemisphere
and are excluded from regional readouts, retained in the original export.

Primary trace is stored470nmgreen `_zscore_datta`; `_dff` and global `_zscore`
are sensitivity views. No processing, clock shift/scale fitting or detector fit.
LED samples must agree in native-frame and actual timestamp membership. Bouts
require>=3finite samples and>=50% coverage estimated from actual clocks; untimed
exclusive endpoints are censored. Total inside/outside differences accompany
same-source, five-minute, occupancy, movement andduration-stratum differences
with>=3bouts on each side. Missing common support remains missing.

Repeated detector seeds average within mouse. Regional pooled values average
hemispheres/fibres within mouse; separate-side results remain available. Mouse
sign-flip/bootstrap intervals and BH span all settings/regions/suffixes. These are
supporting exploratory associations, without a temporal-shuffle control or an
incremental predictive model; centered processing does not show causal precedence.
A weak or null DA result does not invalidate a behavioural chain generator.
""")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verification","signals","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    kwargs = vars(args)
    execute = kwargs.pop("execute")
    print(json.dumps(measure(**kwargs) if execute else {k:str(v) for k,v in kwargs.items()}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
