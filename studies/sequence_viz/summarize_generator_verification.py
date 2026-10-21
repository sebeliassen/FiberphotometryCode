"""Exploratory mouse-level scorecards for the fixed verification panel.

Frozen training descriptors support conservative correspondences. Unmatched
types and HSMM phase identities stay model-scoped. Seeds are repeated measurements
of the same animals, never extra animals. No method-family recipe averaging.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def correspondences(anchors, *, minimum_calls=3, similarity=.85, margin=.10):
    """One-to-one source-phase cosine matches; ambiguity remains unmatched."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    mapping, audit = {}, []
    by_panel = defaultdict(list)
    for job, result in anchors:
        by_panel[job["panel_id"]].append((job, result))
    for panel, fits in sorted(by_panel.items()):
        fits.sort(key=lambda x:(x[0]["fold"], x[0]["replicate"]))
        reference_job, reference = fits[0]
        ref_ids = sorted(t for t in reference["prototypes"] if reference["training_calls"][t] >= minimum_calls)
        ref = np.asarray([reference["prototypes"][t] for t in ref_ids])
        for job, result in fits:
            if result["vocabulary"] != reference["vocabulary"]:
                raise ValueError("correspondence source vocabularies disagree")
            ids = sorted(t for t in result["prototypes"] if result["training_calls"][t] >= minimum_calls)
            mapping[result["model_key"]] = {}
            if not ids or not ref_ids:
                continue
            own = np.asarray([result["prototypes"][t] for t in ids])
            scores = (own @ ref.T)/np.maximum(np.linalg.norm(own, axis=1)[:,None]*np.linalg.norm(ref, axis=1), 1e-12)
            rows, columns = (np.arange(len(ids)), np.arange(len(ids))) if job["source_job_id"] == reference_job["source_job_id"] else linear_sum_assignment(-scores)
            assigned = dict(zip(rows, columns))
            for i, token in enumerate(ids):
                j = assigned.get(i)
                score = float(scores[i,j]) if j is not None else None
                row_margin = score-float(np.max(np.delete(scores[i], j))) if j is not None and len(ref_ids)>1 else 1.
                col_margin = score-float(np.max(np.delete(scores[:,j], i))) if j is not None and len(ids)>1 else 1.
                same_fit = job["source_job_id"] == reference_job["source_job_id"]
                accepted = j is not None and (same_fit or score >= similarity and min(row_margin,col_margin) >= margin)
                if accepted:
                    mapping[result["model_key"]][token] = "ref:"+ref_ids[j]
                audit.append(dict(panel_id=panel, model_key=result["model_key"], source_job_id=job["source_job_id"],
                    local_type=token, reference_model_key=reference["model_key"], reference_type=ref_ids[j] if j is not None else None,
                    cosine=score, row_margin=row_margin, column_margin=col_margin, training_calls=result["training_calls"][token],
                    accepted=accepted, interpretation="source composition/order correspondence; semantic and phase homology unproven"))
    return mapping, audit


def physical_from_moments(groups, columns, *, omit_subject=None):
    """Same cross-animal estimand as assessment; streamable per-channel sums."""
    import itertools

    import numpy as np

    from studies.sequence_viz.assess_broad_search import mean

    moments = {}
    for (token,subject),row in groups.items():
        if subject == omit_subject or row["calls"] < 2 or not columns:
            continue
        counts, sums, squares = [np.asarray(row[k],float)[columns] for k in ("count","sum","sum_squared")]
        center = np.divide(sums,counts,out=np.full(len(columns),np.nan),where=counts>0)
        variance = np.divide(squares,counts,out=np.full(len(columns),np.nan),where=counts>0)-center**2
        moments[(token,subject)] = (center,np.maximum(0.,variance))
    def distance(a,b):
        return mean(((a[0]-b[0])**2+a[1]+b[1]).tolist())
    types = []
    for token in sorted({t for t,s in moments}):
        own = [(s,v) for (t,s),v in moments.items() if t==token]
        if len(own)<2:
            continue
        within = mean(distance(a[1],b[1]) for a,b in itertools.combinations(own,2))
        between = mean(distance(v,other) for subject,v in own for (t,s),other in moments.items() if t!=token and s!=subject)
        separation = 1-within/between if within is not None and between is not None and between>1e-12 else None
        types.append(dict(type_id=token,within=within,between=between,separation=separation))
    return dict(within=mean(t["within"] for t in types),between=mean(t["between"] for t in types),
        separation=mean(t["separation"] for t in types),eligible_types=len(types))


def stratified_group_contrast(values, groups, folds, *, first, second, seed=101, resamples=1999):
    """Within-fold group contrast with stratified mouse bootstrap/permutation."""
    import numpy as np

    values, groups, folds = np.asarray(values,float), np.asarray(groups), np.asarray(folds)
    valid = np.isfinite(values) & np.isin(groups,[first,second])
    values, groups, folds = values[valid], groups[valid], folds[valid]
    a, b = values[groups==first], values[groups==second]
    out = dict(n_first=len(a), n_second=len(b), mean_first=float(a.mean()) if len(a) else None,
        mean_second=float(b.mean()) if len(b) else None, difference=None, ci_low=None, ci_high=None,
        p_permutation=None, status="insufficient_animals", supported_folds=0)
    strata = [(values[(folds==f)&(groups==first)], values[(folds==f)&(groups==second)]) for f in np.unique(folds)]
    strata = [(x,y) for x,y in strata if len(x) and len(y)]
    if not strata:
        return out
    weights = np.asarray([len(x)*len(y)/(len(x)+len(y)) for x,y in strata])
    weights /= weights.sum()
    out.update(difference=float(sum(w*(y.mean()-x.mean()) for w,(x,y) in zip(weights,strata))), supported_folds=len(strata))
    if min(len(a),len(b)) < 3:
        return out
    rng = np.random.default_rng(seed)
    boot, perm = np.zeros(resamples), np.zeros(resamples)
    for w,(x,y) in zip(weights,strata):
        boot += w*(y[rng.integers(len(y),size=(resamples,len(y)))].mean(axis=1)
                   -x[rng.integers(len(x),size=(resamples,len(x)))].mean(axis=1))
        pooled = np.r_[x,y]
        order = np.argsort(rng.random((resamples,len(pooled))),axis=1)
        perm += w*(pooled[order[:,len(x):]].mean(axis=1)-pooled[order[:,:len(x)]].mean(axis=1))
    low, high = np.quantile(boot,[.025,.975])
    out.update(ci_low=float(low),ci_high=float(high),
        p_permutation=float((1+np.sum(abs(perm)>=abs(out["difference"])-1e-12))/(resamples+1)), status="tested")
    return out


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, *, resamples=1999):
    import numpy as np
    import pandas as pd

    from fpcode.common.cache_manifest import atomic_write_json, atomic_write_text
    from fpcode.statistics.behavior_comparisons import adjust_fdr, paired_contrast
    from studies.sequence_viz.run_broad_search import digest, now_utc

    output = Path(output)
    plan = json.loads((output/"run_manifest.json").read_text())
    connection = sqlite3.connect(output/"trials.sqlite")
    artifacts = connection.execute("SELECT artifact FROM jobs WHERE status='SUCCESS' ORDER BY ordinal").fetchall()
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
    connection.close()
    anchors, recordings = [], []
    for (path,) in artifacts:
        payload = json.loads(Path(path).read_text())
        (anchors if payload["job"]["kind"]=="training_anchor" else recordings).append((payload["job"],payload["result"]))
    mapping, crosswalk = correspondences(anchors)
    _write_csv(output/"catalogue_correspondence.csv",crosswalk)
    database = output/"readouts.sqlite"
    db = sqlite3.connect(database)
    db.execute("DROP TABLE IF EXISTS observations")
    db.execute("""CREATE TABLE observations(panel TEXT,cohort TEXT,model TEXT,fold INTEGER,replicate INTEGER,
        subject TEXT,recording TEXT,group_name TEXT,condition TEXT,drug TEXT,epoch TEXT,family TEXT,feature TEXT,
        scope TEXT,parent TEXT,value REAL,weight REAL,n_support INTEGER,circular INTEGER)""")
    total_calls, record_count, transfer_input = 0, 0, []
    physical_models = {}
    for job,result in recordings:
        path = Path(result["recording_artifact"])
        if digest(path) != result["sha256"]:
            raise ValueError("recording readout artifact changed")
        with gzip.open(path,"rt") as handle:
            payload = json.load(handle)
        identity = payload["identity"]
        model, meta = identity["model_key"],identity["metadata"]
        cohort = identity["cohort"]
        group = meta["group"] if cohort=="karen" else meta["genotype"]
        condition = meta["condition"] if cohort=="karen" else meta["drug"]
        physical = physical_models.setdefault(model,dict(identity=identity,profile={},residual={}))
        for token,part in payload.get("physical_moments",{}).items():
            for kind in ("profile","residual"):
                key = (token,identity["subject"])
                row = physical[kind].setdefault(key,dict(calls=0,count=np.zeros(len(part[kind]["count"])),
                    sum=np.zeros(len(part[kind]["sum"])),sum_squared=np.zeros(len(part[kind]["sum_squared"]))))
                row["calls"] += part["calls"]
                for stat in ("count","sum","sum_squared"):
                    row[stat] += np.asarray(part[kind][stat])
        records = []
        for row in payload["observations"]:
            if row["value"] is None:
                continue
            token, feature = row["type_id"],row["feature"]
            scope = ("__all__" if token == "__all__" else mapping.get(model,{}).get(token,model+":"+token))
            if row["family"]=="hsmm_progression" and token!="__all__":
                scope = model+":"+token  # Phase homology was not demonstrated by primitive profiles.
            if row["family"]=="inter_specific" and "_to:" in feature:
                prefix, target = feature.split("_to:",1)
                target = mapping.get(model,{}).get(target,model+":"+target)
                feature = prefix+"_to:"+target
            weight = row["denominator"] if row["denominator"] is not None and row["denominator"]>0 else 1.
            records.append((identity["panel_id"],cohort,model,identity["fold"],identity["replicate"],identity["subject"],
                identity["recording"],group,condition,meta.get("drug",""),row["epoch"],row["family"],feature,scope,
                row["parent_id"],row["value"],weight,row["n_support"],int(row["detector_consumed"])))
        db.executemany("INSERT INTO observations VALUES("+",".join("?" for _ in range(19))+")",records)
        record_count += 1
        total_calls += result["selected_calls"]
        if cohort=="jonathan":
            transfer_input.append((identity,[r for r in payload["observations"] if r["epoch"]=="pre_injection_15min"
                and r["family"] in {"support","recruitment","primitive","intra","inter"}]))
    db.commit()
    # Rate/probability denominators pool within seed first; seed repetitions then average.
    # Entropy/diversity and other nonadditive features are weighted recording summaries.
    seed_table = pd.read_sql_query("""SELECT panel,cohort,fold,replicate,subject,group_name,condition,drug,epoch,family,
        feature,scope,parent,SUM(value*weight)/SUM(weight) AS value,SUM(n_support) AS n_support,
        COUNT(DISTINCT recording) AS recordings,MAX(circular) AS circular
        FROM observations GROUP BY panel,cohort,fold,replicate,subject,group_name,condition,drug,epoch,family,feature,scope,parent""",db)
    db.close()
    for column in seed_table.select_dtypes(include="object"):
        seed_table[column] = seed_table[column].astype("category")
    identity_columns = ["panel","cohort","fold","subject","group_name","condition","drug","epoch","family","feature","scope","parent"]
    mice = seed_table.groupby(identity_columns,observed=True,as_index=False).agg(
        value=("value","mean"),seed_sd=("value","std"),seeds=("replicate","nunique"),
        n_support=("n_support","mean"),recordings=("recordings","max"),circular=("circular","max"))
    mice.to_csv(output/"mouse_readouts.csv.gz",index=False,compression="gzip")
    contrasts = []
    measure = ["panel","cohort","family","feature","scope","parent"]
    for ordinal,(key, sub) in enumerate(mice.groupby(measure,observed=True)):
        base = dict(zip(measure,key))
        first,second = ("Monomer","PFF") if base["cohort"]=="karen" else ("WT","KI")
        for (condition,epoch), part in sub.groupby(["condition","epoch"],observed=True):
            result = stratified_group_contrast(part.value,part.group_name,part.fold,
                first=first,second=second,seed=101+ordinal,resamples=resamples)
            contrasts.append(dict(base,comparison="between_group",condition=condition,epoch=epoch,
                first=first,second=second,**result))
        # Longitudinal changes are paired by named animal identity.
        if base["cohort"]=="karen":
            pairs = [("3w_PI","8w_PI"),("8w_PI","16w_PI"),("16w_PI","24w_PI"),("3w_PI","24w_PI")]
            for before,after in pairs:
                table = sub[sub.epoch=="whole_supported"].pivot_table(index=["subject","group_name","fold"],columns="condition",values="value",observed=True)
                if before not in table or after not in table:
                    continue
                change = (table[after]-table[before]).dropna().rename("value").reset_index()
                contrasts.append(dict(base,comparison="group_difference_in_wave_change",condition=after+" minus "+before,
                    epoch="whole_supported",first=first,second=second,**stratified_group_contrast(
                        change.value,change.group_name,change.fold,first=first,second=second,seed=101+ordinal,resamples=resamples)))
        else:
            table = sub.pivot_table(index=["subject","group_name","fold","condition"],columns="epoch",values="value",observed=True)
            pre,post = "pre_injection_15min","post_injection_2to17min"
            if pre not in table or post not in table:
                continue
            change = (table[post]-table[pre]).dropna().rename("value").reset_index()
            for condition,part in change.groupby("condition",observed=True):
                contrasts.append(dict(base,comparison="genotype_difference_in_pre_post_change",condition=condition,
                    epoch="post minus pre",first=first,second=second,**stratified_group_contrast(
                        part.value,part.group_name,part.fold,first=first,second=second,seed=101+ordinal,resamples=resamples)))
                for group,animals in part.groupby("group_name",observed=True):
                    result = paired_contrast(animals.value,seed=101+ordinal,resamples=resamples)
                    contrasts.append(dict(base,comparison="within_mouse_pre_post",condition=condition,
                        epoch="post minus pre",first=group,second=group,**result))
            drugs = change.pivot_table(index=["subject","group_name","fold"],columns="condition",values="value",observed=True)
            if "Saline" in drugs and "Amphetamine" in drugs:
                did = (drugs["Amphetamine"]-drugs["Saline"]).dropna().rename("value").reset_index()
                contrasts.append(dict(base,comparison="genotype_difference_in_paired_drug_change",condition="Amphetamine minus Saline",
                    epoch="difference in pre-post change",first=first,second=second,**stratified_group_contrast(
                        did.value,did.group_name,did.fold,first=first,second=second,seed=101+ordinal,resamples=resamples)))
                for group,part in did.groupby("group_name",observed=True):
                    contrasts.append(dict(base,comparison="within_mouse_paired_drug_change",condition="Amphetamine minus Saline",
                        epoch="difference in pre-post change",first=group,second=group,
                        **paired_contrast(part.value,seed=101+ordinal,resamples=resamples)))
    # Multiplicity spans configurations, individual types and constituent measures.
    families = defaultdict(list)
    for i,row in enumerate(contrasts):
        families[(row["cohort"],row["family"],row["comparison"],row["condition"],row["epoch"])].append(i)
    for indices in families.values():
        q = adjust_fdr([contrasts[i].get("p_permutation",np.nan) for i in indices])
        for i,value in zip(indices,q):
            contrasts[i]["q_family_all_settings"] = float(value) if np.isfinite(value) else None
    _write_csv(output/"phenotype_contrasts.csv",contrasts)
    transfer = _transfer(transfer_input)
    _write_csv(output/"baseline_transfer.csv",transfer)
    physical_rows = []
    for model,physical in physical_models.items():
        identity = physical["identity"]
        names = identity.get("feature_names",[])
        consumed = set(identity.get("construction_features",identity.get("consumed_features",[])))
        for mode,kind,allowed in (("all","profile",names),("source_conditioned","residual",names),
                ("unused","profile",[n for n in names if n not in consumed]),
                ("unused_source_conditioned","residual",[n for n in names if n not in consumed]),
                ("strict_source_conditioned","residual",[n for n in names if n not in {"duration_s","path_length_px"}])):
            columns = [phase*len(names)+i for phase in range(3) for i,n in enumerate(names) if n in allowed]
            score = physical_from_moments(physical[kind],columns)
            omissions = [physical_from_moments(physical[kind],columns,omit_subject=subject)["separation"]
                         for subject in sorted({s for t,s in physical[kind]})]
            finite = [v for v in omissions if v is not None]
            physical_rows.append(dict(panel_id=identity["panel_id"],cohort=identity["cohort"],
                model_key=model,fold=identity["fold"],replicate=identity["replicate"],mode=mode,channels=len(allowed),
                **score,mouse_omission_min=min(finite) if finite else None,mouse_omission_max=max(finite) if finite else None,
                unavailable_omissions=len(omissions)-len(finite)))
    _write_csv(output/"physical_consistency.csv",physical_rows)
    summary = dict(completed_at_utc=now_utc(),panel_hash=plan["panel_hash"],job_counts=counts,
        recording_evaluations=record_count,selected_calls=total_calls,mouse_readout_rows=len(mice),
        contrast_rows=len(contrasts),tested_rows=sum(r["status"]=="tested" for r in contrasts),
        accepted_correspondences=sum(r["accepted"] for r in crosswalk),correspondence_rows=len(crosswalk),
        interpretation="exploratory configuration-level axes, no weighted truth score or method-family recipe average",
        limitations=["Five Karen reserved animals unused; exploration/selection exposure remains.",
            "Source-phase matches are approximate; unsupported IDs and HSMM stages stay fit-local.",
            "Intra matching is observational and input-consumed channels are circular diagnostics.",
            "Inter nulls condition on selected call slots and do not measure detector false positives.",
            "Genotype/drug, age/wave and fixed session-order/date confounds retain their original meaning.",
            "No calibrated arena geometry; occupancy matching bins have no wall/centre interpretation.",
            "DA trace/timestamp audit and eligible WT associations are separate; no DA result is implied.",
            "Qualitative video review and prospective next-bout prediction are pending."])
    atomic_write_json(output/"summary.json",summary)
    atomic_write_text(output/"report.md",f"""# Fixed generator verification checkpoint

Completed {record_count} recording evaluations from {len(anchors)} saved fits,
covering {len({j['panel_id'] for j,r in anchors})} fixed settings. {total_calls:,} selected calls; {len(mice):,}
mouse/epoch readouts; {summary['tested_rows']:,} supported exploratory contrast rows.
The random search stayed stopped. No KPMS, codebook or chain model was fitted.

`mouse_readouts.csv.gz` contains individual-animal values after denominator pooling
within detector seed and averaging repeated seeds. `phenotype_contrasts.csv` gives
same-unit group and paired condition effects, animal counts, intervals and family
BH across all fixed settings. Group tests preserve held-fold strata. Chain-general
summaries and primitive baselines use all eligible held animals; unsupported
chain-specific comparisons remain explicitly insufficient. Nonadditive entropy/
diversity values are recording summaries, not entropies of a pooled recording.

`catalogue_correspondence.csv` uses training-only source distributions in three
equal bout-slot portions, cosine>=.85 and both ambiguity margins>=.10, with>=3
training calls. It does not establish semantic or HSMM phase homology. Local
phase identities remain local. `baseline_transfer.csv` compares the two Jonathan
pre-injection sessions with between-mouse same-genotype, same-fit controls and a
primitive frequency comparator. This is baseline transfer under fixed date/order,
not repeated identical-condition trait reliability.

The per-recording compressed artifacts retain raw/exclusive calls, native support,
denominators, execution matching, partial annotation coverage/enrichment, HSMM
phase/completion evidence and{plan.get('null_repeats',499)}-draw primary-gap controls with Monte Carlo SE.
Nulls preserve selected-call slots and slow five-minute regimes, not detector input.
Calls crossing an epoch edge are counted in support diagnostics and excluded,
never shortened. Pixel features remain pixels; training-median occupancy quadrants
are descriptive matching strata. DA has no outcome in this checkpoint.

All phenotype results are exploratory after prior data exposure and selection.
Assess individual complete settings across coherence, coverage, stability and
these added axes; no scalar truth score or mean across random method recipes.
Next: WT DA timestamp/trace eligibility and associations, support/influence review,
then matched qualitative grounding of promising and discordant candidates.
""")
    return summary


def _transfer(inputs):
    """Same-fit, same-genotype baseline frequency transfer with primitive comparator."""
    import numpy as np

    fits = defaultdict(dict)
    for identity, observations in inputs:
        rows = [r for r in observations if r["epoch"]=="pre_injection_15min" and r["value"] is not None]
        exposure = next((r["value"] for r in rows if r["family"]=="support" and r["feature"]=="model_exposure_s"),0)
        if exposure < 600:
            continue
        for family in ("recruitment","primitive"):
            values = {(r["type_id"] if family=="recruitment" else r["parent_id"]):r["value"] for r in rows
                if r["family"]==family and r["feature"]=="calls_per_min" and r["type_id"]!="__all__"}
            if family=="primitive":
                values = {r["parent_id"]:r["value"] for r in rows if r["family"]==family and r["feature"]=="calls_per_min"}
            key = (identity["panel_id"],identity["model_key"],identity["metadata"]["genotype"],family)
            fits[key].setdefault(identity["subject"],{})[identity["metadata"]["drug"]] = values
    result = []
    for (panel,model,group,family),animals in fits.items():
        vocabulary = sorted({t for recs in animals.values() for p in recs.values() for t in p})
        vectors = {s:{d:np.array([p.get(t,0.) for t in vocabulary]) for d,p in recs.items()} for s,recs in animals.items()}
        def cosine(a,b):
            denominator = np.linalg.norm(a)*np.linalg.norm(b)
            return float(a@b/denominator) if denominator>0 else np.nan
        for subject,recs in vectors.items():
            if "Saline" not in recs or "Amphetamine" not in recs:
                continue
            within = cosine(recs["Saline"],recs["Amphetamine"])
            other = [cosine(recs[a],p[b]) for s,p in vectors.items() if s!=subject for a,b in
                     (("Saline","Amphetamine"),("Amphetamine","Saline")) if b in p]
            other = [v for v in other if math.isfinite(v)]
            between = float(np.mean(other)) if other else None
            result.append(dict(panel_id=panel,model_key=model,subject=subject,group=group,family=family,
                within_similarity=within if math.isfinite(within) else None,between_similarity=between,
                within_minus_between=within-between if between is not None and math.isfinite(within) else None,
                other_mouse_comparisons=len(other),interpretation="matched pre-injection baseline transfer; fixed session date/order"))
    return result
