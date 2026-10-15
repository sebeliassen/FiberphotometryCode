"""Finite, supervised verification of explicit saved candidate models.

Default planning is read-only. --execute reuses frozen components, never calls a
fit method, and only applies each fit to its originally held exploration animals.
Original training windows supply group-blind correspondence descriptors only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import zipfile
from collections import OrderedDict
from pathlib import Path

from studies.sequence_viz.run_broad_search import canonical, digest


def load_components(saved, cache):
    """Verify source evaluation and identity-checked data-only cache hits."""
    import numpy as np

    from fpcode.common.cache_manifest import cached_npz_component

    path = Path(saved["path"])
    if digest(path) != saved["sha256"]:
        raise ValueError("selected source evaluation changed")
    artifact = json.loads(path.read_text())
    if artifact["job"]["recipe"] != saved["exact_recipe"]:
        raise ValueError("selected exact recipe changed")
    result = artifact["result"]
    components = []
    for component in ("representation", "model"):
        key = saved["cache"][component+"_key"]
        if result["cache"][component+"_key"] != key:
            raise ValueError("source artifact/component key disagreement")
        target = Path(cache)/(key+".npz")
        # Never permit a missing, corrupt or stale component to trigger refitting.
        if not target.is_file() or target.is_symlink():
            raise ValueError("required frozen component is missing")
        with zipfile.ZipFile(target) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 32*1024**2:
                raise MemoryError("frozen component exceeds uncompressed byte cap")
        with np.load(target, allow_pickle=False) as data:
            identity = json.loads(str(data["__identity_json"]))
        if hashlib.sha256(canonical(identity).encode()).hexdigest() != key:
            raise ValueError("embedded frozen component identity differs from selected key")
        def forbid():
            raise ValueError("frozen cache unavailable; fitting is forbidden")
        arrays, hit, actual_key = cached_npz_component(cache, identity, forbid, max_uncompressed_bytes=32*1024**2)
        if not hit or actual_key != key:
            raise ValueError("frozen component was not a verified cache hit")
        components.append(arrays)
    representation = json.loads(str(components[0]["metadata"]))
    conditions = json.loads(str(components[1]["metadata"]))
    if representation != result["representation"] or conditions != result["conditions"]:
        raise ValueError("cached metadata differs from selected source evaluation")
    return representation, components[1], conditions, artifact


def build_plan(*, panel, manifest, cache, output, code_root=None, workers=4, threads=2,
               timeout_s=1800, null_repeats=499):
    from studies.sequence_viz.run_broad_search import build_search_plan, source_snapshot

    panel, manifest, cache = (Path(p).resolve() for p in (panel, manifest, cache))
    selection = json.loads(panel.read_text())
    inputs = json.loads(manifest.read_text())
    entries = selection["entries"]
    if not entries or len({e["panel_id"] for e in entries}) != len(entries):
        raise ValueError("unique, explicit panel entries required")
    root = Path(code_root or Path(__file__).resolve().parents[2]).resolve()
    plan = build_search_plan(manifest=manifest, output=output, code_root=root, recipes_per_cohort=1,
        recipes=[entries[0]["recipe"]], folds=3, replicates=1, workers=workers, threads=threads,
        trial_timeout_s=timeout_s, worker_address_space_gb=4, max_worker_rss_gb=2,
        min_free_disk_gb=80, min_free_ram_gb=8, cache_quota_gb=16, max_output_gb=16,
        null_repeats=null_repeats)
    allowed = {r["npz_path"] for c in inputs["cohorts"] for r in c["records"]
               if r["subject"] in c["exploration_subjects"] and not r.get("exclude_from_search", False)}
    # Reserved animal arrays are not opened even for the entry hash audit.
    plan["input_snapshot"] = [r for r in plan["input_snapshot"] if r["path"] in allowed or r["path"] == str(manifest)]
    extra_paths = {panel}
    anchors, record_lists = [], []
    for index, entry in enumerate(entries):
        cohort = next(c for c in inputs["cohorts"] if c["cohort_id"] == entry["cohort"])
        for saved in entry["saved_evaluations"]:
            held = set(cohort["folds"][saved["fold"]])
            if held & set(cohort["reserved_subjects"]) or not held <= set(cohort["exploration_subjects"]):
                raise ValueError("saved fit held assignment leaks excluded animals")
            path = Path(saved["path"])
            if digest(path) != saved["sha256"]:
                raise ValueError("selected source artifact hash changed")
            extra_paths.add(path)
            for component in ("representation", "model"):
                extra_paths.add(cache/(saved["cache"][component+"_key"]+".npz"))
            common = dict(cohort=entry["cohort"], base_recipe=index, panel_id=entry["panel_id"],
                panel_group=entry["group"], fold=saved["fold"], replicate=saved["replicate"],
                recipe=saved["exact_recipe"], saved=saved, source_job_id=saved["job_id"])
            def job(kind, recording=""):
                result = dict(common, kind=kind, recording=recording)
                result["job_id"] = hashlib.sha256(canonical(result).encode()).hexdigest()[:24]
                return result
            anchors.append(job("training_anchor"))
            record_lists.append([job("recording", r["recording"]) for r in sorted(cohort["records"], key=lambda r:r["recording"])
                                 if r["subject"] in held and not r.get("exclude_from_search", False)])
    # Every configuration starts getting broader coverage before long folds finish.
    recordings = [jobs[i] for i in range(max(map(len, record_lists))) for jobs in record_lists if i < len(jobs)]
    for path in sorted(extra_paths):
        stat = path.stat()
        plan["input_snapshot"].append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path)))
    plan.update(jobs=anchors+recordings, total_jobs=len(anchors)+len(recordings), panel=str(panel),
        panel_hash=digest(panel), frozen_component_cache=str(cache), source_snapshot=source_snapshot(root),
        enforce_output_quota=True, compare_seed_catalogues=False, write_reserve_gb=2,
        selection_policy="23 fixed settings; held-animal quantitative characterisation, exploratory phenotype axes",
        policies=dict(native_bouts="all prepared supported bouts; original gaps, no resampling or refit",
            reference="training-window descriptors; local IDs never pooled by number",
            epochs="Jonathan pre15min and post2-to17min; whole supported sessions separately",
            intra="source ID × five-minute stratum × training-median occupancy quadrant; >=3 inside/outside bouts",
            inter="gaps10/30s; native-gap and five-minute barriers; fixed-call-slot frequency/first-order controls",
            null_repeats=null_repeats, null_memory_mb=64, dtype="float64",
            multiplicity="mouse-level effects; family BH; all candidate selection effects exploratory",
            da="WT only; separate timestamp/trace audit before associations",
            qualitative="after this fixed quantitative round"))
    return plan


def initialize(plan):
    from studies.sequence_viz.run_broad_search import set_worker_allocation_cap

    set_worker_allocation_cap(plan["worker_address_space_gb"])
    from fpcode.workflows.sequence_search import predict_frozen_recipe
    from studies.sequence_viz.broad_search_inputs import load_full_held_records
    if Path(predict_frozen_recipe.__code__.co_filename).resolve() != Path(plan["code_root"])/"src/fpcode/workflows/sequence_search.py":
        raise ValueError("inference imported outside frozen runtime")
    if Path(load_full_held_records.__code__.co_filename).resolve() != Path(plan["code_root"])/"studies/sequence_viz/broad_search_inputs.py":
        raise ValueError("input loader imported outside frozen runtime")
    from threadpoolctl import threadpool_info

    return dict(plan=plan, manifest=json.loads(Path(plan["manifest"]).read_text()), cache=OrderedDict(),
                threadpools=threadpool_info())


def _state_inputs(state, job):
    from studies.sequence_viz.assess_broad_search import training_context
    from studies.sequence_viz.broad_search_inputs import load_full_held_records, load_job_records

    key = (job["cohort"], job["fold"])
    cached = state["cache"].pop(key, None)
    if cached is None:
        sampled = load_job_records(state["manifest"], *key, per_animal=600, held_per_animal=180, window_seed=73)
        held = load_full_held_records(state["manifest"], *key)
        cached = (sampled, held, training_context(sampled))
    state["cache"][key] = cached
    while len(state["cache"]) > 2:
        state["cache"].popitem(last=False)
    return cached


def _gzip_json(path, value):
    from fpcode.common.cache_manifest import atomic_path
    from fpcode.workflows.sequence_search import _finite_json

    with atomic_path(path) as temporary:
        with gzip.open(temporary, "wt") as handle:
            json.dump(_finite_json(value), handle, separators=(",", ":"), allow_nan=False)


def _prototype(record, prediction, vocabulary):
    """Original primitive distributions in three equal bout-slot portions."""
    import numpy as np

    index = {token:i for i, token in enumerate(vocabulary)}
    result = {}
    for call in prediction["selected_calls"]:
        a = np.searchsorted(record["start_frames"], call["start_frame"])
        b = np.searchsorted(record["stop_frames"], call["stop_frame"], side="right")
        counts = np.zeros((3, len(vocabulary)), dtype=float)
        for phase, positions in enumerate(np.array_split(np.arange(a, b), 3)):
            for parent in record["parent_ids"][positions]:
                counts[phase, index[str(parent)]] += 1
            if counts[phase].sum():
                counts[phase] /= counts[phase].sum()
        result.setdefault(call["type_id"], []).append(counts.ravel())
    return result


def evaluate(state, job):
    import numpy as np

    from fpcode.workflows.sequence_search import predict_frozen_recipe
    from studies.sequence_viz.assess_broad_search import call_profiles
    from studies.sequence_viz.verification_readouts import recording_readouts

    plan, saved = state["plan"], job["saved"]
    representation, arrays, conditions, source = load_components(saved, plan["frozen_component_cache"])
    sampled, held, context = _state_inputs(state, job)
    cohort = next(c for c in state["manifest"]["cohorts"] if c["cohort_id"] == job["cohort"])
    kwargs = dict(representation_id=saved["cache"]["representation_key"], model_id=saved["cache"]["model_key"],
                  max_working_bytes=512*1024**2)
    if job["kind"] == "training_anchor":
        # Verify every original held window before broader application of this fit.
        expected = {r["recording"]:r for r in source["result"]["records"]}
        parity = []
        for record in sampled:
            if record["split"] != "held":
                continue
            predicted = predict_frozen_recipe(record, job["recipe"], representation, arrays, conditions, **kwargs)
            reference = expected[record["recording"]]
            for name in ("raw_calls", "selected_calls", "valid_intervals", "source_exposure_s", "available_exposure_s"):
                if predicted[name] != reference[name]:
                    raise ValueError("saved-window reproduction failed: "+name)
            parity.append(record["recording"])
        profiles, n_calls = {}, {}
        for record in sampled:
            if record["split"] != "train":
                continue
            predicted = predict_frozen_recipe(record, job["recipe"], representation, arrays, conditions, **kwargs)
            for token, values in _prototype(record, predicted, cohort["vocabulary"]).items():
                profiles.setdefault(token, []).extend(values)
        for token in profiles:
            n_calls[token] = len(profiles[token])
            profiles[token] = np.mean(profiles[token], axis=0).tolist()
        return dict(status="SUCCESS", kind=job["kind"], panel_id=job["panel_id"], source_job_id=job["source_job_id"],
            vocabulary=cohort["vocabulary"], prototypes=profiles, training_calls=n_calls,
            saved_window_parity=parity, model_key=saved["cache"]["model_key"], operational_conditions=dict(threadpools=state["threadpools"]))
    record = next(r for r in held if r["recording"] == job["recording"])
    anchor_job = dict(job, kind="training_anchor", recording="")
    anchor_job.pop("job_id")
    anchor_id = hashlib.sha256(canonical(anchor_job).encode()).hexdigest()[:24]
    certificate = Path(plan["output"])/"jobs"/(anchor_id+".json")
    if certificate.is_file():
        check = json.loads(certificate.read_text())
        if check["result"].get("model_key") != saved["cache"]["model_key"] or not check["result"].get("saved_window_parity"):
            raise ValueError("frozen fit reproduction certificate differs")
    else:
        # A slow anchor can still be running when the first recording dispatches.
        expected = {r["recording"]:r for r in source["result"]["records"]}
        for sample in sampled:
            if sample["split"] != "held":
                continue
            check = predict_frozen_recipe(sample, job["recipe"], representation, arrays, conditions, **kwargs)
            for name in ("raw_calls", "selected_calls", "valid_intervals"):
                if check[name] != expected[sample["recording"]][name]:
                    raise ValueError("saved-window reproduction failed before full inference")
    metadata = next(r["metadata"] for r in cohort["records"] if r["recording"] == job["recording"])
    predicted = predict_frozen_recipe(record, job["recipe"], representation, arrays, conditions, **kwargs)
    predicted["construction_features"] = source["result"]["quality"]["physical"]["features_used_in_construction"]
    predicted["annotation_input"] = job["recipe"]["resolution"] == "annotation"
    n_types = (job["recipe"]["factors"] if job["recipe"]["method"] == "seqnmf" else
               len(conditions.get("targets", [])) if job["recipe"]["method"] == "hsmm" else len(conditions.get("motifs", [])))
    catalog = [("factor_" if job["recipe"]["method"] == "seqnmf" else "routine_")+str(i) for i in range(n_types)]
    readouts = recording_readouts(record, predicted, metadata, catalog, cohort["annotation_mapping"], context,
        null_repeats=plan["null_repeats"], seed=int(job["job_id"][:8], 16))
    profiles = call_profiles(record, predicted["selected_calls"], context)
    physical = {}
    for token in catalog:
        own = [p for p in profiles if p["type_id"] == token]
        if not own:
            continue
        result = dict(calls=len(own))
        for kind in ("profile", "residual"):
            data = np.asarray([p[kind] for p in own])
            finite = np.isfinite(data)
            safe = np.where(finite, data, 0.)
            result[kind] = dict(count=finite.sum(axis=0).tolist(), sum=safe.sum(axis=0).tolist(),
                sum_squared=(safe**2).sum(axis=0).tolist())
        physical[token] = result
    readouts["physical_moments"] = physical
    path = Path(plan["output"])/"recordings"/(job["job_id"]+".json.gz")
    identity = dict(panel_id=job["panel_id"], cohort=job["cohort"], source_job_id=job["source_job_id"],
        model_key=saved["cache"]["model_key"], fold=job["fold"], replicate=job["replicate"],
        recording=record["recording"], subject=record["subject"], metadata=metadata,
        input_sha256=record["input_sha256"], consumed_features=predicted["consumed_features"],
        construction_features=predicted["construction_features"],
        feature_names=list(record["feature_names"]), feature_scales=context["scale"].tolist())
    _gzip_json(path, dict(identity=identity, prediction=predicted, **readouts))
    return dict(status="SUCCESS", kind=job["kind"], **identity, recording_artifact=str(path), sha256=digest(path),
        source_bouts=len(record["parent_ids"]), selected_calls=len(predicted["selected_calls"]),
        readout_rows=len(readouts["observations"]), available_exposure_s=predicted["available_exposure_s"],
        operational_conditions=dict(threadpools=state["threadpools"], dtype="float64", null_memory_mb=64,
            pid=os.getpid(), worker_address_space_gb=plan["worker_address_space_gb"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "manifest", "cache", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--null-repeats", type=int, default=499)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="One fit and longest original held recording per setting")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume and (output/"run_manifest.json").is_file():
        plan = json.loads((output/"run_manifest.json").read_text())
        if str(args.panel.resolve()) != plan["panel"] or str(args.manifest.resolve()) != plan["manifest"]:
            raise ValueError("resume must name the original frozen panel/inputs")
    else:
        plan = build_plan(panel=args.panel, manifest=args.manifest, cache=args.cache, output=args.output,
            code_root=args.code_root, workers=args.workers, threads=args.threads,
            timeout_s=args.timeout_s, null_repeats=args.null_repeats)
        if args.preflight:
            cohorts = {c["cohort_id"]:c for c in json.loads(args.manifest.read_text())["cohorts"]}
            anchors = [j for j in plan["jobs"] if j["kind"]=="training_anchor" and j["fold"]==0 and j["replicate"]==0]
            selected = []
            for anchor in anchors:
                lengths = {r["recording"]:r["bout_count"] for r in cohorts[anchor["cohort"]]["records"]}
                eligible = [j for j in plan["jobs"] if j["kind"]=="recording" and j["source_job_id"]==anchor["source_job_id"]]
                selected.append(max(eligible,key=lambda j:lengths[j["recording"]]))
            plan.update(jobs=anchors+selected,total_jobs=len(anchors)+len(selected),preflight=True)
    if not args.execute:
        print(canonical(plan))
        return 0
    from studies.sequence_viz.run_broad_search import execute_search

    outcome = execute_search(plan, resume=args.resume, initializer=initialize, evaluator=evaluate)
    if not plan.get("preflight") and outcome["state"] == "COMPLETED" and set(outcome["counts"]) <= {"SUCCESS"}:
        from fpcode.common.cache_manifest import atomic_write_json
        from studies.sequence_viz.run_broad_search import set_worker_allocation_cap
        from studies.sequence_viz.summarize_generator_verification import summarize

        set_worker_allocation_cap(8)
        atomic_write_json(output/"aggregation_status.json", dict(state="RUNNING",address_space_cap_gb=8))
        try:
            summary = summarize(output)
        except BaseException as exc:
            atomic_write_json(output/"aggregation_status.json",dict(state="FAILED",error=f"{type(exc).__name__}: {exc}"))
            raise
        atomic_write_json(output/"aggregation_status.json",dict(state="COMPLETED",summary=summary))
    return 0 if outcome["state"] == "COMPLETED" and set(outcome["counts"]) <= {"SUCCESS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
