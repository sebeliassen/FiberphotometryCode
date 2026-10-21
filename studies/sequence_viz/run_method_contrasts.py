"""Explicit bounded categorical pilot on verified native KPMS bout windows.

No empirical work happens on import or without --execute. The JSON manifest
selects sources and animal splits; this runner never fits KPMS, guesses native
frame conversions, uses DA, or selects a winner from downstream phenotype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    """Read explicit policy without loading numerical inputs or creating outputs."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("cohorts"):
        raise ValueError("require schema_version 1 and nonempty cohorts")
    cohort_ids = set()
    for cohort in manifest["cohorts"]:
        name = cohort["cohort_id"]
        if not name or name in {".", ".."} or name in cohort_ids or Path(name).name != name:
            raise ValueError("cohort_id must be a unique filesystem-safe name")
        cohort_ids.add(name)
        train, held = set(cohort["train_subjects"]), set(cohort["held_out_subjects"])
        if not train or not held or train & held:
            raise ValueError("training and held-out subject sets must be nonempty and disjoint")
        if not cohort["model_id"] or not cohort["clock"] or cohort["fps"] <= 0:
            raise ValueError("explicit model identity, clock and positive fps are required")
        vocabulary = cohort["vocabulary"]
        if not vocabulary or len(set(vocabulary)) != len(vocabulary) or any(str(int(t)) != t or int(t) < 0 for t in vocabulary):
            raise ValueError("vocabulary must contain unique nonnegative raw integer label identities as strings")
        if not cohort["sessions"] or len(cohort["sessions"]) > 64:
            raise ValueError("require 1–64 explicitly selected sessions per cohort")
        recordings, subjects = set(), set()
        for session in cohort["sessions"]:
            if session.get("native_mapping_verified") is not True or not session.get("native_mapping_evidence"):
                raise ValueError("native mapping requires explicit verified evidence")
            if session["recording"] in recordings or session["subject"] not in train | held:
                raise ValueError("recording identities must be unique and subjects must belong to the split")
            recordings.add(session["recording"])
            subjects.add(session["subject"])
            source = Path(session["npz_path"])
            source = source if source.is_absolute() else path.parent/source
            if not source.is_file():
                raise ValueError("missing selected NPZ input: " + str(source))
            session["npz_path"] = str(source.resolve())
        if subjects != train | held:
            raise ValueError("every declared animal must have an explicitly selected session")
    return manifest


def _load_stream(cohort, session, max_input_bytes):
    import numpy as np

    from fpcode.behavior.unit_adapters import from_frame_labels
    from fpcode.contracts import SessionKey, UnitType

    path = Path(session["npz_path"])
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > max_input_bytes:
            raise ValueError("selected NPZ uncompressed input exceeds the declared byte cap")
    digest = _sha256(path)
    if session.get("sha256") is not None and session["sha256"] != digest:
        raise ValueError("selected NPZ fingerprint differs from the verified manifest")
    with np.load(path, allow_pickle=False) as data:
        frames = data[session.get("frames_key", "native_frames")]
        labels = data[session.get("labels_key", "labels")]
    stream = from_frame_labels(frames, labels, session=SessionKey(session["recording"], session["subject"], cohort["cohort_id"]),
                               model_id=cohort["model_id"], clock=cohort["clock"], fps=cohort["fps"],
                               catalog=tuple(UnitType(t) for t in cohort["vocabulary"]))
    return stream, digest


def select_native_window(stream, *, event_limit, rng, minimum_events=6):
    """Select one contiguous block of whole original bouts; never fill a gap."""
    from fpcode.behavior.unit_adapters import occurrence_blocks
    from fpcode.contracts import FrameSpan

    blocks = [block for block in occurrence_blocks(stream) if len(block) >= minimum_events]
    if not blocks or event_limit < minimum_events:
        raise ValueError("selected animal/recording has no supported bounded chain window")
    sufficient = [block for block in blocks if len(block) >= event_limit]
    longest = max(map(len, blocks))
    eligible = sufficient if sufficient else [block for block in blocks if len(block) == longest]
    block = eligible[int(rng.integers(len(eligible)))]
    count = min(event_limit, len(block))
    onset = int(rng.integers(len(block)-count+1))
    events = tuple(block[onset:onset+count])
    return replace(stream, occurrences=events, valid_intervals=(FrameSpan(events[0].interval.start, events[-1].interval.stop),))


def propose_exact_motifs(training, *, min_prevalence, maximum_per_role=4):
    """Animal-balanced frequency controls and diverse composition proposals.

    Diversity here means >=3 categorical lower types and no exact repeated block,
    not validated separate semantic actions. Preserve both proposal roles and
    disclose these conditions instead of treating frequent loops as bad behaviour.
    """
    from fpcode.behavior.episodes.chains import ngram_counts
    from fpcode.behavior.unit_adapters import primitive_bout_table

    by_animal = {}
    vocabulary = tuple(t.type_id for t in training[0].catalog)
    for stream in training:
        counts = ngram_counts(primitive_bout_table(stream)[0], lengths=(3, 4, 5, 6))
        by_animal.setdefault(stream.session.subject, Counter()).update(counts)
    pooled, support = Counter(), Counter()
    for counts in by_animal.values():
        total = max(sum(counts.values()), 1)
        for motif, count in counts.items():
            pooled[motif] += count/total
            support[motif] += 1
    eligible = sorted((motif for motif in pooled if support[motif] >= min_prevalence), key=lambda m: (-pooled[m], m))
    if not eligible:
        raise ValueError("no exact proposal meets the disclosed training-animal support")
    motifs, evidence = {}, []
    proposals = [("frequent", motif) for motif in eligible[:maximum_per_role]]
    diverse = [m for m in eligible if len(set(m)) >= 3 and not any(len(m) % p == 0 and m == m[:p]*(len(m)//p) for p in range(1, len(m)))]
    selected = []
    for length in (3, 4, 5, 6):
        choice = next((m for m in diverse if len(m) == length), None)
        if choice is not None and len(selected) < maximum_per_role:
            selected.append(choice)
    proposals.extend(("diverse", motif) for motif in selected)
    registered = {}
    for i, (role, motif) in enumerate(proposals):
        if motif in registered:
            registered[motif]["roles"].append(role)
            continue
        name = f"{role}_{i:02d}"
        motifs[name] = tuple(vocabulary[token] for token in motif)
        record = dict(type_id=name, roles=[role], tokens=motifs[name], length=len(motif),
                      training_animals=support[motif], animal_balanced_frequency=pooled[motif])
        evidence.append(record)
        registered[motif] = record
    return motifs, evidence


def _configured_hsmm(training, motif, model_id, memory_budget_bytes):
    import numpy as np

    from fpcode.behavior.episodes.staged import StagedEpisodeModel

    catalogue = training[0].catalog
    counts = Counter(event.type_id for stream in training for event in stream.occurrences)
    total = sum(counts.values())+len(catalogue)
    background = {u.type_id: (counts[u.type_id]+1)/total for u in catalogue}
    phases = tuple(f"phase_{i}" for i in range(len(motif)))
    emissions = {phase: {key: .25*p+.75*(key == token) for key, p in background.items()}
                 for phase, token in zip(phases, motif)}
    bg_duration = .15*.85**np.arange(16)
    bg_duration /= bg_duration.sum()
    model = StagedEpisodeModel.create_linear_routine(model_id, "training_proposed_routine", phases, catalogue, emissions, background,
                                                    phase_durations={p: np.array([.85, .15]) for p in phases},
                                                    background_duration=bg_duration, allow_immediate_reentry=True)
    return replace(model, memory_budget_bytes=memory_budget_bytes)


def _duration_sidecar(stream):
    """An observed bout-duration sidecar, unused by all categorical detectors."""
    import numpy as np

    from fpcode.contracts.behavior_results import FeatureAssociation
    from fpcode.contracts.signals import SignalMetadata, SignalSeries
    from fpcode.contracts.timelines import TimeAxis

    frames = np.array([event.interval.start for event in stream.occurrences], dtype=np.int64)
    values = np.array([(event.interval.stop-event.interval.start)/stream.fps for event in stream.occurrences])
    series = SignalSeries(TimeAxis(frames, unit="frame", clock=stream.clock), values,
                          (SignalMetadata("source_bout_duration", "s"),))
    return FeatureAssociation(series, stream.session, "native_video", provenance="Observed source bout duration at each onset; detectors do not consume this sidecar")


def run_pilot(manifest_path, output, *, seeds=(101, 202), selection_seed=73, train_event_limit=1200,
              held_event_limit=300, nmf_iterations=40, hsmm_iterations=8, memory_budget_bytes=256*1024*1024,
              max_input_bytes=128*1024*1024):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_info

    from fpcode.behavior.analysis.unit_views import (
        aggregate_unit_composition,
        measure_unit_composition,
        measure_unit_organisation,
    )
    from fpcode.behavior.episodes.seqnmf import SeqNMF
    from fpcode.behavior.episodes.staged import fit_staged_model
    from fpcode.behavior.unit_adapters import occurrence_blocks, occurrence_table, primitive_bout_table
    from fpcode.workflows.behavior_layers import compare_frozen_episode_methods
    from fpcode.workflows.sequence_atlas import one_hot_segments

    if len(seeds) != 2 or len(set(seeds)) != 2 or any(s < 0 for s in seeds):
        raise ValueError("pilot requires exactly two distinct nonnegative model seeds")
    if not 6 <= train_event_limit <= 1200 or not 6 <= held_event_limit <= 300:
        raise ValueError("pilot event limits must respect 1200 training/animal and 300 held/window caps")
    manifest = load_manifest(manifest_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(selection_seed)
    all_calls, all_composition, all_organisation, windows, efforts, phases, completion, held_diagnostics = [], [], [], [], [], [], [], []
    for cohort in manifest["cohorts"]:
        name = cohort["cohort_id"]
        target = output/name
        target.mkdir()
        training, held = [], []
        train_sessions = Counter(s["subject"] for s in cohort["sessions"] if s["subject"] in cohort["train_subjects"])
        for session in sorted(cohort["sessions"], key=lambda s: (s["subject"], s["recording"])):
            source, digest = _load_stream(cohort, session, max_input_bytes)
            is_train = session["subject"] in cohort["train_subjects"]
            limit = train_event_limit//train_sessions[session["subject"]] if is_train else held_event_limit
            window = select_native_window(source, event_limit=limit, rng=rng)
            (training if is_train else held).append(window)
            span = window.valid_intervals[0]
            source_blocks = occurrence_blocks(source)
            windows.append(dict(cohort=name, recording=session["recording"], subject=session["subject"],
                                split="train" if is_train else "held_out", start_frame=span.start, stop_frame=span.stop,
                                source_model_id=source.model_id, clock=source.clock, fps=source.fps, source_events=len(window.occurrences),
                                requested_events=limit, source_events_total=len(source.occurrences), source_contiguous_blocks=len(source_blocks),
                                source_longest_block_events=max(map(len, source_blocks)),
                                window_selection_policy="uniform among blocks supplying requested events, else uniform among longest blocks; uniform whole-bout onset",
                                input_sha256=digest, native_mapping_evidence=session["native_mapping_evidence"]))
        proposal_start = time.perf_counter()
        prevalence = max(1, (len(cohort["train_subjects"])+1)//2)
        motifs, proposals = propose_exact_motifs(training, min_prevalence=prevalence)
        (target/"exact_proposals.json").write_text(json.dumps(_jsonable(proposals), indent=2))
        animal_events = Counter()
        for stream in training:
            animal_events[stream.session.subject] += len(stream.occurrences)
        model_settings = dict(windows=[w for w in windows if w["cohort"] == name and w["split"] == "train"], seeds=seeds,
                              selection_seed=selection_seed, nmf_iterations=nmf_iterations, hsmm_iterations=hsmm_iterations,
                              runner_sha256=_sha256(__file__))
        fit_identity = hashlib.sha256(json.dumps(_jsonable(model_settings), sort_keys=True).encode()).hexdigest()[:12]
        namespace = f"{cohort['model_id']}/pilot_{fit_identity}"
        effort_base = dict(cohort=name, source_model_id=cohort["model_id"], train_subjects=cohort["train_subjects"],
                           training_recordings=[s.session.recording for s in training],
                           training_events=sum(len(s.occurrences) for s in training),
                           training_event_counts_by_animal=dict(animal_events),
                           train_event_limit_per_animal=train_event_limit, held_event_limit_per_window=held_event_limit,
                           representation="frozen original categorical KPMS bouts; no DA, pose or feature-conditioned learning")
        efforts.append({**effort_base, "method": "exact", "seconds": time.perf_counter()-proposal_start,
                        "learning": "training-animal-balanced ngram catalogue proposal", "iterations": 0,
                        "lengths": [3, 4, 5, 6], "min_training_animal_prevalence": prevalence,
                        "proposal_roles": ["frequency controls", "one >=3-type non-exact-loop proposal per length when supported"],
                        "seed": None, "catalogue_size": len(motifs)})
        routine = next((p for p in proposals if "diverse" in p["roles"]), proposals[0])
        configured_onset = time.perf_counter()
        configured = _configured_hsmm(training, routine["tokens"], f"{namespace}/hsmm_configured", memory_budget_bytes)
        np.savez_compressed(target/"hsmm_configured.npz", log_initial=configured.log_initial, log_transition=configured.log_transition,
                            log_emissions=configured.log_emissions, log_duration_pmfs=configured.log_duration_pmfs,
                            model_id=configured.model_id, primitive_type_ids=np.asarray(cohort["vocabulary"]),
                            phase_names=np.asarray(configured.phase_names), routine_name=configured.routine_name)
        efforts.append({**effort_base, "method": "hsmm_configured", "learning": "no iterative optimizer",
                        "parameter_source": "routine proposal and smoothed background frequencies derived from training; phase topology/duration rules specified",
                        "seconds": time.perf_counter()-configured_onset,
                        "seed": None, "iterations": 0, "routine_proposal": routine,
                        "phase_duration_events": [.85, .15], "background_duration_events": np.exp(configured.log_duration_pmfs[0]).tolist(),
                        "catalogue_size": 1, "limitation": "one specified target routine, not unsupervised multi-routine catalogue discovery"})
        fitted = {}
        for seed in seeds:
            onset = time.perf_counter()
            fit = fit_staged_model(configured, training, model_id=f"{namespace}/hsmm_fitted_seed{seed}",
                                   source_model_id=cohort["model_id"], max_iterations=hsmm_iterations, seed=seed,
                                   initialization="dirichlet", max_observations=1200*len(cohort["train_subjects"]),
                                   memory_budget_bytes=memory_budget_bytes)
            fitted[seed] = fit.model
            np.savez_compressed(target/f"hsmm_fitted_seed{seed}.npz", log_initial=fit.model.log_initial,
                                log_transition=fit.model.log_transition, log_emissions=fit.model.log_emissions,
                                log_duration_pmfs=fit.model.log_duration_pmfs, model_id=fit.model.model_id,
                                primitive_type_ids=np.asarray(cohort["vocabulary"]), phase_names=np.asarray(fit.model.phase_names),
                                routine_name=fit.model.routine_name)
            efforts.append({**effort_base, "method": f"hsmm_fitted_seed{seed}", "seconds": time.perf_counter()-onset,
                            "learning": "seeded categorical Viterbi reestimation; emissions and supported transitions; frozen durations/initial/topology",
                            "seed": seed, "fit_diagnostics": _jsonable(fit.diagnostics), "catalogue_size": 1,
                            "limitation": "one specified target routine; hard-path objective is not soft EM or global convergence"})
        for length in (3, 6):
            matrices, masks = [], []
            for stream in training:
                bouts, vocabulary = primitive_bout_table(stream)
                matrix, valid, _ = one_hot_segments(bouts, len(vocabulary), length)
                matrices.append(matrix)
                masks.append(valid)
            x, valid = np.concatenate(matrices, axis=1), np.concatenate(masks)
            estimate = 64*x.shape[1]*(len(vocabulary)+4+length)
            if estimate > memory_budget_bytes:
                raise ValueError("seqNMF training working-array estimate exceeds pilot RAM cap")
            for seed in seeds:
                onset = time.perf_counter()
                model = SeqNMF(4, length, penalty=.01, iterations=nmf_iterations, seed=seed).fit(x, valid_starts=valid)
                thresholds = np.maximum(np.quantile(model.h[:, valid], .9, axis=1), 1e-7)
                label = f"seqnmf_L{length}_seed{seed}"
                np.savez_compressed(target/f"{label}.npz", w=model.w, thresholds=thresholds, type_ids=np.asarray(vocabulary),
                                    model_id=f"{namespace}/{label}", penalty=.01, iterations=nmf_iterations, seed=seed)
                efforts.append({**effort_base, "method": label, "seconds": time.perf_counter()-onset,
                                "learning": "train-fitted full-template categorical seqNMF; frozen templates/thresholds on held animals",
                                "seed": seed, "iterations": nmf_iterations, "factors": 4, "length_events": length,
                                "penalty": .01, "loss_history": model.loss_history,
                                "threshold_rule": "max(training supported activation quantile .90, 1e-7)",
                                "initialization": "seeded positive random templates plus sampled supported training windows",
                                "inference_iterations": 35, "estimated_working_bytes": estimate,
                                "dtype": "float32", "limitation": "fixed-iteration pilot; reference-MATLAB equivalence and convergence not established"})
                for stream in held:
                    staged = (configured, fitted[seed]) if length == 3 and seed == seeds[0] else ((fitted[seed],) if length == 3 else ())
                    onset = time.perf_counter()
                    contrast = compare_frozen_episode_methods(stream, _duration_sidecar(stream), exact_motifs=motifs,
                                                              seqnmf_templates=model.w, seqnmf_type_ids=vocabulary,
                                                              seqnmf_thresholds=thresholds, staged_models=staged,
                                                              max_bouts=held_event_limit, memory_budget_bytes=memory_budget_bytes)
                    for method, result in contrast["results"].items():
                        if method == "primitives" or (method == "exact" and (length != 3 or seed != seeds[0])):
                            continue
                        identity = f"{namespace}/{label}" if method == "seqnmf" else (f"{namespace}/exact" if method == "exact" else result.model_id)
                        result = replace(result, stream=replace(result.stream, model_id=identity))
                        organisation = measure_unit_organisation(result.stream, gap_seconds=0, overlap_policy="score_ranked")
                        decisions = organisation["selection_audit"]
                        retained = set(decisions.loc[decisions.selected, "occurrence_id"])
                        selected = replace(result.stream, occurrences=tuple(o for o in result.occurrences if o.occurrence_id in retained))
                        held_diagnostics.append(dict(cohort=name, recording=stream.session.recording, subject=stream.session.subject,
                                                     model_id=identity, raw_calls=len(result.occurrences), selected_calls=len(selected.occurrences),
                                                     diagnostics=_jsonable(result.diagnostics)))
                        all_composition.append(measure_unit_composition(selected, stream))
                        calls = occurrence_table(result.stream)
                        calls["selected"] = calls.occurrence_id.isin(retained)
                        calls["cohort"], calls["recording"], calls["subject"], calls["model_id"] = name, stream.session.recording, stream.session.subject, identity
                        all_calls.append(calls)
                        all_organisation.append(dict(cohort=name, recording=stream.session.recording, subject=stream.session.subject,
                                                     model_id=identity, **organisation["counts"]))
                        for p in result.phases or ():
                            phases.append(dict(cohort=name, recording=stream.session.recording, model_id=identity,
                                               occurrence_id=p.parent_occurrence_id, phase=p.phase_id, order=p.order,
                                               start_frame=p.interval.start, stop_frame=p.interval.stop))
                        for c in result.completion or ():
                            completion.append(dict(cohort=name, recording=stream.session.recording, model_id=identity,
                                                   occurrence_id=c.occurrence_id, status=c.status,
                                                   left_censored=c.left_censored, right_censored=c.right_censored))
                    efforts.append(dict(cohort=name, stage="matched_held_window_inference", seqnmf_method=label,
                                        recording=stream.session.recording, seconds=time.perf_counter()-onset,
                                        inference_methods=list(contrast["results"]), resource_estimate_bytes=contrast["resource_estimate_bytes"]))
    # Empty catalogues remain in quality panels, without affecting concat dtypes.
    nonempty_calls = [table for table in all_calls if len(table)]
    for table in nonempty_calls:
        table["score"] = pd.to_numeric(table["score"]).astype("float64")
    (pd.concat(nonempty_calls, ignore_index=True) if nonempty_calls else all_calls[0]).to_csv(output/"calls.csv", index=False)
    nonempty_composition = [table for table in all_composition if len(table)]
    (pd.concat(nonempty_composition, ignore_index=True) if nonempty_composition else all_composition[0]).to_csv(output/"occurrence_composition.csv", index=False)
    summaries = aggregate_unit_composition(all_composition, max_pairs_per_type=128, seed=selection_seed)
    for key, table in summaries.items():
        table.to_csv(output/f"{key}_quality.csv", index=False)
    pd.DataFrame(all_organisation).to_csv(output/"organisation_counts.csv", index=False)
    pd.DataFrame(windows).to_csv(output/"matched_windows.csv", index=False)
    pd.DataFrame(phases).to_csv(output/"phases.csv", index=False)
    pd.DataFrame(completion).to_csv(output/"completion.csv", index=False)
    (output/"held_out_model_diagnostics.json").write_text(json.dumps(_jsonable(held_diagnostics), indent=2))
    effort = dict(schema_version=1, manifest_sha256=_sha256(manifest_path), runner_sha256=_sha256(__file__), settings=dict(seeds=seeds, selection_seed=selection_seed,
                  memory_budget_bytes=memory_budget_bytes, max_input_bytes=max_input_bytes), conditions=efforts,
                  thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                  observed_threadpools=threadpool_info(), limitations=["Bounded reused-data pilot; held animals remain useful but are not untouched confirmation.",
                  "First contrast is categorical; direct-feature routing, learned durations, multi-routine HSMM discovery and video coherence remain outstanding.",
                  "Different catalogue IDs are not compared across cohorts/models; no score combines methods or selects a biological winner.",
                  "Primary composition summaries use explicitly score-ranked nonoverlapping calls within each detector; all candidates remain in calls.csv."])
    (output/"effort.json").write_text(json.dumps(_jsonable(effort), indent=2))
    return effort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize this bounded empirical pilot")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-events-per-animal", type=int, default=1200)
    parser.add_argument("--held-events-per-window", type=int, default=300)
    parser.add_argument("--memory-budget-mib", type=int, default=256)
    parser.add_argument("--max-input-mib", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=73)
    parser.add_argument("--seeds", type=int, nargs=2, default=[101, 202])
    parser.add_argument("--nmf-iterations", type=int, default=40)
    parser.add_argument("--hsmm-iterations", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.threads, args.nmf_iterations, args.hsmm_iterations, args.memory_budget_mib, args.max_input_mib) < 1:
        parser.error("thread/iteration counts and memory caps must be positive")
    if not args.execute:
        manifest = load_manifest(args.manifest)
        print(json.dumps({"execute": False, "cohorts": [c["cohort_id"] for c in manifest["cohorts"]], "output": str(Path(args.output).resolve())}))
        return
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(min(args.threads, available))
    effort = run_pilot(args.manifest, args.output, seeds=tuple(args.seeds), selection_seed=args.selection_seed,
                       train_event_limit=args.train_events_per_animal, held_event_limit=args.held_events_per_window,
                       nmf_iterations=args.nmf_iterations, hsmm_iterations=args.hsmm_iterations,
                       memory_budget_bytes=args.memory_budget_mib*1024*1024, max_input_bytes=args.max_input_mib*1024*1024)
    print(json.dumps({"output": str(Path(args.output).resolve()), "recorded_conditions": len(effort["conditions"])}))


if __name__ == "__main__":
    main()
