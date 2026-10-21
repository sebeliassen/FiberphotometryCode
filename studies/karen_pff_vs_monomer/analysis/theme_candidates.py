"""Group-blind, bounded inventory of observed pooled Karen seqNMF theme cores.

No fit, video processing, phenotype comparison or pickle loading. The named cores
are exploratory refinements of actual observed paths, never template argmaxes.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def definitions():
    return [
        dict(id="travel_turn_travel", name="Travel–turn–travel", stage_sets=[[0, 1, 2, 3, 4, 5], [6, 7], [0, 1, 2, 3, 4, 5]], recommended=True,
             rationale="Pools labelled travel types and left/right turning as within-theme variants; complete observed three-stage loop.", caution="Direction and travel type remain intra-theme attributes. A turn loop is not automatically a wall-following episode."),
        dict(id="orientation_sniff_orientation", name="Orientation–sniff–orientation", stage_sets=[[8, 9, 10], [11, 12], [8, 9, 10]], recommended=True,
             rationale="Complete ordered sensory/orientation core, allowing direction and generic/directional sniff substitutions.", caution="Could share a broader sensory family with sniff–observe–sniff. These are provisional alternative resolutions, not proof of distinct behaviour."),
        dict(id="sniff_observe_sniff", name="Sniff–observe–sniff", stage_sets=[[11, 12], [13], [11, 12]], recommended=True,
             rationale="Repeated sensory investigation interrupted by the specifically annotated observe state; prominent observed L5 paths.", caution="Observe is human annotation of raw25; do not call this resting or stationary without kinematic/video evidence."),
        dict(id="rear_backoff", name="Rear–back-off", stage_sets=[[14], [15]], recommended=True,
             rationale="A rearing-associated root supported by complete named stages; all next-bout continuations form branch opportunities.", caution="No Karen raw syllable is labelled descent. Back-off must not be renamed descent; this is a two-bout root, not a proved multi-action routine."),
        dict(id="groom_reorient_groom", name="Groom–reorient–groom", stage_sets=[[17], [6, 7, 8, 9, 10, 13], [17]], recommended=True,
             rationale="Grooming recurrence with a brief labelled orientation/turn interruption; combines meaningful orientation variants.", caution="Does not pool scratch or scrunch with grooming. Short interruptions are not imposed by a post-hoc duration filter."),
        dict(id="orientation_scrunch_orientation", name="Orientation–scrunch–orientation", stage_sets=[[8, 9, 10, 13], [16], [8, 9, 10, 13]], recommended=False,
             rationale="Supported actual postural/orientation alternation; reserve candidate if it adds interpretable diversity.", caution="Scrunch is an uncertain supplied label. Do not infer sniffing or grooming; exclude from primary atlas until qualitative support."),
        dict(id="rest_sniff_rest", name="Rest–sniff–rest", stage_sets=[[19], [11, 12], [19]], recommended=False,
             rationale="Presentation-like sensory/rest loop, included as an honest low-support reserve.", caution="Weak support may make this unsuitable for group/branch inference. Do not choose it merely to resemble IFxDN theme C."),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l3-root", type=Path, required=True)
    parser.add_argument("--l5-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False, scope="group-blind saved-call and observed-path inventory", candidates=definitions())))
        return
    if args.threads < 1 or args.memory_gb <= 0:
        parser.error("positive thread/memory limits required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import pandas as pd
    import psutil

    process = psutil.Process()

    def guard():
        if process.memory_info().rss > args.memory_gb * 1024**3:
            raise MemoryError("candidate inventory RAM cap exceeded")

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "recommendation.json").exists():
        raise FileExistsError("refusing to overwrite frozen recommendations")
    source_manifest = json.loads(args.manifest.read_text())
    annotations = source_manifest["annotation_mappings"]["karen"]
    mapping_path = args.l5_root / "mapping.json"
    mapping = np.asarray(json.loads(mapping_path.read_text()), int)
    token_map = {int(token): dict(raw_ids=np.flatnonzero(mapping == token).tolist(),
                                 raw_labels={str(raw): annotations[str(raw)] for raw in np.flatnonzero(mapping == token)},
                                 name=" / ".join(sorted({annotations[str(raw)] for raw in np.flatnonzero(mapping == token)})),
                                 known=bool(token < 20)) for token in np.unique(mapping)}
    # Deliberately do not read condition, group or treatment fields.
    usecols = ["token", "start_frame", "end_frame_exclusive", "session_id", "animal_id", "chunk_id"]
    reference_path = args.l5_root / "fit_references.csv.gz"
    references = pd.read_csv(reference_path, usecols=usecols, dtype={"animal_id": str})
    reference_l3 = args.l3_root / "prepared/L3_unresolved_separate/fit_references.csv.gz"
    r3 = pd.read_csv(reference_l3, usecols=usecols, dtype={"animal_id": str}).dropna(subset=["token"])
    r5 = references.dropna(subset=["token"])
    if not np.array_equal(r3[["token", "start_frame", "end_frame_exclusive"]].to_numpy(), r5[["token", "start_frame", "end_frame_exclusive"]].to_numpy()):
        raise AssertionError("saved L3 and L5 grouped observations differ")
    if not np.array_equal(r3.session_id.to_numpy(), r5.session_id.to_numpy()):
        raise AssertionError("saved L3 and L5 recording order differs")
    refs = r5.copy()
    for field in ("token", "start_frame", "end_frame_exclusive", "chunk_id"):
        refs[field] = refs[field].astype(int)
    if refs.session_id.nunique() != 99 or refs.animal_id.nunique() != 21:
        raise AssertionError("unexpected pooled input scope")
    candidates = definitions()
    occurrence_rows, path_rows = [], []
    all_known_paths = {n: collections.Counter() for n in (2, 3, 4, 5)}
    known_path_mice = {n: collections.defaultdict(set) for n in all_known_paths}
    for chunk_id, table in refs.groupby("chunk_id", sort=False):
        guard()
        if table.session_id.nunique() != 1 or table.animal_id.nunique() != 1:
            raise AssertionError("chunk crosses recording/mouse")
        a = table.token.to_numpy(int)
        starts, stops = table.start_frame.to_numpy(int), table.end_frame_exclusive.to_numpy(int)
        if not np.array_equal(starts[1:], stops[:-1]):
            raise AssertionError("chunk bridges unavailable native frames")
        mouse, sid = str(table.animal_id.iloc[0]), table.session_id.iloc[0]
        for n, counts in all_known_paths.items():
            if len(a) < n:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(a, n)
            good = np.all(windows < 20, axis=1)
            for path in map(tuple, windows[good]):
                counts[path] += 1
                known_path_mice[n][path].add(mouse)
        for candidate in candidates:
            stages = candidate["stage_sets"]
            n = len(stages)
            if len(a) < n:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(a, n)
            mask = np.ones(len(windows), bool)
            for j, tokens in enumerate(stages):
                mask &= np.isin(windows[:, j], tokens)
            for start in np.flatnonzero(mask):
                path = tuple(int(t) for t in a[start:start + n])
                next_token = int(a[start + n]) if start + n < len(a) else None
                occurrence_rows.append(dict(theme=candidate["id"], session_id=sid, animal_id=mouse,
                                            chunk_id=int(chunk_id), grouped_reference_start=int(table.index[start]),
                                            start_frame=int(starts[start]), end_frame_exclusive=int(stops[start + n - 1]),
                                            duration_sec=float((stops[start + n - 1] - starts[start]) / 60.),
                                            path=json.dumps(path), next_token=next_token,
                                            continuation_status="block_end" if next_token is None else "known" if next_token < 20 else "unresolved_or_bad"))
    occurrences = pd.DataFrame(occurrence_rows)
    occurrences.to_csv(args.output / "cohort_core_occurrences.csv.gz", index=False)
    for n, counts in all_known_paths.items():
        for path, number in counts.most_common():
            path_rows.append(dict(length=n, path=json.dumps([int(t) for t in path]),
                                  named_path=" → ".join(token_map[int(t)]["name"] for t in path),
                                  occurrences=number, mice=len(known_path_mice[n][path])))
    pd.DataFrame(path_rows).to_csv(args.output / "all_known_observed_paths.csv.gz", index=False)
    sources, support_rows, factor_path_rows = [], [], []
    for length, root in ((3, args.l3_root), (5, args.l5_root)):
        for k in (10, 18):
            folder = root / f"K{k}"
            call_path, model_path = folder / "pooled_calls.csv.gz", folder / "model.npz"
            columns = ["factor", "start", "end_exclusive", "observed_tokens", "animal_id", "session_id", "start_frame", "end_frame_exclusive"]
            calls = pd.read_csv(call_path, usecols=columns, dtype={"animal_id": str})
            paths = np.asarray([json.loads(s) for s in calls.observed_tokens], int)
            if paths.shape != (len(calls), length):
                raise AssertionError("saved call path length mismatch")
            source_id = f"L{length}_K{k}"
            sources.append(dict(id=source_id, length_bouts=length, K=k,
                                model_path=str(model_path.resolve()), model_sha256=digest(model_path),
                                calls_path=str(call_path.resolve()), calls_sha256=digest(call_path), saved_calls=len(calls),
                                lambda_competition=.3, seed=43, iterations=80))
            for factor, table in calls.assign(path=list(map(tuple, paths))).groupby("factor"):
                for path, subset in table.groupby("path", sort=False):
                    factor_path_rows.append(dict(source=source_id, factor=int(factor), path=json.dumps([int(t) for t in path]),
                                                 named_path=" → ".join(token_map[int(t)]["name"] for t in path),
                                                 calls=len(subset), mice=subset.animal_id.nunique(),
                                                 all_stages_named=all(t < 20 for t in path)))
            for candidate in candidates:
                stages = candidate["stage_sets"]
                n = len(stages)
                matched = np.zeros(len(paths), bool)
                offsets = np.zeros(len(paths), int)
                for offset in range(length - n + 1):
                    hit = np.ones(len(paths), bool)
                    for j, tokens in enumerate(stages):
                        hit &= np.isin(paths[:, offset + j], tokens)
                    matched |= hit
                    offsets += hit
                for factor, table in calls.assign(matches=matched, matched_offsets=offsets).groupby("factor"):
                    good = table[table.matches]
                    support_rows.append(dict(theme=candidate["id"], source=source_id, factor=int(factor),
                                             saved_factor_calls=len(table), supporting_calls=len(good),
                                             call_fraction=float(len(good) / len(table)), supporting_offsets=int(good.matched_offsets.sum()),
                                             supporting_mice=int(good.animal_id.nunique())))
            guard()
    support = pd.DataFrame(support_rows)
    support.to_csv(args.output / "generator_core_support.csv", index=False)
    pd.DataFrame(factor_path_rows).sort_values("calls", ascending=False).to_csv(args.output / "generator_observed_paths.csv.gz", index=False)
    for candidate in candidates:
        own = occurrences[occurrences.theme == candidate["id"]]
        ranked = support[support.theme == candidate["id"]].sort_values("supporting_calls", ascending=False)
        candidate.update(root_stage_sets=candidate["stage_sets"] if candidate["id"] == "rear_backoff" else candidate["stage_sets"][:-1],
                         stage_raw_ids=[[raw for raw, token in enumerate(mapping) if token in stage] for stage in candidate["stage_sets"]],
                         stage_labels=[[token_map[t]["name"] for t in stage] for stage in candidate["stage_sets"]],
                         cohort_occurrence_estimate=len(own), pooled_mice=int(own.animal_id.nunique()), pooled_sessions=int(own.session_id.nunique()),
                         cohort_count_limit="all exact consecutive grouped-core windows within saved valid chunks; overlapping occurrences allowed; no duration/kinematic gates or phenotype selection",
                         source_support=ranked.head(8).to_dict("records"),
                         observed_paths=[dict(tokens=json.loads(path), named_path=" → ".join(token_map[t]["name"] for t in json.loads(path)),
                                              occurrences=len(subset), mice=int(subset.animal_id.nunique()))
                                         for path, subset in sorted(own.groupby("path"), key=lambda item: len(item[1]), reverse=True)[:12]],
                         next_bout_counts=[dict(token=None if pd.isna(token) else int(token), name="block_end" if pd.isna(token) else token_map[int(token)]["name"],
                                                occurrences=int(count)) for token, count in own.groupby("next_token", dropna=False).size().sort_values(ascending=False).items()])
    output = dict(status="FROZEN_EXPLORATORY_PROPOSAL", scientific_acceptance=False,
                  selection="group-blind named-path support and prior developmental feedback; no group, wave, treatment, DA or p-values read or used",
                  source_manifest=dict(path=str(args.manifest.resolve()), sha256=digest(args.manifest)),
                  annotation_source=source_manifest["annotation_source"],
                  mapping=dict(path=str(mapping_path.resolve()), sha256=digest(mapping_path), raw_to_grouped=mapping.tolist(), tokens=token_map),
                  references=dict(path=str(reference_path.resolve()), sha256=digest(reference_path),
                                  l3_path=str(reference_l3.resolve()), l3_sha256=digest(reference_l3),
                                  observations=len(refs), chunks=int(refs.chunk_id.nunique()), recordings=99, mice=21,
                                  valid_seconds=float((refs.end_frame_exclusive - refs.start_frame).sum() / 60.)),
                  policy=dict(clock="native 60Hz; half-open frame bounds", grouping="existing grouped69; raw constituents retained in stage_raw_ids",
                              unknown="all tokens>=20 remain distinct unresolved/bad/dump identities; never wildcard/merge/skip; they break named-core matches but are retained as explicit continuation outcomes",
                              barriers="existing saved chunk/recording boundaries and native discontinuities are absolute; no matching across padding/exclusions/unknown gaps",
                              overlap="candidate cores may overlap within/between themes; these counts are window opportunities, not a mutually exclusive state sequence",
                              generator_support="a complete named core anywhere within an observed saved factor-call path; calls can support multiple cores; no argmax, effect-size selection or all-factor acceptance",
                              root="three-stage root is core without final stage; rear_backoff root is both stages and outcome is next bout; denominator includes every matching root, not just completed cores; block-end opportunities are censored, not exits; parent must recompute raw outcomes and usable continuation support"),
                  resources=dict(threads=args.threads, memory_cap_gb=args.memory_gb, rss_bytes=process.memory_info().rss),
                  sources=sources, candidates=candidates,
                  source_code=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)))
    (args.output / "recommendation.json").write_text(json.dumps(output, indent=2) + "\n")
    summary = ["# Group-blind Karen theme candidates", "", "Frozen exploratory proposal. No phenotype statistics or qualitative acceptance.", "",
               "| Candidate | Core windows | Mice | Priority |", "| --- | ---: | ---: | --- |"]
    summary.extend(f"| {c['name']} | {c['cohort_occurrence_estimate']} | {c['pooled_mice']} | {'Primary' if c['recommended'] else 'Reserve'} |" for c in candidates)
    summary += ["", "See recommendation.json for exact grouped/raw definitions, source identities, generator-call support and limitations.",
                "", "Named-core membership is stricter than partial factor detection. Different sensory cores may be variants of one broad family; no distinct-theme claim is established."]
    (args.output / "README.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({c["id"]: dict(windows=c["cohort_occurrence_estimate"], mice=c["pooled_mice"], source_support=c["source_support"][:2]) for c in candidates}, indent=2))


if __name__ == "__main__":
    main()
