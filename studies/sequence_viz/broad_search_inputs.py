"""Frozen native whole-bout inputs and study policy for the broad episode search.

Import and recipe generation use only the standard library. Explicit preparation
consumes existing verified data-only exports; it never loads pickles, fits KPMS,
processes DA or changes source recordings. All feature-created vocabularies retain
the original KPMS bout intervals, and all fitted encoders are owned downstream.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

MINIMUM_SUPPORT = .8
WINDOW_POLICY = ("Allocate the per-animal budget equally across its recordings; "
                 "choose one whole-bout contiguous window per recording uniformly "
                 "among blocks supplying the allocation, else among longest blocks; "
                 "uniform whole-bout onset. Shortfall is recorded and not filled across gaps.")


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), sha256=_digest(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _load_npz(path, *, max_bytes=256*1024**2):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > max_bytes:
            raise MemoryError("selected input exceeds the uncompressed NPZ cap")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files if name != "__identity_json"}


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _annotation_mapping(path):
    """Keep unresolved/unassigned IDs separate instead of one artificial category."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        token = str(int(row["syllable_id"]))
        description = row["description"].strip()
        if token in result:
            raise ValueError("duplicate model-scoped annotation identity")
        unresolved = description.lower() in {"", "x", "unassigned", "unknown", "unlabeled", "unlabelled"}
        result[token] = "unmapped:"+token if unresolved else "annotation:"+description
    return result


def stratified_animal_folds(subject_groups, *, seed=101, folds=3):
    """Freeze animal-level strata; labels are used here and never as model inputs."""
    _integer(seed, "seed")
    _integer(folds, "folds", minimum=2)
    if len(subject_groups) < folds or any(not str(subject) for subject in subject_groups):
        raise ValueError("require enough explicitly identified animals for all folds")
    groups = defaultdict(list)
    for subject, group in subject_groups.items():
        groups[str(group)].append(str(subject))
    rng, result = random.Random(seed), [[] for _ in range(folds)]
    for group in sorted(groups, key=lambda value: (-len(groups[value]), value)):
        subjects = sorted(groups[group])
        rng.shuffle(subjects)
        counts = [0]*folds
        rotation = rng.randrange(folds)
        for subject in subjects:
            fold = min(range(folds), key=lambda i: (counts[i], len(result[i]), (i-rotation) % folds))
            result[fold].append(subject)
            counts[fold] += 1
    return [sorted(subjects) for subjects in result]


def join_verified_native_features(native, sparse, dense):
    """Attach sparse latents and exact dense heading only inside verified support.

    A sparse row can supply its native frame and one interior partner only when
    the next sparse VFI equals f+2. Final rows and larger gaps supply no partner.
    Labels must match both exports at every supported native frame. This join
    never infers an acquisition conversion from frame counts or video headers.
    """
    import numpy as np

    frames, labels = native["native_frames"], native["labels"]
    source = sparse["table_index"]
    if (frames.ndim != 1 or labels.shape != frames.shape or source.ndim != 1
            or len(source) == 0 or frames.dtype.kind not in "iu" or source.dtype.kind not in "iu"
            or np.any(np.diff(frames) <= 0) or np.any(np.diff(source) <= 0)):
        raise ValueError("native and sparse frames must be explicit increasing integer coordinates")
    positions = np.searchsorted(source, frames, side="right")-1
    if np.any(positions < 0):
        raise ValueError("native feature join precedes sparse support")
    exact = source[positions] == frames
    next_positions = np.minimum(positions+1, len(source)-1)
    partner = ((frames == source[positions]+1) & (positions+1 < len(source))
               & (source[next_positions] == source[positions]+2))
    if not np.all(exact | partner):
        raise ValueError("native feature join attempts to fill an unsupported sparse gap")
    if not np.array_equal(sparse["labels"][positions], labels):
        raise ValueError("native labels differ from the sparse model feature export")
    dense_frames = dense["table_index"]
    if np.any(np.diff(dense_frames) <= 0):
        raise ValueError("dense feature coordinates must be increasing")
    if "frame_index" in dense and not np.array_equal(dense_frames, dense["frame_index"]):
        raise ValueError("dense frame_index and table_index disagree")
    dense_positions = np.searchsorted(dense_frames, frames)
    if (np.any(dense_positions >= len(dense_frames))
            or not np.array_equal(dense_frames[dense_positions], frames)):
        raise ValueError("supported native frames are missing from the dense feature export")
    if not np.array_equal(dense["labels"][dense_positions], labels):
        raise ValueError("native labels differ from the dense feature export")
    for key in ("centroid_x", "centroid_y"):
        if not np.array_equal(native[key], dense[key][dense_positions], equal_nan=True):
            raise ValueError("verified centroid sidecars differ from the new feature export")
    latent_keys = sorted((key for key in sparse if key.startswith("latent_state_")),
                         key=lambda key: int(key.rsplit("_", 1)[1]))
    if latent_keys != [f"latent_state_{i}" for i in range(10)] or "heading" not in dense:
        raise ValueError("require the verified ten-dimensional latent export and explicit heading")
    return dict(frames=frames, labels=labels,
                centroid=np.column_stack([native["centroid_x"], native["centroid_y"]]),
                heading=dense["heading"][dense_positions],
                latent=np.column_stack([sparse[key][positions] for key in latent_keys]),
                join_counts=dict(exact_native_rows=int(exact.sum()), supported_partner_rows=int(partner.sum())))


def _prepare_record(cache, identity, builder, *, recording, subject, metadata, excluded):
    import numpy as np

    from fpcode.common.cache_manifest import cached_npz_component

    arrays, hit, key = cached_npz_component(cache, identity, builder, max_uncompressed_bytes=64*1024**2)
    starts, stops, parents = arrays["start_frames"], arrays["stop_frames"], arrays["parent_ids"]
    if (len(starts) != len(stops) or len(parents) != len(starts) or len(starts) == 0
            or np.any(stops <= starts) or np.any(starts[1:] < stops[:-1])):
        raise ValueError("prepared source bouts must be nonempty, nonoverlapping native intervals")
    names = arrays["feature_names"].tolist()
    return dict(recording=recording, subject=subject, npz_path=str((cache/f"{key}.npz").resolve()),
                **{k: v for k, v in _fingerprint(cache/f"{key}.npz").items() if k != "path"},
                bout_count=len(starts), feature_names=names, feature_units=arrays["feature_units"].tolist(),
                finite_feature_bouts={name: int(np.isfinite(arrays["values"][:, i]).sum()) for i, name in enumerate(names)},
                native_mapping_verified=True, source_provenance=identity,
                metadata=metadata, exclude_from_search=excluded, preparation_cache_hit=hit)


def _payload(summary, provenance):
    import numpy as np

    return dict(feature_names=np.asarray(summary["feature_names"], dtype=str),
                feature_units=np.asarray(summary["feature_units"], dtype=str), values=summary["values"],
                parent_ids=summary["parent_labels"].astype(str), start_frames=summary["start_frames"],
                stop_frames=summary["stop_frames"], source_provenance=np.array(json.dumps(provenance, sort_keys=True)))


def _validate_reviewed_mask(data, intervals):
    """Require exact reviewed exclusions; the documented unmodified view has none."""
    import numpy as np

    expected = np.zeros(len(data["frames"]), dtype=bool)
    for start, stop in intervals:
        expected |= (data["frames"] >= start) & (data["frames"] < stop)
    if "analysis_excluded" not in data:
        if intervals:
            raise ValueError("Karen recording with reviewed exclusions lacks its stored review mask")
    elif not np.array_equal(data["analysis_excluded"], expected):
        raise ValueError("Karen stored review mask differs from the frozen interval policy")
    if np.any(data["labels"][expected] >= 0):
        raise ValueError("Karen reviewed exclusions retain supported model labels")


def prepare_inputs(*, code_root, output):
    """Prepare all 129 selected recordings once and atomically freeze study policy."""
    import resource

    import numpy as np
    from threadpoolctl import threadpool_info

    from fpcode.behavior.analysis.bout_features import summarize_source_bouts
    from fpcode.common.cache_manifest import atomic_write_json

    started = time.monotonic()
    code_root, output = Path(code_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"bout_features"
    feature_module = code_root/"src/fpcode/behavior/analysis/bout_features.py"
    code_identity = {str(path.relative_to(code_root)): _digest(path) for path in
                     (Path(__file__).resolve(), feature_module,
                      code_root/"src/fpcode/behavior/primitives/bouts.py")}
    root = code_root/"outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries"
    split_path, sessions_path, review_path = root/"chains/animal_splits.json", root/"sessions.csv", root/"manifest.json"
    review_policy_path = root/"reviewed_exclusions.csv"
    review_metadata = json.loads(review_path.read_text())
    reviewed_intervals = defaultdict(list)
    with review_policy_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            reviewed_intervals[row["session_id"]].append((int(row["start_frame"]), int(row["end_frame_exclusive"])))
    split = json.loads(split_path.read_text())
    search_subjects = sorted(set(map(str, split["train"])) | set(map(str, split["validation"])))
    reserved = sorted(map(str, split["test"]))
    if set(search_subjects) & set(reserved):
        raise ValueError("Karen reserved animals overlap the exploration pool")
    with sessions_path.open(newline="") as handle:
        sessions = list(csv.DictReader(handle))
    if len(sessions) != 99 or {row["animal_id"] for row in sessions} != set(search_subjects+reserved):
        raise ValueError("Karen requires the verified 99 recordings and declared 21 animals")
    if (len(reviewed_intervals) != review_metadata["masked_recordings"]
            or len(sessions)-len(reviewed_intervals) != review_metadata["unmodified_linked_recordings"]
            or not set(reviewed_intervals) <= {row["session_id"] for row in sessions}):
        raise ValueError("Karen reviewed mask coverage differs from its saved verification manifest")
    annotation = code_root/"outputs/karen_behavior_atlas_20261011/syllable_annotations.csv"
    annotation_map = _annotation_mapping(annotation)
    groups, rows, vocabulary = {}, [], set(annotation_map)
    common = dict(source_model_id="model_21kp_static_50ar_300slds_seed101", clock="karen_native_video_frame", fps=60.,
                  minimum_support=MINIMUM_SUPPORT, code_identity=code_identity,
                  review_manifest=_fingerprint(review_path), review_policy=_fingerprint(review_policy_path),
                  source_sessions=_fingerprint(sessions_path))
    for session in sessions:
        recording, subject = session["session_id"], session["animal_id"]
        group = session["group"]
        if subject in groups and groups[subject] != group:
            raise ValueError("Karen animal stratum changes across recordings")
        groups[subject] = group
        if session["model_name"] != common["source_model_id"] or float(session["fps"]) != 60.:
            raise ValueError("Karen model identity or native clock differs from the verified view")
        source = root/"frame_cache"/f"{recording}.npz"
        identity = {**common, "recording": recording, "subject": subject, "source": _fingerprint(source),
                    "projection": "reviewed labels; excluded frames remain barriers; unchanged gap-aware KPMS RLE"}

        def builder(source=source, identity=identity, intervals=reviewed_intervals[recording]):
            data = _load_npz(source)
            labels = data["labels"]
            _validate_reviewed_mask(data, intervals)
            if float(data["fps"]) != 60.:
                raise ValueError("Karen rate no longer agrees with the verified native clock")
            result = summarize_source_bouts(data["frames"], labels, data["centroid"], fps=60.,
                heading=data["heading"], latent=data["latent"], minimum_support=MINIMUM_SUPPORT)
            return _payload(result, identity)

        row = _prepare_record(cache, identity, builder, recording=recording, subject=subject,
            metadata={k: session[k] for k in ("group", "condition", "week", "drug", "date", "chamber")},
            excluded=subject in reserved)
        vocabulary.update(_load_npz(row["npz_path"])["parent_ids"].tolist())
        rows.append(row)
    karen = dict(cohort_id="karen", model_id=common["source_model_id"], clock=common["clock"], fps=60.,
                 minimum_support=MINIMUM_SUPPORT, vocabulary=sorted(vocabulary, key=int), annotation_mapping=annotation_map,
                 annotation_source=_fingerprint(annotation), exploration_subjects=search_subjects, reserved_subjects=reserved,
                 folds=stratified_animal_folds({s: groups[s] for s in search_subjects}), records=rows,
                 clock_evidence="Verified existing reviewed cache native 60 Hz frames; no conversion inferred.",
                 split_evidence=_fingerprint(split_path))

    native_manifest_path = code_root/"outputs/broad_search_preflight_20261013/native_input_manifest.json"
    native_manifest = json.loads(native_manifest_path.read_text())
    export_manifest_path = code_root/"outputs/broad_search_preparation_20261013/jonathan_feature_exports.json"
    exports = {row["recording"]: row for row in json.loads(export_manifest_path.read_text())["rows"]}
    if len(native_manifest["rows"]) != 30 or set(exports) != {row["session_id"] for row in native_manifest["rows"]}:
        raise ValueError("Jonathan preparation requires the verified thirty matching Run12 exports")
    groups, rows, vocabulary = {}, [], set()
    annotation = Path(native_manifest["annotation_csv"])
    if _digest(annotation) != native_manifest["annotation_sha256"]:
        raise ValueError("Jonathan model-scoped annotation identity changed")
    common = dict(source_model_id="amph_prod_k150k_run_12", clock="jonathan_native_acquisition_frame",
                  fps=float(native_manifest["clock_evidence"]["native_video_fps"]), minimum_support=MINIMUM_SUPPORT,
                  code_identity=code_identity, native_manifest=_fingerprint(native_manifest_path),
                  feature_export_manifest=_fingerprint(export_manifest_path),
                  source_pickle_sha256=native_manifest["input_pickle_sha256"])
    for session in native_manifest["rows"]:
        recording, subject = session["session_id"], session["animal_id"]
        export = exports[recording]
        if str(export["metadata"]["animal_id"]) != subject or not session["original_bout_boundaries_verified"]:
            raise ValueError("Jonathan verified recording/animal/bout identity is missing")
        if subject in groups and groups[subject] != session["genotype"]:
            raise ValueError("Jonathan animal stratum changes across recordings")
        groups[subject] = session["genotype"]
        source = Path(session["input_npz"])
        fingerprint = _fingerprint(source)
        if fingerprint["sha256"] != session["sha256"]:
            raise ValueError("verified Jonathan native labels changed")
        sparse_path, dense_path = Path(export["exports"]["syllables"]), Path(export["exports"]["all_data_df"])
        identity = {**common, "recording": recording, "subject": subject, "source": fingerprint,
                    "sparse_features": _fingerprint(sparse_path), "dense_features": _fingerprint(dense_path),
                    "projection": native_manifest["projection_policy"],
                    "latent_join": "exact sparse VFI or its supported interior partner only; no gap fill",
                    "heading_join": "exact dense native table_index, asserted equal to frame_index and verified labels"}

        def builder(source=source, sparse_path=sparse_path, dense_path=dense_path, identity=identity):
            native = _load_npz(source)
            joined = join_verified_native_features(native, _load_npz(sparse_path), _load_npz(dense_path))
            result = summarize_source_bouts(joined["frames"], joined["labels"], joined["centroid"], fps=common["fps"],
                heading=joined["heading"], latent=joined["latent"], minimum_support=MINIMUM_SUPPORT)
            for target, key in (("start_frames", "bout_start_frames"), ("stop_frames", "bout_end_frames_exclusive"),
                                ("parent_labels", "bout_labels")):
                if not np.array_equal(result[target], native[key]):
                    raise ValueError("Jonathan feature preparation altered the verified source bout boundaries")
            return _payload(result, {**identity, "join_counts": joined["join_counts"]})

        row = _prepare_record(cache, identity, builder, recording=recording, subject=subject,
            metadata={key: session.get(key) for key in ("genotype", "drug", "chamber", "entry_frame", "injection_frame")},
            excluded=False)
        vocabulary.update(_load_npz(row["npz_path"])["parent_ids"].tolist())
        rows.append(row)
    if len(groups) != 15 or Counter(groups.values()) != {"WT": 8, "KI": 7}:
        raise ValueError("Jonathan selected cohort no longer matches the verified eight WT/seven KI animals")
    jonathan = dict(cohort_id="jonathan", model_id=common["source_model_id"], clock=common["clock"], fps=common["fps"],
                    minimum_support=MINIMUM_SUPPORT, vocabulary=sorted(vocabulary, key=int),
                    annotation_mapping=_annotation_mapping(annotation), annotation_source=_fingerprint(annotation),
                    exploration_subjects=sorted(groups), reserved_subjects=[], folds=stratified_animal_folds(groups), records=rows,
                    clock_evidence=native_manifest["clock_evidence"], split_evidence="Three frozen animal folds, stratified WT/KI as feasible.")
    manifest = dict(schema_version=1, cohorts=[karen, jonathan], code_identity=code_identity,
        window_policy=WINDOW_POLICY, native_interval_policy="Unchanged supported KPMS source bouts; exclusive stops; masks and native gaps are barriers.",
        frozen_model_exposure="Existing KPMS was fitted before these animal folds; this is downstream detector/encoder holdout, not independent KPMS holdout.",
        downstream_only="Cohort labels/drug are audit metadata and stratification only; no phenotype or DA enters model selection.")
    # Cache hit and timings describe the operation, not the scientific input identity.
    manifest["manifest_hash"] = _json_hash({**manifest, "cohorts": [dict(c, records=[{k: v for k, v in r.items()
        if k != "preparation_cache_hit"} for r in c["records"]]) for c in manifest["cohorts"]]})
    manifest["preparation_resources"] = dict(wall_seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, numerical_pools=threadpool_info(),
        prepared_recordings=sum(len(c["records"]) for c in manifest["cohorts"]), worker_processes=1,
        per_input_uncompressed_cap_bytes=256*1024**2, per_output_uncompressed_cap_bytes=64*1024**2,
        modelling_imports=[name for name in __import__("sys").modules if name.startswith(("jax", "torch", "keypoint_moseq"))])
    atomic_write_json(output/"input_manifest.json", manifest)
    return manifest


def _manifest(value):
    manifest = value if isinstance(value, dict) else json.loads(Path(value).read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("cohorts"):
        raise ValueError("require a frozen broad input manifest schema_version1")
    return manifest


def _cohort(manifest, name):
    matches = [cohort for cohort in manifest["cohorts"] if cohort["cohort_id"] == name]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate cohort identity")
    return matches[0]


def load_job_records(manifest, cohort, fold, *, per_animal, held_per_animal, window_seed):
    """Load matched train/held whole-bout windows, excluding reserved animals.

    Returned records intentionally have no genotype, group, drug or DA fields.
    A record-local hash seed makes sampling invariant to method, representation,
    numerical seed, worker ordering and record order. Different folds can request
    different train/held budgets; repeated recordings stay animal-disjoint.
    """
    import numpy as np

    manifest = _manifest(manifest)
    selected = _cohort(manifest, cohort)
    _integer(fold, "fold")
    _integer(per_animal, "per_animal", minimum=6)
    _integer(held_per_animal, "held_per_animal", minimum=6)
    _integer(window_seed, "window_seed")
    if fold >= len(selected["folds"]):
        raise ValueError("fold is outside the frozen animal assignment")
    held = set(selected["folds"][fold])
    search, reserved = set(selected["exploration_subjects"]), set(selected["reserved_subjects"])
    flattened = [s for animals in selected["folds"] for s in animals]
    if set(flattened) != search or len(flattened) != len(search) or not held or search & reserved:
        raise ValueError("frozen animal folds are incomplete, overlapping or leak reserved subjects")
    records = sorted((row for row in selected["records"] if row["subject"] in search
                      and not row.get("exclude_from_search", False)), key=lambda row: (row["subject"], row["recording"]))
    counts = Counter(row["subject"] for row in records)
    if set(counts) != search:
        raise ValueError("every exploration animal must have prepared supported input")
    animal_rank, result = Counter(), []
    for row in records:
        subject, recording = row["subject"], row["recording"]
        split = "held" if subject in held else "train"
        budget = held_per_animal if split == "held" else per_animal
        requested = budget//counts[subject]+int(animal_rank[subject] < budget % counts[subject])
        animal_rank[subject] += 1
        if requested < 6:
            raise ValueError("per-animal budget must supply at least six events per selected recording")
        path = Path(row["npz_path"])
        stat = path.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("prepared input changed since its frozen fingerprint")
        data = _load_npz(path, max_bytes=64*1024**2)
        starts, stops = data["start_frames"], data["stop_frames"]
        if len(starts) != row["bout_count"] or np.any(starts[1:] < stops[:-1]) or np.any(stops <= starts):
            raise ValueError("prepared whole-bout identity or native ordering changed")
        boundaries = np.flatnonzero(np.r_[True, starts[1:] != stops[:-1]])
        blocks = [(int(a), int(b)) for a, b in zip(boundaries, np.r_[boundaries[1:], len(starts)]) if b-a >= 6]
        if not blocks:
            raise ValueError("prepared recording has no six-event contiguous source window")
        longest = max(b-a for a, b in blocks)
        eligible = [block for block in blocks if block[1]-block[0] >= requested]
        if not eligible:
            eligible = [block for block in blocks if block[1]-block[0] == longest]
        local_seed = int(_json_hash([window_seed, cohort, subject, recording])[:16], 16)
        rng = random.Random(local_seed)
        first, stop = eligible[rng.randrange(len(eligible))]
        achieved = min(requested, stop-first)
        first += rng.randrange(stop-first-achieved+1)
        positions = np.arange(first, first+achieved)
        result.append(dict(feature_names=tuple(data["feature_names"].tolist()), values=data["values"][positions],
            parent_ids=data["parent_ids"][positions].astype(str), start_frames=starts[positions], stop_frames=stops[positions],
            source_vocabulary=tuple(selected["vocabulary"]), split=split, subject=subject, recording=recording,
            cohort=cohort, clock=selected["clock"], fps=selected["fps"], source_model_id=selected["model_id"],
            input_sha256=row["sha256"], source_indices=positions, requested_events=requested, achieved_events=achieved,
            source_events_total=row["bout_count"], source_contiguous_blocks=len(boundaries),
            window_seed=window_seed, window_policy=WINDOW_POLICY))
    return result


def job_identity(manifest, cohort, fold, records, *, window_seed, per_animal, held_per_animal):
    """Actual sampled native source/clock/split dependencies for shared fit caches."""
    manifest = _manifest(manifest)
    selected = _cohort(manifest, cohort)
    windows = [dict(subject=r["subject"], recording=r["recording"], split=r["split"], input_sha256=r["input_sha256"],
                    source_indices=r["source_indices"].tolist(), start_frames=r["start_frames"].tolist(),
                    stop_frames=r["stop_frames"].tolist(), requested_events=r["requested_events"],
                    achieved_events=r["achieved_events"]) for r in records]
    return dict(manifest_hash=manifest["manifest_hash"], cohort=cohort, source_model_id=selected["model_id"],
                clock=selected["clock"], fps=selected["fps"], minimum_support=selected["minimum_support"],
                vocabulary=selected["vocabulary"], annotation_mapping=selected["annotation_mapping"],
                fold=fold, held_subjects=selected["folds"][fold], window_seed=window_seed,
                per_animal=per_animal, held_per_animal=held_per_animal, window_policy=WINDOW_POLICY, windows=windows)


def load_full_held_records(manifest, cohort, fold):
    """Full supported recordings of the original held animals; no resampling.

    Study metadata remains outside model records. Reserved animals remain excluded.
    This loader only reads prepared data-only components, never session pickles.
    """
    import numpy as np

    manifest = _manifest(manifest)
    selected = _cohort(manifest, cohort)
    _integer(fold, "fold")
    if fold >= len(selected["folds"]):
        raise ValueError("fold outside frozen assignment")
    held, reserved = set(selected["folds"][fold]), set(selected["reserved_subjects"])
    if held & reserved or not held <= set(selected["exploration_subjects"]):
        raise ValueError("held assignment leaks reserved or outside animals")
    out = []
    for row in sorted(selected["records"], key=lambda r: (r["subject"], r["recording"])):
        if row["subject"] not in held or row.get("exclude_from_search", False):
            continue
        path = Path(row["npz_path"])
        stat = path.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("prepared input changed since frozen fingerprint")
        arrays = _load_npz(path, max_bytes=64*1024**2)
        if len(arrays["start_frames"]) != row["bout_count"]:
            raise ValueError("prepared bout count differs from frozen manifest")
        out.append(dict(feature_names=tuple(arrays["feature_names"].tolist()), values=arrays["values"],
            parent_ids=arrays["parent_ids"].astype(str), start_frames=arrays["start_frames"], stop_frames=arrays["stop_frames"],
            source_vocabulary=tuple(selected["vocabulary"]), split="held", subject=row["subject"], recording=row["recording"],
            cohort=cohort, clock=selected["clock"], fps=selected["fps"], source_model_id=selected["model_id"],
            input_sha256=row["sha256"], source_indices=np.arange(row["bout_count"]),
            source_events_total=row["bout_count"], window_policy="all original supported bouts; no gap fill or sampling"))
    if {r["subject"] for r in out} != held:
        raise ValueError("full held input does not cover every declared animal")
    return out


def generate_recipes(seed=101, recipes_per_cohort=120):
    """Balanced six representation cells × five detector families, conditional knobs.

    Inactive parameters remain fixed defaults for the shared API and are disclosed
    through active_parameters. Each repeat samples one representation per cell
    and matched categorical window lengths/counts across all five detector methods.
    The numerical detector seed can vary later without changing creator_seed.
    """
    _integer(seed, "seed")
    _integer(recipes_per_cohort, "recipes_per_cohort", minimum=30)
    if recipes_per_cohort % 30:
        raise ValueError("recipes_per_cohort must be a multiple of thirty for balanced route/method cells")
    cells = [("kpms", "describe"), ("kpms", "episode"), ("feature_only", "describe"),
             ("within_kpms", "describe"), ("joint", "describe"), ("parallel", "describe")]
    methods = ("exact", "tolerant", "seqnmf", "hsmm", "grammar")
    rng, recipes = random.Random(seed), []
    for repetition in range((recipes_per_cohort+29)//30):
        for cell, (route, role) in enumerate(cells):
            creator_seed = seed+repetition*1009+cell*37
            bundle = rng.choice(("duration", "movement", "posture", "spatial", "movement_posture", "all"))
            if route == "parallel":
                bundle = rng.choice(("movement", "posture", "movement_posture"))
            resolution = rng.choice(("identity", "annotation", "quantitative")) if route in {"kpms", "parallel"} else rng.choice(("identity", "quantitative"))
            clusters = rng.choice((4, 8, 12)) if route != "within_kpms" else rng.choice((2, 3, 4))
            if route == "kpms":
                clusters = 8  # No creator is fitted on this route.
            merge_clusters = 1
            if resolution == "quantitative":
                merge_clusters = rng.choice((4, 8, 16))
                if route in {"feature_only", "joint"}:
                    merge_clusters = min(merge_clusters, max(1, clusters//2))
            base = dict(route=route, feature_role=role, resolution=resolution, bundle=bundle, clusters=clusters,
                        merge_clusters=merge_clusters,
                        creator_seed=creator_seed, parent_weight=rng.choice((.5, 1., 2.)) if route == "joint" else 1.,
                        length=rng.choice((3, 5, 8)), factors=rng.choice((4, 8, 12)), min_prevalence=rng.choice((1, 2, 3)),
                        seed=seed, feature_weight=rng.choice((.25, 1., 2.)) if role == "episode" or route == "parallel" else 1.,
                        penalty=.01, iterations=40, activation_quantile=.95, max_error=1.,
                        phase_count=3, phase_max=2, background_max=16, entry_probability=.1,
                        early_exit_probability=0., skip_probability=0., reentry=False, extra_background=False,
                        learn_durations=False, initialization="configured", pseudocount=.1, variance_floor=.05,
                        hsmm_targets=1, edit_budget=1, gate_multiplier=1., grammar_optional=False, grammar_repeat=1)
            for method in methods:
                recipe = {**base, "method": method, "representation_cell": cell,
                          "base_id": f"r{repetition:02d}_c{cell}_{method}"}
                if method in {"exact", "tolerant", "grammar"}:
                    recipe["feature_weight"] = 1.
                active = ["route", "feature_role", "resolution", "length", "factors", "min_prevalence"]
                consumes = route != "kpms" or role == "episode" or resolution == "quantitative"
                if consumes:
                    active.append("bundle")
                if route != "kpms":
                    active.append("clusters")
                if route != "kpms" or resolution == "quantitative":
                    active.append("creator_seed")
                if resolution == "quantitative":
                    active.append("merge_clusters")
                if route == "joint":
                    active.append("parent_weight")
                if role == "episode" or route == "parallel":
                    if method in {"seqnmf", "hsmm"}:
                        active.append("feature_weight")
                    if method in {"exact", "tolerant", "grammar"}:
                        recipe["gate_multiplier"] = rng.choice((.75, 1., 2.))
                        active.append("gate_multiplier")
                if method == "seqnmf":
                    recipe.update(penalty=rng.choice((0., .005, .02, .08)), iterations=rng.choice((30, 40)),
                                  activation_quantile=rng.choice((.90, .95, .98)), max_error=rng.choice((.7, 1., 1.3)))
                    active.extend(("seed", "penalty", "iterations", "activation_quantile", "max_error"))
                    active.remove("min_prevalence")
                elif method == "hsmm":
                    recipe.update(iterations=rng.choice((4, 8)), phase_count=rng.choice((2, 3, 5)),
                        phase_max=rng.choice((1, 2, 4)), background_max=rng.choice((8, 16, 32)),
                        entry_probability=rng.choice((.03, .1, .25)), early_exit_probability=rng.choice((0., .1)),
                        skip_probability=rng.choice((0., .1)), reentry=rng.choice((False, True)),
                        extra_background=rng.choice((False, True)), learn_durations=rng.choice((False, True)),
                        initialization=rng.choice(("configured", "dirichlet")), pseudocount=rng.choice((.05, .2, 1.)),
                        hsmm_targets=rng.choice((1, 2, 3)))
                    if not recipe["extra_background"]:
                        recipe["entry_probability"] = .1
                    if recipe["phase_count"] == 2:
                        recipe["skip_probability"] = 0.
                    active.extend(("seed", "iterations", "phase_count", "phase_max", "background_max", "entry_probability",
                        "early_exit_probability", "skip_probability", "reentry", "extra_background", "learn_durations",
                        "initialization", "pseudocount", "hsmm_targets"))
                    if not recipe["extra_background"]:
                        active.remove("entry_probability")
                    if recipe["phase_count"] == 2:
                        active.remove("skip_probability")
                    if role == "episode" or route == "parallel":
                        recipe["variance_floor"] = rng.choice((.02, .05, .2))
                        active.append("variance_floor")
                elif method == "tolerant":
                    recipe["edit_budget"] = rng.choice((1, 2))
                    active.append("edit_budget")
                elif method == "grammar":
                    recipe.update(grammar_optional=recipe["length"] > 3 and rng.choice((False, True)),
                                  grammar_repeat=rng.choice((1, 2, 3)))
                    active.extend(("grammar_optional", "grammar_repeat"))
                recipe["active_parameters"] = active
                recipes.append(recipe)
    # Random ordering affects dispatch only, while every complete 30-cell block is balanced.
    rng.shuffle(recipes)
    return recipes[:recipes_per_cohort]
