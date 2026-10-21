"""Label-independent, animal-held-out screening of frozen-vocabulary routines.

Scores are exploratory proxies and are comparable only within a method and
vocabulary resolution. Shuffled-bout nulls preserve each sampled segment's counts,
but do not preserve first-order transitions or the no-self-transition constraint
of RLE. Finalists therefore need a first-order-null audit and qualitative review.
No phenotype, condition, accelerator, file loader or output writer is used here.
"""

import base64
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache

import numpy as np

from fpcode.behavior.episodes.seqnmf import SeqNMF, activation_events, reconstruct, template_similarity


def feature_bundle_columns(names, bundle):
    """Select named physical channels; no inferred units or categorical ID arithmetic."""
    groups = {
        "duration": lambda n: n == "duration_s",
        "movement": lambda n: n in {"duration_s", "mean_speed_px_s", "net_speed_px_s", "path_length_px"},
        "posture": lambda n: n == "duration_s" or "heading" in n or n.startswith("latent_"),
        "spatial": lambda n: n in {"centroid_x_px", "centroid_y_px"},
        "movement_posture": lambda n: n not in {"centroid_x_px", "centroid_y_px"},
        "all": lambda n: True,
    }
    if bundle not in groups:
        raise ValueError("unknown named feature bundle")
    columns = [i for i, name in enumerate(names) if groups[bundle](name)]
    if not columns:
        raise ValueError("requested feature bundle unavailable")
    return columns


def _json_arrays(value):
    if isinstance(value, Mapping):
        return {str(k): _json_arrays(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_arrays(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _codebook_payload(model):
    return {k: _json_arrays(getattr(model, k)) for k in
            ("route", "feature_names", "location", "scale", "parents", "centers", "parent_weight")}


def _codebook_from_payload(payload):
    from fpcode.behavior.primitives.segmentation import WholeBoutCodebook
    payload = dict(payload)
    payload["location"], payload["scale"] = np.array(payload["location"]), np.array(payload["scale"])
    payload["centers"] = {k: np.asarray(v) for k, v in payload["centers"].items()}
    payload["parents"], payload["feature_names"] = tuple(payload["parents"]), tuple(payload["feature_names"])
    return WholeBoutCodebook(**payload)


def _representation_inputs(records, recipe, identity, cache):
    """Fit only on training rows; return unchanged intervals with declared new IDs."""
    from fpcode.behavior.primitives.segmentation import WholeBoutCodebook
    from fpcode.common.cache_manifest import cached_npz_component

    names = tuple(records[0]["feature_names"])
    if any(tuple(r["feature_names"]) != names for r in records):
        raise ValueError("feature schemas differ inside a cohort")
    route, resolution = recipe["route"], recipe["resolution"]
    columns = feature_bundle_columns(names, recipe["bundle"])
    if route == "parallel":
        columns = [i for i in columns if names[i] not in {"centroid_x_px", "centroid_y_px"}]
        if not columns:
            raise ValueError("parallel execution view requires nonspatial features")
    training = [r for r in records if r["split"] == "train"]
    if not training or not any(r["split"] == "held" for r in records):
        raise ValueError("explicit animal-disjoint training and held records required")
    if set(r["subject"] for r in training) & set(r["subject"] for r in records if r["split"] == "held"):
        raise ValueError("animal leakage across training and held streams")
    physical_creator = route in {"feature_only", "within_kpms", "joint", "parallel"}
    direct = recipe["feature_role"] == "episode"
    consumes = physical_creator or direct or resolution == "quantitative"
    # Description-only bundle changes do not invalidate unchanged models.
    configuration = {"route": route, "resolution": resolution,
        "bundle": recipe["bundle"] if consumes else None,
        "clusters": recipe["clusters"] if physical_creator or resolution == "quantitative" else None,
        "merge_clusters": recipe.get("merge_clusters", recipe["clusters"]) if resolution == "quantitative" else None,
        "creator_seed": recipe["creator_seed"] if physical_creator or resolution == "quantitative" else None,
        "parent_weight": recipe["parent_weight"] if route == "joint" else None,
        "feature_role": recipe["feature_role"], "minimum_support": identity["minimum_support"]}
    key_identity = {**identity, "component": "representation", "configuration": configuration}

    def fit():
        values = np.concatenate([r["values"] for r in training])
        parents = np.concatenate([r["parent_ids"] for r in training]).astype(str)
        selected = values[:, columns]
        if consumes and not np.isfinite(selected).any(axis=0).all():
            raise ValueError("selected channels lack finite training support")
        location = np.array([np.nanmedian(selected[:, i]) for i in range(len(columns))]) if consumes else np.zeros(len(columns))
        scale = (np.nanquantile(selected, .75, axis=0)-np.nanquantile(selected, .25, axis=0)) if consumes else np.ones(len(columns))
        scale = np.where(scale > 1e-9, scale, 1.)
        codebooks, spatial_columns = [], []
        kwargs = dict(feature_names=tuple(names[i] for i in columns), clusters=recipe["clusters"],
                      seed=recipe["creator_seed"], iterations=40)
        if route in {"feature_only", "within_kpms", "joint"}:
            creator = {"feature_only": "feature", "within_kpms": "within_parent", "joint": "joint"}[route]
            codebooks.append(_codebook_payload(WholeBoutCodebook.fit(selected, parents,
                route=creator, parent_weight=recipe["parent_weight"], **kwargs)))
            catalog = list(_codebook_from_payload(codebooks[0]).catalog_ids)
            assigned = _codebook_from_payload(codebooks[0]).assign(selected, parents)
        else:
            catalog = list(identity["vocabulary"])
            assigned = parents.copy()
        if route == "parallel":
            spatial_columns = feature_bundle_columns(names, "spatial")
            codebooks.append(_codebook_payload(WholeBoutCodebook.fit(selected, parents, **kwargs)))
            codebooks.append(_codebook_payload(WholeBoutCodebook.fit(values[:, spatial_columns], parents,
                feature_names=tuple(names[i] for i in spatial_columns), clusters=recipe["clusters"],
                seed=recipe["creator_seed"]+1, iterations=40)))
        mapping = {t: t for t in catalog}
        if resolution == "annotation":
            if route not in {"kpms", "parallel"}:
                raise ValueError("KPMS annotation map cannot label a new codebook")
            mapping = {t: identity["annotation_mapping"].get(t, "unmapped:"+t) for t in catalog}
        elif resolution == "quantitative":
            present, profiles = [], []
            for token in catalog:
                part = selected[(assigned == token) & np.isfinite(selected).all(axis=1)]
                if len(part):
                    present.append(token)
                    profiles.append(np.mean(part, axis=0))
            if not profiles:
                raise ValueError("no finite training type profiles for quantitative merging")
            merge = WholeBoutCodebook.fit(np.asarray(profiles), present,
                feature_names=kwargs["feature_names"], clusters=min(recipe.get("merge_clusters", recipe["clusters"]), len(profiles)),
                seed=recipe["creator_seed"], iterations=40)
            labels = merge.assign(profiles, present)
            mapping.update(dict(zip(present, ["merge:"+str(t) for t in labels])))
        elif resolution != "identity":
            raise ValueError("unknown primitive resolution")
        catalog = sorted(set(mapping.values()))
        return {"metadata": np.array(json.dumps(_json_arrays(dict(columns=columns, spatial_columns=spatial_columns,
            location=location, scale=scale, codebooks=codebooks, mapping=mapping, catalog=catalog,
            configuration=configuration, feature_names=[names[i] for i in columns])), sort_keys=True))}

    arrays, hit, key = cached_npz_component(cache, key_identity, fit)
    fitted = json.loads(str(arrays["metadata"]))
    outputs = transform_frozen_recipe_records(records, recipe, fitted, representation_id=key)
    return outputs, fitted, hit, key


def transform_frozen_recipe_records(records, recipe, fitted, *, representation_id):
    """Apply stored scales/codebooks/mapping without fitting or writing caches.

    Original native bouts survive unchanged or are removed as unsupported; no
    interpolation, new segmentation, phenotype policy or training operation.
    """
    route = recipe["route"]
    direct = recipe["feature_role"] == "episode"
    for name in ("route", "resolution", "feature_role", "bundle", "clusters", "merge_clusters", "creator_seed", "parent_weight"):
        value = fitted["configuration"].get(name)
        if value is not None and value != recipe.get(name):
            raise ValueError("frozen representation does not match recipe: " + name)
    for r in records:
        names = tuple(r["feature_names"])
        if [names[i] for i in fitted["columns"]] != fitted["feature_names"]:
            raise ValueError("frozen feature schema differs from inference records")
    codebooks = [_codebook_from_payload(p) for p in fitted["codebooks"]]
    outputs = []
    aux_names = []
    if route == "parallel":
        aux_names = [f"{view}_view:{c}" for view, model in zip(("execution", "space"), codebooks) for c in model.catalog_ids]
    for record in records:
        record = dict(record)
        selected = record["values"][:, fitted["columns"]]
        if route in {"feature_only", "within_kpms", "joint"}:
            labels = codebooks[0].assign(selected, record["parent_ids"])
        else:
            labels = np.asarray(record["parent_ids"], dtype=str)
        valid = labels != ""
        continuous = np.clip((selected-np.asarray(fitted["location"]))/np.asarray(fitted["scale"]), -8, 8)
        model_features = continuous if direct else np.empty((len(labels), 0))
        if direct:
            valid &= np.isfinite(continuous).all(axis=1)
        aux = np.empty((len(labels), 0))
        if route == "parallel":
            first = codebooks[0].assign(selected, record["parent_ids"])
            second = codebooks[1].assign(record["values"][:, fitted["spatial_columns"]], record["parent_ids"])
            valid &= (first != "") & (second != "")
            blocks = []
            for labels_view, model in zip((first, second), codebooks):
                crosswalk = {c: i for i, c in enumerate(model.catalog_ids)}
                block = np.zeros((len(labels), len(crosswalk)))
                positions = np.flatnonzero(labels_view != "")
                block[positions, [crosswalk[t] for t in labels_view[positions]]] = 1.
                blocks.append(block)
            aux = np.column_stack(blocks)
            model_features = np.column_stack([model_features, aux])
        labels = np.array([fitted["mapping"].get(str(t), "") for t in labels], dtype=object)
        valid &= labels != ""
        for field in ("start_frames", "stop_frames", "parent_ids", "values"):
            record[field] = record[field][valid]
        record.update(labels=labels[valid], model_features=model_features[valid], auxiliary=aux[valid],
            catalog=fitted["catalog"], representation_id=representation_id, removed_feature_bouts=int((~valid).sum()),
            consumed_features=(fitted["feature_names"] if direct else [])+aux_names)
        outputs.append(record)
    return outputs


def _record_stream(record, model_id, *, source=False):
    from fpcode.contracts.behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
    from fpcode.contracts.sessions import SessionKey
    starts, stops = record["start_frames"], record["stop_frames"]
    bounds = []
    for start, stop in zip(starts, stops):
        if bounds and bounds[-1].stop == int(start):
            bounds[-1] = FrameSpan(bounds[-1].start, int(stop))
        else:
            bounds.append(FrameSpan(int(start), int(stop)))
    labels = record["parent_ids"] if source else record["labels"]
    catalogue = record["source_vocabulary"] if source else record["catalog"]
    return UnitStream(SessionKey(record["recording"], record["subject"], record["cohort"]), "primitive", model_id,
        record["clock"], record["fps"], tuple(bounds), tuple(UnitType(t) for t in catalogue),
        tuple(UnitOccurrence(f"bout_{i}", str(t), FrameSpan(int(a), int(b)))
              for i, (a, b, t) in enumerate(zip(starts, stops, labels))))


def _blocks_of_record(record):
    n = len(record["labels"])
    starts = np.flatnonzero(np.r_[True, record["start_frames"][1:] != record["stop_frames"][:-1]]) if n else np.array([], dtype=int)
    return [np.arange(a, b) for a, b in zip(starts, np.r_[starts[1:], n])]


def _encode_record(record, length, feature_weight=1.):
    """Nonnegative signed feature blocks and categorical views, gap-separated."""
    crosswalk = {t: i for i, t in enumerate(record["catalog"])}
    k, f = len(crosswalk), record["model_features"].shape[1]
    columns, masks, indices = [], [], []
    for rows in _blocks_of_record(record):
        x = np.zeros((k+2*f, len(rows)+length), dtype=np.float32)
        x[[crosswalk[t] for t in record["labels"][rows]], np.arange(len(rows))] = 1.
        if f:
            z = record["model_features"][rows].T
            weight = np.sqrt(feature_weight/max(1, f))
            x[k:k+f, :len(rows)] = np.maximum(z, 0)*weight
            x[k+f:, :len(rows)] = np.maximum(-z, 0)*weight
        columns.append(x)
        masks.extend([True]*max(0, len(rows)-length+1)+[False]*min(len(rows), length-1)+[False]*length)
        indices.extend(rows.tolist()+[-1]*length)
    if not columns:
        return np.zeros((k+2*f, 0), dtype=np.float32), np.array([], dtype=bool), np.array([], dtype=int)
    return np.concatenate(columns, axis=1), np.asarray(masks, dtype=bool), np.asarray(indices, dtype=int)


def _propose_motifs(records, recipe):
    counts, prevalence = Counter(), Counter()
    for animal in sorted({r["subject"] for r in records}):
        local = Counter()
        for r in records:
            if r["subject"] != animal:
                continue
            for positions in _blocks_of_record(r):
                tokens = r["labels"][positions].tolist()
                for length in sorted(set((max(3, recipe["length"]-1), recipe["length"]))):
                    local.update(tuple(tokens[i:i+length]) for i in range(len(tokens)-length+1))
        total = max(1, sum(local.values()))
        for key, value in local.items():
            counts[key] += value/total
            prevalence[key] += 1
    candidates = sorted((m for m in counts if prevalence[m] >= recipe["min_prevalence"]), key=lambda m: (-counts[m], m))
    diverse = [m for m in candidates if len(set(m)) >= 3]
    chosen = []
    for item in diverse[:recipe["factors"]//2]+candidates:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= recipe["factors"]:
            break
    return chosen


def _window_profile(record, start, stop):
    values = record["model_features"][start:stop]
    if not values.shape[1]:
        return np.empty((0,))
    return np.concatenate([np.mean(part, axis=0) for part in np.array_split(values, min(3, len(values)))])


def _exact_rows(record, motif):
    for positions in _blocks_of_record(record):
        for a in motif_hits(record["labels"][positions], np.asarray(motif)):
            yield int(positions[a]), int(positions[a+len(motif)-1])+1


def _fit_recipe_model(training, recipe, representation, identity, cache):
    from dataclasses import replace

    from fpcode.behavior.episodes.staged import StagedEpisodeModel, fit_staged_model
    from fpcode.common.cache_manifest import cached_npz_component

    method = recipe["method"]
    shared = {k: recipe[k] for k in ("method", "length", "factors", "min_prevalence", "seed", "feature_weight")}
    if method == "seqnmf":
        shared.update({k: recipe[k] for k in ("penalty", "iterations", "activation_quantile")})
    if method == "hsmm":
        shared.update({k: recipe[k] for k in ("iterations", "phase_count", "phase_max", "background_max",
            "entry_probability", "early_exit_probability", "skip_probability", "reentry", "extra_background",
            "learn_durations", "initialization", "pseudocount", "variance_floor", "hsmm_targets")})
    # No transform/fit invalidation for descriptions not consumed by the model.
    key_identity = {**identity, "component": "episode_model", "representation": representation, "parameters": shared}

    def fit():
        import time

        started = time.perf_counter()
        payload, metadata = {}, {"method": method, "parameters": shared,
            "training_animals": sorted({r["subject"] for r in training}),
            "training_recordings": [r["recording"] for r in training],
            "training_events": sum(len(r["labels"]) for r in training),
            "consumed_features": training[0]["consumed_features"],
            "catalogue": training[0]["catalog"], "dtype": "float32" if method == "seqnmf" else "float64"}
        if method == "seqnmf":
            encoded = [_encode_record(r, recipe["length"], recipe["feature_weight"]) for r in training]
            x, valid = np.concatenate([v[0] for v in encoded], axis=1), np.concatenate([v[1] for v in encoded])
            if x.nbytes*16 > 256*1024**2:
                raise MemoryError("seqNMF intermediate array budget exceeded")
            model = None
            if valid.any():
                model = SeqNMF(recipe["factors"], recipe["length"], penalty=recipe["penalty"],
                               iterations=recipe["iterations"], seed=recipe["seed"]).fit(x, valid_starts=valid)
                payload.update(w=model.w, thresholds=np.maximum(np.quantile(model.h[:, valid], recipe["activation_quantile"], axis=1), 1e-7))
            else:
                metadata["empty_reason"] = "no supported training windows at requested length"
            metadata.update(loss_history=model.loss_history if model is not None else [],
                fitted_iterations=recipe["iterations"] if model is not None else 0,
                initialized_from="seeded positive templates and training windows",
                encoding="categorical one-hot plus signed nonnegative split of standardized features; separate parallel view blocks",
                estimated_working_bytes=x.nbytes*16,
                fixed_parameters=["factor_count", "template_length", "penalty"],
                score_semantics="uncentered normalized root reconstruction error, not ordinary R squared")
        else:
            motifs = _propose_motifs(training, recipe)
            if not motifs:
                metadata["empty_reason"] = "no proposal met training-animal prevalence"
            metadata["motifs"] = motifs
            gates, limits = [], []
            for motif in motifs:
                profiles = [_window_profile(r, a, b) for r in training for a, b in _exact_rows(r, motif)]
                if profiles and len(profiles[0]):
                    profiles = np.asarray(profiles)
                    center = np.mean(profiles, axis=0)
                    limits.append(max(.05, float(np.quantile(np.mean((profiles-center)**2, axis=1), .9))))
                    gates.append(center)
                else:
                    gates.append(np.empty(0))
                    limits.append(.05)
            if gates:
                payload["gate_centers"] = np.asarray(gates)
                payload["gate_limits"] = np.asarray(limits)
            if method == "hsmm" and motifs:
                streams = [_record_stream(r, f"representation:{representation}") for r in training if len(r["labels"])]
                feature_arrays = [r["model_features"] for r in training if len(r["labels"])]
                background_count = Counter(t for r in training for t in r["labels"])
                catalog = streams[0].catalog
                total = sum(background_count.values())+len(catalog)
                background = {u.type_id: (background_count[u.type_id]+1)/total for u in catalog}
                metadata["targets"] = []
                for target, motif in enumerate(motifs[:recipe["hsmm_targets"]]):
                    phase_count = min(recipe["phase_count"], len(motif))
                    chunks = np.array_split(np.asarray(motif), phase_count)
                    phases = tuple(f"phase_{i}" for i in range(phase_count))
                    emissions = {}
                    for phase, chunk in zip(phases, chunks):
                        counts = Counter(chunk)
                        emissions[phase] = {key: .2*p+.8*counts[key]/len(chunk) for key, p in background.items()}
                    phase_duration = .65*.35**np.arange(recipe["phase_max"])
                    phase_duration /= phase_duration.sum()
                    background_duration = .08*.92**np.arange(recipe["background_max"])
                    background_duration /= background_duration.sum()
                    model = StagedEpisodeModel.create_linear_routine(f"configured:{target}", f"routine_{target}",
                        phases, catalog, emissions, background,
                        phase_durations={p: phase_duration for p in phases}, background_duration=background_duration,
                        allow_immediate_reentry=recipe["reentry"], extra_background=recipe["extra_background"],
                        entry_probability=recipe["entry_probability"], early_exit_probability=recipe["early_exit_probability"],
                        skip_probability=recipe["skip_probability"])
                    f = feature_arrays[0].shape[1]
                    if f:
                        values = np.concatenate(feature_arrays)
                        means = np.tile(values.mean(axis=0), (model.log_emissions.shape[0], 1))
                        examples = [r["model_features"][a:b] for r in training for a, b in _exact_rows(r, motif)]
                        for i in range(phase_count):
                            slices = [np.array_split(v, phase_count)[i] for v in examples]
                            means[i+1] = np.concatenate(slices).mean(axis=0)
                        model = replace(model, feature_names=tuple(training[0]["consumed_features"]),
                            feature_means=means, feature_variances=np.tile(np.maximum(values.var(axis=0), .2), (len(means), 1)),
                            feature_weight=recipe["feature_weight"]/max(1, f))
                    result = fit_staged_model(model, streams, model_id=f"trained:{target}",
                        source_model_id=f"representation:{representation}", max_iterations=recipe["iterations"],
                        seed=recipe["seed"]+target, initialization=recipe["initialization"], pseudocount=recipe["pseudocount"],
                        max_observations=30000, max_total_work=4_000_000_000,
                        training_features=feature_arrays if f else None, learn_durations=recipe["learn_durations"],
                        variance_floor=recipe["variance_floor"])
                    fitted = result.model
                    for key in ("log_initial", "log_transition", "log_emissions", "log_duration_pmfs", "feature_means", "feature_variances"):
                        value = getattr(fitted, key)
                        if value is not None:
                            payload[f"target_{target}_{key}"] = value
                    metadata["targets"].append(dict(phase_names=phases, feature_names=fitted.feature_names,
                        feature_weight=fitted.feature_weight, extra_background=fitted.extra_background,
                        allow_phase_skips=fitted.allow_phase_skips, fit_diagnostics=_json_arrays(dict(result.diagnostics))))
                metadata["score_semantics"] = "selected-path weight per source bout; independent seeded targets, not a joint HSMM catalogue"
            elif method not in {"exact", "tolerant", "grammar", "hsmm"}:
                raise ValueError("unsupported episode method")
            metadata["proposal_conditions"] = "animal-balanced training frequency plus diverse categorical composition; no phenotype selection"
            metadata["feature_gate_semantics"] = "training-calibrated three-part standardized execution profile distance; zero extra features means no gate"
        metadata["fitting_seconds"] = time.perf_counter()-started
        payload["metadata"] = np.array(json.dumps(_json_arrays(metadata), sort_keys=True, allow_nan=False))
        return payload

    arrays, hit, key = cached_npz_component(cache, key_identity, fit)
    return arrays, json.loads(str(arrays["metadata"])), hit, key


def _predict_recipe(record, recipe, arrays, metadata, representation, model_id, *, max_working_bytes=256*1024**2,
                    collect_stages=False):
    """Return a common raw episode stream and method-specific held diagnostics."""
    from fpcode.behavior.episodes.chains import edit_alignment
    from fpcode.behavior.episodes.staged import StagedEpisodeModel
    from fpcode.contracts.behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType

    method = recipe["method"]
    source = _record_stream(record, f"representation:{representation}")
    candidates, diagnostics = [], {}
    if method == "seqnmf":
        catalogue = [f"factor_{i}" for i in range(recipe["factors"])]
        x, valid, positions = _encode_record(record, recipe["length"], recipe["feature_weight"])
        if valid.any() and "w" in arrays:
            if x.nbytes*16 > max_working_bytes:
                raise MemoryError("seqNMF held inference array budget exceeded")
            model = SeqNMF(recipe["factors"], recipe["length"], penalty=recipe["penalty"])
            model.w = arrays["w"]
            h = model.infer(x, valid_starts=valid, iterations=35)
            for c in activation_events(h, model.w, x, thresholds=arrays["thresholds"], valid_starts=valid,
                                       max_error=recipe["max_error"]):
                a, b = int(positions[c["start"]]), int(positions[c["end"]-1])+1
                candidates.append((catalogue[c["factor"]], a, b, 1-c["template_error"]))
            supported = positions >= 0
            error = np.linalg.norm((x-reconstruct(model.w, h))[:, supported])/max(np.linalg.norm(x[:, supported]), 1e-9)
            diagnostics["normalized_root_error"] = float(error)
    elif method == "hsmm":
        catalogue = [f"routine_{i}" for i in range(len(metadata.get("targets", [])))]
        likelihood = 0.
        for target, info in enumerate(metadata.get("targets", [])):
            kw = {key: arrays[f"target_{target}_{key}"] for key in
                  ("log_initial", "log_transition", "log_emissions", "log_duration_pmfs")}
            if info["feature_names"]:
                kw.update(feature_means=arrays[f"target_{target}_feature_means"],
                          feature_variances=arrays[f"target_{target}_feature_variances"],
                          feature_names=tuple(info["feature_names"]), feature_weight=info["feature_weight"])
            model = StagedEpisodeModel(model_id=f"{model_id}/target_{target}", routine_name=f"routine_{target}",
                phase_names=tuple(info["phase_names"]), primitive_catalog=source.catalog,
                extra_background=info["extra_background"], allow_phase_skips=info["allow_phase_skips"], **kw)
            if len(record["labels"]):
                result = model.decode_stream(source, features=record["model_features"] if info["feature_names"] else None)
                likelihood += result.diagnostics["log_likelihood"]
                for o in result.occurrences:
                    a, b = np.searchsorted(record["start_frames"], o.interval.start), np.searchsorted(record["stop_frames"], o.interval.stop, side="right")
                    candidates.append((o.type_id, int(a), int(b), o.score/max(1, b-a)))
                    if collect_stages:
                        completion = next((c for c in result.completion if c.occurrence_id == o.occurrence_id), None)
                        diagnostics.setdefault("stages", []).append(dict(type_id=o.type_id,
                            start_frame=o.interval.start, stop_frame=o.interval.stop,
                            phases=[dict(phase_id=p.phase_id, order=p.order, start_frame=p.interval.start,
                                         stop_frame=p.interval.stop, evidence_kind=p.evidence_kind)
                                    for p in result.phases if p.parent_occurrence_id == o.occurrence_id],
                            completion=None if completion is None else dict(
                                status=completion.status, observed_entry=completion.observed_entry,
                                observed_exit=completion.observed_exit, final_phase_reached=completion.final_phase_reached,
                                left_censored=completion.left_censored, right_censored=completion.right_censored,
                                evidence_kind=completion.evidence_kind)))
        diagnostics["mean_target_log_likelihood_per_event"] = likelihood/max(1, len(record["labels"])*len(catalogue))
    else:
        motifs = metadata.get("motifs", [])
        catalogue = [f"routine_{i}" for i in range(len(motifs))]
        if method == "grammar" and motifs:
            from fpcode.behavior.episodes.grammar import GrammarEpisodeDetector
            from fpcode.contracts.composition import (
                GrammarRule,
                GrammarSpecification,
                OptionalRule,
                RepeatRule,
                SequenceRule,
                TerminalMatch,
            )
            rules = []
            for i, motif in enumerate(motifs):
                parts = [TerminalMatch(t) for t in motif]
                if recipe["grammar_optional"] and len(parts) > 3:
                    parts[len(parts)//2] = OptionalRule(parts[len(parts)//2])
                if recipe["grammar_repeat"] > 1:
                    parts[1] = RepeatRule(parts[1], 1, recipe["grammar_repeat"])
                rules.append(GrammarRule(f"routine_{i}", SequenceRule(tuple(parts))))
            parsed = GrammarEpisodeDetector(GrammarSpecification("training_seeded_finite", tuple(rules))).detect(
                source, max_candidates=20000, max_derivations=4, max_work=2_000_000)
            for o in parsed.occurrences:
                a, b = np.searchsorted(record["start_frames"], o.interval.start), np.searchsorted(record["stop_frames"], o.interval.stop, side="right")
                candidates.append((o.type_id, int(a), int(b), o.score/max(1, b-a)))
            diagnostics["grammar_semantics"] = "training-seeded finite optional/repeat rules, not unrestricted grammar induction"
        else:
            for k, motif in enumerate(motifs):
                if method == "exact":
                    candidates.extend((f"routine_{k}", a, b, 1.) for a, b in _exact_rows(record, motif))
                elif method == "tolerant":
                    for positions in _blocks_of_record(record):
                        for start in range(len(positions)):
                            best = None
                            for length in range(max(3, len(motif)-recipe["edit_budget"]), len(motif)+recipe["edit_budget"]+1):
                                if start+length > len(positions):
                                    continue
                                observed = record["labels"][positions[start:start+length]]
                                edits = edit_alignment(motif, observed)["edit_distance"]
                                rank = (edits, abs(length-len(motif)), length)
                                if best is None or rank < best[0]:
                                    best = (rank, length)
                            if best is not None and best[0][0] <= recipe["edit_budget"]:
                                a, b = int(positions[start]), int(positions[start+best[1]-1])+1
                                candidates.append((f"routine_{k}", a, b, 1-best[0][0]/max(len(motif), best[1])))
                else:
                    raise ValueError("unsupported detector family")
        if record["model_features"].shape[1] and "gate_centers" in arrays:
            retained = []
            for token, a, b, score in candidates:
                k = int(token.split("_")[-1])
                profile = _window_profile(record, a, b)
                if profile.shape == arrays["gate_centers"][k].shape:
                    distance = float(np.mean((profile-arrays["gate_centers"][k])**2))
                    if distance <= recipe["gate_multiplier"]*arrays["gate_limits"][k]:
                        retained.append((token, a, b, score-distance/100))
            diagnostics["feature_gate_removed"] = len(candidates)-len(retained)
            candidates = retained
    occurrences = []
    for token, a, b, score in candidates:
        if not 0 <= a < b <= len(record["labels"]) or np.any(record["start_frames"][a+1:b] != record["stop_frames"][a:b-1]):
            raise ValueError("detector call crosses native support barriers")
        occurrences.append(UnitOccurrence(f"call_{len(occurrences)}", token,
            FrameSpan(int(record["start_frames"][a]), int(record["stop_frames"][b-1])), float(score), "model_score"))
    return UnitStream(source.session, "episode", model_id, source.clock, source.fps,
        source.valid_intervals, tuple(UnitType(t) for t in catalogue), tuple(occurrences)), diagnostics


def _finite_json(value):
    """Serialize missing diagnostics as null, retaining zero as an observed value."""
    value = _json_arrays(value)
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@lru_cache(maxsize=1)
def _recipe_implementation_digest():
    """Fingerprint component owners lazily during explicit numerical execution."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    owners = ("workflows/sequence_search.py", "behavior/primitives/segmentation.py",
              "behavior/episodes/seqnmf.py", "behavior/episodes/staged.py",
              "behavior/episodes/chains.py", "behavior/episodes/grammar.py",
              "statistics/sequence_inference.py", "statistics/finite_composition.py",
              "contracts/behavior_units.py", "common/cache_manifest.py")
    digest = hashlib.sha256()
    for owner in owners:
        digest.update(owner.encode())
        digest.update((root/owner).read_bytes())
    return digest.hexdigest()


def _validate_recipe_records(records, recipe):
    """Validate the narrow executable contract before fitting or writing a cache."""
    if not records or recipe.get("method") not in {"exact", "tolerant", "seqnmf", "hsmm", "grammar"}:
        raise ValueError("nonempty records and a supported episode method required")
    if recipe.get("route") not in {"kpms", "feature_only", "within_kpms", "joint", "parallel"}:
        raise ValueError("unsupported representation route")
    if recipe.get("feature_role") not in {"describe", "episode"}:
        raise ValueError("feature role must be describe or episode")
    if recipe.get("resolution") not in {"identity", "annotation", "quantitative"}:
        raise ValueError("unsupported primitive resolution")
    if recipe["resolution"] == "annotation" and recipe["route"] not in {"kpms", "parallel"}:
        raise ValueError("model-specific KPMS annotations cannot label a new feature vocabulary")
    for name in ("length", "factors", "min_prevalence", "clusters", "iterations"):
        if isinstance(recipe.get(name), bool) or not isinstance(recipe.get(name), int) or recipe[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if recipe["length"] < 3 or not np.isfinite(recipe["feature_weight"]) or recipe["feature_weight"] <= 0:
        raise ValueError("length >=3 and positive finite feature weight required")
    if recipe["method"] in {"exact", "tolerant", "grammar"} and recipe["feature_weight"] != 1:
        raise ValueError("profile-gate methods have no independent feature-weight contrast; use 1")
    allowed = {"method", "route", "feature_role", "bundle", "resolution", "length", "factors", "min_prevalence",
        "seed", "feature_weight", "clusters", "merge_clusters", "creator_seed", "parent_weight", "iterations", "penalty",
        "activation_quantile", "max_error", "phase_count", "phase_max", "background_max", "entry_probability",
        "early_exit_probability", "skip_probability", "reentry", "extra_background", "learn_durations",
        "initialization", "pseudocount", "variance_floor", "hsmm_targets", "edit_budget", "gate_multiplier",
        "grammar_optional", "grammar_repeat", "base_id", "representation_cell", "active_parameters"}
    if set(recipe)-allowed:
        raise ValueError("unsupported recipe parameters: "+str(sorted(set(recipe)-allowed)))
    scope = (records[0]["cohort"], records[0]["clock"], records[0]["fps"], tuple(records[0]["source_vocabulary"]))
    recordings = set()
    for r in records:
        if (r["cohort"], r["clock"], r["fps"], tuple(r["source_vocabulary"])) != scope:
            raise ValueError("one recipe requires a single explicitly scoped cohort/vocabulary/clock")
        if r["split"] not in {"train", "held"} or not r["subject"] or r["recording"] in recordings:
            raise ValueError("explicit split, animal and unique recording identity required")
        recordings.add(r["recording"])
        a, b, v, p = (np.asarray(r[k]) for k in ("start_frames", "stop_frames", "values", "parent_ids"))
        if (a.ndim != 1 or b.shape != a.shape or not np.issubdtype(a.dtype, np.integer)
                or not np.issubdtype(b.dtype, np.integer) or np.any(a < 0) or np.any(b <= a)
                or np.any(a[1:] < b[:-1]) or p.shape != a.shape
                or v.shape != (len(a), len(r["feature_names"]))):
            raise ValueError("prepared features must align with ordered complete nonoverlapping native bouts")
        if any(str(t) not in scope[3] for t in p):
            raise ValueError("source label outside declared model vocabulary")
        if not np.isfinite(r["fps"]) or r["fps"] <= 0:
            raise ValueError("positive declared native frame rate required")


def _selected_stream(stream):
    """Retain raw evidence and apply the existing explicit within-model resolver."""
    from dataclasses import replace

    from fpcode.behavior.analysis.organisation import exclusive_timeline
    from fpcode.behavior.unit_adapters import occurrence_table
    table = occurrence_table(stream).rename(columns={"type_id": "chain_id"})
    table["selection_score"] = table.score
    blocks = np.asarray([(s.start, s.stop) for s in stream.valid_intervals], dtype=np.int64).reshape(-1, 2)
    selected, decisions, _ = exclusive_timeline(table, blocks, fps=stream.fps,
        recording_end_frame=stream.valid_intervals[-1].stop if stream.valid_intervals else 0)
    keep = set(selected.occurrence_id)
    return replace(stream, occurrences=tuple(o for o in stream.occurrences if o.occurrence_id in keep)), decisions


def _call_rows(stream):
    return [dict(occurrence_id=o.occurrence_id, type_id=o.type_id,
                 start_frame=o.interval.start, stop_frame=o.interval.stop, score=o.score)
            for o in stream.occurrences]


def predict_frozen_recipe(record, recipe, representation, model_arrays, conditions, *,
                          representation_id, model_id, max_working_bytes=256*1024**2):
    """Infer one explicit recording using supplied frozen components, never fit.

    The caller owns component identity/file verification and training/held animal
    policy. Returned raw/selected intervals use existing native score-ranked
    selection; scores are never calibrated across detector families.
    """
    if isinstance(max_working_bytes, bool) or not isinstance(max_working_bytes, int) or max_working_bytes < 1:
        raise ValueError("positive inference working-array cap required")
    _validate_recipe_records([record], recipe)
    if conditions["method"] != recipe["method"] or conditions["catalogue"] != representation["catalog"]:
        raise ValueError("frozen model method/catalogue disagrees with representation")
    for name, value in conditions.get("parameters", {}).items():
        if name in recipe and value != recipe[name]:
            raise ValueError("frozen model parameter disagrees with inference recipe: " + name)
    r = transform_frozen_recipe_records([record], recipe, representation, representation_id=representation_id)[0]
    raw, diagnostics = _predict_recipe(r, recipe, model_arrays, conditions, representation_id, model_id,
                                       max_working_bytes=max_working_bytes, collect_stages=True)
    selected, audit = _selected_stream(raw)
    exposure = float(np.sum(record["stop_frames"]-record["start_frames"])/record["fps"])
    supported = sum(s.stop-s.start for s in raw.valid_intervals)/record["fps"]
    covered = sum(o.interval.stop-o.interval.start for o in selected.occurrences)/record["fps"]
    return dict(recording=record["recording"], subject=record["subject"], clock=record["clock"], fps=record["fps"],
        original_source_bouts=len(record["parent_ids"]), available_bouts=len(r["labels"]),
        removed_feature_bouts=r["removed_feature_bouts"], source_exposure_s=exposure,
        available_exposure_s=supported, covered_exposure_s=covered,
        valid_intervals=[[s.start, s.stop] for s in raw.valid_intervals], consumed_features=r["consumed_features"],
        raw_calls=_call_rows(raw), selected_calls=_call_rows(selected),
        overlap_rejections=int((audit.rejection_reason == "overlap").sum()), diagnostics=diagnostics)


def _physical_quality(records, streams, training, consumed, *, seed, pair_cap=32):
    """Bounded physical profile agreement, with feature-selection circularity explicit."""
    names = records[0]["feature_names"]
    all_train = np.concatenate([r["values"] for r in training])
    location, scale = np.zeros(len(names)), np.ones(len(names))
    for i in range(len(names)):
        z = all_train[:, i]
        z = z[np.isfinite(z)]
        if len(z):
            location[i], scale[i] = np.median(z), max(float(np.quantile(z, .75)-np.quantile(z, .25)), 1e-9)
            if scale[i] < 1e-8:
                scale[i] = 1.
    groups = {}
    for r, stream in zip(records, streams):
        z = np.clip((r["values"]-location)/scale, -8, 8)
        sums = np.vstack([np.zeros(len(names)), np.cumsum(np.where(np.isfinite(z), z, 0.), axis=0)])
        counts = np.vstack([np.zeros(len(names)), np.cumsum(np.isfinite(z), axis=0)])
        for o in stream.occurrences:
            a = np.searchsorted(r["start_frames"], o.interval.start)
            b = np.searchsorted(r["stop_frames"], o.interval.stop, side="right")
            count = counts[b]-counts[a]
            profile = np.divide(sums[b]-sums[a], count, out=np.full(len(names), np.nan), where=count > 0)
            groups.setdefault(o.type_id, []).append(profile)
    rng = np.random.default_rng(seed)
    rows = []
    for token in sorted(groups):
        own = np.asarray(groups[token])
        other = np.asarray([p for key, profiles in groups.items() if key != token for p in profiles])
        distances = {}
        for role, values in (("within", own), ("other_type", other)):
            if not len(values) or (role == "within" and len(own) < 2):
                distances[role] = dict(pairs=0, mean_squared_standardized_distance=None)
                continue
            first = rng.integers(len(own), size=pair_cap)
            if role == "within":
                second = (first+1+rng.integers(len(own)-1, size=pair_cap)) % len(own)
            else:
                second = rng.integers(len(values), size=pair_cap)
            delta = (own[first]-values[second])**2
            valid = np.isfinite(delta)
            distance = np.divide(np.where(valid, delta, 0).sum(axis=1), valid.sum(axis=1),
                                 out=np.full(pair_cap, np.nan), where=valid.sum(axis=1) > 0)
            finite = distance[np.isfinite(distance)]
            distances[role] = dict(pairs=len(finite), mean_squared_standardized_distance=float(finite.mean()) if len(finite) else None)
        rows.append(dict(type_id=token, occurrences=len(own), **distances))
    return dict(types=rows, feature_names=names, features_used_in_construction=sorted(consumed),
                interpretation="unweighted whole-bout mean profiles; consumed channels partly reward the fitted objective; no semantic ground truth",
                normalization="training-only median/IQR, clipped at 8", pair_cap=pair_cap, seed=seed)


def _control_record(record, training, *, kind, rng):
    """Fixed event-slot controls; physical durations are deliberately not surrogated."""
    from fpcode.statistics.chain_nulls import sample_first_order_bouts
    result = dict(record)
    labels, features = record["labels"].copy(), record["model_features"].copy()
    reference = [r["labels"][b] for r in training for b in _blocks_of_record(r)]
    catalog = record["catalog"]
    code = {t: i for i, t in enumerate(catalog)}
    encoded = [np.array([code[t] for t in z], dtype=np.int64) for z in reference]
    feature_rows = {}
    if kind == "first_order" and features.shape[1]:
        for token in catalog:
            parts = [r["model_features"][r["labels"] == token] for r in training]
            feature_rows[token] = np.concatenate(parts)
    for rows in _blocks_of_record(record):
        if kind == "shuffle":
            permutation = rng.permutation(rows)
            labels[rows], features[rows] = record["labels"][permutation], record["model_features"][permutation]
        else:
            sampled = sample_first_order_bouts(encoded, length=len(rows), initial=code[record["labels"][rows[0]]],
                vocabulary=len(catalog), replicates=1, seed=int(rng.integers(2**31)), forbid_self=False)[0]
            labels[rows] = np.asarray(catalog)[sampled]
            if features.shape[1]:
                for token in np.unique(labels[rows]):
                    positions = rows[labels[rows] == token]
                    available = feature_rows[token]
                    if len(available):
                        features[positions] = available[rng.integers(len(available), size=len(positions))]
                    elif len(positions) == 1 and positions[0] == rows[0]:
                        # Condition on the observed initial category/feature pair;
                        # subsequent transitions use only training support.
                        features[positions] = record["model_features"][positions]
                    else:
                        raise ValueError("first-order feature control lacks training support for a sampled type")
    result.update(labels=labels, model_features=features)
    return result


def evaluate_recipe(records, recipe, identity, cache_directory, *, null_repeats=2):
    """Fit one bounded animal-held-out recipe and return auditable quantitative evidence.

    No file loader, phenotype/drug/DA selector or cross-cohort ID comparison is used.
    Descriptive metrics, fitting conditions, raw calls and one explicit selected
    view remain separate. Controls do not refit the detector and are exploratory
    event-slot diagnostics, not p-values or faithful physical-trajectory surrogates.
    """
    import time

    from fpcode.behavior.analysis.unit_views import aggregate_unit_composition, measure_unit_composition

    if isinstance(null_repeats, bool) or not isinstance(null_repeats, int) or null_repeats < 0:
        raise ValueError("null_repeats must be a nonnegative integer")
    _validate_recipe_records(records, recipe)
    identity = {**identity, "implementation_digest": _recipe_implementation_digest()}
    if len({r["subject"] for r in records if r["split"] == "held"}) == 0:
        raise ValueError("held animal evidence required")
    started = time.perf_counter()
    represented, representation, representation_hit, representation_key = _representation_inputs(records, recipe, identity, cache_directory)
    training = [r for r in represented if r["split"] == "train"]
    arrays, conditions, model_hit, model_key = _fit_recipe_model(training, recipe, representation_key, identity, cache_directory)
    model_id = "recipe:"+hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    tables, streams, held_original, result_rows = [], [], [], []
    held_events, held_exposure, supported_exposure, covered_exposure, raw_count, selected_count = 0, 0., 0., 0., 0, 0
    for original, r in zip(records, represented):
        if r["split"] != "held":
            continue
        raw, diagnostics = _predict_recipe(r, recipe, arrays, conditions, representation_key, model_id)
        selected, audit = _selected_stream(raw)
        source = _record_stream(original, identity.get("source_model_id", "source:"+identity.get("cohort", r["cohort"])), source=True)
        table = measure_unit_composition(selected, source)
        if len(table) and (not table.lower_support_complete.all() or table.edge_bout_clipped.any()):
            raise ValueError("selected call does not contain complete supported original source bouts")
        tables.append(table)
        streams.append(selected)
        held_original.append(original)
        exposure = float(np.sum(original["stop_frames"]-original["start_frames"])/r["fps"])
        supported = sum(s.stop-s.start for s in raw.valid_intervals)/r["fps"]
        covered = sum(o.interval.stop-o.interval.start for o in selected.occurrences)/r["fps"]
        held_events += len(r["labels"])
        held_exposure += exposure
        supported_exposure += supported
        covered_exposure += covered
        raw_count += len(raw.occurrences)
        selected_count += len(selected.occurrences)
        result_rows.append(dict(recording=r["recording"], subject=r["subject"], clock=r["clock"], fps=r["fps"],
            original_source_bouts=len(original["parent_ids"]), available_bouts=len(r["labels"]),
            removed_feature_bouts=r["removed_feature_bouts"], source_exposure_s=exposure,
            available_exposure_s=supported, covered_exposure_s=covered,
            valid_intervals=[(s.start, s.stop) for s in raw.valid_intervals],
            raw_calls=_call_rows(raw), selected_calls=_call_rows(selected),
            overlap_rejections=int((audit.rejection_reason == "overlap").sum()), diagnostics=diagnostics))
    quality = aggregate_unit_composition(tables, max_pairs_per_type=32, seed=recipe["seed"])
    quality = {key: value.to_dict(orient="records") for key, value in quality.items()}
    consumed = set(conditions["consumed_features"])
    if recipe["route"] != "kpms" or recipe["resolution"] == "quantitative":
        consumed.update(representation["feature_names"])
    if recipe["route"] == "parallel":
        consumed.update(("centroid_x_px", "centroid_y_px"))
    quality["physical"] = _physical_quality(held_original, streams,
        [r for r in records if r["split"] == "train"], consumed, seed=recipe["seed"])
    controls = []
    rng = np.random.default_rng(recipe["seed"]+9001)
    for kind in ("shuffle", "first_order"):
        for repeat in range(null_repeats):
            if kind == "first_order" and not any(len(r["labels"]) for r in training):
                controls.append(dict(kind=kind, replicate=repeat, status="UNAVAILABLE",
                                     reason="no complete training observations after feature support filtering"))
                continue
            events, calls = 0, 0
            for r in represented:
                if r["split"] != "held" or not len(r["labels"]):
                    continue
                control = _control_record(r, training, kind=kind, rng=rng)
                raw, _ = _predict_recipe(control, recipe, arrays, conditions, representation_key, model_id)
                selected, _ = _selected_stream(raw)
                calls += len(selected.occurrences)
                for o in selected.occurrences:
                    a = np.searchsorted(r["start_frames"], o.interval.start)
                    b = np.searchsorted(r["stop_frames"], o.interval.stop, side="right")
                    events += b-a
            controls.append(dict(kind=kind, replicate=repeat, selected_calls=calls,
                                 selected_event_fraction=events/max(held_events, 1)))
    types = quality["types"]
    return _finite_json(dict(status="SUCCESS" if selected_count else "EMPTY", recipe=recipe,
        representation=representation, conditions=conditions,
        cache=dict(representation_hit=representation_hit, model_hit=model_hit,
                   representation_key=representation_key, model_key=model_key),
        summary=dict(raw_calls=raw_count, selected_calls=selected_count, held_events=held_events,
            held_animals=len({r["subject"] for r in records if r["split"] == "held"}),
            source_exposure_s=held_exposure, available_exposure_s=supported_exposure,
            covered_exposure_s=covered_exposure, coverage_fraction=covered_exposure/max(held_exposure, 1e-9),
            feature_support_fraction=supported_exposure/max(held_exposure, 1e-9),
            calls_per_available_minute=selected_count*60/max(supported_exposure, 1e-9),
            observed_types=sum(t["occurrences"] > 0 for t in types),
            one_animal_types=sum(t["known_animals"] == 1 for t in types),
            total_seconds=time.perf_counter()-started),
        records=result_rows, quality=quality, controls=controls,
        policies=dict(selection="existing score-ranked greedy resolver within each fitted method; scores across methods are not compared",
            controls="fixed native event slots; shuffle preserves category/feature pairs inside blocks; first-order labels fitted on training blocks allow self transitions because merged fixed source bouts may repeat; features sampled from training rows conditional on label",
            control_limit="no detector refit, no physical duration/category preservation in first-order controls, no transition uncertainty or p-value calibration; seed is detector seed+9001",
            physical_quality="all available named channels, training-only normalization; consumed features flagged for circularity",
            source_clock="native frames and explicit per-cohort rate; no comparison of numerical labels across models",
            inference="per-recording; gaps are barriers; exclusive selection retains complete original bouts")))


def compare_recipe_catalogues(first, second):
    """Match seed catalogues by native temporal overlap, never by type ID equality.

    Both results must cover identical recording-scoped available native intervals.
    This measures stability of calls after the same explicit exclusive selection,
    not semantic agreement or independent replication. Empty catalogues have no
    defined agreement instead of a perfect score.
    """
    from scipy.optimize import linear_sum_assignment

    a_records = {r["recording"]: r for r in first["records"]}
    b_records = {r["recording"]: r for r in second["records"]}
    if a_records.keys() != b_records.keys():
        raise ValueError("seed comparisons require identical recording support")
    a_types = sorted({c["type_id"] for r in a_records.values() for c in r["selected_calls"]})
    b_types = sorted({c["type_id"] for r in b_records.values() for c in r["selected_calls"]})
    if len(a_types)*len(b_types) > 16384:
        raise MemoryError("catalogue overlap matching budget exceeded")
    ai, bi = {t: i for i, t in enumerate(a_types)}, {t: i for i, t in enumerate(b_types)}
    overlap = np.zeros((len(ai), len(bi)))
    duration_a, duration_b = np.zeros(len(ai)), np.zeros(len(bi))
    for recording, ra in a_records.items():
        rb = b_records[recording]
        if any(ra[k] != rb[k] for k in ("subject", "clock", "fps", "valid_intervals")):
            raise ValueError("seed comparisons require identical animal, clock and native support")
        ca = sorted(ra["selected_calls"], key=lambda c: c["start_frame"])
        cb = sorted(rb["selected_calls"], key=lambda c: c["start_frame"])
        for calls, duration, index in ((ca, duration_a, ai), (cb, duration_b, bi)):
            if any(y["start_frame"] < x["stop_frame"] for x, y in zip(calls, calls[1:])):
                raise ValueError("catalogue comparison requires exclusive selected calls")
            for c in calls:
                duration[index[c["type_id"]]] += (c["stop_frame"]-c["start_frame"])/ra["fps"]
        i = j = 0
        while i < len(ca) and j < len(cb):
            x, y = ca[i], cb[j]
            intersection = max(0, min(x["stop_frame"], y["stop_frame"])-max(x["start_frame"], y["start_frame"]))
            overlap[ai[x["type_id"]], bi[y["type_id"]]] += intersection/ra["fps"]
            if x["stop_frame"] <= y["stop_frame"]:
                i += 1
            else:
                j += 1
    matching, matched = [], 0.
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        for i, j in zip(rows, cols):
            if overlap[i, j] > 0:
                matching.append(dict(first_type=a_types[i], second_type=b_types[j],
                                     overlap_s=float(overlap[i, j])))
                matched += overlap[i, j]
    denominator = duration_a.sum()+duration_b.sum()
    return dict(matched_type_temporal_f1=float(2*matched/denominator) if denominator else None,
                any_type_temporal_f1=float(2*overlap.sum()/denominator) if denominator else None,
                first_observed_types=len(ai), second_observed_types=len(bi), mapping=matching,
                first_covered_s=float(duration_a.sum()), second_covered_s=float(duration_b.sum()),
                matching="Hungarian maximum overlapped seconds over exclusive native calls; IDs carry no cross-seed meaning",
                limit="stable temporal allocation is not semantic ground truth; unmatched types reduce agreement")


def filter_short_bouts(frame, min_seconds):
    """Remove short bouts without allowing chains across deleted native intervals.

    Required columns are ``syllable_id``, ``segment``, ``duration_s``,
    ``start_frame`` and ``end_frame_exclusive``. Rows must be in native temporal
    order. All original columns and surviving bounds/durations are retained;
    segment IDs are replaced by consecutive IDs wherever a row was deleted, an
    original segment changed, or native intervals were not adjacent. Identical
    neighbours on opposite sides of a deletion are never fused.
    """
    required = {"syllable_id", "segment", "duration_s", "start_frame", "end_frame_exclusive"}
    if not required <= set(frame.columns):
        raise ValueError("bout filtering requires native bounds, durations, labels and segments")
    if isinstance(min_seconds, bool) or not np.isfinite(min_seconds) or min_seconds < 0:
        raise ValueError("minimum bout duration must be finite and nonnegative")
    out = frame.copy()
    if out.empty:
        return out.reset_index(drop=True)
    starts, ends = out.start_frame.to_numpy(), out.end_frame_exclusive.to_numpy()
    if (not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(ends.dtype, np.integer)
            or np.any(starts < 0) or np.any(ends <= starts)
            or np.any(starts[1:] < ends[:-1])):
        raise ValueError("native bout bounds must be increasing nonoverlapping integer intervals")
    durations = out.duration_s.to_numpy(float)
    if not np.isfinite(durations).all() or np.any(durations <= 0) or out.segment.isna().any():
        raise ValueError("bout durations and segment identities must be valid")
    positions = np.flatnonzero(durations >= min_seconds)
    out = out.iloc[positions].copy()
    if not len(positions):
        return out.reset_index(drop=True)
    source_segments = frame.segment.to_numpy()
    boundary = np.r_[True, (np.diff(positions) != 1)
                     | (source_segments[positions[1:]] != source_segments[positions[:-1]])
                     | (starts[positions[1:]] != ends[positions[:-1]])]
    out["segment"] = np.cumsum(boundary) - 1
    return out.reset_index(drop=True)


def motif_hits(tokens, template):
    """Vectorised exact comparison with chronological greedy nonoverlap calls."""
    tokens, template = np.asarray(tokens), np.asarray(template)
    if tokens.ndim != 1 or template.ndim != 1 or len(template) < 1:
        raise ValueError("motifs and token sequences must be one-dimensional and motif nonempty")
    size = len(tokens) - len(template) + 1
    if size <= 0:
        return np.empty(0, dtype=int)
    match = np.ones(size, dtype=bool)
    for lag, value in enumerate(template):
        match &= tokens[lag:lag + size] == value
    calls, stop = [], 0
    for start in np.flatnonzero(match):
        if start >= stop:
            calls.append(int(start))
            stop = start + len(template)
    return np.asarray(calls, dtype=int)


def motif_masks(segments, motifs):
    """Return K-by-bout coverage; independent segments can never form a match."""
    offsets = np.r_[0, np.cumsum([len(tokens) for tokens in segments])]
    masks = np.zeros((len(motifs), int(offsets[-1])), dtype=bool)
    for k, motif in enumerate(motifs):
        for tokens, offset in zip(segments, offsets):
            for start in motif_hits(tokens, motif):
                masks[k, offset + start:offset + start + len(motif)] = True
    return masks


def frozen_seqnmf_model(artifact, configuration, *, vocabulary):
    """Decode a saved float32 template without fitting or modifying its payload."""
    payload = artifact["templates"]
    if payload.get("encoding") != "base64" or payload.get("dtype") != "<f4":
        raise ValueError("unsupported frozen template encoding")
    expected = (vocabulary, configuration["n_units"], configuration["chain_len"])
    if tuple(payload["shape"]) != expected:
        raise ValueError("frozen template shape differs from configuration/vocabulary")
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != int(np.prod(expected)) * 4:
        raise ValueError("frozen template byte count differs from shape")
    w = np.frombuffer(raw, dtype="<f4").reshape(expected).copy()
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("frozen templates must be finite and nonnegative")
    model = SeqNMF(expected[1], expected[2], penalty=configuration["penalty"])
    model.w = w
    return model


def score_frozen_sequence_batch(samples, artifact, configuration, *, vocabulary,
                                inference_iterations=50, memory_mb=256):
    """Score separated real/null timelines with one frozen inference call.

    Values are exact bout coverage or seqNMF uncentered normalized root error.
    Explicit padding prevents either reconstruction or competition from joining
    segments or samples. Batching reuses inference dispatch; throughput depends
    on matrix shape and thread settings and must be benchmarked. A conservative
    intermediate-array budget is checked before encoding.
    """
    if not samples or min(inference_iterations, memory_mb) < 1:
        raise ValueError("nonempty samples and positive execution budgets required")
    for segments in samples.values():
        if not segments or not sum(map(len, segments)):
            raise ValueError("every sample requires nonempty token support")
        for tokens in segments:
            z = np.asarray(tokens)
            if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer) or np.any(z < 0) or np.any(z >= vocabulary):
                raise ValueError("sample vocabulary differs from frozen model")
    if configuration["method"] == "exact":
        return {key: float(motif_masks(segments, artifact["motifs"]).any(axis=0).mean()) for key, segments in samples.items()}
    model = frozen_seqnmf_model(artifact, configuration, vocabulary=vocabulary)
    columns = sum(len(z) + model.length for segments in samples.values() for z in segments)
    if columns * (vocabulary + model.factors) * 4 * 12 > memory_mb * 1024 ** 2:
        raise MemoryError("frozen score batch exceeds intermediate-array budget")
    x, valid, slices, _ = _encode(samples, vocabulary, model.length)
    residual = x - reconstruct(model.w, model.infer(x, valid_starts=valid, iterations=inference_iterations))
    return {key: float(np.linalg.norm(residual[:, span]) / np.linalg.norm(x[:, span])) for key, span in slices.items()}


def project_frozen_catalogue(bouts, artifact, configuration, *, vocabulary, fps,
                             thresholds=None, inference_iterations=35, max_error=.95):
    """Project one frozen catalogue onto native bouts; never learn templates.

    Return raw candidate calls and a catalogue, both pandas tables. Selection is
    delegated to the existing native-interval organisation consumer. Thresholds
    for seqNMF must be supplied from explicitly chosen calibration observations;
    no threshold is estimated on these observations. Missing/deleted bouts are
    barriers. All original native bounds are retained and calls are half open.
    """
    import pandas as pd

    from fpcode.workflows.sequence_atlas import one_hot_segments

    if not np.isfinite(fps) or fps <= 0 or inference_iterations < 1:
        raise ValueError("positive FPS and inference iterations required")
    b = filter_short_bouts(bouts, configuration["min_bout_ms"] / 1000.)
    tokens = b.syllable_id.to_numpy()
    if not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0) or np.any(tokens >= vocabulary):
        raise ValueError("bout vocabulary differs from frozen catalogue")
    method = configuration["method"]
    candidates, catalog = [], []
    if method == "exact":
        for k, motif in enumerate(artifact["motifs"]):
            if not motif or min(motif) < 0 or max(motif) >= vocabulary:
                raise ValueError("motif differs from frozen vocabulary")
            chain = f"exact_{k:02d}"
            catalog.append({"chain_id": chain, "tokens": " ".join(map(str, motif)), "length_bouts": len(motif)})
            for _, group in b.groupby("segment", sort=False):
                for start in motif_hits(group.syllable_id.to_numpy(), motif):
                    a = int(group.index[start])
                    candidates.append((chain, a, a + len(motif), float(len(motif)), np.nan, np.nan))
    elif method == "seqnmf":
        model = frozen_seqnmf_model(artifact, configuration, vocabulary=vocabulary)
        thresholds = np.asarray(thresholds, dtype=float)
        if thresholds.shape != (model.factors,) or not np.isfinite(thresholds).all() or np.any(thresholds < 0):
            raise ValueError("explicit finite nonnegative training-calibrated thresholds required")
        for k in range(model.factors):
            catalog.append({"chain_id": f"seqnmf_{k:02d}", "tokens": " ".join(map(str, model.w[:, k].argmax(axis=0))),
                            "length_bouts": model.length})
        if len(b) >= model.length:
            x, valid, original = one_hot_segments(b, vocabulary, model.length)
            if valid.any():
                h = model.infer(x, valid_starts=valid, iterations=inference_iterations)
                for call in activation_events(h, model.w, x, thresholds=thresholds, valid_starts=valid, max_error=max_error):
                    a, end = int(original[call["start"]]), int(original[call["end"] - 1]) + 1
                    candidates.append((f"seqnmf_{call['factor']:02d}", a, end, 1. - call["template_error"],
                                       call["template_error"], call["activation"]))
    else:
        raise ValueError("unsupported frozen catalogue method")
    rows = []
    for chain, start, end, score, error, activation in candidates:
        chunk = b.iloc[start:end]
        if len(chunk) != end - start or chunk.segment.nunique() != 1:
            raise ValueError("candidate crossed a native segment barrier")
        duration = chunk.duration_s.to_numpy()
        rows.append({"chain_id": chain, "method": method, "start_bout": start, "end_bout": end,
                     "segment": int(chunk.segment.iloc[0]), "start_frame": int(chunk.start_frame.iloc[0]),
                     "end_frame_exclusive": int(chunk.end_frame_exclusive.iloc[-1]), "selection_score": score,
                     "duration_s": float(duration.sum()), "phase_duration_cv": float(duration.std() / duration.mean()),
                     "template_error": error, "activation": activation})
    columns = ["chain_id", "method", "start_bout", "end_bout", "segment", "start_frame", "end_frame_exclusive",
               "selection_score", "duration_s", "phase_duration_cv", "template_error", "activation"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(catalog)


def activation_dead_fraction(mass):
    """Fraction below 5% of the positive median activation, including zero mass."""
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or not len(mass) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("activation mass must be a finite nonnegative factor vector")
    positive = mass[mass > 1e-10]
    if not len(positive):
        return 1.0
    threshold = max(1e-10, .05 * float(np.median(positive)))
    return float(np.mean(mass < threshold))


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:4], "little")


def _sample_animals(data, bouts, animals, vocabulary, length, per_animal, fold, replicate):
    """One bounded native-valid chunk per eligible session, budgeted per animal."""
    sampled, evidence = {}, {}
    identity = dict(zip(data.sessions.session_id, data.sessions.animal_id.astype(str)))
    for animal in animals:
        animal = str(animal)
        recordings = sorted(data.sessions_of[animal])
        if any(identity.get(sid) != animal for sid in recordings):
            raise ValueError("session-to-animal ownership mismatch")
        eligible = []
        raw_bouts, available_seconds, available_frames = 0, 0., 0
        for sid in recordings:
            frame = filter_short_bouts(bouts[sid], 0.)
            tokens = frame.syllable_id.to_numpy()
            if (not np.issubdtype(tokens.dtype, np.integer)
                    or np.any(tokens < 0) or np.any(tokens >= vocabulary)):
                raise ValueError("bout labels differ from declared vocabulary")
            raw_bouts += len(frame)
            available_seconds += float(frame.duration_s.sum())
            available_frames += int((frame.end_frame_exclusive - frame.start_frame).sum())
            groups = [sub for _, sub in frame.groupby("segment", sort=False) if len(sub) >= length]
            if groups:
                eligible.append((sid, groups))
        if not eligible:
            raise ValueError(f"animal {animal} has no supported {length}-bout windows")
        if per_animal < length * len(eligible):
            raise ValueError("per-animal budget is too small to represent eligible sessions")
        chunks, records = [], []
        allocation, remainder = divmod(per_animal, len(eligible))
        for index, (sid, groups) in enumerate(eligible):
            rng = np.random.default_rng(_seed(data.base_seed, "sample", fold, replicate, sid))
            weights = np.asarray([len(group) for group in groups], dtype=float)
            group = groups[int(rng.choice(len(groups), p=weights / weights.sum()))]
            size = min(allocation + (index < remainder), len(group))
            start = int(rng.integers(len(group) - size + 1))
            chunk = group.iloc[start:start + size]
            chunks.append(chunk.syllable_id.to_numpy(dtype=int))
            records.append({"session_id": sid, "start_frame": int(chunk.start_frame.iloc[0]),
                            "end_frame_exclusive": int(chunk.end_frame_exclusive.iloc[-1]),
                            "sampled_bouts": int(size), "sampled_seconds": float(chunk.duration_s.sum()),
                            "sampled_native_frames": int((chunk.end_frame_exclusive - chunk.start_frame).sum())})
        sampled[animal] = chunks
        evidence[animal] = {"available_filtered_bouts": raw_bouts,
                            "available_filtered_seconds": available_seconds,
                            "available_native_frames": available_frames,
                            "sampled_bouts": sum(map(len, chunks)),
                            "sampled_seconds": sum(r["sampled_seconds"] for r in records),
                            "sampled_native_frames": sum(r["sampled_native_frames"] for r in records),
                            "eligible_sessions": len(eligible), "chunks": records}
    return sampled, evidence


def _encode(sampled, vocabulary, length):
    matrices, valid, slices, blocks = [], [], {}, []
    offset = 0
    for animal, segments in sampled.items():
        begin = offset
        for tokens in segments:
            x = np.zeros((vocabulary, len(tokens) + length), dtype=np.float32)
            x[tokens, np.arange(len(tokens))] = 1.
            onset = np.zeros(x.shape[1], dtype=bool)
            onset[:len(tokens) - length + 1] = True
            matrices.append(x)
            valid.append(onset)
            blocks.append(np.arange(offset, offset + len(tokens)))
            offset += x.shape[1]
        slices[animal] = slice(begin, offset)
    return np.concatenate(matrices, axis=1), np.concatenate(valid), slices, blocks


def _exact_fold(train_sample, held_sample, cfg, rng, null_repeats):
    pooled, prevalence = Counter(), Counter()
    for segments in train_sample.values():
        counts = Counter()
        for tokens in segments:
            for length in range(3, cfg["chain_len"] + 1):
                counts.update(tuple(tokens[i:i + length]) for i in range(len(tokens) - length + 1))
        total = sum(counts.values())
        for motif, count in counts.items():
            pooled[motif] += count / max(total, 1)
            prevalence[motif] += 1
    floor = cfg["min_prevalence"]
    if floor > len(train_sample):
        raise ValueError("minimum animal prevalence exceeds training-fold size")
    motifs = [motif for motif, _ in pooled.most_common() if prevalence[motif] >= floor][:cfg["n_units"]]
    if not motifs:
        raise ValueError("no exact motif met training animal prevalence")
    masks, rows = [], []
    for animal, segments in held_sample.items():
        real = motif_masks(segments, motifs)
        coverage = float(real.any(axis=0).mean())
        null = [float(motif_masks([rng.permutation(tokens) for tokens in segments], motifs).any(axis=0).mean())
                for _ in range(null_repeats)]
        null_mean = float(np.mean(null))
        rows.append({"animal_id": animal, "explained": coverage, "real_coverage": coverage,
                     "null_coverage": null, "null_mean": null_mean, "null_sd": float(np.std(null)),
                     "order_gain": (coverage - null_mean) / coverage if coverage > 0 else 0.})
        masks.append(real)
    real = np.concatenate(masks, axis=1)
    hits = real.sum(axis=1)
    inter = real.astype(np.float32) @ real.astype(np.float32).T
    union = hits[:, None] + hits[None, :] - inter
    similarity = inter / np.maximum(union, 1)
    np.fill_diagonal(similarity, 0.)
    return {"rows": rows, "dead_fraction": float(np.mean(hits == 0)),
            "redundancy": float(similarity.max()) if len(motifs) > 1 else 0.,
            "motifs": motifs, "discovered_units": len(motifs)}


def _seqnmf_fold(train_sample, held_sample, cfg, vocabulary, seeds, iterations, null_repeats):
    length, units = cfg["chain_len"], cfg["n_units"]
    x, valid, _, _ = _encode(train_sample, vocabulary, length)
    xt, vt, slices, blocks = _encode(held_sample, vocabulary, length)
    model = SeqNMF(units, length, penalty=cfg["penalty"], iterations=iterations, seed=seeds["model"]).fit(x, valid_starts=valid)
    alternate = SeqNMF(units, length, penalty=cfg["penalty"], iterations=iterations,
                       seed=seeds["alternate"]).fit(x, valid_starts=valid)
    h = model.infer(xt, valid_starts=vt)
    residual = xt - reconstruct(model.w, h)
    errors = {animal: float(np.linalg.norm(residual[:, span]) / np.linalg.norm(xt[:, span]))
              for animal, span in slices.items()}
    null_errors = {animal: [] for animal in slices}
    rng = np.random.default_rng(seeds["null"])
    for _ in range(null_repeats):
        null = xt.copy()
        for block in blocks:
            null[:, block] = xt[:, rng.permutation(block)]
        residual_null = null - reconstruct(model.w, model.infer(null, valid_starts=vt))
        for animal, span in slices.items():
            null_errors[animal].append(float(np.linalg.norm(residual_null[:, span]) / np.linalg.norm(null[:, span])))
    rows = []
    for animal, err in errors.items():
        values = null_errors[animal]
        null_mean = float(np.mean(values))
        rows.append({"animal_id": animal, "explained": 1. - err ** 2, "real_nrmse": err,
                     "null_nrmse": values, "null_mean": null_mean, "null_sd": float(np.std(values)),
                     "order_gain": (null_mean - err) / null_mean if null_mean > 0 else 0.})
    flat = model.w.transpose(1, 0, 2).reshape(units, -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
    similarity = flat @ flat.T
    np.fill_diagonal(similarity, 0.)
    return {"rows": rows, "dead_fraction": activation_dead_fraction(h.sum(axis=1)),
            "redundancy": float(similarity.max()) if units > 1 else 0., "w": model.w,
            "same_data_seed_stability": template_similarity(model.w, alternate.w), "discovered_units": units}


def _validate_configuration(data, cfg, replicate, iterations, per_animal, null_repeats):
    common = {"level", "min_bout_ms", "method", "chain_len", "n_units"}
    knob = {"exact": "min_prevalence", "seqnmf": "penalty"}.get(cfg.get("method"))
    if knob is None or set(cfg) != common | {knob}:
        raise ValueError("configuration must contain only discovery choices and its active method knob")
    for name, value, floor in (("replicate", replicate, 0), ("iterations", iterations, 1),
                               ("per_animal", per_animal, 1), ("null_repeats", null_repeats, 1),
                               ("chain_len", cfg["chain_len"], 3), ("n_units", cfg["n_units"], 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < floor:
            raise ValueError(f"{name} must be an integer >= {floor}")
    if cfg["level"] not in data.vocab or not np.isfinite(cfg["min_bout_ms"]) or cfg["min_bout_ms"] < 0:
        raise ValueError("resolution or minimum bout duration is invalid")
    if knob == "min_prevalence":
        if isinstance(cfg[knob], bool) or not isinstance(cfg[knob], (int, np.integer)) or cfg[knob] < 1:
            raise ValueError("minimum prevalence must be a positive animal count")
    elif not np.isfinite(cfg[knob]) or cfg[knob] < 0:
        raise ValueError("seqNMF penalty must be finite and nonnegative")
    pool, test = set(map(str, data.pool)), set(map(str, data.test))
    if len(pool) != len(data.pool) or pool & test:
        raise ValueError("search and reserved animal membership must be unique and disjoint")
    held_sets = []
    for fold in range(len(data.folds)):
        train, held = data.fold_split(fold)
        train, held = set(map(str, train)), set(map(str, held))
        if not train or not held or train & held or train | held != pool or (train | held) & test:
            raise ValueError("every fold must separate all search animals and exclude reserved animals")
        held_sets.append(held)
    if len(held_sets) < 2 or set.union(*held_sets) != pool or sum(map(len, held_sets)) != len(pool):
        raise ValueError("held-out folds must partition the search animals exactly once")


def evaluate_configuration(data, cfg, replicate, *, iterations=80, per_animal=700, null_repeats=5):
    """Evaluate routines using animal-grouped CV and equal-weight animal readouts.

    ``data`` supplies sessions/session ownership, pool/test lists, fixed folds,
    ``fold_split(fold)``, ``vocab``, ``base_seed`` and a gap-preserving
    ``bouts(level,min_ms)`` view. Only search-animal recordings are accessed here.
    Exact ``chain_len`` is the maximum motif length (minimum three), whereas
    seqNMF uses that fixed length. Exact prevalence is a training-animal count.

    Returns JSON-safe metrics, fold/null/native-support evidence and compressed
    float32 finalist templates or exact catalogues. ``explained`` means bout
    coverage for exact and ``1-SSE/sum(x**2)`` for seqNMF, never ordinary R².
    Fitness uses disclosed proxy weights, valid only within (level,method).
    """
    _validate_configuration(data, cfg, replicate, iterations, per_animal, null_repeats)
    vocabulary = int(data.vocab[cfg["level"]])
    bouts = data.bouts(cfg["level"], cfg["min_bout_ms"])
    folds, support, artifacts = [], [], []
    for fold in range(len(data.folds)):
        train, held = data.fold_split(fold)
        training, tr_evidence = _sample_animals(data, bouts, train, vocabulary, cfg["chain_len"], per_animal, fold, replicate)
        testing, te_evidence = _sample_animals(data, bouts, held, vocabulary, cfg["chain_len"], per_animal, fold, replicate)
        seeds = {name: _seed(data.base_seed, cfg["level"], cfg["min_bout_ms"], fold, replicate, name)
                 for name in ("model", "alternate", "null")}
        # Common null seeds reduce Monte Carlo variation between otherwise comparable configurations.
        seeds["null"] = _seed(data.base_seed, "null", cfg["level"], cfg["min_bout_ms"], fold, replicate)
        if cfg["method"] == "exact":
            result = _exact_fold(training, testing, cfg, np.random.default_rng(seeds["null"]), null_repeats)
            artifact = {"motifs": [list(map(int, motif)) for motif in result["motifs"]]}
        else:
            result = _seqnmf_fold(training, testing, cfg, vocabulary, seeds, iterations, null_repeats)
            w = np.ascontiguousarray(result["w"], dtype="<f4")
            artifact = {"templates": {"encoding": "base64", "dtype": "<f4", "shape": list(w.shape),
                                      "data": base64.b64encode(w.tobytes()).decode("ascii")}}
        folds.append(result)
        support.append({"fold": fold, "train": tr_evidence, "heldout": te_evidence, "seeds": seeds})
        artifacts.append({"fold": fold, "training_animals": list(map(str, train)),
                          "heldout_animals": list(map(str, held)), **artifact})
    pairs = [(i, j) for i in range(len(folds)) for j in range(i + 1, len(folds))]
    if cfg["method"] == "exact":
        sets = [set(result["motifs"]) for result in folds]
        crossfold = float(np.mean([len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1) for i, j in pairs]))
        stability, seed_stability = crossfold, None
    else:
        crossfold = float(np.mean([template_similarity(folds[i]["w"], folds[j]["w"]) for i, j in pairs]))
        seed_stability = float(np.mean([result["same_data_seed_stability"] for result in folds]))
        stability = (crossfold + seed_stability) / 2.
    rows = [row for result in folds for row in result["rows"]]
    gain = float(np.mean([row["order_gain"] for row in rows]))
    explained = float(np.mean([row["explained"] for row in rows]))
    dead = float(np.mean([result["dead_fraction"] for result in folds]))
    redundancy = float(np.mean([result["redundancy"] for result in folds]))
    fitness = float(.4 * max(0., gain) + .25 * np.clip(stability, 0, 1)
                    + .2 * np.clip(explained, 0, 1) + .15 * (1. - dead)
                    - .5 * max(redundancy - .7, 0.))
    fold_gains = [float(np.mean([row["order_gain"] for row in result["rows"]])) for result in folds]
    metrics = {"order_gain": gain, "order_gain_sd": float(np.std(fold_gains)), "stability": stability,
               "same_data_seed_stability": seed_stability, "crossfold_stability": crossfold,
               "explained": explained, "dead_fraction": dead, "redundancy": redundancy,
               "fitness": fitness, "null_sd_mean": float(np.mean([row["null_sd"] for row in rows])),
               "sampled_heldout_bouts": sum(e["sampled_bouts"] for f in support for e in f["heldout"].values()),
               "heldout_animals": len(rows), "discovered_units_mean": float(np.mean([r["discovered_units"] for r in folds])),
               "details": {"metric": "bout_coverage" if cfg["method"] == "exact" else "uncentered_reconstruction_score",
                           "ranking_scope": [cfg["level"], cfg["method"]], "null_kind": "within_segment_shuffle",
                           "null_repeats": null_repeats, "order_gain_by_fold": fold_gains,
                           "explained_by_fold": [float(np.mean([r["explained"] for r in f["rows"]])) for f in folds],
                           "animal_readouts_by_fold": [f["rows"] for f in folds], "support_by_fold": support,
                           "fitness_weights": {"order_gain": .4, "stability": .25, "explained": .2,
                                               "alive": .15, "redundancy_penalty": .5, "redundancy_free_limit": .7}},
               "artifacts": {"configuration": cfg, "replicate": replicate, "folds": artifacts}}
    json.dumps(metrics, allow_nan=False)
    return metrics
