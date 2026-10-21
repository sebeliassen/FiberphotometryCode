"""Shared bout-motif benchmarking and inter/intra-chain table construction."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.episodes.chains import exact_occurrences, group_occurrences, ngram_counts, prefix_progression
from fpcode.behavior.episodes.seqnmf import SeqNMF, activation_events, reconstruct, template_similarity


def validate_animal_splits(sessions, splits):
    """Validate a complete frozen animal split; repeated sessions inherit its role.

    This function selects no animals, fits no model and imposes no cohort policy.
    IDs are compared as strings, matching exported study manifests.
    """
    required = {'session_id', 'animal_id', 'group'}
    if not required <= set(sessions) or sessions[list(required)].isna().any().any():
        raise ValueError("sessions require nonmissing recording, animal and group identities")
    if sessions.session_id.duplicated().any():
        raise ValueError("recording identities must be unique")
    animals = sessions.animal_id.astype(str)
    if sessions.assign(animal_id=animals).groupby('animal_id').group.nunique().max() > 1:
        raise ValueError("animal group changes between recordings")
    roles = {}
    for name in ('train', 'validation', 'test'):
        if name not in splits or isinstance(splits[name], str):
            raise ValueError("split requires explicit train, validation and test animal lists")
        for animal in splits[name]:
            animal = str(animal)
            if animal in roles:
                raise ValueError("animal split overlap or duplicate membership")
            roles[animal] = name
    if set(roles) != set(animals):
        raise ValueError("animal split must exactly cover the selected cohort")
    return roles
def one_hot_segments(bouts, vocabulary, length):
    """Encode separated bouts with enough zero padding to prohibit false joins."""
    tokens, original, valid = [], [], []
    for _, sub in bouts.groupby("segment", sort=False):
        z = sub.syllable_id.to_numpy(int)
        tokens.extend(z)
        original.extend(sub.index)
        valid.extend([True] * max(0, len(z) - length + 1) + [False] * min(len(z), length - 1))
        tokens.extend([-1] * length)
        original.extend([-1] * length)
        valid.extend([False] * length)
    x = np.zeros((vocabulary, len(tokens)), dtype=np.float32)
    good = np.flatnonzero(np.asarray(tokens) >= 0)
    x[np.asarray(tokens)[good], good] = 1.
    return x, np.asarray(valid, bool), np.asarray(original, int)


def _sample_matrix(sessions, bouts, animals, vocabulary, length, rng, per_animal=700):
    matrices, masks = [], []
    for animal in animals:
        selected = sessions[sessions.animal_id == animal]
        per_session = max(60, per_animal // len(selected))
        for meta in selected.itertuples():
            sub = bouts[meta.session_id]
            candidates = [(seg, group) for seg, group in sub.groupby("segment") if len(group) >= 40]
            if not candidates:
                continue
            weights = np.array([len(group) for _, group in candidates], float)
            _, group = candidates[rng.choice(len(candidates), p=weights / weights.sum())]
            size = min(per_session, len(group))
            start = int(rng.integers(max(1, len(group) - size + 1)))
            chunk = group.iloc[start:start + size]
            x, valid, _ = one_hot_segments(chunk, vocabulary, length)
            matrices.append(x)
            masks.append(valid)
    return np.concatenate(matrices, axis=1), np.concatenate(masks)


def _exclusive_events(events):
    """Greedy score-ranked nonoverlap stream; original overlapping calls retained."""
    selected = []
    occupied = defaultdict(list)
    for idx in events.sort_values("selection_score", ascending=False).index:
        row = events.loc[idx]
        intervals = occupied[row.segment]
        if any(row.start_bout < end and row.end_bout > start for start, end in intervals):
            continue
        selected.append(idx)
        intervals.append((row.start_bout, row.end_bout))
    return events.loc[selected].sort_values("start_s").reset_index(drop=True)


def run_sequence_atlas(sessions, measurement_dir, output_dir, *, vocabulary, groups,
                       seed=101, motif_limit=30, iterations=80):
    """Discover on training animals, tune on validation animals, audit on test animals.

    Templates remain frozen for every cohort comparison. SeqNMF is reconstructed
    on held-out data, not trained there; reconstruction is not future prediction.
    """
    output_dir, measurement_dir = Path(output_dir), Path(measurement_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for group in groups:
        animals = sessions.loc[sessions.group == group, "animal_id"].unique().copy()
        rng.shuffle(animals)
        test.extend(animals[:2 if len(animals) < 10 else 3])
        validation.extend(animals[2 if len(animals) < 10 else 3:4 if len(animals) < 10 else 5])
        train.extend(animals[4 if len(animals) < 10 else 5:])
    (output_dir / "animal_splits.json").write_text(json.dumps({"train": train, "validation": validation, "test": test,
                                                              "seed": seed}, indent=2))
    bouts = {m.session_id: pd.read_csv(measurement_dir / "bouts_by_session" / (m.session_id + ".csv"))
             for m in sessions.itertuples()}
    mouse_counts = defaultdict(Counter)
    for meta in sessions.itertuples():
        if meta.animal_id in train:
            mouse_counts[meta.animal_id].update(ngram_counts(bouts[meta.session_id], (3, 4, 5, 6)))
    pooled, prevalence = Counter(), Counter()
    for animal, count in mouse_counts.items():
        pooled.update({key: value / max(sum(count.values()), 1) for key, value in count.items()})
        prevalence.update(count.keys())
    motifs = [key for key, _ in pooled.most_common() if prevalence[key] >= min(5, len(train))][:motif_limit]
    catalog = [{"chain_id": f"exact_{i:02d}", "method": "exact", "tokens": " ".join(map(str, motif)),
                "length_bouts": len(motif), "training_mouse_prevalence": prevalence[motif]}
               for i, motif in enumerate(motifs)]
    benchmarks, models = [], []
    settings = [(6, 3, .01), (10, 3, .01), (6, 6, .01), (10, 6, .01), (10, 3, .05), (10, 6, .05)]
    for factors, length, penalty in settings:
        x, valid = _sample_matrix(sessions, bouts, train, vocabulary, length, np.random.default_rng(seed))
        xv, vv = _sample_matrix(sessions, bouts, validation, vocabulary, length, np.random.default_rng(seed + 1))
        model = SeqNMF(factors, length, penalty=penalty, iterations=iterations, seed=seed).fit(x, valid_starts=valid)
        h = model.infer(xv, valid_starts=vv)
        error = np.linalg.norm(xv - reconstruct(model.w, h)) / np.linalg.norm(xv)
        benchmarks.append({"factors": factors, "length_bouts": length, "penalty": penalty,
                           "training_error": model.loss_history[-1], "validation_error": error,
                           "train_columns": x.shape[1], "validation_columns": xv.shape[1]})
        models.append(model)
        print(f"seqNMF benchmark K={factors} L={length} penalty={penalty}: validation error {error:.4f}", flush=True)
    best = int(np.argmin([b["validation_error"] for b in benchmarks]))
    model = models[best]
    x, valid = _sample_matrix(sessions, bouts, train, vocabulary, model.length, np.random.default_rng(seed))
    stability = []
    for other_seed in (seed + 1, seed + 2):
        alternate = SeqNMF(model.factors, model.length, penalty=model.penalty, iterations=iterations,
                           seed=other_seed).fit(x, valid_starts=valid)
        stability.append(template_similarity(model.w, alternate.w))
    xt, vt = _sample_matrix(sessions, bouts, test, vocabulary, model.length, np.random.default_rng(seed + 2))
    ht = model.infer(xt, valid_starts=vt)
    real_error = float(np.linalg.norm(xt - reconstruct(model.w, ht)) / np.linalg.norm(xt))
    null_rows = []
    support = np.flatnonzero(xt.sum(axis=0) > 0)
    boundaries = np.flatnonzero(np.r_[True, np.diff(support) != 1, True])
    blocks = [support[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
    for shuffle in range(10):
        shuffled = xt.copy()
        for block in blocks:
            shuffled[:, block] = shuffled[:, rng.permutation(block)]
        hnull = model.infer(shuffled, valid_starts=vt)
        error = float(np.linalg.norm(shuffled - reconstruct(model.w, hnull)) / np.linalg.norm(shuffled))
        null_rows.append({"shuffle": shuffle, "heldout_real_error": real_error, "shuffled_order_error": error,
                          "order_gain": error - real_error})
    pd.DataFrame(null_rows).to_csv(output_dir / "seqnmf_heldout_order_null.csv", index=False)
    benchmarks[best].update(selected=True, heldout_test_error=real_error, stability_mean=float(np.mean(stability)))
    pd.DataFrame(benchmarks).to_csv(output_dir / "seqnmf_benchmarks.csv", index=False)
    thresholds = np.quantile(model.h[:, valid], .90, axis=1)
    np.savez_compressed(output_dir / "seqnmf_templates.npz", w=model.w, thresholds=thresholds,
                        length=model.length, factors=model.factors, penalty=model.penalty,
                        loss=np.array(model.loss_history))
    for k in range(model.factors):
        tokens = np.argmax(model.w[:, k, :], axis=0)
        distribution = model.w[:, k, :] / np.maximum(model.w[:, k, :].sum(axis=0), 1e-9)
        temporal_specificity = np.mean(np.sum(distribution * np.log(np.maximum(distribution, 1e-9)
                                                    / np.maximum(distribution.mean(axis=1, keepdims=True), 1e-9)), axis=0))
        catalog.append({"chain_id": f"seqnmf_{k:02d}", "method": "seqnmf", "tokens": " ".join(map(str, tokens)),
                        "length_bouts": model.length, "distinct_dominant_syllables": len(np.unique(tokens)),
                        "temporal_specificity_nats": temporal_specificity})
    pd.DataFrame(catalog).to_csv(output_dir / "chain_catalog.csv", index=False)
    all_events, metrics, progression, grouping, transition_rows = [], [], [], [], []
    for session_index, meta in enumerate(sessions.itertuples()):
        b = bouts[meta.session_id]
        minutes = b.duration_s.sum() / 60
        identity = {k: getattr(meta, k) for k in ("session_id", "animal_id", "group", "condition")}
        events = []
        for i, motif in enumerate(motifs):
            chain_id = f"exact_{i:02d}"
            for a, end in exact_occurrences(b, motif):
                events.append(_event(b, a, end, chain_id, "exact", meta.fps, selection_score=len(motif)))
            p = prefix_progression(b, motif)
            if len(p):
                p["chain_id"] = chain_id
                for key, value in identity.items():
                    p[key] = value
                progression.append(p)
        xs, vs, original = one_hot_segments(b, vocabulary, model.length)
        hs = model.infer(xs, valid_starts=vs, iterations=35)
        for event in activation_events(hs, model.w, xs, thresholds=thresholds, valid_starts=vs):
            start, end = original[event["start"]], original[event["end"] - 1] + 1
            if start < 0 or end <= start:
                raise ValueError("episode crosses a segment boundary")
            row = _event(b, start, end, f"seqnmf_{event['factor']:02d}", "seqnmf", meta.fps,
                         selection_score=1 - event["template_error"])
            row.update(template_error=event["template_error"], activation=event["activation"])
            events.append(row)
        table = pd.DataFrame(events)
        if table.empty:
            continue
        for method, sub in table.groupby("method"):
            stream = _exclusive_events(sub)
            for gap in (0., .5, 1., 2., 5.):
                grouped = group_occurrences(stream, gap_seconds=gap)
                for chain, calls in grouped.groupby("chain_id"):
                    grouping.append({**identity, "method": method, "chain_id": chain, "gap_threshold_s": gap,
                                     "exclusive_occurrences": len(calls), "repeat_fraction": float((calls.repeat_position > 1).mean()),
                                     "max_run_length": calls.run_length.max(), "mean_run_length": calls.run_length.mean(),
                                     "initiations_per_min": float((calls.repeat_position == 1).sum() / minutes),
                                     "first_mean_speed_px_s": calls.loc[calls.repeat_position == 1, "mean_speed_px_s"].mean(),
                                     "repeat_mean_speed_px_s": calls.loc[calls.repeat_position > 1, "mean_speed_px_s"].mean()})
            for a, following in zip(stream.itertuples(), list(stream.itertuples())[1:]):
                if a.segment == following.segment:
                    transition_rows.append({**identity, "method": method, "from_chain": a.chain_id,
                                            "to_chain": following.chain_id, "gap_s": following.start_s - a.end_s})
        for row in catalog:
            sub = table[table.chain_id == row["chain_id"]]
            intervals = sorted(zip(sub.start_bout, sub.end_bout))
            union = []
            for a, end in intervals:
                if not union or a > union[-1][1]:
                    union.append([a, end])
                else:
                    union[-1][1] = max(union[-1][1], end)
            coverage = sum(b.iloc[a:end].duration_s.sum() for a, end in union) / (minutes * 60)
            metrics.append({**identity, "chain_id": row["chain_id"], "method": row["method"],
                            "occurrences": len(sub), "occurrences_per_min": len(sub) / minutes,
                            "coverage_fraction": coverage, "duration_s": sub.duration_s.mean(),
                            "duration_cv": sub.duration_s.std() / sub.duration_s.mean() if len(sub) > 1 else np.nan,
                            "mean_speed_px_s": sub.mean_speed_px_s.mean(), "speed_cv": sub.mean_speed_px_s.std()
                            / sub.mean_speed_px_s.mean() if len(sub) > 1 else np.nan,
                            "speed_change_px_s": sub.speed_change_px_s.mean(),
                            "mean_phase_duration_cv": sub.phase_duration_cv.mean()})
        for key, value in identity.items():
            table[key] = value
        all_events.append(table)
        print(f"Chain measurements {session_index + 1}/{len(sessions)}", flush=True)
    for name, tables in (("chain_occurrences", all_events), ("prefix_progression", progression)):
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(output_dir / (name + ".csv"), index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "chain_session_metrics.csv", index=False)
    pd.DataFrame(grouping).to_csv(output_dir / "chain_grouping_sensitivity.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output_dir / "chain_transitions.csv", index=False)


def _event(bouts, start, end, chain_id, method, fps, *, selection_score):
    chunk = bouts.iloc[start:end]
    return {"chain_id": chain_id, "method": method, "segment": int(chunk.segment.iloc[0]),
            "start_bout": start, "end_bout": end, "start_frame": int(chunk.start_frame.iloc[0]),
            "end_frame_exclusive": int(chunk.end_frame_exclusive.iloc[-1]),
            "start_s": chunk.start_frame.iloc[0] / fps, "end_s": chunk.end_frame_exclusive.iloc[-1] / fps,
            "duration_s": chunk.duration_s.sum(), "selection_score": selection_score,
            "mean_speed_px_s": np.average(chunk.mean_speed_px_s, weights=chunk.duration_s),
            "speed_change_px_s": chunk.exit_speed_px_s.iloc[-1] - chunk.entry_speed_px_s.iloc[0],
            "phase_duration_cv": chunk.duration_s.std() / chunk.duration_s.mean()}


def refresh_chain_kinematics(sessions, measurement_dir, chains_dir):
    """Update movement measurements on frozen calls, without redetecting episodes."""
    measurement_dir, chains_dir = Path(measurement_dir), Path(chains_dir)
    events = pd.read_csv(chains_dir / "chain_occurrences.csv")
    metrics = pd.read_csv(chains_dir / "chain_session_metrics.csv")
    grouping = pd.read_csv(chains_dir / "chain_grouping_sensitivity.csv")
    for meta in sessions.itertuples():
        bouts = pd.read_csv(measurement_dir / "bouts_by_session" / (meta.session_id + ".csv"))
        indices = events.index[events.session_id == meta.session_id]
        for idx in indices:
            row = events.loc[idx]
            chunk = bouts.iloc[int(row.start_bout):int(row.end_bout)]
            events.loc[idx, "mean_speed_px_s"] = np.average(chunk.mean_speed_px_s, weights=chunk.duration_s)
            events.loc[idx, "speed_change_px_s"] = chunk.exit_speed_px_s.iloc[-1] - chunk.entry_speed_px_s.iloc[0]
        local = events.loc[indices]
        for chain, calls in local.groupby("chain_id"):
            target = (metrics.session_id == meta.session_id) & (metrics.chain_id == chain)
            metrics.loc[target, "mean_speed_px_s"] = calls.mean_speed_px_s.mean()
            metrics.loc[target, "speed_change_px_s"] = calls.speed_change_px_s.mean()
            metrics.loc[target, "speed_cv"] = (calls.mean_speed_px_s.std() / calls.mean_speed_px_s.mean()
                                               if len(calls) > 1 and calls.mean_speed_px_s.mean() > 0 else np.nan)
        for _, calls in local.groupby("method"):
            stream = _exclusive_events(calls)
            for gap in grouping.gap_threshold_s.unique():
                grouped = group_occurrences(stream, gap_seconds=gap)
                for chain, calls in grouped.groupby("chain_id"):
                    target = ((grouping.session_id == meta.session_id) & (grouping.chain_id == chain)
                              & (grouping.gap_threshold_s == gap))
                    grouping.loc[target, "first_mean_speed_px_s"] = calls.loc[calls.repeat_position == 1, "mean_speed_px_s"].mean()
                    grouping.loc[target, "repeat_mean_speed_px_s"] = calls.loc[calls.repeat_position > 1, "mean_speed_px_s"].mean()
    events.to_csv(chains_dir / "chain_occurrences.csv", index=False)
    metrics.to_csv(chains_dir / "chain_session_metrics.csv", index=False)
    grouping.to_csv(chains_dir / "chain_grouping_sensitivity.csv", index=False)


def audit_frozen_sequence_order(sessions, measurement_dir, chains_dir, *, vocabulary, seed=101, repeats=10):
    """Compare held-out reconstruction with block shuffle and first-order nulls.

    Nulls preserve each sampled block's support and length. Shuffles preserve its
    exact syllable counts. Markov simulations preserve fitted block transition
    probabilities in expectation, with original starting state; they do not
    preserve exact counts. Neither null is a future-prediction task.
    """
    chains_dir, measurement_dir = Path(chains_dir), Path(measurement_dir)
    splits = json.loads((chains_dir / "animal_splits.json").read_text())
    with np.load(chains_dir / "seqnmf_templates.npz") as saved:
        model = SeqNMF(int(saved["factors"]), int(saved["length"]), penalty=float(saved["penalty"]), seed=seed)
        model.w = saved["w"]
    bouts = {m.session_id: pd.read_csv(measurement_dir / "bouts_by_session" / (m.session_id + ".csv"))
             for m in sessions.itertuples()}
    x, valid = _sample_matrix(sessions, bouts, splits["test"], vocabulary, model.length,
                              np.random.default_rng(seed + 2))
    support = np.flatnonzero(x.sum(axis=0) > 0)
    boundaries = np.flatnonzero(np.r_[True, np.diff(support) != 1, True])
    blocks = [support[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
    h = model.infer(x, valid_starts=valid)
    real = float(np.linalg.norm(x - reconstruct(model.w, h)) / np.linalg.norm(x))
    rng, rows = np.random.default_rng(seed + 3), []
    for null_kind in ("within_block_shuffle", "first_order_markov"):
        for repeat in range(repeats):
            null = x.copy()
            for block in blocks:
                if null_kind == "within_block_shuffle":
                    null[:, block] = x[:, rng.permutation(block)]
                    continue
                tokens = np.argmax(x[:, block], axis=0)
                counts = np.zeros((vocabulary, vocabulary))
                np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
                marginal = np.bincount(tokens, minlength=vocabulary) / len(tokens)
                null[:, block] = 0
                state = int(tokens[0])
                for position in block:
                    null[state, position] = 1
                    distribution = counts[state] / counts[state].sum() if counts[state].sum() else marginal
                    state = int(rng.choice(vocabulary, p=distribution))
            hn = model.infer(null, valid_starts=valid)
            error = float(np.linalg.norm(null - reconstruct(model.w, hn)) / np.linalg.norm(null))
            rows.append({"null_kind": null_kind, "repeat": repeat, "heldout_real_error": real,
                         "null_error": error, "order_gain": error - real, "heldout_mice": len(splits["test"])})
    pd.DataFrame(rows).to_csv(chains_dir / "seqnmf_frozen_order_audit.csv", index=False)
