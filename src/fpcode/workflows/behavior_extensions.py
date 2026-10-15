"""Reusable execution-variant, history and chain-context analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.prediction import HistoryPredictor
from fpcode.behavior.analysis.variants import assign_execution_variants, fit_execution_variants
from fpcode.behavior.episodes.chains import segmented_tokens


def run_behavior_extensions(sessions, measurement_dir, chains_dir, *, vocabulary, seed=101):
    measurement_dir, chains_dir = Path(measurement_dir), Path(chains_dir)
    splits = json.loads((chains_dir / "animal_splits.json").read_text())
    train = splits["train"]
    bouts = []
    for meta in sessions.itertuples():
        table = pd.read_csv(measurement_dir / "bouts_by_session" / (meta.session_id + ".csv"))
        for name in ("session_id", "animal_id", "group", "condition"):
            table[name] = getattr(meta, name)
        bouts.append(table)
    bouts = pd.concat(bouts, ignore_index=True)
    features = ["duration_s", "mean_speed_px_s", "peak_phase", "mean_abs_yaw_rad_s", "latent_dispersion_units2"]
    usage, catalog, calls = [], [], []
    rng = np.random.default_rng(seed)
    for syllable, sub in bouts.groupby("syllable_id"):
        eligible = sub.dropna(subset=features)
        eligible = eligible[~eligible.left_censored & ~eligible.right_censored]
        selected = []
        for _, animal in eligible[eligible.animal_id.isin(train)].groupby("animal_id"):
            selected.append(animal.iloc[np.sort(rng.choice(len(animal), min(len(animal), 150), replace=False))])
        if len(selected) < 5 or sum(map(len, selected)) < 100:
            continue
        fit = pd.concat(selected)
        model = fit_execution_variants(fit[features].to_numpy(), seed=seed)
        eligible = eligible.copy()
        eligible["variant"] = assign_execution_variants(eligible[features].to_numpy(), model)
        for variant in (0, 1):
            catalog.append({"syllable_id": syllable, "variant": variant, "training_bouts": len(fit),
                            "training_animals": fit.animal_id.nunique(),
                            "seed_assignment_agreement": model["seed_assignment_agreement"],
                            "explained_dispersion": model["explained_dispersion"],
                            **dict(zip(features, model["raw_centers"][variant]))})
        counts = eligible.groupby(["session_id", "variant"]).size()
        for meta in sessions.itertuples():
            denominator = int((eligible.session_id == meta.session_id).sum())
            total_minutes = bouts.loc[bouts.session_id == meta.session_id, "duration_s"].sum() / 60
            for variant in (0, 1):
                n = counts.get((meta.session_id, variant), 0)
                usage.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                              "condition": meta.condition, "variant_id": f"{syllable}/v{variant}",
                              "fraction": n / denominator if denominator >= 10 else np.nan,
                              "rate_per_min": n / total_minutes, "eligible_bouts": denominator})
        calls.append(eligible[["session_id", "bout_index", "syllable_id", "variant"]])
    pd.DataFrame(catalog).to_csv(measurement_dir / "variant_catalog.csv", index=False)
    pd.DataFrame(usage).to_csv(measurement_dir / "variant_usage.csv", index=False)
    pd.concat(calls).to_csv(measurement_dir / "bout_variant_assignments.csv", index=False)
    predictor = HistoryPredictor(vocabulary, max_history=5)
    sequences = []
    for sid, sub in bouts[bouts.animal_id.isin(train)].groupby("session_id"):
        sequences.extend(tokens for _, tokens in segmented_tokens(sub))
    predictor.fit(sequences)
    prediction = []
    for meta in sessions.itertuples():
        sub = bouts[bouts.session_id == meta.session_id]
        sequences = [tokens for _, tokens in segmented_tokens(sub)]
        base = predictor.score(sequences, 0)["cross_entropy_bits"]
        for history in (0, 1, 2, 3, 5):
            score = predictor.score(sequences, history)
            prediction.append({"session_id": meta.session_id, "animal_id": meta.animal_id,
                               "group": meta.group, "condition": meta.condition, "history": history,
                               "discovery_mouse": meta.animal_id in train, "gain_vs_unigram_bits": base - score["cross_entropy_bits"],
                               **score})
    pd.DataFrame(prediction).to_csv(measurement_dir / "history_prediction.csv", index=False)
    # Compare the same syllable inside any detected chain versus outside. This
    # is an observational context contrast; overlapping membership is a union.
    events = pd.read_csv(chains_dir / "chain_occurrences.csv")
    context = []
    for meta in sessions.itertuples():
        sub = bouts[bouts.session_id == meta.session_id].copy()
        local = events[events.session_id == meta.session_id]
        membership = np.zeros(len(sub), bool)
        for event in local.itertuples():
            membership[event.start_bout:event.end_bout] = True
        sub["in_chain"] = membership
        for syllable, action in sub.groupby("syllable_id"):
            first, second = action[~action.in_chain], action[action.in_chain]
            if min(len(first), len(second)) < 10:
                continue
            for feature in ("duration_s", "mean_speed_px_s", "peak_phase", "latent_dispersion_units2"):
                context.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                                "condition": meta.condition, "context_feature": f"{syllable}/{feature}",
                                "n_outside": len(first), "n_inside": len(second),
                                "difference_inside_minus_outside": second[feature].mean() - first[feature].mean()})
    pd.DataFrame(context).to_csv(measurement_dir / "chain_context.csv", index=False)
    print("Execution variants, behavioural history and chain context saved", flush=True)
