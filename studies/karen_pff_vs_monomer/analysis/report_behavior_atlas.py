"""Build the Karen exploratory report from completed, reusable atlas products."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.run_record import RunRecord
from fpcode.statistics.followup_planning import (
    exploratory_effect_stability,
    pilot_effect_scenarios,
    sensitivity_scenarios,
    two_group_power,
)
from fpcode.viz.behavior_atlas import (
    effect_forest,
    factor_templates,
    portable_html,
    syllable_profiles,
    syllable_usage_heatmap,
    timecourse_panels,
)

GROUPS = ("Monomer", "PFF")
WAVES = ("3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa")
UNTREATED = WAVES[:-1]


def describe_feature(family, feature, annotations, catalog):
    first = feature.split("/")[0]
    descriptions = annotations.set_index("syllable_id").description
    categories = annotations.set_index("syllable_id").category
    label = feature
    annotation = ""
    if first.isdigit() and family.startswith(("syllable", "frame_posture", "frame_micro")):
        sid = int(first)
        annotation = categories.get(sid, "")
        label = f"S{sid:02d} {descriptions.get(sid, '')}: {'/'.join(feature.split('/')[1:])}"
    elif first in catalog.index:
        tokens = str(catalog.loc[first, "tokens"]).split()
        annotation = " → ".join(f"S{int(s):02d} {descriptions.get(int(s), '')} [{categories.get(int(s), '')}]" for s in tokens)
        label = f"{feature}: {' → '.join(tokens)}"
    elif family == "named_category":
        annotation = feature.rsplit("/", 1)[0]
    elif family == "syllable_transitions":
        states = first.split(">")
        if len(states) == 2 and all(s.isdigit() for s in states):
            annotation = " → ".join(f"S{int(s):02d} {descriptions.get(int(s), '')} [{categories.get(int(s), '')}]" for s in states)
            label = annotation + ": " + feature.rsplit("/", 1)[-1]
    elif family == "chain_transitions":
        pair = feature.split("/")[1].split(">")
        if len(pair) == 2 and all(chain in catalog.index for chain in pair):
            annotation = " → next chain → ".join(describe_feature("chain_usage", chain + "/rate", annotations, catalog)[1]
                                                   for chain in pair)
    return label, annotation


def units(feature):
    if feature.endswith("_elapsed_slope"):
        return units(feature.removesuffix("_elapsed_slope")) + " per normalised recording fraction"
    if "bodylengths_s" in feature:
        return "body lengths/s; 2D projected"
    if "bodylength" in feature:
        return "body lengths; 2D projected"
    for ending, unit in (("px_s3", "px/s³"), ("px_s2", "px/s²"), ("px_s", "px/s"), ("rad_s", "rad/s"),
                         ("units2", "latent units²"), ("units_s", "latent units/s"), ("units", "latent units")):
        if ending in feature:
            return unit
    if "per_min" in feature:
        return "per modelled minute"
    if "per_syllable_second" in feature:
        return "per sampled syllable second"
    if "bits" in feature:
        return "bits"
    if any(x in feature for x in ("fraction", "probability", "straightness", "_cv", "peak_phase", "asymmetry", "_ratio")):
        return "dimensionless"
    if "duration" in feature or "latency_s" in feature:
        return "seconds"
    if "turn_rad" in feature or "angle_rad" in feature:
        return "radians"
    if "displacement_px" in feature or "length_px" in feature:
        return "pixels"
    return "count/dimensionless; see feature definition"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    record = RunRecord("karen-behaviour-atlas-report", root / "provenance/report",
                       config={"groups": GROUPS, "waves": WAVES, "followup_min_mice_per_group": 6,
                               "followup_min_abs_g": .4, "followup_min_supported_waves": 2,
                               "followup_min_wave_direction_fraction": 2 / 3,
                               "prospective_alphas": [.05, .005], "pilot_multipliers": [.5, 1.]})
    code_root = Path(__file__).resolve().parents[3]
    record.extra["source_sha256_at_start"] = {
        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*list((code_root / "src/fpcode").rglob("*.py")), Path(__file__)]}
    stats_dir, measurements, chains = root / "statistics", root / "measurements", root / "chains"
    annotations = pd.read_csv(root / "syllable_annotations.csv")
    sessions = pd.read_csv(root / "sessions.csv", dtype={"animal_id": str})
    panel = pd.read_csv(stats_dir / "animal_wave_measurements.csv", dtype={"animal_id": str})
    contrasts = pd.read_csv(stats_dir / "group_contrasts.csv")
    adjusted = pd.read_csv(stats_dir / "wave_adjusted_contrasts.csv")
    paired = pd.read_csv(stats_dir / "paired_wave_changes.csv")
    catalog = pd.read_csv(chains / "chain_catalog.csv").set_index("chain_id")
    for directory in (stats_dir, measurements, chains):
        for path in directory.glob("*.csv"):
            record.add_input_file(path)
    candidate = exploratory_effect_stability(panel, contrasts, groups=GROUPS, untreated=UNTREATED,
                                              consistency_waves=WAVES[:3])
    labels = [describe_feature(r.family, r.feature, annotations, catalog) for r in candidate.itertuples()]
    candidate["label"] = [p[0] for p in labels]
    candidate["annotation"] = [p[1] for p in labels]
    candidate["units"] = candidate.feature.map(units)
    candidate["annotation_needs_review"] = candidate.annotation.str.contains("Unassigned|Dump", case=False)
    candidate["quality_measure"] = (candidate.family == "session") & candidate.feature.isin(["modelled_fraction", "valid_duration_s"])
    candidate["requires_tracking_review"] = candidate.family.isin(["frame_posture_paws", "frame_micro_motifs"])
    candidate.to_csv(stats_dir / "followup_effect_stability_all.csv", index=False)
    promising = candidate[candidate.meets_exploratory_heuristic & ~candidate.quality_measure].copy()
    promising.to_csv(stats_dir / "promising_effects.csv", index=False)
    wave_candidates = contrasts[(contrasts.view.isin(WAVES + ("change_untreated_mean_to_L_dopa",)))
                               & (contrasts.n_first >= 6) & (contrasts.n_second >= 6)
                               & (abs(contrasts.hedges_g) >= .4)].copy()
    wave_candidates["label"] = [describe_feature(r.family, r.feature, annotations, catalog)[0] for r in wave_candidates.itertuples()]
    wave_candidates["magnitude"] = abs(wave_candidates.hedges_g)
    wave_candidates = wave_candidates.sort_values("magnitude", ascending=False)
    wave_candidates.to_csv(stats_dir / "wave_specific_effect_candidates.csv", index=False)
    sensitivity = sensitivity_scenarios()
    sensitivity.to_csv(stats_dir / "prospective_fixed_effect_scenarios.csv", index=False)
    history = pd.read_csv(measurements / "history_prediction.csv", dtype={"animal_id": str})
    history = history[~history.discovery_mouse]
    history_panel = history.groupby(["animal_id", "group", "condition", "history"], as_index=False).agg(
        cross_entropy_bits=("cross_entropy_bits", "mean"), gain_vs_unigram_bits=("gain_vs_unigram_bits", "mean"))
    history_panel.to_csv(stats_dir / "heldout_history_by_animal_wave.csv", index=False)
    history_summary = history_panel.groupby(["condition", "group", "history"], as_index=False).agg(
        mice=("animal_id", "nunique"), mean_cross_entropy_bits=("cross_entropy_bits", "mean"),
        mean_gain_vs_unigram_bits=("gain_vs_unigram_bits", "mean"))
    history_summary.to_csv(stats_dir / "heldout_history_summary.csv", index=False)
    dictionary = panel[["family", "feature"]].drop_duplicates()
    dictionary["units"] = dictionary.feature.map(units)
    dictionary["label"] = [describe_feature(r.family, r.feature, annotations, catalog)[0] for r in dictionary.itertuples()]
    dictionary.to_csv(root / "feature_dictionary.csv", index=False)
    roster = sessions.groupby(["condition", "group"]).agg(sessions=("session_id", "size"), mice=("animal_id", "nunique")).reset_index()
    tested = int(contrasts.p_permutation.notna().sum())
    q_family = int((contrasts.q_family < .05).sum())
    q_global = int((contrasts.q_global < .05).sum())
    adjusted_hits = adjusted[(adjusted.view == "untreated_adjusted") & (adjusted.q_family < .05)]
    summary = pd.read_csv(measurements / "session_metrics.csv")
    bench = pd.read_csv(chains / "seqnmf_benchmarks.csv")
    chosen = bench[bench.selected == True].iloc[0]  # noqa: E712
    order = pd.read_csv(chains / "seqnmf_frozen_order_audit.csv")
    null_summary = order.groupby("null_kind").agg(real_error=("heldout_real_error", "first"), null_mean_error=("null_error", "mean"),
                                                 mean_gain=("order_gain", "mean"), min_gain=("order_gain", "min"), max_gain=("order_gain", "max")).reset_index()
    n_features = len(dictionary)
    paragraphs = [
        f"This is an exploratory behavioural atlas of {len(sessions)} recordings from {sessions.animal_id.nunique()} mice (12 PFF, 9 monomer), with {n_features:,} feature definitions. PFF/monomer is the experimental group, not a genetic grouping. The independent statistical unit is the mouse; repeated recordings are averaged within mouse and wave.",
        f"The broad screen contains {tested:,} tested mouse-level group contrasts: {q_family} have family-and-scope q < .05 and {q_global} have global q < .05. Failure to survive correction does not establish behavioural equivalence. The follow-up table deliberately includes promising non-significant patterns, with original-unit differences, bootstrap intervals, Hedges g and mouse support. The wave-adjusted regression is a separate model-dependent analysis; {len(adjusted_hits)} untreated effects have family q < .05 there.",
        f"There are {len(promising)} effects meeting the exploratory magnitude/stability heuristic: |g| ≥ .4, at least six mice per group, at least two supported early waves, direction agreement in at least two thirds of supported 3/8/16-week waves, and unchanged direction after leaving out any one mouse. These criteria are discovery filters, not confirmation. Nearby measures of the same state, mirrored variant fractions, transitions and different gap thresholds often describe the same pattern; the row count is not the number of independent biological findings.",
        "Pre/post L-DOPA means earlier untreated waves versus the separate L_dopa wave, around 26 weeks. The closest baseline is 24 weeks, with only 6 PFF and 3 monomer mice, all paired to the later wave. Earlier-wave means provide more pairs but combine ages. Drug, age, wave, handling and acquisition differences cannot be separated by this design. Adjacent untreated-wave and 3-to-16/24-week paired changes are also exported.",
        f"Raw syllable IDs are retained alongside the latest labels: 51 named states, 44 unassigned states and 5 Dump states. All included native-frame labels were checked exactly against V2 results and frame mappings: zero mismatches. The atlas contains {summary.valid_duration_s.sum() / 3600:.1f} modelled hours. Unmodelled frames and discontinuities split bouts and sequences; rates use modelled exposure. Quality differences, especially modelled coverage across waves, must be considered alongside behavioural effects.",
        "Syllable analyses cover time fraction, bout rates, duration distributions, diversity, transitions, recording-time trends, speed, turning, displacement, path straightness, latent dispersion, acceleration and jerk. Phase-normalised and native-onset curves describe execution; two group-blind clusters describe execution variants. The same syllable is compared inside versus outside detected chains. Category time fractions sum raw states; category bout rates sum raw-state bouts and are not new category-level episodes.",
        "Chains use existing gap-aware RLE semantics: repeated label frames form one bout. Exact motifs of 3–6 bouts are discovered in training mice, with 30 templates frozen before cohort comparisons. Exact matches do not overlap within a motif. SeqNMF suppresses nearby peaks within a factor. Overlaps between different motifs/factors remain in the full occurrence table; a score-ranked exclusive stream is used for chain transitions and run grouping. These are operational event definitions and require video review.",
        "Continuation/repetition/restart is assessed with gap thresholds of 0, 0.5, 1, 2 and 5 seconds; intervening different chains and data gaps break a run. Counts, run lengths, repeat fractions, initiations, first/repeated-call speed and transitions are saved. A motif-prefix analysis records observed progression and edit operations; prefix completion is a conditional sequence statistic, not evidence that an animal intended a routine. Omission/insertion alignment is descriptive and not a probabilistic grammar.",
        f"Bout seqNMF uses an independent CPU implementation of data-dependent cross-factor competition, not a MATLAB-identity claim. Six settings were compared on validation mice. Selected K={int(chosen.factors)}, L={int(chosen.length_bouts)} bouts, penalty={chosen.penalty:g}; held-out reconstruction error {chosen.heldout_test_error:.4f}, alternate-seed matched cosine {chosen.stability_mean:.3f}. The split is 12 training, 4 validation and 5 test mice. See the null table: a gain over shuffling alone is insufficient to establish structure beyond one-step transitions. Ten null replicates give a coarse diagnostic, not a formal significance claim. Test reconstruction is not future prediction.",
        "In this benchmark the held-out reconstruction advantage over order shuffling is small, and the advantage over simulated first-order sequences fluctuates around zero. Alternate-seed template agreement is modest. These results do not yet make a strong case that the fitted bout factors capture reproducible higher-order routines beyond ordinary transitions. Treat their group patterns as hypotheses; review concrete exact motifs, improve factor stability and repeat the order audit before choosing a seqNMF factor as a new-cohort primary endpoint.",
        "Native-time seqNMF uses 100 ms occupancy bins and 1.2 s templates; sustained-state components may primarily reflect dwell time. Frame-level micro-motifs use speed, absolute yaw and latent motion inside five common named syllables, with eight-frame templates. Target states were chosen from overall usage before group comparisons; these are exploratory numeric shapes, with no validated micromovement identity. Templates use training mice. History prediction uses common eligible positions for 0/1/2/3/5-bout history and provides held-out-mouse flags; training-mouse performance is not an estimate of generalisation.",
        "Native frames are 60 Hz. Model centroid positions and circular heading are median-smoothed over 9 frames (150 ms) before derivatives; speed uses an additional 5-frame median. This reduces tracking jitter but limits temporal resolution. Raw centroid derivatives are retained for auditing. Pose features use exact native-frame reindexing and confidence ≥ .5, body-relative 2D positions, widths, projected paw motion and asymmetry; inferential means require ≥80% valid feature frames. Relative paw speeds can contain tracking noise. There is no height, paw contact, millimetre calibration or validated tremor measurement.",
        "Inference uses 1,999 mouse-label permutations and 1,999 independent-mouse bootstrap resamples for group contrasts; paired changes use sign flips (exact for up to 12 pairs). The smallest Monte Carlo p value is .0005, which limits tail resolution in a screen this large; this pass should not be used to rule out a promising effect. Pooled estimates average each mouse's available waves, so uneven attendance affects the estimand. Separate wave-adjusted estimates use an additive group + wave model with mouse-clustered CR1 uncertainty and a cluster t reference. Broad-screen permutation q values and regression q values answer different model questions. Bootstrap intervals are individual 95% intervals, not simultaneous coverage across the screen. Rare states and execution with fewer than ten bouts/calls remain missing.",
        "For a new cohort, first review videos and tracking for candidate endpoints, decide a biologically useful difference in natural units, freeze the endpoint and observation window, and then use independent animals with balanced groups. Choose a small endpoint set before collecting the new data. A larger N can improve precision but cannot fix ambiguous annotations, tracking artefacts, wave/drug confounding or a poorly defined chain. Record coverage and exclusions, retain a blinded comparison and use an independently fixed chain library or an explicit training/test split.",
        "The prospective tables assume independent equal-size groups, normally distributed equal-variance outcomes, a fixed true standardised effect and no attrition. Alpha .05 illustrates one endpoint; .005 illustrates a conservative ten-endpoint Bonferroni allocation, not the thousands-feature BH screen. Fixed d = .3/.5/.8 scenarios are the starting point. Separate tables show selected pilot |g| and 50% of that magnitude; both are assumptions, and the 50% value is not a validated shrinkage estimator. These are design sensitivities, not achieved-power evidence that any pilot effect is real.",
        "Forty native-frame example clips cover four concrete motifs across both groups and all five waves. They are linked in the gallery and selected near the median duration in each cell. They have not been behaviourally reviewed. Inspect the clips before assigning ethological names to motifs or using them as future primary endpoints. Original pickle, model and fused-pose inputs were not modified.",
    ]
    result_examples = [
        ("session", "mean_speed_px_s", "Mean movement speed"),
        ("syllable_usage", "71/time_fraction", "S71, labelled turn left, time fraction"),
        ("syllable_execution", "55/latent_dispersion_units2", "S55, labelled walk right, within-bout latent dispersion"),
        ("chain_progression", "exact_14/phases_reached", "Turn left → dash → turn left prefix progression"),
    ]
    statements = []
    for family, feature, name in result_examples:
        sub = candidate[(candidate.family == family) & (candidate.feature == feature)]
        if not len(sub):
            continue
        r = sub.iloc[0]
        statements.append(f"{name}: PFF minus monomer {r.difference:.4g} {r.units}, individual 95% interval [{r.ci_low:.4g}, {r.ci_high:.4g}], g={r.hedges_g:.2f}, family q={r.q_family:.3g}; direction agrees in {r.agreeing_consistency_waves}/{r.supported_consistency_waves} supported early waves")
    paragraphs.insert(3, "Concrete follow-up examples from the untreated analysis. " + "; ".join(statements)
                      + ". These are selected pilot estimates. The speed interval includes zero; the turn/dispersion/prefix patterns remain unconfirmed after the broad correction. S55 latent dispersion is a model-space variability measure, not a direct physical amplitude. Prefix progression is conditional on the initial two bouts and is not intended-task completion.")
    # Select distinct objects within each family to avoid filling the summary with
    # multiple highly correlated metrics from one syllable or motif.
    shortlist_parts = []
    for _, sub in promising.groupby("family", sort=False):
        sub = sub.copy()
        sub["object"] = sub.feature.str.split("/").str[0]
        shortlist_parts.append(sub.drop_duplicates("object").head(2))
    shortlist = pd.concat(shortlist_parts).sort_values("priority_score", ascending=False) if shortlist_parts else promising
    shortlist.to_csv(stats_dir / "followup_shortlist.csv", index=False)
    pilot_effect_scenarios(shortlist).to_csv(stats_dir / "prospective_pilot_shrinkage_scenarios.csv", index=False)
    tables = [("Recordings and independent mice by wave", roster),
              ("Exploratory follow-up shortlist: nonsignificant effects retained", shortlist[["family", "label", "annotation", "difference", "ci_low", "ci_high", "hedges_g", "q_family", "n_first", "n_second", "agreeing_consistency_waves", "requires_tracking_review"]]),
              ("Wave-adjusted untreated effects with family q < .05: separate model", adjusted_hits),
              ("Held-out seqNMF order diagnostics", null_summary),
              ("History prediction in mice excluded from predictor training", history_summary),
              ("Fixed-effect design scenarios", sensitivity[sensitivity.animals_per_group.isin([20, 40, 60, 100])]),
              ("Within-group paired drug-wave changes: session metrics", paired[(paired.family == "session") & (paired.target == "L_dopa")])]
    for scope in ("L_dopa", "change_untreated_mean_to_L_dopa"):
        top = wave_candidates[wave_candidates.view == scope].copy()
        top["object"] = top.feature.str.split("/").str[0]
        top = top.drop_duplicates(["family", "object"]).head(12)
        tables.append(("Exploratory effect magnitudes: " + scope, top[["family", "label", "difference", "ci_low", "ci_high", "hedges_g", "q_family", "n_first", "n_second"]]))
    figures = []
    figure_dir = root / "figures"
    feature_labels = {(r.family, r.feature): r.label for r in dictionary.itertuples()}

    def curves(name, features, title):
        if not features:
            return
        path = figure_dir / name
        timecourse_panels(panel, features, conditions=WAVES, groups=GROUPS, output=path, title=title,
                          feature_labels=feature_labels)
        figures.append((title, path.with_suffix(".png")))

    curves("session_overview", [("session", f) for f in ("mean_speed_px_s", "pause_fraction", "bouts_per_min", "mean_bout_duration_s", "entropy_bits", "effective_shannon", "top5_fraction", "modelled_fraction", "valid_duration_s")], "Mouse-level session measures and coverage")
    categories = panel[panel.family == "named_category"].feature.unique()
    curves("category_usage", [("named_category", f) for f in categories if f.endswith("/time_fraction")][:18], "Label-category time fractions")
    curves("followup_patterns", [(r.family, r.feature) for r in shortlist.head(18).itertuples()], "Follow-up patterns: individual mice, means and SEM")
    for family in ("syllable_usage", "syllable_execution", "chain_usage", "chain_execution", "chain_progression", "chain_grouping", "chain_transitions", "syllable_variants", "syllable_chain_context", "behavioural_history", "frame_posture_paws", "native_time_motifs", "frame_micro_motifs"):
        sub = candidate[candidate.family == family].head(9)
        curves(family, [(family, f) for f in sub.feature], family.replace("_", " ").capitalize() + ": exploratory largest stable effects")
    for i, ids in enumerate(np.array_split(catalog.index.to_numpy(), 4)):
        curves(f"chain_rates_{i}", [("chain_usage", f"{chain}/occurrences_per_min") for chain in ids], f"Frozen chain library: rates, page {i + 1}")
    syllable_usage_heatmap(contrasts, annotations, conditions=WAVES, output=figure_dir / "all_syllable_usage",
                           difference_label="PFF minus monomer")
    figures.append(("All raw syllable usage differences (PFF − monomer)", figure_dir / "all_syllable_usage.png"))
    with np.load(chains / "seqnmf_templates.npz") as templates:
        factor_templates(templates["w"], annotations, output=figure_dir / "seqnmf_templates")
    figures.append(("Bout seqNMF templates", figure_dir / "seqnmf_templates.png"))
    if len(shortlist):
        effect_forest(shortlist, output=figure_dir / "followup_intervals", title="Exploratory untreated differences: separate original-unit axes")
        figures.append(("Original-unit effect estimates and individual 95% intervals", figure_dir / "followup_intervals.png"))
    common = pd.read_csv(measurements / "syllables.csv").groupby("syllable_id").time_fraction.mean().nlargest(12).index.tolist()
    for filename, coordinate in (("phase_profiles", "phase"), ("onset_profiles", "time_s")):
        profile = pd.read_csv(measurements / (filename + ".csv"))
        syllable_profiles(profile, common, groups=GROUPS, output=figure_dir / filename,
                          title="Common syllables: untreated execution profiles", coordinate=coordinate, conditions=UNTREATED)
        figures.append((filename.replace("_", " "), figure_dir / (filename + ".png")))
    links = [("Full CSV inventory", "csv_manifest.csv"), ("Feature labels and units", "feature_dictionary.csv"),
             ("Promising effects including nonsignificant estimates", "statistics/promising_effects.csv"),
             ("Full mouse-level contrasts", "statistics/group_contrasts.csv"), ("Paired longitudinal changes", "statistics/paired_wave_changes.csv"),
             ("Prospective sample-size scenarios", "statistics/prospective_fixed_effect_scenarios.csv"),
             ("Shortlist-specific pilot/shrinkage design assumptions", "statistics/prospective_pilot_shrinkage_scenarios.csv"),
             ("Native-frame clip gallery", "examples/gallery.html"), ("seqNMF reference paper", "https://elifesciences.org/articles/38471"),
             ("Noncentral-t sensitivity implementation", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html")]
    portable_html("Karen PFF versus monomer: exploratory behaviour atlas", paragraphs, tables, figures, links, output=root / "report.html")
    markdown = ["# Karen PFF versus monomer: exploratory behaviour atlas", *paragraphs]
    for caption, table in tables:
        markdown.extend(["## " + caption, table.to_markdown(index=False, floatfmt=".4g")])
    for caption, path in figures:
        markdown.extend(["## " + caption, f"![{caption}]({path.relative_to(root)})"])
    markdown.extend(["## Data and methods references", *[f"- [{name}]({url})" for name, url in links]])
    (root / "report.md").write_text("\n\n".join(markdown))
    overview = {"sessions": len(sessions), "mice": sessions.animal_id.nunique(), "features": n_features,
                "tested_group_contrasts": tested, "family_q_lt_05": q_family, "global_q_lt_05": q_global,
                "exploratory_heuristic_rows": len(promising), "shortlist_rows": len(shortlist), "figures": len(figures),
                "planning_n80_fixed_d": {str(d): next(n for n in range(2, 501) if two_group_power(d, n) >= .8) for d in (.3, .5, .8)}}
    (root / "report_summary.json").write_text(json.dumps(overview, indent=2))
    manifest = []
    for path in sorted(root.rglob("*.csv")):
        if "bouts_by_session" in path.parts or path.name == "csv_manifest.csv":
            continue
        with path.open() as handle:
            columns = handle.readline().strip()
            count = sum(1 for _ in handle)
        manifest.append({"path": str(path.relative_to(root)), "rows": count, "bytes": path.stat().st_size, "columns": columns})
    pd.DataFrame(manifest).to_csv(root / "csv_manifest.csv", index=False)
    for filename in ("report.html", "report.md", "report_summary.json", "csv_manifest.csv", "feature_dictionary.csv"):
        record.add_output_file(root / filename)
    record.finish(extra=overview)
    print(json.dumps(overview), flush=True)


if __name__ == "__main__":
    main()
