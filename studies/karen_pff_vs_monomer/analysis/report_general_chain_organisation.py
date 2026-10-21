"""Report Karen's overall chain tendencies and their coverage/support limits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.viz.behavior_atlas import effect_forest, portable_html, timecourse_panels
from studies.karen_pff_vs_monomer.analysis.run_general_chain_organisation import (
    PRIMARY_LIBRARY,
    PRIMARY_METRICS,
    UNTREATED,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output
    panel = pd.read_csv(output / "animal_wave_measurements.csv", dtype={"animal_id": str})
    contrasts = pd.read_csv(output / "group_contrasts.csv")
    candidates = pd.read_csv(output / "exploratory_candidates.csv")
    support = pd.read_csv(output / "library_support.csv")
    sessions = pd.read_csv(output / "sessions.csv", dtype={"animal_id": str})
    counts = pd.read_csv(output / "animal_wave_counts.csv", dtype={"animal_id": str})
    primary = contrasts[contrasts.family == "general_chain_primary"].copy()
    primary["metric"] = primary.feature.str.split("/").str[-1]
    primary.to_csv(output / "primary_contrasts.csv", index=False)
    primary_pooled = primary[primary.view.isin(["untreated_pooled", "all_waves_pooled", "change_24w_PI_to_L_dopa"])].copy()
    primary_pooled = primary_pooled[["metric", "view", "n_first", "n_second", "mean_first", "mean_second", "difference",
                                    "ci_low", "ci_high", "hedges_g", "p_permutation", "q_family", "status"]]
    # Do not present many correlated aliases of the same effect as many discoveries.
    inspected_metrics = {"self_excess", "mean_run_calls", "return2_excess", "initiations_per_min", "longest_run",
                         "first_order_return2_probability_excess_equal_recording", "first_order_longest_run_excess_equal_recording"}
    select = candidates[candidates.feature.str.split("/").str[-1].isin(inspected_metrics)
                        & candidates.feature.str.contains("/g1/")
                        & ~candidates.feature.str.startswith(("L1_exact/", "L2_anchors/", "L3_unresolved", "L4_unresolved"))].copy()
    select["metric"] = select.feature.str.split("/").str[-1]
    select = select.drop_duplicates(["metric", "feature"]).head(16)
    spotlight_features = ["L3_seqnmf/interpretable/g1/self_excess", "L4_seqnmf/interpretable/g1/mean_run_calls",
                          "L4_anchors/interpretable/g1/mean_run_calls", "L3_anchors/inclusive/g1/initiations_per_min"]
    spotlight = candidates[candidates.feature.isin(spotlight_features)].copy()
    spotlight["metric"] = spotlight.feature.str.split("/").str[-1]
    select = pd.concat([spotlight, select], ignore_index=True).drop_duplicates("feature").head(16)
    select.to_csv(output / "report_exploratory_shortlist.csv", index=False)
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    figures = []
    conditions = (*UNTREATED, "L_dopa")
    primary_features = [("general_chain_primary", f"{PRIMARY_LIBRARY}/interpretable/g1/{metric}") for metric in PRIMARY_METRICS]
    primary_features += [("general_chain_exploratory", f"{PRIMARY_LIBRARY}/interpretable/g1/{metric}")
                         for metric in ("coverage_fraction", "calls_per_min", "right_censored_fraction")]
    timecourse_panels(panel, primary_features, conditions=conditions, groups=("Monomer", "PFF"),
                      output=figs / "primary", title="Frozen named L3 anchors: endpoints, coverage and censoring")
    figures.append(("Primary endpoints and their limited observed support", figs / "primary.png"))
    for kind, libraries, measures in (
        ("anchor_levels", ("L1_anchors", "L2_anchors", "L3_anchors", "L4_anchors"), PRIMARY_METRICS),
        ("seqnmf_levels", ("L2_seqnmf", "L3_seqnmf", "L4_seqnmf"), PRIMARY_METRICS),
        ("initiations", ("L3_anchors", "L3_seqnmf", "L4_seqnmf"), ("initiations_per_min", "calls_per_min", "self_probability")),
    ):
        features = [("general_chain_primary" if library == PRIMARY_LIBRARY and metric in PRIMARY_METRICS else "general_chain_exploratory",
                     f"{library}/interpretable/g1/{metric}") for library in libraries for metric in measures]
        timecourse_panels(panel, features, conditions=conditions, groups=("Monomer", "PFF"), output=figs / kind,
                          title=kind.replace("_", " ") + ": named-member libraries, one point per mouse/wave")
        figures.append((kind.replace("_", " ").capitalize(), figs / (kind + ".png")))
    gap = contrasts[(contrasts.view == "untreated_pooled") & contrasts.feature.str.startswith(PRIMARY_LIBRARY + "/interpretable/")
                    & contrasts.feature.str.split("/").str[-1].isin(PRIMARY_METRICS)].copy()
    gap["label"] = gap.feature.str.replace(PRIMARY_LIBRARY + "/interpretable/", "", regex=False) + " (PFF − monomer)"
    effect_forest(gap, output=figs / "gap_sensitivity", title="Primary anchor endpoints across gap policies", limit=15)
    figures.append(("Gap-policy sensitivity: individual-mouse bootstrap intervals", figs / "gap_sensitivity.png"))
    if len(select):
        forest = select.copy()
        forest["label"] = forest.feature + " (PFF − monomer)"
        effect_forest(forest, output=figs / "exploratory", title="Selected exploratory effects; inspect support and uncertainty", limit=12)
        figures.append(("Promising patterns remain candidates even when not significant", figs / "exploratory.png"))
    local = support[(support.library == PRIMARY_LIBRARY) & (support.scope == "interpretable")].iloc[0]
    tested = primary[primary.status == "tested"]
    hits = int((tested.q_family < .05).sum())
    all_hits = int(((contrasts.status == "tested") & (contrasts.q_global < .05)).sum())
    primary_counts = counts[(counts.library == PRIMARY_LIBRARY) & (counts.scope == "interpretable") & (counts.gap_seconds == 1.)]
    wave_support = primary_counts.groupby("condition", as_index=False).agg(
        animals=("animal_id", "nunique"), calls=("calls", "sum"), pairs=("pair_opportunities", "sum"),
        return2_windows=("return2_opportunities", "sum"), animals_with_pairs=("pair_opportunities", lambda x: int((x > 0).sum())),
        animals_with_return2=("return2_opportunities", lambda x: int((x > 0).sum())))
    wave_support.to_csv(output / "primary_wave_support.csv", index=False)
    paragraphs = [
        f"Completed general-chain steps 1–6 on {len(sessions)} recordings from {sessions.animal_id.nunique()} mice, using 14 frozen libraries, inclusive/named-member scopes, five gaps and 1,999 configurable surrogate replicates. Models and labels were not refitted.",
        f"The primary named L3 anchor library covers a median {100 * local.median_coverage:.2f}% of modelled time. It has {int(local.total_pairs)} nearby successor pairs and {int(local.total_return2_opportunities)} A→B→A opportunities across all sessions. These are tendencies among detected routines; they cannot describe all behaviour. Few complete runs are observed, and many runs end through loss of observation rather than an observed switch.",
        f"The three frozen primary endpoints produce {hits} family/scope q<.05 group contrasts among {len(tested)} tested rows. The full exploratory group screen has {all_hits} global q<.05 rows. Sparse or missing primary loop comparisons are shown as insufficient support. Neither missing significance nor insufficient support establishes equivalence.",
        "PFF-minus-monomer estimates use independent mice. Within mouse/wave, opportunities, run sums and exposure are pooled; available waves are averaged equally per mouse for untreated/all-wave summaries. Separate regression coefficients adjust for wave with mouse-clustered uncertainty. Equal-session and balanced early-attendance sensitivities are exported.",
        "At g0, calls must abut; at g1, up to one second of valid unmatched time is allowed. Missing data and excluded calls break links. Chain-only next detections may bridge long unrecognised intervals and are a separate diagnostic. Initiations include singleton starts; observed initiations exclude left-censored run starts. All duration fractions use nonoverlapping call intervals.",
        "Strict self-adjacency can also be structurally prevented by the templates and RLE: 14 of the 19 named primary anchors begin and end with the same syllable. Abutting copies would place identical endpoint syllables together, which RLE merges. A g0 zero for those exact templates therefore cannot establish that the animal does not repeat the underlying routine.",
        "Frequency excess subtracts an exact count-preserving within-reference-block expectation. The shuffled intervals preserve timestamps, counts and barriers. Quarter-stratified controls omit cross-quarter links. First-order controls use session transition estimates with fixed disconnected starts; their fitting uncertainty is not integrated. They address higher-order shapes/returns, and do not test self probability against a null fitted to that probability.",
        "SeqNMF's wider coverage improves descriptive support, but dominant-token labels do not certify every minor factor weight. Prior held-out audits found limited/unstable advantage over first-order controls, especially at L3. A group difference in factor calls alone does not establish a reproducible higher-order routine.",
        "L_dopa is a separate later wave, approximately 26 weeks. Drug, age and wave are confounded. The closest 24-week paired comparison has only nine animals overall (six PFF, three monomer), and endpoint-specific support can be lower.",
        "The exploratory shortlist uses magnitude, week direction and leave-one-mouse-out direction without requiring p<.05. Repeated weeks, multiple gaps, clustering levels and complementary metrics are correlated evidence. Leave-one-chain-out retains the removed routine as a barrier. Future-cohort tables assume specified true effects; pilot-selected estimates are not validated planning effects.",
    ]
    for row in spotlight.itertuples():
        paragraphs.append(f"Exploratory untreated {row.feature}: PFF − monomer {row.difference:+.5f} "
                          f"(95% mouse-bootstrap interval {row.ci_low:+.5f} to {row.ci_high:+.5f}; Hedges g {row.hedges_g:+.2f}; "
                          f"broad family/scope q {row.q_family:.3g}). Direction agrees in {row.agreeing_consistency_waves}/{row.supported_consistency_waves} "
                          f"supported early waves, with leave-one-mouse-out direction agreement {row.leave_one_animal_out_direction_fraction:.0%}. "
                          "This is a follow-up candidate, not a confirmed difference.")
    usage = contrasts[(contrasts.view == "untreated_pooled") & (contrasts.feature == "L3_anchors/inclusive/g1/calls_per_min")].iloc[0]
    initiation = contrasts[(contrasts.view == "untreated_pooled") & (contrasts.feature == "L3_anchors/inclusive/g1/initiations_per_min")].iloc[0]
    paragraphs.append(f"Lower inclusive-anchor initiations ({initiation.difference:+.3f}/min) closely track lower call rates "
                      f"({usage.difference:+.3f}/min). Most of this candidate concerns fewer detected routines, rather than longer repetition runs.")
    sensitivity = pd.read_csv(output / "sensitivity_contrasts.csv")
    chain_out = sensitivity[sensitivity.policy == "leave_chain_out"]
    leave_chain_rows = []
    for row in spotlight.itertuples():
        library, scope, _, metric = row.feature.split("/")
        checks = chain_out[(chain_out.library == library) & (chain_out.scope == scope) & (chain_out.metric == metric)]
        finite = checks.difference.dropna()
        if len(finite):
            leave_chain_rows.append({"feature": row.feature, "removals": len(finite),
                                     "same_direction_fraction": float(np.mean(np.sign(finite) == np.sign(row.difference))),
                                     "difference_min": finite.min(), "difference_max": finite.max()})
    leave_chain_summary = pd.DataFrame(leave_chain_rows)
    leave_chain_summary.to_csv(output / "spotlight_leave_chain_summary.csv", index=False)
    # Make a reusable feature dictionary and an indexed inventory of the outputs.
    units = {"self_excess": "probability difference", "return2_excess": "probability difference", "mean_run_calls": "calls/run",
             "complete_mean_run_calls": "calls/run",
             "initiations_per_min": "runs/modelled minute", "calls_per_min": "calls/modelled minute",
             "observed_initiations_per_min": "observed starts/modelled minute", "longest_run": "calls",
             "effective_routines": "effective IDs", "transition_entropy_nats": "nats"}
    dictionary = []
    for feature, family in panel[["feature", "family"]].drop_duplicates().itertuples(index=False, name=None):
        library, scope, gap_tag, metric = feature.split("/", 3)
        base_metric = metric.removeprefix("frequency_").removeprefix("first_order_").removesuffix("_excess_equal_recording")
        dictionary.append({"family": family, "feature": feature, "library": library, "scope": scope, "gap": gap_tag,
                           "metric": metric, "units": units.get(base_metric, "dimensionless"),
                           "aggregation": "equal recording surrogate diagnostics" if metric.endswith("equal_recording") else "count/exposure pooling; nonadditive diagnostics use recording mean/max"})
    pd.DataFrame(dictionary).to_csv(output / "feature_definitions.csv", index=False)
    tables = [("Primary estimates and endpoint-specific animal support", primary_pooled),
              ("Primary observation opportunities by wave", wave_support),
              ("Library coverage and succession support", support),
              ("Exploratory candidate patterns (not independent discoveries)", select[["feature", "n_first", "n_second", "difference", "ci_low", "ci_high", "hedges_g", "q_family", "wave_direction_fraction", "leave_one_animal_out_direction_fraction"]])]
    tables.append(("Does each spotlight direction survive removing a detected routine?", leave_chain_summary))
    indexed = []
    for path in sorted(output.glob("*.csv")):
        data = pd.read_csv(path)
        indexed.append({"path": path.name, "rows": len(data), "columns": len(data.columns), "bytes": path.stat().st_size})
    pd.DataFrame(indexed).to_csv(output / "csv_index.csv", index=False)
    links = [(row["path"], row["path"]) for row in indexed] + [("Frozen policy", "policy.json"), ("Interval/call/pair/run audits", "audit/")]
    portable_html("Karen: general chain organisation", paragraphs, tables, figures, links, output=output / "report.html")
    lines = ["# Karen: general chain organisation", "", *[p + "\n" for p in paragraphs]]
    for caption, table in tables:
        lines += ["## " + caption, "", table.to_markdown(index=False, floatfmt=".4g"), ""]
    for caption, path in figures:
        lines += ["## " + caption, "", f"![{caption}]({path.relative_to(output)})", ""]
    lines += ["## Output locations", "", "Full compressed audits are under `audit/<library>/<session>/`; all comparison tables are indexed in `csv_index.csv`. Shared mechanisms belong in `src/fpcode`, with Karen choices and reporting in the study folder."]
    (output / "report.md").write_text("\n".join(lines))
    (output / "report_summary.json").write_text(json.dumps({"primary_tested": len(tested), "primary_family_hits": hits,
                                                           "all_group_global_hits": all_hits, "figures": len(figures),
                                                           "primary_median_coverage": float(local.median_coverage)}, indent=2))


if __name__ == "__main__":
    main()
