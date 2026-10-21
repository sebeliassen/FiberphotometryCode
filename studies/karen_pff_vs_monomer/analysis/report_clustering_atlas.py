"""Report the Karen state-resolution experiment and its qualitative limits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.run_record import RunRecord
from fpcode.statistics.behavior_comparisons import adjust_fdr
from fpcode.viz.behavior_atlas import factor_templates, portable_html, timecourse_panels

LEVELS = ("L1", "L2", "L3", "L4", "L3_unresolved_separate", "L4_unresolved_separate")
WAVES = ("3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root, original = args.output.resolve(), args.atlas.resolve()
    record = RunRecord("karen-clustering-report", root / "provenance/report", config={"levels": LEVELS, "groups": ["Monomer", "PFF"]})
    membership = pd.read_csv(root / "cluster_membership.csv")
    annotations = {level: pd.read_csv(root / level / "cluster_annotations.csv").set_index("syllable_id") for level in LEVELS}
    catalogs, anchors = {}, {}
    for level in LEVELS:
        chain_root = original / "chains" if level == "L1" else root / level / "chains"
        catalogs[level] = pd.read_csv(chain_root / "chain_catalog.csv").set_index("chain_id") if chain_root.exists() else pd.DataFrame()
        anchors[level] = pd.read_csv(root / level / "anchors/anchor_mapping_audit.csv").dropna(subset=["anchor_id"]).drop_duplicates("anchor_id").set_index("anchor_id")

    def describe(level, family, feature):
        name, unresolved = feature, False
        first = feature.split("/")[0]
        ids = []
        if family in ("syllable_usage", "syllable_execution", "syllable_variants", "syllable_chain_context",
                      "raw_member_composition", "fixed_mixture_raw_execution", "observed_mixture_raw_execution") and first.isdigit():
            ids = [int(first)]
        elif family == "syllable_transitions":
            ids = [int(v) for v in first.split(">")]
        elif family.startswith("anchored") and first in anchors[level].index:
            ids = [int(v) for v in anchors[level].loc[first, "mapped_tokens"].split()]
        elif family.startswith("chain_"):
            chains = feature.split("/")[1].split(">") if family == "chain_transitions" else [first]
            if all(chain in catalogs[level].index for chain in chains):
                ids = [int(v) for chain in chains for v in str(catalogs[level].loc[chain, "tokens"]).split()]
        if ids:
            names = [str(annotations[level].loc[v, "description"]) for v in ids]
            unresolved = bool(annotations[level].loc[ids, "unresolved"].any())
            name = " → ".join(names) + ": " + "/".join(feature.split("/")[1:])
        return name, unresolved

    grouped, panels, candidates, models, paired = [], [], [], [], []
    for level in LEVELS:
        base = root / level / "statistics"
        for filename, destination in (("group_contrasts", grouped), ("animal_wave_measurements", panels),
                                      ("effect_stability", candidates), ("wave_adjusted_contrasts", models), ("paired_wave_changes", paired)):
            table = pd.read_csv(base / (filename + ".csv"), dtype={"animal_id": str})
            table["level"] = level
            destination.append(table)
            record.add_input_file(base / (filename + ".csv"))
    contrasts, panel, candidate, adjusted, changes = [pd.concat(values, ignore_index=True) for values in (grouped, panels, candidates, models, paired)]
    contrasts["q_across_primary_levels_family_scope"] = np.nan
    adjusted["q_across_primary_levels_family_scope"] = np.nan
    primary = contrasts.level.isin(LEVELS[:4])
    contrasts.loc[primary, "q_across_primary_levels_family_scope"] = contrasts[primary].groupby(["family", "view"]).p_permutation.transform(adjust_fdr)
    primary_model = adjusted.level.isin(LEVELS[:4])
    adjusted.loc[primary_model, "q_across_primary_levels_family_scope"] = adjusted[primary_model].groupby(["family", "view"]).p_cluster.transform(adjust_fdr)
    contrasts["q_all_levels_global"] = adjust_fdr(contrasts.p_permutation)
    for table in (contrasts, candidate, adjusted):
        described = [describe(r.level, r.family, r.feature) for r in table.itertuples()]
        table["label"] = [p[0] for p in described]
        table["unresolved_members"] = [p[1] for p in described]
    contrasts.to_csv(root / "all_level_group_contrasts.csv", index=False)
    adjusted.to_csv(root / "all_level_wave_adjusted_contrasts.csv", index=False)
    changes.to_csv(root / "all_level_paired_changes.csv", index=False)
    candidate.to_csv(root / "all_level_effect_stability.csv", index=False)
    named = candidate[candidate.meets_exploratory_heuristic & ~candidate.unresolved_members].copy()
    named["object"] = named.feature.str.split("/").str[0]
    chosen = named[named.level.isin(["L2", "L3", "L4"])].sort_values("priority_score", ascending=False)
    shortlist = chosen.drop_duplicates(["level", "family", "object"]).groupby(["level", "family"], sort=False).head(2)
    shortlist = shortlist.sort_values("priority_score", ascending=False)
    shortlist.to_csv(root / "interpretable_followup_shortlist.csv", index=False)
    usage = contrasts[(contrasts.family == "syllable_usage") & contrasts.feature.str.endswith("/time_fraction")].copy()
    signatures = {(r.level, r.cluster_id): tuple(sorted(membership[(membership.level == r.level) & (membership.cluster_id == r.cluster_id)].raw_syllable_id))
                  for r in membership[["level", "cluster_id"]].drop_duplicates().itertuples()}
    usage["raw_members"] = [" ".join(map(str, signatures[(r.level, int(r.feature.split('/')[0]))])) for r in usage.itertuples()]
    usage.to_csv(root / "membership_matched_usage_effects.csv", index=False)
    mechanics = []
    raw_summary = pd.read_csv(original / "measurements/session_metrics.csv").set_index("session_id")
    for level in LEVELS:
        summary = raw_summary if level == "L1" else pd.read_csv(root / level / "measurements/session_metrics.csv").set_index("session_id")
        assert np.allclose(summary.valid_duration_s, raw_summary.valid_duration_s)
        assert np.allclose(summary.mean_speed_px_s, raw_summary.mean_speed_px_s)
        assert np.all(summary.bout_count <= raw_summary.bout_count)
        audit = pd.read_csv(root / level / "anchors/anchor_mapping_audit.csv")
        mechanics.append({"level": level, "states": len(annotations[level]), "mean_bout_count": summary.bout_count.mean(),
                          "mean_bout_duration_s": summary.mean_bout_duration_s.mean(),
                          "mean_entropy_bits": summary.entropy_bits.mean(), "collapsed_raw_templates": int((~audit.retained).sum()),
                          "distinct_mapped_anchors": audit.anchor_id.nunique()})
    mechanics = pd.DataFrame(mechanics)
    mechanics.to_csv(root / "resolution_mechanics.csv", index=False)
    benchmarks, nulls = [], []
    for level in LEVELS[:4]:
        chain_root = original / "chains" if level == "L1" else root / level / "chains"
        benchmark = pd.read_csv(chain_root / "seqnmf_benchmarks.csv")
        selected = benchmark[benchmark.selected == True].copy()  # noqa: E712
        selected["level"] = level
        benchmarks.append(selected)
        audit = pd.read_csv(chain_root / "seqnmf_frozen_order_audit.csv")
        table = audit.groupby("null_kind").agg(real_error=("heldout_real_error", "first"),
                                               mean_order_gain=("order_gain", "mean"), minimum_gain=("order_gain", "min"),
                                               maximum_gain=("order_gain", "max")).reset_index()
        table["level"] = level
        nulls.append(table)
    benchmarks, nulls = pd.concat(benchmarks, ignore_index=True), pd.concat(nulls, ignore_index=True)
    benchmarks.to_csv(root / "seqnmf_resolution_benchmarks.csv", index=False)
    nulls.to_csv(root / "seqnmf_resolution_nulls.csv", index=False)
    summary = contrasts.groupby("level").agg(tested=("p_permutation", "count"), family_q_hits=("q_family", lambda s: (s < .05).sum()),
                                              across_level_family_q_hits=("q_across_primary_levels_family_scope", lambda s: (s < .05).sum())).reset_index()
    summary.to_csv(root / "screen_summary.csv", index=False)
    concrete = []
    for level, name in (("L3", "Turn_L"), ("L4", "Steering_Turn")):
        cid = annotations[level][annotations[level].description == name].index[0]
        row = contrasts[(contrasts.level == level) & (contrasts.family == "syllable_usage")
                        & (contrasts.feature == f"{cid}/time_fraction") & (contrasts.view == "untreated_pooled")].iloc[0]
        concrete.append(f"{level} {name}: PFF minus monomer {row.difference * 100:.2f} percentage points, "
                        f"individual 95% interval [{row.ci_low * 100:.2f}, {row.ci_high * 100:.2f}], g={row.hedges_g:.2f}, family q={row.q_family:.3g}")
    local_hits = contrasts[(contrasts.level == "L3") & (contrasts.q_family < .05)]
    paragraphs = [
        "This companion experiment compares the same 99 Karen sessions and 21 mice at the existing four hand-curated annotation resolutions: 100 raw states, 93 conservative synonyms, 22 meso categories and 14 broad families. They are semantic groupings, not newly fitted distance clusters. The raw atlas and its input files are retained.",
        "Aggregation changes the question. L2 combines several forward-dash, sniff and scrunch states. L3 retains walking and turning direction while grouping subtypes; L4 merges those directions into broad families. Remapping is followed by fresh gap-aware RLE. Different raw states that become the same category form one continuous bout; invalid frames and gaps remain boundaries. Longer bouts, fewer switches and lower absolute entropy at coarse levels are partly mechanical consequences of aggregation, not disease findings.",
        "My qualitative preference is L2 as a conservative sensitivity check and L3 as the most useful main interpretation level. L3 preserves distinctions such as left versus right steering while pooling raw variants. L4 is useful for broad summaries but can cancel opposing changes or conceal execution subtypes. The L2 dash merge is not uniform in speed: S3 was labelled walking pace and S4 fast. Historical group names such as Groom_Body and Rearing_Wall do not establish body location or wall specificity; use the raw labels/member table and videos before making those claims.",
        "Forty-four unassigned states and five Dump states require special care. Canonical L3/L4 pool them, which can create generic Unknown → action → Unknown motifs from unrelated behaviours. They remain in the full outputs, but the interpretable shortlist excludes unresolved members. Additional 69-/61-state sensitivity maps preserve each unknown and Dump state separately. Those sensitivity analyses recompute usage, execution, transitions and frozen mapped anchors; they do not refit seqNMF or execution-variant discovery.",
        "Fresh exact motifs, seqNMF, repetition grouping, prefix edits, execution variants, chain context and behavioural-history prediction were analysed at primary L2/L3/L4 using the same 12 training/4 validation/5 test mice as L1. Factor numbers are local to each fit. A separate anchor analysis maps the original 30 training-derived raw motifs into every level, RLE-collapses their token sequences and records routines that collapse below three bouts. This provides comparable hypotheses without equating unrelated factor IDs.",
        "The fixed-mixture tables help distinguish member composition from execution. Observed raw-member summaries use each recording's conditional state-time shares; fixed summaries use untreated discovery-mouse reference weights, with at least 80% reference-weight coverage. They average original raw-state execution measures, not newly merged-bout measures. Differences between them are a descriptive sensitivity, not causal mediation. Some genuine phenotypic changes consist precisely of altered member selection, so a composition effect is still worth investigating.",
        "Mouse-level means, individual bootstrap intervals, permutations, wave-adjusted models and paired longitudinal comparisons retain the original policies. Within-level family q values are exported alongside correction across the four primary levels within family/scope. Global q includes all six levels. A smaller q at a coarser level can reflect fewer tests even when the member set and effect estimate are identical; membership_matched_usage_effects.csv exposes this distinction. Four levels are correlated analyses of the same animals, not four independent replications.",
        "Non-significant effects remain visible and are ranked by magnitude, early-wave direction and leave-one-mouse-out stability. Ranking does not confirm a finding. The L-DOPA wave remains confounded with age/recording wave; its closest untreated baseline has only nine paired mice. The permutation resolution is .0005 with 1,999 resamples. No high-frequency physiological, paw-contact, height or millimetre claims are introduced by changing the label hierarchy.",
        "Absolute seqNMF reconstruction errors are not comparable evidence across vocabulary sizes: a smaller vocabulary is easier to reconstruct. Compare real-versus-null gains and alternate-seed stability within each level. Apparent additional history at coarse levels may partly reflect hidden raw-member identity. Treat all fitted routines as exploratory until video review and independent validation. The merging_review.csv file describes member speed/duration heterogeneity using discovery mice, without choosing clusters to maximise group differences.",
        "Reusable state mapping, composition sensitivity and anchored-chain measurement live in fpcode. Karen memberships and interpretation choices live in the study folder. Remaining comparison runs used four OpenMP/OpenBLAS threads per process and parallel independent levels; provenance records the thread environment. No source pickle reload was needed: the previously verified numerical frame cache was reused.",
    ]
    paragraphs.insert(1, "The steering difference survives aggregation. " + "; ".join(concrete)
                      + ". Both remain exploratory after correction. L3 has " + str(len(local_hits))
                      + " within-level family-corrected rows, describing related progression measures for one routine containing the unassigned pool. These are not four independent discoveries; inspect cross-level correction and raw members. L3 seqNMF has essentially no mean advantage over the first-order Markov null in this pilot; lower reconstruction error alone does not show that clustering solved routine discovery.")
    columns = ["level", "family", "label", "difference", "ci_low", "ci_high", "hedges_g", "q_family", "n_first", "n_second", "agreeing_consistency_waves"]
    tables = [("Resolution mechanics: descriptive changes caused partly by aggregation", mechanics), ("Screen counts; effect estimates matter beyond significance", summary),
              ("Interpretable exploratory candidates at levels 2–3", shortlist[shortlist.level.isin(["L2", "L3"])][columns].head(24)),
              ("Selected model fits: compare null gains rather than absolute error", benchmarks), ("Held-out order-null diagnostics", nulls)]
    unknown = candidate[candidate.meets_exploratory_heuristic & candidate.unresolved_members].sort_values("priority_score", ascending=False)
    tables.append(("Unresolved-state candidates: require annotation review", unknown[columns].head(12)))
    figures = []
    panel["source_family"] = panel.family
    panel["family"] = panel.level + "::" + panel.family
    descriptions = {(r.level + "::" + r.family, r.feature): r.level + " " + r.label for r in candidate.itertuples()}

    def curves(name, features, title):
        if not features:
            return
        path = root / "figures" / name
        timecourse_panels(panel, features, conditions=WAVES, groups=("Monomer", "PFF"), output=path, title=title, feature_labels=descriptions)
        figures.append((title, path.with_suffix(".png")))

    curves("mechanics", [(level + "::session", f) for level in LEVELS[:4] for f in ("bouts_per_min", "mean_bout_duration_s", "entropy_bits")], "Behavioural resolution changes bout definitions")
    for level in LEVELS[:4]:
        names = annotations[level]
        ids = [i for i in names.index if not names.loc[i, "unresolved"]]
        ids = ids[:18] if level in ("L3", "L4") else []
        if not ids:
            sub = shortlist[(shortlist.level == level) & (shortlist.family == "syllable_usage")].head(9)
            features = [(level + "::syllable_usage", f) for f in sub.feature]
        else:
            features = [(level + "::syllable_usage", f"{i}/time_fraction") for i in ids]
            descriptions.update({(level + "::syllable_usage", f"{i}/time_fraction"): names.loc[i, "description"] + " time fraction" for i in ids})
        curves(level + "_usage", features, level + " usage: individual mice and descriptive SEM")
        sub = shortlist[shortlist.level == level].head(12)
        curves(level + "_followup", [(level + "::" + r.family, r.feature) for r in sub.itertuples()], level + " follow-up candidates, including non-significant effects")
    for level in ("L3", "L4"):
        sub = shortlist[(shortlist.level == level) & (shortlist.family.str.startswith("anchored"))].head(6)
        curves(level + "_anchors", [(level + "::" + r.family, r.feature) for r in sub.itertuples()], level + " frozen raw-motif hypotheses after merging")
    for level in ("L2", "L3", "L4"):
        with np.load(root / level / "chains/seqnmf_templates.npz") as source:
            path = root / "figures" / (level + "_templates")
            factor_templates(source["w"], annotations[level].reset_index(), output=path)
        figures.append((level + " local seqNMF templates (cluster IDs)", path.with_suffix(".png")))
    links = [("All group contrasts with cross-level correction", "all_level_group_contrasts.csv"),
             ("Interpretable follow-up shortlist", "interpretable_followup_shortlist.csv"),
             ("Cluster memberships", "cluster_membership.csv"), ("Member heterogeneity review", "merging_review.csv"),
             ("Matched-member usage effects", "membership_matched_usage_effects.csv"),
             ("Wave-adjusted models", "all_level_wave_adjusted_contrasts.csv"), ("Paired longitudinal changes", "all_level_paired_changes.csv")]
    portable_html("Karen clustering levels 1–4: behavioural sensitivity", paragraphs, tables, figures, links, output=root / "report.html")
    markdown = ["# Karen clustering levels 1–4", *paragraphs]
    for caption, table in tables:
        markdown += ["## " + caption, table.to_markdown(index=False, floatfmt=".4g")]
    markdown += [f"![{caption}]({path.relative_to(root)})" for caption, path in figures]
    (root / "report.md").write_text("\n\n".join(markdown))
    result = {"levels": len(LEVELS), "sessions_per_level": 99, "mice": 21, "tested_contrasts": int(contrasts.p_permutation.notna().sum()),
              "primary_across_level_family_hits": int((contrasts.q_across_primary_levels_family_scope < .05).sum()),
              "interpretable_shortlist_rows": len(shortlist), "figures": len(figures)}
    (root / "report_summary.json").write_text(json.dumps(result, indent=2))
    record.finish(extra=result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
