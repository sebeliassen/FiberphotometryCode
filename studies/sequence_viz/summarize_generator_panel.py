"""Compact configuration scorecard from completed readouts, without new tests/fits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(output):
    import pandas as pd

    from fpcode.common.cache_manifest import atomic_write_text

    output = Path(output)
    plan = json.loads((output/"run_manifest.json").read_text())
    selection = json.loads(Path(plan["panel"]).read_text())
    if json.loads((output/"aggregation_status.json").read_text())["state"]!="COMPLETED":
        raise ValueError("completed quantitative scorecard required")
    kept = []
    for chunk in pd.read_csv(output/"mouse_readouts.csv.gz",chunksize=50000):
        eligible = (chunk.scope=="__all__")&chunk.parent.isna()
        baseline = ((chunk.cohort=="karen")&(chunk.epoch=="whole_supported")&(chunk.condition!="L_dopa"))|(
            (chunk.cohort=="jonathan")&(chunk.epoch=="pre_injection_15min"))
        kept.append(chunk[eligible&baseline])
    rows = pd.concat(kept,ignore_index=True)
    # First average available baseline sessions/waves within mouse, then mice.
    mice = rows.groupby(["panel","subject","family","feature"],observed=True).value.mean().reset_index()
    values = mice.groupby(["panel","family","feature"],observed=True).value.mean()
    physical = pd.read_csv(output/"physical_consistency.csv")
    physical_mean = physical.groupby(["panel_id","mode"],observed=True).separation.mean()
    physical_min = physical.groupby(["panel_id","mode"],observed=True).mouse_omission_min.min()
    crosswalk = pd.read_csv(output/"catalogue_correspondence.csv")
    accepted = crosswalk.groupby("panel_id").accepted.mean()
    trials = []
    for entry in selection["entries"]:
        panel = entry["panel_id"]
        def val(family,feature):
            return values.get((panel,family,feature),float("nan"))
        trials.append(dict(panel_id=panel,group=entry["group"],cohort=entry["cohort"],base_recipe=entry["base_recipe"],
            method=entry["recipe"]["method"],route=entry["recipe"]["route"],resolution=entry["recipe"]["resolution"],
            held_animals=int(mice[mice.panel==panel].subject.nunique()),
            baseline_coverage=val("recruitment","coverage_fraction"),baseline_calls_per_min=val("recruitment","calls_per_min"),
            baseline_mean_duration_s=val("execution","mean_call_duration_s"),
            baseline_effective_types=val("inter","g10_effective_routines"),
            baseline_repetition_excess=val("inter","g10_self_excess"),
            baseline_ABA_excess=val("inter","g10_return2_excess"),
            full_session_physical=physical_mean.get((panel,"all")),
            full_session_source_conditioned=physical_mean.get((panel,"source_conditioned")),
            full_session_unused_source_conditioned=physical_mean.get((panel,"unused_source_conditioned")),
            source_conditioned_mouse_omission_min=physical_min.get((panel,"source_conditioned")),
            original_window_seed_f1=entry["search_metrics"].get("seed_stability"),
            training_correspondence_acceptance=accepted.get(panel)))
    pd.DataFrame(trials).to_csv(output/"configuration_scorecard.csv",index=False)
    # A manageable general-axes companion; do not cherry-pick significant rows.
    general = []
    with (output/"phenotype_contrasts.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["scope"]=="__all__" and not row["parent"] and row["family"] in {"recruitment","execution","intra","inter","inter_null"}:
                general.append(row)
    if general:
        with (output/"general_phenotype_contrasts.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(general[0]))
            writer.writeheader()
            writer.writerows(general)
    def number(value,percent=False):
        if value is None or pd.isna(value):
            return "—"
        return f"{value*100:.1f}%" if percent else f"{value:.2f}"
    parts = ["# Fixed configuration scorecard\n\nAll23settings and2,038jobs completed successfully. "
        "This is a comparison of complete configurations; there is no method-family recipe average or weighted truth score.\n\n"
        "Baseline values average sessions/waves within mouse before mice: Karen's available untreated waves; "
        "Jonathan's two pre-injection15minute epochs. Physical separation uses all broader supported sessions "
        "and averages each setting's own saved folds/seeds. Those scopes differ deliberately. Original seedF1 "
        "is the previous window measurement, not a new full-session stability estimate.\n\n"]
    for cohort in ("karen","jonathan"):
        parts += [f"## {cohort.capitalize()}\n\n",
            "| Setting | Method / input route | Coverage | Calls/min | Source-conditioned physical | Unused-channel physical | Prior seedF1 |\n",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"]
        for row in trials:
            if row["cohort"]!=cohort:
                continue
            parts.append(f"| {row['panel_id']} / {row['base_recipe']} | {row['method']} / {row['route']} | "
                f"{number(row['baseline_coverage'],True)} | {number(row['baseline_calls_per_min'])} | "
                f"{number(row['full_session_source_conditioned'])} | {number(row['full_session_unused_source_conditioned'])} | "
                f"{number(row['original_window_seed_f1'])} |\n")
        parts.append("\n")
    parts.append("C01–C09 are the core; D01–D14 the diversity challenges. Missing unused-channel evidence "
        "stays missing where all physical channels entered construction. Physical separation is descriptive and "
        "partly circular; condition-specific intra changes can increase within-type variation.\n\n"
        "The full CSV also gives call duration, effective type count, repetition/ABA frequency excess, "
        "mouse-omission sensitivity and conservative training correspondence support. "
        "`general_phenotype_contrasts.csv` retains every general recruitment/intra/inter contrast, rather than "
        "only significant examples. Chain-specific effects remain in the complete file with their support and "
        "local/reference identity. Only238of679proposed source-phase correspondences passed the fixed similarity/ambiguity "
        "policy; this is not semantic or phase homology.\n\n"
        "Frequency controls permute labels within native/five-minute reference blocks. First-order controls fit "
        "a pooled within-epoch transition model, retain disconnected starts and fixed slots, and do not fix exact "
        "local frequencies or integrate transition-fit uncertainty. Group intervals condition on the finite null "
        "means; Monte Carlo meanSE is reported separately. These are selected-call organisation controls, not detector-input "
        "false-positive controls.\n\n"
        "Phenotype selection remains exploratory after prior data exposure. Review effects, intervals, mouse/fold "
        "support and controls together, then narrow for matched video. WT DA is another supporting axis; no "
        "strong DA signal or significant group result is required of every useful generator. Arena-zone geometry, "
        "prospective forecasting and targeted boundary/surrogate checks remain separate from this completed round.\n")
    atomic_write_text(output/"scorecard.md","".join(parts))
    return dict(settings=len(trials),general_contrast_rows=len(general))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.output)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
