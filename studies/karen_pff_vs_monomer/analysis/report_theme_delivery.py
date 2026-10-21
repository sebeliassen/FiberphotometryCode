"""Compact, portable exploratory theme atlas from already computed readouts.

This builder performs no fitting, event selection, inference or video rendering.
It displays fixed themes, mouse-level summaries and every supplied statistical
result, with nominal and corrected findings kept visibly distinct.
"""

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WAVES = ("3w_PI", "8w_PI", "16w_PI", "24w_PI", "Ldopa")
GROUPS = ("Monomer", "PFF")
COLORS = {"Monomer": "#276a9b", "PFF": "#c05e29"}
WAVE_LABELS = ("3w", "8w", "16w", "24w", "L-dopa*")


def _read(path):
    return pd.read_csv(path, dtype={"animal_id": str, "session_id": str}) if path.exists() else pd.DataFrame()


def _number(value, digits=3):
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value)


def _safe(value):
    return html.escape(str(value), quote=True)


def _label(feature):
    return (str(feature).replace("__gap_1s", " (gap ≤1 s)").replace("_gap_1s", " (gap ≤1 s)")
            .replace("_", " ").replace(" px s", " px/s").replace(" rad s", " rad/s"))


def _token_names(library):
    mapping = library.get("token_names", library.get("mapping", {}).get("tokens", {}))
    return {str(key): value.get("name", str(key)) if isinstance(value, dict) else str(value)
            for key, value in mapping.items()}


def _stages(theme, names):
    labels = theme.get("stage_labels")
    if not labels:
        labels = [[names.get(str(token), f"token {token}") for token in stage] for stage in theme["stage_sets"]]
    return " → ".join(" / ".join(stage) if isinstance(stage, list) else str(stage) for stage in labels)


def _choose_feature(table, priorities, *, reject=()):
    if table.empty:
        return None
    available = sorted(str(x) for x in table.feature.unique())
    for pattern in priorities:
        matches = [feature for feature in available if re.search(pattern, feature)
                   and not any(re.search(bad, feature) for bad in reject)]
        if matches:
            return matches[0]
    return None


def _primary_gap(table):
    if table.empty:
        return table
    features = table.feature.astype(str)
    candidates = features.str.contains(r"(?:gap[_=:]?|_g)1(?:\.0)?(?:s|$|_)", regex=True)
    has_gap_variant = features.str.contains(r"gap|_g\d", case=False, regex=True)
    if candidates.any():
        return table[candidates | ~has_gap_variant]
    # Unsuffixed inputs are allowed; avoid silently selecting 0/2-second variants.
    return table[~has_gap_variant]


def _save(figure, directory, stem, caption, figures):
    figure.savefig(directory / f"{stem}.png", dpi=150, bbox_inches="tight", facecolor="white")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    figures.append({"stem": stem, "caption": caption})


def _points(axis, table, *, positions, lookup, xlabel=True):
    """All dots are animal means; error bars are descriptive SEM across animals."""
    for group_index, group in enumerate(GROUPS):
        offset = (-.15, .15)[group_index]
        for position in positions:
            sub = table[(table.group == group) & (table[lookup] == position)].dropna(subset=["value"])
            x = positions.index(position) + offset
            jitter = np.array([int(hashlib.sha256(str(animal).encode()).hexdigest()[:4], 16) / 65535 - .5
                               for animal in sub.animal_id]) * .18
            axis.scatter(x + jitter, sub.value, s=14, color=COLORS[group], alpha=.5, linewidths=0)
            if len(sub):
                mean = sub.value.mean()
                sem = sub.value.std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.
                axis.errorbar(x, mean, yerr=sem, fmt="o", color=COLORS[group], ms=5, capsize=3, zorder=5)
            axis.text(x, -.13, str(len(sub)), transform=axis.get_xaxis_transform(), color=COLORS[group],
                      ha="center", va="top", fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", alpha=.6, linewidth=.5)
    axis.set_xticks(range(len(positions)))
    if xlabel:
        axis.set_xticklabels(WAVE_LABELS if positions == list(WAVES) else positions, fontsize=8)
    else:
        axis.set_xticklabels([])


def _recruitment(panel, themes, directory, figures, plt):
    rate = _choose_feature(panel, [r"^core_calls_per_min$", r"^calls_per_min$", r"^occurrences_per_min$",
                                  r"calls_per_min", r"events_per_min", r"occurrences_per_min"])
    coverage = _choose_feature(panel, [r"^union_time_fraction$", r"^union_coverage_fraction$", r"^coverage_fraction$", r"union.*coverage",
                                      r"coverage_fraction"])
    features = [feature for feature in [rate, coverage] if feature]
    if not features:
        return
    figure, axes = plt.subplots(len(themes), len(features), figsize=(7 * len(features), 2.5 * len(themes)),
                                squeeze=False, constrained_layout=True)
    for row, theme in enumerate(themes):
        for col, feature in enumerate(features):
            sub = panel[(panel.theme == theme["id"]) & (panel.feature == feature)]
            _points(axes[row, col], sub, positions=list(WAVES), lookup="wave")
            axes[row, col].set_title(theme["name"], loc="left", fontsize=11)
            axes[row, col].set_ylabel(_label(feature), fontsize=8)
    figure.suptitle("Theme recruitment · each dot is one mouse-wave; mean ± SEM", fontsize=13)
    _save(figure, directory, "recruitment", "Recruitment across waves. Blue Monomer; orange PFF. "
          "Numbers below each group are finite mice. L-dopa combines age and treatment. "
          "Theme occurrences may overlap; coverage uses interval unions, not summed durations.", figures)
    plt.close(figure)


def _intra(pooled, themes, directory, figures, plt):
    pooled = pooled[pooled.scope == "pooled_untreated"]
    features = [_choose_feature(pooled, [r"^mean_duration_s$", r"^duration_s$", r"mean.*duration"],
                                reject=[r"stage", r"exclusive"]),
                _choose_feature(pooled, [r"^mean_speed_px_s$", r"^mean_speed", r"speed"],
                                reject=[r"preceding", r"outside", r"inside_minus"]),
                _choose_feature(pooled, [r"^mean_abs_angular_velocity_rad_s$", r"angular_velocity"],
                                reject=[r"preceding", r"outside", r"inside_minus"])]
    features = [feature for feature in features if feature]
    if not features:
        return
    figure, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 4.2),
                                squeeze=False, constrained_layout=True)
    ids, names = [t["id"] for t in themes], [t["name"].replace("–", "–\n", 1) for t in themes]
    for col, feature in enumerate(features):
        axis = axes[0, col]
        _points(axis, pooled[pooled.feature == feature], positions=ids, lookup="theme")
        axis.set_xticklabels(names, fontsize=8)
        axis.set_title(_label(feature), fontsize=11)
    figure.suptitle("Within-theme execution · equal untreated waves within each mouse", fontsize=13)
    _save(figure, directory, "intra_execution", "Execution summaries before viewing selected examples. "
          "Mouse means receive equal weight; raw occurrence-level kinematics remain downloadable. "
          "Speed/displacement are pixels, without a claimed physical calibration.", figures)
    plt.close(figure)


def _stage_timing(panel, themes, directory, figures, plt):
    table = panel[panel.feature.str.match(r"^stage\d+_duration_s$")]
    if table.empty:
        return
    table = table.groupby(["animal_id", "group", "theme", "feature"], as_index=False).value.mean()
    figure, axes = plt.subplots(1, len(themes), figsize=(3.4 * len(themes), 3.8),
                                squeeze=False, constrained_layout=True)
    for index, theme in enumerate(themes):
        own = table[table.theme == theme["id"]]
        positions = sorted(own.feature.unique())
        _points(axes[0, index], own, positions=positions, lookup="feature")
        axes[0, index].set_xticklabels([str(i + 1) for i in range(len(positions))])
        axes[0, index].set_xlabel("Ordered stage")
        axes[0, index].set_title(theme["name"].replace("–", "–\n", 1), fontsize=10)
        axes[0, index].set_ylabel("Mean stage duration (s)")
    _save(figure, directory, "stage_timing", "Stage timing: equal available untreated waves within mouse; "
          "all supported mice shown. Stage identities are given in the theme cards.", figures)
    plt.close(figure)


def _transition_matrix(pooled, themes, directory, figures, plt):
    table = _primary_gap(pooled[(pooled.scope == "pooled_untreated")
                               & pooled.feature.str.startswith("transition_to:")])
    if table.empty:
        return
    ids = [t["id"] for t in themes]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    for axis, group in zip(axes, GROUPS):
        matrix = np.full((len(ids), len(ids)), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        for i, source in enumerate(ids):
            for j, target in enumerate(ids):
                own = table[(table.group == group) & (table.theme == source)
                            & table.feature.str.startswith("transition_to:" + target)]
                values = own.value.dropna()
                matrix[i, j], counts[i, j] = values.mean(), len(values)
        image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="YlGnBu")
        for i, j in np.ndindex(matrix.shape):
            axis.text(j, i, f"{matrix[i, j]:.2f}\nn={counts[i, j]}" if np.isfinite(matrix[i, j]) else "—",
                      ha="center", va="center", fontsize=8,
                      color="white" if matrix[i, j] > .55 else "black")
        axis.set_xticks(range(len(ids)), [str(i + 1) for i in range(len(ids))])
        axis.set_yticks(range(len(ids)), [f"{i + 1}. {t['name']}" for i, t in enumerate(themes)], fontsize=8)
        axis.set_xlabel("Next theme")
        axis.set_title(group)
    figure.colorbar(image, ax=axes, label="Mean mouse transition probability", shrink=.7)
    figure.suptitle("Inter-theme transitions · exclusive projection, gap ≤1 s", fontsize=13)
    _save(figure, directory, "inter_transitions", "Transition denominator includes eligible self-transitions. "
          "Each cell is the equal-mouse mean of equal-wave probabilities, with finite mouse count. "
          "Rows need not sum to one when cells have different missingness; sparse rows are not evidence of no transition.", figures)
    plt.close(figure)


def _organisation(pooled, themes, directory, figures, plt):
    table = _primary_gap(pooled[pooled.scope == "pooled_untreated"])
    features = [_choose_feature(table, [r"^self_repeat_probability"]),
                _choose_feature(table, [r"^return2_probability"]),
                _choose_feature(table, [r"^mean_run_calls"])]
    features = [feature for feature in features if feature]
    if not features:
        return
    figure, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 4.3),
                                squeeze=False, constrained_layout=True)
    ids = [t["id"] for t in themes]
    for col, feature in enumerate(features):
        axis = axes[0, col]
        _points(axis, table[table.feature == feature], positions=ids, lookup="theme")
        axis.set_xticklabels([f"{i + 1}" for i in range(len(themes))])
        axis.set_xlabel("Theme number (see atlas)")
        axis.set_title(_label(feature), fontsize=11)
    figure.suptitle("Repetition and combinations · independent mouse summaries", fontsize=13)
    _save(figure, directory, "inter_repetition", "Exclusive, nonoverlapping events with unknown-frame barriers; "
          "primary gap ≤1 s, with 0/2 s sensitivity retained in full results. Returns A→B→A require B≠A; "
          "run summaries include singletons and explicitly handle boundary censoring.", figures)
    plt.close(figure)


def _branching(branches, themes, names, directory, figures, plt):
    if branches.empty:
        return
    outcome = next((column for column in ["next_label", "next_name", "outcome", "next_token", "token"]
                    if column in branches), None)
    if outcome is None:
        return
    table = branches.copy()
    def outcome_label(value):
        if pd.isna(value):
            return "Censored"
        try:
            token = int(float(value))
            return "Censored" if token < 0 else names.get(str(token), f"Unresolved token {token}")
        except (TypeError, ValueError):
            return str(value)

    if "next_token" in table:
        table["outcome_label"] = table.next_token.map(lambda value: "Unresolved / bad" if pd.notna(value)
                                                      and float(value) >= 20 else outcome_label(value))
    else:
        table["outcome_label"] = table[outcome].map(outcome_label)
    # Explicit block-end categories are censoring, not observed biological exits.
    censored = table.outcome_label.str.contains("block.end|censor|no.next", case=False, regex=True)
    if "right_censored" in table:
        censored |= table.right_censored.astype(str).str.lower().eq("true")
    eligible = table.loc[~censored].copy()
    if "continuation_status" in eligible:
        eligible = eligible[eligible.continuation_status != "block_end"]
    if "wave" in eligible:
        eligible = eligible[eligible.wave.isin(WAVES[:-1])]
    if eligible.empty:
        return
    if "count" not in eligible:
        eligible["count"] = 1
    totals = eligible.groupby("outcome_label")["count"].sum().sort_values(ascending=False)
    outcomes = list(totals.head(10).index)
    eligible["outcome_label"] = eligible.outcome_label.where(eligible.outcome_label.isin(outcomes), "Other observed")
    if "Other observed" in set(eligible.outcome_label):
        outcomes.append("Other observed")
    figure, axes = plt.subplots(1, 2, figsize=(max(12, len(outcomes) * 1.15), 5.5), constrained_layout=True)
    for axis, group in zip(axes, GROUPS):
        own = eligible[eligible.group == group] if "group" in eligible else eligible
        if "wave" in own:
            own = own[own.wave.isin(WAVES[:-1])]
        counts = own.groupby(["theme", "outcome_label"])["count"].sum().unstack(fill_value=0)
        matrix = np.full((len(themes), len(outcomes)), np.nan)
        nroots = []
        for i, theme in enumerate(themes):
            row = counts.loc[theme["id"]] if theme["id"] in counts.index else pd.Series(dtype=float)
            total = row.sum()
            nroots.append(int(total))
            if total:
                matrix[i] = [row.get(outcome, 0) / total for outcome in outcomes]
        image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if matrix[i, j] > .55 else "black")
        axis.set_xticks(range(len(outcomes)), outcomes, rotation=60, ha="right", fontsize=8)
        axis.set_yticks(range(len(themes)), [f"{t['name']} (roots={n})" for t, n in zip(themes, nroots)], fontsize=8)
        axis.set_title(group)
    figure.colorbar(image, ax=axes, label="Fraction of observed root continuations", shrink=.65)
    figure.suptitle("Branches after the shared root · pooled untreated descriptive counts", fontsize=13)
    _save(figure, directory, "branching", "Branch populations include every matching root with an observed "
          "continuation, including unresolved/bad outcomes. Block-end roots are censored and excluded from the "
          "continuation denominator. This count-weighted descriptive heatmap differs from mouse-level inference; "
          "it is not a plot restricted to completed theme cores.", figures)
    plt.close(figure)


def _frequency_controls(pooled, directory, figures, plt):
    table = _primary_gap(pooled[(pooled.scope == "pooled_untreated") & pooled.family.str.contains(
        r"frequency|first_order|sequence_control", regex=True)])
    features = [feature for feature in sorted(table.feature.unique()) if "excess" in feature][:6]
    if not features:
        return
    figure, axis = plt.subplots(figsize=(max(8, len(features) * 2.2), 4.5), constrained_layout=True)
    _points(axis, table[table.feature.isin(features)], positions=features, lookup="feature")
    axis.set_xticklabels([_label(feature).replace(" ", "\n", 2) for feature in features], fontsize=8)
    axis.axhline(0, color="#888888", linewidth=.8)
    axis.set_ylabel("Observed minus null-expected probability")
    axis.set_title("Sequence controls · equal untreated waves within mouse")
    _save(figure, directory, "sequence_controls", "Frequency-only expectations preserve within-block theme "
          "counts and eligible gap slots. First-order results, if supplied, use a plug-in fitted transition null; "
          "these are descriptive controls with finite surrogate precision, not independent biological validation.", figures)
    plt.close(figure)


def _primitive_controls(pooled, name_map, directory, figures, plt):
    table = pooled[(pooled.scope == "pooled_untreated") & pooled.theme.str.match(r"^__primitive_token\d+__$")]
    if table.empty:
        return
    table = table.copy()
    table["token"] = table.theme.str.extract(r"token(\d+)", expand=False).astype(int)
    table = table[table.token < 20]
    features = [_choose_feature(table, [r"^bout_rate_per_min$", r"occurrences_per_min", r"bouts_per_min", r"rate_per_min"]),
                _choose_feature(table, [r"^time_fraction$", r"time_fraction", r"coverage_fraction"])]
    features = [feature for feature in features if feature]
    if not features or table.empty:
        return
    ids = table[["theme", "token"]].drop_duplicates().sort_values("token").theme.tolist()
    figure, axes = plt.subplots(len(features), 1, figsize=(16, 4 * len(features)),
                                squeeze=False, constrained_layout=True)
    for row, feature in enumerate(features):
        axis = axes[row, 0]
        _points(axis, table[table.feature == feature], positions=ids, lookup="theme")
        axis.set_xticklabels([name_map.get(token, token).replace("Primitive: ", "") for token in ids],
                             rotation=50, ha="right", fontsize=8)
        axis.set_ylabel(_label(feature))
    figure.suptitle("Constituent usage controls · equal untreated waves within mouse", fontsize=13)
    _save(figure, directory, "primitive_controls", "Fixed set of known annotated primitive groups; no selection "
          "by effect or p-value. Native source-bout usage is a descriptive context control. It does not statistically "
          "adjust theme effects or establish causality; unresolved labels and motion baselines remain in full tables.", figures)
    plt.close(figure)


def _status(row):
    if pd.isna(row.get("p_raw")):
        return "Not tested / insufficient support"
    if row.get("q_global", 1) < .05:
        return "Global FDR < .05"
    if row.get("q_family", 1) < .05:
        return "Family FDR < .05; global fails"
    if row["p_raw"] < .05:
        return "Nominal p < .05; FDR fails"
    return "No nominal difference"


def _shortlist(inference):
    tested = inference[inference.p_raw.notna()].copy()
    tested["magnitude"] = tested.hedges_g.abs() if "hedges_g" in tested else np.nan
    selected = []
    # Keep strong supported estimates per scientific family independently of p.
    for _, own in tested[tested.contrast_type != "paired_change"].groupby("family", sort=True):
        primary = own[(own.scope == "pooled_untreated") | (own.contrast_type == "difference_in_changes")]
        if primary.empty:
            primary = own
        selected.extend(primary.sort_values("magnitude", ascending=False, na_position="last").head(2).index)
    selected.extend(tested.index[(tested.q_family < .05) & (tested.contrast_type != "paired_change")])
    selected = list(dict.fromkeys(selected))
    return tested.loc[selected].drop(columns="magnitude")


def _table(records, names, limit=None):
    records = records if limit is None else records[:limit]
    headers = ["Theme", "Readout", "Scope", "Contrast", "Effect [95% CI]", "N", "Raw p", "Family q", "Global q", "Status"]
    rows = []
    for row in records:
        effect = row.get("mean_change") if row.get("contrast_type") == "paired_change" else row.get("difference")
        effect_text = f"{_number(effect)} [{_number(row.get('ci_low'))}, {_number(row.get('ci_high'))}]"
        n = f"pairs={_number(row.get('n_pairs'))}" if row.get("contrast_type") == "paired_change" else (
            f"M={_number(row.get('n_monomer'))}; PFF={_number(row.get('n_pff'))}")
        values = [names.get(row["theme"], row["theme"]), _label(row["feature"]), row["scope"],
                  row["contrast_type"] + (" / " + row["group"] if isinstance(row.get("group"), str) else ""),
                  effect_text, n, _number(row.get("p_raw")), _number(row.get("q_family")),
                  _number(row.get("q_global")), _status(row)]
        rows.append("<tr>" + "".join(f"<td>{_safe(value)}</td>" for value in values) + "</tr>")
    return "<table><thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + (
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def build(output):
    """Build report/ from frozen definitions and finished empirical readout tables."""
    output = Path(output)
    library = json.loads((output / "library.json").read_text())
    themes = library["themes"]
    if not themes:
        raise ValueError("a theme report requires frozen theme definitions")
    sessions = _read(output / "sessions.csv")
    occurrences = _read(output / "occurrences.csv.gz")
    panel = _read(output / "statistics/mouse_readouts.csv")
    pooled = _read(output / "statistics/pooled_mouse_readouts.csv")
    inference = _read(output / "statistics/all_inference.csv")
    required = ["family", "theme", "feature", "scope", "contrast_type", "p_raw", "q_family", "q_global"]
    if panel.empty or pooled.empty or inference.empty or not set(required).issubset(inference):
        raise ValueError("completed mouse-level statistics are required before report generation")
    report = output / "report"
    figures_dir, tables_dir = report / "figures", report / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted([*output.glob("*.csv"), *output.glob("*.csv.gz"), *output.glob("*.json"),
                          *output.glob("statistics/*.csv"), *output.glob("statistics/*.csv.gz"),
                          *output.glob("statistics/*.json")]):
        prefix = "statistics_" if source.parent.name == "statistics" else ""
        destination = tables_dir / (prefix + source.name)
        shutil.copyfile(source, destination)
        copied.append({"name": destination.name, "bytes": destination.stat().st_size})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    figures = []
    names = _token_names(library)
    _recruitment(panel, themes, figures_dir, figures, plt)
    _intra(pooled, themes, figures_dir, figures, plt)
    _stage_timing(panel[panel.wave.isin(WAVES[:-1])], themes, figures_dir, figures, plt)
    _branching(_read(output / "branches.csv.gz"), themes, names, figures_dir, figures, plt)
    _transition_matrix(pooled, themes, figures_dir, figures, plt)
    _organisation(pooled, themes, figures_dir, figures, plt)
    _frequency_controls(pooled, figures_dir, figures, plt)
    finite = inference[inference.p_raw.notna()]
    counts = {"all_rows": len(inference), "tested": len(finite),
              "nominal": int((finite.p_raw < .05).sum()), "family_fdr": int((finite.q_family < .05).sum()),
              "global_fdr": int((finite.q_global < .05).sum())}
    name_map = {theme["id"]: theme["name"] for theme in themes}
    name_map["__catalogue__"] = "Whole atlas"
    name_map.update({f"__primitive_token{token}__": f"Primitive: {name}" for token, name in names.items()})
    name_map["__primitive_unresolved__"] = "Primitive: unresolved / bad"
    name_map["__unresolved_annotation__"] = "Unresolved annotation share"
    _primitive_controls(pooled, name_map, figures_dir, figures, plt)
    shortlist = _shortlist(inference)
    shortlist.to_csv(tables_dir / "effect_shortlist.csv", index=False)
    copied.append({"name": "effect_shortlist.csv", "bytes": (tables_dir / "effect_shortlist.csv").stat().st_size})
    n_mice = sessions.animal_id.nunique() if "animal_id" in sessions else occurrences.animal_id.nunique()
    n_sessions = sessions.session_id.nunique() if "session_id" in sessions else occurrences.session_id.nunique()
    cards, md_cards = [], []
    for index, theme in enumerate(themes):
        own = occurrences[occurrences.theme == theme["id"]]
        core = _stages(theme, names)
        support = theme.get("source_support", [])[:2]
        support_text = "; ".join(f"{item.get('model', item.get('trial', item.get('source', 'saved fit')))} "
                                 f"F{item.get('factor', '?')}: {item.get('supporting_calls', '?')} calls"
                                 for item in support)
        caution = theme.get("caution", "Unreviewed operational theme definition.")
        paths = theme.get("observed_paths", [])[:3]
        path_text = "<br>".join(_safe(path.get("named_path", path.get("tokens", ""))) +
                                   f" · {_safe(path.get('occurrences', '?'))} windows" for path in paths)
        cards.append(f"<article><h3>{index + 1}. {_safe(theme['name'])}</h3><p class='core'>{_safe(core)}</p>"
                     f"<p><strong>{len(own):,}</strong> complete core windows · "
                     f"{own.animal_id.nunique()} mice · {own.session_id.nunique()} recordings</p>"
                     f"<p>{_safe(theme.get('rationale', ''))}</p><p class='caution'>{_safe(caution)}</p>"
                     f"<details><summary>Observed paths and generator support</summary><p>{path_text}</p>"
                     f"<p>{_safe(support_text)}</p></details></article>")
        md_cards.extend([f"### {index + 1}. {theme['name']}", "", core, "",
                         f"{len(own):,} complete windows across {own.animal_id.nunique()} mice / "
                         f"{own.session_id.nunique()} recordings. {caution}", ""])
    summary = (f"{len(themes)} provisional themes · {n_mice} mice · {n_sessions} recordings. "
               f"{counts['tested']:,} finite primary tests: {counts['nominal']} nominal p<.05, "
               f"{counts['family_fdr']} family-BH q<.05, {counts['global_fdr']} global-BH q<.05. "
               "Counts overlap and include correlated endpoints/sensitivity analyses; they are not independent discoveries.")
    family_hits = finite[finite.q_family < .05].groupby("contrast_type").size()
    summary += (f" Family-FDR results comprise {family_hits.get('paired_change', 0)} within-group longitudinal "
                f"changes, {family_hits.get('group_difference', 0)} group differences and "
                f"{family_hits.get('difference_in_changes', 0)} group differences in change.")
    if {"wave", "group", "animal_id"}.issubset(sessions):
        wave_support = sessions.groupby(["wave", "group"]).animal_id.nunique()
        summary += (f" The 24w subset contains {wave_support.get(('24w_PI', 'Monomer'), 0)} Monomer and "
                    f"{wave_support.get(('24w_PI', 'PFF'), 0)} PFF mice; see mouse counts and complete-wave sensitivity.")
    readouts = _read(output / "readouts.csv.gz")
    opportunities = (readouts[(readouts.theme == "__catalogue__") &
                              (readouts.feature == "self_repeat_probability_g1s")]
                     if {"theme", "feature"}.issubset(readouts) else pd.DataFrame())
    if len(opportunities):
        summary += (f" Primary inter-theme succession has only {int(opportunities.denominator.sum())} "
                    f"eligible pairs across {len(opportunities)} recordings; sparse conditional effects need caution.")
    if {"theme", "feature", "numerator", "denominator"}.issubset(readouts):
        coverage = readouts[(readouts.theme == "__catalogue__") & (readouts.feature == "union_time_fraction")]
        if coverage.denominator.sum() > 0:
            summary += (f" The five-theme union covers {100 * coverage.numerator.sum() / coverage.denominator.sum():.2f}% "
                        "of total valid modelled time, counting overlaps once.")
    methods = [
        ("What was extracted", "Complete observed, human-labelled ordered cores suggested by pooled Karen "
         "seqNMF calls and prior feedback. Cores were fixed before group statistics. This is explicit theme "
         "refinement, not acceptance of every component or proof that the generator has recovered a full behavioural ontology."),
        ("What is pooled", "One source generator per setting used all 99 recordings and 21 mice, including L-dopa. "
         "No new KPMS fit or phenotype-driven generator selection. Current Karen source fits use K10/K18, "
         "L3/L5, λ=.3, 80 updates, seed43 and the frozen corrected legacy ranking. These are training-cohort descriptions."),
        ("Validity and denominators", "Native 60 Hz, half-open frame bounds. Exposure includes valid unknown-labelled "
         "frames; named event cores require known labels. Missing frames and exclusions are barriers. Recruitment "
         "retains overlapping core windows and uses union time coverage. Organisation uses a separately audited "
         "exclusive event projection; unknown frames break adjacency. Branches begin with every matching root, "
         "not only completed themes; unobserved block-end continuations are censored."),
        ("Inference", "Animals are independent units. Recording ratios pool numerators/denominators within "
         "mouse-wave; pooled untreated means weight available waves equally within mouse, with complete-four-wave "
         "sensitivity exported. Group effects are PFF−Monomer, changes follow-up−3w. Primary p-values use "
         "mouse-label partitions or paired mouse sign flips; 95% Welch/paired-t CIs are unadjusted. "
         "No nominal endpoint is removed because correction fails."),
        ("Constituent and movement controls", "Primitive-group usage and within-bout motion baselines "
         "describe the same recorded population. They help distinguish possible constituent-frequency context "
         "from sequence organisation, without fitting an adjusted causal model. Separate same-raw-syllable "
         "inside/outside comparisons match support block, five-minute period and coarse occupancy quadrant; "
         "they still describe associations rather than independent confirmation."),
        ("Multiplicity", "Family BH keys contrast type × scientific feature family × scope (paired changes also "
         "group), including all themes/endpoints. Global BH includes every finite primary group, paired and "
         "change-interaction test, including sensitivity scopes. Auxiliary t-test p-values do not define these corrections."),
        ("Primitive and movement controls", "Known constituent-group usage and time fractions are counted from the "
         "same saved bouts. Movement controls weight each finite within-raw-bout mean by its adjacent native-pair "
         "count; they omit pairs spanning raw-bout transitions and are not all-frame means. These planned controls "
         "were added before their comparisons, with all corrections recomputed jointly. They provide descriptive "
         "context, not a regression-adjusted or causal explanation of theme differences."),
        ("Limits", "L-dopa contrasts confound age and treatment. Conditional sparsity is explicit, with at least "
         "three finite mice per group for inference; predecessor contexts need at least three executions in each "
         "context per recording. Pixel kinematics are uncalibrated. These are exploratory, previously exposed "
         "training data and provisional themes; qualitative movie review and independent confirmation remain useful."),
        ("IFxDN presentation lineage", "The earlier three-theme source used 14 broad groups, K8, L5, λ=.5, "
         "100 updates, seed42. The branching replay instead used raw100/K10/L3/λ1/80/42; the quantitative "
         "slide candidate code used grouped20/K10/L3/λ.005/150/42, with the exact figure-producing version "
         "still partly unresolved. Presentation themes were refined into observed "
         "sequences and selected exemplars; their old p-values are not transplanted to Karen.")]
    figures_html = "".join(f"<figure><a href='figures/{item['stem']}.pdf'><img "
                           f"src='figures/{item['stem']}.png' alt='{_safe(item['stem'])}' loading='lazy'></a>"
                           f"<figcaption>{_safe(item['caption'])} <a href='figures/{item['stem']}.pdf'>PDF</a>"
                           "</figcaption></figure>" for item in figures)
    methods_html = "".join(f"<details><summary>{_safe(title)}</summary><p>{_safe(text)}</p></details>"
                           for title, text in methods)
    downloads = "".join(f"<li><a href='tables/{item['name']}' download>{_safe(item['name'])}</a> "
                        f"({item['bytes'] / 1024:.0f} KiB)</li>" for item in copied)
    records = inference.replace({np.nan: None}).to_dict("records")
    data = json.dumps({"rows": records, "names": name_map}, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karen exploratory theme atlas</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1300px;margin:auto;padding:24px;color:#17212b;background:#fafbfc}
h1{line-height:1.15}h2{margin-top:2.2em}p{max-width:100ch}a{color:#145e91}nav{display:flex;gap:20px;flex-wrap:wrap}
.notice{padding:18px;background:#fff4db;border-left:5px solid #bd7620}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
article{background:white;border:1px solid #d7e0e5;border-radius:8px;padding:18px}.core{font-weight:650}.caution{color:#6b4c22;font-size:.92em}
figure{margin:28px 0;background:white;border:1px solid #dde3e8;padding:12px}img{max-width:100%;height:auto}figcaption{font-size:.88em;color:#465361}
details{padding:10px 0}summary{font-weight:650;cursor:pointer}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;background:white}
th,td{border:1px solid #dce2e6;padding:7px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eaf0f4}input,select,button{font:inherit;padding:7px;margin:5px}
.controls{display:flex;gap:8px;flex-wrap:wrap}.muted{color:#5c6670}.counts{font-variant-numeric:tabular-nums}footer{margin:40px 0;color:#52606c}
</style><h1>Karen: exploratory behavioural themes</h1><p class="notice">__SUMMARY__</p>
<nav><a href="#atlas">Theme atlas</a><a href="#figures">Readouts</a><a href="#effects">Effects</a><a href="media/index.html">Six-panel movies</a><a href="#downloads">Full tables</a><a href="#methods">Methods / limits</a></nav>
<p>Start with the named cores and recruitment plots. Compare within-theme execution and inter-theme organisation, then read raw and corrected evidence together. <strong>Neither statistical significance nor a selected exemplar establishes behavioural coherence.</strong></p>
<h2 id="atlas">Frozen, provisional theme atlas</h2><div class="cards">__CARDS__</div>
<h2 id="figures">Recruitment, execution, branching and combinations</h2><p class="muted">Blue: Monomer. Orange: PFF. All finite mice shown; mean ± SEM is descriptive. Click any figure for its PDF.</p><p>Full tables also retain constituent substitutions, matched same-syllable inside/outside comparisons, predecessor-context effects, longer returns and sensitivity gaps. Search the result table by feature or scientific family to inspect these readouts.</p>__FIGURES__
<h2 id="effects">Exploratory effects with their correction status</h2><p>This first table retains up to two large standardized supported group/change effects per scientific family, independent of p-values, plus every family-FDR group/change-interaction finding. Within-group longitudinal findings remain in the full table: choose Family q &lt; .05, then search paired_change. This is an inspection shortlist, not a ranking of validated discoveries. Full results below include nonsignificant and unsupported estimates.</p>
<div class="scroll">__SHORTLIST__</div>
<h3>All supplied statistical results</h3><div class="controls"><input id="search" placeholder="Theme, readout or scope" aria-label="Search results"><select id="status"><option value="all">All results</option><option value="tested">Finite tests</option><option value="nominal">Raw p &lt; .05</option><option value="family">Family q &lt; .05</option><option value="global">Global q &lt; .05</option></select><button id="prev">Previous</button><button id="next">Next</button><span id="count" class="counts"></span></div>
<div id="results" class="scroll"></div><h2 id="methods">Scope, provenance and limits</h2>__METHODS__
<h2 id="downloads">Full data and numerical results</h2><p>All links are relative to this portable report. Raw and corrected p-values, effect sizes, CIs, mouse counts and supporting opportunities are retained.</p><ul>__DOWNLOADS__</ul>
<footer>Exploratory delivery, 21 October 2026. Theme definitions are operational and remain subject to qualitative interpretation; no overall chain-generator acceptance is claimed.</footer>
<script>const DATA=__DATA__;let page=0;const size=50,$=id=>document.getElementById(id);
function esc(x){return String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function num(x){return x===null||x===undefined?'—':typeof x==='number'?Number(x).toPrecision(3):String(x);}
function status(r){return r.p_raw===null?'Not tested / insufficient support':r.q_global<.05?'Global FDR < .05':r.q_family<.05?'Family FDR < .05; global fails':r.p_raw<.05?'Nominal p < .05; FDR fails':'No nominal difference';}
function draw(){const q=$('search').value.toLowerCase(),s=$('status').value;const rows=DATA.rows.filter(r=>[r.theme,DATA.names[r.theme],r.feature,r.scope,r.family,r.contrast_type].join(' ').toLowerCase().includes(q)&&
(s==='all'||s==='tested'&&r.p_raw!==null||s==='nominal'&&r.p_raw!==null&&r.p_raw<.05||s==='family'&&r.q_family!==null&&r.q_family<.05||s==='global'&&r.q_global!==null&&r.q_global<.05));
page=Math.max(0,Math.min(page,Math.ceil(rows.length/size)-1));const chunk=rows.slice(page*size,(page+1)*size);let out='<table><thead><tr>'+['Theme','Readout','Scope','Contrast','Effect [95% CI]','N','Raw p','Family q','Global q','Status'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
for(const r of chunk){const paired=r.contrast_type==='paired_change',effect=paired?r.mean_change:r.difference,n=paired?'pairs='+num(r.n_pairs):'M='+num(r.n_monomer)+'; PFF='+num(r.n_pff);const vals=[DATA.names[r.theme]||r.theme,r.feature,r.scope,r.contrast_type+(r.group?' / '+r.group:''),num(effect)+' ['+num(r.ci_low)+', '+num(r.ci_high)+']',n,num(r.p_raw),num(r.q_family),num(r.q_global),status(r)];out+='<tr>'+vals.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>';}
$('results').innerHTML=out+'</tbody></table>';$('count').textContent=rows.length+' rows · page '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/size));$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=rows.length;}
$('search').oninput=()=>{page=0;draw()};$('status').onchange=()=>{page=0;draw()};$('prev').onclick=()=>{page--;draw()};$('next').onclick=()=>{page++;draw()};draw();</script></html>"""
    for key, value in {"SUMMARY": _safe(summary), "CARDS": "".join(cards), "FIGURES": figures_html,
                       "SHORTLIST": _table(shortlist.to_dict("records"), name_map), "METHODS": methods_html,
                       "DOWNLOADS": downloads, "DATA": data}.items():
        page = page.replace(f"__{key}__", value)
    (report / "index.html").write_text(page)
    markdown = ["# Karen exploratory theme atlas", "", summary, "", "## Provisional themes", "", *md_cards,
                "## Readouts", ""]
    for item in figures:
        markdown.extend([f"![{item['stem']}](figures/{item['stem']}.png)", "", item["caption"], ""])
    markdown.extend(["## Supported effect inspection shortlist", "",
                     "Selected by standardized effect magnitude within scientific families, plus corrected hits. "
                     "This is exploratory inspection, not a validated-discovery ranking. Full results retain all endpoints.", "",
                     "| Theme | Readout / scope | Contrast | N | Effect [95% CI] | Raw p | Family q | Global q |",
                     "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in shortlist.head(24).to_dict("records"):
        effect = row.get("mean_change") if row.get("contrast_type") == "paired_change" else row.get("difference")
        n = (f"pairs={_number(row.get('n_pairs'))}" if row.get("contrast_type") == "paired_change"
             else f"M={_number(row.get('n_monomer'))}; PFF={_number(row.get('n_pff'))}")
        markdown.append(f"| {name_map.get(row['theme'], row['theme'])} | {row['feature']} / {row['scope']} | "
                        f"{row['contrast_type']} | {n} | "
                        f"{_number(effect)} [{_number(row.get('ci_low'))}, {_number(row.get('ci_high'))}] | "
                        f"{_number(row.get('p_raw'))} | {_number(row.get('q_family'))} | {_number(row.get('q_global'))} |")
    markdown.extend(["", "## Interpretation and methods", ""])
    for title, text in methods:
        markdown.extend([f"**{title}.** {text}", ""])
    markdown.extend(["## Complete tables", "", *[f"- [{item['name']}](tables/{item['name']})" for item in copied],
                     "", "[Six-panel movies](media/index.html).", ""])
    (report / "report.md").write_text("\n".join(markdown))
    manifest = {"counts": counts, "themes": [t["id"] for t in themes], "figures": figures,
                "copied_tables": copied, "builder": str(Path(__file__).resolve()),
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "limitations": "No new inference or theme selection; all data, significance and claims come "
                               "from frozen supplied tables. Native movie rendering is separate."}
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"report": str(report.resolve()), "index": str((report / "index.html").resolve()),
            "figures": len(figures), "counts": counts}
