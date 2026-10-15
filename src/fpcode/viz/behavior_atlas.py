"""Reusable static plots and portable HTML for behavioural comparison atlases."""

import base64
import html
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def timecourse_panels(panel, features, *, conditions, groups, output, title, feature_labels=None):
    """One point per animal/wave; means and descriptive animal-level SEM."""
    fig, axes = plt.subplots(int(np.ceil(len(features) / 3)), 3, figsize=(15, 3.3 * int(np.ceil(len(features) / 3))),
                             squeeze=False, constrained_layout=True)
    colors = ("#246b9c", "#c34b36")
    for ax, (family, feature) in zip(axes.flat, features):
        sub = panel[(panel.family == family) & (panel.feature == feature)]
        for offset, (group, color) in enumerate(zip(groups, colors)):
            means, sems = [], []
            for i, condition in enumerate(conditions):
                values = sub.loc[(sub.group == group) & (sub.condition == condition), "value"].dropna()
                x = i + (offset - .5) * .12
                ax.scatter(np.full(len(values), x), values, s=15, alpha=.35, color=color)
                means.append(values.mean())
                sems.append(values.sem())
            ax.errorbar(np.arange(len(conditions)) + (offset - .5) * .12, means, yerr=sems,
                        color=color, marker="o", linewidth=1.4, label=group)
        caption = (feature_labels or {}).get((family, feature), feature.replace("_", " "))
        ax.set_title(textwrap.fill(caption, width=58), fontsize=9)
        ax.set_xticks(range(len(conditions)), conditions, rotation=25, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, output)


def syllable_usage_heatmap(contrasts, annotations, *, conditions, output, difference_label="Second minus first group"):
    sub = contrasts[(contrasts.family == "syllable_usage") & contrasts.feature.str.endswith("/time_fraction")]
    matrix = sub.pivot(index="feature", columns="view", values="difference").reindex(columns=conditions)
    matrix.index = matrix.index.str.split("/").str[0].astype(int)
    matrix = matrix.reindex(annotations.syllable_id.sort_values()) * 100
    names = annotations.set_index("syllable_id").description
    labels = [f"S{i:02d} {names.get(i, '')}" for i in matrix.index]
    fig, ax = plt.subplots(figsize=(10, 23), constrained_layout=True)
    limit = max(float(np.nanquantile(abs(matrix.to_numpy()), .97)), .01)
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(conditions)), conditions, rotation=25)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title(difference_label + ": syllable time fraction (percentage points)")
    fig.colorbar(im, ax=ax, label="Percentage-point difference; colour clipped at 97th percentile", shrink=.4)
    _save(fig, output)


def factor_templates(w, annotations, *, output):
    fig, axes = plt.subplots(int(np.ceil(w.shape[1] / 2)), 2,
                             figsize=(12, 3 * int(np.ceil(w.shape[1] / 2))), constrained_layout=True, squeeze=False)
    names = annotations.set_index("syllable_id").description
    for k, ax in enumerate(axes.flat):
        if k >= w.shape[1]:
            ax.set_visible(False)
            continue
        strongest = np.argsort(w[:, k, :].sum(axis=1))[-12:][::-1]
        ax.imshow(w[strongest, k, :], aspect="auto", cmap="magma")
        ax.set_yticks(range(len(strongest)), [f"S{i:02d} {names.get(i, '')}" for i in strongest], fontsize=8)
        ax.set_xticks(range(w.shape[2]), [str(i + 1) for i in range(w.shape[2])])
        ax.set_title(f"seqNMF factor {k:02d}")
        ax.set_xlabel("Bout position")
    _save(fig, output)


def effect_forest(table, *, output, title, limit=20):
    """Original-unit estimates and confidence intervals, with per-row units."""
    table = table.head(limit).iloc[::-1]
    fig, axes = plt.subplots(len(table), 1, figsize=(11, max(4, .55 * len(table))), constrained_layout=True)
    for ax, row in zip(np.atleast_1d(axes), table.itertuples()):
        ax.axvline(0, color="#aaa", linewidth=.8)
        ax.plot([row.ci_low, row.ci_high], [0, 0], color="#246b9c", linewidth=2)
        ax.scatter([row.difference], [0], color="#c34b36", s=25)
        ax.set_title(getattr(row, "label", row.feature), fontsize=8, loc="left")
        ax.set_yticks([])
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(title)
    _save(fig, output)


def syllable_profiles(table, syllables, *, groups, output, title, coordinate="phase", conditions=None):
    """Mean curves after averaging sessions within each animal/wave."""
    if conditions is not None:
        table = table[table.condition.isin(conditions)]
    table = table[(table.n_bouts >= 10) & (table.feature == "smooth_speed_px_s")]
    animal = table.groupby(["animal_id", "group", "condition", "syllable_id", coordinate], as_index=False)["mean"].mean()
    animal = animal.groupby(["animal_id", "group", "syllable_id", coordinate], as_index=False)["mean"].mean()
    fig, axes = plt.subplots(int(np.ceil(len(syllables) / 3)), 3, figsize=(14, 3 * int(np.ceil(len(syllables) / 3))),
                             constrained_layout=True, squeeze=False)
    for ax, syllable in zip(axes.flat, syllables):
        for group, color in zip(groups, ("#246b9c", "#c34b36")):
            sub = animal[(animal.syllable_id == syllable) & (animal.group == group)]
            average = sub.groupby(coordinate)["mean"].agg(["mean", "sem"])
            x, y, se = average.index.to_numpy(), average["mean"].to_numpy(), average["sem"].to_numpy()
            ax.plot(x, y, color=color, label=group)
            ax.fill_between(x, y - se, y + se, alpha=.15, color=color)
        ax.set_title(f"S{syllable:02d}")
        ax.set_xlabel("Normalised bout phase" if coordinate == "phase" else "Seconds from bout onset")
        ax.set_ylabel("Speed (px/s)")
    for ax in list(axes.flat)[len(syllables):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, output)


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def portable_html(title, paragraphs, tables, figures, links, *, output):
    """Embed generated PNGs so the report can be opened or shared as one file."""
    content = [f"<h1>{html.escape(title)}</h1>"]
    content += [f"<p>{html.escape(text)}</p>" for text in paragraphs]
    for caption, table in tables:
        content += [f"<h2>{html.escape(caption)}</h2>", table.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4g}")]
    for caption, path in figures:
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        content += [f"<h2>{html.escape(caption)}</h2>", f'<img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">']
    content += ["<h2>Data tables</h2><ul>"]
    content += [f'<li><a href="{html.escape(url, quote=True)}">{html.escape(name)}</a></li>' for name, url in links]
    content += ["</ul>"]
    style = "body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 25px;color:#203040}p{line-height:1.65}img{max-width:100%}table{font-size:12px;border-collapse:collapse;width:100%;overflow:auto;display:block}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}h2{margin-top:40px}a{color:#246b9c}"
    Path(output).write_text("<!doctype html><html><head><meta charset='utf-8'><title>" + html.escape(title)
                           + "</title><style>" + style + "</style></head><body>" + "\n".join(content) + "</body></html>")
