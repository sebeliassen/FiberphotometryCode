"""Build a local qualitative labelling page from existing verified grid movies."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def merge_intervals(intervals):
    """Half-open native-frame union, retaining recording separation in the caller."""
    result = []
    for start, stop in sorted(intervals):
        if stop <= start:
            raise ValueError("invalid native interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def interval_intersection(first, second):
    total, i, j = 0, 0, 0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return total


def generator_comparisons(families, anchors=None):
    """Suggest comparison witnesses from exact native-time overlap, without fitting."""
    generators, intervals = [], {}
    trials = sorted({f["trial"] for f in families}, key=lambda t: next(
        (f["K"], f["bout_length"], t) for f in families if f["trial"] == t))
    if len(trials) != 2:
        raise ValueError("this comparison view requires two generator settings")
    for trial in trials:
        own = [f for f in families if f["trial"] == trial]
        artifacts = own[0]["artifacts"]
        folder = Path(artifacts["folder"])
        call_path, model_path = folder / "pooled_calls.csv.gz", folder / "model.npz"
        if digest(call_path) != artifacts["calls_sha256"] or digest(model_path) != artifacts["model_sha256"]:
            raise ValueError("generator payload changed")
        lookup = {f["factor"]: f["stem"] for f in own}
        for stem in lookup.values():
            intervals[stem] = {}
        count, unrendered = 0, {}
        with gzip.open(call_path, "rt") as stream:
            for row in csv.DictReader(stream):
                factor = int(row["factor"])
                count += 1
                if factor not in lookup:
                    unrendered[factor] = unrendered.get(factor, 0) + 1
                    continue
                stem = lookup[factor]
                intervals[stem].setdefault(row["session_id"], []).append(
                    (int(row["start_frame"]), int(row["end_frame_exclusive"])))
        for f in own:
            intervals[f["stem"]] = {sid: merge_intervals(rows) for sid, rows in intervals[f["stem"]].items()}
        result = json.loads((folder / "result.json").read_text())
        if result["fit_calls"] != count or result["model_sha256"] != artifacts["model_sha256"]:
            raise ValueError("generator result identity differs")
        generators.append(dict(id=trial, label=f"K{own[0]['K']} · {own[0]['bout_length']} bouts", K=own[0]["K"],
                               length=own[0]["bout_length"], calls=count, families=len(own),
                               penalty=result["penalty"], seed=result["seed"], iterations=result["iterations"],
                               union_coverage=result["fit_union_time_coverage"], artifacts=artifacts,
                               unrendered_factors=[dict(factor=k, calls=n) for k, n in sorted(unrendered.items())]))
    first = [f for f in families if f["trial"] == trials[0]]
    second = [f for f in families if f["trial"] == trials[1]]
    lengths = {stem: sum(stop - start for rows in sessions.values() for start, stop in rows)
               for stem, sessions in intervals.items()}
    overlaps = []
    for a in first:
        for b in second:
            aa, bb = intervals[a["stem"]], intervals[b["stem"]]
            common = sum(interval_intersection(aa[sid], bb[sid]) for sid in aa.keys() & bb.keys())
            union = lengths[a["stem"]] + lengths[b["stem"]] - common
            overlaps.append(dict(a=a["stem"], b=b["stem"], native_time_jaccard=common / union if union else 0.))
    def best_a(stem):
        return max((p for p in overlaps if p["a"] == stem), key=lambda p: (p["native_time_jaccard"], p["b"]))
    def best_b(stem):
        return max((p for p in overlaps if p["b"] == stem), key=lambda p: (p["native_time_jaccard"], p["a"]))
    pairs = []
    def add(pair, purpose, recommended):
        if any(p["a"] == pair["a"] and p["b"] == pair["b"] for p in pairs):
            return
        pairs.append(dict(pair, id="comparison_" + str(len(pairs) + 1), purpose=purpose, recommended=recommended))
    if anchors is None:
        anchors = [s for s in ("K10_F09", "K10_F02") if any(f["stem"] == s for f in first)]
        if not anchors:
            anchors = [f["stem"] for f in sorted(first, key=lambda f: f["occurrence_rank"])[:2]]
    for stem in anchors:
        add(best_a(stem), "Compare these candidate routines: are their behaviours related, and which generator handles them better?", True)
    # The frequent larger-catalogue family with the weakest best time overlap is
    # an added-diversity witness, not an assertion of semantic novelty.
    candidates = [best_b(f["stem"]) for f in second if f["occurrence_rank"] <= 10]
    extra = min(candidates, key=lambda p: (p["native_time_jaccard"], p["b"]))
    add(extra, "Check whether generator B adds a useful routine or another version of an existing one.", True)
    if any(f["stem"] == "K18_F16" for f in second):
        add(best_b("K18_F16"), "Optional: another ambiguous family and its closest time-overlap counterpart.", False)
    if any(f["stem"] == "K10_F06" for f in first):
        add(best_a("K10_F06"), "Optional: compare execution variation in another part of the catalogue.", False)
    return generators, pairs, overlaps


def build(source, output, reference_labels=None, view="core"):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["panels"] != 6 or manifest["rows"] != 2 or manifest["columns"] != 3:
        raise ValueError("review requires six-panel 2x3 grids")
    references = manifest.get("reference_labels", {})
    if reference_labels:
        incoming = json.loads(reference_labels.read_text())
        if incoming["package"] != manifest["labels_key"]:
            raise ValueError("reference labels do not match source review package")
        references = incoming["labels"]
    families, movies, bindings = [], 0, {}
    for original in manifest["families"]:
        family = dict(original, movies={})
        lengths = {int(e["end_exclusive"]) - int(e["start"])
                   for examples in original["examples_by_mode"].values() for e in examples}
        if len(lengths) != 1:
            raise ValueError("ambiguous template length for " + original["stem"])
        family["bout_length"] = lengths.pop()
        for mode in manifest["modes"]:
            metadata = source / f"{original['stem']}_{mode}_grid.json"
            movie = json.loads(metadata.read_text())
            path = source / movie["filename"]
            if movie["family"] != original["stem"] or movie["selection_mode"] != mode:
                raise ValueError("movie metadata identity mismatch")
            if digest(path) != movie["sha256"]:
                raise ValueError("movie content changed: " + str(path))
            examples = original["examples_by_mode"][mode]
            if len(examples) != 6 or len(movie["clips"]) != 6:
                raise ValueError("missing execution panels")
            item = dict(movie, filename=os.path.relpath(path, output))
            item["poster"] = os.path.relpath(path.with_suffix(".png"), output)
            item["panels"] = []
            for number, (example, clip) in enumerate(zip(examples, movie["clips"]), 1):
                if example["panel_number"] != number or example["filename"] != clip["filename"]:
                    raise ValueError("source panel order differs from grid")
                item["panels"].append(dict(
                    number=number, recording=example["recording"], subject=str(example["subject"]),
                    start_frame=example["start_frame"], stop_frame=example["stop_frame"],
                    source_index=example["source_index"], clip_sha256=clip["sha256"],
                    clip=os.path.relpath(source / clip["filename"], output),
                    percentile=example["actual_percentile"], segments=example["segments"]))
            key = original["stem"] + "::" + mode
            if key in references and references[key]["movie_sha256"] != movie["sha256"]:
                raise ValueError("earlier judgement belongs to another movie: " + key)
            family["movies"][mode] = item
            bindings[key] = movie["sha256"]
            movies += 1
        families.append(family)
    if references.keys() - bindings.keys():
        raise ValueError("reference labels contain unavailable movies")
    data = dict(package="karen-chain-core-review-20261016-v1", source_package=manifest["labels_key"],
                source_manifest=str(manifest_path), source_manifest_sha256=digest(manifest_path),
                families=families, reference_labels=references, focused=["K10_F09", "K10_F02", "K18_F16", "K10_F06"],
                names=manifest["annotation_mappings"]["karen"],
                notice="These are the existing three-bout Karen models with corrected legacy ranking. The proposed five-bout comparison has not been run.")
    if view == "generator":
        data["package"] = manifest.get("generator_review_key", "karen-generator-review-20261019-v1")
        data["export_filename"] = manifest.get("export_filename", (
            "karen_generator_feedback.json" if data["package"] == "karen-generator-review-20261019-v1"
            else data["package"] + ".json"))
        data["generators"], data["comparisons"], data["cross_generator_overlaps"] = generator_comparisons(families, manifest.get("comparison_anchors"))
        first, second = data["generators"]
        data["default_mode"] = manifest.get("default_mode", "p80" if first["length"] != second["length"] else "random")
        data["notice"] = manifest.get("generator_review_description", (
            f"Compare pooled seqNMF settings {first['label']} (A) and {second['label']} (B), "
            "using the same inputs and corrected ranking. The movies support a generator-level judgement, not automatic catalogue acceptance."))
    template = Path(__file__).with_name("generator_review.html" if view == "generator" else "core_review.html")
    page = template.read_text().replace("__REVIEW_DATA__", json.dumps(data).replace("<", "\\u003c"))
    target = output / "index.html"
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(target)
    provenance = dict(package=data["package"], source_package=data["source_package"],
                      source_manifest=str(manifest_path), source_manifest_sha256=data["source_manifest_sha256"],
                      families=len(families), movies=movies, grid_rows=2, grid_columns=3,
                      movie_bindings=bindings, reference_entries=len(references),
                      source_hashes={str(p.resolve()): digest(p) for p in (Path(__file__), template)},
                      html_sha256=digest(target), media_policy="Existing files referenced; no rendering, fitting or reassignment.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-labels", type=Path)
    parser.add_argument("--view", choices=("core", "generator"), default="core")
    args = parser.parse_args()
    result = build(args.source, args.output, args.reference_labels, args.view)
    print(json.dumps({k: result[k] for k in ("package", "families", "movies", "reference_entries")}))


if __name__ == "__main__":
    main()
