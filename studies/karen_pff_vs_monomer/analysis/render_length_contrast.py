"""Rank and render saved five-bout fits, then build length-controlled review pages."""

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def prepare(args):
    import numpy as np
    import pandas as pd

    from studies.sequence_viz.recreate_generator_videos import (
        ClockCache,
        Tracks,
        execution_descriptors,
        rank_exemplar_neighbors,
        sample_indexes,
        syllable_segments,
    )
    from studies.sequence_viz.render_generator_review_grids import select_movie_sets

    from .render_reference_review import verify_call
    from .render_six_panel_review import guard
    from .run_reference_seqnmf import digest, log, write_json

    source = json.loads((args.source / "manifest.json").read_text())
    scale = json.loads((args.source.parent / "selected_ranking_recipe.json").read_text())
    mapping = np.asarray(json.loads((args.fits / "mapping.json").read_text()), int)
    plan = dict(families=[], bindings=source["bindings"], track_rows=source["track_rows"],
                annotation_mappings=source["annotation_mappings"], annotation_source=source["annotation_source"],
                modes=["p100", "p80", "random"], panels=6, rows=2, columns=3, fps=60, context_sec_each_side=.5,
                labels_key="karen-L5-catalogue-20261019-v1", legacy_labels_key="karen-pooled-p80-20261015-v1",
                fit_design="pooled99recordings/21mice, grouped69, L5/K10/K18/lambda.3/seed43/80updates",
                source_hashes={str(Path(__file__).resolve()): digest(__file__)})
    tables, vectors, records = {}, [], []
    unavailable = []
    for k in (10, 18):
        t = pd.read_csv(args.fits / f"K{k}/pooled_calls.csv.gz", dtype={"animal_id": str})
        t["source_index"], t["K"] = np.arange(len(t)), k
        tables[k] = t
    tracks, clocks = Tracks(plan["track_rows"]), ClockCache()
    for number, (sid, binding) in enumerate(plan["bindings"].items(), 1):
        guard(args)
        track, clock = tracks.get(sid), clocks.get(binding)
        for k, table in tables.items():
            calls = []
            for row in table[table.session_id == sid].to_dict("records"):
                row.update(recording=sid, subject=str(row["animal_id"]), stop_frame=int(row["end_frame_exclusive"]),
                           case=f"L5_K{k}")
                if row["stop_frame"] >= len(clock):
                    unavailable.append(dict(K=k, source_index=row["source_index"], reason="encoded video endpoint unavailable"))
                    continue
                verify_call(track, row, mapping)
                a, b = np.searchsorted(track["frames"], [row["start_frame"], row["stop_frame"]])
                if not np.isfinite(track["latent"][a:b]).all():
                    raise ValueError("call lacks full latent support")
                calls.append(row)
            if calls:
                values, okay, _ = execution_descriptors(track, clock, calls, dict(latent=np.ones(10), path=1.))
                if not okay.all():
                    raise ValueError("descriptor support unavailable")
                values[:, :320] /= np.tile(scale["latent_iqr_native_units"], 32)
                values[:, 320:384] /= scale["displacement_px_p75"]
                values[:, -1] /= scale["log_duration_iqr"]
                vectors.append(values)
                records.extend(calls)
        log(f"L5 RANK INPUT {number}/99 {sid}")
    table = pd.DataFrame(records)
    values = np.concatenate(vectors)
    table.to_csv(args.output / "eligible_calls.csv.gz", index=False)
    pd.DataFrame(unavailable).to_csv(args.output / "unavailable_calls.csv", index=False)
    unrendered = []
    for (k, factor), sub in table.groupby(["K", "factor"], sort=True):
        if len(sub) < 6:
            unrendered.append(dict(K=int(k), factor=int(factor), eligible_calls=len(sub), reason="fewer than six unique executions"))
            continue
        d, pct, exemplar = rank_exemplar_neighbors(values[sub.index], sub.subject.to_numpy())
        stem = f"L5_K{k}_F{factor:02d}"
        seed = int(hashlib.sha256(("20261019:" + stem).encode()).hexdigest()[:8], 16)
        examples = select_movie_sets(sub.to_dict("records"), d, pct, seed, exemplar, panels=6)
        counts = tables[k].groupby("factor").size().sort_values(ascending=False, kind="stable")
        fit = json.loads((args.fits / f"K{k}/result.json").read_text())
        plan["families"].append(dict(stem=stem, cohort="karen", K=int(k), factor=int(factor),
                                     trial=f"L5_K{k}_seed43", model_label=f"K{k} · 5 bouts · corrected ranking",
                                     eligible_calls=len(sub), eligible_mice=int(sub.subject.nunique()),
                                     saved_calls=int(counts.loc[factor]), occurrence_rank=counts.index.tolist().index(factor) + 1,
                                     crop_native_pixels=320, examples_by_mode=examples,
                                     artifacts=dict(folder=str((args.fits / f"K{k}").resolve()),
                                                    calls_sha256=fit["calls_sha256"], model_sha256=fit["model_sha256"])))
    write_json(args.output / "unrendered_factors.json", unrendered)
    selected = defaultdict(list)
    for f in plan["families"]:
        for examples in f["examples_by_mode"].values():
            for e in examples:
                e["filename"] = f"components/{f['stem']}_{e['role']}.mp4"
                selected[e["recording"]].append((f, e))
    extents = defaultdict(lambda: 160.)
    for sid, examples in selected.items():
        guard(args)
        track, clock = tracks.get(sid), clocks.get(plan["bindings"][sid])
        for f, e in examples:
            a = int(np.searchsorted(track["frames"], e["start_frame"]))
            e["onset_centroid"], e["onset_heading"] = track["centroid"][a].tolist(), float(track["heading"][a])
            e["segments"] = syllable_segments(track, clock, e["start_frame"], e["stop_frame"])
            indexes, _ = sample_indexes(clock, e["start_frame"], e["stop_frame"], fps=60)
            pos = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
            known = (track["frames"][pos] == indexes) & np.isfinite(track["centroid"][pos]).all(axis=1)
            xy = track["centroid"][pos[known]] - e["onset_centroid"]
            h = e["onset_heading"]
            rot = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
            extents[f["stem"]] = max(extents[f["stem"]], float(np.max(np.abs(xy @ rot.T))) + 55)
            e["frames"] = len(indexes)
    for f in plan["families"]:
        f["crop_native_pixels"] = math.ceil(2 * extents[f["stem"]] / 20) * 20
    write_json(args.output / "manifest.json", plan)
    log(f"L5 RANK COMPLETE: {len(table)} eligible calls, {len(plan['families'])} rendered families, {len(unrendered)} too-small factors")
    return plan


def render(args, plan):
    from .render_six_panel_review import guard, init_worker, render_family, write_gallery
    from .run_reference_seqnmf import log, write_json
    results = []
    for batch in (plan["families"][:1], plan["families"][1:]):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(plan, str(args.output))) as pool:
            pending = [pool.submit(render_family, f) for f in batch]
            for future in as_completed(pending):
                guard(args)
                results.extend(future.result())
                write_gallery(args.output, plan, results)
                log(f"L5 MOVIES {len(results)}/{len(plan['families']) * 3}")
        if len(results) == 3:
            write_json(args.output / "preview.json", dict(status="PASS", first_family=plan["families"][0]["stem"],
                                                          movies=3, full_native_decode_and_encoded_frame_counts=True))
    write_json(args.output / "status.json", dict(state="COMPLETE", movies=len(results),
                                                execution_positions=6 * len(results)))


def package_reviews(args, plan):
    from .build_core_review import build
    from .run_reference_seqnmf import write_json
    parent = json.loads((args.source / "manifest.json").read_text())
    prior = json.loads(args.labels.read_text())["labels"]
    for k in (10, 18):
        output = args.output.parent / f"length_review_K{k}_20261019"
        output.mkdir(exist_ok=True)
        combined = {key: copy.deepcopy(parent[key]) for key in ("panels", "rows", "columns", "modes", "annotation_mappings")}
        combined.update(families=[], labels_key=f"karen-length-K{k}-movies-20261019-v1",
                        generator_review_key=f"karen-length-K{k}-review-20261019-v1", reference_labels={},
                        comparison_anchors=[f"L3_K{k}_F{factor:02d}" for factor in ((9, 2) if k == 10 else (14, 16))],
                        generator_review_description=f"K{k}: compare three-bout (A) and five-bout (B) pooled seqNMF generators. Same grouped69 inputs, λ.3, seed43,80updates and corrected ranking; no held-out or equal-precision claim.")
        for length, source, families in ((3, args.source, parent["families"]), (5, args.output, plan["families"])):
            for original in families:
                if original["K"] != k:
                    continue
                f = copy.deepcopy(original)
                old_stem = f["stem"]
                f["stem"] = f"L{length}_K{k}_F{f['factor']:02d}"
                f["trial"] = f"L{length}_K{k}_seed43"
                for mode, examples in f["examples_by_mode"].items():
                    movie = json.loads((source / f"{old_stem}_{mode}_grid.json").read_text())
                    movie["family"] = f["stem"]
                    movie["filename"] = str((source / movie["filename"]).resolve())
                    for example, clip in zip(examples, movie["clips"]):
                        clip["filename"] = str((source / clip["filename"]).resolve())
                        example["filename"] = clip["filename"]
                    write_json(output / f"{f['stem']}_{mode}_grid.json", movie)
                    if length == 3 and old_stem + "::" + mode in prior:
                        combined["reference_labels"][f["stem"] + "::" + mode] = prior[old_stem + "::" + mode]
                combined["families"].append(f)
        write_json(output / "manifest.json", combined)
        build(output, output, view="generator")
    hub = args.output.parent / "length_comparison_20261019"
    hub.mkdir(exist_ok=True)
    (hub / "index.html").write_text('''<!doctype html><meta charset="utf-8"><title>Karen generator review</title>
<style>body{font:19px system-ui;max-width:900px;margin:50px auto;padding:20px;background:#15181e;color:#eee}a{color:#add1ff}p{line-height:1.6}</style>
<h1>Which generator produces useful chains?</h1><p>New controlled template-length comparisons are ready. Start with K10; K18 is a second capacity check. Existing notes and movies remain separate.</p>
<p><a href="../length_review_K10_20261019/">Compare 3 versus 5 bouts · K10</a></p>
<p><a href="../length_review_K18_20261019/">Compare 3 versus 5 bouts · K18</a></p>
<p><a href="../generator_review_20261019/">Earlier capacity comparison · K10 versus K18, both 3 bouts</a></p>
<p>Use the overall generator judgements: behavioural variety, coherence, splitting/merging and useful variants. Random, P80 and p100 six-panel examples are available. Export feedback from each page.</p>
<p>Detection thresholds use the same normalized policy; the minimum discrete stage match differs with length. Coverage and sampling context are supporting evidence, not a claim that longer templates are better.</p>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fits", "output", "labels"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=6.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(source=str(args.source), fits=str(args.fits), output=str(args.output), execute=False)))
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "components").mkdir(exist_ok=True)
    plan = (json.loads((args.output / "manifest.json").read_text()) if (args.output / "manifest.json").exists()
            else prepare(args))
    render(args, plan)
    package_reviews(args, plan)


if __name__ == "__main__":
    main()
