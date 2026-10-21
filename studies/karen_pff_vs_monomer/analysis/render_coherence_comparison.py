"""Versioned small six-panel review before refreshing the Karen catalogue.

Six previously judged anchors: corrected ranking on frozen fits, then cleaned
pooled fits with phase/semantic ranking. Original movies and labels remain intact.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .render_reference_review import verify_call
from .render_six_panel_review import MODES, guard, init_worker, render_family, write_gallery
from .run_reference_seqnmf import log, write_json

ANCHORS = ((10, 0), (10, 3), (10, 6), (18, 11), (18, 14), (18, 16))


def prepare(args):
    import numpy as np
    import pandas as pd

    from studies.sequence_viz.recreate_generator_videos import ClockCache, Tracks, sample_indexes, syllable_segments
    from studies.sequence_viz.render_generator_review_grids import select_movie_sets
    from studies.sequence_viz.render_generator_screening import digest

    original = json.loads((args.source / "manifest.json").read_text())
    labels = json.loads(args.labels.read_text())
    feature_recipe = json.loads((args.repair / "feature_recipe.json").read_text())
    recipe = json.loads((args.repair / "fits/recipe.json").read_text())
    mapping = np.asarray(json.loads(Path(recipe["input_metadata"][3]["path"]).read_text()), int)
    ranks = pd.read_csv(args.repair / "execution_rankings.csv.gz", dtype={"subject": str})
    calls = pd.read_csv(args.repair / "eligible_calls.csv.gz", dtype={"animal_id": str, "subject": str})
    matches = pd.read_csv(args.repair / "old_to_repaired_factors.csv")
    catalogue = getattr(args, "catalogue", False)
    anchors = ANCHORS
    variants = (("ranking_correction", "frozen", "corrected_legacy"),
                ("cleaned_semantic", "clean", "phase_semantic"))
    if catalogue:
        metrics = pd.read_csv(args.repair / "family_comparison.csv")
        supported = metrics[(metrics.profile == "corrected_legacy") & metrics.case.str.startswith("clean_") & metrics.supported]
        anchors = []
        for r in supported.itertuples():
            match = matches[(matches.K == r.K) & (matches.repaired_factor == r.factor)].iloc[0]
            anchors.append((int(r.K), int(match.original_factor)))
        variants = (("cleaned_ranking_correction", "clean", "corrected_legacy"),)
    plans = []
    for name, cohort, profile in variants:
        output = args.output / name
        output.mkdir(parents=True, exist_ok=True)
        (output / "components").mkdir(exist_ok=True)
        if (output / "manifest.json").exists():
            raise ValueError("refusing to overwrite a review manifest")
        rows = copy.deepcopy(original["track_rows"])
        if cohort == "clean":
            for sid, row in rows.items():
                path = args.repair / "reviewed_view/frame_cache" / (sid + ".npz")
                previous = Path(row["source_provenance"]["source"]["path"])
                row["source_provenance"]["source"] = dict(path=str(path.resolve()), sha256=(
                    row["source_provenance"]["source"]["sha256"] if path.resolve() == previous.resolve() else digest(path)))
        manifest = dict(families=[], bindings=original["bindings"], track_rows=rows,
                        annotation_mappings=original["annotation_mappings"], annotation_source=original["annotation_source"],
                        modes=list(MODES), panels=6, rows=2, columns=3, fps=60, context_sec_each_side=.5,
                        labels_key=("karen-ranking-repair-catalogue-20261016-v1" if catalogue else
                                    "karen-coherence-repair-20261016-" + name),
                        legacy_labels_key="karen-pooled-p80-20261015-v1",
                        reference_labels_key=labels["package"], reference_labels=labels["labels"],
                        review_description=("Corrected ranking on the original fits." if cohort == "frozen" else
                                            "Cleaned pooled fits with the preferred corrected legacy ranking. This combination awaits review." if profile == "corrected_legacy" else
                                            "Cleaned pooled fits with bout-aligned posture, movement, tempo and soft known-label agreement."),
                        profile=profile, fit_state=cohort, feature_recipe=feature_recipe,
                        fit_design="pooled99recordings/all21mice/allwaves; two K settings; no phenotype selection",
                        selection="six unique executions/mode; random uniform with seed20261016 (fresh relative to original review)",
                        source_hashes={str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("render_six_panel_review.py"),
                                                                 Path(__file__).with_name("six_panel_gallery.html"))})
        for k, old_factor in anchors:
            if cohort == "frozen":
                factor, cosine = old_factor, 1.
            else:
                match = matches[(matches.K == k) & (matches.original_factor == old_factor)].iloc[0]
                factor, cosine = int(match.repaired_factor), float(match.template_cosine)
            case = f"{cohort}_K{k}"
            sub = calls[(calls.case == case) & (calls.factor == factor)]
            ranked = ranks[(ranks.case == case) & (ranks.profile == profile) & (ranks.factor == factor)]
            lookup = sub.set_index("source_index")
            ordered = [lookup.loc[r.source_index].to_dict() | dict(source_index=int(r.source_index))
                       for r in ranked.itertuples(index=False)]
            if len(ordered) < 6:
                raise ValueError("matched anchor has fewer than six eligible calls")
            distance, pct = ranked.distance.to_numpy(), ranked.percentile.to_numpy()
            exemplar = int(np.argmin(distance))
            stem = f"K{k}_F{factor:02d}"
            seed = int(hashlib.sha256(("20261016:" + f"K{k}_F{old_factor:02d}").encode()).hexdigest()[:8], 16)
            examples = select_movie_sets(ordered, distance, pct, seed, exemplar, panels=6)
            all_counts = calls[calls.case == case].groupby("factor").size().sort_values(ascending=False, kind="stable")
            trial = f"{case}_{profile}_seed43"
            manifest["families"].append(dict(stem=stem, cohort="karen", trial=trial, K=k, factor=factor,
                                                reference_family=f"K{k}_F{old_factor:02d}", matched_template_cosine=cosine,
                                                model_label=f"K{k} · " + ("ranking correction / original fit" if cohort == "frozen" else
                                                                          "cleaned fit / corrected legacy ranking" if profile == "corrected_legacy" else
                                                                          "cleaned fit / phase + semantics"),
                                                eligible_calls=len(ordered), eligible_mice=sub.animal_id.nunique(),
                                                occurrence_rank=all_counts.index.tolist().index(factor) + 1,
                                                saved_calls=int(all_counts.loc[factor]), crop_native_pixels=320,
                                                examples_by_mode=examples,
                                                artifacts=feature_recipe["datasets"][case]))
        plans.append((output, manifest))
    extents = defaultdict(lambda: 160.)
    for output, manifest in plans:
        tracks, clocks = Tracks(manifest["track_rows"]), ClockCache()
        selected = defaultdict(list)
        for family in manifest["families"]:
            old = next((f for f in original["families"] if f["stem"] == family["reference_family"]), None)
            if old is not None:
                extents[family["reference_family"]] = max(extents[family["reference_family"]], old["crop_native_pixels"] / 2)
            for mode, examples in family["examples_by_mode"].items():
                for e in examples:
                    e["filename"] = "components/" + family["stem"] + "_" + e["role"] + ".mp4"
                    selected[e["recording"]].append((family, e))
        for sid, examples in selected.items():
            guard(args)
            track, clock = tracks.get(sid), clocks.get(manifest["bindings"][sid])
            for family, e in examples:
                verify_call(track, e, mapping)
                a, b = np.searchsorted(track["frames"], [e["start_frame"], e["stop_frame"]])
                if not np.isfinite(track["latent"][a:b]).all():
                    raise ValueError("selected execution lacks full latent support")
                e["onset_centroid"], e["onset_heading"] = track["centroid"][a].tolist(), float(track["heading"][a])
                e["segments"] = syllable_segments(track, clock, e["start_frame"], e["stop_frame"])
                indexes, _ = sample_indexes(clock, e["start_frame"], e["stop_frame"], fps=60)
                pos = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
                known = (track["frames"][pos] == indexes) & np.isfinite(track["centroid"][pos]).all(axis=1)
                xy = track["centroid"][pos[known]] - e["onset_centroid"]
                h = e["onset_heading"]
                rotation = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
                extent = float(np.max(np.abs(xy @ rotation.T))) + 55
                key = family["reference_family"]
                extents[key] = max(extents[key], extent)
                e["frames"] = len(indexes)
    for output, manifest in plans:
        manifest["families"].sort(key=lambda f: (f["K"], f["occurrence_rank"]))
        for family in manifest["families"]:
            family["crop_native_pixels"] = int(16 * math.ceil(2 * extents[family["reference_family"]] / 16))
            if family["crop_native_pixels"] > 2048:
                raise ValueError("selected motion exceeds bounded fixed crop")
        write_json(output / "manifest.json", manifest)
    write_json(args.output / "comparison_recipe.json", dict(anchors=anchors, variants=variants,
                                                            matched_family_policy="Hungarian template correspondence; actual factor IDs retained",
                                                            catalogue=catalogue, full_catalogue_next=not catalogue))
    return plans


def render(args, plans):
    for output, manifest in plans:
        results = []
        # Bounded preview first, then remaining anchors on the identical recipe.
        for batch in (manifest["families"][:1], manifest["families"][1:]):
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                     initargs=(manifest, str(output))) as pool:
                pending = [pool.submit(render_family, f) for f in batch]
                for future in as_completed(pending):
                    guard(args)
                    results.extend(future.result())
                    write_gallery(output, manifest, results)
                    log(f"REVIEW {output.name} {len(results)}/{len(manifest['families']) * len(MODES)} movies")
        write_json(output / "status.json", dict(state="COMPLETE", movies=len(results),
                                                execution_panels=len(results) * 6,
                                                playback_seconds=sum(r["playback_s"] for r in results),
                                                p80_playback_seconds=sum(r["playback_s"] for r in results if r["selection_mode"] == "p80")))
    if getattr(args, "catalogue", False):
        manifest = plans[0][1]
        rows = []
        for k, factor in ANCHORS:
            old = f"K{k}_F{factor:02d}"
            matched = next(f["stem"] for f in manifest["families"] if f["reference_family"] == old)
            rows.append(f'<tr><td>{old}</td><td><a href="../coherence_repair_20261016/ranking_correction/{old}_p80_grid.mp4">Preferred original-fit example</a></td>'
                        f'<td><a href="cleaned_ranking_correction/{matched}_p80_grid.mp4">Same ranker on cleaned fit ({matched})</a></td></tr>')
        page = ('<!doctype html><meta charset="utf-8"><title>Karen ranking-repair catalogue</title>'
                '<style>body{font:18px system-ui;max-width:1200px;margin:30px auto;background:#181a20;color:#eee}a{color:#9cf}td,th{padding:12px;border:1px solid #555}</style>'
                '<h1>Catalogue with corrected legacy ranking</h1>'
                '<p><a href="cleaned_ranking_correction/">Open the catalogue and label</a> — top ten per K by default; all supported families accessible.</p>'
                '<p>The ranking-repair examples on the original fits were preferred in the small review. '
                'This catalogue applies that ranker to the cleaned models, retaining handling exclusions. '
                'Check these anchors before the wider pass; this combination has not been accepted yet.</p>'
                '<table><tr><th>Original anchor</th><th>Preferred reference</th><th>Cleaned input control</th></tr>' + ''.join(rows) + '</table>')
        (args.output / "index.html").write_text(page)
        write_json(args.output / "status.json", dict(state="COMPLETE", families=len(manifest["families"]),
                                                    new_movies=len(manifest["families"]) * len(MODES),
                                                    profile="corrected_legacy", fit_state="clean", human_review="pending"))
        return
    rows = []
    for k, factor in ANCHORS:
        old = f"K{k}_F{factor:02d}"
        matched = next(f["stem"] for f in plans[1][1]["families"] if f["reference_family"] == old)
        rows.append(f'<tr><td>{old}</td><td><a href="../six_panel_20261016/{old}_p80_grid.mp4">Original P80</a></td>'
                    f'<td><a href="ranking_correction/{old}_p80_grid.mp4">Corrected ranking P80</a></td>'
                    f'<td><a href="cleaned_semantic/{matched}_p80_grid.mp4">Cleaned / phase + semantics P80 ({matched})</a></td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Karen coherence repair</title>'
            '<style>body{font:18px system-ui;max-width:1200px;margin:30px auto;background:#181a20;color:#eee}a{color:#9cf}td,th{padding:12px;border:1px solid #555}</style>'
            '<h1>Small comparison before a refreshed catalogue review</h1>'
            '<p>Start with the six P80 anchors. Use p100 and fresh uniform random examples where uncertainty remains. '
            'The original videos and labels stay available; revised ratings are separate.</p>'
            '<p><a href="ranking_correction/">Corrected ranking on original fits — label here</a> · '
            '<a href="cleaned_semantic/">Cleaned pooled fits / phase + semantics — label here</a></p>'
            '<table><tr><th>Original anchor</th><th>Original</th><th>Ranking repair only</th><th>Cleaned alternative</th></tr>'
            + ''.join(rows) + '</table><p>Family correspondence is descriptive, not automatic acceptance. '
            'No calibrated wall/corner distance was used. Model, sampling mode and label export controls are in each viewer.</p>')
    (args.output / "index.html").write_text(page)
    write_json(args.output / "status.json", dict(state="COMPLETE", new_movies=36, anchors=6, catalogue_review_next=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repair", "source", "labels", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=6.)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--catalogue", action="store_true",
                        help="All supported cleaned K10/K18 families with corrected legacy ranking; separate review package")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(scope="supported cleaned K10/K18 catalogue" if args.catalogue else "six-anchor comparison",
                              profile="corrected_legacy" if args.catalogue else "comparison", modes=list(MODES),
                              panels_per_movie=6, state="PLAN")))
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    for key in ("repair", "source", "labels", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    render(args, prepare(args))


if __name__ == "__main__":
    main()
