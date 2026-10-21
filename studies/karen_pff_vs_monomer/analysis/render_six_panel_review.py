"""Six natural-speed executions per grid, all sampling modes, saved Karen fits only."""

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from studies.karen_pff_vs_monomer.analysis.render_reference_review import verify_call
from studies.karen_pff_vs_monomer.analysis.run_reference_seqnmf import log, write_json
from studies.sequence_viz.recreate_generator_videos import (
    ClockCache,
    Tracks,
    render_example,
    sample_indexes,
    syllable_segments,
)
from studies.sequence_viz.render_generator_review_grids import render_grid, select_movie_sets
from studies.sequence_viz.render_generator_screening import digest

MODES = ("p80", "p100", "random")
_WORKER = {}


def guard(args):
    import psutil

    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if rss > args.memory_gb * 1024**3:
        raise MemoryError("parent/worker/encoder combined RSS guard exceeded")
    if shutil.disk_usage(args.output).free < args.min_free_disk_gb * 1024**3:
        raise RuntimeError("free disk floor reached")
    if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > args.max_output_gb * 1024**3:
        raise RuntimeError("output-size cap reached")


def prepare(args):
    import numpy as np
    import pandas as pd

    source = json.loads((args.source / "manifest.json").read_text())
    ranking_path = args.source / "execution_ranking.csv.gz"
    if digest(ranking_path) != source["ranking"]["sha256"]:
        raise ValueError("original exemplar ranking changed")
    recipe_path = args.base / "recipe.json"
    if digest(recipe_path) != source["source_recipe_sha256"]:
        raise ValueError("pooled fit input recipe changed")
    recipe = json.loads(recipe_path.read_text())
    mapping_path = Path(recipe["input_metadata"][3]["path"])
    if digest(mapping_path) != recipe["input_metadata"][3]["sha256"]:
        raise ValueError("fitted grouping map changed")
    mapping = np.asarray(json.loads(mapping_path.read_text()), dtype=int)
    rank = pd.read_csv(ranking_path, dtype={"subject": str})
    tables = {}
    for trial, item in source["inputs"].items():
        folder = Path(item["directory"])
        if digest(folder / "model.npz") != item["model_sha256"]:
            raise ValueError("saved model fingerprint changed")
        path = folder / "pooled_calls.csv.gz"
        if digest(path) != item["calls_sha256"]:
            raise ValueError("saved call fingerprint changed")
        tables[trial] = pd.read_csv(path, dtype={"animal_id": str})
    manifest = dict(source_manifest=str(args.source / "manifest.json"),
                    source_manifest_sha256=digest(args.source / "manifest.json"),
                    source_ranking_sha256=digest(ranking_path), inputs=source["inputs"],
                    track_rows=source["track_rows"], bindings=source["bindings"],
                    annotation_mappings=source["annotation_mappings"], annotation_source=source["annotation_source"],
                    ranking=source["ranking"], families=[], modes=list(MODES), panels=6, rows=2, columns=3,
                    fps=60, context_sec_each_side=.5, fit_design="same pooled99recordings/21mice, no refits",
                    selection="six unique executions per mode; ranked animal diversity inside declared pools; random uniform",
                    legacy_labels_key="karen-pooled-p80-20261015-v1",
                    labels_key="karen-pooled-six-panel-20261016-v1")
    for old in source["families"]:
        guard(args)
        table = tables[old["trial"]]
        own = table[table.factor == old["factor"]]
        lookup = {(r.session_id, int(r.start_frame), int(r.end_frame_exclusive)): r._asdict()
                  for r in own.itertuples(index=False)}
        ranks = rank[rank.family == old["stem"]]
        calls = []
        for r in ranks.itertuples(index=False):
            original = lookup[(r.recording, int(r.start_frame), int(r.stop_frame))]
            calls.append(dict(original, recording=r.recording, subject=str(r.subject), stop_frame=int(r.stop_frame)))
        if len(calls) != old["eligible_calls"]:
            raise ValueError("original ranked call population changed")
        exemplar = int(np.argmin(ranks.distance.to_numpy()))
        seed = int(hashlib.sha256(old["stem"].encode()).hexdigest()[:8], 16)
        sets = select_movie_sets(calls, ranks.distance.to_numpy(), ranks.percentile.to_numpy(), seed, exemplar, panels=6)
        order = table.groupby("factor").size().sort_values(ascending=False, kind="stable").index.tolist()
        family = {k: v for k, v in old.items() if k not in ("examples", "crop_native_pixels")}
        family.update(examples_by_mode=sets, occurrence_rank=order.index(old["factor"]) + 1,
                      saved_calls=len(own), crop_native_pixels=320)
        manifest["families"].append(family)
    by_recording = defaultdict(list)
    for family in manifest["families"]:
        for mode, examples in family["examples_by_mode"].items():
            for e in examples:
                e["filename"] = "components/" + family["stem"] + "_" + e["role"] + ".mp4"
                by_recording[e["recording"]].append((family, mode, e))
    tracks, clocks = Tracks(manifest["track_rows"]), ClockCache()
    extents = defaultdict(lambda: 160.)
    for number, (sid, selected) in enumerate(sorted(by_recording.items()), 1):
        guard(args)
        track = tracks.get(sid)
        binding = manifest["bindings"][sid]
        video = Path(binding["video_path"])
        if (video.stat().st_size, video.stat().st_mtime_ns) != (binding["video_size"], binding["video_mtime_ns"]):
            raise ValueError("native video changed")
        clock = clocks.get(binding)
        for family, _, e in selected:
            verify_call(track, e, mapping)
            a, b = np.searchsorted(track["frames"], [e["start_frame"], e["stop_frame"]])
            if not np.isfinite(track["latent"][a:b]).all():
                raise ValueError("selected core lacks finite latent support")
            e["onset_centroid"] = track["centroid"][a].tolist()
            e["onset_heading"] = float(track["heading"][a])
            e["segments"] = syllable_segments(track, clock, e["start_frame"], e["stop_frame"])
            indexes, _ = sample_indexes(clock, e["start_frame"], e["stop_frame"], fps=60)
            pos = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
            valid = (track["frames"][pos] == indexes) & np.isfinite(track["centroid"][pos]).all(axis=1)
            xy = track["centroid"][pos[valid]] - np.asarray(e["onset_centroid"])
            h = e["onset_heading"]
            rotation = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
            extents[family["stem"]] = max(extents[family["stem"]], float(np.max(np.abs(xy @ rotation.T))) + 55)
            e["frames"] = len(indexes)
        log(f"PREPARE {number}/{len(by_recording)} recordings; native support verified")
    for family in manifest["families"]:
        family["crop_native_pixels"] = 16 * math.ceil(2 * extents[family["stem"]] / 16)
        if family["crop_native_pixels"] > 2048:
            raise ValueError("selected movement exceeds bounded fixed crop")
    manifest["families"].sort(key=lambda f: (f["K"], f["occurrence_rank"]))
    manifest["source_hashes"] = {str(p): digest(p) for p in
                                 (Path(__file__), Path(render_example.__code__.co_filename),
                                  Path(render_grid.__code__.co_filename), Path(__file__).with_name("six_panel_gallery.html"))}
    write_json(args.output / "manifest.json", manifest)
    log("PREPARE COMPLETE:49families ×3modes ×6executions; original ranking unchanged")


def write_gallery(output, manifest, results):
    ready = defaultdict(dict)
    for r in results:
        ready[r["family"]][r["selection_mode"]] = r
    families = []
    for f in manifest["families"]:
        if f["stem"] in ready:
            families.append(dict(**f, movies=ready[f["stem"]]))
    expected = len(manifest["families"]) * len(manifest.get("modes", MODES))
    data = dict(package=manifest["labels_key"], legacy_key=manifest["legacy_labels_key"],
                families=families, names=manifest["annotation_mappings"]["karen"],
                complete=len(results) == expected, expected_movies=expected, ready_movies=len(results),
                reference_key=manifest.get("reference_labels_key"),
                reference_labels=manifest.get("reference_labels", {}),
                description=manifest.get("review_description", "Model parameters and rankings are unchanged."))
    template = Path(__file__).with_name("six_panel_gallery.html").read_text()
    page = template.replace("__REVIEW_DATA__", json.dumps(data).replace("<", "\\u003c"))
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(output / "index.html")


def init_worker(manifest, output):
    import cv2

    cv2.setNumThreads(1)
    _WORKER.update(manifest=manifest, output=Path(output), tracks=Tracks(manifest["track_rows"]), clocks=ClockCache())


def render_family(family):
    plan, output = _WORKER["manifest"], _WORKER["output"]
    results = []
    for mode in MODES:
        path = output / (family["stem"] + "_" + mode + "_grid.json")
        if path.exists():
            result = json.loads(path.read_text())
            if digest(output / result["filename"]) != result["sha256"] or result.get("rows") != 2:
                raise ValueError("existing six-panel grid fingerprint/layout changed")
            results.append(result)
            continue
        clips = [render_example(family, e, plan, output, _WORKER["tracks"], _WORKER["clocks"])
                 for e in family["examples_by_mode"][mode]]
        result = render_grid(family, clips, output, selection_mode=mode, columns=3)
        result["dot_offscreen_frames"] = sum(c["dot_offscreen_frames"] for c in clips)
        result["missing_tracking_frames"] = sum(c["missing_tracking_context_or_call_frames"] for c in clips)
        if result["dot_offscreen_frames"]:
            raise ValueError("centroid escaped fixed crop")
        write_json(path, result)
        results.append(result)
    return results


def render(args):
    plan = json.loads((args.output / "manifest.json").read_text())
    for path, expected in plan["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("prepared renderer source changed")
    if digest(plan["source_manifest"]) != plan["source_manifest_sha256"]:
        raise ValueError("original P80manifest changed")
    families = sorted(plan["families"], key=lambda f: (f["occurrence_rank"] > 10, f["K"], f["occurrence_rank"]))
    if args.preview:
        families = [next(f for f in families if f["K"] == k) for k in (10, 18, 22)]
    results = []
    workers = min(args.workers, max(1, len(os.sched_getaffinity(0)) // 4))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(plan, str(args.output))) as pool:
        pending = {pool.submit(render_family, f): f for f in families}
        while pending:
            guard(args)
            completed = next(as_completed(pending))
            family = pending.pop(completed)
            results.extend(completed.result())
            write_gallery(args.output, plan, results)
            log(f"RENDER {len(results)}/{len(families) * 3} movies; {family['stem']} all modes complete")
    status = dict(state="PREVIEW_COMPLETE" if args.preview else "COMPLETE", families=len(families),
                  movies=len(results), executions=len(results) * 6, rows=2, columns=3, modes=list(MODES),
                  total_playback_sec=sum(r["playback_s"] for r in results),
                  playback_by_mode={m: sum(r["playback_s"] for r in results if r["selection_mode"] == m) for m in MODES},
                  dot_offscreen_frames=sum(r.get("dot_offscreen_frames", 0) for r in results),
                  missing_tracking_frames=sum(r.get("missing_tracking_frames", 0) for r in results),
                  actual_workers=workers, numerical_threads=2, encoder_threads=2,
                  ranking="unchanged original exemplar ranks", human_review="pending for six-panel/mode-specific views")
    write_json(args.output / ("preview_status.json" if args.preview else "status.json"), status)
    log(json.dumps(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "render"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=8.)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, modes=MODES, panels=6, rows=2, columns=3,
                              families=49, execute=False, output=str(args.output))))
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "components").mkdir(exist_ok=True)
    guard(args)
    if args.stage == "prepare":
        prepare(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
