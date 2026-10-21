"""Native-time movies of three supported execution neighbourhoods per theme.

Explicit empirical execution only. Reuses the frozen corrected distance, core
population, masks and video bindings; does not fit models or select group effects.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


def bind_measurements(calls, measurements):
    """Bind media/statistics identities by exact native interval, never ID text.

    The frozen index and measured tables use different internal bout offsets.
    Their occurrence-ID strings are retained in an explicit one-to-one crosswalk.
    """
    keys = ["theme", "session_id", "start_frame", "end_frame_exclusive"]
    if calls.duplicated(keys).any() or measurements.duplicated(keys).any():
        raise ValueError("duplicate native occurrence key")
    renamed = measurements.rename(columns={"occurrence_id": "measurement_occurrence_id"})
    joined = calls[keys + ["occurrence_id"]].merge(renamed, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(joined) != len(calls) or not joined._merge.eq("both").all():
        raise ValueError("media and measurements differ in native occurrence population")
    return joined.drop(columns="_merge").set_index("occurrence_id")


def write_gallery(output, plan, results):
    from .run_reference_seqnmf import digest, write_json

    ready = {(r["family"], r["view"]): r for r in results}
    families = copy.deepcopy(plan["families"])
    for family in families:
        family["overview"] = ready.get((family["stem"], "overview"))
        for region in family["neighbourhoods"]:
            region["movie"] = ready.get((family["stem"], region["id"]))
    data = dict(package=plan["labels_key"], source_manifest_sha256=digest(output / "manifest.json"),
                families=families, ready=len(results), expected=20,
                previous_random_url="../theme_delivery_20261021/media/")
    template = Path(__file__).with_name("theme_showcase_gallery.html").read_text()
    page = template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(output / "index.html")
    write_json(output / "gallery_data.json", data)


def distribution_figure(output, family, measurements, assignment):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ("#2878b5", "#e08027", "#2d9657")
    fig, all_axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = all_axes.ravel()
    for ax, (column, label) in zip(axes[:3], [("duration_s", "Core duration (s)"),
                                         ("mean_speed_px_s", "Mean speed (px/s)"),
                                         ("mean_abs_angular_velocity_rad_s", "Mean absolute angular velocity (rad/s)")]):
        values = measurements[column].to_numpy(float)
        finite = np.isfinite(values)
        bins = np.histogram_bin_edges(values[finite], bins=30)
        ax.hist(values[finite], bins=bins, color="0.8", label="All executions")
        for region, colour in zip(family["neighbourhoods"], colours):
            own = assignment.nearest_region.to_numpy() == region["id"]
            ax.hist(values[finite & own], bins=bins, histtype="step", color=colour,
                    label=f"Nearest anchor {region['region_number']}")
            selected = values[region["selected_indices"]]
            ax.plot(selected, np.zeros(len(selected)), "|", color=colour, markersize=14, clip_on=False)
        ax.set(xlabel=label, ylabel="Execution count")
    distance = np.sqrt(assignment.distance_squared.to_numpy(float))
    bins = np.histogram_bin_edges(distance, bins=30)
    axes[3].hist(distance, bins=bins, color="0.8")
    for region, colour in zip(family["neighbourhoods"], colours):
        own = assignment.nearest_region.to_numpy() == region["id"]
        axes[3].hist(distance[own], bins=bins, histtype="step", color=colour)
        selected = distance[region["selected_indices"]]
        axes[3].plot(selected, np.zeros(len(selected)), "|", color=colour, markersize=14, clip_on=False)
    axes[3].set(xlabel="Corrected distance to nearest anchor", ylabel="Execution count")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{family['theme_name']}: full population and nearest-anchor partitions\n"
                 "Partitions describe this display; they do not establish biological clusters. Bars below mark selected clips.", fontsize=11)
    name = family["stem"] + "_distribution"
    fig.savefig(output / (name + ".png"), dpi=160)
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)
    family["plot_filename"] = name + ".png"


def prepare(args):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata

    from studies.sequence_viz.recreate_generator_videos import (
        ClockCache,
        Tracks,
        execution_descriptors,
        rank_exemplar_neighbors,
        sample_indexes,
        syllable_segments,
    )

    from .render_reference_review import verify_call
    from .render_six_panel_review import guard
    from .run_reference_seqnmf import digest, log, write_json
    from .theme_neighbourhoods import select_neighbourhoods

    source = json.loads((args.base / "manifest.json").read_text())
    if json.loads((args.base / "media_audit.json").read_text())["status"] != "PASS":
        raise ValueError("base media population has not passed its audit")
    for name in ("library", "occurrences", "cleaned_recipe", "frozen_ranking", "mapping", "annotation_source"):
        item = source[name]
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"base {name} changed")
    for path, expected in source["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("base renderer/metric source changed")
    population_path = args.base / "eligible_calls.csv.gz"
    calls = pd.read_csv(population_path, dtype={"animal_id": str, "subject": str})
    if len(calls) != 7319 or calls.occurrence_id.duplicated().any():
        raise ValueError("expected the existing unique 7319-core population")
    metric_columns = ["duration_s", "mean_speed_px_s", "mean_abs_angular_velocity_rad_s",
                      "net_displacement_px", "signed_heading_change_rad"]
    columns = ["occurrence_id", "theme", "session_id", "start_frame", "end_frame_exclusive"] + metric_columns
    measurement_path = args.delivery / "occurrences.csv.gz"
    measured = bind_measurements(calls, pd.read_csv(measurement_path, usecols=columns))
    measured[["measurement_occurrence_id", "theme", "session_id", "start_frame", "end_frame_exclusive"]].to_csv(
        args.output / "occurrence_identity_crosswalk.csv.gz")
    mapping = np.asarray(json.loads(Path(source["mapping"]["path"]).read_text()), int)
    scaling = source["frozen_ranking"]["values"]
    tracks, clocks = Tracks(source["track_rows"]), ClockCache()
    values = np.empty((len(calls), 449), dtype=np.float64)
    for number, (sid, table) in enumerate(calls.groupby("recording", sort=False), 1):
        guard(args)
        track, clock = tracks.get(sid), clocks.get(source["bindings"][sid])
        records = table.to_dict("records")
        for row in records:
            verify_call(track, row, mapping)
        vectors, okay, _ = execution_descriptors(track, clock, records, dict(latent=np.ones(10), path=1.))
        if vectors.shape != (len(table), 449) or not okay.all():
            raise ValueError("original descriptor support unavailable")
        vectors[:, :320] /= np.tile(scaling["latent_iqr_native_units"], 32)
        vectors[:, 320:384] /= scaling["displacement_px_p75"]
        vectors[:, -1] /= scaling["log_duration_iqr"]
        values[table.index] = vectors
        if number % 10 == 0 or number == 99:
            log(f"SHOWCASE DESCRIPTORS {number}/99 recordings")
    plan = {key: copy.deepcopy(source[key]) for key in
            ("bindings", "track_rows", "annotation_mappings", "annotation_source", "library", "occurrences",
             "cleaned_recipe", "frozen_ranking", "mapping")}
    plan.update(families=[], fps=60, context_sec_each_side=.5,
                labels_key="karen-theme-neighbourhoods-20261021-v1", rows=3, columns=3,
                base=dict(path=str((args.base / "manifest.json").resolve()), sha256=digest(args.base / "manifest.json")),
                population=dict(path=str(population_path.resolve()), sha256=digest(population_path)),
                descriptive_measurements=dict(path=str(measurement_path.resolve()), sha256=digest(measurement_path)),
                selection="three cross-mouse-supported local anchors, exemplar plus two neighbours per row; no group/wave inputs",
                limitation="selected illustrations; neighbourhoods/nearest-anchor partitions are not evidence for three biological clusters")
    all_assignments, all_support = [], []
    for old in source["families"]:
        guard(args)
        sub = calls[calls.theme == old["stem"]].reset_index(drop=True)
        own = values[calls.theme.to_numpy() == old["stem"]]
        if hashlib.sha256(own.tobytes()).hexdigest() != old["descriptor_sha256"]:
            raise ValueError("descriptor values/order differ from frozen corrected metric")
        selection = select_neighbourhoods(own, sub.to_dict("records"), max_memory_gb=1.)
        if len(selection["regions"]) != 3:
            raise ValueError("fewer than three supported display neighbourhoods; inspect selection before rendering")
        family = {key: copy.deepcopy(old[key]) for key in
                  ("stem", "theme_name", "cohort", "stage_labels", "stage_sets", "caution", "saved_calls", "eligible_calls", "eligible_mice")}
        family.update(neighbourhoods=[], descriptor_sha256=old["descriptor_sha256"], selection_policy=selection["policy"])
        assigned = pd.DataFrame(selection["assignments"])
        if assigned.occurrence_id.tolist() != sub.occurrence_id.tolist():
            raise ValueError("selection assignment identities changed")
        assigned.insert(0, "theme", family["stem"])
        all_assignments.append(assigned)
        metrics = measured.loc[sub.occurrence_id].reset_index()
        for number, region in enumerate(selection["regions"], 1):
            anchor = int(region["anchor_index"])
            distance = cdist(own[anchor:anchor + 1], own, metric="sqeuclidean")[0]
            pct = 100 * (len(own) - rankdata(distance, method="average")) / (len(own) - 1)
            local = distance <= region["support_radius_squared"] + selection["distance_tolerance_squared"]
            nearest = assigned.nearest_region.to_numpy() == region["id"]
            summary = {}
            for column in metric_columns:
                x = metrics.loc[local, column].to_numpy(float)
                x = x[np.isfinite(x)]
                summary[column] = dict(median=float(np.median(x)), p10=float(np.quantile(x, .1)),
                                       p90=float(np.quantile(x, .9))) if len(x) else None
            r = dict(id=f"region_{number}", region_number=number, label=f"Neighbourhood {number}",
                     selection_kind=region["status"], support_calls=int(local.sum()),
                     support_mice=int(sub.loc[local, "subject"].nunique()), support_share=float(local.mean()),
                     assigned_calls=int(nearest.sum()), assigned_share=float(nearest.mean()),
                     summary=summary, anchor_occurrence_id=sub.iloc[anchor].occurrence_id,
                     selected_indices=region["selected_indices"], selection_details=region, examples=[])
            for panel, index in enumerate(region["selected_indices"], 1):
                example = dict(sub.iloc[index].to_dict(), role=f"N{number}_{'anchor' if panel == 1 else 'neighbour' + str(panel - 1)}",
                               selection_mode="neighbourhood", panel_number=panel, neighbourhood=number,
                               actual_percentile=float(pct[index]), distance_squared=float(distance[index]))
                example["filename"] = f"components/{family['stem']}_{example['role']}.mp4"
                r["examples"].append(example)
            family["neighbourhoods"].append(r)
            all_support.append(dict(theme=family["stem"], neighbourhood=number,
                                    **{k: r[k] for k in ("selection_kind", "support_calls", "support_mice", "support_share", "assigned_calls", "assigned_share", "anchor_occurrence_id")}))
        # Global-medoid distances are preserved as reference, never used to pick these anchors.
        original_distance, _, _ = rank_exemplar_neighbors(own, sub.subject.to_numpy())
        assigned["original_medoid_distance_squared"] = original_distance
        np.save(args.output / (family["stem"] + "_descriptors.npy"), own, allow_pickle=False)
        distribution_figure(args.output, family, metrics, assigned)
        plan["families"].append(family)
        write_json(args.output / (family["stem"] + "_selection.json"), selection)
        log(f"SHOWCASE SELECTED {family['stem']}: 3 anchors / 9 clips; {len(sub)} population")
    pd.concat(all_assignments).to_csv(args.output / "population_assignments.csv.gz", index=False)
    pd.DataFrame(all_support).to_csv(args.output / "neighbourhood_support.csv", index=False)
    # Fit one crop per theme across all nine selected executions, including context.
    for family in plan["families"]:
        extent = 160.
        for region in family["neighbourhoods"]:
            for example in region["examples"]:
                track = tracks.get(example["recording"])
                clock = clocks.get(plan["bindings"][example["recording"]])
                a = int(np.searchsorted(track["frames"], example["start_frame"]))
                example["onset_centroid"] = track["centroid"][a].tolist()
                example["onset_heading"] = float(track["heading"][a])
                example["segments"] = syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
                indexes, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
                if not np.all(np.diff(indexes) == 1):
                    raise ValueError("selected movie lacks one-to-one native60Hz stream")
                pos = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
                valid = (track["frames"][pos] == indexes) & np.isfinite(track["centroid"][pos]).all(axis=1)
                if not valid.all():
                    raise ValueError("selected context lacks verified tracking")
                xy = track["centroid"][pos] - example["onset_centroid"]
                h = example["onset_heading"]
                rotation = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
                extent = max(extent, float(np.max(np.abs(xy @ rotation.T))) + 55)
                example["frames"] = len(indexes)
        family["crop_native_pixels"] = 16 * math.ceil(2 * extent / 16)
        if family["crop_native_pixels"] > 2048:
            raise ValueError("selected movement exceeds bounded onset-fixed crop")
    from studies.sequence_viz.recreate_generator_videos import render_example
    from studies.sequence_viz.render_generator_review_grids import grid_filters

    plan["source_hashes"] = {str(p.resolve()): digest(p) for p in
                             (Path(__file__), Path(select_neighbourhoods.__code__.co_filename),
                              Path(render_example.__code__.co_filename), Path(sample_indexes.__code__.co_filename),
                              Path(grid_filters.__code__.co_filename), Path(verify_call.__code__.co_filename),
                              Path(__file__).with_name("theme_showcase_gallery.html"))}
    write_json(args.output / "manifest.json", plan)
    write_gallery(args.output, plan, [])
    return plan


def render_showcase_grid(family, clips, output, view):
    import cv2

    from studies.sequence_viz.render_generator_review_grids import grid_filters

    from .run_reference_seqnmf import digest, write_json

    path = output / f"{family['stem']}_{view}.mp4"
    cache = path.with_suffix(".json")
    if cache.exists():
        result = json.loads(cache.read_text())
        if digest(path) != result["sha256"]:
            raise ValueError("existing showcase changed")
        return result
    counts = [clip["frames"] for clip in clips]
    if len(clips) not in (3, 9):
        raise ValueError("showcase requires one three-panel row or three rows")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    for clip in clips:
        command += ["-threads", "1", "-i", str(output / clip["filename"])]
    filters = grid_filters(counts, columns=3)
    temporary = path.with_suffix(".partial.mp4")
    command += ["-filter_complex", filters, "-map", "[out]", "-frames:v", str(max(counts)),
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    encoded = subprocess.run(command, capture_output=True, timeout=600)
    if encoded.returncode:
        raise RuntimeError(encoded.stderr.decode()[-2000:])
    temporary.replace(path)
    capture, count = cv2.VideoCapture(str(path)), 0
    try:
        if (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (2160, 960 * len(clips) // 3):
            raise ValueError("showcase dimensions differ")
        if abs(capture.get(cv2.CAP_PROP_FPS) - 60) > .001:
            raise ValueError("showcase clock differs")
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            if count == min(33, min(counts) - 1):
                cv2.imwrite(str(path.with_suffix(".png")), frame)
            count += 1
        if count != max(counts):
            raise ValueError("showcase full decode count differs")
    finally:
        capture.release()
    result = dict(filename=path.name, family=family["stem"], view=view, frames=count, playback_s=count / 60,
                  sha256=digest(path), columns=3, rows=len(clips) // 3, width=2160, height=960 * len(clips) // 3,
                  panel_order=[clip["example"]["role"] for clip in clips],
                  occurrence_ids=[clip["example"]["occurrence_id"] for clip in clips],
                  clips=[dict(filename=c["filename"], sha256=c["sha256"], frames=c["frames"]) for c in clips],
                  dot_offscreen_frames=sum(c["dot_offscreen_frames"] for c in clips),
                  missing_tracking_frames=sum(c["missing_tracking_context_or_call_frames"] for c in clips),
                  filters=filters, timing="natural60Hz; independent dark finish; row=anchor plus two neighbours")
    if result["dot_offscreen_frames"] or result["missing_tracking_frames"]:
        raise ValueError("showcase has missing tracking or escaped centroid")
    write_json(cache, result)
    return result


def render_family(family, plan, output):
    import cv2

    from studies.sequence_viz.recreate_generator_videos import ClockCache, Tracks, render_example

    from .run_reference_seqnmf import digest

    cv2.setNumThreads(1)
    output = Path(output)
    tracks, clocks = Tracks(plan["track_rows"]), ClockCache()
    results, all_clips = [], []
    for region in family["neighbourhoods"]:
        clips = []
        for example in region["examples"]:
            cached = (output / example["filename"]).with_suffix(".json")
            if cached.exists():
                clip = json.loads(cached.read_text())
                if digest(output / example["filename"]) != clip["sha256"] or clip["example"] != example:
                    raise ValueError("cached native clip identity changed")
            else:
                clip = render_example(family, example, plan, output, tracks, clocks)
            clips.append(clip)
        results.append(render_showcase_grid(family, clips, output, region["id"]))
        all_clips.extend(clips)
    results.append(render_showcase_grid(family, all_clips, output, "overview"))
    return results


def render(args, plan):
    from .render_six_panel_review import guard
    from .run_reference_seqnmf import digest, log, write_json

    for path, expected in plan["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("prepared source changed")
    for name in ("base", "population", "library", "occurrences", "cleaned_recipe", "frozen_ranking", "mapping", "descriptive_measurements"):
        item = plan[name]
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("prepared input changed")
    results = []
    batches = [plan["families"][:1]] if args.preview else [plan["families"][:1], plan["families"][1:]]
    for batch in batches:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(batch))) as pool:
            pending = {pool.submit(render_family, f, plan, str(args.output)) for f in batch}
            while pending:
                guard(args)
                completed, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in completed:
                    results.extend(future.result())
                    write_gallery(args.output, plan, results)
                    log(f"SHOWCASE MOVIES {len(results)}/20")
    status = dict(state="PREVIEW_COMPLETE" if args.preview else "COMPLETE", movies=len(results),
                  unique_executions=9 * len({r["family"] for r in results}),
                  overview_playback_s=sum(r["playback_s"] for r in results if r["view"] == "overview"),
                  separate_rows_playback_s=sum(r["playback_s"] for r in results if r["view"] != "overview"),
                  workers=args.workers, numerical_threads=2, encoder_threads=2,
                  dot_offscreen_frames=sum(r["dot_offscreen_frames"] for r in results),
                  missing_tracking_frames=sum(r["missing_tracking_frames"] for r in results))
    write_json(args.output / ("preview.json" if args.preview else "status.json"), status)
    log(json.dumps(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "delivery", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "render", "all"), default="all")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=3.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False, stage=args.stage, themes=5, neighbourhoods_per_theme=3,
                              executions_per_neighbourhood=3, unique_executions=45, overview_movies=5,
                              row_movies=15, output=str(args.output))))
        return
    if args.workers < 1 or args.memory_gb < 2:
        parser.error("positive workers and at least2GiB memory required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "components").mkdir(exist_ok=True)
    start = time.monotonic()
    manifest = args.output / "manifest.json"
    if args.stage == "render" and not manifest.exists():
        parser.error("render stage requires a prepared manifest")
    plan = json.loads(manifest.read_text()) if manifest.exists() else prepare(args)
    if args.stage in ("render", "all"):
        render(args, plan)
    print(f"SHOWCASE stage elapsed {time.monotonic() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
