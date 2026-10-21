"""Separate p100, P80 and random movies with one selection setting per grid."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from studies.sequence_viz.recreate_generator_videos import (
    ClockCache,
    Tracks,
    digest,
    inputs,
    onset_matrix,
    render_example,
    sample_indexes,
    syllable_segments,
    token_colour,
)


def grid_filters(counts):
    """Pad only after each execution ends; never stretch, repeat or freeze it."""
    maximum = max(counts)
    parts = []
    for i, count in enumerate(counts):
        parts.append(
            f"[{i}:v]tpad=stop={maximum - count}:stop_mode=add:color=0x121212,"
            f"trim=end_frame={maximum},setpts=PTS-STARTPTS,"
            f"drawtext=text='Finished (natural speed)':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=h/2:enable='gte(n,{count})'[p{i}]"
        )
    parts.append("".join(f"[p{i}]" for i in range(len(counts))) + f"hstack=inputs={len(counts)}[out]")
    return ";".join(parts)


def render_grid(family, clips, output, *, selection_mode):
    import cv2

    if selection_mode not in ("p100", "p80", "random") or len(clips) != 3:
        raise ValueError("one movie requires three panels with one common selection mode")
    if any(c["example"].get("selection_mode") != selection_mode for c in clips):
        raise ValueError("mixed selection modes in one movie are forbidden")
    path = output / (family["stem"] + "_" + selection_mode + "_grid.mp4")
    temporary = path.with_suffix(".partial.mp4")
    counts = [c["frames"] for c in clips]
    filters = grid_filters(counts)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    for clip in clips:
        command += ["-threads", "1", "-i", str(output / clip["filename"])]
    command += [
        "-filter_complex",
        filters,
        "-map",
        "[out]",
        "-frames:v",
        str(max(counts)),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    encoded = subprocess.run(command, capture_output=True, timeout=600)
    if encoded.returncode:
        raise RuntimeError(encoded.stderr.decode()[-2000:])
    temporary.replace(path)
    capture = cv2.VideoCapture(str(path))
    count = 0
    poster_index = min(33, min(counts) - 1)
    try:
        if not capture.isOpened():
            raise ValueError("grid cannot be decoded")
        if (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (720 * len(clips), 960):
            raise ValueError("grid dimensions differ")
        if abs(capture.get(cv2.CAP_PROP_FPS) - 60) > 0.001:
            raise ValueError("grid clock differs")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count == poster_index:
                cv2.imwrite(str(path.with_suffix(".png")), frame)
            count += 1
        if count != max(counts):
            raise ValueError("full grid decode count differs from longest natural-speed panel")
    finally:
        capture.release()
    result = dict(
        filename=path.name,
        family=family["stem"],
        selection_mode=selection_mode,
        frames=count,
        playback_s=count / 60,
        sha256=digest(path),
        panel_order=[c["example"]["role"] for c in clips],
        clips=[dict(filename=c["filename"], sha256=c["sha256"], frames=c["frames"]) for c in clips],
        filters=filters,
        timing="each source clip at original 60fps; dark finished panel after its end",
        width=720 * len(clips),
        height=960,
    )
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def select_movie_sets(calls, distance, percentiles, seed, exemplar, *, panels=3):
    """Three unique calls per mode; random is uniform over the entire timed pool.

    Ranked animal diversity is preferred only within a declared near-target pool.
    Top pool: closest max(panels, ceil(2% N)); P80: +/-5 percentile points, extended
    only to supply panels when discrete ranks leave too few candidates. Animals
    can repeat when those pools do not supply three different mice.
    """
    import numpy as np

    if len(calls) < panels:
        raise ValueError("insufficient distinct executions for requested movie panels")

    def choose_diverse(order):
        selected, animals = [], set()
        for index in order:
            if calls[index]["subject"] not in animals:
                selected.append(index)
                animals.add(calls[index]["subject"])
                if len(selected) == panels:
                    return selected
        return (selected + [i for i in order if i not in selected])[:panels]

    def tie(i):
        return calls[i]["recording"], calls[i]["start_frame"]

    top_order = [exemplar] + sorted(
        (i for i in range(len(calls)) if i != exemplar), key=lambda i: (distance[i], tie(i))
    )
    top_pool = top_order[: max(panels, math.ceil(0.02 * len(calls)))]
    middle_order = sorted(range(len(calls)), key=lambda i: (abs(percentiles[i] - 80), tie(i)))
    middle_pool = [i for i in middle_order if abs(percentiles[i] - 80) <= 5]
    if len(middle_pool) < panels:
        middle_pool = middle_order[:panels]
    chosen = dict(
        p100=choose_diverse(top_pool),
        p80=choose_diverse(middle_pool),
        random=np.random.default_rng(seed).choice(len(calls), size=panels, replace=False).tolist(),
    )
    result = {}
    for mode, indexes in chosen.items():
        result[mode] = [
            dict(
                **calls[index],
                selection_mode=mode,
                panel_number=number,
                role=f"{mode}_{number}",
                requested_percentile={"p100": 100, "p80": 80, "random": None}[mode],
                actual_percentile=float(percentiles[index]),
                distance_squared=float(distance[index]),
                random_seed=seed,
            )
            for number, index in enumerate(indexes, start=1)
        ]
    return result


def build_separate_review(ranking_source, output):
    """Reuse the validated complete ranking and call artifacts; no re-profiling."""
    import numpy as np

    previous = json.loads((ranking_source / "manifest.json").read_text())
    if json.loads((ranking_source / "validation.json").read_text())["status"] != "PASSED":
        raise ValueError("ranking source has not passed validation")
    if previous["ranking"]["distance_reference"] != "actual animal-balanced medoid execution":
        raise ValueError("unexpected ranking reference")
    if digest(ranking_source / "execution_ranking.csv.gz") != previous["ranking"]["ranking_sha256"]:
        raise ValueError("ranking source fingerprint changed")
    old, _, rows = inputs(Path(previous["source"]))
    if digest(Path(previous["source"]) / "manifest.json") != previous["source_manifest_sha256"]:
        raise ValueError("screening source changed")
    tracks, clocks = Tracks(rows), ClockCache()
    candidates = {}
    for path, fingerprint in previous["artifact_hashes"].items():
        if digest(path) != fingerprint:
            raise ValueError("completed call artifact changed")
        artifact = json.loads(gzip.decompress(Path(path).read_bytes()))
        identity = artifact["identity"]
        if identity["fold"] != 0 or identity["replicate"] != 0:
            raise ValueError("non-reference fit requested")
        for call in artifact["prediction"]["selected_calls"]:
            key = (
                identity["panel_id"] + "_" + call["type_id"],
                identity["recording"],
                call["start_frame"],
                call["stop_frame"],
            )
            candidates[key] = dict(
                **call,
                recording=identity["recording"],
                subject=identity["subject"],
                metadata=identity["metadata"],
                artifact=path,
            )
    ranked = defaultdict(list)
    with gzip.open(ranking_source / "execution_ranking.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            ranked[row["family"]].append(row)
    for family in previous["families"]:
        original_exemplar = family["examples"][:1]
        family["examples"] = []
        own = ranked[family["stem"]]
        if not own:
            continue
        if len(own) != family["eligible_calls"]:
            raise ValueError("eligible ranking pool differs")
        calls = [candidates[(family["stem"], r["recording"], int(r["start_frame"]), int(r["stop_frame"]))] for r in own]
        if any(c["subject"] != r["subject"] for c, r in zip(calls, own)):
            raise ValueError("ranked animal identity differs")
        ex = original_exemplar[0]
        exemplar = next(
            i
            for i, c in enumerate(calls)
            if (c["recording"], c["start_frame"], c["stop_frame"])
            == (ex["recording"], ex["start_frame"], ex["stop_frame"])
        )
        distance = np.array([float(r["distance_squared"]) for r in own])
        percentiles = np.array([float(r["exemplarness_percentile"]) for r in own])
        if distance[exemplar] != 0:
            raise ValueError("exemplar identity does not match frozen distance reference")
        seed = int(hashlib.sha256(("separate-chain-movies-20261015:" + family["stem"]).encode()).hexdigest()[:8], 16)
        sets = select_movie_sets(calls, distance, percentiles, seed, exemplar)
        family["examples"] = [e for es in sets.values() for e in es]
        family["sampling_pools"] = dict(
            top_size=max(3, math.ceil(0.02 * len(calls))), p80_band=[75, 85], random_pool_size=len(calls)
        )
        extent = 160.0
        for example in family["examples"]:
            track = tracks.get(example["recording"])
            clock = clocks.get(old["bindings"][example["recording"]])
            onset = int(np.searchsorted(track["frames"], example["start_frame"]))
            example["onset_centroid"] = track["centroid"][onset].tolist()
            example["onset_heading"] = float(track["heading"][onset])
            example["duration_s"] = float(clock[example["stop_frame"]] - clock[example["start_frame"]])
            example["segments"] = syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
            indexes, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
            loc = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
            known = (track["frames"][loc] == indexes) & np.isfinite(track["centroid"][loc]).all(axis=1)
            xy = track["centroid"][loc[known]] - np.asarray(example["onset_centroid"])
            heading = example["onset_heading"]
            rotation = np.array([[math.cos(heading), math.sin(heading)], [-math.sin(heading), math.cos(heading)]])
            extent = max(extent, float(np.max(np.abs(xy @ rotation.T))) + 55)
            example["frames"], example["playback_s"] = len(indexes), len(indexes) / 60
            example["filename"] = family["stem"] + "_" + example["role"] + ".mp4"
        family["crop_native_pixels"] = int(16 * math.ceil(2 * extent / 16))
        if family["crop_native_pixels"] > 2048:
            raise ValueError("selected movement exceeds bounded crop; inspect before rendering")
    shutil.copyfile(ranking_source / "execution_ranking.csv.gz", output / "execution_ranking.csv.gz")
    previous.update(
        view_mode="separate_p100_p80_random_movies",
        panels_per_movie=3,
        ranking_source=str(ranking_source),
        ranking_source_manifest_sha256=digest(ranking_source / "manifest.json"),
        ranking_source_validation_sha256=digest(ranking_source / "validation.json"),
    )
    previous["ranking"]["selection"] = (
        "three homogeneous movies; top max(3,ceil(2% N)) pool; P80 +/-5 rank-point pool (minimum three); animal diversity within ranked pools then distinct-call fallback; uniform full-pool random sample without replacement; modes may overlap"
    )
    previous["render"].update(
        grid="three simultaneous executions of one family and one selection mode per movie",
        camera_scale="common per family across all nine selected executions",
    )
    (output / "manifest.json").write_text(json.dumps(previous, indent=2) + "\n")
    return previous


def gallery(plan, grids, output, preview):
    available = {(g["family"], g["selection_mode"]): g for g in grids}
    payload = []
    for family in plan["families"]:
        movies = {
            mode: available[(family["stem"], mode)]
            for mode in ("p100", "p80", "random")
            if (family["stem"], mode) in available
        }
        if movies:
            payload.append(
                dict(
                    stem=family["stem"],
                    cohort=family["cohort"],
                    pool=family["eligible_calls"],
                    animals=family["eligible_animals"],
                    examples=family["examples"],
                    movies=movies,
                    mapping=plan["annotation_mappings"][family["cohort"]],
                )
            )
    page = """<!doctype html><meta charset="utf-8"><title>Qualitative chain review</title>
<style>body{font:16px system-ui;background:#16181d;color:#eee;max-width:1800px;margin:20px auto;padding:0 16px}select,button{font:inherit;padding:8px;margin:4px}video{display:block;width:100%;max-height:78vh;margin:12px auto}a{color:#9cd1ff}#sequences{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;line-height:1.6}.muted{color:#bfc3cd}</style>
<h1>Qualitative chain review</h1><p>Each video shows three executions of the same chain with <strong>one common selection setting</strong>. Choose p100, P80 or random for the whole video.</p>
<p class="muted">Start with P80. Replay to focus on another panel. Natural pace, onset-locked camera, coloured source-syllable dot and live syllable strip. Finished panels go dark. Similarity ranks are descriptive, not behavioural correctness.</p>
<button id="previous">Previous chain</button><select id="family"></select><button id="next">Next chain</button><div id="choices"></div><button id="replay">Replay movie</button>
<p id="details"></p><video id="player" controls preload="none"></video><div id="sequences"></div>
<p><a id="direct">Open movie</a> · <a href="START_HERE.md">Viewing guide</a> · <a href="report.md">Methods</a> · <a href="family_review_notes.csv">Family notes</a> · <a href="review_notes.csv">Execution notes</a></p>
<script>const families=PAYLOAD;const picker=document.getElementById('family'),player=document.getElementById('player');let mode='p80';
families.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.cohort+' / '+f.stem;picker.append(o)});
function choose(m){mode=m;const f=families[+picker.value],g=f.movies[mode];player.pause();player.src=g.filename;player.poster=g.filename.replace('.mp4','.png');document.getElementById('direct').href=g.filename;
const es=f.examples.filter(e=>e.selection_mode===mode);document.getElementById('details').textContent=mode+' movie — all three panels use '+mode+' selection; '+g.playback_s.toFixed(2)+'s per replay; pool '+f.pool+' occurrences / '+f.animals+' mice';
const sequences=document.getElementById('sequences');sequences.replaceChildren();es.forEach(e=>{const p=document.createElement('p');p.textContent='Panel '+e.panel_number+' | mouse '+e.subject+' | actual similarity percentile '+e.actual_percentile.toFixed(1)+' — '+e.segments.map(s=>{let label=(f.mapping[s.syllable_id]||'unlabelled').replace('annotation:','');if(label.startsWith('unmapped:'))label='unlabelled';return 'S'+s.syllable_id+' '+label+' '+s.duration_s.toFixed(2)+'s'}).join(' → ');sequences.append(p)});}
function change(){const f=families[+picker.value],choices=document.getElementById('choices');choices.replaceChildren();['p100','p80','random'].forEach(m=>{if(f.movies[m]){const b=document.createElement('button');b.textContent=m==='p100'?'p100 / top exemplars':m==='p80'?'P80':'Random';b.onclick=()=>choose(m);choices.append(b)}});choose(f.movies[mode]?mode:'p80');}
picker.onchange=change;document.getElementById('previous').onclick=()=>{picker.value=(+picker.value+families.length-1)%families.length;change()};document.getElementById('next').onclick=()=>{picker.value=(+picker.value+1)%families.length;change()};document.getElementById('replay').onclick=()=>{player.currentTime=0;player.play()};if(families.length)change();</script>"""
    (output / ("preview.html" if preview else "index.html")).write_text(
        page.replace("PAYLOAD", json.dumps(payload).replace("</", "<\\/"))
    )


def validate_collection(plan, families, grids, clips, output):
    """Audit selection grouping, frozen identities, clocks, overlays and grid timing."""
    import cv2
    import numpy as np
    from scipy.stats import rankdata

    _, _, rows = inputs(Path(plan["source"]))
    tracks, clocks = Tracks(rows), ClockCache()
    if digest(output / "execution_ranking.csv.gz") != plan["ranking"]["ranking_sha256"]:
        raise ValueError("ranking changed")
    for key in ("manifest", "validation"):
        if digest(Path(plan["ranking_source"]) / (key + ".json")) != plan["ranking_source_" + key + "_sha256"]:
            raise ValueError("frozen ranking provenance changed")
    ranked, original = defaultdict(list), {}
    with gzip.open(output / "execution_ranking.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            ranked[row["family"]].append(row)
    for path, fingerprint in plan["artifact_hashes"].items():
        if digest(path) != fingerprint:
            raise ValueError("original calls changed")
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        identity = data["identity"]
        for call in data["prediction"]["selected_calls"]:
            original[
                (
                    identity["panel_id"] + "_" + call["type_id"],
                    identity["recording"],
                    call["start_frame"],
                    call["stop_frame"],
                )
            ] = (identity["model_key"], path)
    by_grid = {(g["family"], g["selection_mode"]): g for g in grids}
    by_clip = {c["filename"]: c for c in clips}
    checks, dot_checks, finished_checks, active_missing, context_missing, offscreen = 0, 0, 0, 0, 0, 0
    max_error = 0.0
    selection_stats = []
    for family in families:
        pool = ranked[family["stem"]]
        distances = np.array([float(r["distance_squared"]) for r in pool])
        percentiles = np.array([float(r["exemplarness_percentile"]) for r in pool])
        np.testing.assert_allclose(
            percentiles, 100 * (len(pool) - rankdata(distances, method="average")) / (len(pool) - 1), atol=1e-12
        )
        lookup = {(r["recording"], int(r["start_frame"]), int(r["stop_frame"])): i for i, r in enumerate(pool)}
        for mode in ("p100", "p80", "random"):
            examples = [e for e in family["examples"] if e["selection_mode"] == mode]
            if len(examples) != 3 or len({(e["recording"], e["start_frame"], e["stop_frame"]) for e in examples}) != 3:
                raise ValueError("movie must contain three distinct executions of the same selection setting")
            selected = [lookup[(e["recording"], e["start_frame"], e["stop_frame"])] for e in examples]
            if mode == "p100":
                allowed = set(np.argsort(distances, kind="stable")[: max(3, math.ceil(0.02 * len(pool)))].tolist())
                # Equal-distance boundary ties are equivalent members of the top pool.
                limit = max(distances[list(allowed)])
                if any(distances[i] > limit for i in selected) or distances[selected[0]] != 0:
                    raise ValueError("top-tier selection outside declared neighbourhood")
            elif mode == "p80":
                allowed = np.flatnonzero(np.abs(percentiles - 80) <= 5).tolist()
                if len(allowed) < 3:
                    allowed = np.argsort(np.abs(percentiles - 80), kind="stable")[:3].tolist()
                if not set(selected) <= set(allowed):
                    raise ValueError("P80 selection outside near-target pool")
            else:
                expected = (
                    np.random.default_rng(examples[0]["random_seed"]).choice(len(pool), size=3, replace=False).tolist()
                )
                if selected != expected:
                    raise ValueError("random is not the frozen uniform full-pool draw")
            grid = by_grid[(family["stem"], mode)]
            grid_path = output / grid["filename"]
            if digest(grid_path) != grid["sha256"] or grid["frames"] != max(e["frames"] for e in examples):
                raise ValueError("grid identity or natural duration differs")
            if grid["panel_order"] != [e["role"] for e in examples]:
                raise ValueError("grid panel order differs")
            capture = cv2.VideoCapture(str(grid_path))
            try:
                if capture.get(cv2.CAP_PROP_FRAME_COUNT) != grid["frames"]:
                    raise ValueError("grid frame count differs")
                for panel, example in enumerate(examples):
                    row = pool[selected[panel]]
                    if (float(row["distance_squared"]), float(row["exemplarness_percentile"]), row["subject"]) != (
                        example["distance_squared"],
                        example["actual_percentile"],
                        example["subject"],
                    ):
                        raise ValueError("score/rank/animal identity differs")
                    native_id = (family["stem"], example["recording"], example["start_frame"], example["stop_frame"])
                    if (
                        original[native_id] != (family["model_key"], example["artifact"])
                        or rows[example["recording"]]["exclude_from_search"]
                    ):
                        raise ValueError("source call/model or animal exclusion differs")
                    track = tracks.get(example["recording"])
                    clock = clocks.get(plan["bindings"][example["recording"]])
                    if (
                        syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
                        != example["segments"]
                    ):
                        raise ValueError("original source sequence differs")
                    onset = np.searchsorted(track["frames"], example["start_frame"])
                    np.testing.assert_allclose(track["centroid"][onset], example["onset_centroid"], atol=1e-12)
                    if float(track["heading"][onset]) != example["onset_heading"]:
                        raise ValueError("source onset heading differs")
                    matrix = onset_matrix(
                        example["onset_centroid"], example["onset_heading"], family["crop_native_pixels"]
                    )
                    meta = by_clip[example["filename"]]
                    file = output / example["filename"]
                    if digest(file) != meta["sha256"] or meta["example"] != example:
                        raise ValueError("component media identity differs")
                    np.testing.assert_allclose(matrix, meta["transform"], atol=1e-12)
                    native, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
                    if len(native) != meta["frames"] or (int(native[0]), int(native[-1])) != (
                        meta["source_native_first"],
                        meta["source_native_last"],
                    ):
                        raise ValueError("native/actual camera timing differs")
                    loc = np.minimum(np.searchsorted(track["frames"], native), len(track["frames"]) - 1)
                    known = (track["frames"][loc] == native) & np.isfinite(track["centroid"][loc]).all(axis=1)
                    active = (native >= example["start_frame"]) & (native < example["stop_frame"])
                    active_missing += int(np.sum(active & ~known))
                    context_missing += int(np.sum(~active & ~known))
                    dots = np.c_[track["centroid"][loc[known]], np.ones(known.sum())] @ matrix.T
                    offscreen += int(np.sum(np.any((dots < 0) | (dots >= 720), axis=1)))
                    middle = int(np.flatnonzero(active)[np.sum(active) // 2])
                    component = cv2.VideoCapture(str(file))
                    try:
                        for index in sorted({0, middle, len(native) - 1}):
                            component.set(cv2.CAP_PROP_POS_FRAMES, index)
                            ok, image = component.read()
                            if not ok:
                                raise ValueError("component seek failed")
                            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                            ok, combined = capture.read()
                            if not ok:
                                raise ValueError("grid seek failed")
                            tile = combined[:, panel * 720 : (panel + 1) * 720]
                            error = float(np.mean(np.abs(tile.astype(float) - image.astype(float))))
                            if error > 5:
                                raise ValueError("grid changed execution timing or panel pixels")
                            max_error = max(max_error, error)
                            checks += 1
                            if index == middle:
                                x, y = np.rint(matrix @ np.r_[track["centroid"][loc[index]], 1]).astype(int)
                                colour = token_colour(family["cohort"], int(track["labels"][loc[index]]))
                                if np.max(np.abs(combined[y, panel * 720 + x].astype(float) - colour)) > 85:
                                    raise ValueError("encoded active source-syllable centroid dot differs")
                                dot_checks += 1
                        if len(native) < grid["frames"]:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, len(native))
                            ok, image = capture.read()
                            if not ok or np.max(np.abs(image[10, panel * 720 + 10].astype(float) - 18)) > 4:
                                raise ValueError("finished panel not dark at its own natural endpoint")
                            finished_checks += 1
                    finally:
                        component.release()
            finally:
                capture.release()
            selection_stats.append(
                dict(
                    family=family["stem"],
                    mode=mode,
                    actual_percentiles=[e["actual_percentile"] for e in examples],
                    animals=len({e["subject"] for e in examples}),
                )
            )
        print("validated", family["stem"], flush=True)
    if active_missing or offscreen:
        raise ValueError("active tracking missing or centroid outside crop")
    result = dict(
        status="PASSED",
        families=len(families),
        movies=len(grids),
        components=len(clips),
        all_movies_have_one_common_selection_setting=True,
        all_within_movie_calls_unique=True,
        frozen_source_calls_ranks_models_clocks_sequences_and_transforms="passed",
        full_pool_random_draws="passed",
        encoded_panel_frame_checks=checks,
        encoded_dot_checks=dot_checks,
        finished_panel_checks=finished_checks,
        max_mean_pixel_error=max_error,
        active_tracking_missing=active_missing,
        context_tracking_missing=context_missing,
        offscreen_dot_frames=offscreen,
        selection_stats=selection_stats,
        full_media_decode="all movie and component frames decoded by renderer",
        actual_browser_playback="not exercised",
        qualitative_interpretation="pending",
    )
    (output / ("preview_validation.json" if len(families) < 49 else "validation.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def write_review_files(plan, families, grids, summary, output, *, preview):
    gallery(plan, grids, output, preview)
    notes = output / "review_notes.csv"
    if not notes.exists():
        with notes.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "family",
                    "movie_setting",
                    "panel",
                    "mouse",
                    "actual_percentile",
                    "similarity",
                    "order",
                    "boundaries",
                    "decision",
                    "notes",
                )
            )
            writer.writerows(
                (
                    f["stem"],
                    e["selection_mode"],
                    e["panel_number"],
                    e["subject"],
                    e["actual_percentile"],
                    "",
                    "",
                    "",
                    "pending",
                    "",
                )
                for f in plan["families"]
                for e in f["examples"]
            )
    notes = output / "family_review_notes.csv"
    if not notes.exists():
        with notes.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "cohort",
                    "family",
                    "p80_similarity",
                    "random_similarity",
                    "top_similarity",
                    "shared_progression",
                    "meaningful_variants",
                    "boundary_issues",
                    "merge_split_concern",
                    "decision",
                    "notes",
                )
            )
            writer.writerows(
                (f["cohort"], f["stem"], "", "", "", "", "", "", "", "pending", "")
                for f in plan["families"]
                if f["examples"]
            )
    durations = summary["playback_s_by_setting"]
    budget_lines = "\n".join(
        f"| {mode} | {seconds / 60:.2f} min | {seconds * 3 / 60:.2f} min |" for mode, seconds in durations.items()
    )
    (output / "START_HERE.md").write_text(f"""# Start here: qualitative chain movies

Open [the viewer](index.html). Choose a family; **P80** is the default whole-video
setting. All three panels in every movie show executions of the same chain with
one common selection setting. Use Replay to focus on a different mouse.

- **p100 / top exemplars:** the exemplar and nearby executions, showing the clearest reference.
- **P80:** three executions near the same within-family percentile target.
- **Random:** three distinct executions sampled uniformly from the full eligible pool.

Actual ranks and mouse IDs are shown. Natural playback, onset-locked camera,
coloured centroid dot and live original syllable strip are retained. Finished
panels turn dark. Unknown annotations remain unknown.

Start with P80 movies; use random for promising/ambiguous families and the top
movie as a reference. Other modes are available, not mandatory extra viewings.
Record similarity, common progression, meaningful variation, boundaries and
possible merging/splitting in [family notes](family_review_notes.csv). Detailed
[execution notes](review_notes.csv) are optional. Names/decisions can stay provisional.

## Actual playback budget (excluding notes)

| Whole-video setting | All families, once | Three replays |
| --- | ---: | ---: |
{budget_lines}

This package contains {summary["movies"]} movies across {summary["families"]} families.
The same five configurations per cohort are retained. Counts refer to supported
original reference families, not every nominal factor or every fold. Full details
are in [methods](report.md), [manifest](manifest.json) and validation JSON.
""")
    status = "two-cohort preview" if preview else "full collection"
    (output / "report.md").write_text(f"""# Qualitative chain movies — 2026-10-15

This {status} implements the user's corrected legacy viewing structure: separate
p100, P80 and random **movie files**, each containing three executions of one
chain family selected with the same setting. The preceding mixed-selection
package is superseded. Source calls/ranks are reused; no refitting or new search.

## Presentation

One fixed onset-position/heading transform per execution, a moving centroid dot
in the active original source-syllable colour, actual ordered syllable IDs,
model-bound partial labels, durations and live highlighting. Unknowns remain
unknown. No generator-phase/source-syllable equivalence is implied. Crop scale is
common per family across all nine selected executions. No trajectory inset,
following, recentering, time warping or stage waiting. Full calls plus 0.5 s context;
finished panels turn dark with a Finished caption. Movies are 2160×960 H.264/
yuv420p at 60fps. Karen's verified native 60Hz policy and Jonathan's actual
CenterTop FP-system clock are preserved, distinct from AVI/model rates. No native
ID multiplication, interpolation over missing support or extrapolated endpoints.

## Fixed data and selection

49 supported /68 nominal original fold-0/replicate-0 families, ten settings
(five per cohort), 45,261 eligible occurrences. The 19 unused reference factors
remain unavailable. Eight recording-end calls have no stored half-open timestamp
and remain excluded under the existing timing policy. No reserved Karen animals,
phenotype/DA clip filtering, new KPMS, detector fit or signal processing.

The exact validated ranking is reused from `{plan["ranking_source"]}`. Its 32-phase
float64 descriptor has four equally weighted blocks: original 10D pose latents,
onset-relative displacement, relative heading sin/cos and log actual duration,
with saved training IQR scales. The animal-balanced squared-distance medoid is
the actual exemplar; ranks measure squared distance to that execution, with
within-family midranks (high=closer). Descriptor/time-phase sampling does not warp
movies. Scores are descriptive curation, not behavioural correctness or an
independent generator metric. This sampler differs from the historical density-
ratio/cosine methods; only the legacy whole-video grouping is restored. Settings
retain their unequal historical tuning and are not averaged across methods.

- **p100** is the top-exemplar viewing tier, not a claim that three distinct
  executions have exactly rank100. Include the exemplar; choose neighbours from
  the closest max(3,ceil(2% N)) occurrences, preferring distinct mice within that
  pool, then distinct-call fallback. Actual achieved ranks are shown.
- **P80:** select near rank80 within +/-5 percentile points. Prefer different
  mice within that band; use distinct-call fallback. If a discrete/small pool
  supplies fewer than three, take the three closest ranks and record actuals.
- **Random:** seeded uniform full-pool sample of three occurrences without
  replacement. No similarity/animal/condition/duration filtering. It may overlap
  either ranked movie. A common mode within a movie is enforced before encoding.

Percentile ranks/random draws are occurrence-based; frequent mice can contribute
more. Animal balancing applies to the exemplar reference. The full ranking table,
exact source identities, seed, actual values, selected native intervals, original
sequences, camera transforms and component/grid hashes are retained. No new
behavioural names or qualitative winners have been assigned.

## Viewing and budgets

See [Start here](START_HERE.md) and the initially blank [family notes](family_review_notes.csv).
P80 first, random as a robustness check and p100 as the clearest reference;
three passes of all three settings are not mandatory. The measured budget is:

| Setting | Once | Three replays |
| --- | ---: | ---: |
{budget_lines}

{summary["movies"]} movies, {summary["components"]} component clips,
{summary["decoded_movie_frames"]:,} movie and {summary["decoded_component_frames"]:,}
component frames fully decoded. Render/audit runtime {summary["elapsed_s"]:.2f}s,
peak process RSS {summary["peak_rss_mib"]:.1f}MiB. See summary/status/validation JSON.
This verifies media, provenance and timing; continuous qualitative interpretation
and actual-browser playback remain pending. Prior notes belong to different clips.

## Execution and safeguards

Detached nohup with a finite 7200s outer timeout, configurable 8GiB AS cap,
80GiB free-disk floor and 6GiB output cap. Two numerical/encoder threads, one
OpenCV/grid-filter thread, one ffmpeg grid decoder thread per input; CPU libx264
veryfast/crf20. No GPU/model backend needed. Tracking cache two recordings,
160MiB NPZ bound; cached ranking avoids descriptor recomputation. Each grid
encoding has a 600s timeout. Failure stops the job and records FAILED status;
no automatic refits or search restart. Frozen render sources are saved.
Independent source/rank/native-clock/transform/sequence checks plus encoded panel
and centroid-colour checks run after rendering. The final status is only COMPLETE
once that audit passes. Resource bounds and exact evidence are saved in manifest.
""")


def budget(output, args):
    if shutil.disk_usage(output).free < args.min_free_disk_gib * 1024**3:
        raise ValueError("free-disk floor reached")
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > args.output_cap_gib * 1024**3:
        raise ValueError("output cap reached")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--min-free-disk-gib", type=float, default=80)
    parser.add_argument("--output-cap-gib", type=float, default=6)
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                dict(
                    ranking_source=str(args.ranking_source.resolve()),
                    output=str(args.output.resolve()),
                    scope="147 separate p100/P80/random movies; three same-setting executions per movie; existing 49 supported families",
                    execute=False,
                )
            )
        )
        return
    if args.memory_gib <= 0 or args.min_free_disk_gib < 0 or args.output_cap_gib <= 0:
        raise ValueError("resource caps must be positive; free-disk floor may be zero")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    limit = int(args.memory_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    import cv2

    cv2.setNumThreads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        budget(output, args)
        manifest = output / "manifest.json"
        plan = (
            json.loads(manifest.read_text())
            if manifest.exists()
            else build_separate_review(args.ranking_source.resolve(), output)
        )
        if plan.get("view_mode") != "separate_p100_p80_random_movies" or plan["ranking_source"] != str(
            args.ranking_source.resolve()
        ):
            raise ValueError("wrong output/source selection plan")
        if digest(Path(plan["ranking_source"]) / "manifest.json") != plan["ranking_source_manifest_sha256"]:
            raise ValueError("ranking source changed")
        plan["resources"].update(
            address_space_gib=args.memory_gib,
            min_free_disk_gib=args.min_free_disk_gib,
            output_cap_gib=args.output_cap_gib,
            grid_filter_threads=1,
            grid_decoder_threads_per_input=1,
        )
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
        for module, name in (
            (Path(__file__), "grid_render_source.py"),
            (Path(__file__).with_name("recreate_generator_videos.py"), "clip_render_source.py"),
        ):
            destination = output / name
            if destination.exists() and digest(destination) != digest(module):
                raise ValueError("renderer source changed; use a new output directory")
            destination.write_text(module.read_text())
        _, _, rows = inputs(Path(plan["source"]))
        tracks, clocks = Tracks(rows), ClockCache()
        families = [f for f in plan["families"] if f["examples"]]
        if args.preview:
            families = [next(f for f in families if f["cohort"] == c) for c in ("karen", "jonathan")]
        grids, clips = [], []
        for family in families:
            for mode in ("p100", "p80", "random"):
                own = []
                for example in [e for e in family["examples"] if e["selection_mode"] == mode]:
                    budget(output, args)
                    destination = output / example["filename"]
                    sidecar = destination.with_suffix(".json")
                    if sidecar.exists():
                        result = json.loads(sidecar.read_text())
                        if (
                            digest(destination) != result["sha256"]
                            or result["example"] != example
                            or result["crop_native_pixels"] != family["crop_native_pixels"]
                        ):
                            raise ValueError("completed component or selection changed")
                    else:
                        result = render_example(family, example, plan, output, tracks, clocks)
                    own.append(result)
                budget(output, args)
                sidecar = output / (family["stem"] + "_" + mode + "_grid.json")
                if sidecar.exists():
                    grid = json.loads(sidecar.read_text())
                    if digest(output / grid["filename"]) != grid["sha256"] or grid["clips"] != [
                        dict(filename=c["filename"], sha256=c["sha256"], frames=c["frames"]) for c in own
                    ]:
                        raise ValueError("completed movie or panel identity changed")
                else:
                    grid = render_grid(family, own, output, selection_mode=mode)
                grids.append(grid)
                clips.extend(own)
                (output / "status.json").write_text(
                    json.dumps(dict(status="RENDERING", movies=len(grids), last=grid["filename"])) + "\n"
                )
                print("movie", grid["filename"], grid["playback_s"], flush=True)
        (output / "status.json").write_text(json.dumps(dict(status="VALIDATING", movies=len(grids))) + "\n")
        validation = validate_collection(plan, families, grids, clips, output)
        summary = dict(
            status="PREVIEW_COMPLETE" if args.preview else "COMPLETE",
            families=len(families),
            movies=len(grids),
            components=len(clips),
            playback_s_by_setting={
                m: sum(g["playback_s"] for g in grids if g["selection_mode"] == m) for m in ("p100", "p80", "random")
            },
            decoded_movie_frames=sum(g["frames"] for g in grids),
            decoded_component_frames=sum(c["frames"] for c in clips),
            elapsed_s=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            active_tracking_missing=validation["active_tracking_missing"],
            context_tracking_missing=validation["context_tracking_missing"],
            offscreen_dot_frames=validation["offscreen_dot_frames"],
            unsupported_families=[f["stem"] for f in plan["families"] if not f["examples"]],
            source_model_refits=0,
            qualitative_review="pending",
        )
        write_review_files(plan, families, grids, summary, output, preview=args.preview)
        for name in ("preview_summary.json",) if args.preview else ("summary.json",):
            (output / name).write_text(json.dumps(summary, indent=2) + "\n")
        (output / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        (output / "status.json").write_text(json.dumps(dict(status="FAILED", error=str(exc))) + "\n")
        raise


if __name__ == "__main__":
    main()
