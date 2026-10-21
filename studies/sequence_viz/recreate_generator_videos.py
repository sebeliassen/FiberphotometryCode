"""Onset-locked, natural-time generator review with source-syllable progression.

Only completed reference-fit calls and fingerprinted data-only tracks are used.
Exemplarness is an explicit descriptive execution ranking, not detector confidence.
"""

import argparse
import colorsys
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
from collections import OrderedDict, defaultdict
from pathlib import Path

from studies.sequence_viz.render_generator_screening import ClockCache, digest, sample_indexes


def onset_matrix(centroid, heading, window, output_size=720):
    """Fixed native-pixel to view transform: starting heading points right."""
    import numpy as np

    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, s], [-s, c]], dtype=float)
    scale = output_size / window
    return np.column_stack([scale * rotation, output_size / 2 - scale * rotation @ centroid])


def execution_descriptors(track, clock, calls, scales, *, samples=32):
    """Batch fixed-time-phase original latents, displacement, heading and duration.

    Phase sampling is for ranking only. No video or label timing is warped.
    Missing support is unavailable, not interpolated or repaired.
    """
    import numpy as np

    starts = np.array([c["start_frame"] for c in calls])
    stops = np.array([c["stop_frame"] for c in calls])
    timed = (stops < len(clock)) & (starts >= 0)
    safe_stop = np.minimum(stops, len(clock) - 1)
    duration = clock[safe_stop] - clock[starts]
    targets = clock[starts, None] + duration[:, None] * np.arange(samples)[None, :] / samples
    native = np.searchsorted(clock, targets, side="right") - 1
    native = np.maximum(starts[:, None], np.minimum(native, stops[:, None] - 1))
    positions = np.searchsorted(track["frames"], native)
    positions = np.minimum(positions, len(track["frames"]) - 1)
    onset = np.minimum(np.searchsorted(track["frames"], starts), len(track["frames"]) - 1)
    valid = timed & (duration > 0) & (track["frames"][onset] == starts)
    valid &= np.all(track["frames"][positions] == native, axis=1)
    pose = track["latent"][positions] / np.asarray(scales["latent"])[None, None, :]
    xy = track["centroid"][positions] - track["centroid"][onset, None, :]
    h0 = track["heading"][onset]
    c, s = np.cos(h0)[:, None], np.sin(h0)[:, None]
    moved = np.stack([xy[:, :, 0] * c + xy[:, :, 1] * s, -xy[:, :, 0] * s + xy[:, :, 1] * c], axis=-1)
    moved /= scales["path"]
    heading = track["heading"][positions] - h0[:, None]
    turned = np.stack([np.sin(heading), np.cos(heading)], axis=-1)
    blocks = [
        pose.reshape(len(calls), -1),
        moved.reshape(len(calls), -1),
        turned.reshape(len(calls), -1),
        np.log(np.maximum(duration, 1e-12))[:, None],
    ]
    # Equal contribution per block, rather than weighting ten latent channels tenfold.
    values = np.concatenate([b / math.sqrt(4 * b.shape[1]) for b in blocks], axis=1)
    valid &= np.isfinite(values).all(axis=1)
    return values, valid, duration


def rank_executions(values, subjects):
    """Exact animal-balanced squared-Euclidean medoid ranking in O(ND), no NxN array.

    Sum_j w_j ||x_i-x_j||² differs from ||x_i-weighted_mean||² by a constant.
    Midrank percentiles keep tied executions tied; larger means more exemplary.
    """
    import numpy as np
    from scipy.stats import rankdata

    subjects = np.asarray(subjects)
    weights = np.empty(len(subjects), float)
    for subject in np.unique(subjects):
        own = subjects == subject
        weights[own] = 1 / own.sum()
    weights /= weights.sum()
    centre = weights @ values
    distance = np.mean((values - centre) ** 2, axis=1) * values.shape[1]
    if len(values) == 1:
        percentiles = np.array([100.0])
    else:
        percentiles = 100 * (len(values) - rankdata(distance, method="average")) / (len(values) - 1)
    return distance, percentiles


def select_views(calls, distance, percentiles, seed):
    import numpy as np

    if not calls:
        return []
    available, used_animals, selections = set(range(len(calls))), set(), []
    exemplar = min(available, key=lambda i: (distance[i], calls[i]["recording"], calls[i]["start_frame"]))
    for role, target in (("exemplar", 100), ("p90", 90), ("p50", 50), ("p10", 10), ("random", None)):
        if not available:
            break
        if role == "exemplar":
            selected = exemplar
        elif role == "random":
            selected = int(np.random.default_rng(seed).choice(sorted(available)))
        else:
            # Animal diversity is a preference only inside a declared ±5-percentile band.
            band = [i for i in available if abs(percentiles[i] - target) <= 5]
            pool = band or sorted(available)
            selected = min(
                pool,
                key=lambda i: (
                    bool(band) and calls[i]["subject"] in used_animals,
                    abs(percentiles[i] - target),
                    calls[i]["recording"],
                    calls[i]["start_frame"],
                ),
            )
        selections.append(
            dict(
                **calls[selected],
                role=role,
                requested_percentile=target,
                actual_percentile=float(percentiles[selected]),
                distance_squared=float(distance[selected]),
                random_seed=seed,
            )
        )
        available.remove(selected)
        used_animals.add(calls[selected]["subject"])
    return selections


def rank_exemplar_neighbors(values, subjects):
    """Keep the animal-balanced medoid; rank distance to that actual execution."""
    import numpy as np
    from scipy.stats import rankdata

    centre_distance, _ = rank_executions(values, subjects)
    exemplar = int(np.argmin(centre_distance))
    distance = np.sum((values - values[exemplar]) ** 2, axis=1)
    percentile = (
        np.array([100.0])
        if len(values) == 1
        else 100 * (len(values) - rankdata(distance, method="average")) / (len(values) - 1)
    )
    return distance, percentile, exemplar


def select_grid_views(calls, distance, percentiles, seed, exemplar):
    """Exemplar, nearest p80 and an unconditional uniform occurrence draw.

    Random includes the exemplar/p80 and every eligible animal. Repeats are
    retained and flagged rather than redrawn. p80 prefers a different animal
    only when percentile-target errors tie exactly.
    """
    import numpy as np

    if not calls:
        return []
    chosen = [("exemplar", 100, exemplar)]
    others = [i for i in range(len(calls)) if i != exemplar]
    if others:
        near = min(
            others,
            key=lambda i: (
                abs(percentiles[i] - 80),
                calls[i]["subject"] == calls[exemplar]["subject"],
                calls[i]["recording"],
                calls[i]["start_frame"],
            ),
        )
        chosen.append(("p80", 80, near))
    chosen.append(("random", None, int(np.random.default_rng(seed).integers(len(calls)))))
    result = []
    for role, target, index in chosen:
        same = [old_role for old_role, _, old_index in chosen[: len(result)] if old_index == index]
        result.append(
            dict(
                **calls[index],
                role=role,
                requested_percentile=target,
                actual_percentile=float(percentiles[index]),
                distance_squared=float(distance[index]),
                random_seed=seed,
                repeats_roles=same,
            )
        )
    return result


def syllable_segments(track, clock, start, stop):
    import numpy as np

    a, b = np.searchsorted(track["frames"], [start, stop])
    frames, labels = track["frames"][a:b], track["labels"][a:b]
    if len(frames) != stop - start or frames[0] != start or np.any(np.diff(frames) != 1) or np.any(labels < 0):
        raise ValueError("chain source syllables cross missing or excluded frames")
    breaks = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    result = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        begin, end = int(frames[lo]), int(frames[hi - 1] + 1)
        result.append(
            dict(
                syllable_id=int(labels[lo]),
                start_frame=begin,
                stop_frame=end,
                duration_s=float(clock[end] - clock[begin]),
            )
        )
    return result


class Tracks:
    """Two-recording cache, using only the original verified native joins."""

    def __init__(self, rows):
        self.rows, self.values, self.hashes = rows, OrderedDict(), {}

    def get(self, recording):
        import numpy as np

        from studies.sequence_viz.broad_search_inputs import _load_npz, join_verified_native_features

        if recording not in self.values:
            row = self.rows[recording]
            if row["exclude_from_search"] or not row["native_mapping_verified"]:
                raise ValueError("reserved or unsupported recording requested")
            provenance = row["source_provenance"]
            data = {}
            for key in ("source", "sparse_features", "dense_features"):
                if key not in provenance:
                    continue
                info = provenance[key]
                if info["path"] not in self.hashes:
                    if digest(info["path"]) != info["sha256"]:
                        raise ValueError("native tracking fingerprint changed")
                    self.hashes[info["path"]] = info["sha256"]
                data[key] = _load_npz(info["path"], max_bytes=160 * 1024**2)
            if "sparse_features" in data:
                track = join_verified_native_features(data["source"], data["sparse_features"], data["dense_features"])
            else:
                original = data["source"]
                supported = original["labels"] >= 0
                if "analysis_excluded" in original:
                    supported &= ~original["analysis_excluded"]
                track = {k: original[k][supported] for k in ("frames", "labels", "centroid", "heading", "latent")}
            if np.any(np.diff(track["frames"]) <= 0):
                raise ValueError("native tracking coordinates not increasing")
            self.values[recording] = track
            if len(self.values) > 2:
                self.values.popitem(last=False)
        self.values.move_to_end(recording)
        return self.values[recording]


def inputs(source):
    old = json.loads((source / "manifest.json").read_text())
    run = json.loads(Path(old["source_manifest"]).read_text())
    if (
        digest(old["source_manifest"]) != old["source_manifest_sha256"]
        or digest(run["manifest"]) != old["input_manifest_sha256"]
    ):
        raise ValueError("frozen verification manifests changed")
    frozen = json.loads(Path(run["manifest"]).read_text())
    rows = {r["recording"]: r for c in frozen["cohorts"] for r in c["records"]}
    return old, frozen, rows


def build_review(source, output, *, grid=False):
    import numpy as np

    old, frozen, rows = inputs(source)
    tracks, clocks = Tracks(rows), ClockCache()
    grouped, artifacts, unavailable = defaultdict(list), {}, defaultdict(int)
    entries = {e["stem"]: e for e in old["entries"]}
    rank_path = output / "execution_ranking.csv.gz"
    # Process each recording once; all five reference models share the original held fold.
    paths = sorted(
        old["artifacts"], key=lambda p: (json.loads(gzip.decompress(Path(p).read_bytes()))["identity"]["recording"], p)
    )
    for path in paths:
        if digest(path) != old["artifacts"][path]:
            raise ValueError("completed calls changed")
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        identity, prediction = data["identity"], data["prediction"]
        if identity["fold"] != 0 or identity["replicate"] != 0:
            raise ValueError("only original reference catalogue allowed")
        recording = identity["recording"]
        track, clock = tracks.get(recording), clocks.get(old["bindings"][recording])
        names, scales = identity["feature_names"], identity["feature_scales"]
        scale = dict(
            latent=[scales[names.index("latent_mean_" + str(i))] for i in range(10)],
            path=scales[names.index("path_length_px")],
        )
        calls = prediction["selected_calls"]
        artifacts[path] = old["artifacts"][path]
        for start in range(0, len(calls), 256):
            batch = calls[start : start + 256]
            vectors, valid, duration = execution_descriptors(track, clock, batch, scale)
            for number, call in enumerate(batch):
                stem = identity["panel_id"] + "_" + call["type_id"]
                if stem not in entries or entries[stem]["model_key"] != identity["model_key"]:
                    raise ValueError("local chain/model identity differs")
                if not valid[number]:
                    unavailable[stem] += 1
                    continue
                if not any(
                    a <= call["start_frame"] < call["stop_frame"] <= b for a, b in prediction["valid_intervals"]
                ):
                    raise ValueError("call crosses model support gap")
                grouped[stem].append(
                    (
                        dict(
                            **call,
                            recording=recording,
                            subject=identity["subject"],
                            metadata=identity["metadata"],
                            duration_s=float(duration[number]),
                            artifact=path,
                        ),
                        vectors[number].copy(),
                    )
                )
        print("profiled", identity["panel_id"], recording, len(calls), flush=True)
    result = []
    with gzip.open(rank_path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "family",
                "recording",
                "subject",
                "start_frame",
                "stop_frame",
                "distance_squared",
                "exemplarness_percentile",
            )
        )
        for stem, entry in entries.items():
            pairs = grouped.pop(stem, [])
            own = dict(
                stem=stem,
                cohort=entry["cohort"],
                panel_id=entry["panel_id"],
                type_id=entry["type_id"],
                recipe=entry["recipe"],
                model_key=entry["model_key"],
                source_fit=entry["source_fit"],
                eligible_calls=len(pairs),
                unavailable_calls=unavailable[stem],
                original_calls=entry["available_calls"],
            )
            if pairs:
                calls = [c for c, _ in pairs]
                values = np.stack([v for _, v in pairs])
                if grid:
                    distance, percentiles, exemplar = rank_exemplar_neighbors(values, [c["subject"] for c in calls])
                else:
                    distance, percentiles = rank_executions(values, [c["subject"] for c in calls])
                seed = int(hashlib.sha256(("chain-review-20261015:" + stem).encode()).hexdigest()[:8], 16)
                own["examples"] = (
                    select_grid_views(calls, distance, percentiles, seed, exemplar)
                    if grid
                    else select_views(calls, distance, percentiles, seed)
                )
                own["eligible_animals"] = len({c["subject"] for c in calls})
                own["descriptor_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
                for c, d, percentile in zip(calls, distance, percentiles):
                    writer.writerow(
                        (stem, c["recording"], c["subject"], c["start_frame"], c["stop_frame"], d, percentile)
                    )
            else:
                own["examples"], own["eligible_animals"] = [], 0
            result.append(own)
    for family in result:
        extent = 160.0
        for example in family["examples"]:
            track, binding = tracks.get(example["recording"]), old["bindings"][example["recording"]]
            clock = clocks.get(binding)
            example["segments"] = syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
            onset = int(np.searchsorted(track["frames"], example["start_frame"]))
            example["onset_centroid"] = track["centroid"][onset].tolist()
            example["onset_heading"] = float(track["heading"][onset])
            indexes, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
            loc = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
            known = (track["frames"][loc] == indexes) & np.isfinite(track["centroid"][loc]).all(axis=1)
            xy = track["centroid"][loc[known]] - np.asarray(example["onset_centroid"])
            heading = example["onset_heading"]
            rot = np.array([[math.cos(heading), math.sin(heading)], [-math.sin(heading), math.cos(heading)]])
            extent = max(extent, float(np.max(np.abs(xy @ rot.T))) + 55)
            example["frames"] = len(indexes)
            example["playback_s"] = len(indexes) / 60
            example["filename"] = family["stem"] + "_" + example["role"] + ".mp4"
        # One constant scale per family, planned from selected examples; never follows during playback.
        family["crop_native_pixels"] = int(16 * math.ceil(2 * extent / 16))
        if family["crop_native_pixels"] > 2048:
            raise ValueError("tracking extent exceeds bounded crop; inspect before rendering")
    manifest = dict(
        view_mode="grid_exemplar_p80_random" if grid else "independent_five_choices",
        source=str(source),
        source_manifest_sha256=digest(source / "manifest.json"),
        frozen_input_manifest_sha256=digest(Path(json.loads(Path(old["source_manifest"]).read_text())["manifest"])),
        families=result,
        bindings=old["bindings"],
        track_hashes=tracks.hashes,
        artifact_hashes=artifacts,
        annotation_mappings={c["cohort_id"]: c["annotation_mapping"] for c in frozen["cohorts"]},
        ranking=dict(
            samples=32,
            dtype="float64",
            blocks="equal weights: original latent pose / onset-relative displacement / relative heading sin-cos / log actual duration",
            scaling="original reference-training latent-mean and path-length IQRs; heading dimensionless; log duration",
            centre="animal-balanced mean; actual nearest execution is weighted squared-Euclidean medoid",
            distance_reference="actual animal-balanced medoid execution" if grid else "animal-balanced mean",
            percentile=(
                "within family midrank; high=close to actual exemplar; not behavioural correctness"
                if grid
                else "within family midrank; high=close to centre; not behavioural correctness"
            ),
            selection=(
                "exemplar; nearest p80 (other animal only breaks exact ties); unconditional seeded uniform random from full eligible occurrence pool, repeats retained/flagged"
                if grid
                else "exemplar; p90/p50/p10 with animal diversity within ±5percentile band; seeded uniform random among remaining calls; no duplicate executions"
            ),
            timing="equal actual elapsed-time phases for ranking only; videos never warped",
            ranking_sha256=digest(rank_path),
        ),
        render=dict(
            fps=60,
            video_size=720,
            hud_height=240,
            context_s=0.5,
            crop="one fixed onset position/heading transform for entire clip; common planned pixel scale per family",
            dot="current source centroid in active syllable colour; unsupported tracking omitted",
            annotations="original model-bound partial labels; no phase/syllable equivalence",
        ),
        resources=dict(
            address_space_gib=8, threads=2, opencv_threads=1, min_free_disk_gib=80, output_cap_gib=6, timeout_s=7200
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def token_colour(cohort, token):
    # Stable within vocabulary; no cross-cohort identity implied by the colour.
    code = int(hashlib.sha256(f"{cohort}:{token}".encode()).hexdigest()[:8], 16)
    rgb = colorsys.hsv_to_rgb((code % 360) / 360, 0.65, 0.95)
    return tuple(int(round(255 * x)) for x in rgb[::-1])


def draw_hud(family, example, frame, relative, current_token, mapping):
    import cv2
    import numpy as np

    hud = np.full((240, 720, 3), 18, np.uint8)
    active = example["start_frame"] <= frame < example["stop_frame"]
    segments = example["segments"]
    current = next((i for i, s in enumerate(segments) if s["start_frame"] <= frame < s["stop_frame"]), -1)
    label = mapping.get(str(current_token), "unlabelled").removeprefix("annotation:")
    if label.startswith("unmapped:"):
        label = "unlabelled"
    top = [
        f"{family['cohort']} {family['stem']} | {example['role']} | actual percentile {example['actual_percentile']:.1f}",
        f"Mouse {example['subject']} | {relative:+.2f}s | {'CHAIN' if active else 'context'} | native {frame}",
        f"Current source syllable {current_token if current_token is not None else 'unavailable'}: {label}",
    ]
    for i, text in enumerate(top):
        cv2.putText(
            hud,
            text[:100],
            (12, 22 + 25 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (100, 235, 150) if active else (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    begin = max(0, min(max(0, len(segments) - 16), current - 7))
    for index, segment in enumerate(segments[begin : begin + 16], start=begin):
        local = index - begin
        x, y = 12 + (local % 8) * 87, 88 + (local // 8) * 59
        colour = token_colour(family["cohort"], segment["syllable_id"])
        cv2.rectangle(hud, (x, y), (x + 82, y + 53), colour if index == current else (30, 30, 30), -1)
        cv2.rectangle(
            hud, (x, y), (x + 82, y + 53), (245, 245, 245) if index == current else colour, 2 if index == current else 1
        )
        text_colour = (5, 5, 5) if index == current else (215, 215, 215)
        name = mapping.get(str(segment["syllable_id"]), "?").removeprefix("annotation:")
        if name.startswith("unmapped:"):
            name = "?"
        for line, text in enumerate(
            (f"{index + 1}: S{segment['syllable_id']}", name[:12], f"{segment['duration_s']:.2f}s")
        ):
            cv2.putText(
                hud, text, (x + 3, y + 14 + 16 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_colour, 1, cv2.LINE_AA
            )
    cv2.putText(
        hud,
        f"Observed sequence {begin + 1}-{min(begin + 16, len(segments))} of {len(segments)} | onset-locked camera | natural speed",
        (12, 232),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return hud


def render_example(family, example, plan, output, tracks, clocks):
    import cv2
    import numpy as np

    binding = plan["bindings"][example["recording"]]
    source = Path(binding["video_path"])
    if (source.stat().st_size, source.stat().st_mtime_ns) != (binding["video_size"], binding["video_mtime_ns"]):
        raise ValueError("native video changed")
    if binding["timestamp_path"] and digest(binding["timestamp_path"]) != binding["timestamp_sha256"]:
        raise ValueError("camera timestamp source changed")
    track, clock = tracks.get(example["recording"]), clocks.get(binding)
    indexes, relative = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
    transform = onset_matrix(example["onset_centroid"], example["onset_heading"], family["crop_native_pixels"])
    path = output / example["filename"]
    temporary = path.with_suffix(".partial.mp4")
    capture = cv2.VideoCapture(str(source))
    process = None
    posters, missing_tracking, offscreen = [], 0, 0
    points = set(np.linspace(0, len(indexes) - 1, 8).astype(int))
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "720x960",
        "-r",
        "60",
        "-i",
        "pipe:0",
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
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, int(indexes[0])):
            raise ValueError("native seek failed")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        next_index, image = int(indexes[0]), None
        for output_index, (index, elapsed) in enumerate(zip(indexes, relative)):
            while next_index <= index:
                ok, image = capture.read()
                if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != next_index + 1:
                    raise ValueError("native decode/seek position mismatch")
                next_index += 1
            tile = cv2.warpAffine(
                image,
                transform,
                (720, 720),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            position = int(np.searchsorted(track["frames"], index))
            token = None
            if position < len(track["frames"]) and track["frames"][position] == index:
                token = int(track["labels"][position])
                centroid = track["centroid"][position]
                if np.isfinite(centroid).all():
                    dot = transform @ np.r_[centroid, 1.0]
                    if np.all((dot >= 0) & (dot < 720)):
                        point = tuple(np.rint(dot).astype(int))
                        cv2.circle(tile, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(tile, point, 5, token_colour(family["cohort"], token), -1, cv2.LINE_AA)
                    else:
                        offscreen += 1
                else:
                    missing_tracking += 1
            else:
                missing_tracking += 1
            hud = draw_hud(
                family, example, int(index), float(elapsed), token, plan["annotation_mappings"][family["cohort"]]
            )
            full = np.concatenate([tile, hud], axis=0)
            process.stdin.write(full.tobytes())
            if output_index in points:
                posters.append(full)
        process.stdin.close()
        errors = process.stderr.read().decode()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("encoder failed: " + errors[-1500:])
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    temporary.replace(path)
    poster = path.with_suffix(".png")
    sheet = path.with_suffix(".jpg")
    cv2.imwrite(str(poster), posters[min(2, len(posters) - 1)])
    # Eight time points from ONE execution, laid out as two rows of four.
    thumbs = [cv2.resize(p, (360, 480)) for p in posters]
    thumbs += [np.zeros((480, 360, 3), np.uint8)] * (8 - len(thumbs))
    cv2.imwrite(
        str(sheet), np.concatenate([np.concatenate(thumbs[:4], axis=1), np.concatenate(thumbs[4:8], axis=1)], axis=0)
    )
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while capture.read()[0]:
            count += 1
        if count != len(indexes) or abs(capture.get(cv2.CAP_PROP_FPS) - 60) > 0.001:
            raise ValueError("full encoded playback validation failed")
    finally:
        capture.release()
    result = dict(
        filename=example["filename"],
        frames=count,
        playback_s=count / 60,
        sha256=digest(path),
        transform=transform.tolist(),
        crop_native_pixels=family["crop_native_pixels"],
        source_native_first=int(indexes[0]),
        source_native_last=int(indexes[-1]),
        dot_offscreen_frames=offscreen,
        missing_tracking_context_or_call_frames=missing_tracking,
        encoder="CPU libx264,60fps actual-time resampling,crf20,2threads",
        example=example,
    )
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def gallery(plan, results, output, *, preview):
    supported = [f for f in plan["families"] if f["examples"]]
    available = {r["filename"] for r in results}
    payload = []
    for f in supported:
        es = [e for e in f["examples"] if e["filename"] in available]
        if es:
            payload.append(
                dict(
                    stem=f["stem"],
                    cohort=f["cohort"],
                    pool=f["eligible_calls"],
                    animals=f["eligible_animals"],
                    examples=es,
                    mapping=plan["annotation_mappings"][f["cohort"]],
                )
            )
    page = """<!doctype html><meta charset="utf-8"><title>Chain review</title>
<style>body{font:16px system-ui;background:#16181d;color:#eee;max-width:1100px;margin:24px auto;padding:0 16px}select,button{font:inherit;padding:8px;margin:4px}video{display:block;width:min(100%,600px);max-height:76vh;margin:12px auto}#sequence{line-height:1.9}a{color:#9cd1ff}.muted{color:#bfc3cd}</style>
<h1>Chain review: independent natural-speed clips</h1>
<p>Camera fixed to position and heading at chain onset. Dot and syllable strip share colours. No trajectory inset or time warping.</p>
<p class="muted">Start with p90; other percentiles and random are optional checks, not extra required viewing. Percentiles describe execution similarity within this family, not behavioural correctness. Random samples uniformly among remaining occurrences after the four ranked selections.</p>
<button id="previous">Previous chain</button><select id="family"></select><button id="next">Next chain</button><div id="choices"></div><p id="details"></p>
<video id="player" controls preload="none"></video><p id="sequence"></p><p><a id="direct">Open video file</a> · <a href="report.md">Methods and review budget</a> · <a href="review_notes.csv">Notes sheet</a></p>
<script>const families=PAYLOAD; const picker=document.getElementById('family'), player=document.getElementById('player');
families.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.cohort+' / '+f.stem;picker.append(o)});
function choose(f,e){player.pause();player.src=e.filename;player.poster=e.filename.replace('.mp4','.png');document.getElementById('direct').href=e.filename;
document.getElementById('details').textContent=e.role+' — actual percentile '+e.actual_percentile.toFixed(1)+'; mouse '+e.subject+'; eligible pool '+f.pool+' calls / '+f.animals+' animals';
document.getElementById('sequence').textContent='Actual observed chain: '+e.segments.map(s=>'S'+s.syllable_id+' '+(f.mapping[s.syllable_id]||'unlabelled').replace('annotation:','')+' ('+s.duration_s.toFixed(2)+'s)').join(' → ');}
function change(){const f=families[+picker.value], choices=document.getElementById('choices');choices.replaceChildren();f.examples.forEach(e=>{const b=document.createElement('button');b.textContent=e.role==='exemplar'?'Most exemplary':e.role;b.onclick=()=>choose(f,e);choices.append(b)});choose(f,f.examples.find(e=>e.role==='p90')||f.examples[0]);}
picker.onchange=change;document.getElementById('previous').onclick=()=>{picker.value=(+picker.value+families.length-1)%families.length;change()};document.getElementById('next').onclick=()=>{picker.value=(+picker.value+1)%families.length;change()};if(families.length)change();</script>"""
    (output / ("preview.html" if preview else "index.html")).write_text(
        page.replace("PAYLOAD", json.dumps(payload).replace("</", "<\\/"))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                dict(
                    source=str(args.source.resolve()),
                    output=str(args.output.resolve()),
                    scope="existing ten reference configurations; exemplar,90th,50th,10th,random per supported type",
                    execute=False,
                )
            )
        )
        return
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    import cv2

    cv2.setNumThreads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = output / "manifest.json"
    plan = json.loads(manifest.read_text()) if manifest.exists() else build_review(args.source.resolve(), output)
    if (
        plan["source"] != str(args.source.resolve())
        or digest(args.source / "manifest.json") != plan["source_manifest_sha256"]
    ):
        raise ValueError("review source changed")
    old, _, rows = inputs(args.source.resolve())
    tracks, clocks = Tracks(rows), ClockCache()
    (output / "render_source.py").write_text(Path(__file__).read_text())
    families = [f for f in plan["families"] if f["examples"]]
    if args.preview:
        families = [next(f for f in families if f["cohort"] == c) for c in ("karen", "jonathan")]
    results = []
    for family in families:
        examples = (
            family["examples"] if not args.preview else [next(e for e in family["examples"] if e["role"] == "p90")]
        )
        for example in examples:
            if (
                shutil.disk_usage(output).free < 80 * 1024**3
                or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 6 * 1024**3
            ):
                raise ValueError("free-disk floor or output cap reached")
            destination = output / example["filename"]
            sidecar = destination.with_suffix(".json")
            if sidecar.exists():
                result = json.loads(sidecar.read_text())
                if digest(destination) != result["sha256"]:
                    raise ValueError("completed output changed")
            else:
                result = render_example(family, example, plan, output, tracks, clocks)
            results.append(result)
            (output / "status.json").write_text(
                json.dumps(dict(status="RUNNING", completed=len(results), last=example["filename"])) + "\n"
            )
            print(example["filename"], result["playback_s"], flush=True)
    summary = dict(
        status="PREVIEW_COMPLETE" if args.preview else "COMPLETE",
        files=len(results),
        families=len(families),
        unsupported_families=[f["stem"] for f in plan["families"] if not f["examples"]],
        all_optional_clips_playback_s=sum(r["playback_s"] for r in results),
        default_p90_playback_s=sum(r["playback_s"] for r in results if r["example"]["role"] == "p90"),
        decoded_frames=sum(r["frames"] for r in results),
        elapsed_s=time.monotonic() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        dot_offscreen_frames=sum(r["dot_offscreen_frames"] for r in results),
        missing_tracking_frames=sum(r["missing_tracking_context_or_call_frames"] for r in results),
        source_model_refits=0,
        review_status="media validated; qualitative interpretation pending",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    gallery(plan, results, output, preview=args.preview)
    notes = output / "review_notes.csv"
    if not notes.exists():
        with notes.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "family",
                    "selection",
                    "actual_percentile",
                    "recording",
                    "mouse",
                    "within_type_similarity",
                    "action_order",
                    "boundaries",
                    "decision",
                    "notes",
                )
            )
            writer.writerows(
                (f["stem"], e["role"], e["actual_percentile"], e["recording"], e["subject"], "", "", "", "pending", "")
                for f in plan["families"]
                for e in f["examples"]
            )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
