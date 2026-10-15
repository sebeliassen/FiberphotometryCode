"""Group-blind screening of ten frozen generator catalogues, with native video clocks."""

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import resource
import shutil
from collections import OrderedDict
from pathlib import Path

PANEL = ("C05", "D03", "C01", "D02", "D04", "C06", "D10", "C09", "D09", "D14")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def choose_examples(calls):
    """Ordinary duration examples, animal diversity and a declared distinct-type foil.

    No score, phenotype, DA, annotation or favourable tracking filter is used.
    Foils are descriptive nearest matches, not a statistically matched control.
    """
    import numpy as np

    result = []
    for token in sorted({c["type_id"] for c in calls}):
        own = [c for c in calls if c["type_id"] == token]
        median = float(np.median([c["duration_s"] for c in own]))
        ranked = sorted(own, key=lambda c: (abs(c["duration_s"] - median), c["recording"], c["start_frame"]))
        first = ranked[0]
        other = next((c for c in ranked if c["subject"] != first["subject"]), None)
        if other is None:
            other = next((c for c in ranked if c["recording"] != first["recording"]), None)
        if other is None:
            other = next((c for c in ranked if c["start_frame"] != first["start_frame"]), None)
        alternatives = [
            c
            for c in calls
            if c["type_id"] != token
            and c["recording"] == first["recording"]
            and (c["stop_frame"] <= first["start_frame"] or c["start_frame"] >= first["stop_frame"])
        ]

        def distance(c):
            keys = set(first["parent_counts"]) | set(c["parent_counts"])
            a, b = sum(first["parent_counts"].values()), sum(c["parent_counts"].values())
            source_l1 = sum(abs(first["parent_counts"].get(k, 0) / a - c["parent_counts"].get(k, 0) / b) for k in keys)
            duration = abs(math.log(c["duration_s"] / first["duration_s"]))
            temporal = abs(c["start_frame"] - first["start_frame"]) / first["fps"]
            return (temporal > 300, source_l1 + duration, temporal, c["start_frame"])

        foil = min(alternatives, key=distance) if alternatives else None
        result.append(
            dict(
                type_id=token,
                available_calls=len(own),
                available_animals=len({c["subject"] for c in own}),
                panels=[first, other, foil],
                second_different_animal=bool(other and other["subject"] != first["subject"]),
                foil_policy="same recording; different detected type; prefer within300s; minimise original-source frequency L1 + absolute log-duration ratio",
                foil_distance=list(distance(foil)) if foil else None,
            )
        )
    return result


def sample_indexes(clock, start, stop, *, fps=30, context=0.5):
    """Real-time native lookup: never convert acquisition IDs using AVI header FPS."""
    import numpy as np

    clock = np.asarray(clock, float)
    if clock.ndim != 1 or not np.isfinite(clock).all() or np.any(np.diff(clock) < 0):
        raise ValueError("camera clock must be finite, nondecreasing")
    if not 0 <= start < stop < len(clock) or clock[stop] <= clock[start]:
        raise ValueError("call lacks an independently timed half-open endpoint")
    begin = max(clock[0], clock[start] - context)
    end = min(clock[-1], clock[stop] + context)
    targets = begin + np.arange(math.ceil((end - begin) * fps)) / fps
    indexes = np.searchsorted(clock, targets, side="right") - 1
    indexes = np.clip(indexes, 0, len(clock) - 1)
    return indexes, targets - clock[start]


class ClockCache:
    def __init__(self):
        self.values = OrderedDict()

    def get(self, binding):
        import numpy as np

        key = binding["timestamp_path"] or binding["video_path"]
        if key not in self.values:
            if binding["timestamp_path"]:
                self.values[key] = np.fromiter(
                    (float(row["FP3002_System_Timestamp"]) for row in csv.DictReader(Path(key).open(newline=""))), float
                )
            else:
                self.values[key] = np.arange(binding["video_frames"]) / binding["acquisition_fps"]
            value = self.values[key]
            if len(value) > binding["video_frames"] + 2 or len(value) < binding["video_frames"] - 2:
                raise ValueError("camera timestamp rows differ from native video by more than2")
            # A timestamp may be unavailable for the last one or two video frames.
            self.values[key] = value[: binding["video_frames"]]
            if not np.isfinite(value).all() or np.any(np.diff(value) < 0):
                raise ValueError("invalid native camera clock")
            if len(self.values) > 4:
                self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]


def build_plan(root, source):
    import cv2

    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads(Path(manifest["manifest"]).read_text())
    input_rows = {r["recording"]: r for c in frozen["cohorts"] for r in c["records"]}
    karen = {
        r["session_id"]: r
        for r in csv.DictReader(
            (root / "outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries/sessions.csv").open()
        )
    }
    native = json.loads((root / "outputs/broad_search_preflight_20261013/native_input_manifest.json").read_text())
    jonathan = {r["session_id"]: r for r in native["rows"]}
    audit = json.loads((root / "outputs/broad_search_preflight_20261013/amph_run12/manifest.json").read_text())
    metadata = {r["session_id"]: r["metadata"] for r in audit["sessions"]}
    bindings, entries, snapshots = {}, [], {}
    clock_cache = ClockCache()
    for panel in PANEL:
        jobs = [j for j in manifest["jobs"] if j["panel_id"] == panel and j["fold"] == 0 and j["replicate"] == 0]
        anchor = next(j for j in jobs if not j["recording"])
        if digest(anchor["saved"]["path"]) != anchor["saved"]["sha256"]:
            raise ValueError("saved catalogue fingerprint changed")
        original = json.loads(Path(anchor["saved"]["path"]).read_text())
        recipe, conditions = anchor["recipe"], original["result"]["conditions"]
        count = (
            recipe["factors"]
            if recipe["method"] == "seqnmf"
            else len(conditions["targets"] if recipe["method"] == "hsmm" else conditions["motifs"])
        )
        calls = []
        for job in jobs:
            if not job["recording"]:
                continue
            path = source / "recordings" / (job["job_id"] + ".json.gz")
            checked = json.loads((source / "jobs" / (job["job_id"] + ".json")).read_text())["result"]
            if digest(path) != checked["sha256"]:
                raise ValueError("completed recording fingerprint changed")
            data = json.loads(gzip.decompress(path.read_bytes()))
            identity, prediction = data["identity"], data["prediction"]
            if (
                identity["model_key"] != anchor["saved"]["cache"]["model_key"]
                or identity["fold"] != 0
                or identity["replicate"] != 0
            ):
                raise ValueError("reference catalogue identity differs")
            recording = identity["recording"]
            if input_rows[recording]["exclude_from_search"] or not input_rows[recording]["native_mapping_verified"]:
                raise ValueError("reserved or unverified recording")
            snapshots[str(path)] = checked["sha256"]
            if recording not in bindings:
                if identity["cohort"] == "karen":
                    video = (
                        root / "data/raw/karen_pff_vs_monomer/video_files" / Path(karen[recording]["video_path"]).name
                    )
                    timestamp = None
                else:
                    video = (
                        root
                        / "data/raw/oft_baseline/amph_vids_ssd"
                        / (jonathan[recording]["recording_key"].split("DLC_")[0] + ".avi")
                    )
                    stored = Path(metadata[recording]["alltimestamps_csv_path"])
                    timestamp = stored.parent / (
                        "CenterTopCam_AllTimestampsCombined" + stored.name.split("_AllTimestampsCombined", 1)[1]
                    )
                capture = cv2.VideoCapture(str(video))
                try:
                    if not capture.isOpened():
                        raise ValueError("missing model-bound native video: " + str(video))
                    header_fps = capture.get(cv2.CAP_PROP_FPS)
                    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    capture.release()
                if abs(header_fps - 60) > 0.001:
                    raise ValueError("unexpected source encoded FPS")
                bindings[recording] = dict(
                    video_path=str(video),
                    video_frames=frames,
                    encoded_fps=header_fps,
                    acquisition_fps=prediction["fps"],
                    timestamp_path=str(timestamp) if timestamp else None,
                    timestamp_sha256=digest(timestamp) if timestamp else None,
                    video_size=video.stat().st_size,
                    video_mtime_ns=video.stat().st_mtime_ns,
                    frame_mapping="identity: original native acquisition index names cropped CenterTop AVI frame; no VFI multiplication",
                    clock_policy="actual CenterTop FP-system seconds"
                    if timestamp
                    else "verified native60Hz acquisition",
                )
            timed = clock_cache.get(bindings[recording])
            descriptors = {
                (d["type_id"], d["start_frame"], d["stop_frame"]): d
                for d in data["descriptors"]
                if d["epoch"] == "whole_supported"
            }
            for call in prediction["selected_calls"]:
                if call["stop_frame"] >= len(timed):
                    continue
                descriptor = descriptors[(call["type_id"], call["start_frame"], call["stop_frame"])]
                if timed[call["stop_frame"]] <= timed[call["start_frame"]]:
                    continue
                if not any(
                    a <= call["start_frame"] < call["stop_frame"] <= b for a, b in prediction["valid_intervals"]
                ):
                    raise ValueError("selected call crosses unsupported native gap")
                calls.append(
                    dict(
                        **call,
                        recording=recording,
                        subject=identity["subject"],
                        metadata=identity["metadata"],
                        fps=prediction["fps"],
                        duration_s=float(timed[call["stop_frame"]] - timed[call["start_frame"]]),
                        parent_counts=descriptor["parent_counts"],
                        artifact=str(path),
                    )
                )
        selected = {e["type_id"]: e for e in choose_examples(calls)}
        for i in range(count):
            token = ("factor_" if recipe["method"] == "seqnmf" else "routine_") + str(i)
            entry = selected.get(
                token,
                dict(
                    type_id=token,
                    available_calls=0,
                    available_animals=0,
                    panels=[None] * 3,
                    second_different_animal=False,
                    foil_distance=None,
                ),
            )
            entries.append(
                dict(
                    **entry,
                    panel_id=panel,
                    cohort=anchor["cohort"],
                    method=recipe["method"],
                    recipe=recipe,
                    model_key=anchor["saved"]["cache"]["model_key"],
                    source_fit=anchor["saved"],
                    stem=panel + "_" + token,
                )
            )
    return dict(
        entries=entries,
        bindings=bindings,
        source_manifest=str(manifest_path),
        source_manifest_sha256=digest(manifest_path),
        input_manifest_sha256=digest(manifest["manifest"]),
        artifacts=snapshots,
        render_fps=30,
        pre_post_seconds=0.5,
        output_policy="full chain plus0.5s context on both sides; align on chain onset; black after panel finishes; no speed normalization or chain truncation",
        screening_passes=3,
        selection="median-duration executions; second animal preferred; same-recording other-type source/duration foil; no group/DA scores",
        limits=dict(ram_gib=8, threads=2, opencv_threads=1, disk_floor_gib=80, output_cap_gib=4),
    )


def render_one(entry, plan, output, clocks):
    import cv2
    import numpy as np

    panels, audits, contact = [], [], []
    writer = None
    path = output / (entry["stem"] + ".mp4")
    temporary = output / (entry["stem"] + ".partial.mp4")
    try:
        for number, call in enumerate(entry["panels"]):
            if call is None:
                panels.append(None)
                audits.append(dict(panel=number + 1, status="INSUFFICIENT_SUPPORT"))
                continue
            binding = plan["bindings"][call["recording"]]
            video = Path(binding["video_path"])
            if (video.stat().st_size, video.stat().st_mtime_ns) != (binding["video_size"], binding["video_mtime_ns"]):
                raise ValueError("source video stat changed")
            if binding["timestamp_path"] and digest(binding["timestamp_path"]) != binding["timestamp_sha256"]:
                raise ValueError("camera clock changed")
            indexes, times = sample_indexes(clocks.get(binding), call["start_frame"], call["stop_frame"])
            capture = cv2.VideoCapture(str(video))
            panels.append(
                dict(capture=capture, indexes=indexes, times=times, next_index=int(indexes[0]), image=None, call=call)
            )
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, int(indexes[0])):
                raise ValueError("native seek failed")
            audits.append(
                dict(
                    panel=number + 1,
                    status="RENDERED",
                    video_path=str(video),
                    native_first=int(indexes[0]),
                    native_last=int(indexes[-1]),
                    sampled_frames=len(indexes),
                    start_frame=call["start_frame"],
                    stop_frame=call["stop_frame"],
                    duration_s=call["duration_s"],
                    recording=call["recording"],
                    subject=call["subject"],
                    type_id=call["type_id"],
                )
            )
        length = max((len(p["indexes"]) for p in panels if p is not None), default=1)
        checkpoints = set(np.linspace(0, length - 1, 8).astype(int).tolist())
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1080, 440))
        if not writer.isOpened():
            raise ValueError("video writer unavailable")
        for position in range(length):
            tiles = []
            for number, panel in enumerate(panels):
                tile = np.zeros((440, 360, 3), np.uint8)
                if panel is None or position >= len(panel["indexes"]):
                    label = "NO SUPPORTED EXAMPLE" if panel is None else "PANEL FINISHED"
                    cv2.putText(tile, label, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    index, relative = panel["indexes"][position], panel["times"][position]
                    while panel["next_index"] <= index:
                        ok, panel["image"] = panel["capture"].read()
                        if (
                            not ok
                            or int(round(panel["capture"].get(cv2.CAP_PROP_POS_FRAMES))) != panel["next_index"] + 1
                        ):
                            raise ValueError("native decoding/seek position differs")
                        panel["next_index"] += 1
                    tile[80:] = cv2.resize(panel["image"], (360, 360))
                    call = panel["call"]
                    active = call["start_frame"] <= index < call["stop_frame"]
                    role = ("execution A", "execution B", "other-type comparison")[number]
                    text = (
                        role + " | " + call["type_id"],
                        "mouse " + call["subject"] + " | " + call["recording"],
                        f"native {index} | {relative:+.2f}s | " + ("CHAIN" if active else "context"),
                    )
                    for line, label in enumerate(text):
                        cv2.putText(
                            tile,
                            label,
                            (5, 20 + 23 * line),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (80, 225, 140) if active else (225, 225, 225),
                            1,
                        )
                tiles.append(tile)
            frame = np.concatenate(tiles, axis=1)
            writer.write(frame)
            if position in checkpoints:
                contact.append(frame)
    finally:
        if writer is not None:
            writer.release()
        for panel in panels:
            if panel is not None:
                panel["capture"].release()
    sheet = output / (entry["stem"] + ".jpg")
    if not cv2.imwrite(str(sheet), np.concatenate(contact, axis=0)):
        raise ValueError("contact sheet encoding failed")
    capture = cv2.VideoCapture(str(temporary))
    try:
        decoded = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
        if decoded != length or abs(capture.get(cv2.CAP_PROP_FPS) - 30) > 0.001:
            raise ValueError("complete output decoding failed")
    finally:
        capture.release()
    temporary.replace(path)
    return dict(
        stem=entry["stem"],
        panels=audits,
        frames=length,
        duration_s=length / 30,
        video_sha256=digest(path),
        contact_sheet_sha256=digest(sheet),
        decoded_frames=decoded,
    )


def refresh_contact_sheet(output, result):
    """Sample each panel on its own elapsed timeline so short clips remain visible.

    Only the still sheet uses independently spaced rows; movie timing is unchanged.
    Every tile retains its original native index and elapsed-time caption.
    """
    import cv2
    import numpy as np

    columns = []
    capture = cv2.VideoCapture(str(output / (result["stem"] + ".mp4")))
    try:
        for panel in result["panels"]:
            samples = []
            if panel["status"] == "RENDERED":
                for index in np.linspace(0, panel["sampled_frames"] - 1, 8).astype(int):
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                        raise ValueError("contact sheet video seek failed")
                    ok, frame = capture.read()
                    if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                        raise ValueError("contact sheet decoded index differs")
                    left = (panel["panel"] - 1) * 360
                    samples.append(frame[:, left : left + 360].copy())
            else:
                samples = [np.zeros((440, 360, 3), np.uint8)] * 8
            columns.append(np.concatenate(samples, axis=0))
    finally:
        capture.release()
    path = output / (result["stem"] + ".jpg")
    if not cv2.imwrite(str(path), np.concatenate(columns, axis=1)):
        raise ValueError("contact sheet encoding failed")
    result["contact_sheet_sha256"] = digest(path)
    result["contact_sheet_policy"] = "Eight independently spaced times per panel; captions preserve native/time coordinates; movie has natural synchronous timing"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preview", action="store_true", help="One supported reference type per cohort")
    args = parser.parse_args(argv)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    import cv2

    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    manifest = output / "manifest.json"
    plan = json.loads(manifest.read_text()) if manifest.exists() else build_plan(root, args.source.resolve())
    if (
        Path(plan["source_manifest"]).parent != args.source.resolve()
        or digest(plan["source_manifest"]) != plan["source_manifest_sha256"]
    ):
        raise ValueError("source manifest changed")
    print(
        json.dumps(
            dict(
                types=len(plan["entries"]),
                supported_panels=sum(c is not None for e in plan["entries"] for c in e["panels"]),
                execute=args.execute,
            )
        ),
        flush=True,
    )
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "render_source.py").write_text(Path(__file__).read_text())
    clocks = ClockCache()
    selected = [e for e in plan["entries"] if any(e["panels"])]
    if args.preview:
        selected = [next(e for e in selected if e["cohort"] == c and any(e["panels"])) for c in ("karen", "jonathan")]
    results = []
    for entry in selected:
        if (
            shutil.disk_usage(output).free < 80 * 1024**3
            or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 4 * 1024**3
        ):
            raise ValueError("disk floor/output quota reached")
        check = output / (entry["stem"] + ".json")
        if check.exists():
            result = json.loads(check.read_text())
            if (
                digest(output / (entry["stem"] + ".mp4")) != result["video_sha256"]
                or digest(output / (entry["stem"] + ".jpg")) != result["contact_sheet_sha256"]
            ):
                raise ValueError("completed media fingerprint changed")
        else:
            result = render_one(entry, plan, output, clocks)
        result = refresh_contact_sheet(output, result)
        check.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(entry["stem"], result["duration_s"], flush=True)
    summary = dict(
        status="PREVIEW_COMPLETE" if args.preview else "COMPLETE",
        files=len(results),
        reference_capacity=len(plan["entries"]),
        unsupported_reference_types=[e["stem"] for e in plan["entries"] if not any(e["panels"])],
        playback_once_s=sum(r["duration_s"] for r in results),
        playback_three_passes_s=3 * sum(r["duration_s"] for r in results),
        decoded_frames=sum(r["decoded_frames"] for r in results),
        model_refits=0,
        review_status="UNREVIEWED",
        resource_limits=plan["limits"],
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cards = []
    for entry in selected:
        stem = html.escape(entry["stem"])
        cards.append(
            f'<article><h2>{stem}</h2><p>{entry["cohort"]}; {entry["method"]}; reference fold0/seed0</p><video controls preload="none" src="{stem}.mp4"></video><p><a href="{stem}.jpg">Eight-time-point contact sheet</a></p></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Generator screening</title><style>body{font-family:sans-serif;max-width:1150px;margin:auto}video{width:100%}article{margin:2em 0}</style><h1>Five settings per cohort: initial screening</h1><p>Three playback passes initially. Left/right are executions of the reference type; third panel is a matched different detected type, not ground truth. Descriptive source/duration match, no phenotype/DA selection. Natural movement timing; black padding after completed panels. Still-sheet rows sample each panel independently; elapsed-time captions identify the actual times. IDs remain cohort/model-local.</p>'
        + "\n".join(cards)
    )
    notes = output / "review_notes.csv"
    if not notes.exists():
        with notes.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "reference_type",
                    "cohort",
                    "method",
                    "within_type_similarity",
                    "action_order",
                    "split_merge_concern",
                    "boundary_concern",
                    "foil_distinction",
                    "decision",
                    "notes",
                )
            )
            writer.writerows(
                (e["stem"], e["cohort"], e["method"], "", "", "", "", "", "pending", "") for e in plan["entries"]
            )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
