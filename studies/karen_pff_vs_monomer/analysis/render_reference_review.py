"""Explicit P80-first review of saved pooled Karen catalogues; never fit a model."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from studies.karen_pff_vs_monomer.analysis.run_reference_seqnmf import guard, log, write_json
from studies.sequence_viz.recreate_generator_videos import (
    ClockCache,
    Tracks,
    execution_descriptors,
    rank_exemplar_neighbors,
    render_example,
    sample_indexes,
    syllable_segments,
)
from studies.sequence_viz.render_generator_review_grids import render_grid, select_movie_sets
from studies.sequence_viz.render_generator_screening import digest

TRIALS = (
    "L3_unresolved_separate_K10_lambda0.3_seed43",
    "L3_unresolved_separate_K18_lambda0.3_seed43",
    "L3_unresolved_separate_K22_lambda0.3_seed42",
)


def verify_call(track, call, mapping):
    """Check the entire native core, including gaps and actual grouped token order."""
    import numpy as np

    start, stop = int(call["start_frame"]), int(call["end_frame_exclusive"])
    a, b = np.searchsorted(track["frames"], [start, stop])
    if stop <= start or not np.array_equal(track["frames"][a:b], np.arange(start, stop)):
        raise ValueError("call crosses missing or excluded native support")
    labels = mapping[track["labels"][a:b].astype(int)]
    observed = labels[np.r_[True, np.diff(labels) != 0]].tolist()
    if observed != json.loads(call["observed_tokens"]):
        raise ValueError("saved grouped call differs from raw native labels")
    if not np.isfinite(track["centroid"][a:b]).all() or not np.isfinite(track["heading"][a:b]).all():
        raise ValueError("call core lacks finite position/heading")


def balance_sample(values, animals, per_animal=512):
    """Deterministic equal-mouse calibration; no condition/genotype input."""
    import numpy as np

    rng = np.random.default_rng(101)
    animals = np.asarray(animals)
    selected = []
    for animal in np.unique(animals):
        own = np.flatnonzero(animals == animal)
        selected.extend(rng.choice(own, per_animal, replace=len(own) < per_animal))
    return values[selected]


def normalize_descriptors(values, calibration):
    """Common robust feature scales, retaining the established four-block weights."""
    import numpy as np

    # execution_descriptors flattens 32x10 latent, 32x2 displacement, 32x2 heading, log duration.
    if values.shape[1] != 449:
        raise ValueError("expected 10 original latent channels and 32 phase samples")
    latent = calibration[:, :320].reshape(-1, 32, 10)
    latent_iqr = np.diff(np.quantile(latent, [.25, .75], axis=(0, 1)), axis=0)[0]
    latent_iqr = np.maximum(latent_iqr, 1e-6)
    # Undo existing equal-block normalisation only to express native-pixel path scale.
    moved = calibration[:, 320:384].reshape(-1, 32, 2) * math.sqrt(4 * 64)
    path = max(float(np.quantile(np.linalg.norm(moved, axis=-1), .75)), 1.)
    duration_iqr = max(float(np.diff(np.quantile(calibration[:, -1] * 2, [.25, .75]))[0]), .1)
    result = values.copy()
    result[:, :320] /= np.tile(latent_iqr, 32)
    result[:, 320:384] /= path
    result[:, -1] /= duration_iqr
    return result, dict(latent_iqr=latent_iqr.tolist(), displacement_px_p75=path,
                        log_duration_iqr=duration_iqr)


def prepare(args):
    import cv2
    import numpy as np
    import pandas as pd

    recipe = json.loads((args.base / "recipe.json").read_text())
    metrics = pd.read_csv(args.capacity / "capacity_metrics.csv")
    families = pd.read_csv(args.capacity / "family_metrics.csv")
    sessions_path = Path(recipe["input_metadata"][0]["path"])
    if digest(sessions_path) != recipe["input_metadata"][0]["sha256"]:
        raise ValueError("fitted session metadata changed")
    sessions = pd.read_csv(sessions_path, dtype={"animal_id": str}).set_index("session_id")
    mapping_path = Path(recipe["input_metadata"][3]["path"])
    if digest(mapping_path) != recipe["input_metadata"][3]["sha256"]:
        raise ValueError("fitted vocabulary mapping changed")
    mapping = np.asarray(json.loads(mapping_path.read_text()), dtype=int)
    annotations = sessions_path.parent / "syllable_annotations.csv"
    names = {str(r.syllable_id): r.description for r in pd.read_csv(annotations).itertuples()}
    rows, bindings, fingerprints = {}, {}, {}
    tables, groups = {}, defaultdict(list)
    for trial in TRIALS:
        row = metrics[metrics.trial == trial].iloc[0]
        folder = Path(row.directory)
        if digest(folder / "model.npz") != row.model_sha256 or digest(folder / "pooled_calls.csv.gz") != row.calls_sha256:
            raise ValueError("frozen fit/calls changed")
        supported = families[(families.trial == trial) & (families.calls >= 20) & (families.mice >= 5)]
        table = pd.read_csv(folder / "pooled_calls.csv.gz", dtype={"animal_id": str})
        tables[trial] = table[table.factor.isin(supported.factor)].copy()
        fingerprints[trial] = dict(model_sha256=row.model_sha256, calls_sha256=row.calls_sha256,
                                   directory=str(folder))
        for factor in supported.factor:
            groups[(trial, int(factor))] = []
    unavailable = defaultdict(int)
    unavailable_reasons = defaultdict(int)
    for number, source in enumerate(recipe["source_arrays"], 1):
        guard(args)
        sid, path = source["session_id"], Path(source["path"])
        with np.load(path, allow_pickle=False) as saved:
            frames, labels = saved["frames"], saved["labels"]
            excluded = saved["analysis_excluded"] if "analysis_excluded" in saved.files else np.zeros(len(frames), bool)
            identity = hashlib.sha256()
            for array in (frames, labels, excluded):
                identity.update(str((array.shape, str(array.dtype))).encode())
                identity.update(np.ascontiguousarray(array).tobytes())
            if identity.hexdigest() != source["relevant_frame_label_mask_sha256"] or float(saved["fps"]) != 60.:
                raise ValueError("fitted native labels/masks/clock changed")
            support = (labels >= 0) & ~excluded
            track = {k: saved[k][support] for k in ("frames", "labels", "centroid", "heading", "latent")}
        fingerprint = digest(path)
        rows[sid] = dict(exclude_from_search=False, native_mapping_verified=True,
                         source_provenance={"source": dict(path=str(path), sha256=fingerprint)})
        video = args.video_dir / Path(sessions.loc[sid, "video_path"]).name
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 60.) > .001:
                raise ValueError("missing native60Hz video: " + str(video))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if len(frames) != int(sessions.loc[sid, "n_frames"]):
            raise ValueError("native cache/metadata row-count mismatch")
        bindings[sid] = dict(video_path=str(video), video_frames=count, encoded_fps=60., acquisition_fps=60.,
                             timestamp_path=None, timestamp_sha256=None, video_size=video.stat().st_size,
                             video_mtime_ns=video.stat().st_mtime_ns,
                             cache_rows=len(frames), cache_last_native_frame=int(frames[-1]),
                             frame_mapping="verified native60Hz cropped TopCam AVI index, identity mapping")
        clock = np.arange(count) / 60.
        for trial, table in tables.items():
            sub = table[table.session_id == sid]
            calls = []
            for original in sub.to_dict("records"):
                if int(original["end_frame_exclusive"]) >= count:
                    unavailable[(trial, int(original["factor"]))] += 1
                    unavailable_reasons["native_video_endpoint_unavailable"] += 1
                    continue
                try:
                    verify_call(track, original, mapping)
                except ValueError as error:
                    if "finite position/heading" not in str(error):
                        raise
                    unavailable[(trial, int(original["factor"]))] += 1
                    unavailable_reasons["nonfinite_core_centroid_heading"] += 1
                    continue
                calls.append(dict(original, recording=sid, subject=str(original["animal_id"]),
                                  stop_frame=int(original["end_frame_exclusive"])))
            if not calls:
                continue
            for a in range(0, len(calls), 256):
                batch = calls[a:a + 256]
                vectors, valid, _ = execution_descriptors(track, clock, batch, dict(latent=np.ones(10), path=1.))
                for call, vector, okay in zip(batch, vectors, valid):
                    key = (trial, int(call["factor"]))
                    if okay:
                        groups[key].append((call, vector))
                    else:
                        unavailable[key] += 1
                        unavailable_reasons["descriptor_support_unavailable"] += 1
        log(f"PROFILE {number}/99 {sid}")
    all_values = np.stack([v for pairs in groups.values() for _, v in pairs])
    animals = [c["subject"] for pairs in groups.values() for c, _ in pairs]
    calibration = balance_sample(all_values, animals)
    _, scales = normalize_descriptors(all_values[:1], calibration)
    tracks, clocks = Tracks(rows), ClockCache()
    manifest = dict(families=[], bindings=bindings, track_rows=rows, annotation_mappings={"karen": names},
                    inputs=fingerprints, annotation_source=dict(path=str(annotations), sha256=digest(annotations)),
                    ranking=dict(policy="animal-balanced actual medoid; distance to exemplar; within-family midrank",
                                 samples=32, dtype="float64", scales=scales,
                                 blocks="equal: original latent / onset-relative displacement / relative heading / log duration",
                                 calibration="pooled equal-mouse512calls; deterministic seed101; no phenotype selection",
                                 selection="P80 +/-5percentile band, nearest first, prefer different mice within band",
                                 limitation="descriptive rank, not behavioural validity; phase sampling for ranking only"),
                    source_recipe_sha256=digest(args.base / "recipe.json"), generated_modes=["p80"],
                    unavailable_reasons=dict(unavailable_reasons),
                    deferred_modes=["p100", "random"], fps=60, context_sec_each_side=.5)
    rank_path = args.output / "execution_ranking.csv.gz"
    with gzip.open(rank_path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "recording", "subject", "start_frame", "stop_frame", "distance", "percentile"])
        for (trial, factor), pairs in groups.items():
            guard(args)
            calls = [c for c, _ in pairs]
            if len(calls) < 3:
                raise ValueError("supported family has fewer than3rankable executions")
            values, _ = normalize_descriptors(np.stack([v for _, v in pairs]), calibration)
            distance, percentiles, exemplar = rank_exemplar_neighbors(values, [c["subject"] for c in calls])
            k = int(trial.split("_K")[1].split("_")[0])
            stem = f"K{k}_F{factor:02d}"
            seed = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)
            examples = select_movie_sets(calls, distance, percentiles, seed, exemplar)["p80"]
            family = dict(stem=stem, cohort="karen", trial=trial, K=k, factor=factor, examples=examples,
                          eligible_calls=len(calls), eligible_mice=len({c["subject"] for c in calls}),
                          unavailable_calls=unavailable[(trial, factor)], crop_native_pixels=320)
            extent = 160.
            for example in examples:
                track = tracks.get(example["recording"])
                clock = clocks.get(bindings[example["recording"]])
                a = int(np.searchsorted(track["frames"], example["start_frame"]))
                example["onset_centroid"] = track["centroid"][a].tolist()
                example["onset_heading"] = float(track["heading"][a])
                example["segments"] = syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
                indexes, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
                loc = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
                known = (track["frames"][loc] == indexes) & np.isfinite(track["centroid"][loc]).all(axis=1)
                xy = track["centroid"][loc[known]] - np.asarray(example["onset_centroid"])
                h = example["onset_heading"]
                rot = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
                extent = max(extent, float(np.max(np.abs(xy @ rot.T))) + 55)
                example["filename"] = stem + "_" + example["role"] + ".mp4"
                example["frames"] = len(indexes)
            family["crop_native_pixels"] = int(16 * math.ceil(2 * extent / 16))
            if family["crop_native_pixels"] > 2048:
                raise ValueError("selected movement exceeds bounded fixed crop")
            manifest["families"].append(family)
            for call, d, p in zip(calls, distance, percentiles):
                writer.writerow([stem, call["recording"], call["subject"], call["start_frame"], call["stop_frame"], d, p])
    if len(manifest["families"]) != 49:
        raise ValueError("expected all49supported families")
    manifest["ranking"]["sha256"] = digest(rank_path)
    manifest["renderer_source_hashes"] = {
        str(p): digest(p) for p in (Path(__file__), Path(render_example.__code__.co_filename),
                                   Path(render_grid.__code__.co_filename))}
    write_json(args.output / "manifest.json", manifest)
    log("PREPARE COMPLETE:49families,147selected executions")
    return manifest


def write_gallery(output, manifest, results):
    """A portable viewer with browser-local notes and an explicit JSON export."""
    by_family = {r["family"]: r for r in results}
    families = []
    for family in manifest["families"]:
        if family["stem"] not in by_family:
            continue
        families.append(dict(**family, movie=by_family[family["stem"]]))
    payload = json.dumps(families).replace("<", "\\u003c")
    names = json.dumps(manifest["annotation_mappings"]["karen"]).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><title>Karen P80 chain review</title>
<style>body{background:#16181d;color:#eee;font:17px system-ui;max-width:1800px;margin:20px auto;padding:0 18px}
button,select,textarea,input{font:inherit;padding:7px;margin:4px}video{width:100%;max-height:76vh;background:#111}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}textarea{width:95%;height:85px}.muted{color:#bfc3cd}</style>
<h1>Karen P80 review</h1><p>Three executions of one family. Start with three plays, focusing on a different panel each time.
Use the similarity and distinctness fields independently. Flag uncertain families for p100/random follow-up.</p>
<p class="muted">P80 is a within-family exemplar-distance rank, not 80% behavioural correctness.
Raw human syllable labels are provisional. Videos retain natural pace and a camera fixed at onset.
Notes save in this browser only: export them before changing browser or URL.</p>
<button id="prev">Previous</button><select id="pick"></select><button id="next">Next</button>
<button id="replay">Replay</button><button id="export">Export labels JSON</button><span id="status"></span>
<h2 id="title"></h2><p id="info"></p><video id="player" controls preload="metadata"></video>
<div id="panels" class="panels"></div><p><a id="direct">Open MP4</a> · <a href="labels.csv">Blank CSV labelling sheet</a></p>
<label>Within-family similarity <select id="coherence"><option value="">Unreviewed</option><option>Similar</option>
<option>Mixed / unclear</option><option>Dissimilar</option></select></label>
<label>Distinct from other families <select id="distinct"><option value="">Unreviewed</option><option>Distinct</option>
<option>Possible duplicate / variant</option><option>Unclear</option></select></label><br>
<label><input id="follow" type="checkbox">Request p100/random</label>
<label>Possible related family IDs <input id="related" placeholder="e.g. K18_F04"></label>
<textarea id="notes" placeholder="Observed theme, variants, confusing labels, tracking/crop problems…"></textarea>
<script>const families=PAYLOAD,names=NAMES,key='karen-pooled-p80-20261015-v1';
const $=id=>document.getElementById(id);let notes={};try{notes=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
families.forEach((f,i)=>{let o=document.createElement('option');o.value=i;o.textContent=f.stem;$('pick').appendChild(o)});
function save(){let f=families[+$('pick').value];notes[f.stem]={family:f.stem,trial:f.trial,coherence:$('coherence').value,
distinct:$('distinct').value,followup:$('follow').checked,related:$('related').value,notes:$('notes').value};
try{localStorage.setItem(key,JSON.stringify(notes));$('status').textContent='Saved locally'}catch(e){$('status').textContent='Storage unavailable: export labels'}}
function show(){let f=families[+$('pick').value],n=notes[f.stem]||{};let p=$('player');p.pause();p.src=f.movie.filename;
p.poster=f.movie.filename.replace('.mp4','.png');$('direct').href=f.movie.filename;$('title').textContent=f.stem+' — P80';
$('info').textContent=f.eligible_calls+' rankable executions / '+f.eligible_mice+' mice; '+f.unavailable_calls+
' unavailable for ranking; '+f.movie.playback_s.toFixed(2)+' seconds/play; '+(+ $('pick').value+1)+' / '+families.length;
$('panels').replaceChildren();f.examples.forEach(e=>{let x=document.createElement('p');x.textContent='Panel '+e.panel_number+
' · Mouse '+e.subject+' · actual P'+e.actual_percentile.toFixed(1)+' · '+e.segments.map(s=>'S'+s.syllable_id+' '+
(names[s.syllable_id]||'?')+' ('+s.duration_s.toFixed(2)+'s)').join(' → ');$('panels').appendChild(x)});
['coherence','distinct','related','notes'].forEach(id=>$(id).value=n[id]||'');$('follow').checked=!!n.followup;$('status').textContent=''}
['coherence','distinct','related','notes','follow'].forEach(id=>$(id).addEventListener('input',save));$('pick').onchange=show;
$('prev').onclick=()=>{$('pick').value=Math.max(0,+$('pick').value-1);show()};
$('next').onclick=()=>{$('pick').value=Math.min(families.length-1,+$('pick').value+1);show()};
$('replay').onclick=()=>{$('player').currentTime=0;$('player').play()};
$('export').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({package:key,labels:notes},null,2)],
{type:'application/json'}));a.download='karen_p80_labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
if(families.length)show();</script>'''
    (output / "index.html").write_text(page.replace("PAYLOAD", payload).replace("NAMES", names))
    with (output / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "trial", "within_family_similarity", "distinctness", "related_families",
                         "request_p100_random", "notes"])
        for family in families:
            writer.writerow([family["stem"], family["trial"], "", "", "", "", ""])


def render(args, manifest):
    import cv2

    cv2.setNumThreads(1)
    tracks, clocks = Tracks(manifest["track_rows"]), ClockCache()
    results = []
    own = manifest["families"][:2] if args.preview else manifest["families"]
    for family in own:
        guard(args)
        path = args.output / (family["stem"] + "_p80_grid.json")
        if path.exists():
            saved = json.loads(path.read_text())
            if digest(args.output / saved["filename"]) != saved["sha256"]:
                raise ValueError("existing grid fingerprint changed")
            results.append(saved)
            log("REUSE " + family["stem"])
            continue
        clips = []
        for example in family["examples"]:
            guard(args)
            clips.append(render_example(family, example, manifest, args.output, tracks, clocks))
        grid = render_grid(family, clips, args.output, selection_mode="p80")
        grid["missing_tracking_frames"] = sum(c["missing_tracking_context_or_call_frames"] for c in clips)
        grid["dot_offscreen_frames"] = sum(c["dot_offscreen_frames"] for c in clips)
        if grid["dot_offscreen_frames"]:
            raise ValueError("selected centroid escaped fixed crop")
        write_json(path, grid)
        results.append(grid)
        write_gallery(args.output, manifest, results)
        log(f"RENDER {len(results)}/{len(own)} {family['stem']} {grid['playback_s']:.2f}s")
    write_gallery(args.output, manifest, results)
    summary = dict(state="PREVIEW_COMPLETE" if args.preview else "COMPLETE", families=len(results),
                   grids=len(results), panels=3 * len(results), modes=["p80"],
                   playback_sec=sum(r["playback_s"] for r in results), fps=60,
                   dot_offscreen_frames=sum(r.get("dot_offscreen_frames", 0) for r in results),
                   missing_tracking_frames=sum(r.get("missing_tracking_frames", 0) for r in results),
                   annotation_status="Human chain review pending; media decode/timing verified",
                   final_rank_sha256=manifest["ranking"]["sha256"])
    write_json(args.output / ("preview_status.json" if args.preview else "status.json"), summary)
    log(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "render"), required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--memory-gb", type=float, default=6.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=6.)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, trials=TRIALS, modes=["p80"], panels=3,
                              output=str(args.output), execute=False)))
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    args.output.mkdir(parents=True, exist_ok=True)
    guard(args)
    if args.stage == "prepare":
        prepare(args)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if digest(args.output / "execution_ranking.csv.gz") != manifest["ranking"]["sha256"]:
            raise ValueError("frozen execution ranking changed")
        for path, fingerprint in manifest["renderer_source_hashes"].items():
            if digest(path) != fingerprint:
                raise ValueError("frozen renderer source changed")
        render(args, manifest)


if __name__ == "__main__":
    main()
