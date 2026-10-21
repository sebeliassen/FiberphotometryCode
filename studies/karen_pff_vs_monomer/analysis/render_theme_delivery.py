"""Render explicit group-blind Karen theme cores with the frozen legacy ranker.

No fitting or inferential comparisons. Explicit empirical execution uses native
60Hz cleaned tracks and half-open occurrence bounds. Movies never warp timing.
"""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


def write_gallery(output, manifest, results):
    from .run_reference_seqnmf import digest

    ready = defaultdict(dict)
    for result in results:
        ready[result["family"]][result["selection_mode"]] = result
    families = [{**family, "movies": ready[family["stem"]]} for family in manifest["families"]
                if family["stem"] in ready]
    data = dict(package=manifest["labels_key"], source_manifest_sha256=digest(output / "manifest.json"),
                families=families, ready_movies=len(results), expected_movies=3 * len(manifest["families"]))
    page = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Karen exploratory themes</title>
<style>body{font:18px system-ui;background:#15181e;color:#eee;max-width:1400px;margin:30px auto;padding:16px}p{line-height:1.5}select,button,textarea,input{font:inherit;padding:9px;margin:4px;max-width:95%}a{color:#abd0ff}video{width:100%;max-height:85vh;background:#000}textarea{width:95%;min-height:85px}.muted{color:#c0c4cd}.controls{position:sticky;top:0;background:#15181ef5;padding:8px;z-index:1}fieldset{margin-top:20px;border:1px solid #777}#error{color:#ffb0a0}</style></head><body>
<h1>Karen: exploratory behavioural themes</h1>
<p>Five explicit observed cores suggested by the existing seqNMF catalogues. These are refined theme populations, not five newly trained components. Names use supplied human syllable labels and remain provisional. No group-effect selection.</p>
<p class="muted">Six executions per movie, 2×3. p100: closest exemplar neighbours; P80: near the 80th percentile of exemplar proximity; random: uniform over eligible occurrences. Percentile is similarity, not correctness. The camera stays fixed at each execution's onset position/heading. Each panel plays at its natural pace and goes dark when finished.</p>
<div class="controls"><label>Theme <select id="theme"></select></label><label>Examples <select id="mode"><option value="p80">P80 · start here</option><option value="p100">p100</option><option value="random">Random</option></select></label><button id="replay">Replay</button><button id="full">Fullscreen</button></div>
<h2 id="title"></h2><p id="definition"></p><p id="details"></p><video id="player" controls preload="metadata"></video><p><a id="direct">Open movie file</a> · <a href="manifest.json">Provenance and selected occurrences</a> · <a href="status.json">Render status</a></p>
<p id="caution" class="muted"></p><fieldset><legend>Optional quick feedback for this theme and example mode</legend>
<label>Coherence <select id="coherence"><option value="">Unrated</option><option>Similar</option><option>Mixed</option><option>Dissimilar</option><option>Unsure</option></select></label>
<label>Useful theme? <select id="useful"><option value="">Unrated</option><option>Yes</option><option>Maybe</option><option>No</option></select></label>
<p><textarea id="note" placeholder="What behaviour do you see? Mixed panels, meaningful variants, overlap with another theme, or boundary problems?"></textarea></p>
<p><button id="export">Export feedback JSON</button><label>Import <input type="file" id="import" accept="application/json,.json"></label></p><p id="saved" class="muted"></p><p id="error"></p></fieldset>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id), controls=['coherence','useful','note'];let store={};
try{const prior=JSON.parse(localStorage.getItem(DATA.package)||'{}');if(prior.source_manifest_sha256===DATA.source_manifest_sha256)store=prior.labels||{}}catch(e){}
DATA.families.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.theme_name;$('theme').append(o)});
function current(){const f=DATA.families[+$('theme').value];return {f,m:f?.movies[$('mode').value],key:f?.stem+'::'+$('mode').value}}
function update(){const {f,m,key}=current();if(!f)return;$('title').textContent=f.theme_name;$('definition').textContent='Observed grouped core: '+f.stage_labels.map(x=>x.join(' / ')).join(' → ');$('caution').textContent=f.caution;
if(!m){$('details').textContent='This mode is still rendering.';$('player').removeAttribute('src');controls.forEach(id=>$(id).disabled=true);return}
controls.forEach(id=>$(id).disabled=false);$('player').pause();$('player').src=m.filename;$('player').poster=m.filename.replace('.mp4','.png');$('direct').href=m.filename;
$('details').textContent=`${DATA.ready_movies}/${DATA.expected_movies} movies ready · ${f.saved_calls} full-cohort core windows · ${f.eligible_calls} movie-eligible / ${f.eligible_mice} mice · ${m.playback_s.toFixed(2)} seconds/play`;
const label=store[key]||{};controls.forEach(id=>$(id).value=label[id]||'');$('saved').textContent=label.updated_at?'Saved '+label.updated_at:'Feedback autosaves in this browser. Export to share it.'}
function persist(){localStorage.setItem(DATA.package,JSON.stringify({package:DATA.package,source_manifest_sha256:DATA.source_manifest_sha256,labels:store}))}
function save(){const {f,m,key}=current();if(!m)return;store[key]={theme:f.stem,mode:$('mode').value,movie_sha256:m.sha256,updated_at:new Date().toISOString()};controls.forEach(id=>store[key][id]=$(id).value);persist();$('saved').textContent='Saved in this browser. Export JSON to share.'}
$('theme').onchange=update;$('mode').onchange=update;controls.forEach(id=>$(id).oninput=save);$('replay').onclick=()=>{$('player').currentTime=0;$('player').play().catch(()=>{})};$('full').onclick=()=>$('player').requestFullscreen?.();
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({package:DATA.package,source_manifest_sha256:DATA.source_manifest_sha256,exported_at:new Date().toISOString(),labels:store},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='karen_theme_feedback_20261021.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('import').onchange=async e=>{try{const incoming=JSON.parse(await e.target.files[0].text());if(incoming.package!==DATA.package||incoming.source_manifest_sha256!==DATA.source_manifest_sha256||!incoming.labels||typeof incoming.labels!=='object'||Array.isArray(incoming.labels))throw Error('This export belongs to a different review package.');
for(const [key,label]of Object.entries(incoming.labels)){const [stem,mode]=key.split('::'),f=DATA.families.find(f=>f.stem===stem),m=f?.movies[mode];if(!m||label.theme!==stem||label.mode!==mode||label.movie_sha256!==m.sha256)throw Error('Movie identity mismatch. Nothing imported.');if(!controls.every(id=>label[id]===undefined||typeof label[id]==='string'))throw Error('Invalid feedback values. Nothing imported.')}
store={...store,...incoming.labels};persist();$('error').textContent='';update()}catch(err){$('error').textContent=err.message}};
if(DATA.families.length)update();
</script></body></html>'''
    temporary = output / "index.html.tmp"
    temporary.write_text(page.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")))
    temporary.replace(output / "index.html")


def prepare(args):
    import copy

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

    source_path, recipe_path = args.source / "manifest.json", args.fits / "recipe.json"
    source, recipe = json.loads(source_path.read_text()), json.loads(recipe_path.read_text())
    library_path = args.delivery / "library.json"
    occurrences_path = args.occurrences or args.delivery / "occurrences.csv.gz"
    library = json.loads(library_path.read_text())
    themes = library["themes"]
    if len(themes) != 5:
        raise ValueError("this bounded media delivery expects five explicit themes")
    scaling_path = args.source.parent / "selected_ranking_recipe.json"
    scaling = json.loads(scaling_path.read_text())
    mapping_path = args.fits / "prepared/L3_unresolved_separate/mapping.json"
    mapping = np.asarray(json.loads(mapping_path.read_text()), int)
    if not np.array_equal(mapping, np.asarray(library["mapping"]["raw_to_grouped"], int)):
        raise ValueError("frozen theme and cleaned fit token namespaces differ")
    annotation = source["annotation_source"]
    if digest(annotation["path"]) != annotation["sha256"]:
        raise ValueError("source human annotation fingerprint changed")
    rows = copy.deepcopy(source["track_rows"])
    plan = dict(families=[], bindings=source["bindings"], track_rows=rows,
                annotation_mappings=source["annotation_mappings"], annotation_source=source["annotation_source"],
                modes=["p100", "p80", "random"], panels=6, rows=2, columns=3, fps=60,
                context_sec_each_side=.5, labels_key="karen-theme-delivery-20261021-v1",
                selection="new refined-core population, existing449Dblockweights/native scales; animal-balanced medoid, exemplarneighbours; ranked diversity only in declared pools; uniformrandom withoutreplacement",
                fit_design="five exploratory observed-core definitions from existing pooledL3/L5K10/K18 fits; not new fits or wholefactor acceptance",
                library=dict(path=str(library_path.resolve()), sha256=digest(library_path)),
                occurrences=dict(path=str(occurrences_path.resolve()), sha256=digest(occurrences_path)),
                source_manifest=dict(path=str(source_path.resolve()), sha256=digest(source_path)),
                cleaned_recipe=dict(path=str(recipe_path.resolve()), sha256=digest(recipe_path)),
                frozen_ranking=dict(path=str(scaling_path.resolve()), sha256=digest(scaling_path), values=scaling),
                mapping=dict(path=str(mapping_path.resolve()), sha256=digest(mapping_path)))
    # Replace every stale pre-repair row with the cleaned input used by the fits.
    for number, item in enumerate(recipe["source_arrays"], 1):
        guard(args)
        sid, path = item["session_id"], Path(item["path"])
        with np.load(path, allow_pickle=False) as cache:
            frames, labels = cache["frames"], cache["labels"]
            excluded = cache["analysis_excluded"] if "analysis_excluded" in cache.files else np.zeros(len(frames), bool)
            identity = hashlib.sha256()
            for array in (frames, labels, excluded):
                identity.update(str((array.shape, str(array.dtype))).encode())
                identity.update(np.ascontiguousarray(array).tobytes())
            if identity.hexdigest() != item["relevant_frame_label_mask_sha256"] or float(cache["fps"]) != 60.:
                raise ValueError("cleaned fit source frame/label/mask identity changed")
        rows[sid] = dict(exclude_from_search=False, native_mapping_verified=True,
                         source_provenance=dict(source=dict(path=str(path.resolve()), sha256=digest(path))))
        if number % 20 == 0:
            log(f"THEME TRACK IDENTITY {number}/99")
    if len(rows) != 99:
        raise ValueError("unexpected cleaned tracking scope")
    # Group and wave columns are deliberately not loaded for media selection.
    columns = ["session_id", "animal_id", "theme", "occurrence_id", "start_frame", "end_frame_exclusive", "observed_tokens", "duration_s"]
    occurrences = pd.read_csv(occurrences_path, usecols=columns, dtype={"animal_id": str})
    if set(occurrences.theme) != {theme["id"] for theme in themes}:
        raise ValueError("occurrence theme identity differs from frozen library")
    if occurrences.occurrence_id.duplicated().any():
        raise ValueError("duplicate occurrence identities")
    tracks, clocks = Tracks(rows), ClockCache()
    vectors, records, unavailable = [], [], []
    for number, (sid, table) in enumerate(occurrences.groupby("session_id", sort=False), 1):
        guard(args)
        binding = plan["bindings"][sid]
        video = Path(binding["video_path"])
        if (video.stat().st_size, video.stat().st_mtime_ns) != (binding["video_size"], binding["video_mtime_ns"]):
            raise ValueError("native video fingerprint changed")
        track, clock = tracks.get(sid), clocks.get(binding)
        calls = []
        for row in table.to_dict("records"):
            row.update(recording=sid, subject=str(row["animal_id"]), stop_frame=int(row["end_frame_exclusive"]))
            if row["stop_frame"] >= len(clock):
                unavailable.append(dict(occurrence_id=row["occurrence_id"], theme=row["theme"], session_id=sid,
                                        excludeFromMovieReason="encoded video endpoint unavailable; statistical occurrence retained"))
                continue
            verify_call(track, row, mapping)
            a, b = np.searchsorted(track["frames"], [row["start_frame"], row["stop_frame"]])
            if not np.isfinite(track["latent"][a:b]).all():
                unavailable.append(dict(occurrence_id=row["occurrence_id"], theme=row["theme"], session_id=sid,
                                        excludeFromMovieReason="nonfinite native latent support; statistical occurrence retained"))
                continue
            calls.append(row)
        if calls:
            values, okay, _ = execution_descriptors(track, clock, calls, dict(latent=np.ones(10), path=1.))
            if values.shape[1] != 449 or not okay.all():
                raise ValueError("descriptor width or exact native support unavailable")
            values[:, :320] /= np.tile(scaling["latent_iqr_native_units"], 32)
            values[:, 320:384] /= scaling["displacement_px_p75"]
            values[:, -1] /= scaling["log_duration_iqr"]
            vectors.append(values)
            records.extend(calls)
        log(f"THEME RANK INPUT {number}/99 {sid}: {len(calls)} eligible")
    eligible = pd.DataFrame(records)
    values = np.concatenate(vectors)
    rank_rows = []
    for theme in themes:
        sub = eligible[eligible.theme == theme["id"]]
        if len(sub) < 6:
            raise ValueError(f"fewer than six unique executions for {theme['id']}")
        distance, percentile, exemplar = rank_exemplar_neighbors(values[sub.index], sub.subject.to_numpy())
        seed = int(hashlib.sha256(("20261021:" + theme["id"]).encode()).hexdigest()[:8], 16)
        examples = select_movie_sets(sub.to_dict("records"), distance, percentile, seed, exemplar, panels=6)
        plan["families"].append(dict(stem=theme["id"], theme_name=theme["name"], cohort="karen",
                                     stage_labels=theme["stage_labels"], caution=theme["caution"], stage_sets=theme["stage_sets"],
                                     eligible_calls=len(sub), eligible_mice=int(sub.subject.nunique()),
                                     saved_calls=int((occurrences.theme == theme["id"]).sum()),
                                     examples_by_mode=examples, crop_native_pixels=320,
                                     descriptor_sha256=hashlib.sha256(values[sub.index].tobytes()).hexdigest()))
        for row, d, pct in zip(sub.to_dict("records"), distance, percentile):
            rank_rows.append(dict(occurrence_id=row["occurrence_id"], theme=theme["id"], distance_squared=float(d), percentile=float(pct)))
    pd.DataFrame(rank_rows).to_csv(args.output / "execution_ranking.csv.gz", index=False)
    eligible.to_csv(args.output / "eligible_calls.csv.gz", index=False)
    pd.DataFrame(unavailable, columns=["occurrence_id", "theme", "session_id", "excludeFromMovieReason"]).to_csv(args.output / "unavailable_calls.csv", index=False)
    selected = defaultdict(list)
    for family in plan["families"]:
        for examples in family["examples_by_mode"].values():
            for example in examples:
                example["filename"] = f"components/{family['stem']}_{example['role']}.mp4"
                selected[example["recording"]].append((family, example))
    extents = defaultdict(lambda: 160.)
    for sid, examples in selected.items():
        guard(args)
        track, clock = tracks.get(sid), clocks.get(plan["bindings"][sid])
        for family, example in examples:
            a = int(np.searchsorted(track["frames"], example["start_frame"]))
            example["onset_centroid"], example["onset_heading"] = track["centroid"][a].tolist(), float(track["heading"][a])
            example["segments"] = syllable_segments(track, clock, example["start_frame"], example["stop_frame"])
            indexes, _ = sample_indexes(clock, example["start_frame"], example["stop_frame"], fps=60)
            pos = np.minimum(np.searchsorted(track["frames"], indexes), len(track["frames"]) - 1)
            valid = (track["frames"][pos] == indexes) & np.isfinite(track["centroid"][pos]).all(axis=1)
            xy = track["centroid"][pos[valid]] - example["onset_centroid"]
            h = example["onset_heading"]
            rotation = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])
            extents[family["stem"]] = max(extents[family["stem"]], float(np.max(np.abs(xy @ rotation.T))) + 55)
            example["frames"] = len(indexes)
    for family in plan["families"]:
        family["crop_native_pixels"] = 16 * math.ceil(2 * extents[family["stem"]] / 16)
        if family["crop_native_pixels"] > 2048:
            raise ValueError("selected movement exceeds bounded onset-fixed crop")
    from studies.sequence_viz.recreate_generator_videos import render_example
    from studies.sequence_viz.render_generator_review_grids import render_grid
    plan["source_hashes"] = {str(path.resolve()): digest(path) for path in
                             (Path(__file__), Path(render_example.__code__.co_filename), Path(render_grid.__code__.co_filename),
                              Path(guard.__code__.co_filename), Path(verify_call.__code__.co_filename),
                              Path(sample_indexes.__code__.co_filename))}
    write_json(args.output / "manifest.json", plan)
    log(f"THEME RANK COMPLETE: {len(eligible)} eligible / {len(unavailable)} unavailable; five themes")
    return plan


def render(args, plan):
    from .render_six_panel_review import guard, init_worker, render_family
    from .run_reference_seqnmf import digest, log, write_json

    for path, expected in plan["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("prepared renderer source changed")
    for item in (plan["library"], plan["occurrences"], plan["cleaned_recipe"], plan["frozen_ranking"]):
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("prepared input identity changed")
    results = []
    batches = (plan["families"][:1],) if args.preview else (plan["families"][:1], plan["families"][1:])
    for batch in batches:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(batch)), initializer=init_worker,
                                 initargs=(plan, str(args.output))) as pool:
            pending = {pool.submit(render_family, family) for family in batch}
            while pending:
                guard(args)
                completed, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in completed:
                    results.extend(future.result())
                    write_gallery(args.output, plan, results)
                    log(f"THEME MOVIES {len(results)}/15")
        if len(results) == 3:
            write_json(args.output / "preview.json", dict(status="PASS", first_theme=plan["families"][0]["stem"],
                                                          movies=3, full_native_decode_and_encoded_frame_counts=True))
    status = dict(state="PREVIEW_COMPLETE" if args.preview else "COMPLETE", movies=len(results), execution_positions=6 * len(results),
                  playback_by_mode={mode: sum(r["playback_s"] for r in results if r["selection_mode"] == mode) for mode in plan["modes"]},
                  dot_offscreen_frames=sum(r.get("dot_offscreen_frames", 0) for r in results),
                  missing_tracking_frames=sum(r.get("missing_tracking_frames", 0) for r in results),
                  workers=args.workers, numerical_threads=2, encoder_threads=2,
                  human_review="pending; selected media do not establish whole-population coherence")
    write_json(args.output / "status.json", status)
    log(json.dumps(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fits", "delivery", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, help="immutable core index; defaults to delivery/occurrences.csv.gz")
    parser.add_argument("--stage", choices=("prepare", "render", "all"), default="all")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=3.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, execute=False, themes=5, movies=15, panels=6, rows=2, columns=3, output=str(args.output))))
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "components").mkdir(exist_ok=True)
    manifest = args.output / "manifest.json"
    if args.stage in ("prepare", "all"):
        plan = json.loads(manifest.read_text()) if manifest.exists() else prepare(args)
    else:
        plan = json.loads(manifest.read_text())
    if args.stage in ("render", "all"):
        render(args, plan)


if __name__ == "__main__":
    main()
