# Karen six-panel chain review — 16 October 2026

Open http://localhost:8766/six_panel_20261016/ on your Mac through the existing
SSH tunnel towoody(10.209.3.161). Or refresh the original gallery and use its
six-panel link to preserve your exact browser origin. Same port8766; no new tunnel
needed if already forwarded. Root three-panel views are preserved.

49supported families across grouped K10/lambda.3/seed43(9families),K18/.3/s43(18),
K22/.3/s42(22). Every family has p100,P80anduniformrandom:147movies/882executions.
Model dropdown selects K; family dropdown lists occurrence-ranked families;
Examples dropdown switches sampling mode. Top10filter ON bydefault(9+10+10).
Uncheck to access all supported components. Fitting and ranking are unchanged.

Six executions per file,2rows×3columns,row-major panels1–3/4–6;2160x1920,60fps.
All six panels use ONE common sampling mode. Videos run at natural pace with
0.5scontext eachside; shorter panels turn dark after finishing. No time warping,
stage waiting, following camera or trajectory inset. Each camera is fixed at
onset position/heading; family crop scale is shared acrossall18selected examples
so mode switches do not change spatial scale. Current raw-source-syllable colour
centroid dot and live ordered labels/durations remain. Use fullscreen/directMP4
and panel detail text for inspection.

## Selection

Original frozen execution_ranking.csv.gz is reused byte-for-byte. Per family,
original animal-balanced medoid exemplar, original four-block descriptors/scales
and within-family distance percentiles are unchanged. New six P80sets retain all
original three P80examples as panels1–3 for all49families.

p100:the exemplar and closest neighbours in topmax(6,ceil(2%N))pool, preferring
animal diversity within that pool. P80:nearest80, prefer different mice inside
±5percentile band; extend only when too few calls to fill6panels. Random:seeded
uniform sample of6distinct eligible calls, without animal-diversity filtering.
Each mode has6unique executions; overlap between modes is permitted.

Actual p100ranks90.91–100(lower values arise for small populations supplying six
nearest examples); P80ranks74.55–83.64(discrete-rank/six-call fallback); random0–100.
Ranks are sampling descriptors, not percentages of behavioural correctness.
No new phenotype/DA selection, generator fitting, latent optimisation or video
interpretation was performed. Full selected native paths/latent support verified.

## Labels

Earlier three-panel P80judgements are displayed separately as read-only reference
from the original browser-storage key. They are NOT silently treated as judgements
of six-panel p100/random movies. New ratings are stored separately per family AND
sampling mode, with trial/panelcount/movieSHA. Browser reload/navigation preserve
notes. Export labels JSON downloads karen_six_panel_labels.json including original
and new ratings; import accepts both original and new package formats. Original
browser key is untouched. If changing localhost/127.0.0.1/port/browser, import your
previous JSON because browser storage belongs to that origin.

## Runtime and checks

Total single-play grid material906.53s(15min7s):P80284.38s,p100345.97s,
random276.18s. Top10P80single-play material~3.14min. Replays/notes/navigation extra.
Generated under detached nohup/3h timeout, fourCPUworkers after2worker preflight,
2numerical/encoderthreads each, combinedRSSguard8GiB,free-diskfloor80GiB,
outputcap8GiB. No accelerator/model initialisation. Parameters inlaunch.json.

Three-model/all-mode preview9movies/54panels passed full decoding/timing and no
missing/offscreen dots;2x3poster inspected. Every final movie/component fully
decoded during rendering; dimensions/FPS/framecounts and longest-panel timing
checked; finalstatus reports zero missing/offscreen dots in selected clips.
Mediahash/native/pixel audit in final_media_audit.json. Browser mode/model
switching, per-mode note separation, old-key preservation, reload/import/export
passed in isolatedChrome; continuous Mac playback is for user review.
IndexHTTP200/MP4Range206exactbytes verified on existingTornado viewer.

12focused tests and fullCPU868tests passed(6existingdeprecation warnings).
Lint3078reviewed/no newdebt;146folder maps synchronized. Raw models/masks/calls
and original movie artifacts preserved. Human six-panel chain review pending.

Runtime folder:outputs/karen_pooled_p80_review_20261015/six_panel_20261016.
