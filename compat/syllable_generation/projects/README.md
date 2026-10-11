# Projects: overrides, enrich, and examples

## Goal

* Keep **code** generic inside `syllable_generation/`.
* Keep **manual, study-specific info** (genotypes, fiber mappings, per-session notes) in **project folders** as flat files.
* Allow **zero code changes** when a lab adds or edits manual info.

---

## Folder layout

```
syllable_generation/
  projects/
    README.md                     # this file
    jonathan_openfield_example/
      overrides/
        animals.csv               # animal-level info (current minimal needed)
        sessions.csv              # session-level info (optional)
        experiment.csv            # experiment-level defaults (optional)
        time_overlays.csv         # time-window annotations (optional)
        join_spec.yaml            # mapping spec (future-friendly; optional today)
      enrich.py                   # project-specific enrich entry (thin wrapper)
```

---

## Current status (what works today)

* Pipeline builds sessions from data (H5, videos, photometry).
* `projects/enrich_adhoc.py` provides **ad-hoc enrichment**:

  * Reads `animals.csv` with headers: `ID, Geno, Left Hemisphere, Right Hemisphere`.
  * Sets per-session:

    * `metadata.genotype` (uppercased).
    * `metadata.signal_0`, `metadata.signal_1` (normalized, e.g. `l-DMS` → `L-DMS`).
    * If `metadata.fiber_indices` exists, creates `metadata.fiber_regions_by_index` mapping indices → regions.
  * Adds provenance under `metadata.manual_sources`.

**Assumptions**

* `metadata.animal_id` exists (derived earlier from `trial_id` + `chamber_index`).
* `metadata.fiber_indices` may exist (auto-pair from chamber index or manual). If missing, enrichment still sets genotype/signals; you can re-run after fibers exist.

---

## Quickstart (animals-only)

1. Place `animals.csv` in your project’s `overrides/`:

```
ID,Geno,Left Hemisphere,Right Hemisphere
1658,KI,L-DLS,R-VS
740,WT,L-DMS,R-VS
```

2. Run the pipeline, then ad-hoc enrich:

```python
ctx = BuildContext(
    root=Path("/path/to/study"),
    h5_dir=Path("/path/to/study/h5_files"),
    video_dir=Path("/path/to/study/video_files"),
    csv_root=Path("/path/to/study/Jonathan_IFxDN_csvs"),
    overrides_dir=Path("syllable_generation/projects/jonathan_openfield_example/overrides"),
    project="jonathan_openfield_example",
)

sessions = build(ctx)
sessions = enrich_animals_only(ctx, sessions)  # from projects/enrich_adhoc.py
```

3. Inspect one session:

```python
s = sessions[0]
s.metadata["animal_id"], s.metadata.get("genotype"), s.metadata.get("signal_0"), s.metadata.get("fiber_regions_by_index")
```

---

## Planned (CSV + join-spec; future-friendly)

* Support **4 layers** of manual info with a deterministic precedence:

  1. **Session-level** (`sessions.csv` or `session_id` / `trial_id + chamber_index`).
  2. **Animal-level** (`animals.csv` keyed by `animal_id`).
  3. **Experiment-level** (`experiment.csv` single-row defaults).
  4. **Time overlays** (`time_overlays.csv` appended to `metadata.time_overlays`).

* Keep mapping rules in `join_spec.yaml` so labs can change columns without code edits.

### Example CSV schemas

**animals.csv**

```
ID,Geno,Left Hemisphere,Right Hemisphere
728,WT,L-VS,R-DLS
...
```

**sessions.csv** *(optional)*

```
session_id,trial_id,chamber_index,exclude,comment,fps_override,fiber_indices
2024-12-13T12_20_06_bottom_left,T1_1658.740.1666.e,0,false,good cable,59.94,"[0,1]"
```

**experiment.csv** *(optional; one row)*

```
experiment_title,experiment_type,group,fiber_auto_pair_fallback
Open Field IFxDN,openfield,baseline,true
```

**time_overlays.csv** *(optional)*

```
trial_id,chamber_index,t_start_s,t_end_s,drug,dose_mgkg,route
T1_*,0,600,660,CLZ,0.1,IP
```

**join_spec.yaml** *(optional; describes how to map into `metadata`)*

```yaml
sources:
  - name: experiment
    file: experiment.csv
    one_row: true
    map:
      experiment_title: metadata.experiment_title
      experiment_type:  metadata.experiment_type
      group:            metadata.group
      fiber_auto_pair_fallback: metadata.fiber_auto_pair_fallback

  - name: animals
    file: animals.csv
    key: ["ID"]
    map:
      Geno:             metadata.genotype
      Left Hemisphere:  metadata.signal_0
      Right Hemisphere: metadata.signal_1

  - name: sessions
    file: sessions.csv
    key: ["session_id"]
    alt_key: ["trial_id", "chamber_index"]
    map:
      exclude:        metadata.exclude
      comment:        metadata.comment
      fps_override:   metadata.fps_override
      fiber_indices:  metadata.fiber_indices

  - name: time_overlays
    file: time_overlays.csv
    key: ["trial_id","chamber_index"]
    collect:
      "*": metadata.time_overlays
```

**Precedence**

* Apply in order: `experiment → animals → sessions → time_overlays`.
* For scalars: last writer wins.
* For dicts/lists: merge keys; last writer wins on conflicts.

**Fallbacks**

* If `fiber_indices` absent and `chamber_index` present, auto-pair as `[2*k, 2*k+1]`.

**Normalization**

* `genotype` uppercased (`ki` → `KI`).
* Hemisphere labels canonicalized (`l-DMS` → `L-DMS`, `R-vs` → `R-VS`).

**Provenance**

* Every applied file recorded at `metadata.manual_sources` with path and SHA256.

---

## Discovery order (where overrides are read)

1. `ctx.overrides_dir` (explicit).
2. `<ctx.root>/manual_overrides` (next to your data).
3. In-repo example: `syllable_generation/projects/<project_name>/overrides`.

---

## Troubleshooting

* **Missing genotype**: row in `animals.csv` lacks `Geno` → field stays unset.
* **No animal match**: `metadata.animal_id` not found in `animals.csv` → no enrich applied.
* **No fiber mapping produced**: `fiber_indices` not set yet → run enrich again after fibers exist.
* **Multiple rows for same ID**: last row wins (or clean the CSV).

---

## FAQ

* **Can I keep only `animals.csv`?** Yes. Leave other CSVs empty or omit them.
* **Can I use wildcards?** Yes, in `time_overlays.csv` for `trial_id` (planned in join-spec flow).
* **Can I switch to JSON later?** Yes. A JSON manifest variant can coexist or replace CSVs when needed.

---

## Contact points in code

* **Ad-hoc enrich (today)**: `syllable_generation/projects/enrich_adhoc.py::enrich_animals_only`
* **Project entry**: `syllable_generation/projects/jonathan_openfield_example/enrich.py` (thin wrapper)
* **Dialects**: `syllable_generation/dialects/jonathan/*` (builders for sessions + attachments)

---

That’s it. Start simple with `animals.csv`, add more CSVs or a join-spec only when you need them.
