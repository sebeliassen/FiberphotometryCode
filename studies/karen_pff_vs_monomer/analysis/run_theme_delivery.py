"""Execute the frozen small Karen theme atlas; no new model fitting or search."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")
    temporary.replace(path)


def log(message):
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def guard(args):
    import psutil
    total = psutil.Process().memory_info().rss
    if total > args.memory_gb * 1024 ** 3:
        raise MemoryError("configured RSS budget exceeded")
    if shutil.disk_usage(args.output).free < args.min_free_disk_gb * 1024 ** 3:
        raise RuntimeError("free disk floor reached")
    if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > args.max_output_gb * 1024 ** 3:
        raise RuntimeError("output size budget reached")


def prepare(args):
    import numpy as np
    import pandas as pd

    library_path = args.output / "library.json"
    library = json.loads(library_path.read_text())
    if not library.get("frozen_before_group_statistics"):
        raise ValueError("group-blind theme definitions must be frozen first")
    recipe_path = args.fits / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    sessions_path = Path(recipe["input_metadata"][0]["path"])
    if digest(sessions_path) != recipe["input_metadata"][0]["sha256"]:
        raise ValueError("pooled source sessions changed")
    mapping_path = Path(library["mapping"]["path"])
    if digest(mapping_path) != library["mapping"]["sha256"]:
        raise ValueError("theme grouping changed")
    mapping = json.loads(mapping_path.read_text())
    if mapping != library["mapping"]["raw_to_grouped"]:
        raise ValueError("raw constituent mapping mismatch")
    sessions = pd.read_csv(sessions_path, dtype={"animal_id": str})
    sessions["wave"] = sessions.condition.replace({"L_dopa": "Ldopa"})
    if len(sessions) != 99 or sessions.animal_id.nunique() != 21:
        raise ValueError("expected all 99 recordings and 21 mice")
    if set(sessions.session_id) != {r["session_id"] for r in recipe["source_arrays"]}:
        raise ValueError("source recording roster differs")
    sessions.to_csv(args.output / "sessions.csv", index=False)
    sources = []
    for row in recipe["source_arrays"]:
        guard(args)
        path = Path(row["path"])
        with np.load(path, allow_pickle=False) as cache:
            expected = cache["analysis_excluded"] if "analysis_excluded" in cache else np.zeros(len(cache["frames"]), bool)
            identity = hashlib.sha256()
            for a in (cache["frames"], cache["labels"], expected):
                identity.update(str((a.shape, str(a.dtype))).encode())
                identity.update(np.ascontiguousarray(a).tobytes())
            if identity.hexdigest() != row["relevant_frame_label_mask_sha256"] or float(cache["fps"]) != 60.:
                raise ValueError("fitted native frames/labels/exclusions changed")
        sources.append(dict(row, sha256=digest(path)))
    manifest = dict(library_sha256=digest(library_path), recipe_path=str(recipe_path),
                    recipe_sha256=digest(recipe_path), sessions_sha256=digest(args.output / "sessions.csv"),
                    source_arrays=sources, input_clock="explicit native 60Hz, half-open intervals",
                    selection="frozen group-blind named cores; no new factor fit or group-based selection",
                    software_sources={str(p.resolve()): digest(p) for p in
                                      (Path(__file__), Path(__file__).with_name("theme_measurements.py"),
                                       Path(__file__).with_name("theme_statistics.py"),
                                       Path(__file__).with_name("theme_organisation.py"))},
                    resources=dict(threads=args.threads, memory_gb=args.memory_gb,
                                   max_output_gb=args.max_output_gb, min_free_disk_gb=args.min_free_disk_gb),
                    policy=dict(recruitment="all valid modelled time including unresolved; inclusive core count and union time",
                                organisation="exclusive score1 calls; unresolved frames are barriers; gaps0/1/2s,1sprimary",
                                motion="native finite consecutive pairs; missing pair => unavailable whole-window mean; px/s and rad/s",
                                intra="same raw constituent/block/300s/recording-median occupancy quadrant; >=3bouts each side; outside every theme",
                                first_order="199 plug-in Markov simulations/session; fitted transition uncertainty not integrated",
                                inference="mouse independent, ratio pooling recordings then equal-wave means; no held-out or causal drug claim"))
    dump(args.output / "input_manifest.json", manifest)
    log(f"PREPARE COMPLETE: {len(sessions)} recordings / {sessions.animal_id.nunique()} mice / {len(library['themes'])} frozen themes")


def measure(args):
    import numpy as np
    import pandas as pd

    from .theme_measurements import session_measurements

    manifest = json.loads((args.output / "input_manifest.json").read_text())
    if digest(args.output / "library.json") != manifest["library_sha256"]:
        raise ValueError("theme definitions changed after freeze")
    library = json.loads((args.output / "library.json").read_text())
    mapping = np.asarray(library["mapping"]["raw_to_grouped"], int)
    sessions = pd.read_csv(args.output / "sessions.csv", dtype={"animal_id": str})
    if args.limit_sessions:
        sessions = sessions.head(args.limit_sessions)
    sources = {row["session_id"]: row for row in manifest["source_arrays"]}
    pieces, audits = {}, []
    started = time.perf_counter()
    for count, meta in enumerate(sessions.to_dict("records"), 1):
        guard(args)
        directory = args.output / "session_tables" / meta["session_id"]
        directory.mkdir(parents=True, exist_ok=True)
        audit_path = directory / "audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["library_sha256"] != manifest["library_sha256"]:
                raise ValueError("saved recording measurements belong to different themes")
        else:
            source = sources[meta["session_id"]]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("native feature cache changed")
            with np.load(source["path"], allow_pickle=False) as cache:
                data = {key: cache[key] for key in cache.files}
            tables, organisation, audit = session_measurements(meta, data, mapping, library["themes"])
            for name, table in tables.items():
                if name == "branches":
                    names = library["token_names"]
                    table["next_label"] = table.next_token.map(lambda t: "block_end" if t < 0 else names[str(t)])
                table.to_csv(directory / (name + ".csv.gz"), index=False)
            for name in ("decisions", "selected", "timeline", "pairs", "runs"):
                organisation[name].to_csv(directory / ("inter_" + name + ".csv.gz"), index=False)
            audit.update(library_sha256=manifest["library_sha256"],
                         source_sha256=source["sha256"],
                         outputs={p.name: digest(p) for p in directory.glob("*.csv.gz")})
            dump(audit_path, audit)
        audits.append(audit)
        for path in directory.glob("*.csv.gz"):
            if path.name not in audit["outputs"] or digest(path) != audit["outputs"][path.name]:
                raise ValueError("saved measurement identity changed")
            if not path.name.startswith("inter_"):
                pieces.setdefault(path.name, []).append(path)
        log(f"MEASURE {count}/{len(sessions)} {meta['session_id']}: {audit['full_calls']} full cores")
    for name, paths in pieces.items():
        chunks = [pd.read_csv(path, dtype={"animal_id": str}) for path in paths]
        combined = pd.concat(chunks, ignore_index=True)
        combined.to_csv(args.output / name, index=False)
    occurrences = pd.read_csv(args.output / "occurrences.csv.gz", dtype={"animal_id": str})
    counts = occurrences.groupby("theme").size().to_dict()
    if not args.limit_sessions:
        if counts != {theme["id"]: theme["cohort_occurrence_estimate"] for theme in library["themes"]}:
            raise ValueError("independent raw-frame counts differ from group-blind candidate inventory")
        index = pd.read_csv(args.output / "occurrence_index.csv.gz", dtype={"animal_id": str})
        columns = ["session_id", "animal_id", "theme", "start_frame", "end_frame_exclusive", "observed_tokens"]
        a = occurrences[columns].sort_values(columns[:-1]).reset_index(drop=True)
        b = index[columns].sort_values(columns[:-1]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    dump(args.output / "measurement_audit.json", dict(status="PASS", sessions=len(sessions),
        mice=int(sessions.animal_id.nunique()), calls=len(occurrences), counts=counts,
        total_valid_seconds=sum(a["valid_seconds"] for a in audits),
        total_named_seconds=sum(a["named_seconds"] for a in audits),
        selected_calls=sum(a["organisation"]["included_calls"] for a in audits),
        selected_pair_opportunities_primary="see per-recording inter_pairs audits",
        elapsed_seconds=time.perf_counter() - started, complete_cohort=not bool(args.limit_sessions),
        library_sha256=manifest["library_sha256"], input_manifest_sha256=digest(args.output / "input_manifest.json")))
    log(f"MEASUREMENT COMPLETE: {len(occurrences)} occurrences, {time.perf_counter() - started:.1f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "measure", "statistics", "report", "all"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--max-output-gb", type=float, default=4.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--replicates", type=int, default=9999)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, execute=False, output=str(args.output))))
        return
    if args.threads < 1 or args.replicates < 99 or min(args.memory_gb, args.max_output_gb) <= 0:
        parser.error("invalid positive execution/resource settings")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    args.output = args.output.resolve()
    args.fits = args.fits.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    guard(args)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("measure", "all"):
        measure(args)
    if args.stage in ("statistics", "all"):
        import pandas as pd

        from .theme_baselines import build as build_baselines
        from .theme_statistics import infer
        if not json.loads((args.output / "measurement_audit.json").read_text())["complete_cohort"]:
            raise ValueError("profiling excerpt cannot become a full-cohort inference")
        baseline = build_baselines(args.output)
        readouts = pd.concat([pd.read_csv(args.output / "readouts.csv.gz", dtype={"animal_id": str}),
                              baseline], ignore_index=True)
        dump(args.output / "statistics_inputs.json", dict(
            rows=len(readouts), base_rows=len(readouts) - len(baseline), baseline_rows=len(baseline),
            note="Planned primitive/movement controls, counted without phenotype selection; all BH corrections recomputed jointly",
            input_hashes={name: digest(args.output / name) for name in
                          ("library.json", "readouts.csv.gz", "baseline_readouts.csv.gz", "baseline_metadata.json")},
            source_hashes={str(p.resolve()): digest(p) for p in
                           (Path(__file__), Path(__file__).with_name("theme_baselines.py"),
                            Path(__file__).with_name("theme_statistics.py"))}))
        result = infer(readouts, args.output / "statistics", replicates=args.replicates)
        log(f"STATISTICS COMPLETE: {result['n_tests']} tests; all raw p/family q/global q saved")
    if args.stage in ("report", "all"):
        from .report_theme_delivery import build
        build(args.output)
        log("REPORT COMPLETE")
    dump(args.output / "stage_status.json", dict(stage=args.stage, status="COMPLETE", limited=args.limit_sessions))


if __name__ == "__main__":
    main()
