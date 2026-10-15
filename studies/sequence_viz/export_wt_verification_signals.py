"""Export already processed WT 470-nm traces with original clocks and fibre IDs.

Explicit --execute reads the selected trusted local Run12 pickle. No signal is
filtered, normalized, interpolated or refitted. Camera/photometry timestamps stay
in documented FP-system seconds; native IDs retain their independent audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def audit_clock(photo_frames, photo_time, camera_time, *, maximum_error_s=.025):
    """Audit the stored nearest-frame map without fitting an offset or scale."""
    import numpy as np

    frames, times, camera = np.asarray(photo_frames),np.asarray(photo_time,float),np.asarray(camera_time,float)
    if camera.ndim!=1 or not np.isfinite(camera).all() or np.any(np.diff(camera)<0):
        raise ValueError("actual native camera clock is missing or decreases")
    if times.ndim!=1 or frames.shape!=times.shape or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("photometry requires explicit increasing acquisition timestamps")
    valid = np.isfinite(frames) & (frames>=0) & (frames==np.floor(frames)) & (frames<len(camera))
    integer = np.where(valid,frames,0).astype(np.int64)
    if np.any(np.diff(integer[valid])<0):
        raise ValueError("stored native photometry frame mapping decreases")
    error = np.full(len(times),np.nan)
    error[valid] = times[valid]-camera[integer[valid]]
    valid &= abs(error)<=maximum_error_s
    finite = np.isfinite(error)
    positive = np.diff(camera)
    positive = positive[positive>0]
    report = dict(eligible_samples=int(valid.sum()),total_samples=len(times),alignment_fraction=float(valid.mean()),
        maximum_allowed_error_s=maximum_error_s,maximum_absolute_error_s=float(np.max(abs(error[finite]))) if finite.any() else None,
        p99_absolute_error_s=float(np.quantile(abs(error[finite]),.99)) if finite.any() else None,
        photometry_median_dt_s=float(np.median(np.diff(times))),camera_median_positive_dt_s=float(np.median(positive)),
        camera_duplicate_timestamps=int((np.diff(camera)==0).sum()),duplicate_native_photo_frames=int((np.diff(integer[valid])==0).sum()),
        policy="original frame map; no alignment offset/scale fitted; samples outside25ms excluded")
    return integer, valid, report


def export(*, source, audit, manifest, output, cap_gib=24):
    import numpy as np
    import pandas as pd

    from fpcode.common.cache_manifest import atomic_path, atomic_write_json
    from fpcode.common.file_io import safe_load_pickle
    from studies.sequence_viz.broad_search_inputs import _load_npz
    from studies.sequence_viz.run_broad_search import digest

    source,output = Path(source).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new output; preserve previous signal export")
    resource.setrlimit(resource.RLIMIT_AS,(int(cap_gib*1024**3),int(cap_gib*1024**3)))
    started,before = time.monotonic(),source.stat()
    old = json.loads(Path(audit).read_text())
    if digest(source)!=old["sha256"]:
        raise ValueError("selected session pickle differs from existing Run12 identity audit")
    frozen = json.loads(Path(manifest).read_text())
    cohort = next(c for c in frozen["cohorts"] if c["cohort_id"]=="jonathan")
    if cohort["model_id"]!="amph_prod_k150k_run_12":
        raise ValueError("WT DA export must use the selected Run12 behavioural model")
    sessions = safe_load_pickle(source)
    selected = [s for s in sessions if s.metadata.get("genotype")=="WT"]
    if len(selected)!=16 or len({str(s.metadata["animal_id"]) for s in selected})!=8:
        raise ValueError("WT cohort differs from documented8mice/16recordings")
    output.mkdir(parents=True)
    reports,camera_cache = [],{}
    for session in selected:
        recording,meta = str(session.session_id),session.metadata
        row = next(r for r in cohort["records"] if r["recording"]==recording)
        if str(meta["animal_id"])!=row["subject"] or meta["drug"]!=row["metadata"]["drug"]:
            raise ValueError("WT identity/drug differs from frozen behavioural input")
        dense = session.all_data_df
        if not np.array_equal(dense.index.to_numpy(),dense.frame_index.to_numpy()):
            raise ValueError("dense frame coordinate differs from independent native audit")
        native_ref = row["source_provenance"]["source"]
        if digest(native_ref["path"])!=native_ref["sha256"]:
            raise ValueError("native reference export changed")
        native = _load_npz(native_ref["path"],max_bytes=32*1024**2)
        frames = native["native_frames"]
        if not np.array_equal(dense.syllable.iloc[frames].to_numpy(float),native["labels"].astype(float)):
            raise ValueError("stored DA session labels differ from selected native model")
        # The model-bound cropped video is CenterTop; audit that camera clock.
        original_camera_path = Path(meta["alltimestamps_csv_path"])
        suffix = original_camera_path.name.split("_AllTimestampsCombined",1)[1]
        camera_path = original_camera_path.parent/("CenterTopCam_AllTimestampsCombined"+suffix)
        if str(camera_path) not in camera_cache:
            clock = pd.read_csv(camera_path,usecols=["FP3002_System_Timestamp"])["FP3002_System_Timestamp"].to_numpy(float)
            camera_cache[str(camera_path)] = (clock,digest(camera_path))
        camera,camera_hash = camera_cache[str(camera_path)]
        # Densification can retain one or two trailing frames without a camera
        # timestamp. Keep the actual clock unchanged; never pad or extrapolate.
        native_clock_fraction = float(np.mean(frames < len(camera)))
        if native_clock_fraction < .8:
            raise ValueError("less than80% of verified native support has an actual camera timestamp")
        table = session.photometry["470"]
        photo_frames,valid,clock_audit = audit_clock(table.frame_idx.to_numpy(),table.SystemTimestamp.to_numpy(float),camera)
        if clock_audit["alignment_fraction"]<.8:
            raise ValueError("less than80% of stored photometry samples pass declared clock mapping")
        arrays = dict(native_camera_time_s=camera,native_photo_frames=photo_frames,
            photo_time_s=table.SystemTimestamp.to_numpy(float),alignment_valid=valid)
        signals = []
        for base,info in sorted(meta["signal_meta"].items()):
            if info["orig_letter"]!="G":
                continue
            for suffix in ("","_dff","_zscore","_zscore_datta"):
                column = base+suffix
                if column not in table:
                    continue
                arrays[column] = table[column].to_numpy(dtype=float,na_value=np.nan)
                signals.append(dict(column=column,base_signal=base,**info,
                    finite_aligned_samples=int((valid & np.isfinite(arrays[column])).sum())))
        if not signals:
            raise ValueError("no original green fibre traces available")
        target = output/(hashlib.sha256(recording.encode()).hexdigest()[:16]+".npz")
        with atomic_path(target) as temporary:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle,**arrays)
        reports.append(dict(recording=recording,subject=row["subject"],genotype="WT",drug=meta["drug"],
            source_model_id=cohort["model_id"],npz_path=str(target),sha256=digest(target),
            signals=signals,clock_audit=clock_audit,camera_timestamp_source=str(camera_path),camera_timestamp_sha256=camera_hash,
            camera_rows=len(camera),dense_native_rows=len(dense),native_supported_clock_fraction=native_clock_fraction,
            original_mapping_timestamp_source=meta["alltimestamps_csv_path"],source_label_rows_verified=len(frames),
            signal_processing_metadata={k:meta.get(k) for k in ("dff_iso_channel","dff_computed_channels",
                "datta_replication_enabled","reference_interpolation","photometry_source")}))
    after = source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError("source pickle changed during export")
    backend = [m for m in sys.modules if m.split(".")[0] in {"jax","torch","keypoint_moseq"}]
    if backend:
        raise ValueError("unexpected modelling backend import")
    report = dict(source=str(source),source_sha256=old["sha256"],source_model_id=cohort["model_id"],records=reports,
        trace_policy="Stored470nm green fibres; primary_zscore_datta; dff/global_zscore sensitivity; raw for audit only",
        unit_policy="FPsystem timestamps in documented seconds; no nominal-frame-rate conversion for DA",
        processing_policy="No filtering/interpolation/normalization; existing centered Datta transform is descriptive, not causal",
        sensor_evidence="User DA designation and existing dLight1 study narrative; per-fibre metadata supplies colour/region/hemisphere, not sensor name",
        resource_settings=dict(address_space_cap_gib=cap_gib,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            elapsed_s=time.monotonic()-started),backend_imports=backend)
    atomic_write_json(output/"manifest.json",report)
    return {k:report[k] for k in ("source_model_id","resource_settings")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","audit","manifest","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({k:str(v) for k,v in vars(args).items()}))
        return 0
    kwargs = vars(args)
    kwargs.pop("execute")
    print(json.dumps(export(**kwargs)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
