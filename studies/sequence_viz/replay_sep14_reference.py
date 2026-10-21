"""Replay the pinned September seqNMF detector from saved W/H without fitting.

Historical endpoint formulas are reproduced for identity checks, then audited
separately. This command does not choose a Karen recipe or repair old results.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

LEGACY = "b83298fcfb15ffd5ce3f6dbbd863b1f8e98af03e"
SOLVER = "syllable_generation/analysis/sequence_viz/optimized_seqnmf/convolutional_seqnmf.py"
PREFIX = "compat/syllable_generation/analysis/sequence_viz/"
MODEL = PREFIX + "optimized_seqnmf/optimized_seqnmf_model_results.pkl"
DATA = PREFIX + "data/unified_sequence_dataset.pkl"
CALLS = PREFIX + "optimized_seqnmf/optimized_seqnmf_verified_episodes.csv"
METADATA = PREFIX + "data/session_metadata_ground_truth.csv"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def detector_type(source, numpy):
    """Compile only two inspected historical methods; skip module initialization."""
    tree = ast.parse(source)
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "ConvolutionalSeqNMF")
    names = {"get_dominant_token_progression", "retrieve_episodes"}
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise ValueError("pinned detector methods are incomplete")
    module = ast.Module(body=[ast.ClassDef(name=owner.name, bases=[], keywords=[], body=methods,
                                          decorator_list=[])], type_ignores=[])
    namespace = {"np": numpy}
    exec(compile(ast.fix_missing_locations(module), f"{LEGACY}:{SOLVER}", "exec"), namespace)
    return namespace[owner.name]


def bind_native(full, metadata_path, native_path, output):
    """Bind every historical bout to saved KPMS rows and explicit native VFI."""
    import h5py
    import numpy as np
    import pandas as pd

    native = json.loads(native_path.read_text())
    metadata = pd.read_csv(metadata_path, dtype={"animal_id": str})
    summaries = []
    model_path, vfi_path = Path(native["model_results"]), Path(native["model_vfi"])
    source_files = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in (native_path, model_path, vfi_path)]
    with h5py.File(model_path, "r") as model, np.load(vfi_path, allow_pickle=False) as index:
        for sid, frame in full.groupby("session_idx", sort=False):
            first = frame.iloc[0]
            matches = metadata[(metadata.animal_id == first.animal_id) & (metadata.drug == first.drug)]
            if len(matches) != 1:
                raise ValueError(f"ambiguous historical recording identity: session {sid}")
            row = matches.iloc[0]
            native_rows = [r for r in native["rows"] if r["recording_key"] == row.results_dict_key]
            if len(native_rows) != 1 or str(native_rows[0]["animal_id"]) != first.animal_id:
                raise ValueError("native manifest recording/animal mismatch")
            vfi, labels = index[row.results_dict_key], model[row.results_dict_key]["syllable"][:]
            starts, ends = frame.start_frame.to_numpy(dtype=int), frame.end_frame.to_numpy(dtype=int)
            a, b = np.searchsorted(vfi, starts), np.searchsorted(vfi, ends)
            if not ((a < len(vfi)).all() and (b < len(vfi)).all()):
                raise ValueError("historical coordinates outside saved VFI")
            if not (np.array_equal(vfi[a], starts) and np.array_equal(vfi[b], ends) and (b >= a).all()):
                raise ValueError("historical starts/ends do not bind to exact saved native coordinates")
            expected = frame.syllable.to_numpy(dtype=int)
            changes = np.r_[0, np.cumsum(labels[1:] != labels[:-1])]
            if not (np.array_equal(labels[a], expected) and ((changes[b] - changes[a]) == 0).all()):
                raise ValueError("historical bout labels disagree with saved KPMS rows")
            model_fps = float(native["clock_evidence"]["model_fps"])
            durations = (b - a + 1) / model_fps
            if not np.allclose(durations, frame.duration_sec.to_numpy(), atol=1e-9, rtol=0):
                raise ValueError("historical duration does not match inclusive saved model-row span")
            # The existing reviewed Run12 policy explicitly projects regular
            # +2 VFI steps to two native frames, block-terminal rows to one.
            gaps = np.r_[0, np.cumsum(np.diff(vfi) != 2)]
            next_row = np.minimum(b + 1, len(vfi) - 1)
            regular_successor = (b + 1 < len(vfi)) & (vfi[next_row] == vfi[b] + 2)
            full.loc[frame.index, "_native_end_exclusive"] = ends + np.where(regular_successor, 2, 1)
            full.loc[frame.index, "_native_internal_gap"] = (gaps[b] - gaps[a]) > 0
            summaries.append({"session_idx": int(sid), "session_id": row.session_id,
                              "recording_key": row.results_dict_key, "animal_id": first.animal_id,
                              "genotype": first.genotype, "drug": first.drug, "bouts_bound": len(frame),
                              "start_end_and_label_identity": True, "inclusive_model_row_duration_matches": True,
                              "bouts_spanning_native_gaps": int(((gaps[b] - gaps[a]) > 0).sum())})
    pd.DataFrame(summaries).to_csv(output / "source_native_binding.csv", index=False)
    return {"status": "ALL_HISTORICAL_BOUTS_BOUND_TO_SAVED_KPMS_ROWS", "source_files": source_files,
            "sessions": len(summaries), "bouts": len(full),
            "historical_end_frame_semantics": "inclusive native coordinate of final model row",
            "duration_semantics": "inclusive model-row count divided by documented 40 Hz model clock",
            "clock_evidence": native["clock_evidence"],
            "limit": "Uses existing reviewed Run12 projection policy; no new all-recording video timestamp calibration."}


def replay(args):
    # Cap address space and numerical workers before loading numerical libraries.
    import resource

    cap = int(args.memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import pickle
    import sys

    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_info, threadpool_limits

    started = time.monotonic()
    code = Path(__file__).resolve().parents[2]
    manifest_path = code / "studies/sequence_viz/baselines/seqnmf_sep14/reference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {row["path"]: row["sha256"] for row in manifest["artifact_inventory"]}
    inputs = []
    for relative in (MODEL, DATA, CALLS, METADATA):
        path = args.input_root / relative
        actual = digest(path)
        if actual != expected[relative]:
            raise ValueError(f"reference fingerprint mismatch: {relative}")
        inputs.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    source = subprocess.check_output(["git", "show", f"{LEGACY}:{SOLVER}"], cwd=code)
    reference_source = next(row for row in manifest["legacy_sources"] if row["path"] == SOLVER)
    if hashlib.sha256(source).hexdigest() != reference_source["sha256"]:
        raise ValueError("historical solver source identity changed")
    # These are explicitly selected, fingerprint-verified local research payloads.
    with (args.input_root / MODEL).open("rb") as stream:
        model = pickle.load(stream)
    with (args.input_root / DATA).open("rb") as stream:
        bundle = pickle.load(stream)
    W, H, hp = model["W"], model["H"], model["hyperparameters"]
    if hp != {"K": 10, "L": 3, "lambda_ortho": 1.0, "max_iter": 80, "bouts_fitted": 50000}:
        raise ValueError("saved model recipe differs from the recovered reference")
    if W.shape != (100, 10, 3) or H.shape != (10, 50000):
        raise ValueError("unexpected saved weight/activation shapes")
    if not (np.isfinite(W).all() and np.isfinite(H).all() and (W >= 0).all() and (H >= 0).all()):
        raise ValueError("invalid saved nonnegative model arrays")

    sessions, frames = [], []
    offset = 0
    for session in bundle["sessions"]:
        frame = session["bouts_df"].copy()
        frame["animal_id"], frame["genotype"], frame["drug"] = (
            str(session["animal_id"]), session["genotype"], session["drug"])
        frame["session_idx"] = int(session["session_idx"])
        frame["source_fps"] = float(session["fps"])
        end = offset + len(frame)
        fitted = max(0, min(end, H.shape[1]) - offset)
        sessions.append({"session_idx": int(session["session_idx"]), "animal_id": str(session["animal_id"]),
                         "genotype": session["genotype"], "drug": session["drug"],
                         "global_bout_start": offset, "global_bout_end_exclusive": end,
                         "source_bouts": len(frame), "fitted_bouts": fitted,
                         "source_declared_fps": float(session["fps"])})
        frames.append(frame)
        offset = end
    full = pd.concat(frames, ignore_index=True)
    native_binding = bind_native(full, args.input_root / METADATA, args.native_manifest, args.output)
    fitted = full.iloc[:H.shape[1]]
    tokens = fitted.syllable.to_numpy(dtype=int)
    if not ((tokens >= 0) & (tokens < W.shape[0])).all():
        raise ValueError("out-of-vocabulary training token")
    X = np.zeros((W.shape[0], H.shape[1]), dtype=np.float32)
    X[tokens, np.arange(len(tokens))] = 1
    solver_class = detector_type(source, np)
    solver = solver_class.__new__(solver_class)
    solver.W, solver.H, solver.L = W, H, hp["L"]
    rows = []
    with threadpool_limits(limits=args.threads):
        for k in range(hp["K"]):
            episodes = solver.retrieve_episodes(X, factor_idx=k, tau_percentile=90.,
                                                max_whole_template_err=.68, min_stage_support=.50)
            for episode in episodes:
                a, b = episode["start_bout"], episode["end_bout"]
                origin = fitted.iloc[a]
                # Exact historical formula, including its next-bout endpoint.
                next_bout = fitted.iloc[min(b, len(fitted) - 1)]
                episode.update(factor_id=k + 1, animal_id=origin.animal_id, genotype=origin.genotype,
                               drug=origin.drug, session_idx=int(origin.session_idx),
                               start_frame=int(origin.start_frame),
                               end_frame=int(next_bout.start_frame + int(next_bout.duration_sec * 80.)),
                               duration_sec=float(fitted.duration_sec.iloc[a:b].sum()))
                rows.append(episode)
    actual = pd.DataFrame(rows)
    exported = pd.read_csv(args.input_root / CALLS, dtype={"animal_id": str})
    for column in ("observed_tokens", "dominant_template"):
        exported[column] = exported[column].map(ast.literal_eval)
    key = ["factor_id", "start_bout", "end_bout"]
    def ordered(frame):
        return frame.sort_values(key).reset_index(drop=True)

    actual, exported, embedded = ordered(actual), ordered(exported), ordered(model["episodes"])
    pd.testing.assert_frame_equal(actual[exported.columns], exported, check_dtype=False, atol=1e-7, rtol=1e-6)
    pd.testing.assert_frame_equal(actual[embedded.columns], embedded, check_dtype=False, atol=1e-7, rtol=1e-6)
    float_columns = ["peak_activation", "template_error", "stage_support", "duration_sec"]
    errors = {name: float(np.max(np.abs(actual[name].to_numpy() - exported[name].to_numpy())))
              for name in float_columns}
    summaries = actual.groupby(["factor_id", "animal_id", "genotype", "drug"], as_index=False).agg(
        episodes=("start_bout", "size"), historical_summed_duration_sec=("duration_sec", "sum"))
    expected_summaries = exported.groupby(["factor_id", "animal_id", "genotype", "drug"], as_index=False).agg(
        episodes=("start_bout", "size"), historical_summed_duration_sec=("duration_sec", "sum"))
    pd.testing.assert_frame_equal(summaries, expected_summaries, check_dtype=False, atol=1e-7, rtol=1e-6)

    audits = []
    for episode in actual.itertuples():
        window = fitted.iloc[episode.start_bout:episode.end_bout]
        crosses_session = window.session_idx.nunique() != 1
        crosses_segment = crosses_session or window.segment_id.nunique() != 1
        support_gap = (window.start_frame.to_numpy()[1:] > window._native_end_exclusive.to_numpy()[:-1]).any()
        support_gap = bool(support_gap or window._native_internal_gap.any() or crosses_session)
        last_end = int(window._native_end_exclusive.iloc[-1])
        audits.append({"factor_id": episode.factor_id, "start_bout": episode.start_bout,
                       "end_bout_exclusive": episode.end_bout,
                       "crosses_session": bool(crosses_session), "crosses_segment": bool(crosses_segment),
                       "native_support_gap": support_gap,
                       "historical_end_frame": episode.end_frame, "last_included_bout_end_frame": last_end,
                       "end_frame_difference": int(episode.end_frame - last_end)})
    audit = pd.DataFrame(audits)
    actual.to_csv(args.output / "replayed_episodes.csv", index=False)
    summaries.to_csv(args.output / "animal_factor_summaries.csv", index=False)
    pd.DataFrame(sessions).to_csv(args.output / "training_session_order.csv", index=False)
    audit.to_csv(args.output / "historical_boundary_audit.csv", index=False)
    fitted_sessions = [row for row in sessions if row["fitted_bouts"]]
    report = {"status": "SAVED_DETECTOR_REPLAY_MATCHED", "inputs": inputs,
              "legacy_source": reference_source, "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=code, text=True).strip(), "driver_sha256": digest(__file__),
              "model": {"W_shape": list(W.shape), "H_shape": list(H.shape), "W_dtype": str(W.dtype),
                        "H_dtype": str(H.dtype), "hyperparameters": hp},
              "comparison": {"replayed_episodes": len(actual), "embedded_and_exported_catalogues_match": True,
                             "animal_factor_summaries_match": True, "maximum_absolute_csv_errors": errors,
                             "float_atol": 1e-7, "float_rtol": 1e-6,
                             "integer_identifiers_and_token_strings": "exact"},
              "training_scope": {"input_sessions": len(sessions), "input_bouts": len(full),
                                 "prefix_bouts": len(fitted), "contributing_sessions": len(fitted_sessions),
                                 "contributing_animals": len({r["animal_id"] for r in fitted_sessions}),
                                 "drug_bouts": fitted.groupby("drug").size().to_dict()},
              "native_binding": native_binding,
              "transfer_audit": {"session_crossing_calls": int(audit.crosses_session.sum()),
                                 "segment_crossing_calls": int(audit.crosses_segment.sum()),
                                 "calls_with_native_support_gap": int(audit.native_support_gap.sum()),
                                 "historical_end_differs_from_last_included_bout": int((audit.end_frame_difference != 0).sum()),
                                 "clock_status": "Source rows and native VFI bound; existing clock evidence retained with its stated limits."},
              "runtime": {"threads_requested": args.threads, "thread_pools": threadpool_info(),
                          "memory_cap_gib": args.memory_gb, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "elapsed_sec": time.monotonic() - started,
                          "jax_loaded": "jax" in sys.modules, "model_fits": 0},
              "limits": ["Replays detection using saved W AND saved H; does not independently re-infer activations or refit.",
                         "Historical frames/durations reproduced literally; match does not establish valid physical clocks/endpoints.",
                         "Reference is the raw-state branching model, not the unresolved unified-sniff quantitative slide.",
                         "No Karen model fitted, vocabulary selected, or biological interpretation accepted."]}
    (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Original checkout containing ignored reference artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True,
                        help="Existing reviewed Run12 native-frame provenance manifest")
    parser.add_argument("--memory-gb", type=float, default=4.)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="Explicitly permit loading the pinned local research pickles")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("replay requires --execute; no files were loaded")
    if args.memory_gb < 1 or args.threads < 1:
        parser.error("positive thread count and at least 1 GiB memory cap required")
    args.input_root, args.output = args.input_root.resolve(), args.output.resolve()
    args.native_manifest = args.native_manifest.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    replay(args)


if __name__ == "__main__":
    main()
