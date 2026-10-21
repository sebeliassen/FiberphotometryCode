"""Two pooled five-bout fits against the existing cleaned three-bout references.

Inputs, vocabulary, penalty, seed, updates and normalized detection policy remain
fixed. Execution is explicit; import/help/plan initialize no numerical backend.
"""

import argparse
import json
import os
import time
from pathlib import Path


def prepare(args):
    import numpy as np
    import pandas as pd

    from .run_reference_seqnmf import digest, guard, log, pack_chunks, write_json

    parent = args.baseline / "recipe.json"
    old = json.loads(parent.read_text())
    folder = args.baseline / "prepared/L3_unresolved_separate"
    reference_path = folder / "fit_references.csv.gz"
    if digest(reference_path) != old["prepared"]["L3_unresolved_separate"]["fit_reference_sha256"]:
        raise ValueError("baseline native references changed")
    refs = pd.read_csv(reference_path, dtype={"animal_id": str})
    refs = refs[refs.token.notna()].copy()
    if refs.session_id.nunique() != 99 or refs.animal_id.nunique() != 21:
        raise ValueError("expected all99recordings/all21mice")
    mapping = json.loads((folder / "mapping.json").read_text())
    vocabulary = max(mapping) + 1
    chunks = []
    for _, chunk in refs.groupby("chunk_id", sort=False):
        guard(args)
        if chunk.session_id.nunique() != 1:
            raise ValueError("chunk crosses recordings")
        if not np.array_equal(chunk.start_frame.to_numpy()[1:], chunk.end_frame_exclusive.to_numpy()[:-1]):
            raise ValueError("chunk bridges unavailable native frames")
        chunk = chunk.copy()
        chunk["token"] = chunk.token.astype(int)
        chunks.append(chunk)
    x, valid, tokens, current = pack_chunks(chunks, np.arange(vocabulary), vocabulary, 5)
    if not np.array_equal(tokens[tokens >= 0], refs.token.to_numpy(int)):
        raise ValueError("length repacking changed grouped tokens or their order")
    guard(args)
    np.savez_compressed(args.output / "matrices.npz", fit_x=x, fit_valid=valid, fit_tokens=tokens)
    current.to_csv(args.output / "fit_references.csv.gz", index=False)
    write_json(args.output / "mapping.json", mapping)
    recipe = dict(parent_recipe=str(parent.resolve()), parent_recipe_sha256=digest(parent),
                  parent_references_sha256=digest(reference_path), length_bouts=5, factors=[10, 18],
                  penalty=.3, seed=43, iterations=80, vocabulary=vocabulary,
                  pooled_recordings=99, pooled_mice=21, parent_source_arrays=old["source_arrays"],
                  detector=old["detector"], activation_threshold=old["training_threshold"],
                  detection_limit="same normalized error/support and radius2 policy; 3/5 may pass where L3 requires2/3; no equal-precision claim",
                  matrix_sha256=digest(args.output / "matrices.npz"),
                  reference_sha256=digest(args.output / "fit_references.csv.gz"),
                  mapping_sha256=digest(args.output / "mapping.json"),
                  source={str(Path(__file__).resolve()): digest(__file__)},
                  resources=dict(threads=args.threads, memory_gb=args.memory_gb,
                                 min_free_disk_gb=args.min_free_disk_gb, max_output_gb=args.max_output_gb),
                  scope="new pooled five-bout settings, not independent generalisation; no condition/DA selection")
    write_json(args.output / "recipe.json", recipe)
    log(f"PREPARED L5: {vocabulary} tokens, {len(tokens)} padded columns, {valid.sum()} supported starts")


def run_fits(args):
    import numpy as np
    import pandas as pd

    from .reference_seqnmf import backend, fit
    from .run_reference_seqnmf import digest, evaluate, guard, log, write_json

    recipe = json.loads((args.output / "recipe.json").read_text())
    if digest(args.output / "matrices.npz") != recipe["matrix_sha256"]:
        raise ValueError("prepared matrices changed")
    if digest(args.output / "fit_references.csv.gz") != recipe["reference_sha256"]:
        raise ValueError("prepared native references changed")
    with np.load(args.output / "matrices.npz", allow_pickle=False) as saved:
        x, valid, tokens = saved["fit_x"], saved["fit_valid"], saved["fit_tokens"]
    refs = pd.read_csv(args.output / "fit_references.csv.gz", dtype={"animal_id": str})
    guard(args)
    runtime = backend(args.device)
    write_json(args.output / "backend.json", runtime[2])
    n = min(4096, x.shape[1])
    supported = valid[:n].copy()
    supported[-4:] = False
    w, h, _ = fit(x[:, :n], supported, factors=10, length=5, penalty=.3,
                  iterations=3, seed=43, runtime=runtime)
    if not np.isfinite(w).all() or not np.isfinite(h).all() or np.any(h[:, ~supported] != 0):
        raise AssertionError("bounded GPU/CPU preflight failed")
    write_json(args.output / "preflight.json", dict(status="PASS", columns=n, updates=3,
                                                    limit="execution/mask smoke check, not convergence or scientific validation"))
    results = []
    for k in (10, 18):
        folder = args.output / f"K{k}"
        folder.mkdir(exist_ok=True)
        if (folder / "result.json").exists():
            result = json.loads((folder / "result.json").read_text())
            for filename, key in (("model.npz", "model_sha256"), ("pooled_calls.csv.gz", "calls_sha256")):
                if digest(folder / filename) != result[key]:
                    raise ValueError("completed fit changed")
            results.append(result)
            continue
        started = time.monotonic()
        log(f"FIT L5 K{k} START: lambda.3/seed43/80updates, same pooled inputs")
        def progress(iteration, error):
            guard(args)
            log(f"FIT L5 K{k} {iteration}/80 reconstruction={error:.6f}")
        w, h, history = fit(x, valid, factors=k, length=5, penalty=.3, iterations=80,
                            seed=43, runtime=runtime, progress=progress)
        if not np.isfinite(w).all() or not np.isfinite(h).all() or np.any(h[:, ~valid] != 0):
            raise AssertionError("nonfinite or unsupported activations")
        thresholds = np.array([np.percentile(v[valid & (v > 1e-4)], 90)
                               if np.any(valid & (v > 1e-4)) else np.inf for v in h])
        metrics, calls = evaluate(w, h, tokens, valid, refs, thresholds)
        np.savez_compressed(folder / "model.npz", W=w, pooled_H=h, thresholds=thresholds,
                            factors=k, length=5, penalty=.3, seed=43)
        calls.to_csv(folder / "pooled_calls.csv.gz", index=False)
        result = dict(K=k, directory=str(folder.resolve()), length_bouts=5, penalty=.3, seed=43,
                      iterations=80, recipe_sha256=digest(args.output / "recipe.json"), **metrics,
                      history=history, elapsed_seconds=time.monotonic() - started,
                      model_sha256=digest(folder / "model.npz"), calls_sha256=digest(folder / "pooled_calls.csv.gz"),
                      stage_support_counts={str(v): int(n) for v, n in calls.stage_support.value_counts().items()})
        write_json(folder / "result.json", result)
        results.append(result)
        log(f"FIT L5 K{k} COMPLETE calls={len(calls)}, coverage={metrics['fit_union_time_coverage']:.4f}")
    write_json(args.output / "status.json", dict(state="FITS_COMPLETE", models=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "fit", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=6.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, pooled_recordings=99, K=[10, 18], length=5,
                              penalty=.3, seed=43, iterations=80, execute=False)))
        return
    if args.threads < 1:
        parser.error("threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        if (args.output / "recipe.json").exists():
            raise ValueError("refusing to overwrite a prepared input recipe")
        prepare(args)
    if args.stage in ("fit", "all"):
        run_fits(args)


if __name__ == "__main__":
    main()
