"""Extract the explicitly selected Karen pickle after exact V2 label verification."""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fpcode.adapters.behavior_sessions import record_frame_arrays, verify_result_projection
from fpcode.common.run_record import RunRecord

CONDITIONS = ("3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--vfi", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "frame_cache"
    cache.mkdir(exist_ok=True)
    provenance = RunRecord("karen-atlas-extraction", args.output / "extraction_provenance", config=vars(args))
    provenance.config = {k: str(v) for k, v in vars(args).items()}
    for path in (args.pickle, args.results, args.vfi, args.labels):
        provenance.add_input_file(path)
    print("Loading approved Karen pickle", flush=True)
    with args.pickle.open("rb") as stream:
        sessions = pickle.load(stream)
    pd.read_csv(args.labels).to_csv(args.output / "syllable_annotations.csv", index=False)
    records, excluded = [], []
    with h5py.File(args.results, "r") as results, np.load(args.vfi, allow_pickle=False) as vfi:
        for s in sessions:
            md = s.metadata
            if md.get("genotype") not in ("PFF", "Monomer") or md.get("condition") not in CONDITIONS:
                excluded.append({"session_id": s.session_id, "reason": "outside_requested_groups_or_waves"})
                continue
            arrays = record_frame_arrays(s, export_columns=("velocity_px_s", "static_21kp_v2_syllable"))
            keys = [k for k in results if k == s.session_id or k.startswith(s.session_id + "_seg")]
            if not keys:
                raise ValueError(f"no selected-model segment for {s.session_id}")
            check = verify_result_projection(arrays, {k: {"syllable": results[k]["syllable"][:]}
                                                        for k in keys}, vfi)
            if "exported_static_21kp_v2_syllable" in arrays:
                exp = arrays["exported_static_21kp_v2_syllable"]
                if not np.array_equal(np.where(np.isfinite(exp), exp, -1), arrays["labels"]):
                    raise ValueError("exported V2 labels disagree with syllable table")
            np.savez_compressed(cache / (s.session_id + ".npz"), **arrays)
            records.append({"session_id": s.session_id, "animal_id": str(md["animal_id"]),
                            "group": md["genotype"], "condition": md["condition"],
                            "week": md["weeks_post_injection"], "drug": md.get("drug"),
                            "date": str(md.get("timestamp", s.session_id))[:10],
                            "chamber": md.get("chamber"), "fps": float(arrays["fps"]),
                            "n_frames": len(arrays["frames"]), "latent_dimensions": arrays["latent"].shape[1],
                            "video_path": str(s.video_refs_df.loc[s.video_refs_df["role"] == "primary", "path"].iloc[0]),
                            "model_name": getattr(s, "kpms_model_name", None), **check})
            print(f"Verified/extracted {len(records)}: {s.session_id}", flush=True)
    frame = pd.DataFrame(records)
    if frame.groupby("animal_id")["group"].nunique().max() != 1:
        raise ValueError("animal group changes across recordings")
    frame.to_csv(args.output / "sessions.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output / "excluded_sessions.csv", index=False)
    (args.output / "input_paths.json").write_text(json.dumps({k: str(v.resolve()) for k, v in vars(args).items()}, indent=2))
    provenance.finish(extra={"sessions_included": len(records), "sessions_excluded": len(excluded),
                             "exact_native_frame_label_verification": True})
    print(f"Extraction complete: {len(records)} sessions", flush=True)


if __name__ == "__main__":
    main()
