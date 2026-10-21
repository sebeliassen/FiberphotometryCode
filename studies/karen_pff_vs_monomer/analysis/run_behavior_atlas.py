"""Run the Karen atlas using shared algorithms and explicit study policy."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from fpcode.common.run_record import RunRecord
from fpcode.workflows.atlas_statistics import run_atlas_statistics
from fpcode.workflows.behavior_atlas import extract_measurements
from fpcode.workflows.behavior_extensions import run_behavior_extensions
from fpcode.workflows.pose_atlas import run_pose_atlas
from fpcode.workflows.sequence_atlas import audit_frozen_sequence_order, refresh_chain_kinematics, run_sequence_atlas
from fpcode.workflows.temporal_motifs import run_temporal_motifs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("measurements", "chains", "chain-refresh", "extensions", "pose", "temporal", "statistics"), default="measurements")
    parser.add_argument("--fused-dir", type=Path)
    parser.add_argument("--reuse-native", action="store_true", help="Keep existing native-time fits while refreshing frame motifs")
    parser.add_argument("--reuse-native-templates", action="store_true", help="Infer with saved native-time templates without refitting")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    sessions = pd.read_csv(args.output / "sessions.csv", dtype={"animal_id": str})
    record = RunRecord("karen-behaviour-atlas-" + args.stage, args.output / "provenance" / args.stage,
                       config={"output": str(args.output.resolve()), "stage": args.stage, "seed": 101,
                               "vocabulary": 100, "pause_speed_px_s": 5., "filter_frames": 5, "position_filter_frames": 9,
                               "groups": ["Monomer", "PFF"], "untreated_weeks": [3, 8, 16, 24],
                               "fused_dir": str(args.fused_dir) if args.fused_dir else None})
    record.config["reuse_native"] = args.reuse_native
    record.config["reuse_native_templates"] = args.reuse_native_templates
    record.config["workers"] = args.workers
    if args.reuse_native_templates:
        record.add_input_file(args.output / "chains/native_time_templates.npz")
    record.add_input_file(args.output / "sessions.csv")
    code_root = Path(__file__).resolve().parents[3]
    sources = list((code_root / "src/fpcode").rglob("*.py"))
    sources.append(Path(__file__))
    source_hashes = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record.extra["source_sha256_at_start"] = source_hashes
    if args.stage == "measurements":
        extract_measurements(sessions, args.output / "frame_cache", args.output / "measurements",
                             num_states=100, pause_speed_px_s=5., filter_frames=5)
    elif args.stage == "chains":
        run_sequence_atlas(sessions, args.output / "measurements", args.output / "chains",
                           vocabulary=100, groups=("Monomer", "PFF"))
    elif args.stage == "extensions":
        run_behavior_extensions(sessions, args.output / "measurements", args.output / "chains", vocabulary=100)
    elif args.stage == "chain-refresh":
        refresh_chain_kinematics(sessions, args.output / "measurements", args.output / "chains")
        audit_frozen_sequence_order(sessions, args.output / "measurements", args.output / "chains", vocabulary=100)
    elif args.stage == "pose":
        if args.fused_dir is None:
            parser.error("pose stage requires --fused-dir")
        paths = {sid: args.fused_dir / (sid + "_tandem_21kp.h5") for sid in sessions.session_id}
        landmarks = {"nose": "nose", "tail": "tail_base", "neck": "neck",
                     "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
                     "left_hip": "left_hip", "right_hip": "right_hip", "left_front": "left_front_paw",
                     "right_front": "right_front_paw", "left_hind": "left_hind_paw", "right_hind": "right_hind_paw"}
        run_pose_atlas(sessions, args.output / "frame_cache", paths, args.output / "measurements",
                       landmarks=landmarks, confidence_threshold=.5, vocabulary=100)
    elif args.stage == "temporal":
        annotations = pd.read_csv(args.output / "syllable_annotations.csv")
        run_temporal_motifs(sessions, args.output / "frame_cache", args.output / "measurements", args.output / "chains",
                            vocabulary=100, annotations=annotations, reuse_native=args.reuse_native,
                            reuse_native_templates=args.reuse_native_templates, workers=args.workers)
    else:
        annotations = pd.read_csv(args.output / "syllable_annotations.csv")
        run_atlas_statistics(sessions, args.output / "measurements", args.output / "statistics",
                             annotations=annotations, chains_dir=args.output / "chains",
                             groups=("Monomer", "PFF"), untreated=("3w_PI", "8w_PI", "16w_PI", "24w_PI"),
                             drug_condition="L_dopa", baseline_condition="24w_PI",
                             longitudinal_pairs=(("3w_PI", "8w_PI"), ("8w_PI", "16w_PI"), ("16w_PI", "24w_PI"),
                                                 ("3w_PI", "16w_PI"), ("3w_PI", "24w_PI")))
    record.finish(extra={"selected_sessions": len(sessions), "independent_mice": sessions.animal_id.nunique()})


if __name__ == "__main__":
    main()
