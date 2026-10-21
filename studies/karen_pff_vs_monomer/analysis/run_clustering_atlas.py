"""Compare Karen's explicit annotation levels on the verified original frame cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.run_record import RunRecord
from fpcode.statistics.behavior_comparisons import compare_feature_atlas
from fpcode.statistics.followup_planning import exploratory_effect_stability
from fpcode.workflows.anchored_sequences import measure_anchored_sequences
from fpcode.workflows.atlas_statistics import build_feature_table
from fpcode.workflows.behavior_atlas import extract_measurements
from fpcode.workflows.behavior_extensions import run_behavior_extensions
from fpcode.workflows.resolution_atlas import raw_merging_audit, resolution_composition_features
from fpcode.workflows.sequence_atlas import audit_frozen_sequence_order, run_sequence_atlas
from studies.karen_pff_vs_monomer.analysis.karen_hierarchy import build_hierarchy

LEVELS = ("L1", "L2", "L3", "L4", "L3_unresolved_separate", "L4_unresolved_separate")
UNTREATED = ("3w_PI", "8w_PI", "16w_PI", "24w_PI")
PAIRS = (("3w_PI", "8w_PI"), ("8w_PI", "16w_PI"), ("16w_PI", "24w_PI"), ("3w_PI", "16w_PI"), ("3w_PI", "24w_PI"))
CORE_FAMILIES = ("session", "syllable_usage", "syllable_execution", "syllable_transitions", "chain_usage", "chain_execution",
                 "chain_grouping", "chain_transitions", "chain_progression", "syllable_variants", "behavioural_history", "syllable_chain_context")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "measurements", "chains", "anchors", "statistics"))
    parser.add_argument("--level", choices=LEVELS, default="L3")
    args = parser.parse_args(argv)
    sessions = pd.read_csv(args.atlas / "sessions.csv", dtype={"animal_id": str})
    annotations = pd.read_csv(args.atlas / "syllable_annotations.csv")
    maps, tables, membership = build_hierarchy(annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    level_dir = args.output / args.level
    record = RunRecord("karen-resolution-" + args.stage, level_dir / "provenance" / args.stage,
                       config={"atlas": str(args.atlas.resolve()), "output": str(args.output.resolve()), "stage": args.stage,
                               "level": args.level, "map": maps[args.level].tolist(), "seed": 101,
                               "filter_frames": 5, "position_filter_frames": 9, "groups": ["Monomer", "PFF"],
                               "untreated": UNTREATED, "drug_condition": "L_dopa", "discovery_split": "same animals as raw atlas"})
    code_root = Path(__file__).resolve().parents[3]
    record.extra["source_sha256_at_start"] = {
        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*list((code_root / "src/fpcode").rglob("*.py")), Path(__file__), Path(__file__).with_name("karen_hierarchy.py")]}
    record.extra["thread_environment"] = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    record.add_input_file(args.atlas / "sessions.csv")
    record.add_input_file(args.atlas / "syllable_annotations.csv")
    split = json.loads((args.atlas / "chains/animal_splits.json").read_text())
    if args.stage == "prepare":
        sessions.to_csv(args.output / "sessions.csv", index=False)
        membership.to_csv(args.output / "cluster_membership.csv", index=False)
        raw = pd.read_csv(args.atlas / "measurements/syllables.csv", dtype={"animal_id": str})
        audit = raw_merging_audit(raw, membership, training_animals=split["train"], untreated=UNTREATED)
        audit.to_csv(args.output / "merging_review.csv", index=False)
        for level in LEVELS:
            target = args.output / level
            target.mkdir(exist_ok=True)
            tables[level].to_csv(target / "cluster_annotations.csv", index=False)
            (target / "mapping.json").write_text(json.dumps(maps[level].tolist()))
        print({key: int(value.max() + 1) for key, value in maps.items()}, flush=True)
    elif args.stage == "measurements":
        if args.level == "L1":
            raise ValueError("L1 reuses original verified measurements; no extraction required")
        extract_measurements(sessions, args.atlas / "frame_cache", level_dir / "measurements",
                             num_states=int(maps[args.level].max() + 1), pause_speed_px_s=5., label_map=maps[args.level])
    elif args.stage == "chains":
        if args.level not in ("L2", "L3", "L4"):
            raise ValueError("fresh chain discovery is scoped to primary L2/L3/L4")
        run_sequence_atlas(sessions, level_dir / "measurements", level_dir / "chains",
                           vocabulary=int(maps[args.level].max() + 1), groups=("Monomer", "PFF"))
        actual = json.loads((level_dir / "chains/animal_splits.json").read_text())
        if actual != split:
            raise ValueError("resolution discovery/test animals differ from the raw atlas")
        audit_frozen_sequence_order(sessions, level_dir / "measurements", level_dir / "chains",
                                    vocabulary=int(maps[args.level].max() + 1))
        run_behavior_extensions(sessions, level_dir / "measurements", level_dir / "chains",
                                vocabulary=int(maps[args.level].max() + 1))
    elif args.stage == "anchors":
        measured = args.atlas / "measurements" if args.level == "L1" else level_dir / "measurements"
        catalog = pd.read_csv(args.atlas / "chains/chain_catalog.csv")
        measure_anchored_sequences(sessions, measured, level_dir / "anchors", catalog[catalog.method == "exact"], maps[args.level])
    else:
        original = args.level == "L1"
        measured = args.atlas / "measurements" if original else level_dir / "measurements"
        chain_dir = args.atlas / "chains" if original else level_dir / "chains"
        chain_dir = chain_dir if chain_dir.exists() else None
        feature = (pd.read_csv(args.atlas / "statistics/session_features_long.csv", dtype={"animal_id": str}) if original
                   else build_feature_table(sessions, measured, annotations=tables[args.level], chains_dir=chain_dir))
        feature = feature[feature.family.isin(CORE_FAMILIES)]
        if not original:
            raw = pd.read_csv(args.atlas / "measurements/syllables.csv", dtype={"animal_id": str})
            mixture = resolution_composition_features(raw, maps[args.level], level_dir / "measurements",
                                                       training_animals=split["train"], untreated=UNTREATED)
            feature = pd.concat([feature, mixture], ignore_index=True)
        anchors = level_dir / "anchors/anchor_features_long.csv"
        if anchors.exists():
            feature = pd.concat([feature, pd.read_csv(anchors, dtype={"animal_id": str})], ignore_index=True)
        feature.value = feature.value.replace([np.inf, -np.inf], np.nan)
        out = level_dir / "statistics"
        out.mkdir(exist_ok=True)
        feature.to_csv(out / "session_features_long.csv", index=False)
        print(f"{args.level}: comparing {feature[['family', 'feature']].drop_duplicates().shape[0]} features", flush=True)
        result = compare_feature_atlas(feature, groups=("Monomer", "PFF"), untreated=UNTREATED,
                                       drug_condition="L_dopa", baseline_condition="24w_PI", longitudinal_pairs=PAIRS)
        for name, table in result.items():
            table.to_csv(out / (name + ".csv"), index=False)
        candidates = exploratory_effect_stability(result["animal_wave_measurements"], result["group_contrasts"],
                                                   groups=("Monomer", "PFF"), untreated=UNTREATED, consistency_waves=UNTREATED[:3])
        candidates.to_csv(out / "effect_stability.csv", index=False)
    record.finish(extra={"sessions": len(sessions), "mice": sessions.animal_id.nunique()})


if __name__ == "__main__":
    main()
