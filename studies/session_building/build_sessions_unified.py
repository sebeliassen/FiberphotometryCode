import argparse
import importlib
import pickle
from pathlib import Path
from typing import Optional

# Physical owners: IFdN and monoamine baseline intentionally have distinct policy tables.
_PROJECT_OWNERS = {
    "jonathan_full": ("studies.oft_baseline.ifdn_policy.enrich", None, "h5"),
    "jonathan_openfield_example": ("studies.oft_baseline.ifdn_policy.enrich", None, "h5"),
    "jonathan_oft_baseline": ("studies.oft_baseline.enrich", "studies.oft_baseline.config", "hybrid"),
    "jonathan_juvenile_intruder": ("studies.juvenile_intruder.enrich", "studies.juvenile_intruder.config", "hybrid"),
    "karen_project": ("studies.karen_pff_vs_monomer.analysis.enrich", "studies.karen_pff_vs_monomer.session_policy.config", "hybrid"),
}


def _resolve_project(project):
    try:
        enrich_name, config_name, default_mode = _PROJECT_OWNERS[project]
    except KeyError:
        raise ValueError(
            f"No audited policy owner for project {project!r}. "
            "AMPH, flipped and external project policies require separate source review."
        ) from None

    from studies.session_building.jonathan.config import JonathanConfig

    enrich = importlib.import_module(enrich_name)
    project_config = importlib.import_module(config_name) if config_name else None
    cfg = project_config.JCFG if project_config is not None else JonathanConfig()
    policy_module = project_config if project_config is not None else enrich
    overrides_dir = Path(policy_module.__file__).resolve().parent / "overrides"
    return enrich, project_config, cfg, overrides_dir, default_mode

def build_sessions(
    *,
    data_root: Path | str,
    h5_dir: Path | str,
    video_dir: Path | str,
    csv_root: Path | str,
    output_pickle: Path | str,
    project: str,
    overwrite: bool = True,
    force_dirty: bool = True,
    discovery_mode: Optional[str] = None
):
    """Unified session builder function for open field experiments."""
    data_root = Path(data_root).resolve()
    h5_dir = Path(h5_dir).resolve()
    video_dir = Path(video_dir).resolve()
    csv_root = Path(csv_root).resolve()
    out_path = Path(output_pickle).resolve()

    # Fast path: load cached sessions if overwrite is False
    if out_path.exists() and not overwrite:
        print(f"[sessions-unified] Loading cached sessions from: {out_path}")
        with out_path.open("rb") as f:
            return pickle.load(f)

    # 1. Select explicit source owners without changing module or path state.
    enrich, project_config, cfg, overrides_dir, default_mode = _resolve_project(project)
    print(f"[sessions-unified] Using project overrides from: {overrides_dir}")

    from studies.session_building.jonathan import steps as jonathan_steps
    from studies.session_building.jonathan.build import build
    from studies.session_building.jonathan.context import BuildContext

    # 2. Preserve the historical default and forced discovery choices.
    if discovery_mode is None:
        discovery_mode = default_mode
    if discovery_mode not in ("h5", "hybrid"):
        raise ValueError(f"Unknown discovery mode: {discovery_mode!r}")
    print(f"[sessions-unified] Discovery mode: {discovery_mode}")

    if discovery_mode == "h5":
        discovery_step = jonathan_steps.tracking.discover_sessions
    elif project == "karen_project":
        from studies.karen_pff_vs_monomer.analysis.tracking import discover_sessions_hybrid

        discovery_step = discover_sessions_hybrid
    else:
        # This fallback historically owns baseline discovery policy, including
        # when hybrid discovery is explicitly forced for another project.
        from studies.monoamine_interactions.steps.tracking import discover_sessions_hybrid

        discovery_step = discover_sessions_hybrid

    # 3. Setup Enrichment wrappers
    def _custom_enrich_pre(ctx, sessions):
        return enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=force_dirty)

    def _custom_enrich_pre_wrapper(ctx, sessions):
        return _custom_enrich_pre(ctx, sessions)

    def _custom_enrich_post(ctx, sessions):
        return enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=force_dirty)

    def _custom_enrich_post_wrapper(ctx, sessions):
        return _custom_enrich_post(ctx, sessions)

    # 4. Check for Flipping
    has_flipping = (project_config is not None and hasattr(project_config, "OPTIMAL_FLIP_IDS")) or project == "jonathan_flipped"

    if has_flipping:
        flip_ids = getattr(project_config, "OPTIMAL_FLIP_IDS", [728, 740, 1767, 1773, 1664, 1666, 1779, 1781])
        print(f"[sessions-unified] Flipping enabled. Optimal flip IDs: {flip_ids}")
        def _flag_flipped_sessions(ctx, sessions):
            flip_set = set(str(x) for x in flip_ids)
            count = 0
            for s in sessions:
                aid = str(s.metadata.get("animal_id", "")).strip()
                if aid in flip_set:
                    s.metadata["is_flipped"] = True
                    count += 1
                else:
                    s.metadata["is_flipped"] = False
            print(f"[sessions-unified] Flagged {count}/{len(sessions)} sessions as flipped.")
            return sessions

    # 5. Assemble Custom Steps
    CUSTOM_STEPS = [
        discovery_step,
        jonathan_steps.video.attach_video_refs,
        jonathan_steps.fiberphotometry.attach_photometry_refs,
        _custom_enrich_pre_wrapper,
        jonathan_steps.fiberphotometry.realize_photometry_columns,
        _custom_enrich_post_wrapper,
    ]

    if has_flipping:
        CUSTOM_STEPS.append(_flag_flipped_sessions)

    CUSTOM_STEPS.extend([
        jonathan_steps.fiberphotometry.split_photometry,
        jonathan_steps.fiberphotometry.compute_dff_zscore,
    ])

    # 6. Execute Build
    ctx = BuildContext(
        root=data_root,
        h5_dir=h5_dir,
        video_dir=video_dir,
        csv_root=csv_root,
        project=project,
        cfg=cfg,
    )

    print(f"[sessions-unified] Building sessions for {project}...")
    sessions = build(ctx, steps=CUSTOM_STEPS)

    # 7. Save to pickle
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[sessions-unified] Saved {len(sessions)} sessions to: {out_path}")

    return sessions

def main():
    parser = argparse.ArgumentParser(description="Unified Sessions Builder")
    parser.add_argument("--project", required=True, help="Project name (e.g. jonathan_oft_baseline)")
    parser.add_argument("--data-dir", required=True, help="Data root folder containing video_files, h5_files, csvs")
    parser.add_argument("--output", required=True, help="Path where the output pickle file should be saved")
    parser.add_argument("--dirty", action="store_true", help="Skip exclusions and save as dirty.pkl")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output pickle")
    parser.add_argument("--csv-root", default=None, help="Optional path to photometry CSV files root folder (if separate from data-dir)")
    parser.add_argument("--discovery-mode", choices=["h5", "hybrid"], default=None, help="Force specific discovery mode")
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    h5_dir = data_root / "h5_files"
    video_dir = data_root / "video_files"

    if args.csv_root:
        csv_root = Path(args.csv_root).resolve()
    else:
        # Resolve CSV candidates
        csv_candidates = [data_root / "csvs", data_root / "Jonathan_IFxDN_csvs", data_root]
        csv_root = next((p for p in csv_candidates if p.exists()), data_root)

    build_sessions(
        data_root=data_root,
        h5_dir=h5_dir,
        video_dir=video_dir,
        csv_root=csv_root,
        output_pickle=args.output,
        project=args.project,
        overwrite=args.overwrite,
        force_dirty=args.dirty,
        discovery_mode=args.discovery_mode
    )

if __name__ == "__main__":
    main()
