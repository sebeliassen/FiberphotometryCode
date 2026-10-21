import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from fpcode.common.paths import get_repo_root

# Ensure project root is in sys.path
REPO_ROOT = str(get_repo_root())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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

    # 1. Discover Project Module
    sg_proj_dir = (get_repo_root() / 'studies/oft_baseline/ifdn_policy') / project
    mi_proj_dir = Path(REPO_ROOT) / "studies" / "monoamine_interactions" / "projects" / project

    proj_dir = None
    if sg_proj_dir.exists():
        proj_dir = sg_proj_dir
    elif mi_proj_dir.exists():
        proj_dir = mi_proj_dir
    else:
        # Check fallback (e.g. if the project is jonathan_full but config is under jonathan_openfield_example)
        if project == "jonathan_full":
            proj_dir = (get_repo_root() / 'studies/oft_baseline/ifdn_policy') / "jonathan_openfield_example"
        else:
            raise ValueError(f"Project directory for '{project}' not found.")

    print(f"[sessions-unified] Using project overrides from: {proj_dir}")

    # Dynamic imports
    if str(proj_dir) not in sys.path:
        sys.path.insert(0, str(proj_dir))

    # Reload modules to make sure we don't have cached configs from other runs
    if "enrich" in sys.modules:
        importlib.reload(sys.modules["enrich"])

    if "config" in sys.modules:
        try:
            importlib.reload(sys.modules["config"])
        except Exception:
            # If config module is cached but cannot be reloaded, delete it
            sys.modules.pop("config", None)

    import enrich

    project_config = None
    try:
        import config as project_config
    except (ImportError, ModuleNotFoundError):
        print(f"[sessions-unified] No custom config.py found in {proj_dir}. Using defaults.")

    # Clean up sys.path
    if str(proj_dir) in sys.path:
        sys.path.remove(str(proj_dir))

    # Overwrite the global JCFG in syllable_generation library
    from studies.session_building.jonathan import config as lib_config
    from studies.session_building.jonathan.config import JonathanConfig

    if project_config is not None and hasattr(project_config, "JCFG"):
        print("[sessions-unified] Overwriting JCFG with project config.")
        lib_config.JCFG = project_config.JCFG
    else:
        print("[sessions-unified] Resetting JCFG to default JonathanConfig.")
        lib_config.JCFG = JonathanConfig()

    from studies.session_building.jonathan import steps as jonathan_steps
    from studies.session_building.jonathan.build import build
    from studies.session_building.jonathan.context import BuildContext

    # 2. Determine Discovery Mode
    # Default to H5-based for syllable_generation example, and hybrid video/H5 for others
    if discovery_mode is None:
        if project in ["jonathan_full", "jonathan_openfield_example"]:
            discovery_mode = "h5"
        else:
            discovery_mode = "hybrid"

    print(f"[sessions-unified] Discovery mode: {discovery_mode}")

    if discovery_mode == "h5":
        discovery_step = jonathan_steps.tracking.discover_sessions
    else:
        # Check if project has custom tracking steps override
        custom_tracking = proj_dir / "tracking.py"
        if custom_tracking.exists():
            print(f"[sessions-unified] Loading custom tracking step from: {custom_tracking}")
            if str(proj_dir) not in sys.path:
                sys.path.insert(0, str(proj_dir))

            if "tracking" in sys.modules:
                importlib.reload(sys.modules["tracking"])
            import tracking

            discovery_step = tracking.discover_sessions_hybrid

            if str(proj_dir) in sys.path:
                sys.path.remove(str(proj_dir))
        else:
            # Import hybrid discovery step from monoamine_interactions/steps/tracking.py
            sys.path.insert(0, os.path.join(REPO_ROOT, "studies", "monoamine_interactions"))
            from steps.tracking import discover_sessions_hybrid
            sys.path.remove(os.path.join(REPO_ROOT, "studies", "monoamine_interactions"))
            discovery_step = discover_sessions_hybrid

    # 3. Setup Enrichment wrappers
    def _custom_enrich_pre(ctx, sessions):
        overrides_dir = proj_dir / "overrides"
        return enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=force_dirty)

    def _custom_enrich_pre_wrapper(ctx, sessions):
        return _custom_enrich_pre(ctx, sessions)

    def _custom_enrich_post(ctx, sessions):
        overrides_dir = proj_dir / "overrides"
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
