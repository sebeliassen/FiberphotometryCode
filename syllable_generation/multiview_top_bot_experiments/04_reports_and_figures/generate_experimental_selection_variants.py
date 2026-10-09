#!/usr/bin/env python3
"""
generate_experimental_selection_variants.py
=============================================================================
Implementation of Reviewer Recommendation 3:
Generate experimental comparison grids with:
  - Variant 0: Canonical 4-PC Baseline (current installed behavior).
  - Variant A: Supported-Active RMS Reranking (pose similarity during the syllable).
  - Variant B: Duration-Stratified Display (matched execution length + active RMS).

Renders movies, trajectories, and builds a side-by-side comparative HTML gallery.
=============================================================================
"""

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import viz_v2_common as vc

from core.pair_matching import find_matched_pairs

TARGET_SYLLABLES = [1, 67, 42]
COMPARISON_DIR = vc.DEFAULT_VIZ_DIR / "experimental_comparisons"


def build_bout_dict(nb: Dict[str, Any], rank: int, is_anchor: bool, anchor_type: str,
                    vfi_data: np.ndarray, pairs_by_sid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sid, s_idx_str = nb["bout_id"].split(":")
    s_idx = int(s_idx_str)
    dur = int(nb["duration_frames"])
    e_idx = s_idx + dur
    vfi = vfi_data[sid]
    paths = vc.get_session_files(sid, pairs_by_sid)
    if not paths:
        raise ValueError(f"Cannot resolve paths for {sid}")

    return {
        "tile_index": rank,
        "is_blank": False,
        "is_medoid": is_anchor,
        "is_anchor": is_anchor,
        "anchor_type": anchor_type if is_anchor else None,
        "rank": rank,
        "bout_id": nb["bout_id"],
        "segment_name": sid,
        "onset_segment_frame": s_idx,
        "offset_segment_frame": e_idx,
        "duration_frames": dur,
        "source_top_onset": int(vfi[s_idx]),
        "source_top_offset": int(vfi[e_idx]),
        "top_vid": paths["top_vid"],
        "bot_vid": paths["bot_vid"],
        "top_h5": paths["top_h5"],
        "bot_h5": paths["bot_h5"],
        "qc_score": nb.get("qc_score"),
        "qc_active": nb.get("qc_active"),
        "metrics": {
            "pca_distance": nb.get("pca_distance"),
            "shared_active_full_rms_px": nb.get("shared_active_full_rms_px"),
            "shared_active_dorsal_rms_px": nb.get("shared_active_dorsal_rms_px"),
            "shared_active_paw_rms_px": nb.get("shared_active_paw_rms_px"),
            "full_21kp_rms_px": nb.get("full_21kp_rms_px"),
        }
    }


def create_manifest_variants():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    manifest_orig_path = vc.DEFAULT_VIZ_DIR / "unified_syllable_selection_manifest.json"
    diag_path = vc.DEFAULT_VIZ_DIR / "neighbor_diagnostics.json"

    with open(manifest_orig_path) as f:
        manifest_base = json.load(f)
    with open(diag_path) as f:
        diagnostics = json.load(f)

    vfi_data = np.load(vc.VFI_PATH)
    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # Prepare 3 variant manifest structures
    manifests = {
        "canonical": copy.deepcopy(manifest_base),
        "active_reranked": copy.deepcopy(manifest_base),
        "duration_matched": copy.deepcopy(manifest_base),
    }

    metrics_comparison = {}

    for s in TARGET_SYLLABLES:
        s_str = str(s)
        if s_str not in diagnostics:
            continue

        diag_s = diagnostics[s_str]
        nbrs = diag_s.get("neighbors_50", [])
        if not nbrs or len(nbrs) < 12:
            print(f"[Warning] Syllable {s} has fewer than 12 neighbors; skipping variant reranking.")
            continue

        ex_nb = nbrs[0]
        ex_dur = ex_nb["duration_frames"]

        # 1. Variant 0: Canonical 4-PC (first 12 neighbors)
        chosen_canonical = nbrs[:12]

        # 2. Variant A: Active-RMS Reranked
        # Candidate pool: neighbors 1..49 sorted by shared_active_full_rms_px
        valid_candidates = [nb for nb in nbrs[1:] if nb.get("shared_active_full_rms_px") is not None]
        sorted_by_active_rms = sorted(valid_candidates, key=lambda x: x["shared_active_full_rms_px"])
        chosen_active = [ex_nb] + sorted_by_active_rms[:11]

        # 3. Variant B: Duration-Matched + Active RMS
        # Filter duration to 0.6x - 1.8x of exemplar duration
        min_dur = max(5, int(round(ex_dur * 0.6)))
        max_dur = int(round(ex_dur * 1.8))
        dur_matched_pool = [nb for nb in sorted_by_active_rms if min_dur <= nb["duration_frames"] <= max_dur]

        if len(dur_matched_pool) >= 11:
            chosen_duration = [ex_nb] + dur_matched_pool[:11]
            dur_status = "exact_duration_matched"
        else:
            # Fall back to nearest duration distance to exemplar
            remaining_needed = 11 - len(dur_matched_pool)
            other_candidates = sorted(
                [nb for nb in sorted_by_active_rms if nb not in dur_matched_pool],
                key=lambda x: abs(x["duration_frames"] - ex_dur)
            )
            chosen_duration = [ex_nb] + dur_matched_pool + other_candidates[:remaining_needed]
            dur_status = f"partial_duration_fallback_{len(dur_matched_pool)}_of_11"

        # Populate manifests
        for v_name, chosen_list in [("canonical", chosen_canonical),
                                    ("active_reranked", chosen_active),
                                    ("duration_matched", chosen_duration)]:
            bouts_list = []
            for r, nb in enumerate(chosen_list):
                bouts_list.append(build_bout_dict(
                    nb, rank=r, is_anchor=(r == 0),
                    anchor_type="density_exemplar",
                    vfi_data=vfi_data, pairs_by_sid=pairs_by_sid
                ))
            manifests[v_name]["syllables"][s_str]["bouts"] = bouts_list
            manifests[v_name]["syllables"][s_str]["selection_policy"] = v_name

        # Compute summary metrics for comparison
        def extract_stats(chosen_list):
            durs = [nb["duration_frames"] for nb in chosen_list]
            rms_vals = [nb["shared_active_full_rms_px"] for nb in chosen_list[1:] if nb.get("shared_active_full_rms_px") is not None]
            pca_dists = [nb["pca_distance"] for nb in chosen_list[1:]]
            return {
                "exemplar_dur_frames": ex_dur,
                "exemplar_dur_ms": round(ex_dur / 60.0 * 1000, 1),
                "neighbor_dur_min": min(durs[1:]),
                "neighbor_dur_max": max(durs[1:]),
                "neighbor_dur_median": float(np.median(durs[1:])),
                "neighbor_dur_std": round(float(np.std(durs[1:])), 2),
                "mean_active_rms_px": round(float(np.mean(rms_vals)), 2) if rms_vals else None,
                "median_active_rms_px": round(float(np.median(rms_vals)), 2) if rms_vals else None,
                "max_active_rms_px": round(float(np.max(rms_vals)), 2) if rms_vals else None,
                "mean_pca_dist": round(float(np.mean(pca_dists)), 2),
                "tiles_unique_from_canonical": len(set(nb["bout_id"] for nb in chosen_list[1:]) - set(nb["bout_id"] for nb in chosen_canonical[1:]))
            }

        metrics_comparison[s_str] = {
            "syllable": s,
            "exemplar_bout_id": ex_nb["bout_id"],
            "exemplar_duration_frames": ex_dur,
            "duration_matched_status": dur_status,
            "canonical": extract_stats(chosen_canonical),
            "active_reranked": extract_stats(chosen_active),
            "duration_matched": extract_stats(chosen_duration),
        }

    # Save manifests
    manifest_paths = {}
    for v_name, m_data in manifests.items():
        v_dir = COMPARISON_DIR / v_name
        v_dir.mkdir(parents=True, exist_ok=True)
        p = v_dir / f"manifest_{v_name}.json"
        with open(p, "w") as f:
            json.dump(m_data, f, indent=2)
        manifest_paths[v_name] = p
        print(f"✅ Generated {v_name} manifest: {p}")

    metrics_path = COMPARISON_DIR / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_comparison, f, indent=2)
    print(f"✅ Saved quantitative metrics: {metrics_path}")

    return manifest_paths, metrics_comparison


def render_all_variants(manifest_paths: Dict[str, Path]):
    kpms_python = "/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
    movie_renderer = SCRIPT_DIR / "render_all_cohort_static_21kp_v2_seed101_grid_movies.py"
    traj_renderer = SCRIPT_DIR / "render_all_cohort_static_21kp_v2_seed101_trajectories.py"

    syl_args = [str(s) for s in TARGET_SYLLABLES]

    for v_name, m_path in manifest_paths.items():
        v_out_dir = COMPARISON_DIR / v_name
        movies_dir = v_out_dir / "grid_movies"
        traj_dir = v_out_dir / "trajectories"
        movies_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)

        print("\n========================================================")
        print(f"🎬 RENDERING VARIANT: {v_name.upper()}")
        print("========================================================")

        # 1. Grid Movies
        cmd_movies = [
            kpms_python, str(movie_renderer),
            "--manifest", str(m_path),
            "--out_dir", str(movies_dir),
            "--syllables", *syl_args,
            "--overlay_keypoints",
            "--overwrite"
        ]
        print(f"Running grid movies: {' '.join(cmd_movies)}")
        subprocess.run(cmd_movies, check=True)

        # 2. Trajectories
        cmd_traj = [
            kpms_python, str(traj_renderer),
            "--manifest", str(m_path),
            "--out_dir", str(traj_dir),
            "--syllables", *syl_args,
            "--overwrite"
        ]
        print(f"Running trajectories: {' '.join(cmd_traj)}")
        subprocess.run(cmd_traj, check=True)


def build_comparison_html(metrics: Dict[str, Any]):
    html_file = COMPARISON_DIR / "comparison_gallery.html"

    cards_html = []
    for s_str, m in metrics.items():
        s = m["syllable"]
        ex_dur = m["exemplar_duration_frames"]
        c_stats = m["canonical"]
        a_stats = m["active_reranked"]
        d_stats = m["duration_matched"]

        card = f"""
        <div class="syllable-card" id="syllable-{s}">
            <div class="card-header">
                <h2>Syllable {s:02d}</h2>
                <div class="badge">Exemplar Duration: {ex_dur} frames ({ex_dur/60.0*1000:.1f} ms)</div>
                <div class="badge-sub">Exemplar ID: {m["exemplar_bout_id"]}</div>
            </div>

            <div class="variants-grid">
                <!-- Variant 0: Canonical -->
                <div class="variant-col">
                    <div class="col-header canonical-header">
                        <h3>1. Canonical 4-PC Baseline</h3>
                        <div class="tag">4-PC Distance Nearest Neighbors</div>
                    </div>
                    <div class="video-container">
                        <video controls autoplay loop muted playsinline width="100%">
                            <source src="canonical/grid_movies/syllable_{s:02d}_tandem_grid_density.mp4" type="video/mp4">
                        </video>
                    </div>
                    <div class="stats-box">
                        <table>
                            <tr><th>Active Point RMS</th><td><strong>{c_stats['mean_active_rms_px']} px</strong> (med {c_stats['median_active_rms_px']})</td></tr>
                            <tr><th>Duration Range</th><td>{c_stats['neighbor_dur_min']} – {c_stats['neighbor_dur_max']} f (med {c_stats['neighbor_dur_median']} f)</td></tr>
                            <tr><th>Duration Std Dev</th><td>±{c_stats['neighbor_dur_std']} frames</td></tr>
                            <tr><th>Mean 4-PC Dist</th><td>{c_stats['mean_pca_dist']}</td></tr>
                        </table>
                    </div>
                    <div class="traj-preview">
                        <h4>Superimposed Native Trajectory:</h4>
                        <img src="canonical/trajectories/superimposed/syllable_{s:02d}_superimposed.png" width="100%" loading="lazy">
                    </div>
                </div>

                <!-- Variant A: Active Reranked -->
                <div class="variant-col">
                    <div class="col-header active-header">
                        <h3>2. Active-RMS Reranked</h3>
                        <div class="tag">Lowest Pose Error During Syllable</div>
                    </div>
                    <div class="video-container">
                        <video controls autoplay loop muted playsinline width="100%">
                            <source src="active_reranked/grid_movies/syllable_{s:02d}_tandem_grid_density.mp4" type="video/mp4">
                        </video>
                    </div>
                    <div class="stats-box">
                        <table>
                            <tr><th>Active Point RMS</th><td><strong><span class="highlight">{a_stats['mean_active_rms_px']} px</span></strong> (med {a_stats['median_active_rms_px']})</td></tr>
                            <tr><th>Duration Range</th><td>{a_stats['neighbor_dur_min']} – {a_stats['neighbor_dur_max']} f (med {a_stats['neighbor_dur_median']} f)</td></tr>
                            <tr><th>Duration Std Dev</th><td>±{a_stats['neighbor_dur_std']} frames</td></tr>
                            <tr><th>Tiles Changed</th><td>{a_stats['tiles_unique_from_canonical']} / 11 tiles replaced</td></tr>
                        </table>
                    </div>
                    <div class="traj-preview">
                        <h4>Superimposed Native Trajectory:</h4>
                        <img src="active_reranked/trajectories/superimposed/syllable_{s:02d}_superimposed.png" width="100%" loading="lazy">
                    </div>
                </div>

                <!-- Variant B: Duration Matched -->
                <div class="variant-col">
                    <div class="col-header duration-header">
                        <h3>3. Duration-Matched + Active RMS</h3>
                        <div class="tag">Comparable Execution Length</div>
                    </div>
                    <div class="video-container">
                        <video controls autoplay loop muted playsinline width="100%">
                            <source src="duration_matched/grid_movies/syllable_{s:02d}_tandem_grid_density.mp4" type="video/mp4">
                        </video>
                    </div>
                    <div class="stats-box">
                        <table>
                            <tr><th>Active Point RMS</th><td><strong><span class="highlight">{d_stats['mean_active_rms_px']} px</span></strong> (med {d_stats['median_active_rms_px']})</td></tr>
                            <tr><th>Duration Range</th><td>{d_stats['neighbor_dur_min']} – {d_stats['neighbor_dur_max']} f (med {d_stats['neighbor_dur_median']} f)</td></tr>
                            <tr><th>Duration Std Dev</th><td><strong><span class="highlight">±{d_stats['neighbor_dur_std']} frames</span></strong></td></tr>
                            <tr><th>Tiles Changed</th><td>{d_stats['tiles_unique_from_canonical']} / 11 tiles replaced</td></tr>
                        </table>
                    </div>
                    <div class="traj-preview">
                        <h4>Superimposed Native Trajectory:</h4>
                        <img src="duration_matched/trajectories/superimposed/syllable_{s:02d}_superimposed.png" width="100%" loading="lazy">
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KPMS Grid Comparison: Canonical vs Active-Reranked vs Duration-Matched</title>
<style>
    :root {{
        --bg-color: #0e1117;
        --card-bg: #161b22;
        --border-color: #30363d;
        --text-color: #e6edf3;
        --accent-canonical: #58a6ff;
        --accent-active: #3fb950;
        --accent-duration: #d29922;
    }}
    body {{
        margin: 0;
        padding: 24px;
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    header {{
        margin-bottom: 28px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 16px;
    }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    p.lead {{ color: #8b949e; font-size: 14px; max-width: 1000px; line-height: 1.5; }}
    .syllable-card {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 32px;
        padding: 20px;
    }}
    .card-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 12px;
    }}
    .badge {{
        background: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    .badge-sub {{ color: #8b949e; font-size: 12px; font-family: monospace; }}
    .variants-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
    }}
    .variant-col {{
        background: #0d1117;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px;
        display: flex;
        flex-direction: column;
    }}
    .col-header h3 {{ margin: 0 0 4px 0; font-size: 16px; }}
    .canonical-header h3 {{ color: var(--accent-canonical); }}
    .active-header h3 {{ color: var(--accent-active); }}
    .duration-header h3 {{ color: var(--accent-duration); }}
    .tag {{ font-size: 11px; color: #8b949e; margin-bottom: 10px; }}
    .video-container {{
        background: #000;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 12px;
    }}
    .stats-box {{
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 12px;
        font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; color: #8b949e; font-weight: normal; padding: 3px 0; }}
    td {{ text-align: right; padding: 3px 0; }}
    .highlight {{ color: #3fb950; font-weight: bold; }}
    .traj-preview h4 {{ font-size: 12px; color: #8b949e; margin: 8px 0 4px 0; }}
    .traj-preview img {{ border-radius: 4px; border: 1px solid #21262d; }}
</style>
</head>
<body>
<header>
    <h1>🔬 KPMS Syllable Grid Reranking Comparison Gallery</h1>
    <p.lead>
        Direct evaluation of the reviewer hypothesis: <em>"Rendering the same selected bouts again will not make them more similar... pairing/rendering fixes address whether you see the correct execution; selection changes address whether the displayed executions resemble one another."</em><br>
        <strong>Variant 1 (Canonical 4-PC):</strong> Nearest 11 in 4-PC space over 40-frame scoring window.<br>
        <strong>Variant 2 (Active-RMS Reranked):</strong> Reranked by point RMS strictly during the active execution of the syllable.<br>
        <strong>Variant 3 (Duration-Matched + Active RMS):</strong> Constrained to comparable execution durations (0.6x–1.8x of exemplar) and sorted by active RMS.
    </p>
</header>

{"".join(cards_html)}

</body>
</html>
"""
    with open(html_file, "w") as f:
        f.write(full_html)
    print(f"\n🌟 Interactive comparison HTML gallery written to: {html_file}")


def main():
    print("=" * 80)
    print("🚀 GENERATING EXPERIMENTAL SELECTION VARIANTS & COMPARISONS")
    print(f"Target Syllables: {TARGET_SYLLABLES}")
    print(f"Output Directory: {COMPARISON_DIR}")
    print("=" * 80)

    # 1. Create manifests
    manifest_paths, metrics = create_manifest_variants()

    # 2. Render all variants
    render_all_variants(manifest_paths)

    # 3. Build HTML gallery
    build_comparison_html(metrics)

    print("\n✅ All experimental variants and comparisons completed successfully!")


if __name__ == "__main__":
    main()
