#!/usr/bin/env python3
"""
build_cohort_static_21kp_v2_seed101_labeling_gallery.py
=========================================================
Builds an interactive single-page HTML labeling dashboard and markdown catalog
for Cohort Static 21kp V2 Seed 101 syllables, linking:
- Tandem side-by-side animated GIF / vector PDF
- Superimposed animated GIF / vector PDF
- 3-Tier density / compact-neighborhood tandem 3x4 grid movie
- Real-time kinematic metrics (usage %, bouts, dwell, velocity, yaw rate)
- Form fields for the ethologist to label syllable taxonomy, notes, and confidence
- Client-side JSON and CSV label exporter

Environment: Python 3
"""

import os
import sys
import json
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "visualizations_cohort_static_21kp_v2_seed101"
TRAJ_SIDE = VIZ_DIR / "trajectories" / "side_by_side"
TRAJ_SUPER = VIZ_DIR / "trajectories" / "superimposed"
GRID_DENSITY = VIZ_DIR / "grid_movies_density"

COHORT_DIR = SCRIPT_DIR.parent / "cohort_static_21kp_v2"
MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
RESULTS_PATH = COHORT_DIR / MODEL_NAME / "results.h5"
MANIFEST_PATH = VIZ_DIR / "unified_syllable_selection_manifest.json"

HTML_OUT = VIZ_DIR / "syllable_labeling_gallery_cohort_static_21kp_v2_seed101.html"
MD_OUT = VIZ_DIR / "SEED101_STATIC21KP_V2_SYLLABLE_CATALOG_FOR_LABELING.md"


def main():
    print("=" * 80)
    print("📋 BUILDING COHORT STATIC 21kp V2 SEED 101 LABELING GALLERY & CATALOG")
    print(f"Output HTML:     {HTML_OUT}")
    print(f"Output Markdown: {MD_OUT}")
    print("=" * 80)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)

    # 2. Compute cohort-wide metrics from results.h5
    print("[1/3] Computing comprehensive syllable statistics from results.h5...", flush=True)
    syll_counts = {}
    bout_counts = {}
    durations = {}
    vel_means = {}
    ang_means = {}

    total_frames = 0
    if RESULTS_PATH.exists():
        with h5py.File(RESULTS_PATH, "r") as f:
            for k in f.keys():
                s = f[k]["syllable"][:]
                c = f[k]["centroid"][:]
                h = f[k]["heading"][:]
                total_frames += len(s)

                vel = np.linalg.norm(np.diff(c, axis=0), axis=1) * 60.0
                dhead = np.diff(h)
                dhead = np.arctan2(np.sin(dhead), np.cos(dhead))
                ang_vel = np.abs(dhead) * 60.0 * (180.0 / np.pi)

                u, counts = np.unique(s, return_counts=True)
                for val, cnt in zip(u, counts):
                    val = int(val)
                    syll_counts[val] = syll_counts.get(val, 0) + cnt

                diffs = np.diff(s)
                bounds = np.where(diffs != 0)[0] + 1
                starts = np.r_[0, bounds]
                ends = np.r_[bounds, len(s)]
                for start, end in zip(starts, ends):
                    dur = end - start
                    val = int(s[start])
                    if dur >= 5:
                        bout_counts[val] = bout_counts.get(val, 0) + 1
                        if val not in durations:
                            durations[val] = []
                            vel_means[val] = []
                            ang_means[val] = []
                        durations[val].append(dur)
                        if end > start + 1 and end - 1 < len(vel):
                            vel_means[val].append(float(np.mean(vel[start:end-1])))
                            ang_means[val].append(float(np.mean(ang_vel[start:end-1])))

    # Discover all syllables from manifest or results
    discovered_sylls = set()
    if "syllables" in manifest:
        for s_str in manifest["syllables"].keys():
            discovered_sylls.add(int(s_str))
    for s in syll_counts.keys():
        discovered_sylls.add(s)

    syll_ids = sorted(list(discovered_sylls))
    syll_ids.sort(key=lambda s: syll_counts.get(s, 0), reverse=True)

    print(f"[2/3] Generating catalog items for {len(syll_ids)} syllables...", flush=True)

    cards_html = []
    catalog_md_rows = []

    for rank, s in enumerate(syll_ids, 1):
        cnt = syll_counts.get(s, 0)
        pct = (cnt / total_frames * 100.0) if total_frames > 0 else 0.0
        nbouts = bout_counts.get(s, 0)
        med_dur_ms = (np.median(durations[s]) / 60.0 * 1000.0) if s in durations and durations[s] else 0.0
        mean_v = np.mean(vel_means[s]) if s in vel_means and vel_means[s] else 0.0
        mean_w = np.mean(ang_means[s]) if s in ang_means and ang_means[s] else 0.0

        manifest_entry = manifest.get("syllables", {}).get(str(s), {})
        tier = manifest_entry.get("tier", "tier_1_density")

        tier_badge_color = "#10b981" if "tier_1" in tier else ("#f59e0b" if "tier_2" in tier else "#8b5cf6")
        tier_label = "Built-in Density (N>50)" if "tier_1" in tier else ("Compact Neighborhood (G<N<=50)" if "tier_2" in tier else "Full Bout Enumeration (N<=G)")

        # Media paths
        vid_density = GRID_DENSITY / f"syllable_{s:02d}_tandem_grid_density.mp4"
        traj_side_gif = TRAJ_SIDE / f"syllable_{s:02d}_trajectory_side_by_side.gif"
        traj_side_png = TRAJ_SIDE / f"syllable_{s:02d}_trajectory_side_by_side.png"
        traj_side_pdf = TRAJ_SIDE / f"syllable_{s:02d}_trajectory_side_by_side.pdf"
        traj_super_gif = TRAJ_SUPER / f"syllable_{s:02d}_trajectory_superimposed.gif"
        traj_super_png = TRAJ_SUPER / f"syllable_{s:02d}_trajectory_superimposed.png"
        traj_super_pdf = TRAJ_SUPER / f"syllable_{s:02d}_trajectory_superimposed.pdf"

        has_vid_density = vid_density.exists()
        has_traj_side = traj_side_gif.exists()
        has_traj_super = traj_super_gif.exists()

        vid_density_rel = f"grid_movies_density/{vid_density.name}" if has_vid_density else ""
        traj_side_gif_rel = f"trajectories/side_by_side/{traj_side_gif.name}" if has_traj_side else ""
        traj_side_png_rel = f"trajectories/side_by_side/{traj_side_png.name}" if traj_side_png.exists() else ""
        traj_side_pdf_rel = f"trajectories/side_by_side/{traj_side_pdf.name}" if traj_side_pdf.exists() else ""
        traj_super_gif_rel = f"trajectories/superimposed/{traj_super_gif.name}" if has_traj_super else ""
        traj_super_png_rel = f"trajectories/superimposed/{traj_super_png.name}" if traj_super_png.exists() else ""
        traj_super_pdf_rel = f"trajectories/superimposed/{traj_super_pdf.name}" if traj_super_pdf.exists() else ""

        # Markdown catalog row
        catalog_md_rows.append(
            f"| `{s:02d}` | #{rank} | **{pct:.2f}%** | {nbouts:,} | {med_dur_ms:.0f} ms | {mean_v:.1f} | {mean_w:.1f} | `{tier}` | [Movie]({vid_density_rel}) | [SideBySide GIF]({traj_side_gif_rel}) |"
        )

        # HTML Card
        card = f"""
        <div class="card" id="card-syllable-{s}" data-syllable="{s}" data-usage="{pct:.4f}" data-bouts="{nbouts}" data-tier="{tier}">
          <div class="card-header">
            <div>
              <span class="card-title">Syllable {s:02d}</span>
              <span class="rank-tag">Rank #{rank}</span>
              <span class="tier-tag" style="background: {tier_badge_color}22; border-color: {tier_badge_color}; color: {tier_badge_color};">{tier_label}</span>
            </div>
            <div class="stats-row">
              <span>Usage: <b>{pct:.2f}%</b></span>
              <span>Bouts: <b>{nbouts:,}</b></span>
              <span>Dwell: <b>{med_dur_ms:.0f} ms</b></span>
              <span>Speed: <b>{mean_v:.1f} px/s</b></span>
            </div>
          </div>
          
          <div class="media-wrapper">
            <div class="video-container">
              <div class="section-label">Dual-View Prototype Movie (TopCam + BotCam)</div>
              {f'<video class="grid-video" controls loop muted preload="metadata" src="{vid_density_rel}"></video>' if has_vid_density else '<div class="placeholder">Grid Movie Rendering In Progress</div>'}
            </div>
            <div class="traj-container">
              <div class="section-label">Observed 21kp Trajectory Prototype</div>
              <div class="traj-view-toggle">
                <button class="toggle-btn active" onclick="switchTraj({s}, 'side')">Side-by-Side</button>
                <button class="toggle-btn" onclick="switchTraj({s}, 'super')">Superimposed</button>
              </div>
              <div class="traj-image-wrapper">
                <img id="traj-img-{s}" class="traj-gif" src="{traj_side_gif_rel if has_traj_side else (traj_super_gif_rel if has_traj_super else '')}" alt="Trajectory GIF" />
              </div>
              <div class="vector-links">
                {f'<a href="{traj_side_pdf_rel}" target="_blank" class="pdf-link">Vector PDF (Side)</a>' if traj_side_pdf_rel else ''}
                {f'<a href="{traj_side_png_rel}" target="_blank" class="pdf-link" style="background:#e0e7ff;color:#3730a3;">Hi-Res PNG (Side)</a>' if traj_side_png_rel else ''}
                {f'<a href="{traj_super_pdf_rel}" target="_blank" class="pdf-link">Vector PDF (Super)</a>' if traj_super_pdf_rel else ''}
                {f'<a href="{traj_super_png_rel}" target="_blank" class="pdf-link" style="background:#fef3c7;color:#92400e;">Hi-Res PNG (Super)</a>' if traj_super_png_rel else ''}
              </div>
            </div>
          </div>

          <div class="annotation-box">
            <div class="anno-row">
              <input type="text" class="anno-input name-input" id="name-{s}" placeholder="Proposed Name (e.g., Left Turn, Rearing, Walk)" oninput="saveLabel({s})" />
              <select class="anno-select" id="cat-{s}" onchange="saveLabel({s})">
                <option value="">-- Behavior Class --</option>
                <option value="Locomotion">Locomotion</option>
                <option value="Turning / Turning Gait">Turning</option>
                <option value="Rearing / Unsupported">Rearing</option>
                <option value="Grooming">Grooming</option>
                <option value="Pause / Quiescence">Pause / Quiescence</option>
                <option value="Sniffing / Exploration">Sniffing</option>
                <option value="Other">Other</option>
              </select>
              <select class="anno-select" id="conf-{s}" onchange="saveLabel({s})">
                <option value="">Confidence</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div>
            <textarea class="anno-textarea" id="notes-{s}" placeholder="Ethological notes & kinematics rationale..." oninput="saveLabel({s})"></textarea>
          </div>
        </div>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cohort Static 21kp V2 Ethologist Labeling Gallery (Seed 101)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #0f131f;
      --card-bg: rgba(23, 29, 46, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --primary-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: radial-gradient(circle at 50% 0%, #1e2640 0%, var(--bg-dark) 75%);
      color: var(--text-main);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      padding: 24px;
    }}
    header {{
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }}
    header h1 {{
      font-size: 26px;
      font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .meta-badges {{
      display: flex;
      gap: 12px;
      margin-top: 8px;
    }}
    .badge {{
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 13px;
      color: var(--accent-cyan);
      font-family: 'JetBrains Mono', monospace;
    }}
    .controls-bar {{
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .search-input {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 14px;
      min-width: 260px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{ border-color: var(--primary-blue); }}
    .btn {{
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 500;
      transition: transform 0.15s, opacity 0.15s;
      font-family: 'Outfit', sans-serif;
    }}
    .btn:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    .grid-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(620px, 1fr));
      gap: 24px;
    }}
    .card {{
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      display: flex;
      flex-direction: column;
    }}
    .card:hover {{
      transform: translateY(-3px);
      border-color: rgba(59, 130, 246, 0.4);
      box-shadow: 0 8px 30px rgba(37, 99, 235, 0.15);
    }}
    .card-header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
    }}
    .card-title {{
      font-size: 19px;
      font-weight: 600;
      color: #fff;
    }}
    .rank-tag {{
      background: rgba(59, 130, 246, 0.2);
      border: 1px solid var(--primary-blue);
      color: #93c5fd;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      margin-left: 8px;
    }}
    .tier-tag {{
      border: 1px solid;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      margin-left: 6px;
    }}
    .stats-row {{
      display: flex;
      gap: 12px;
      font-size: 12px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}
    .media-wrapper {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 16px;
      background: rgba(0, 0, 0, 0.25);
    }}
    .section-label {{
      font-size: 11px;
      color: var(--accent-cyan);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .grid-video {{
      width: 100%;
      height: 220px;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }}
    .traj-container {{
      display: flex;
      flex-direction: column;
    }}
    .traj-view-toggle {{
      display: flex;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .toggle-btn {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      cursor: pointer;
    }}
    .toggle-btn.active {{
      background: var(--primary-blue);
      color: #fff;
      border-color: var(--primary-blue);
    }}
    .traj-image-wrapper {{
      width: 100%;
      height: 180px;
      background: #fff;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border: 1px solid var(--border-color);
    }}
    .traj-gif {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }}
    .vector-links {{
      display: flex;
      gap: 10px;
      margin-top: 8px;
    }}
    .pdf-link {{
      color: var(--accent-cyan);
      font-size: 11px;
      text-decoration: none;
    }}
    .pdf-link:hover {{ text-decoration: underline; }}
    .placeholder {{
      height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #666;
      font-size: 12px;
      background: #000;
      border-radius: 8px;
    }}
    .annotation-box {{
      padding: 16px;
      border-top: 1px solid var(--border-color);
      background: rgba(0, 0, 0, 0.15);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .anno-row {{
      display: flex;
      gap: 10px;
    }}
    .anno-input, .anno-select {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
    }}
    .name-input {{ flex: 2; }}
    .anno-select {{ flex: 1; }}
    .anno-textarea {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      resize: vertical;
      min-height: 48px;
      font-family: inherit;
    }}
  </style>
</head>
<body>

  <header>
    <div>
      <h1>Cohort Static 21kp V2 Ethologist Labeling Gallery</h1>
      <div class="meta-badges">
        <span class="badge">Model: 50 AR + 300 SLDS (Seed 101)</span>
        <span class="badge">Epoch-Corrected Static 21kp</span>
        <span class="badge">ARI Convergence: 0.9330</span>
        <span class="badge">Total Frames: {total_frames:,}</span>
      </div>
    </div>
    <div>
      <button class="btn" onclick="exportLabelsJSON()">Export Labels (JSON)</button>
      <button class="btn" style="margin-left: 8px; background: linear-gradient(135deg, #059669, #047857);" onclick="exportLabelsCSV()">Export CSV</button>
    </div>
  </header>

  <div class="controls-bar">
    <input type="text" class="search-input" id="searchBar" placeholder="Search by Syllable ID or Name..." oninput="filterCards()" />
    <select class="anno-select" id="tierFilter" onchange="filterCards()">
      <option value="all">All Tiers</option>
      <option value="tier_1">Tier 1: Built-in Density (N > 50)</option>
      <option value="tier_2">Tier 2: Compact Neighborhood (G < N <= 50)</option>
      <option value="tier_3">Tier 3: All Bouts (N <= G)</option>
    </select>
  </div>

  <div class="grid-container" id="cardsGrid">
    {cards_joined}
  </div>

  <script>
    const storageKey = 'kpms_v2_seed101_ethologist_labels';
    let labelsState = JSON.parse(localStorage.getItem(storageKey) || '{{}}');

    window.addEventListener('DOMContentLoaded', () => {{
      for (const [s, data] of Object.entries(labelsState)) {{
        const nameEl = document.getElementById(`name-${{s}}`);
        const catEl = document.getElementById(`cat-${{s}}`);
        const confEl = document.getElementById(`conf-${{s}}`);
        const notesEl = document.getElementById(`notes-${{s}}`);
        if (nameEl && data.name) nameEl.value = data.name;
        if (catEl && data.category) catEl.value = data.category;
        if (confEl && data.confidence) confEl.value = data.confidence;
        if (notesEl && data.notes) notesEl.value = data.notes;
      }}
    }});

    function saveLabel(s) {{
      const name = document.getElementById(`name-${{s}}`)?.value || '';
      const category = document.getElementById(`cat-${{s}}`)?.value || '';
      const confidence = document.getElementById(`conf-${{s}}`)?.value || '';
      const notes = document.getElementById(`notes-${{s}}`)?.value || '';
      labelsState[s] = {{ name, category, confidence, notes, timestamp: new Date().toISOString() }};
      localStorage.setItem(storageKey, JSON.stringify(labelsState));
    }}

    function switchTraj(s, mode) {{
      const img = document.getElementById(`traj-img-${{s}}`);
      if (!img) return;
      const pad = String(s).padStart(2, '0');
      img.src = mode === 'side' 
        ? `trajectories/side_by_side/syllable_${{pad}}_trajectory_side_by_side.gif`
        : `trajectories/superimposed/syllable_${{pad}}_trajectory_superimposed.gif`;
    }}

    function filterCards() {{
      const query = document.getElementById('searchBar').value.toLowerCase();
      const tierFilter = document.getElementById('tierFilter').value;
      const cards = document.querySelectorAll('.card');

      cards.forEach(card => {{
        const s = card.getAttribute('data-syllable');
        const tier = card.getAttribute('data-tier');
        const customName = (labelsState[s]?.name || '').toLowerCase();
        const matchesQuery = s.includes(query) || customName.includes(query);
        const matchesTier = (tierFilter === 'all') || tier.includes(tierFilter);
        card.style.display = (matchesQuery && matchesTier) ? 'flex' : 'none';
      }});
    }}

    function exportLabelsJSON() {{
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(labelsState, null, 2));
      const dl = document.createElement('a');
      dl.setAttribute("href", dataStr);
      dl.setAttribute("download", "cohort_static_21kp_v2_seed101_labels.json");
      dl.click();
    }}

    function exportLabelsCSV() {{
      let csv = "syllable_id,name,category,confidence,notes\\n";
      for (const [s, data] of Object.entries(labelsState)) {{
        csv += `"${{s}}","${{data.name || ''}}","${{data.category || ''}}","${{data.confidence || ''}}","${{(data.notes || '').replace(/\\n/g, ' ')}}"\\n`;
      }}
      const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
      const dl = document.createElement('a');
      dl.setAttribute("href", dataStr);
      dl.setAttribute("download", "cohort_static_21kp_v2_seed101_labels.csv");
      dl.click();
    }}
  </script>
</body>
</html>
"""

    with open(HTML_OUT, "w") as f:
        f.write(full_html)

    # Markdown Catalog
    catalog_md = f"""# Cohort Static 21kp V2 Seed 101 Syllable Catalog & Ethologist Reference

This document catalogs all **{len(syll_ids)} syllables** inferred by the **Full Cohort Static 21kp V2 Production Model** (50 AR + 300 SLDS = 350 iterations, Seed 101).

---

## 3-Tier Prototype Selection Rules

1. **Tier 1 (\(N > 50\))**: Built-in KPMS Density Sampling in 4D PCA space (Modal exemplar first, nearest neighbors ordered).
2. **Tier 2 (\(12 < N \le 50\))**: Compact-Neighborhood Selection (Subset of 12 with lowest mean pairwise distance, medoid first).
3. **Tier 3 (\(N \le 12\))**: Complete Bout Enumeration (All available bouts shown once, medoid first, unused tiles blank).

---

## Syllable Catalog Table

| Syllable ID | Usage Rank | Usage % | Cohort Bouts | Med. Duration | Mean Speed (px/s) | Mean Yaw (deg/s) | Prototype Tier | Dual-View Movie | Trajectory GIF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(catalog_md_rows)}

---

*Generated for interactive browser annotation via:* [`{HTML_OUT.name}`](file://{HTML_OUT})
"""

    with open(MD_OUT, "w") as f:
        f.write(catalog_md)

    print(f"\n✅ HTML Gallery generated: {HTML_OUT} ({HTML_OUT.stat().st_size / 1024:.1f} KB)")
    print(f"✅ Markdown Catalog generated: {MD_OUT} ({MD_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
