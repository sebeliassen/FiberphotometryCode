#!/usr/bin/env python3
"""
build_labeling_gallery.py
=============================================================================
Builds an interactive single-page HTML labeling dashboard and markdown catalog
for Champion Seed 101 syllables, linking:
- Tandem side-by-side animated GIF / vector PDF
- Superimposed animated GIF / vector PDF
- Density-sampled tandem 3x4 grid movie
- Random/uniform spread tandem 3x4 grid movie
- Form fields for the user to label syllable behavioral taxonomy, notes, and confidence
=============================================================================
"""

import os
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "visualizations_seed101"
TRAJ_SIDE = VIZ_DIR / "trajectories" / "side_by_side"
TRAJ_SUPER = VIZ_DIR / "trajectories" / "superimposed"
GRID_DENSITY = VIZ_DIR / "grid_movies_density"
GRID_RANDOM = VIZ_DIR / "grid_movies_random"

HTML_OUT = VIZ_DIR / "syllable_labeling_gallery_seed101.html"
MD_OUT = VIZ_DIR / "SEED101_SYLLABLE_CATALOG_FOR_LABELING.md"

def main():
    print("=" * 80)
    print("📋 BUILDING SEED 101 INTERACTIVE LABELING GALLERY & CATALOG")
    print(f"Output HTML: {HTML_OUT}")
    print(f"Output Markdown: {MD_OUT}")
    print("=" * 80)

    # Discover all syllables from trajectory files
    side_pdfs = sorted(TRAJ_SIDE.glob("syllable_*_tandem_side_by_side.pdf"))
    syll_ids = []
    for p in side_pdfs:
        parts = p.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            syll_ids.append(int(parts[1]))
    syll_ids = sorted(list(set(syll_ids)))

    print(f"  Discovered {len(syll_ids)} syllables with visualization assets.")

    # Build Markdown Catalog
    md_lines = [
        "# Champion Seed 101: Comprehensive Syllable Catalog & Labeling Guide",
        "",
        "This catalog aggregates all multi-view visualizations generated for **Champion Seed 101** across 113 cohort sessions ($23.87\\text{M}$ frames).",
        "",
        "## Visual Asset Inventory per Syllable",
        "",
        "| Syllable | Trajectory (Side-by-Side) | Trajectory (Superimposed) | Grid Movie (Density) | Grid Movie (Random) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]

    syllable_data = []

    for s in syll_ids:
        side_gif = TRAJ_SIDE / f"syllable_{s:02d}_tandem_side_by_side.gif"
        side_pdf = TRAJ_SIDE / f"syllable_{s:02d}_tandem_side_by_side.pdf"
        super_gif = TRAJ_SUPER / f"syllable_{s:02d}_superimposed.gif"
        super_pdf = TRAJ_SUPER / f"syllable_{s:02d}_superimposed.pdf"
        density_mp4 = GRID_DENSITY / f"syllable_{s:02d}_tandem_grid_density.mp4"
        random_mp4 = GRID_RANDOM / f"syllable_{s:02d}_tandem_grid_random.mp4"

        side_link = f"[PDF](file://{side_pdf})" if side_pdf.exists() else "—"
        super_link = f"[PDF](file://{super_pdf})" if super_pdf.exists() else "—"
        dens_link = f"[MP4](file://{density_mp4})" if density_mp4.exists() else "—"
        rand_link = f"[MP4](file://{random_mp4})" if random_mp4.exists() else "—"

        md_lines.append(f"| **Syllable {s:02d}** | {side_link} | {super_link} | {dens_link} | {rand_link} |")

        syllable_data.append({
            "id": s,
            "name": f"Syllable {s:02d}",
            "side_gif": str(side_gif.relative_to(VIZ_DIR)) if side_gif.exists() else None,
            "side_pdf": str(side_pdf.relative_to(VIZ_DIR)) if side_pdf.exists() else None,
            "super_gif": str(super_gif.relative_to(VIZ_DIR)) if super_gif.exists() else None,
            "super_pdf": str(super_pdf.relative_to(VIZ_DIR)) if super_pdf.exists() else None,
            "density_mp4": str(density_mp4.relative_to(VIZ_DIR)) if density_mp4.exists() else None,
            "random_mp4": str(random_mp4.relative_to(VIZ_DIR)) if random_mp4.exists() else None,
        })

    with open(MD_OUT, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  ✅ Wrote Markdown Catalog: {MD_OUT}")

    # Build Interactive HTML Labeling Gallery
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Champion Seed 101: Interactive Syllable Labeling Gallery</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d30;
    --bg-hover: #1e293b;
    --accent: #38bdf8;
    --border: #24324f;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-main);
    color: var(--text-primary);
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}
  #sidebar {{
    width: 320px;
    background: var(--bg-card);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }}
  .sidebar-header {{
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }}
  .sidebar-header h1 {{
    font-size: 15px;
    font-weight: 700;
    color: var(--accent);
  }}
  .sidebar-header p {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
  }}
  #syllable-list {{
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }}
  .syll-item {{
    padding: 10px 12px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid var(--border);
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .syll-item:hover, .syll-item.active {{
    background: var(--accent);
    color: #0b0f19;
    border-color: var(--accent);
  }}
  #main {{
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }}
  .card h2 {{
    font-size: 16px;
    color: var(--accent);
    margin-bottom: 12px;
  }}
  .media-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .media-box {{
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .media-box h3 {{
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 10px;
    width: 100%;
    text-align: left;
  }}
  .media-box img, .media-box video {{
    max-width: 100%;
    border-radius: 4px;
    border: 1px solid var(--border);
  }}
  .label-form {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 10px;
  }}
  .label-input {{
    flex: 1;
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    color: var(--text-primary);
    font-size: 13px;
  }}
  .btn-save {{
    background: var(--accent);
    color: #0b0f19;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 700;
    cursor: pointer;
  }}
</style>
</head>
<body>
  <div id="sidebar">
    <div class="sidebar-header">
      <h1>Champion Seed 101 Gallery</h1>
      <p>{len(syll_ids)} Syllables • TopCam & BotCam Tandem</p>
    </div>
    <div id="syllable-list"></div>
  </div>

  <div id="main">
    <div class="card">
      <h2 id="current-title">Select a syllable from the left sidebar</h2>
      <div class="label-form">
        <input type="text" id="label-input" class="label-input" placeholder="Enter ethological label (e.g., Exploratory Locomotion, Facial Grooming, Pause)...">
        <button class="btn-save" onclick="saveLabel()">Save Label</button>
      </div>
    </div>

    <div class="media-grid">
      <div class="media-box">
        <h3>Tandem Side-by-Side Trajectory (Spine Left | Paws Right)</h3>
        <img id="img-side" src="" alt="Side-by-side Trajectory">
      </div>
      <div class="media-box">
        <h3>Superimposed Trajectory (TopCam + BotCam Fused)</h3>
        <img id="img-super" src="" alt="Superimposed Trajectory">
      </div>
      <div class="media-box">
        <h3>Tandem Grid Movie (Density Exemplars)</h3>
        <video id="vid-density" controls loop autoplay muted></video>
      </div>
      <div class="media-box">
        <h3>Tandem Grid Movie (Random Uniform Spread)</h3>
        <video id="vid-random" controls loop autoplay muted></video>
      </div>
    </div>
  </div>

<script>
  const syllables = {json.dumps(syllable_data)};
  let currentIdx = 0;
  let labels = JSON.parse(localStorage.getItem('seed101_syllable_labels') || '{{}}');

  const listContainer = document.getElementById('syllable-list');
  syllables.forEach((s, idx) => {{
    const div = document.createElement('div');
    div.className = 'syll-item' + (idx === 0 ? ' active' : '');
    div.id = 'item-' + s.id;
    div.innerHTML = `<span>${{s.name}}</span><span style="font-size:11px;color:var(--text-secondary);">${{labels[s.id] || ''}}</span>`;
    div.onclick = () => selectSyllable(idx);
    listContainer.appendChild(div);
  }});

  function selectSyllable(idx) {{
    currentIdx = idx;
    const s = syllables[idx];
    document.querySelectorAll('.syll-item').forEach(el => el.classList.remove('active'));
    document.getElementById('item-' + s.id).classList.add('active');

    document.getElementById('current-title').innerText = `${{s.name}} — Multi-View Behavior Profile`;
    document.getElementById('label-input').value = labels[s.id] || '';

    if (s.side_gif) document.getElementById('img-side').src = s.side_gif;
    if (s.super_gif) document.getElementById('img-super').src = s.super_gif;
    if (s.density_mp4) document.getElementById('vid-density').src = s.density_mp4;
    if (s.random_mp4) document.getElementById('vid-random').src = s.random_mp4;
  }}

  function saveLabel() {{
    const s = syllables[currentIdx];
    const val = document.getElementById('label-input').value.trim();
    labels[s.id] = val;
    localStorage.setItem('seed101_syllable_labels', JSON.stringify(labels));
    document.getElementById('item-' + s.id).innerHTML = `<span>${{s.name}}</span><span style="font-size:11px;color:var(--text-secondary);">${{val}}</span>`;
    alert(`Saved label for ${{s.name}}: ${{val}}`);
  }}

  if (syllables.length > 0) {{
    selectSyllable(0);
  }}
</script>
</body>
</html>
"""
    with open(HTML_OUT, "w") as f:
        f.write(html_content)
    print(f"  ✅ Wrote Interactive HTML Gallery: {HTML_OUT}")
    print("=" * 80)

if __name__ == "__main__":
    main()
