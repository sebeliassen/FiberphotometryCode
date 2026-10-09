#!/usr/bin/env python3
"""
build_seed404_labeling_gallery.py
=============================================================================
Builds an interactive single-page HTML labeling dashboard and markdown catalog
for Champion Seed 404 syllables, linking:
- Tandem side-by-side animated GIF / vector PDF
- Superimposed animated GIF / vector PDF
- Density-sampled tandem 3x4 grid movie
- Random/uniform spread tandem 3x4 grid movie
- Form fields for the user to label syllable behavioral taxonomy, notes, and confidence
=============================================================================
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "visualizations_seed404"
TRAJ_SIDE = VIZ_DIR / "trajectories" / "side_by_side"
TRAJ_SUPER = VIZ_DIR / "trajectories" / "superimposed"
GRID_DENSITY = VIZ_DIR / "grid_movies_density"
GRID_RANDOM = VIZ_DIR / "grid_movies_random"

MODEL_NAME = "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073"
HTML_OUT = VIZ_DIR / "syllable_labeling_gallery_seed404.html"
MD_OUT = VIZ_DIR / "SEED404_SYLLABLE_CATALOG_FOR_LABELING.md"

def main():
    print("=" * 80)
    print("📋 BUILDING SEED 404 INTERACTIVE LABELING GALLERY & CATALOG")
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
        "# Champion Seed 404: Comprehensive Syllable Catalog & Labeling Guide",
        "",
        "This catalog aggregates all multi-view visualizations generated for **Champion Seed 404** across the 118-session continuous crystal-calibrated cohort ($17.37\\text{M}$ active frames).",
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
<title>Champion Seed 404: Interactive Syllable Labeling Gallery</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151c2e;
    --border-color: #24304f;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-blue: #388bfd;
    --accent-green: #238636;
    --accent-gold: #d29922;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 24px;
    line-height: 1.5;
  }}
  header {{
    max-width: 1400px;
    margin: 0 auto 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
  }}
  h1 {{ font-size: 24px; margin-bottom: 8px; color: #58a6ff; }}
  p.subtitle {{ color: var(--text-secondary); font-size: 14px; }}
  
  .stats-bar {{
    display: flex;
    gap: 20px;
    margin-top: 12px;
    font-size: 13px;
    color: #7ee787;
  }}

  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }}

  .syllable-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}

  .card-header {{
    padding: 12px 16px;
    background: #1c2438;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
  }}
  .card-title {{ font-size: 16px; font-weight: 600; color: #79c0ff; }}

  .card-media {{
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .preview-row {{
    display: flex;
    gap: 12px;
  }}
  .preview-box {{
    flex: 1;
    background: #0d1117;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
  }}
  .preview-box img, .preview-box video {{
    width: 100%;
    height: auto;
    border-radius: 4px;
    display: block;
  }}
  .box-label {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  .card-form {{
    padding: 12px 16px 16px;
    background: #111726;
    border-top: 1px solid var(--border-color);
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .form-label {{ font-size: 12px; font-weight: 600; color: var(--text-secondary); }}
  .form-input, .form-select, .form-textarea {{
    width: 100%;
    background: #0d1117;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 13px;
  }}
  .form-textarea {{ resize: vertical; min-height: 48px; }}
  .toolbar {{
    margin-top: 14px;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .btn {{
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .btn:hover {{ background: #30363d; border-color: #8b949e; color: #fff; }}
  .btn-primary {{ background: #238636; border-color: #2ea043; color: #fff; }}
  .btn-primary:hover {{ background: #2ea043; }}
  .btn-danger {{ background: #da3633; border-color: #f85149; color: #fff; }}
  .btn-danger:hover {{ background: #b62324; }}
  .progress-badge {{
    margin-left: auto;
    background: #161b22;
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    color: #58a6ff;
    font-weight: 600;
  }}
  .save-status {{
    font-size: 12px;
    color: #7ee787;
    margin-left: 8px;
    transition: opacity 0.3s;
  }}
</style>
</head>
<body>

<header>
  <h1>Tandem Multi-View KPMS: Champion Seed 404 Syllable Gallery</h1>
  <p class="subtitle">118-session continuous crystal-calibrated cohort (17.37M active frames, 3,473 segments). Rank 1 EML Winner.</p>
  <div class="stats-bar">
    <span>🏆 Model: Seed 404 (Rank 1)</span>
    <span>⚡ Total Syllables Audited: {len(syll_ids)}</span>
    <span>🎯 Dual-View Scale: 144px TopCam vs 384px BotCam (160x160 Normalized)</span>
  </div>
  <div class="toolbar">
    <button class="btn btn-primary" onclick="exportJSON()">💾 Export JSON</button>
    <button class="btn btn-primary" onclick="exportCSV()">📊 Export CSV</button>
    <label class="btn" style="cursor: pointer;">
      📂 Import Annotations <input type="file" id="import-file" accept=".json,.csv" style="display:none;" onchange="importAnnotations(event)">
    </label>
    <button class="btn btn-danger" onclick="clearAllAnnotations()">🗑️ Clear Form</button>
    <span id="save-status" class="save-status">✓ Autosaved to browser</span>
    <div id="progress-badge" class="progress-badge">Labeled: 0 / {len(syll_ids)} (0%)</div>
  </div>
</header>

<div class="gallery-grid">
"""

    for item in syllable_data:
        s_id = item["id"]
        side_gif_tag = f'<img src="{item["side_gif"]}" alt="Tandem Trajectory">' if item["side_gif"] else '<div style="padding:40px 0; color:#555;">No Trajectory</div>'
        density_mp4_tag = f'<video src="{item["density_mp4"]}" autoplay loop muted playsinline></video>' if item["density_mp4"] else '<div style="padding:40px 0; color:#555;">No Grid Video</div>'

        html_content += f"""
  <div class="syllable-card" id="card-syll-{s_id}">
    <div class="card-header">
      <span class="card-title">Syllable {s_id:02d}</span>
      <div style="font-size: 12px;">
        {'<a href="' + item["side_pdf"] + '" style="color:#58a6ff; text-decoration:none; margin-right:8px;">[Side PDF]</a>' if item["side_pdf"] else ''}
        {'<a href="' + item["density_mp4"] + '" style="color:#7ee787; text-decoration:none;">[Grid MP4]</a>' if item["density_mp4"] else ''}
      </div>
    </div>
    <div class="card-media">
      <div class="preview-row">
        <div class="preview-box">
          {side_gif_tag}
          <div class="box-label">Dorsal Left | Ventral Right</div>
        </div>
      </div>
      <div class="preview-row">
        <div class="preview-box">
          {density_mp4_tag}
          <div class="box-label">Tandem Grid Video (Median Duration)</div>
        </div>
      </div>
    </div>
    <div class="card-form">
      <div>
        <label class="form-label">Behavioral Taxonomy Name:</label>
        <input type="text" class="form-input annot-input" id="name-{s_id}" data-syll="{s_id}" placeholder="e.g. forward locomotion, rearing, grooming...">
      </div>
      <div>
        <label class="form-label">Kinematic Notes:</label>
        <textarea class="form-textarea annot-input" id="notes-{s_id}" data-syll="{s_id}" placeholder="Paw excursions, spine bending, heading change..."></textarea>
      </div>
    </div>
  </div>
"""

    html_content += f"""
</div>

<script>
const TOTAL_SYLLS = {len(syll_ids)};
const MODEL_NAME = "{MODEL_NAME}";
const STORAGE_KEY = "kpms_annot_" + MODEL_NAME;

function loadStored() {{
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return {{}};
  try {{ return JSON.parse(raw); }} catch (e) {{ return {{}}; }}
}}

function saveStored(data) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  const el = document.getElementById("save-status");
  el.textContent = "✓ Autosaved (" + new Date().toLocaleTimeString() + ")";
  el.style.opacity = "1";
  updateProgress(data);
}}

function updateProgress(data) {{
  let count = 0;
  for (let s = 0; s < TOTAL_SYLLS; s++) {{
    if (data[s] && data[s].name && data[s].name.trim().length > 0) {{
      count++;
    }}
  }}
  const pct = Math.round((count / TOTAL_SYLLS) * 100);
  document.getElementById("progress-badge").textContent = `Labeled: ${{count}} / ${{TOTAL_SYLLS}} (${{pct}}%)`;
}}

function init() {{
  const data = loadStored();
  for (let s = 0; s < TOTAL_SYLLS; s++) {{
    const nameEl = document.getElementById(`name-${{s}}`);
    const notesEl = document.getElementById(`notes-${{s}}`);
    if (data[s]) {{
      if (nameEl && data[s].name) nameEl.value = data[s].name;
      if (notesEl && data[s].notes) notesEl.value = data[s].notes;
    }}
  }}
  updateProgress(data);

  document.querySelectorAll(".annot-input").forEach(el => {{
    el.addEventListener("input", () => {{
      const s = el.getAttribute("data-syll");
      const nameVal = document.getElementById(`name-${{s}}`).value;
      const notesVal = document.getElementById(`notes-${{s}}`).value;
      const cur = loadStored();
      cur[s] = {{ name: nameVal, notes: notesVal, updated_at: new Date().toISOString() }};
      saveStored(cur);
    }});
  }});
}}

function exportJSON() {{
  const cur = loadStored();
  const payload = {{
    model_name: MODEL_NAME,
    eml_rank: 1,
    eml_score: 225001430.22,
    iterations: 250,
    export_time: new Date().toISOString(),
    total_syllables: TOTAL_SYLLS,
    annotations: []
  }};
  for (let s = 0; s < TOTAL_SYLLS; s++) {{
    payload.annotations.push({{
      syllable_id: s,
      taxonomy_name: cur[s]?.name || "",
      kinematic_notes: cur[s]?.notes || "",
      updated_at: cur[s]?.updated_at || null
    }});
  }}
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `seed404_behavioral_annotations_${{new Date().toISOString().slice(0, 10)}}.json`;
  a.click();
}}

function exportCSV() {{
  const cur = loadStored();
  let csv = "syllable_id,taxonomy_name,kinematic_notes,model_name,updated_at\\n";
  for (let s = 0; s < TOTAL_SYLLS; s++) {{
    const name = (cur[s]?.name || "").replace(/"/g, '""');
    const notes = (cur[s]?.notes || "").replace(/"/g, '""');
    const updated = cur[s]?.updated_at || "";
    csv += `${{s}},"${{name}}","${{notes}}","${{MODEL_NAME}}","${{updated}}"\\n`;
  }}
  const blob = new Blob([csv], {{ type: "text/csv" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `seed404_behavioral_annotations_${{new Date().toISOString().slice(0, 10)}}.csv`;
  a.click();
}}

function importAnnotations(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const text = e.target.result;
      const cur = loadStored();
      if (file.name.endsWith(".json")) {{
        const parsed = JSON.parse(text);
        const list = parsed.annotations || parsed;
        list.forEach(item => {{
          const sid = item.syllable_id !== undefined ? item.syllable_id : item.id;
          if (sid !== undefined) {{
            cur[sid] = {{
              name: item.taxonomy_name || item.name || "",
              notes: item.kinematic_notes || item.notes || "",
              updated_at: item.updated_at || new Date().toISOString()
            }};
          }}
        }});
      }}
      saveStored(cur);
      location.reload();
    }} catch (err) {{
      alert("Failed to parse imported file: " + err.message);
    }}
  }};
  reader.readAsText(file);
}}

function clearAllAnnotations() {{
  if (confirm("Are you sure you want to clear all entered behavioral names and notes from browser storage?")) {{
    localStorage.removeItem(STORAGE_KEY);
    location.reload();
  }}
}}

window.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""

    with open(HTML_OUT, "w") as f:
        f.write(html_content)
    print(f"  ✅ Wrote Interactive HTML Gallery: {HTML_OUT}")

if __name__ == "__main__":
    main()
