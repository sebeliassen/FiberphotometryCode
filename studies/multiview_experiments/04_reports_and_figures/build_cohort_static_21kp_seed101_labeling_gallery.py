#!/usr/bin/env python3
"""
build_cohort_static_21kp_seed101_labeling_gallery.py
=============================================================================
Builds an interactive single-page HTML labeling dashboard and markdown catalog
for Cohort Static 21kp Seed 101 syllables, linking:
- Tandem side-by-side animated GIF / vector PDF
- Superimposed animated GIF / vector PDF
- Density-sampled tandem 3x4 grid movie
- Random/uniform spread tandem 3x4 grid movie
- Real-time kinematic metrics (usage %, bouts, dwell, velocity, yaw rate)
- Form fields for the user to label syllable taxonomy, notes, and confidence

Environment: Python 3
=============================================================================
"""

import json
import re
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "visualizations_cohort_static_21kp_seed101"
TRAJ_SIDE = VIZ_DIR / "trajectories" / "side_by_side"
TRAJ_SUPER = VIZ_DIR / "trajectories" / "superimposed"
GRID_DENSITY = VIZ_DIR / "grid_movies_density"
GRID_RANDOM = VIZ_DIR / "grid_movies_random"

PROJECT_DIR = SCRIPT_DIR.parent / "cohort_static_21kp"
MODEL_NAME = "model_21kp_static_50ar_200slds_seed101"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"

HTML_OUT = VIZ_DIR / "syllable_labeling_gallery_cohort_static_21kp_seed101.html"
MD_OUT = VIZ_DIR / "SEED101_STATIC21KP_SYLLABLE_CATALOG_FOR_LABELING.md"


def main():
    print("=" * 80)
    print("📋 BUILDING COHORT STATIC 21kp SEED 101 LABELING GALLERY & CATALOG")
    print(f"Output HTML: {HTML_OUT}")
    print(f"Output Markdown: {MD_OUT}")
    print("=" * 80)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compute cohort-wide metrics from results.h5
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
                    if dur >= 3:
                        bout_counts[val] = bout_counts.get(val, 0) + 1
                        if val not in durations:
                            durations[val] = []
                            vel_means[val] = []
                            ang_means[val] = []
                        durations[val].append(dur)
                        if end > start + 1 and end - 1 < len(vel):
                            vel_means[val].append(float(np.mean(vel[start:end-1])))
                            ang_means[val].append(float(np.mean(ang_vel[start:end-1])))

    # Discover all syllables from files or results
    discovered_sylls = set()
    for p in TRAJ_SIDE.glob("syllable_*_tandem_side_by_side.pdf"):
        m = re.search(r"syllable_(\d+)", p.name)
        if m:
            discovered_sylls.add(int(m.group(1)))
    for p in GRID_DENSITY.glob("syllable_*_tandem_grid_density.mp4"):
        m = re.search(r"syllable_(\d+)", p.name)
        if m:
            discovered_sylls.add(int(m.group(1)))
    for s in syll_counts.keys():
        discovered_sylls.add(s)

    syll_ids = sorted(list(discovered_sylls))
    # Sort primarily by usage % descending
    syll_ids = sorted(syll_ids, key=lambda s: syll_counts.get(s, 0), reverse=True)

    print(f"  Aggregated {len(syll_ids)} syllables across cohort.", flush=True)

    # 2. Build Markdown Catalog
    print("[2/3] Writing Markdown Catalog...", flush=True)
    md_lines = [
        "# Cohort Static 21kp Production: Seed 101 Syllable Catalog & Labeling Guide",
        "",
        "This catalog aggregates all multi-view visualizations and quantitative metrics generated for **Cohort Static 21kp Seed 101** across the 118-session cohort ($18,929,934$ active frames across 338 continuous segments).",
        "",
        "## Cohort Summary",
        f"- **Total Frames:** {total_frames:,}",
        f"- **Total Bouts:** {sum(bout_counts.values()):,}",
        f"- **Active Syllables (>=0.1% usage):** {sum(1 for s in syll_ids if (syll_counts.get(s, 0) / max(1, total_frames)) >= 0.001)}",
        f"- **Interactive Gallery:** [syllable_labeling_gallery_cohort_static_21kp_seed101.html](file://{HTML_OUT})",
        "",
        "## Visual Asset Inventory & Kinematic Metrics per Syllable",
        "",
        "| Syllable | Usage % | Frames | Bouts | Med Dwell | Velocity | Yaw Rate | Trajectory (Side-by-Side) | Trajectory (Superimposed) | Grid Movie (Density) | Grid Movie (Random) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    syllable_data = []

    for s in syll_ids:
        f_cnt = syll_counts.get(s, 0)
        pct = (f_cnt / max(1, total_frames)) * 100.0
        b_cnt = bout_counts.get(s, 0)
        durs = durations.get(s, [0])
        med_dur_ms = (np.median(durs) / 60.0) * 1000.0 if durs else 0.0
        v_mean = np.mean(vel_means[s]) if vel_means.get(s) else 0.0
        w_mean = np.mean(ang_means[s]) if ang_means.get(s) else 0.0

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

        md_lines.append(
            f"| **Syllable {s:02d}** | {pct:.2f}% | {f_cnt:,} | {b_cnt:,} | {med_dur_ms:.0f} ms | {v_mean:.1f} px/s | {w_mean:.1f}°/s | {side_link} | {super_link} | {dens_link} | {rand_link} |"
        )

        syllable_data.append({
            "id": int(s),
            "name": f"Syllable {int(s):02d}",
            "usage_pct": float(round(pct, 2)),
            "frames": int(f_cnt),
            "bouts": int(b_cnt),
            "dwell_ms": float(round(med_dur_ms, 1)),
            "vel": float(round(v_mean, 1)),
            "yaw": float(round(w_mean, 1)),
            "side_gif": str(side_gif.relative_to(VIZ_DIR)) if side_gif.exists() else None,
            "side_pdf": str(side_pdf.relative_to(VIZ_DIR)) if side_pdf.exists() else None,
            "super_gif": str(super_gif.relative_to(VIZ_DIR)) if super_gif.exists() else None,
            "super_pdf": str(super_pdf.relative_to(VIZ_DIR)) if super_pdf.exists() else None,
            "density_mp4": str(density_mp4.relative_to(VIZ_DIR)) if density_mp4.exists() else None,
            "random_mp4": str(random_mp4.relative_to(VIZ_DIR)) if random_mp4.exists() else None,
        })

    with open(MD_OUT, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  ✅ Wrote Markdown Catalog: {MD_OUT}", flush=True)

    # 3. Build Interactive HTML Labeling Gallery
    print("[3/3] Generating Interactive HTML Labeling Dashboard...", flush=True)
    json_data = json.dumps(syllable_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cohort Static 21kp Seed 101: Syllable Labeling Gallery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-dark: #0f1117;
    --card-bg: rgba(26, 31, 46, 0.85);
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
    grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
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
  .stats-row {{
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }}
  .media-wrapper {{
    position: relative;
    background: #000;
    width: 100%;
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid var(--border-color);
    overflow: hidden;
  }}
  .media-wrapper video, .media-wrapper img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }}
  .tab-buttons {{
    display: flex;
    background: rgba(0, 0, 0, 0.3);
    border-bottom: 1px solid var(--border-color);
  }}
  .tab-btn {{
    flex: 1;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
  }}
  .tab-btn.active {{
    color: #fff;
    background: rgba(255, 255, 255, 0.08);
    font-weight: 600;
  }}
  .card-body {{
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    flex-grow: 1;
  }}
  .label-group {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .label-group label {{
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .text-field {{
    background: rgba(15, 17, 23, 0.7);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
    font-family: inherit;
  }}
  .text-field:focus {{ border-color: var(--primary-blue); }}
  textarea.text-field {{ resize: vertical; min-height: 52px; }}
  .action-links {{
    display: flex;
    gap: 12px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color);
  }}
  .action-links a {{
    font-size: 12px;
    color: var(--accent-cyan);
    text-decoration: none;
    font-weight: 500;
  }}
  .action-links a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Cohort Static 21kp: Syllable Labeling Dashboard (Seed 101)</h1>
    <div class="meta-badges">
      <span class="badge">Model: 50 AR + 200 SLDS</span>
      <span class="badge">118 Sessions</span>
      <span class="badge">18.93M Frames</span>
      <span class="badge">100 Syllables</span>
    </div>
  </div>
  <div>
    <button class="btn" onclick="exportAnnotations()">Export Annotations (JSON)</button>
  </div>
</header>

<div class="controls-bar">
  <input type="text" class="search-input" id="searchBox" placeholder="Filter syllables (e.g. 37, locomotion, groom)..." oninput="filterCards()">
  <span style="font-size: 13px; color: var(--text-muted);" id="syllableCount">Showing all syllables</span>
</div>

<div class="grid-container" id="cardsGrid"></div>

<script>
const data = {json_data};
let annotations = JSON.parse(localStorage.getItem('cohort_static_21kp_seed101_labels') || '{{}}');

function renderCards(filterText = '') {{
  const container = document.getElementById('cardsGrid');
  container.innerHTML = '';
  let count = 0;

  data.forEach(item => {{
    const sId = item.id;
    const sStr = sId.toString().padStart(2, '0');
    const note = annotations[sId]?.notes || '';
    const label = annotations[sId]?.label || '';

    if (filterText && !sStr.includes(filterText) && !label.toLowerCase().includes(filterText.toLowerCase()) && !note.toLowerCase().includes(filterText.toLowerCase())) {{
      return;
    }}
    count++;

    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card-${{sId}}`;

    const initialMedia = item.density_mp4 ? 
      `<video src="${{item.density_mp4}}" autoplay loop muted playsinline controls></video>` :
      (item.side_gif ? `<img src="${{item.side_gif}}" alt="Syllable ${{sStr}}">` : `<div style="color:#666">No Media Available</div>`);

    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">Syllable ${{sStr}}</div>
        <div class="stats-row">
          <span>${{item.usage_pct}}%</span>
          <span>${{item.dwell_ms}} ms</span>
          <span>${{item.vel}} px/s</span>
          <span>${{item.yaw}}°/s</span>
        </div>
      </div>
      <div class="tab-buttons">
        <button class="tab-btn active" onclick="switchMedia(${{sId}}, 'density', this)">Density Grid</button>
        <button class="tab-btn" onclick="switchMedia(${{sId}}, 'random', this)">Random Grid</button>
        <button class="tab-btn" onclick="switchMedia(${{sId}}, 'side', this)">Trajectory Side</button>
        <button class="tab-btn" onclick="switchMedia(${{sId}}, 'super', this)">Superimposed</button>
      </div>
      <div class="media-wrapper" id="media-${{sId}}">
        ${{initialMedia}}
      </div>
      <div class="card-body">
        <div class="label-group">
          <label>Taxonomy Label</label>
          <input type="text" class="text-field" placeholder="e.g. forward_locomotion, rearing, groom" value="${{label}}" onchange="saveLabel(${{sId}}, 'label', this.value)">
        </div>
        <div class="label-group">
          <label>Observations & Notes</label>
          <textarea class="text-field" placeholder="Kinematics, paw alternation, snout angle..." onchange="saveLabel(${{sId}}, 'notes', this.value)">${{note}}</textarea>
        </div>
        <div class="action-links">
          ${{item.side_pdf ? `<a href="${{item.side_pdf}}" target="_blank">Side PDF</a>` : ''}}
          ${{item.super_pdf ? `<a href="${{item.super_pdf}}" target="_blank">Super PDF</a>` : ''}}
          ${{item.density_mp4 ? `<a href="${{item.density_mp4}}" download>Density MP4</a>` : ''}}
        </div>
      </div>
    `;
    container.appendChild(card);
  }});

  document.getElementById('syllableCount').innerText = `Showing ${{count}} of ${{data.length}} syllables`;
}}

function switchMedia(sId, type, btn) {{
  const card = document.getElementById(`card-${{sId}}`);
  const wrapper = document.getElementById(`media-${{sId}}`);
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const item = data.find(d => d.id === sId);
  if (!item) return;

  if (type === 'density') {{
    wrapper.innerHTML = item.density_mp4 ? `<video src="${{item.density_mp4}}" autoplay loop muted playsinline controls></video>` : '<div style="color:#666">Density MP4 Not Found</div>';
  }} else if (type === 'random') {{
    wrapper.innerHTML = item.random_mp4 ? `<video src="${{item.random_mp4}}" autoplay loop muted playsinline controls></video>` : '<div style="color:#666">Random MP4 Not Found</div>';
  }} else if (type === 'side') {{
    wrapper.innerHTML = item.side_gif ? `<img src="${{item.side_gif}}" alt="Side Trajectory">` : '<div style="color:#666">Side GIF Not Found</div>';
  }} else if (type === 'super') {{
    wrapper.innerHTML = item.super_gif ? `<img src="${{item.super_gif}}" alt="Superimposed Trajectory">` : '<div style="color:#666">Superimposed GIF Not Found</div>';
  }}
}}

function saveLabel(sId, field, val) {{
  if (!annotations[sId]) annotations[sId] = {{}};
  annotations[sId][field] = val;
  localStorage.setItem('cohort_static_21kp_seed101_labels', JSON.stringify(annotations));
}}

function filterCards() {{
  const query = document.getElementById('searchBox').value.trim();
  renderCards(query);
}}

function exportAnnotations() {{
  const blob = new Blob([JSON.stringify(annotations, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'seed101_static21kp_syllable_annotations.json';
  a.click();
}}

renderCards();
</script>
</body>
</html>
"""

    with open(HTML_OUT, "w") as f:
        f.write(html_content)
    print(f"  ✅ Wrote HTML Gallery: {HTML_OUT}", flush=True)


if __name__ == "__main__":
    main()
