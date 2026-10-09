#!/usr/bin/env python3
"""
generate_4state_interactive_html.py
=============================================================================
Builds a high-performance interactive HTML5 canvas visualization of the
30-minute hierarchical timeline with M=4 Upper States and a continuous
Posterior Probability Ribbon showing P(U1)..P(U4) across time.

Features:
- Pan and zoom across the entire 1,800-second (30-minute) window.
- 4 tracks per session:
    1. Upper Macro-States (U1..U4) with neutral numerical labels
    2. Shared Subroutines (R1..R6) with explicit boundaries
    3. KPMS Syllable Bouts (0..99)
    4. Posterior Probability Ribbon for P(U1), P(U2), P(U3), P(U4)
- Hover tooltips showing exact timestamps, duration, constituent bouts,
  and the exact posterior probability distribution.
=============================================================================
"""

import json
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"

UPPER_COLORS_4 = {
    0: "#6C8EBF", # U1: Slate Blue
    1: "#82B366", # U2: Sage Green
    2: "#D6B656", # U3: Warm Ochre
    3: "#B85450"  # U4: Muted Terracotta
}

ROUTINE_COLORS = {
    0: "#E41A1C", # R1: Vivid Crimson
    1: "#377EB8", # R2: Sapphire Blue
    2: "#4DAF4A", # R3: Emerald Green
    3: "#984EA3", # R4: Royal Amethyst
    4: "#FF7F00", # R5: Amber Orange
    5: "#00A896"  # R6: Deep Teal
}

def build_4state_interactive_html(sessions_data, output_path):
    # Prepare compact JSON data
    json_sessions = []
    for s in sessions_data:
        # Downsample posterior curve for smooth rendering (step of ~0.5s or per bout)
        onsets = s["onsets"]
        gamma = s["gamma_macro"]

        # Subsample posterior points every 1-2 bouts to keep HTML compact
        step = max(1, len(onsets) // 2500)
        prob_curve = []
        for i in range(0, len(onsets), step):
            prob_curve.append({
                "t": round(float(onsets[i]), 2),
                "p": [round(float(gamma[i, u]), 3) for u in range(4)]
            })

        json_sessions.append({
            "session_idx": s["session_idx"],
            "animal_id": s["animal_id"],
            "genotype": s["genotype"],
            "drug": s["drug"],
            "total_sec": round(s["total_sec"], 2),
            "macro_blocks": [{
                "m": b["macro_state"],
                "s": round(b["start_sec"], 2),
                "e": round(b["end_sec"], 2),
                "d": round(b["duration_sec"], 2),
                "b": b["bouts"],
                "c": round(b.get("mean_certainty", 1.0), 3)
            } for b in s["macro_blocks"]],
            "rout_blocks": [{
                "r": b["routine_id"],
                "m": b["macro_state"],
                "s": round(b["start_sec"], 2),
                "e": round(b["end_sec"], 2),
                "d": round(b["duration_sec"], 2),
                "b": b["bouts"]
            } for b in s["rout_blocks"]],
            "syll_blocks": [{
                "s_id": b["syllable"],
                "r": b["routine"],
                "st": b["stage"],
                "m": b["macro"],
                "s": round(b["start_sec"], 2),
                "e": round(b["end_sec"], 2),
                "d": round(b["duration_sec"], 2)
            } for b in s["syll_blocks"]],
            "prob_curve": prob_curve
        })

    json_str = json.dumps(json_sessions)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>30-Minute Hierarchical Timeline (M=4 Upper States + Posterior Certainty)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0f172a;
    color: #f8fafc;
    overflow-x: hidden;
    padding: 20px;
  }}
  header {{
    margin-bottom: 16px;
    border-bottom: 1px solid #334155;
    padding-bottom: 12px;
  }}
  h1 {{
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .badge {{
    font-size: 11px;
    font-weight: 600;
    background: #0369a1;
    color: #e0f2fe;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
    background: #1e293b;
    padding: 10px 16px;
    border-radius: 8px;
    border: 1px solid #334155;
  }}
  button {{
    background: #3b82f6;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }}
  button:hover {{ background: #2563eb; }}
  .legend-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 10px;
    font-size: 11px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(0,0,0,0.3);
  }}
  #canvas-container {{
    position: relative;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
  }}
  canvas {{ display: block; }}
  #tooltip {{
    position: absolute;
    display: none;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
    line-height: 1.4;
    color: #f8fafc;
    pointer-events: none;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    min-width: 180px;
  }}
</style>
</head>
<body>

<header>
  <h1>
    30-Minute Hierarchical Subroutine Timeline (M=4 Upper States)
    <span class="badge">4 Upper States + Posterior Certainty Track</span>
  </h1>
  <p style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
    Continuous 0 to 1,800s recording. Tracks: <strong>Upper Macro-State (U1..U4)</strong>, <strong>Shared Subroutines (R1..R6)</strong>, <strong>KPMS Syllable Bouts</strong>, and <strong>Posterior Probabilities P(U1..U4)</strong>.
  </p>
</header>

<div class="controls">
  <button id="btn-reset">Full 30 Min (0-1800s)</button>
  <button id="btn-zoom-600">Zoom Window 10–12m (600–720s)</button>
  <button id="btn-zoom-in">Zoom In (+)</button>
  <button id="btn-zoom-out">Zoom Out (-)</button>
  <span style="font-size: 12px; color: #94a3b8; margin-left: 10px;">Drag mouse to pan • Scroll wheel to zoom</span>
  <span id="window-display" style="font-size: 12px; font-weight: bold; color: #38bdf8; margin-left: auto;"></span>
</div>

<div class="legend-bar">
  <strong style="color: #94a3b8;">Upper States:</strong>
  <div class="legend-item"><div class="swatch" style="background:#6C8EBF;"></div> U1</div>
  <div class="legend-item"><div class="swatch" style="background:#82B366;"></div> U2</div>
  <div class="legend-item"><div class="swatch" style="background:#D6B656;"></div> U3</div>
  <div class="legend-item"><div class="swatch" style="background:#B85450;"></div> U4</div>
  <span style="border-left: 1px solid #475569; margin: 0 4px;"></span>
  <strong style="color: #94a3b8;">Routines:</strong>
  <div class="legend-item"><div class="swatch" style="background:#E41A1C;"></div> R1</div>
  <div class="legend-item"><div class="swatch" style="background:#377EB8;"></div> R2</div>
  <div class="legend-item"><div class="swatch" style="background:#4DAF4A;"></div> R3</div>
  <div class="legend-item"><div class="swatch" style="background:#984EA3;"></div> R4</div>
  <div class="legend-item"><div class="swatch" style="background:#FF7F00;"></div> R5</div>
  <div class="legend-item"><div class="swatch" style="background:#00A896;"></div> R6</div>
</div>

<div id="canvas-container" style="margin-top: 12px;">
  <canvas id="timeline-canvas"></canvas>
  <div id="tooltip"></div>
</div>

<script>
const SESSIONS = {json_str};

const UPPER_COLORS = ["#6C8EBF", "#82B366", "#D6B656", "#B85450"];
const ROUTINE_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#00A896"];

const SYLLABLE_COLORS = [];
for (let i = 0; i < 100; i++) {{
  const hue = (i * 137.5) % 360;
  SYLLABLE_COLORS.push(`hsl(${{hue}}, 65%, 60%)`);
}}

const canvas = document.getElementById("timeline-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const windowDisplay = document.getElementById("window-display");

let viewStart = 0;
let viewEnd = 1800;
const TOTAL_MAX_TIME = 1800;

const MARGIN_LEFT = 160;
const MARGIN_RIGHT = 30;
const MARGIN_TOP = 40;
const MARGIN_BOTTOM = 40;

const TRACK_H = 24;
const PROB_H = 28;
const TRACK_GAP = 6;
const TRACK_GROUP_HEIGHT = 150;

function resize() {{
  const container = document.getElementById("canvas-container");
  canvas.width = container.clientWidth;
  canvas.height = MARGIN_TOP + SESSIONS.length * TRACK_GROUP_HEIGHT + MARGIN_BOTTOM;
  render();
}}
window.addEventListener("resize", resize);

function timeToX(t) {{
  const plotWidth = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;
  return MARGIN_LEFT + ((t - viewStart) / (viewEnd - viewStart)) * plotWidth;
}}

function xToTime(x) {{
  const plotWidth = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;
  return viewStart + ((x - MARGIN_LEFT) / plotWidth) * (viewEnd - viewStart);
}}

function render() {{
  windowDisplay.innerText = `Window: ${{viewStart.toFixed(1)}}s – ${{viewEnd.toFixed(1)}}s (Span: ${{(viewEnd - viewStart).toFixed(1)}}s / ${{( (viewEnd - viewStart)/60 ).toFixed(1)}}m)`;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const plotWidth = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;
  const span = viewEnd - viewStart;

  // Grid lines
  let gridStep = 300;
  if (span <= 120) gridStep = 10;
  else if (span <= 300) gridStep = 30;
  else if (span <= 600) gridStep = 60;

  const firstGrid = Math.floor(viewStart / gridStep) * gridStep;
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";

  for (let t = firstGrid; t <= viewEnd; t += gridStep) {{
    if (t < viewStart) continue;
    const x = timeToX(t);
    ctx.beginPath();
    ctx.moveTo(x, MARGIN_TOP);
    ctx.lineTo(x, canvas.height - MARGIN_BOTTOM);
    ctx.stroke();

    const min = Math.floor(t / 60);
    const sec = Math.floor(t % 60);
    const label = span > 300 ? `${{min}}m` : `${{min}}m ${{sec}}s`;
    ctx.fillText(label, x, canvas.height - MARGIN_BOTTOM + 18);
  }}

  // Sessions
  SESSIONS.forEach((sess, sIdx) => {{
    const groupY = MARGIN_TOP + sIdx * TRACK_GROUP_HEIGHT;

    // Margin labels
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`Session ${{sess.session_idx}} | Animal ${{sess.animal_id}}`, MARGIN_LEFT - 12, groupY + 20);
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`(${{sess.genotype}}, ${{sess.drug}})`, MARGIN_LEFT - 12, groupY + 36);

    ctx.font = "9px sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Upper (M=4)", MARGIN_LEFT - 12, groupY + 18);
    ctx.fillText("Routine", MARGIN_LEFT - 12, groupY + 18 + TRACK_H + TRACK_GAP);
    ctx.fillText("Syllable", MARGIN_LEFT - 12, groupY + 18 + (TRACK_H + TRACK_GAP) * 2);
    ctx.fillText("P(U1..U4)", MARGIN_LEFT - 12, groupY + 18 + (TRACK_H + TRACK_GAP) * 2 + PROB_H);

    // Track 1: Upper Macro-States
    const y1 = groupY + 5;
    sess.macro_blocks.forEach(b => {{
      if (b.e < viewStart || b.s > viewEnd) return;
      const x1 = Math.max(timeToX(b.s), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.e), MARGIN_LEFT + plotWidth);
      const w = Math.max(1, x2 - x1);

      ctx.fillStyle = UPPER_COLORS[b.m] || "#888";
      ctx.fillRect(x1, y1, w, TRACK_H);
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, w, TRACK_H);

      if (w > 35) {{
        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`U${{b.m + 1}}`, x1 + w/2, y1 + 16);
      }}
    }});

    // Track 2: Routines
    const y2 = y1 + TRACK_H + TRACK_GAP;
    sess.rout_blocks.forEach(b => {{
      if (b.e < viewStart || b.s > viewEnd) return;
      const x1 = Math.max(timeToX(b.s), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.e), MARGIN_LEFT + plotWidth);
      const w = Math.max(1, x2 - x1);

      ctx.fillStyle = ROUTINE_COLORS[b.r] || "#555";
      ctx.fillRect(x1, y2, w, TRACK_H);
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x1, y2, w, TRACK_H);

      if (w > 25 && span <= 300) {{
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`R${{b.r + 1}}`, x1 + w/2, y2 + 16);
      }}
    }});

    // Track 3: Syllables
    const y3 = y2 + TRACK_H + TRACK_GAP;
    sess.syll_blocks.forEach(b => {{
      if (b.e < viewStart || b.s > viewEnd) return;
      const x1 = Math.max(timeToX(b.s), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.e), MARGIN_LEFT + plotWidth);
      const w = Math.max(0.5, x2 - x1);

      ctx.fillStyle = SYLLABLE_COLORS[b.s_id] || "#aaa";
      ctx.fillRect(x1, y3, w, TRACK_H);

      if (span <= 60 && w > 12) {{
        ctx.fillStyle = "#111";
        ctx.font = "8px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`${{b.s_id}}`, x1 + w/2, y3 + 15);
      }}
    }});

    // Track 4: Posterior Probabilities Ribbon
    const y4 = y3 + TRACK_H + TRACK_GAP;
    ctx.fillStyle = "#090d16";
    ctx.fillRect(MARGIN_LEFT, y4, plotWidth, PROB_H);
    ctx.strokeStyle = "#334155";
    ctx.strokeRect(MARGIN_LEFT, y4, plotWidth, PROB_H);

    // Render 4 colored lines for P(U1)..P(U4)
    for (let u = 0; u < 4; u++) {{
      ctx.strokeStyle = UPPER_COLORS[u];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;

      sess.prob_curve.forEach(pt => {{
        if (pt.t < viewStart - 10 || pt.t > viewEnd + 10) return;
        const x = timeToX(pt.t);
        const y = y4 + PROB_H - (pt.p[u] * PROB_H);
        if (!started) {{
          ctx.moveTo(x, y);
          started = true;
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();
    }}
  }});
}}

// Interactions: Pan and Zoom
let isDragging = false;
let startDragX = 0;
let startViewStart = 0;
let startViewEnd = 0;

canvas.addEventListener("mousedown", e => {{
  if (e.clientX >= MARGIN_LEFT) {{
    isDragging = true;
    startDragX = e.clientX;
    startViewStart = viewStart;
    startViewEnd = viewEnd;
  }}
}});

window.addEventListener("mouseup", () => {{ isDragging = false; }});

canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  if (isDragging) {{
    const plotWidth = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;
    const dt = ((mouseX - startDragX) / plotWidth) * (startViewEnd - startViewStart);
    const span = startViewEnd - startViewStart;
    viewStart = Math.max(0, startViewStart - dt);
    viewEnd = Math.min(TOTAL_MAX_TIME, viewStart + span);
    render();
    return;
  }}

  // Tooltip hit test
  const hoverTime = xToTime(mouseX);
  if (hoverTime < viewStart || hoverTime > viewEnd || mouseX < MARGIN_LEFT) {{
    tooltip.style.display = "none";
    return;
  }}

  const sessIdx = Math.floor((mouseY - MARGIN_TOP) / TRACK_GROUP_HEIGHT);
  if (sessIdx < 0 || sessIdx >= SESSIONS.length) {{
    tooltip.style.display = "none";
    return;
  }}

  const sess = SESSIONS[sessIdx];
  const relY = (mouseY - MARGIN_TOP) % TRACK_GROUP_HEIGHT;
  let hit = null;

  const y1 = 5;
  const y2 = y1 + TRACK_H + TRACK_GAP;
  const y3 = y2 + TRACK_H + TRACK_GAP;
  const y4 = y3 + TRACK_H + TRACK_GAP;

  if (relY >= y1 && relY <= y1 + TRACK_H) {{
    const b = sess.macro_blocks.find(blk => hoverTime >= blk.s && hoverTime <= blk.e);
    if (b) {{
      hit = {{
        title: `Upper Macro-State U${{b.m + 1}}`,
        onset: b.s.toFixed(1),
        end: b.e.toFixed(1),
        dur: b.d.toFixed(1),
        bouts: b.b,
        cert: (b.c * 100).toFixed(1) + "%",
        extra: `Session ${{sess.session_idx}} (${{sess.genotype}})`
      }};
    }}
  }} else if (relY >= y2 && relY <= y2 + TRACK_H) {{
    const b = sess.rout_blocks.find(blk => hoverTime >= blk.s && hoverTime <= blk.e);
    if (b) {{
      hit = {{
        title: `Shared Subroutine R${{b.r + 1}}`,
        onset: b.s.toFixed(1),
        end: b.e.toFixed(1),
        dur: b.d.toFixed(1),
        bouts: b.b,
        cert: null,
        extra: `In Upper State U${{b.m + 1}} | Session ${{sess.session_idx}}`
      }};
    }}
  }} else if (relY >= y3 && relY <= y3 + TRACK_H) {{
    const b = sess.syll_blocks.find(blk => hoverTime >= blk.s && hoverTime <= blk.e);
    if (b) {{
      hit = {{
        title: `Syllable ${{b.s_id}} (Stage ${{b.st + 1}})`,
        onset: b.s.toFixed(2),
        end: b.e.toFixed(2),
        dur: b.d.toFixed(2),
        bouts: 1,
        cert: null,
        extra: `Routine R${{b.r + 1}} | Upper U${{b.m + 1}}`
      }};
    }}
  }} else if (relY >= y4 && relY <= y4 + PROB_H) {{
    // Closest prob point
    let closest = sess.prob_curve[0];
    let minD = 99999;
    sess.prob_curve.forEach(pt => {{
      const d = Math.abs(pt.t - hoverTime);
      if (d < minD) {{ minD = d; closest = pt; }}
    }});
    if (closest) {{
      hit = {{
        title: `Posterior Probabilities @ ${{closest.t}}s`,
        onset: (closest.t - 0.5).toFixed(1),
        end: (closest.t + 0.5).toFixed(1),
        dur: "1.0",
        bouts: "-",
        probs: closest.p,
        extra: `Session ${{sess.session_idx}}`
      }};
    }}
  }}

  if (hit) {{
    tooltip.style.display = "block";
    tooltip.style.left = (e.pageX + 15) + "px";
    tooltip.style.top = (e.pageY + 15) + "px";
    
    let probDetails = "";
    if (hit.probs) {{
      probDetails = `
        <div style="margin-top: 4px; border-top: 1px solid #475569; padding-top: 4px;">
          <div><strong style="color:#6C8EBF">P(U1):</strong> ${{ (hit.probs[0]*100).toFixed(1) }}%</div>
          <div><strong style="color:#82B366">P(U2):</strong> ${{ (hit.probs[1]*100).toFixed(1) }}%</div>
          <div><strong style="color:#D6B656">P(U3):</strong> ${{ (hit.probs[2]*100).toFixed(1) }}%</div>
          <div><strong style="color:#B85450">P(U4):</strong> ${{ (hit.probs[3]*100).toFixed(1) }}%</div>
        </div>
      `;
    }}

    tooltip.innerHTML = `
      <div style="font-weight:bold; color:#38bdf8; margin-bottom:3px;">${{hit.title}}</div>
      <div><strong>Time:</strong> ${{hit.onset}}s – ${{hit.end}}s (${{hit.dur}}s)</div>
      ${{hit.bouts !== "-" ? `<div><strong>Bouts:</strong> ${{hit.bouts}}</div>` : ""}}
      ${{hit.cert ? `<div><strong>Mean Certainty:</strong> ${{hit.cert}}</div>` : ""}}
      ${{probDetails}}
      <div style="color:#94a3b8; font-size:10px; margin-top:3px;">${{hit.extra}}</div>
    `;
  }} else {{
    tooltip.style.display = "none";
  }}
}});

canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const hoverTime = xToTime(mouseX);

  const zoomFactor = e.deltaY < 0 ? 0.8 : 1.25;
  const curSpan = viewEnd - viewStart;
  const newSpan = Math.max(10, Math.min(TOTAL_MAX_TIME, curSpan * zoomFactor));

  const mouseRatio = (mouseX - MARGIN_LEFT) / (canvas.width - MARGIN_LEFT - MARGIN_RIGHT);
  viewStart = Math.max(0, hoverTime - mouseRatio * newSpan);
  viewEnd = Math.min(TOTAL_MAX_TIME, viewStart + newSpan);
  render();
}});

document.getElementById("btn-reset").addEventListener("click", () => {{
  viewStart = 0;
  viewEnd = 1800;
  render();
}});

document.getElementById("btn-zoom-600").addEventListener("click", () => {{
  viewStart = 600;
  viewEnd = 720;
  render();
}});

document.getElementById("btn-zoom-in").addEventListener("click", () => {{
  const curSpan = viewEnd - viewStart;
  const center = (viewStart + viewEnd) / 2;
  const newSpan = Math.max(10, curSpan * 0.7);
  viewStart = Math.max(0, center - newSpan / 2);
  viewEnd = Math.min(TOTAL_MAX_TIME, viewStart + newSpan);
  render();
}});

document.getElementById("btn-zoom-out").addEventListener("click", () => {{
  const curSpan = viewEnd - viewStart;
  const center = (viewStart + viewEnd) / 2;
  const newSpan = Math.min(TOTAL_MAX_TIME, curSpan * 1.4);
  viewStart = Math.max(0, center - newSpan / 2);
  viewEnd = Math.min(TOTAL_MAX_TIME, viewStart + newSpan);
  render();
}});

resize();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"✓ Saved 4-State Interactive HTML: {output_path.name}")
