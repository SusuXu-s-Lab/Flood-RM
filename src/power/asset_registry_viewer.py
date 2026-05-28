"""Offline asset-registry viewer for the Marshfield OpenDSS infrastructure.

The local viewer launcher uses this module so the visualization path is
testable outside Jupyter. The generated HTML embeds the selected registry rows
and draws them on a browser canvas, with no deck.gl, WebGL, basemap, or CDN
requirement.
"""

from __future__ import annotations

import colorsys
import csv
import json
import webbrowser
from html import escape
from pathlib import Path


from power.paths import POWER_GRID

DEFAULT_ASSETS_DIR = POWER_GRID / "asset_registry"
DEFAULT_OUTPUT = POWER_GRID / "visualizations" / "asset_registry_canvas.html"
DEFAULT_VIEWER_PORT = 8765


def find_assets_dir(start: Path | None = None) -> Path:
    """Find the generated Asset Registry directory from a notebook or repo cwd."""

    cwd = (start or Path.cwd()).resolve()
    candidates = [
        POWER_GRID / "asset_registry",
        cwd / "asset_registry",
        DEFAULT_ASSETS_DIR,
    ]
    for candidate in candidates:
        if (candidate / "summary.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find locations/marshfield/data/power_grid/asset_registry. "
        "Run python -m power.build_asset_registry first."
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except ValueError:
        return default


def _has_buscoords(row: dict[str, str]) -> bool:
    return row.get("has_buscoords", "true").lower() == "true"


def _evenly_sample_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[0]]

    step = (len(rows) - 1) / (limit - 1)
    return [rows[round(index * step)] for index in range(limit)]


def _select_load_buses_for_overlay(
    rows: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    """Select high-load buses without letting tied loads collapse spatial coverage."""

    valid_rows = [row for row in rows if _has_buscoords(row)]
    if len(valid_rows) <= limit:
        return sorted(valid_rows, key=lambda row: (-_float(row, "load_kw"), row["feeder_id"], row["bus"]))

    ranked = sorted(
        valid_rows,
        key=lambda row: (-_float(row, "load_kw"), row["feeder_id"], row["bus"]),
    )
    cutoff_kw = _float(ranked[limit - 1], "load_kw")
    above_cutoff = [row for row in ranked if _float(row, "load_kw") > cutoff_kw]
    tied_at_cutoff = [row for row in ranked if _float(row, "load_kw") == cutoff_kw]

    return above_cutoff + _evenly_sample_rows(tied_at_cutoff, limit - len(above_cutoff))


def _feeder_color(feeder_id: str) -> list[int]:
    number = int(feeder_id.removeprefix("f")) if feeder_id.startswith("f") else 0
    hue = ((number * 47 + 176) % 360) / 360
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.66, 0.78)
    return [round(red * 255), round(green * 255), round(blue * 255), 215]


def build_asset_registry_payload(assets_dir: Path, load_bus_limit: int = 1000) -> dict:
    """Build a compact browser payload from normalized registry CSV files."""

    summary = json.loads((assets_dir / "summary.json").read_text(encoding="utf-8"))
    feeders = [row["feeder_id"] for row in read_csv_rows(assets_dir / "feeders.csv")]
    buses_raw = read_csv_rows(assets_dir / "buses.csv")
    lines_raw = read_csv_rows(assets_dir / "lines.csv")
    sources_raw = read_csv_rows(assets_dir / "sources.csv")
    transformers_raw = read_csv_rows(assets_dir / "transformers.csv")
    load_buses_raw = read_csv_rows(assets_dir / "load_buses.csv")

    valid_buses = [row for row in buses_raw if row.get("lon") and row.get("lat")]
    if not valid_buses:
        raise ValueError(f"No bus coordinates found in {assets_dir / 'buses.csv'}")

    buses = [
        {
            "name": row["bus"],
            "feeder": row["feeder_id"],
            "position": [_float(row, "lon"), _float(row, "lat")],
            "line_degree": _int(row, "line_degree"),
            "load_kw": _float(row, "load_kw"),
            "color": _feeder_color(row["feeder_id"]),
        }
        for row in valid_buses
    ]
    lines = [
        {
            "name": row["line_name"],
            "feeder": row["feeder_id"],
            "line_class": row["line_class"],
            "phases": _int(row, "phases", 1),
            "length": _float(row, "length"),
            "source": [_float(row, "from_lon"), _float(row, "from_lat")],
            "target": [_float(row, "to_lon"), _float(row, "to_lat")],
            "color": _feeder_color(row["feeder_id"]),
        }
        for row in lines_raw
        if _has_buscoords(row)
    ]
    sources = [
        {
            "name": row["source_name"],
            "feeder": row["feeder_id"],
            "position": [_float(row, "lon"), _float(row, "lat")],
            "basekv": _float(row, "basekv"),
        }
        for row in sources_raw
        if _has_buscoords(row)
    ]
    transformers = [
        {
            "name": row["transformer_name"],
            "feeder": row["feeder_id"],
            "position": [_float(row, "location_lon"), _float(row, "location_lat")],
            "kva": _float(row, "max_kva"),
            "kv": _float(row, "max_kv"),
        }
        for row in transformers_raw
        if _has_buscoords(row)
    ]
    top_load_buses = _select_load_buses_for_overlay(load_buses_raw, load_bus_limit)
    load_buses = [
        {
            "name": row["bus"],
            "feeder": row["feeder_id"],
            "position": [_float(row, "lon"), _float(row, "lat")],
            "load_kw": _float(row, "load_kw"),
            "load_count": _int(row, "load_count"),
        }
        for row in top_load_buses
    ]

    lons = [bus["position"][0] for bus in buses]
    lats = [bus["position"][1] for bus in buses]
    return {
        "feeders": feeders,
        "bounds": {
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
        },
        "counts": summary["outputs"],
        "lines": lines,
        "buses": buses,
        "sources": sources,
        "transformers": transformers,
        "load_buses": load_buses,
    }


def render_asset_registry_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marshfield Asset Registry Viewer</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #172026;
    --muted: #5b6670;
    --panel: #f7f4ed;
    --line: #d6d0c2;
    --source: #c2410c;
    --transformer: #7c3aed;
    --load: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    color: var(--ink);
    background: #e8edf0;
    font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr);
    min-height: 100vh;
  }}
  aside {{
    padding: 18px;
    background: var(--panel);
    border-right: 1px solid var(--line);
    overflow-y: auto;
  }}
  h1 {{
    margin: 0 0 4px;
    font-size: 20px;
    letter-spacing: 0;
  }}
  .subtle {{ color: var(--muted); }}
  .control {{
    display: grid;
    gap: 7px;
    margin-top: 16px;
  }}
  label {{ font-weight: 650; }}
  select, button {{
    width: 100%;
    min-height: 34px;
    border: 1px solid #b9b2a5;
    border-radius: 6px;
    background: #fffdfa;
    color: var(--ink);
    font: inherit;
  }}
  button {{
    cursor: pointer;
    font-weight: 650;
  }}
  .checks {{
    display: grid;
    gap: 8px;
    margin-top: 9px;
  }}
  .checks label {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 16px;
  }}
  .stat {{
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px;
    background: #fffdfa;
  }}
  .stat strong {{
    display: block;
    font-size: 17px;
  }}
  .legend {{
    display: grid;
    gap: 7px;
    margin-top: 16px;
  }}
  .key {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
  }}
  .swatch {{
    width: 26px;
    height: 4px;
    border-radius: 999px;
    background: #555;
  }}
  .dot {{
    width: 11px;
    height: 11px;
    border-radius: 999px;
    background: #555;
  }}
  .map-shell {{
    position: relative;
    min-width: 0;
    background: #f9faf8;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 100vh;
    cursor: grab;
  }}
  canvas.dragging {{ cursor: grabbing; }}
  .hud {{
    position: absolute;
    left: 14px;
    bottom: 14px;
    padding: 7px 9px;
    max-width: min(620px, calc(100% - 28px));
    border: 1px solid rgba(23, 32, 38, 0.18);
    border-radius: 6px;
    background: rgba(255, 253, 250, 0.9);
    color: var(--muted);
    backdrop-filter: blur(8px);
  }}
  @media (max-width: 820px) {{
    main {{ grid-template-columns: 1fr; }}
    aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    canvas {{ height: 70vh; }}
  }}
</style>
</head>
<body>
<main>
  <aside>
    <h1>Marshfield Grid</h1>
    <div class="subtle">Asset Registry infrastructure map. No external map, WebGL, or CDN dependency.</div>

    <div class="control">
      <label for="feeder">Feeder</label>
      <select id="feeder"></select>
    </div>

    <div class="control">
      <label for="kind">Line class</label>
      <select id="kind">
        <option value="all">All line classes</option>
        <option value="overhead">Overhead</option>
        <option value="underground">Underground</option>
        <option value="fuse">Fuse / protective</option>
        <option value="line">Other line</option>
      </select>
    </div>

    <div class="checks">
      <label><input id="showLines" type="checkbox" checked> Lines</label>
      <label><input id="showBuses" type="checkbox"> Buses</label>
      <label><input id="showSources" type="checkbox" checked> Substation sources</label>
      <label><input id="showTransformers" type="checkbox" checked> Transformer pads</label>
      <label><input id="showLoads" type="checkbox" checked> Load buses</label>
    </div>

    <div class="control">
      <button id="reset">Reset view</button>
    </div>

    <div class="stats" id="stats"></div>

    <div class="legend">
      <div class="key"><span class="swatch" style="background:#1f8a70"></span> Feeder-colored line</div>
      <div class="key"><span class="swatch" style="background:#6b7280"></span> Filtered line class</div>
      <div class="key"><span class="dot" style="background:var(--source)"></span> Source</div>
      <div class="key"><span class="dot" style="background:var(--transformer)"></span> Transformer</div>
      <div class="key"><span class="dot" style="background:var(--load)"></span> Large load bus</div>
    </div>
  </aside>

  <section class="map-shell">
    <canvas id="map"></canvas>
    <div class="hud">Drag to pan. Scroll to zoom. This is a lon/lat canvas map drawn from the registry CSVs.</div>
  </section>
</main>
<script>
const DATA = {data};
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const controls = {{
  feeder: document.getElementById("feeder"),
  kind: document.getElementById("kind"),
  showLines: document.getElementById("showLines"),
  showBuses: document.getElementById("showBuses"),
  showSources: document.getElementById("showSources"),
  showTransformers: document.getElementById("showTransformers"),
  showLoads: document.getElementById("showLoads"),
}};
const stats = document.getElementById("stats");
let view = {{...DATA.bounds}};
let dragging = false;
let last = null;

function resetView() {{
  const b = DATA.bounds;
  const lonPad = (b.max_lon - b.min_lon) * 0.05;
  const latPad = (b.max_lat - b.min_lat) * 0.05;
  view = {{
    minLon: b.min_lon - lonPad,
    maxLon: b.max_lon + lonPad,
    minLat: b.min_lat - latPad,
    maxLat: b.max_lat + latPad,
  }};
  draw();
}}

function includeFeeder(item) {{
  return controls.feeder.value === "all" || item.feeder === controls.feeder.value;
}}

function includeKind(line) {{
  return controls.kind.value === "all" || line.line_class === controls.kind.value;
}}

function classColor(line) {{
  if (line.line_class === "underground") return [59, 130, 246, 210];
  if (line.line_class === "fuse") return [107, 114, 128, 150];
  if (line.line_class === "overhead") return [30, 126, 104, 215];
  return [75, 85, 99, 180];
}}

function rgba(color) {{
  return `rgba(${{color[0]}}, ${{color[1]}}, ${{color[2]}}, ${{(color[3] || 255) / 255}})`;
}}

function screenX(lon) {{
  return (lon - view.minLon) * canvas.clientWidth / (view.maxLon - view.minLon);
}}

function screenY(lat) {{
  return (view.maxLat - lat) * canvas.clientHeight / (view.maxLat - view.minLat);
}}

function worldLon(x) {{
  return view.minLon + x * (view.maxLon - view.minLon) / canvas.clientWidth;
}}

function worldLat(y) {{
  return view.maxLat - y * (view.maxLat - view.minLat) / canvas.clientHeight;
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function lineWidth(phase) {{
  if (phase >= 3) return 1.45;
  if (phase === 2) return 1.0;
  return 0.65;
}}

function pointRadius(kind, item) {{
  if (kind === "source") return 5.5;
  if (kind === "transformer") return 2.1;
  const kw = Math.max(0, item.load_kw || 0);
  return Math.max(3.0, Math.min(10.0, 2.5 + Math.sqrt(kw) / 5.5));
}}

function drawPoints(items, color, radiusKind) {{
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(255, 253, 250, 0.85)";
  ctx.lineWidth = 1;
  for (const item of items) {{
    if (!includeFeeder(item)) continue;
    const x = screenX(item.position[0]), y = screenY(item.position[1]);
    if (x < -12 || y < -12 || x > canvas.clientWidth + 12 || y > canvas.clientHeight + 12) continue;
    ctx.beginPath();
    ctx.arc(x, y, pointRadius(radiusKind, item), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }}
}}

function draw() {{
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = "#f9faf8";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  let visibleLines = 0;
  if (controls.showLines.checked) {{
    ctx.lineCap = "round";
    for (const line of DATA.lines) {{
      if (!includeFeeder(line) || !includeKind(line)) continue;
      const ax = screenX(line.source[0]), ay = screenY(line.source[1]);
      const bx = screenX(line.target[0]), by = screenY(line.target[1]);
      if ((ax < -20 && bx < -20) || (ay < -20 && by < -20) ||
          (ax > canvas.clientWidth + 20 && bx > canvas.clientWidth + 20) ||
          (ay > canvas.clientHeight + 20 && by > canvas.clientHeight + 20)) continue;
      visibleLines++;
      const color = controls.kind.value === "all" ? line.color : classColor(line);
      ctx.strokeStyle = rgba(color);
      ctx.globalAlpha = line.line_class === "fuse" ? 0.5 : 0.78;
      ctx.lineWidth = lineWidth(line.phases);
      ctx.setLineDash(line.line_class === "underground" ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }}
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }}

  if (controls.showBuses.checked) {{
    ctx.fillStyle = "rgba(22, 78, 99, 0.46)";
    for (const bus of DATA.buses) {{
      if (!includeFeeder(bus)) continue;
      const x = screenX(bus.position[0]), y = screenY(bus.position[1]);
      if (x < -4 || y < -4 || x > canvas.clientWidth + 4 || y > canvas.clientHeight + 4) continue;
      ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
    }}
  }}

  if (controls.showTransformers.checked) drawPoints(DATA.transformers, "#7c3aed", "transformer");
  if (controls.showLoads.checked) drawPoints(DATA.load_buses, "#b45309", "load");
  if (controls.showSources.checked) drawPoints(DATA.sources, "#c2410c", "source");

  updateStats(visibleLines);
}}

function countVisible(items) {{
  return items.reduce((n, item) => n + (includeFeeder(item) ? 1 : 0), 0);
}}

function updateStats(visibleLines) {{
  const feeder = controls.feeder.value;
  const busCount = feeder === "all" ? DATA.counts["buses.csv"] : countVisible(DATA.buses);
  const sourceCount = feeder === "all" ? DATA.counts["sources.csv"] : countVisible(DATA.sources);
  const txCount = feeder === "all" ? DATA.counts["transformers.csv"] : countVisible(DATA.transformers);
  const loadCount = feeder === "all" ? DATA.load_buses.length : countVisible(DATA.load_buses);
  stats.innerHTML = `
    <div class="stat"><strong>${{busCount.toLocaleString()}}</strong><span>Buses</span></div>
    <div class="stat"><strong>${{visibleLines.toLocaleString()}}</strong><span>Lines visible</span></div>
    <div class="stat"><strong>${{sourceCount.toLocaleString()}}</strong><span>Sources</span></div>
    <div class="stat"><strong>${{txCount.toLocaleString()}}</strong><span>Transformers</span></div>
    <div class="stat"><strong>${{loadCount.toLocaleString()}}</strong><span>Load buses shown</span></div>
    <div class="stat"><strong>${{DATA.feeders.length}}</strong><span>Feeders</span></div>
  `;
}}

function populateControls() {{
  controls.feeder.innerHTML = '<option value="all">All feeders</option>' +
    DATA.feeders.map(f => `<option value="${{f}}">${{f}}</option>`).join("");
  for (const element of Object.values(controls)) element.addEventListener("change", draw);
  document.getElementById("reset").addEventListener("click", resetView);
}}

canvas.addEventListener("mousedown", event => {{
  dragging = true;
  last = {{x: event.clientX, y: event.clientY}};
  canvas.classList.add("dragging");
}});
window.addEventListener("mouseup", () => {{
  dragging = false;
  last = null;
  canvas.classList.remove("dragging");
}});
window.addEventListener("mousemove", event => {{
  if (!dragging || !last) return;
  const dx = event.clientX - last.x;
  const dy = event.clientY - last.y;
  const lonSpan = view.maxLon - view.minLon;
  const latSpan = view.maxLat - view.minLat;
  const lonDelta = dx * lonSpan / canvas.clientWidth;
  const latDelta = dy * latSpan / canvas.clientHeight;
  view.minLon -= lonDelta;
  view.maxLon -= lonDelta;
  view.minLat += latDelta;
  view.maxLat += latDelta;
  last = {{x: event.clientX, y: event.clientY}};
  draw();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  const lon = worldLon(mx);
  const lat = worldLat(my);
  const factor = event.deltaY < 0 ? 0.86 : 1.16;
  const minLon = lon - (lon - view.minLon) * factor;
  const maxLon = lon + (view.maxLon - lon) * factor;
  const minLat = lat - (lat - view.minLat) * factor;
  const maxLat = lat + (view.maxLat - lat) * factor;
  view = {{minLon, maxLon, minLat, maxLat}};
  draw();
}}, {{passive: false}});
window.addEventListener("resize", resize);

populateControls();
resetView();
resize();
</script>
</body>
</html>
"""


def write_asset_registry_viewer(
    assets_dir: Path | None = None,
    output: Path | None = None,
    load_bus_limit: int = 1000,
) -> Path:
    assets = assets_dir or find_assets_dir()
    target = output or DEFAULT_OUTPUT
    payload = build_asset_registry_payload(assets, load_bus_limit=load_bus_limit)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_asset_registry_html(payload), encoding="utf-8")
    return target


def iframe_html(viewer_path: Path, cwd: Path | None = None, height: int = 860) -> str:
    root = (cwd or Path.cwd()).resolve()
    resolved = viewer_path.resolve()
    try:
        src = "/files/" + resolved.relative_to(root).as_posix()
    except ValueError:
        src = resolved.as_uri()
    return f"""
<iframe
  title="Marshfield power grid asset registry viewer"
  src="{escape(src, quote=True)}"
  style="width:100%; height:{height}px; border:0; display:block;"
></iframe>
<div style="margin-top:6px; color:#64748b; font-size:12px; font-family:system-ui, -apple-system, Segoe UI, sans-serif;">
  Viewer HTML: <code>{escape(str(viewer_path), quote=False)}</code>
</div>
"""


def open_viewer_in_browser(
    viewer_path: Path,
    opener=webbrowser.open_new_tab,
) -> str:
    """Open a standalone viewer HTML file in the user's default browser."""

    uri = viewer_path.resolve().as_uri()
    opener(uri)
    return uri


def viewer_urls(host: str = "127.0.0.1", port: int = DEFAULT_VIEWER_PORT) -> dict[str, str]:
    """Return stable local HTTP URLs for the generated registry viewers."""

    base = f"http://{host}:{port}/visualizations"
    return {
        "recommended": f"{base}/asset_registry_canvas.html",
        "deckgl": f"{base}/asset_registry_deckgl.html",
        "grid_map": f"{base}/grid_map.html",
    }


def display_asset_registry_viewer(
    assets_dir: Path | None = None,
    output: Path | None = None,
    load_bus_limit: int = 1000,
    height: int = 860,
):
    """Write and return an IPython HTML iframe for notebook display."""

    from IPython.display import HTML

    viewer_path = write_asset_registry_viewer(assets_dir, output, load_bus_limit=load_bus_limit)
    return HTML(iframe_html(viewer_path, height=height))
