"""Render the current Marshfield OpenDSS infrastructure as a browser map.

The output is a dependency-free HTML file with embedded canvas drawing data.
It intentionally visualizes only the infrastructure present in Layer 1:
network buses, lines, transformer locations, substation sources, and load
hotspots derived from the OpenDSS artifacts.

Run:
    python -m power.visualize_grid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


from power.paths import POWER_GRID

DSS_DIR = POWER_GRID / "derived_opendss"
DEFAULT_OUTPUT = POWER_GRID / "visualizations" / "grid_map.html"
BUS_RE = re.compile(r"\bbus(?:1|2)?=([^ ]+)", re.IGNORECASE)
NAME_RE = re.compile(r"^New\s+([A-Za-z]+)\.([^ ]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Coord:
    lon: float
    lat: float


def feeder_id(bus_name: str) -> str:
    return bus_name.split("_", 1)[0]


def strip_nodes(bus_ref: str) -> str:
    return bus_ref.split(".", 1)[0]


def parse_name(line: str) -> str:
    match = NAME_RE.search(line)
    return match.group(2) if match else ""


def parse_value(line: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}=([^ ]+)", line, re.IGNORECASE)
    return match.group(1) if match else None


def parse_float(line: str, key: str, default: float = 0.0) -> float:
    raw = parse_value(line, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def read_buscoords(path: Path) -> dict[str, Coord]:
    coords: dict[str, Coord] = {}
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            coords[row[0]] = Coord(float(row[1]), float(row[2]))
    return coords


def classify_line(linecode: str) -> str:
    lowered = linecode.lower()
    if "_ug_" in lowered:
        return "underground"
    if "_oh_" in lowered:
        return "overhead"
    if "fuse" in lowered:
        return "fuse"
    return "line"


def read_lines(path: Path, coords: dict[str, Coord]) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("New Line."):
            continue
        buses = [strip_nodes(bus) for bus in BUS_RE.findall(line)]
        if len(buses) < 2 or buses[0] not in coords or buses[1] not in coords:
            continue
        linecode = parse_value(line, "linecode") or ""
        a = coords[buses[0]]
        b = coords[buses[1]]
        records.append(
            {
                "name": parse_name(line),
                "feeder": feeder_id(buses[0]),
                "from": buses[0],
                "to": buses[1],
                "phase": int(parse_float(line, "phases", 0)),
                "kind": classify_line(linecode),
                "linecode": linecode,
                "length_m": parse_float(line, "length", 0.0),
                "a_lon": a.lon,
                "a_lat": a.lat,
                "b_lon": b.lon,
                "b_lat": b.lat,
            }
        )
    return records


def read_sources(master_path: Path, extra_path: Path, coords: dict[str, Coord]) -> list[dict]:
    records = []
    for path in (master_path, extra_path):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not (line.startswith("New Circuit.") or line.startswith("New Vsource.")):
                continue
            bus = parse_value(line, "bus1")
            if bus is None:
                continue
            bus_name = strip_nodes(bus)
            coord = coords.get(bus_name)
            if coord is None:
                continue
            records.append(
                {
                    "name": parse_name(line),
                    "bus": bus_name,
                    "feeder": feeder_id(bus_name),
                    "lon": coord.lon,
                    "lat": coord.lat,
                }
            )
    return records


def read_transformers(path: Path, coords: dict[str, Coord]) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("New Transformer."):
            continue
        buses = [strip_nodes(bus) for bus in BUS_RE.findall(line)]
        located = [bus for bus in buses if bus in coords]
        if not located:
            continue
        # Last winding is the lowest-voltage side in the generated OpenDSS.
        bus = located[-1]
        coord = coords[bus]
        records.append(
            {
                "name": parse_name(line),
                "bus": bus,
                "feeder": feeder_id(bus),
                "lon": coord.lon,
                "lat": coord.lat,
            }
        )
    return records


def read_load_hotspots(path: Path, coords: dict[str, Coord], limit: int) -> list[dict]:
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.startswith("New Load."):
            continue
        bus = parse_value(line, "bus1")
        if bus is None:
            continue
        bus_name = strip_nodes(bus)
        if bus_name not in coords:
            continue
        totals[bus_name] += parse_float(line, "kW", 0.0)
        counts[bus_name] += 1

    ranked = select_hotspot_buses(totals, limit)
    records = []
    for bus in ranked:
        coord = coords[bus]
        records.append(
            {
                "bus": bus,
                "feeder": feeder_id(bus),
                "kw": round(totals[bus], 3),
                "loads": counts[bus],
                "lon": coord.lon,
                "lat": coord.lat,
            }
        )
    return records


def evenly_sample(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]

    step = (len(items) - 1) / (limit - 1)
    return [items[round(index * step)] for index in range(limit)]


def select_hotspot_buses(totals: dict[str, float], limit: int) -> list[str]:
    ranked = sorted(totals, key=lambda bus: (-totals[bus], feeder_id(bus), bus))
    if len(ranked) <= limit:
        return ranked

    cutoff_kw = totals[ranked[limit - 1]]
    above_cutoff = [bus for bus in ranked if totals[bus] > cutoff_kw]
    tied_at_cutoff = [bus for bus in ranked if totals[bus] == cutoff_kw]
    return above_cutoff + evenly_sample(tied_at_cutoff, limit - len(above_cutoff))


def project_records(payload: dict) -> None:
    lon_values = [bus["lon"] for bus in payload["buses"]]
    lat_values = [bus["lat"] for bus in payload["buses"]]
    min_lon, max_lon = min(lon_values), max(lon_values)
    min_lat, max_lat = min(lat_values), max(lat_values)
    mean_lat = math.radians((min_lat + max_lat) / 2)
    x_span = max((max_lon - min_lon) * math.cos(mean_lat), 1e-9)
    y_span = max(max_lat - min_lat, 1e-9)
    width = 1400.0
    height = max(520.0, width * y_span / x_span)
    margin = 36.0

    def xy(lon: float, lat: float) -> tuple[float, float]:
        x = margin + ((lon - min_lon) * math.cos(mean_lat) / x_span) * width
        y = margin + ((max_lat - lat) / y_span) * height
        return round(x, 2), round(y, 2)

    for bus in payload["buses"]:
        bus["x"], bus["y"] = xy(bus.pop("lon"), bus.pop("lat"))
    for line in payload["lines"]:
        line["ax"], line["ay"] = xy(line.pop("a_lon"), line.pop("a_lat"))
        line["bx"], line["by"] = xy(line.pop("b_lon"), line.pop("b_lat"))
    for layer in ("sources", "transformers", "load_hotspots"):
        for item in payload[layer]:
            item["x"], item["y"] = xy(item.pop("lon"), item.pop("lat"))

    payload["world"] = {
        "width": round(width + margin * 2, 2),
        "height": round(height + margin * 2, 2),
        "bounds": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
    }


def build_payload(dss_dir: Path, load_hotspot_limit: int) -> dict:
    coords = read_buscoords(dss_dir / "buscoords.csv")
    lines = read_lines(dss_dir / "lines.dss", coords)
    buses = [
        {"name": name, "feeder": feeder_id(name), "lon": coord.lon, "lat": coord.lat}
        for name, coord in coords.items()
    ]
    payload = {
        "buses": buses,
        "lines": lines,
        "sources": read_sources(dss_dir / "master.dss", dss_dir / "vsources_extra.dss", coords),
        "transformers": read_transformers(dss_dir / "transformers.dss", coords),
        "load_hotspots": read_load_hotspots(dss_dir / "loads.dss", coords, load_hotspot_limit),
        "feeders": sorted({bus["feeder"] for bus in buses}),
        "counts": {
            "buses": len(buses),
            "lines": len(lines),
            "sources": 0,
            "transformers": 0,
            "load_hotspots": 0,
        },
    }
    payload["counts"]["sources"] = len(payload["sources"])
    payload["counts"]["transformers"] = len(payload["transformers"])
    payload["counts"]["load_hotspots"] = len(payload["load_hotspots"])
    project_records(payload)
    return payload


def render_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marshfield OpenDSS Infrastructure</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #172026;
    --muted: #5b6670;
    --panel: #f7f4ed;
    --line: #d6d0c2;
    --accent: #0f766e;
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
    max-width: min(560px, calc(100% - 28px));
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
    <div class="subtle">Layer 1 OpenDSS infrastructure, generated from SHIFT GDM outputs.</div>

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
      <label><input id="showLoads" type="checkbox" checked> Load hotspots</label>
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
    <div class="hud">Drag to pan. Scroll to zoom. The map is schematic over lon/lat geometry; no basemap is required.</div>
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
let view = {{x: 0, y: 0, w: DATA.world.width, h: DATA.world.height}};
let dragging = false;
let last = null;

function feederColor(feeder) {{
  const n = Number(feeder.replace(/^f/, ""));
  const hue = (n * 47 + 176) % 360;
  return `hsl(${{hue}} 66% 36%)`;
}}

function screenX(x) {{ return (x - view.x) * canvas.clientWidth / view.w; }}
function screenY(y) {{ return (y - view.y) * canvas.clientHeight / view.h; }}
function worldX(x) {{ return view.x + x * view.w / canvas.clientWidth; }}
function worldY(y) {{ return view.y + y * view.h / canvas.clientHeight; }}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function resetView() {{
  const pad = 25;
  view = {{x: -pad, y: -pad, w: DATA.world.width + pad * 2, h: DATA.world.height + pad * 2}};
  draw();
}}

function includeFeeder(item) {{
  return controls.feeder.value === "all" || item.feeder === controls.feeder.value;
}}

function includeKind(line) {{
  return controls.kind.value === "all" || line.kind === controls.kind.value;
}}

function lineWidth(phase) {{
  if (phase >= 3) return 1.45;
  if (phase === 2) return 1.0;
  return 0.65;
}}

function pointRadius(kind, item) {{
  if (kind === "source") return 5.5;
  if (kind === "transformer") return 2.1;
  const kw = Math.max(0, item.kw || 0);
  return Math.max(3.0, Math.min(10.0, 2.5 + Math.sqrt(kw) / 5.5));
}}

function drawPoints(items, color, radiusKind) {{
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(255, 253, 250, 0.85)";
  ctx.lineWidth = 1;
  for (const item of items) {{
    if (!includeFeeder(item)) continue;
    const x = screenX(item.x), y = screenY(item.y);
    if (x < -12 || y < -12 || x > canvas.clientWidth + 12 || y > canvas.clientHeight + 12) continue;
    ctx.beginPath();
    ctx.arc(x, y, pointRadius(radiusKind, item), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }}
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = "#f9faf8";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  let visibleLines = 0;
  if (controls.showLines.checked) {{
    ctx.lineCap = "round";
    for (const line of DATA.lines) {{
      if (!includeFeeder(line) || !includeKind(line)) continue;
      const ax = screenX(line.ax), ay = screenY(line.ay);
      const bx = screenX(line.bx), by = screenY(line.by);
      if ((ax < -20 && bx < -20) || (ay < -20 && by < -20) ||
          (ax > canvas.clientWidth + 20 && bx > canvas.clientWidth + 20) ||
          (ay > canvas.clientHeight + 20 && by > canvas.clientHeight + 20)) continue;
      visibleLines++;
      ctx.strokeStyle = feederColor(line.feeder);
      ctx.globalAlpha = line.kind === "fuse" ? 0.5 : 0.78;
      ctx.lineWidth = lineWidth(line.phase);
      ctx.setLineDash(line.kind === "underground" ? [4, 3] : []);
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
      const x = screenX(bus.x), y = screenY(bus.y);
      if (x < -4 || y < -4 || x > canvas.clientWidth + 4 || y > canvas.clientHeight + 4) continue;
      ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
    }}
  }}

  if (controls.showTransformers.checked) drawPoints(DATA.transformers, "#7c3aed", "transformer");
  if (controls.showLoads.checked) drawPoints(DATA.load_hotspots, "#b45309", "load");
  if (controls.showSources.checked) drawPoints(DATA.sources, "#c2410c", "source");

  updateStats(visibleLines);
}}

function countVisible(items) {{
  return items.reduce((n, item) => n + (includeFeeder(item) ? 1 : 0), 0);
}}

function updateStats(visibleLines) {{
  const feeder = controls.feeder.value;
  const busCount = feeder === "all" ? DATA.counts.buses : countVisible(DATA.buses);
  const sourceCount = feeder === "all" ? DATA.counts.sources : countVisible(DATA.sources);
  const txCount = feeder === "all" ? DATA.counts.transformers : countVisible(DATA.transformers);
  const loadCount = feeder === "all" ? DATA.counts.load_hotspots : countVisible(DATA.load_hotspots);
  stats.innerHTML = `
    <div class="stat"><strong>${{busCount.toLocaleString()}}</strong><span>Buses</span></div>
    <div class="stat"><strong>${{visibleLines.toLocaleString()}}</strong><span>Lines visible</span></div>
    <div class="stat"><strong>${{sourceCount.toLocaleString()}}</strong><span>Sources</span></div>
    <div class="stat"><strong>${{txCount.toLocaleString()}}</strong><span>Transformers</span></div>
    <div class="stat"><strong>${{loadCount.toLocaleString()}}</strong><span>Load hotspots</span></div>
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
  view.x -= dx * view.w / canvas.clientWidth;
  view.y -= dy * view.h / canvas.clientHeight;
  last = {{x: event.clientX, y: event.clientY}};
  draw();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  const wx = worldX(mx);
  const wy = worldY(my);
  const factor = event.deltaY < 0 ? 0.86 : 1.16;
  view.w *= factor;
  view.h *= factor;
  view.x = wx - mx * view.w / canvas.clientWidth;
  view.y = wy - my * view.h / canvas.clientHeight;
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


def write_html(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dss-dir", type=Path, default=DSS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--load-hotspots",
        type=int,
        default=250,
        help="Number of largest load buses to overlay.",
    )
    args = parser.parse_args()

    payload = build_payload(args.dss_dir, args.load_hotspots)
    write_html(payload, args.output)

    counts = payload["counts"]
    print(f"Wrote {args.output}")
    print(
        "Mapped "
        f"{counts['buses']:,} buses, {counts['lines']:,} lines, "
        f"{counts['sources']:,} sources, {counts['transformers']:,} transformers, "
        f"and {counts['load_hotspots']:,} load hotspots."
    )


if __name__ == "__main__":
    main()
