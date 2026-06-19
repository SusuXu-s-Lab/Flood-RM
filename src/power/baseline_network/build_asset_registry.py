"""Build a normalized asset registry from generated OpenDSS artifacts.

This script is intentionally deterministic: it parses the named properties
emitted by the SHIFT -> GDM -> DiTTo -> OpenDSS pipeline
(`notebooks/regions/marshfield/grid_network/base_network.ipynb`) and writes
CSV tables that later layers can reuse. It does not synthesize DERs,
controllable switches, sensors, fragility states, or any other assets.

Input layout (per DiTTo's OpenDSS writer):
    <dss_dir>/<region_name>/Master.dss
    <dss_dir>/<region_name>/Lines.dss
    <dss_dir>/<region_name>/Loads.dss
    <dss_dir>/<region_name>/Transformers.dss   (when transformers exist)
    <dss_dir>/<region_name>/BusCoords.dss
    <dss_dir>/<region_name>/LineCodes.dss

Each `<region_name>` becomes a feeder. Bus names parsed from a region are
prefixed with `<region_name>__` so the global registry is unambiguous when
two regions happen to use the same bus identifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


from power.artifacts import parse_float, parse_int
from power.artifacts import POWER_GRID

DSS_DIR = POWER_GRID / "derived_opendss"
DEFAULT_OUTPUT_DIR = POWER_GRID / "asset_registry"

OBJECT_RE = re.compile(r"^New\s+([A-Za-z]+)\.([^ ]+)", re.IGNORECASE)
PROPERTY_RE = re.compile(r"(?<!%)\b([A-Za-z][A-Za-z0-9_]*)=([^ ]+)")
PERCENT_PROPERTY_RE = re.compile(r"\B(%[A-Za-z][A-Za-z0-9_]*)=([^ ]+)")
LIST_PROPERTY_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=([\(\[])([^\)\]]*)[\)\]]")
SETBUSXY_RE = re.compile(r"^SetBusXY\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)

# Feeder-scope prefix; chosen to be DSS-safe-ish and visually distinct.
FEEDER_SEP = "__"

DITTO_FILE_NAMES = {
    "master": "Master.dss",
    "lines": "Lines.dss",
    "loads": "Loads.dss",
    "transformers": "Transformers.dss",
    "buscoords": "BusCoords.dss",
    "linecodes": "LineCodes.dss",
}


@dataclass(frozen=True)
class Coord:
    lon: float
    lat: float


@dataclass(frozen=True)
class DssObject:
    class_name: str
    name: str
    properties: dict[str, str]
    repeated: dict[str, list[str]]
    source_file: str
    source_line: int
    raw: str


def qualify_bus(feeder_id: str, bus: str) -> str:
    return f"{feeder_id}{FEEDER_SEP}{bus}" if bus else ""


def split_bus_ref(bus_ref: str | None) -> tuple[str, str]:
    if not bus_ref:
        return "", ""
    parts = bus_ref.split(".")
    return parts[0], ".".join(parts[1:])


def fmt_float(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_properties(line: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse DSS ``Key=Value`` properties on one line.

    Supports list-style values (``Key=(a, b, c)`` or ``Key=[a, b, c]``) used
    by OpenDSS Transformer / XfmrCode for Buses, kVs, kVAs, Conns, Taps,
    %Rs. List items expand under the (lower-cased) key in the repeated
    dict; the scalar value in ``properties`` is the last list element.
    """
    repeated: dict[str, list[str]] = defaultdict(list)
    masked = list(line)
    for match in LIST_PROPERTY_RE.finditer(line):
        key = match.group(1).lower()
        inner = match.group(3)
        items = [item.strip() for item in inner.split(",") if item.strip()]
        repeated[key].extend(items)
        for i in range(match.start(), match.end()):
            masked[i] = " "
    masked_line = "".join(masked)
    for key, value in PROPERTY_RE.findall(masked_line):
        repeated[key.lower()].append(value.strip(","))
    for key, value in PERCENT_PROPERTY_RE.findall(masked_line):
        repeated[key.lower()].append(value.strip(","))
    props = {key: values[-1] for key, values in repeated.items()}
    return props, dict(repeated)


def parse_dss_file(path: Path, class_names: set[str] | None = None) -> list[DssObject]:
    objects: list[DssObject] = []
    if not path.exists():
        return objects
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        match = OBJECT_RE.search(line)
        if match is None:
            continue
        class_name = match.group(1)
        if class_names is not None and class_name.lower() not in class_names:
            continue
        props, repeated = parse_properties(line)
        objects.append(
            DssObject(
                class_name=class_name,
                name=match.group(2),
                properties=props,
                repeated=repeated,
                source_file=path.name,
                source_line=line_no,
                raw=line,
            )
        )
    return objects


def read_buscoords_dss(path: Path, feeder_id: str) -> dict[str, Coord]:
    """Parse `SetBusXY <bus> <lon> <lat>` lines from a DiTTo BusCoords.dss."""
    coords: dict[str, Coord] = {}
    if not path.exists():
        return coords
    for raw in path.read_text().splitlines():
        match = SETBUSXY_RE.match(raw.strip())
        if match is None:
            continue
        bus_raw, x, y = match.group(1), match.group(2), match.group(3)
        try:
            coords[qualify_bus(feeder_id, bus_raw)] = Coord(float(x), float(y))
        except ValueError:
            continue
    return coords


def iter_feeder_dirs(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (feeder_id, region_dir) for every region subdirectory under `root`.

    A region directory is anything directly under `root` that contains a
    Master.dss. The directory name is taken as the feeder_id.
    """
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / DITTO_FILE_NAMES["master"]).exists():
            yield child.name, child


def classify_line(linecode: str) -> str:
    lowered = linecode.lower()
    if "_ug_" in lowered:
        return "underground"
    if "_oh_" in lowered:
        return "overhead"
    if "fuse" in lowered:
        return "fuse"
    return "line"


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def build_lines_for_region(
    feeder_id: str, region_dir: Path, coords: dict[str, Coord]
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    degree: Counter[str] = Counter()
    for obj in parse_dss_file(region_dir / DITTO_FILE_NAMES["lines"], {"line"}):
        bus1_raw, nodes1 = split_bus_ref(obj.properties.get("bus1"))
        bus2_raw, nodes2 = split_bus_ref(obj.properties.get("bus2"))
        bus1 = qualify_bus(feeder_id, bus1_raw)
        bus2 = qualify_bus(feeder_id, bus2_raw)
        if bus1:
            degree[bus1] += 1
        if bus2:
            degree[bus2] += 1
        coord1 = coords.get(bus1)
        coord2 = coords.get(bus2)
        linecode = obj.properties.get("linecode", "")
        rows.append(
            {
                "line_name": obj.name,
                "feeder_id": feeder_id,
                "from_bus": bus1,
                "from_nodes": nodes1,
                "from_lon": fmt_float(coord1.lon if coord1 else None),
                "from_lat": fmt_float(coord1.lat if coord1 else None),
                "to_bus": bus2,
                "to_nodes": nodes2,
                "to_lon": fmt_float(coord2.lon if coord2 else None),
                "to_lat": fmt_float(coord2.lat if coord2 else None),
                "phases": fmt_int(parse_int(obj.properties.get("phases"))),
                "linecode": linecode,
                "line_class": classify_line(linecode),
                "length": fmt_float(parse_float(obj.properties.get("length")), 4),
                "units": obj.properties.get("units", ""),
                "has_buscoords": str(coord1 is not None and coord2 is not None).lower(),
                "source_file": f"{feeder_id}/{obj.source_file}",
                "source_line": str(obj.source_line),
            }
        )
    return rows, degree


def build_loads_for_region(
    feeder_id: str, region_dir: Path, coords: dict[str, Coord]
) -> tuple[list[dict[str, str]], dict[str, dict[str, float | int]]]:
    rows: list[dict[str, str]] = []
    by_bus: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"load_count": 0, "total_kw": 0.0, "total_kvar": 0.0}
    )
    for obj in parse_dss_file(region_dir / DITTO_FILE_NAMES["loads"], {"load"}):
        bus_raw, nodes = split_bus_ref(obj.properties.get("bus1"))
        bus = qualify_bus(feeder_id, bus_raw)
        coord = coords.get(bus)
        kw = parse_float(obj.properties.get("kw"), 0.0) or 0.0
        kvar = parse_float(obj.properties.get("kvar"), 0.0) or 0.0
        by_bus[bus]["load_count"] = int(by_bus[bus]["load_count"]) + 1
        by_bus[bus]["total_kw"] = float(by_bus[bus]["total_kw"]) + kw
        by_bus[bus]["total_kvar"] = float(by_bus[bus]["total_kvar"]) + kvar
        rows.append(
            {
                "load_name": obj.name,
                "feeder_id": feeder_id,
                "bus": bus,
                "nodes": nodes,
                "lon": fmt_float(coord.lon if coord else None),
                "lat": fmt_float(coord.lat if coord else None),
                "phases": fmt_int(parse_int(obj.properties.get("phases"))),
                "conn": obj.properties.get("conn", ""),
                "kv": fmt_float(parse_float(obj.properties.get("kv")), 4),
                "kw": fmt_float(kw, 4),
                "kvar": fmt_float(kvar, 4),
                "model": obj.properties.get("model", ""),
                "has_buscoords": str(coord is not None).lower(),
                "source_file": f"{feeder_id}/{obj.source_file}",
                "source_line": str(obj.source_line),
            }
        )
    return rows, by_bus


def transformer_windings(
    obj: DssObject,
    xfmrcode_index: dict[str, DssObject] | None = None,
) -> list[dict[str, str | float | None]]:
    """Extract one row per Transformer winding.

    Reads ``Buses=(...)`` / ``bus=`` and ``kVs=[...]`` / ``kv=`` etc. from
    the Transformer line. If kV / kVA / conn entries are absent and the
    Transformer references an ``XfmrCode``, those properties are resolved
    from the referenced XfmrCode line.
    """
    buses = obj.repeated.get("buses") or obj.repeated.get("bus", [])
    kvs = obj.repeated.get("kvs") or obj.repeated.get("kv", [])
    kvas = obj.repeated.get("kvas") or obj.repeated.get("kva", [])
    conns = obj.repeated.get("conns") or obj.repeated.get("conn", [])
    xfmrcode_name = obj.properties.get("xfmrcode")
    if xfmrcode_index and xfmrcode_name:
        ref = xfmrcode_index.get(xfmrcode_name.lower())
        if ref is not None:
            if not kvs:
                kvs = ref.repeated.get("kvs") or ref.repeated.get("kv", [])
            if not kvas:
                kvas = ref.repeated.get("kvas") or ref.repeated.get("kva", [])
            if not conns:
                conns = ref.repeated.get("conns") or ref.repeated.get("conn", [])
    count = max(len(buses), len(kvs), len(kvas), len(conns))
    out: list[dict[str, str | float | None]] = []
    for i in range(count):
        bus_ref = buses[i] if i < len(buses) else ""
        bus, nodes = split_bus_ref(bus_ref)
        out.append(
            {
                "bus": bus,
                "nodes": nodes,
                "kv": parse_float(kvs[i] if i < len(kvs) else None),
                "kva": parse_float(kvas[i] if i < len(kvas) else None),
                "conn": conns[i] if i < len(conns) else "",
            }
        )
    return out


def transformer_windings_from_text(
    *, transformer_line: str, xfmrcode_lines: list[str] | None = None
) -> list[dict[str, str | float | None]]:
    """Test helper: parse a Transformer line plus optional XfmrCode lines."""
    xfmrcode_index: dict[str, DssObject] = {}
    for line in xfmrcode_lines or []:
        match = OBJECT_RE.search(line)
        if match is None:
            continue
        props, repeated = parse_properties(line)
        xfmrcode_index[match.group(2).lower()] = DssObject(
            class_name=match.group(1),
            name=match.group(2),
            properties=props,
            repeated=repeated,
            source_file="",
            source_line=0,
            raw=line,
        )
    match = OBJECT_RE.search(transformer_line)
    assert match is not None, f"transformer line did not match: {transformer_line!r}"
    props, repeated = parse_properties(transformer_line)
    obj = DssObject(
        class_name=match.group(1),
        name=match.group(2),
        properties=props,
        repeated=repeated,
        source_file="",
        source_line=0,
        raw=transformer_line,
    )
    return transformer_windings(obj, xfmrcode_index=xfmrcode_index)


def build_transformers_for_region(
    feeder_id: str, region_dir: Path, coords: dict[str, Coord]
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    by_bus: Counter[str] = Counter()
    # Transformers.dss also defines the XfmrCode entries that carry the
    # actual kV/kVA values; index them so transformer_windings can resolve.
    xfmrcode_objs = parse_dss_file(
        region_dir / DITTO_FILE_NAMES["transformers"], {"xfmrcode"}
    )
    xfmrcode_index = {obj.name.lower(): obj for obj in xfmrcode_objs}
    for obj in parse_dss_file(region_dir / DITTO_FILE_NAMES["transformers"], {"transformer"}):
        windings = transformer_windings(obj, xfmrcode_index=xfmrcode_index)
        buses = [qualify_bus(feeder_id, str(w["bus"])) for w in windings if w["bus"]]
        for bus in set(buses):
            by_bus[bus] += 1
        location_bus = buses[-1] if buses else ""
        location_coord = coords.get(location_bus)
        kva_values = [w["kva"] for w in windings if isinstance(w["kva"], float)]
        kv_values = [w["kv"] for w in windings if isinstance(w["kv"], float)]
        rows.append(
            {
                "transformer_name": obj.name,
                "feeder_id": feeder_id,
                "location_bus": location_bus,
                "location_basis": "last_winding_bus",
                "location_lon": fmt_float(location_coord.lon if location_coord else None),
                "location_lat": fmt_float(location_coord.lat if location_coord else None),
                "primary_bus": buses[0] if buses else "",
                "winding_buses": ",".join(buses),
                "winding_count": fmt_int(parse_int(obj.properties.get("windings"))),
                "phases": fmt_int(parse_int(obj.properties.get("phases"))),
                "max_kv": fmt_float(max(kv_values) if kv_values else None, 4),
                "min_kv": fmt_float(min(kv_values) if kv_values else None, 4),
                "max_kva": fmt_float(max(kva_values) if kva_values else None, 4),
                "pct_loadloss": fmt_float(parse_float(obj.properties.get("%loadloss")), 6),
                "pct_noloadloss": fmt_float(parse_float(obj.properties.get("%noloadloss")), 6),
                "has_buscoords": str(location_coord is not None).lower(),
                "source_file": f"{feeder_id}/{obj.source_file}",
                "source_line": str(obj.source_line),
            }
        )
    return rows, by_bus


def build_sources_for_region(
    feeder_id: str, region_dir: Path, coords: dict[str, Coord]
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Sources (Circuit / Vsource) are declared inside each Master.dss."""
    rows: list[dict[str, str]] = []
    by_bus: Counter[str] = Counter()
    for obj in parse_dss_file(region_dir / DITTO_FILE_NAMES["master"], {"circuit", "vsource"}):
        bus_raw, nodes = split_bus_ref(obj.properties.get("bus1"))
        bus = qualify_bus(feeder_id, bus_raw)
        coord = coords.get(bus)
        if bus:
            by_bus[bus] += 1
        rows.append(
            {
                "source_name": obj.name,
                "source_class": obj.class_name,
                "feeder_id": feeder_id,
                "bus": bus,
                "nodes": nodes,
                "lon": fmt_float(coord.lon if coord else None),
                "lat": fmt_float(coord.lat if coord else None),
                "phases": fmt_int(parse_int(obj.properties.get("phases"))),
                "basekv": fmt_float(parse_float(obj.properties.get("basekv")), 4),
                "pu": fmt_float(parse_float(obj.properties.get("pu")), 6),
                "angle": fmt_float(parse_float(obj.properties.get("angle")), 6),
                "r1": fmt_float(parse_float(obj.properties.get("r1")), 6),
                "x1": fmt_float(parse_float(obj.properties.get("x1")), 6),
                "r0": fmt_float(parse_float(obj.properties.get("r0")), 6),
                "x0": fmt_float(parse_float(obj.properties.get("x0")), 6),
                "has_buscoords": str(coord is not None).lower(),
                "source_file": f"{feeder_id}/{obj.source_file}",
                "source_line": str(obj.source_line),
            }
        )
    return rows, by_bus


def build_buses(
    coords: dict[str, Coord],
    bus_to_feeder: dict[str, str],
    degree: Counter[str],
    loads_by_bus: dict[str, dict[str, float | int]],
    transformers_by_bus: Counter[str],
    sources_by_bus: Counter[str],
) -> list[dict[str, str]]:
    rows = []
    for bus, coord in coords.items():
        load = loads_by_bus.get(bus, {})
        rows.append(
            {
                "bus": bus,
                "feeder_id": bus_to_feeder.get(bus, ""),
                "lon": fmt_float(coord.lon),
                "lat": fmt_float(coord.lat),
                "line_degree": str(degree.get(bus, 0)),
                "load_count": str(int(load.get("load_count", 0))),
                "load_kw": fmt_float(float(load.get("total_kw", 0.0)), 4),
                "load_kvar": fmt_float(float(load.get("total_kvar", 0.0)), 4),
                "transformer_count": str(transformers_by_bus.get(bus, 0)),
                "source_count": str(sources_by_bus.get(bus, 0)),
            }
        )
    rows.sort(key=lambda row: (row["feeder_id"], row["bus"]))
    return rows


def build_load_buses(
    coords: dict[str, Coord],
    bus_to_feeder: dict[str, str],
    loads_by_bus: dict[str, dict[str, float | int]],
) -> list[dict[str, str]]:
    rows = []
    for bus, values in loads_by_bus.items():
        coord = coords.get(bus)
        rows.append(
            {
                "bus": bus,
                "feeder_id": bus_to_feeder.get(bus, ""),
                "lon": fmt_float(coord.lon if coord else None),
                "lat": fmt_float(coord.lat if coord else None),
                "load_count": str(int(values["load_count"])),
                "load_kw": fmt_float(float(values["total_kw"]), 4),
                "load_kvar": fmt_float(float(values["total_kvar"]), 4),
                "has_buscoords": str(coord is not None).lower(),
            }
        )
    rows.sort(key=lambda row: (row["feeder_id"], row["bus"]))
    return rows


def build_feeders(
    bus_rows: list[dict[str, str]],
    line_rows: list[dict[str, str]],
    transformer_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    load_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    feeder_ids = sorted({row["feeder_id"] for row in bus_rows if row["feeder_id"]})
    by_feeder = {
        feeder: {
            "feeder_id": feeder,
            "bus_count": 0,
            "line_count": 0,
            "transformer_count": 0,
            "source_count": 0,
            "load_count": 0,
            "load_kw": 0.0,
            "load_kvar": 0.0,
        }
        for feeder in feeder_ids
    }
    for row in bus_rows:
        by_feeder[row["feeder_id"]]["bus_count"] += 1
    for row in line_rows:
        by_feeder[row["feeder_id"]]["line_count"] += 1
    for row in transformer_rows:
        by_feeder[row["feeder_id"]]["transformer_count"] += 1
    for row in source_rows:
        by_feeder[row["feeder_id"]]["source_count"] += 1
    for row in load_rows:
        target = by_feeder[row["feeder_id"]]
        target["load_count"] += 1
        target["load_kw"] += float(row["kw"] or 0.0)
        target["load_kvar"] += float(row["kvar"] or 0.0)
    return [
        {
            "feeder_id": feeder,
            "bus_count": str(values["bus_count"]),
            "line_count": str(values["line_count"]),
            "transformer_count": str(values["transformer_count"]),
            "source_count": str(values["source_count"]),
            "load_count": str(values["load_count"]),
            "load_kw": fmt_float(values["load_kw"], 4),
            "load_kvar": fmt_float(values["load_kvar"], 4),
        }
        for feeder, values in by_feeder.items()
    ]


def write_summary(
    path: Path,
    feeder_files: list[Path],
    csv_counts: dict[str, int],
    warnings: list[str],
) -> None:
    summary = {
        "method": "deterministic extraction from generated OpenDSS named properties",
        "methodology_sources": [
            {
                "title": "OpenDSS Parameters",
                "url": "https://opendss.epri.com/Parameters.html",
                "used_for": "Named property parsing with key=value DSS parameters.",
            },
            {
                "title": "DSS-Extensions Line object format",
                "url": "https://dss-extensions.org/dss-format/Line.html",
                "used_for": "Line Bus1, Bus2, LineCode, Length, Phases, and Units fields.",
            },
            {
                "title": "DSS-Extensions Load object format",
                "url": "https://dss-extensions.org/dss-format/Load.html",
                "used_for": "Load Bus1, Phases, Conn, kV, kW, kvar, and Model fields.",
            },
            {
                "title": "OpenDSS Transformer object",
                "url": "https://opendss.epri.com/Transformer1.html",
                "used_for": "Transformer winding structure and wdg-selected winding properties.",
            },
        ],
        "source_layout": "per-feeder DiTTo OpenDSS output (DSS_DIR/<feeder>/<File>.dss)",
        "inputs": {
            f"{path.parent.name}/{path.name}": {"sha256": sha256(path)}
            for path in feeder_files
        },
        "outputs": csv_counts,
        "warnings": warnings,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def build_registry(dss_dir: Path, output_dir: Path) -> dict[str, int]:
    feeders = list(iter_feeder_dirs(dss_dir))
    if not feeders:
        raise FileNotFoundError(
            f"No DiTTo-style per-feeder directories under {dss_dir}. "
            f"Run base_network.ipynb to produce <feeder>/Master.dss subdirs."
        )

    coords: dict[str, Coord] = {}
    bus_to_feeder: dict[str, str] = {}
    line_rows: list[dict[str, str]] = []
    load_rows: list[dict[str, str]] = []
    transformer_rows: list[dict[str, str]] = []
    source_rows: list[dict[str, str]] = []
    loads_by_bus: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"load_count": 0, "total_kw": 0.0, "total_kvar": 0.0}
    )
    degree: Counter[str] = Counter()
    transformers_by_bus: Counter[str] = Counter()
    sources_by_bus: Counter[str] = Counter()
    feeder_files: list[Path] = []

    for feeder_id, region_dir in feeders:
        region_coords = read_buscoords_dss(
            region_dir / DITTO_FILE_NAMES["buscoords"], feeder_id
        )
        coords.update(region_coords)
        for bus in region_coords:
            bus_to_feeder[bus] = feeder_id

        region_lines, region_degree = build_lines_for_region(feeder_id, region_dir, region_coords)
        line_rows.extend(region_lines)
        degree.update(region_degree)

        region_loads, region_loads_by_bus = build_loads_for_region(feeder_id, region_dir, region_coords)
        load_rows.extend(region_loads)
        for bus, agg in region_loads_by_bus.items():
            target = loads_by_bus[bus]
            target["load_count"] = int(target["load_count"]) + int(agg["load_count"])
            target["total_kw"] = float(target["total_kw"]) + float(agg["total_kw"])
            target["total_kvar"] = float(target["total_kvar"]) + float(agg["total_kvar"])

        region_transformers, region_transformers_by_bus = build_transformers_for_region(
            feeder_id, region_dir, region_coords
        )
        transformer_rows.extend(region_transformers)
        transformers_by_bus.update(region_transformers_by_bus)

        region_sources, region_sources_by_bus = build_sources_for_region(
            feeder_id, region_dir, region_coords
        )
        source_rows.extend(region_sources)
        sources_by_bus.update(region_sources_by_bus)

        for key in ("master", "lines", "loads", "transformers", "buscoords", "linecodes"):
            p = region_dir / DITTO_FILE_NAMES[key]
            if p.exists():
                feeder_files.append(p)

    line_rows.sort(key=lambda row: (row["feeder_id"], row["line_name"]))
    load_rows.sort(key=lambda row: (row["feeder_id"], row["bus"], row["load_name"]))
    transformer_rows.sort(key=lambda row: (row["feeder_id"], row["transformer_name"]))
    source_rows.sort(key=lambda row: (row["feeder_id"], row["source_name"]))

    bus_rows = build_buses(
        coords, bus_to_feeder, degree, loads_by_bus, transformers_by_bus, sources_by_bus
    )
    load_bus_rows = build_load_buses(coords, bus_to_feeder, loads_by_bus)
    feeder_rows = build_feeders(bus_rows, line_rows, transformer_rows, source_rows, load_rows)

    warnings: list[str] = []
    for label, rows in {
        "lines": line_rows,
        "loads": load_rows,
        "transformers": transformer_rows,
        "sources": source_rows,
        "load_buses": load_bus_rows,
    }.items():
        missing = sum(1 for row in rows if row.get("has_buscoords") == "false")
        if missing:
            warnings.append(f"{label}: {missing} records reference buses without coordinates")

    outputs = {
        "buses.csv": write_csv(
            output_dir / "buses.csv",
            bus_rows,
            [
                "bus",
                "feeder_id",
                "lon",
                "lat",
                "line_degree",
                "load_count",
                "load_kw",
                "load_kvar",
                "transformer_count",
                "source_count",
            ],
        ),
        "lines.csv": write_csv(
            output_dir / "lines.csv",
            line_rows,
            [
                "line_name",
                "feeder_id",
                "from_bus",
                "from_nodes",
                "from_lon",
                "from_lat",
                "to_bus",
                "to_nodes",
                "to_lon",
                "to_lat",
                "phases",
                "linecode",
                "line_class",
                "length",
                "units",
                "has_buscoords",
                "source_file",
                "source_line",
            ],
        ),
        "transformers.csv": write_csv(
            output_dir / "transformers.csv",
            transformer_rows,
            [
                "transformer_name",
                "feeder_id",
                "location_bus",
                "location_basis",
                "location_lon",
                "location_lat",
                "primary_bus",
                "winding_buses",
                "winding_count",
                "phases",
                "max_kv",
                "min_kv",
                "max_kva",
                "pct_loadloss",
                "pct_noloadloss",
                "has_buscoords",
                "source_file",
                "source_line",
            ],
        ),
        "sources.csv": write_csv(
            output_dir / "sources.csv",
            source_rows,
            [
                "source_name",
                "source_class",
                "feeder_id",
                "bus",
                "nodes",
                "lon",
                "lat",
                "phases",
                "basekv",
                "pu",
                "angle",
                "r1",
                "x1",
                "r0",
                "x0",
                "has_buscoords",
                "source_file",
                "source_line",
            ],
        ),
        "loads.csv": write_csv(
            output_dir / "loads.csv",
            load_rows,
            [
                "load_name",
                "feeder_id",
                "bus",
                "nodes",
                "lon",
                "lat",
                "phases",
                "conn",
                "kv",
                "kw",
                "kvar",
                "model",
                "has_buscoords",
                "source_file",
                "source_line",
            ],
        ),
        "load_buses.csv": write_csv(
            output_dir / "load_buses.csv",
            load_bus_rows,
            [
                "bus",
                "feeder_id",
                "lon",
                "lat",
                "load_count",
                "load_kw",
                "load_kvar",
                "has_buscoords",
            ],
        ),
        "feeders.csv": write_csv(
            output_dir / "feeders.csv",
            feeder_rows,
            [
                "feeder_id",
                "bus_count",
                "line_count",
                "transformer_count",
                "source_count",
                "load_count",
                "load_kw",
                "load_kvar",
            ],
        ),
    }
    write_summary(output_dir / "summary.json", feeder_files, outputs, warnings)
    return outputs
