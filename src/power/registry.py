"""Native GDM/DiTTo asset-registry export.

OpenDSS parsing belongs to DiTTo; validation, quantities, serialization, and
connectivity belong to GDM.  This module converts already-validated GDM systems
into the small CSV Asset Registry used by the downstream artifact contract.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import file_sha256, parse_float, parse_int, safe_token, write_csv, write_json
from .native import read_opendss

feeder_sep = "__"

DITTO_FEEDER_FILES = (
    "Master.dss",
    "Lines.dss",
    "Loads.dss",
    "Transformers.dss",
    "BusCoords.dss",
    "LineCodes.dss",
)

bus_fields = [
    "bus", "feeder_id", "lon", "lat", "line_degree", "load_count", "load_kw", "load_kvar", "transformer_count", "source_count",
]
line_fields = [
    "line_name", "feeder_id", "from_bus", "from_nodes", "from_lon", "from_lat", "to_bus", "to_nodes", "to_lon", "to_lat",
    "phases", "linecode", "line_class", "length", "units", "has_buscoords", "source_file", "source_line",
]
transformer_fields = [
    "transformer_name", "feeder_id", "location_bus", "location_basis", "location_lon", "location_lat", "primary_bus",
    "winding_buses", "winding_count", "phases", "max_kv", "min_kv", "max_kva", "pct_loadloss", "pct_noloadloss",
    "has_buscoords", "source_file", "source_line",
]
source_fields = [
    "source_name", "source_class", "feeder_id", "bus", "nodes", "lon", "lat", "phases", "basekv", "pu", "angle", "r1", "x1", "r0", "x0", "has_buscoords", "source_file", "source_line",
]
load_fields = [
    "load_name", "feeder_id", "bus", "nodes", "lon", "lat", "phases", "conn", "kv", "kw", "kvar", "model", "has_buscoords", "source_file", "source_line",
]
load_bus_fields = ["bus", "feeder_id", "lon", "lat", "load_count", "load_kw", "load_kvar", "has_buscoords"]
feeder_fields = ["feeder_id", "bus_count", "line_count", "transformer_count", "source_count", "load_count", "load_kw", "load_kvar"]


@dataclass(frozen=True)
class GdmRegistryInputs:
    """One native GDM system plus the feeder namespace assigned to it."""

    feeder_id: str
    system: Any
    source_path: Path | None = None


def qualify_bus(feeder_id: str, bus: str | None) -> str:
    value = _bus_name(bus)
    if not value:
        return ""
    return value if value.startswith(f"{feeder_id}{feeder_sep}") else f"{feeder_id}{feeder_sep}{value}"


def split_bus_ref(bus_ref: Any) -> tuple[str, str]:
    if bus_ref is None:
        return "", ""
    if isinstance(bus_ref, (list, tuple)):
        bus_ref = bus_ref[0] if bus_ref else ""
    text = str(bus_ref).strip()
    if not text:
        return "", ""
    parts = text.split(".")
    return parts[0], ".".join(parts[1:])


def build_registry(opendss_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Parse per-feeder OpenDSS with DiTTo and write normalized registry CSVs.

    ``opendss_dir`` may be a single ``Master.dss`` file, a directory containing
    ``Master.dss``, or a DiTTo-style directory with one child folder per feeder.
    """

    return build_registry_from_systems(_read_ditto_systems(opendss_dir), output_dir)


def build_registry_from_systems(systems: Iterable[GdmRegistryInputs], output_dir: str | Path) -> dict[str, int]:
    """Write the Asset Registry from native GDM systems."""

    all_buses: dict[str, dict[str, Any]] = {}
    line_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    transformer_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    load_by_bus: dict[str, dict[str, float | int]] = defaultdict(lambda: {"load_count": 0, "total_kw": 0.0, "total_kvar": 0.0})
    degree: Counter[str] = Counter()
    transformer_by_bus: Counter[str] = Counter()
    source_by_bus: Counter[str] = Counter()
    input_files: list[Path] = []

    for item in systems:
        if item.source_path is not None:
            input_files.append(Path(item.source_path))
        feeder_id = item.feeder_id
        bus_coord = _bus_coordinates(item.system, feeder_id)
        bus_seen: set[str] = set(bus_coord)

        lines = _components(item.system, ("get_lines", "get_branches", "get_edges"), ("line", "branch", "cable"))
        for component in lines:
            row = _line_row(component, feeder_id, bus_coord, item.source_path)
            if not row["from_bus"] or not row["to_bus"]:
                continue
            line_rows.append(row)
            degree[row["from_bus"]] += 1
            degree[row["to_bus"]] += 1
            bus_seen.update([row["from_bus"], row["to_bus"]])

        loads = _components(item.system, ("get_loads",), ("load",))
        for component in loads:
            row = _load_row(component, feeder_id, bus_coord, item.source_path)
            if not row["bus"]:
                continue
            load_rows.append(row)
            target = load_by_bus[row["bus"]]
            target["load_count"] = int(target["load_count"]) + 1
            target["total_kw"] = float(target["total_kw"]) + float(row["kw"] or 0.0)
            target["total_kvar"] = float(target["total_kvar"]) + float(row["kvar"] or 0.0)
            bus_seen.add(row["bus"])

        transformers = _components(item.system, ("get_transformers",), ("transformer",))
        for component in transformers:
            row = _transformer_row(component, feeder_id, bus_coord, item.source_path)
            if row["winding_buses"]:
                transformer_rows.append(row)
                for bus in set(row["winding_buses"].split(",")):
                    if bus:
                        transformer_by_bus[bus] += 1
                        bus_seen.add(bus)

        sources = _components(item.system, ("get_voltage_sources", "get_sources"), ("voltagesource", "voltage_source", "circuit", "source"))
        for component in sources:
            row = _source_row(component, feeder_id, bus_coord, item.source_path)
            if not row["bus"]:
                continue
            source_rows.append(row)
            source_by_bus[row["bus"]] += 1
            bus_seen.add(row["bus"])

        for bus in bus_seen:
            lon, lat = bus_coord.get(bus, (None, None))
            all_buses.setdefault(bus, {"bus": bus, "feeder_id": feeder_id, "lon": _fmt(lon), "lat": _fmt(lat)})

    bus_rows = _build_bus_rows(all_buses, degree, load_by_bus, transformer_by_bus, source_by_bus)
    load_bus_rows = _build_load_bus_rows(all_buses, load_by_bus)
    feeder_rows = build_feeders(bus_rows, line_rows, transformer_rows, source_rows, load_rows)

    output_dir = Path(output_dir)
    outputs = {
        "buses.csv": write_csv(output_dir / "buses.csv", bus_rows, bus_fields),
        "lines.csv": write_csv(output_dir / "lines.csv", sorted(line_rows, key=lambda r: (r["feeder_id"], r["line_name"])), line_fields),
        "transformers.csv": write_csv(output_dir / "transformers.csv", sorted(transformer_rows, key=lambda r: (r["feeder_id"], r["transformer_name"])), transformer_fields),
        "sources.csv": write_csv(output_dir / "sources.csv", sorted(source_rows, key=lambda r: (r["feeder_id"], r["source_name"])), source_fields),
        "loads.csv": write_csv(output_dir / "loads.csv", sorted(load_rows, key=lambda r: (r["feeder_id"], r["bus"], r["load_name"])), load_fields),
        "load_buses.csv": write_csv(output_dir / "load_buses.csv", load_bus_rows, load_bus_fields),
        "feeders.csv": write_csv(output_dir / "feeders.csv", feeder_rows, feeder_fields),
    }
    warnings = _registry_warnings(line_rows, load_rows, transformer_rows, source_rows, load_bus_rows)
    write_json(
        output_dir / "summary.json",
        {
            "method": "DiTTo OpenDSS Reader -> native GDM DistributionSystem -> normalized Asset Registry",
            "native_api": {
                "ditto": "ditto.readers.opendss.reader.Reader(...).get_system()",
                "gdm": "DistributionSystem component accessors / pydantic objects",
            },
            "inputs": {str(path): {"sha256": file_sha256(path)} for path in input_files if path.exists()},
            "outputs": outputs,
            "warnings": warnings,
        },
    )
    return outputs


def build_feeders(
    bus_rows: list[dict[str, Any]],
    line_rows: list[dict[str, Any]],
    transformer_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    load_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    feeder_ids = sorted({str(row.get("feeder_id")) for row in bus_rows if row.get("feeder_id")})
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
        if row.get("feeder_id") in by_feeder:
            by_feeder[row["feeder_id"]]["bus_count"] += 1
    for rows, field in ((line_rows, "line_count"), (transformer_rows, "transformer_count"), (source_rows, "source_count")):
        for row in rows:
            if row.get("feeder_id") in by_feeder:
                by_feeder[row["feeder_id"]][field] += 1
    for row in load_rows:
        if row.get("feeder_id") in by_feeder:
            target = by_feeder[row["feeder_id"]]
            target["load_count"] += 1
            target["load_kw"] += float(row.get("kw") or 0.0)
            target["load_kvar"] += float(row.get("kvar") or 0.0)
    return [
        {
            "feeder_id": feeder,
            "bus_count": str(int(values["bus_count"])),
            "line_count": str(int(values["line_count"])),
            "transformer_count": str(int(values["transformer_count"])),
            "source_count": str(int(values["source_count"])),
            "load_count": str(int(values["load_count"])),
            "load_kw": _fmt(values["load_kw"], 4),
            "load_kvar": _fmt(values["load_kvar"], 4),
        }
        for feeder, values in by_feeder.items()
    ]


# ---- Native-system discovery --------------------------------------------


def _read_ditto_systems(opendss_dir: str | Path) -> tuple[GdmRegistryInputs, ...]:
    path = Path(opendss_dir)
    masters: list[tuple[str, Path]] = []
    if path.is_file():
        masters.append((safe_token(path.parent.name), path))
    elif (path / "Master.dss").exists():
        masters.append((safe_token(path.name), path / "Master.dss"))
    elif path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "Master.dss").exists():
                masters.append((safe_token(child.name), child / "Master.dss"))
    if not masters:
        raise FileNotFoundError(f"no OpenDSS Master.dss file found under {path}")
    return tuple(GdmRegistryInputs(feeder_id=feeder, system=read_opendss(master).system, source_path=master) for feeder, master in masters)


def _components(system: Any, method_names: Sequence[str], class_tokens: Sequence[str]) -> list[Any]:
    seen: set[int] = set()
    rows: list[Any] = []
    for name in method_names:
        method = getattr(system, name, None)
        if callable(method):
            try:
                values = method()
            except TypeError:
                continue
            for item in _iter_values(values):
                if id(item) not in seen:
                    seen.add(id(item)); rows.append(item)
    if rows:
        return rows
    for name in ("iter_all_components", "iter_components", "get_components"):
        method = getattr(system, name, None)
        if not callable(method):
            continue
        try:
            values = method()
        except TypeError:
            continue
        for item in _iter_values(values):
            cls = item.__class__.__name__.lower()
            if any(token.lower() in cls for token in class_tokens) and id(item) not in seen:
                seen.add(id(item)); rows.append(item)
    return rows


def _iter_values(values: Any) -> Iterable[Any]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        return values.values()
    return values


# ---- Component extraction -------------------------------------------------


def _line_row(component: Any, feeder_id: str, coords: Mapping[str, tuple[float | None, float | None]], source_path: Path | None) -> dict[str, Any]:
    bus1_raw, nodes1 = split_bus_ref(_first(component, "from_bus", "bus1", "from_node", "source_bus", "source"))
    bus2_raw, nodes2 = split_bus_ref(_first(component, "to_bus", "bus2", "to_node", "target_bus", "target"))
    if not (bus1_raw and bus2_raw):
        pair = _bus_list(component)
        if len(pair) >= 2:
            bus1_raw, nodes1 = split_bus_ref(pair[0])
            bus2_raw, nodes2 = split_bus_ref(pair[1])
    bus1, bus2 = qualify_bus(feeder_id, bus1_raw), qualify_bus(feeder_id, bus2_raw)
    lon1, lat1 = coords.get(bus1, (None, None)); lon2, lat2 = coords.get(bus2, (None, None))
    linecode = _equipment_name(component) or str(_first(component, "linecode", "line_code", default=""))
    return {
        "line_name": _name(component), "feeder_id": feeder_id,
        "from_bus": bus1, "from_nodes": nodes1, "from_lon": _fmt(lon1), "from_lat": _fmt(lat1),
        "to_bus": bus2, "to_nodes": nodes2, "to_lon": _fmt(lon2), "to_lat": _fmt(lat2),
        "phases": _fmt_int(_phase_count(component)), "linecode": linecode, "line_class": classify_line(linecode, component),
        "length": _fmt(_quantity(_first(component, "length", "length_m", default=None)), 4), "units": _units(_first(component, "length", default=None), "m"),
        "has_buscoords": str(lon1 is not None and lat1 is not None and lon2 is not None and lat2 is not None).lower(),
        "source_file": str(source_path or "gdm"), "source_line": "",
    }


def _load_row(component: Any, feeder_id: str, coords: Mapping[str, tuple[float | None, float | None]], source_path: Path | None) -> dict[str, Any]:
    bus_raw, nodes = split_bus_ref(_first(component, "bus", "bus1", "load_bus", "node", default=""))
    if not bus_raw:
        buses = _bus_list(component)
        bus_raw, nodes = split_bus_ref(buses[0] if buses else "")
    bus = qualify_bus(feeder_id, bus_raw)
    lon, lat = coords.get(bus, (None, None))
    kw = _quantity(_first(component, "kw", "p", "real_power", "active_power", default=0.0), preferred_units=("kilowatt", "kW"))
    kvar = _quantity(_first(component, "kvar", "q", "reactive_power", default=0.0), preferred_units=("kilovar", "kvar"))
    return {
        "load_name": _name(component), "feeder_id": feeder_id, "bus": bus, "nodes": nodes,
        "lon": _fmt(lon), "lat": _fmt(lat), "phases": _fmt_int(_phase_count(component)),
        "conn": str(_first(component, "conn", "connection_type", default="")),
        "kv": _fmt(_quantity(_first(component, "kv", "voltage", "rated_voltage", default=None), preferred_units=("kilovolt", "kV")), 4),
        "kw": _fmt(kw, 4), "kvar": _fmt(kvar, 4), "model": str(_first(component, "model", default="")),
        "has_buscoords": str(lon is not None and lat is not None).lower(), "source_file": str(source_path or "gdm"), "source_line": "",
    }


def _transformer_row(component: Any, feeder_id: str, coords: Mapping[str, tuple[float | None, float | None]], source_path: Path | None) -> dict[str, Any]:
    raw_buses = _bus_list(component)
    if not raw_buses:
        raw_buses = [_first(w, "bus", "bus1", "node", default="") for w in _as_list(_first(component, "windings", default=[]))]
    buses = [qualify_bus(feeder_id, split_bus_ref(bus)[0]) for bus in raw_buses if split_bus_ref(bus)[0]]
    location_bus = buses[-1] if buses else ""
    lon, lat = coords.get(location_bus, (None, None))
    windings = _as_list(_first(component, "windings", default=[]))
    kvs = [_quantity(_first(w, "kv", "rated_voltage", "voltage", default=None), preferred_units=("kilovolt", "kV")) for w in windings]
    kvas = [_quantity(_first(w, "kva", "rated_power", "power", default=None), preferred_units=("kilovolt_ampere", "kVA")) for w in windings]
    kv_values = [v for v in kvs if v is not None]
    kva_values = [v for v in kvas if v is not None]
    return {
        "transformer_name": _name(component), "feeder_id": feeder_id,
        "location_bus": location_bus, "location_basis": "last_winding_bus", "location_lon": _fmt(lon), "location_lat": _fmt(lat),
        "primary_bus": buses[0] if buses else "", "winding_buses": ",".join(buses), "winding_count": _fmt_int(len(buses) or parse_int(_first(component, "num_windings", "windings", default=0), 0)),
        "phases": _fmt_int(_phase_count(component)), "max_kv": _fmt(max(kv_values) if kv_values else None, 4), "min_kv": _fmt(min(kv_values) if kv_values else None, 4),
        "max_kva": _fmt(max(kva_values) if kva_values else _quantity(_first(component, "kva", "rated_power", default=None), preferred_units=("kilovolt_ampere", "kVA")), 4),
        "pct_loadloss": _fmt(_first(component, "pct_loadloss", "pct_full_load_loss", default=None), 6),
        "pct_noloadloss": _fmt(_first(component, "pct_noloadloss", "pct_no_load_loss", default=None), 6),
        "has_buscoords": str(lon is not None and lat is not None).lower(), "source_file": str(source_path or "gdm"), "source_line": "",
    }


def _source_row(component: Any, feeder_id: str, coords: Mapping[str, tuple[float | None, float | None]], source_path: Path | None) -> dict[str, Any]:
    bus_raw, nodes = split_bus_ref(_first(component, "bus", "bus1", "source_bus", "node", default=""))
    if not bus_raw:
        buses = _bus_list(component)
        bus_raw, nodes = split_bus_ref(buses[0] if buses else "")
    bus = qualify_bus(feeder_id, bus_raw)
    lon, lat = coords.get(bus, (None, None))
    return {
        "source_name": _name(component), "source_class": component.__class__.__name__, "feeder_id": feeder_id, "bus": bus, "nodes": nodes,
        "lon": _fmt(lon), "lat": _fmt(lat), "phases": _fmt_int(_phase_count(component)),
        "basekv": _fmt(_quantity(_first(component, "basekv", "base_kv", "voltage", "rated_voltage", default=None), preferred_units=("kilovolt", "kV")), 4),
        "pu": _fmt(_first(component, "pu", "per_unit", default=1.0), 6), "angle": _fmt(_first(component, "angle", default=0.0), 6),
        "r1": _fmt(_first(component, "r1", default=None), 6), "x1": _fmt(_first(component, "x1", default=None), 6),
        "r0": _fmt(_first(component, "r0", default=None), 6), "x0": _fmt(_first(component, "x0", default=None), 6),
        "has_buscoords": str(lon is not None and lat is not None).lower(), "source_file": str(source_path or "gdm"), "source_line": "",
    }


def _bus_coordinates(system: Any, feeder_id: str) -> dict[str, tuple[float | None, float | None]]:
    coords: dict[str, tuple[float | None, float | None]] = {}
    for bus in _components(system, ("get_buses",), ("bus",)):
        name = qualify_bus(feeder_id, _name(bus))
        lon, lat = _xy(bus)
        coords[name] = (lon, lat)
    return coords


# ---- Row aggregation ------------------------------------------------------


def _build_bus_rows(
    bus_map: Mapping[str, Mapping[str, Any]],
    degree: Counter[str],
    loads: Mapping[str, Mapping[str, float | int]],
    transformers: Counter[str],
    sources: Counter[str],
) -> list[dict[str, Any]]:
    rows = []
    for bus, base in bus_map.items():
        load = loads.get(bus, {})
        rows.append({
            "bus": bus, "feeder_id": base.get("feeder_id", ""), "lon": base.get("lon", ""), "lat": base.get("lat", ""),
            "line_degree": str(degree.get(bus, 0)), "load_count": str(int(load.get("load_count", 0))),
            "load_kw": _fmt(float(load.get("total_kw", 0.0)), 4), "load_kvar": _fmt(float(load.get("total_kvar", 0.0)), 4),
            "transformer_count": str(transformers.get(bus, 0)), "source_count": str(sources.get(bus, 0)),
        })
    return sorted(rows, key=lambda r: (r["feeder_id"], r["bus"]))


def _build_load_bus_rows(bus_map: Mapping[str, Mapping[str, Any]], loads: Mapping[str, Mapping[str, float | int]]) -> list[dict[str, Any]]:
    rows = []
    for bus, values in loads.items():
        base = bus_map.get(bus, {})
        rows.append({
            "bus": bus, "feeder_id": base.get("feeder_id", ""), "lon": base.get("lon", ""), "lat": base.get("lat", ""),
            "load_count": str(int(values.get("load_count", 0))), "load_kw": _fmt(float(values.get("total_kw", 0.0)), 4),
            "load_kvar": _fmt(float(values.get("total_kvar", 0.0)), 4), "has_buscoords": str(bool(base.get("lon") and base.get("lat"))).lower(),
        })
    return sorted(rows, key=lambda r: (r["feeder_id"], r["bus"]))


def _registry_warnings(*tables: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = ("lines", "loads", "transformers", "sources", "load_buses")
    warnings = []
    for label, rows in zip(labels, tables, strict=False):
        missing = sum(1 for row in rows if str(row.get("has_buscoords", "true")).lower() == "false")
        if missing:
            warnings.append(f"{label}: {missing} records reference buses without coordinates")
    return warnings


# ---- Generic GDM introspection helpers -----------------------------------


def _first(obj: Any, *names: str, default: Any = None) -> Any:
    data = _dump(obj)
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if name in data and data[name] is not None:
            return data[name]
    return default


def _dump(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    for method in ("model_dump", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return dict(fn())
            except TypeError:
                continue
    return {}


def _name(obj: Any) -> str:
    return str(_first(obj, "name", "id", "uuid", default=obj.__class__.__name__))


def _bus_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _bus_list(component: Any) -> list[Any]:
    for name in ("buses", "bus_names", "nodes", "terminals"):
        value = _first(component, name, default=None)
        if value:
            return _as_list(value)
    pair = [_first(component, "bus1", "from_bus", default=""), _first(component, "bus2", "to_bus", default="")]
    return [item for item in pair if item]


def _xy(obj: Any) -> tuple[float | None, float | None]:
    lon = parse_float(_first(obj, "lon", "longitude", "x", default=None))
    lat = parse_float(_first(obj, "lat", "latitude", "y", default=None))
    if lon is not None and lat is not None:
        return lon, lat
    location = _first(obj, "location", "coordinates", "geometry", default=None)
    if location is None:
        return None, None
    if hasattr(location, "x") and hasattr(location, "y"):
        return parse_float(location.x), parse_float(location.y)
    if hasattr(location, "longitude") and hasattr(location, "latitude"):
        return parse_float(location.longitude), parse_float(location.latitude)
    if isinstance(location, Sequence) and not isinstance(location, str) and len(location) >= 2:
        return parse_float(location[0]), parse_float(location[1])
    return None, None


def _quantity(value: Any, preferred_units: Sequence[str] = ()) -> float | None:
    if value is None:
        return None
    for units in preferred_units:
        converter = getattr(value, "to", None)
        if callable(converter):
            try:
                converted = converter(units)
                return parse_float(getattr(converted, "magnitude", converted))
            except Exception:
                pass
    return parse_float(getattr(value, "magnitude", value))


def _units(value: Any, default: str = "") -> str:
    units = getattr(value, "units", None)
    return str(units) if units is not None else default


def _phase_count(component: Any) -> int | None:
    value = _first(component, "phases", "num_phases", "phase_count", default=None)
    if value is None:
        phase_list = _first(component, "phase_loads", "phase_windings", default=None)
        if phase_list:
            return len(_as_list(phase_list))
        return None
    if isinstance(value, str):
        return parse_int(value)
    try:
        return len(value)
    except TypeError:
        return parse_int(value)


def _equipment_name(component: Any) -> str:
    equipment = _first(component, "equipment", "linecode", "line_code", default=None)
    if equipment is None:
        return ""
    return _name(equipment) if not isinstance(equipment, str) else equipment


def classify_line(linecode: str, component: Any | None = None) -> str:
    text = f"{linecode} {_name(component) if component is not None else ''} {component.__class__.__name__ if component is not None else ''}".lower()
    if "_ug_" in text or "underground" in text or "cable" in text:
        return "underground"
    if "_oh_" in text or "overhead" in text:
        return "overhead"
    if "fuse" in text:
        return "fuse"
    return "line"


def _fmt(value: Any, digits: int = 6) -> str:
    parsed = parse_float(value)
    return "" if parsed is None else f"{parsed:.{digits}f}"


def _fmt_int(value: Any) -> str:
    parsed = parse_int(value)
    return "" if parsed is None else str(parsed)
