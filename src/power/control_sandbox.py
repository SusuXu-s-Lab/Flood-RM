"""Control-sandbox filtering for Marshfield Asset Registry exports."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from power.build_asset_registry import build_feeders

CONTROL_SANDBOX_FILTER_SCHEMA_VERSION = "marshfield_control_sandbox_filter.v0.1"

REGISTRY_CSV_NAMES = (
    "buses.csv",
    "lines.csv",
    "transformers.csv",
    "sources.csv",
    "loads.csv",
    "load_buses.csv",
    "feeders.csv",
)

FEEDER_FIELDS = [
    "feeder_id",
    "bus_count",
    "line_count",
    "transformer_count",
    "source_count",
    "load_count",
    "load_kw",
    "load_kvar",
]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _as_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object, default: int = 0) -> int:
    parsed = _as_float(value)
    return default if parsed is None else int(parsed)


def _bus_components(
    buses: list[dict[str, str]],
    lines: list[dict[str, str]],
    transformers: list[dict[str, str]],
) -> tuple[dict[str, str], Counter[str]]:
    parent = {row["bus"]: row["bus"] for row in buses if row.get("bus")}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        if not left or not right:
            return
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in lines:
        if (row.get("line_class") or "line") != "line":
            continue
        union(row.get("from_bus", ""), row.get("to_bus", ""))

    for row in transformers:
        windings = [
            bus.strip()
            for bus in (row.get("winding_buses") or "").split(",")
            if bus.strip()
        ]
        if len(windings) < 2:
            continue
        hub = windings[0]
        for bus in windings[1:]:
            union(hub, bus)

    bus_to_component = {bus: find(bus) for bus in parent}
    return bus_to_component, Counter(bus_to_component.values())


def _component_groups_from_tie_candidates(
    buses: list[dict[str, str]],
    bus_to_component: dict[str, str],
    *,
    max_distance_m: float,
    min_line_degree: int,
) -> list[set[str]]:
    components = set(bus_to_component.values())
    graph = {component: set() for component in components}
    candidate_rows = [
        row
        for row in buses
        if row.get("bus") in bus_to_component
        and row.get("feeder_id")
        and _as_int(row.get("line_degree"), default=0) >= min_line_degree
        and _as_float(row.get("lon")) is not None
        and _as_float(row.get("lat")) is not None
    ]
    if not candidate_rows:
        return [{component} for component in sorted(components)]

    reference_lat = sum(float(row["lat"]) for row in candidate_rows) / len(candidate_rows)
    lon_scale = 111_320.0 * math.cos(math.radians(reference_lat))
    lat_scale = 110_540.0
    cell_size = max(max_distance_m, 1.0)
    candidates = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        projected = {
            "row": row,
            "x": float(row["lon"]) * lon_scale,
            "y": float(row["lat"]) * lat_scale,
        }
        bucket = (
            math.floor(projected["x"] / cell_size),
            math.floor(projected["y"] / cell_size),
        )
        candidates.append(projected)
        buckets[bucket].append(index)

    for left_index, projected_left in enumerate(candidates):
        left = projected_left["row"]
        left_component = bus_to_component[left["bus"]]
        left_bucket = (
            math.floor(projected_left["x"] / cell_size),
            math.floor(projected_left["y"] / cell_size),
        )
        neighbor_indexes = [
            index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for index in buckets.get((left_bucket[0] + dx, left_bucket[1] + dy), [])
            if index > left_index
        ]
        for right_index in neighbor_indexes:
            projected_right = candidates[right_index]
            right = projected_right["row"]
            if left.get("feeder_id") == right.get("feeder_id"):
                continue
            right_component = bus_to_component[right["bus"]]
            if left_component == right_component:
                continue
            distance = math.hypot(
                projected_right["x"] - projected_left["x"],
                projected_right["y"] - projected_left["y"],
            )
            if distance > max_distance_m:
                continue
            graph[left_component].add(right_component)
            graph[right_component].add(left_component)

    groups: list[set[str]] = []
    seen: set[str] = set()
    for component in sorted(components):
        if component in seen:
            continue
        stack = [component]
        seen.add(component)
        group: set[str] = set()
        while stack:
            current = stack.pop()
            group.add(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _transformer_in_kept_buses(row: dict[str, str], kept_buses: set[str]) -> bool:
    windings = [
        bus.strip()
        for bus in (row.get("winding_buses") or "").split(",")
        if bus.strip()
    ]
    return bool(windings) and all(bus in kept_buses for bus in windings)


def build_control_sandbox_registry(
    raw_registry_dir: Path | str,
    output_dir: Path | str,
    *,
    max_tie_distance_m: float = 100.0,
    min_tie_bus_line_degree: int = 2,
) -> dict[str, int]:
    """Write the canonical Marshfield control-sandbox registry.

    The raw SHIFT/DiTTo registry may contain source-backed feeder components
    outside the networked-microgrid reconfiguration graph. This filter retains
    the largest baseline-component group connected by eligible cross-feeder tie
    proximity and writes that group as the canonical Asset Registry consumed by
    Stage A and Stage B exports.
    """

    raw_registry_dir = Path(raw_registry_dir)
    output_dir = Path(output_dir)
    tables: dict[str, list[dict[str, str]]] = {}
    fields: dict[str, list[str]] = {}
    for name in REGISTRY_CSV_NAMES:
        rows, header = _read_csv(raw_registry_dir / name)
        tables[name] = rows
        fields[name] = header

    bus_to_component, component_sizes = _bus_components(
        tables["buses.csv"], tables["lines.csv"], tables["transformers.csv"]
    )
    groups = _component_groups_from_tie_candidates(
        tables["buses.csv"],
        bus_to_component,
        max_distance_m=max_tie_distance_m,
        min_line_degree=min_tie_bus_line_degree,
    )
    if groups:
        kept_components = max(
            groups,
            key=lambda group: (
                sum(component_sizes[component] for component in group),
                len(group),
                sorted(group),
            ),
        )
    else:
        kept_components = set(component_sizes.keys())

    kept_buses = {
        bus
        for bus, component in bus_to_component.items()
        if component in kept_components
    }
    filtered = {
        "buses.csv": [
            row for row in tables["buses.csv"] if row.get("bus") in kept_buses
        ],
        "lines.csv": [
            row
            for row in tables["lines.csv"]
            if row.get("from_bus") in kept_buses and row.get("to_bus") in kept_buses
        ],
        "transformers.csv": [
            row
            for row in tables["transformers.csv"]
            if _transformer_in_kept_buses(row, kept_buses)
        ],
        "sources.csv": [
            row for row in tables["sources.csv"] if row.get("bus") in kept_buses
        ],
        "loads.csv": [
            row for row in tables["loads.csv"] if row.get("bus") in kept_buses
        ],
        "load_buses.csv": [
            row for row in tables["load_buses.csv"] if row.get("bus") in kept_buses
        ],
    }
    filtered["feeders.csv"] = build_feeders(
        filtered["buses.csv"],
        filtered["lines.csv"],
        filtered["transformers.csv"],
        filtered["sources.csv"],
        filtered["loads.csv"],
    )

    outputs = {
        name: _write_csv(
            output_dir / name,
            filtered[name],
            FEEDER_FIELDS if name == "feeders.csv" else fields[name],
        )
        for name in REGISTRY_CSV_NAMES
    }

    raw_summary_path = raw_registry_dir / "summary.json"
    raw_summary = (
        json.loads(raw_summary_path.read_text(encoding="utf-8"))
        if raw_summary_path.exists()
        else {}
    )
    excluded_components = set(component_sizes) - set(kept_components)
    excluded_buses = {
        bus
        for bus, component in bus_to_component.items()
        if component in excluded_components
    }
    summary = {
        "method": "canonical Marshfield control-sandbox filter over raw Asset Registry",
        "schema_version": CONTROL_SANDBOX_FILTER_SCHEMA_VERSION,
        "raw_asset_registry_summary": raw_summary,
        "outputs": outputs,
        "control_sandbox_filter": {
            "schema_version": CONTROL_SANDBOX_FILTER_SCHEMA_VERSION,
            "input_registry_dir": str(raw_registry_dir),
            "max_tie_distance_m": float(max_tie_distance_m),
            "min_tie_bus_line_degree": int(min_tie_bus_line_degree),
            "raw_baseline_components": len(component_sizes),
            "retained_baseline_components": len(kept_components),
            "excluded_baseline_components": len(excluded_components),
            "retained_buses": len(kept_buses),
            "excluded_buses": len(excluded_buses),
            "excluded_loads": len(tables["loads.csv"]) - len(filtered["loads.csv"]),
            "excluded_sources": len(tables["sources.csv"]) - len(filtered["sources.csv"]),
            "exclusion_reason": "outside_largest_tie_eligible_component_group",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs
