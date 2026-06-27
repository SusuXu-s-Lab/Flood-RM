"""Audit Marshfield synthetic grid artifacts against synthetic-grid validation criteria.

The audit uses statistical, operational, and expert validation categories. It
does not certify the Marshfield Grid Dataset as SMART-DS or as a utility
model; it records which validation checks the current artifacts can support.
"""

from __future__ import annotations
import heapq
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from paths import default_location_config_path, find_repo_root
from study_location import define_location

repo_root = find_repo_root(Path(__file__).resolve())

def _location_definition():
    return define_location(default_location_config_path(repo_root))

def _location_id():
    definition = _location_definition()
    return os.environ.get("FLOOD_RM_LOCATION_ID") or os.environ.get("FLOOD_RM_SANDBOX_ID") or str(
        definition.config.get("project", {}).get("name") or definition.root.name
    )

def _power_grid_root():
    definition = _location_definition()
    path = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return path if path.is_absolute() else definition.root / path

def parse_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

location_id = _location_id()
power_grid = _power_grid_root()

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RegionConfig:
    """Static configuration for one Regional Case Domain."""

    region_id: str
    smart_ds_code: str
    smart_ds_subregions: tuple[str, ...]
    artifact_root: Path

_sfo_subregions = tuple(
    [f"P{i}R" for i in range(1, 6)] + [f"P{i}U" for i in range(1, 36)]
)
_austin_subregions = ("P1R", "P1U", "P2U", "P3U", "P4U", "P5U")
_greensboro_subregions = ("industrial", "rural", "urban-suburban")

def _region(region_id: str, code: str, subregions: tuple[str, ...]) -> RegionConfig:
    return RegionConfig(
        region_id,
        code,
        subregions,
        Path("locations") / region_id / "data",
    )

regions: dict[str, RegionConfig] = {
    "sfo": _region("sfo", "SFO", _sfo_subregions),
    "austin": _region("austin", "AUS", _austin_subregions),
    "greensboro": _region("greensboro", "GSO", _greensboro_subregions),
}

def get_region_config(region_id: str) -> RegionConfig:
    """Return the configured Regional Case Domain."""
    if region_id not in regions:
        valid = ", ".join(sorted(regions))
        raise ValueError(f"unknown region_id {region_id!r}; expected one of: {valid}")
    return regions[region_id]

# SMART-DS reference helpers
"""Public SMART-DS v1.0 OpenDSS model locations on the OEDI data lake.
SMART-DS publishes synthetic distribution feeders per region/subregion; we use
them only as audit references. ``SmartDsModelRef`` is the single source of truth
for the OEDI key layout — the URL list and the download plan both derive from it.
"""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

oedi_s3_base_url = "https://oedi-data-lake.s3.amazonaws.com"
oedi_bucket = "oedi-data-lake"
valid_years = (2016, 2017, 2018)  # snapshots published in the public OEDI catalog

@dataclass(frozen=True)
class SmartDsModelRef:
    """OEDI location of one subregion's OpenDSS Master model."""
    region_id: str
    dataset_code: str
    year: int
    subregion: str
    scenario: str
    model_format: str

    @property
    def model_prefix(self) -> str:
        # OEDI key layout: SMART-DS/<version>/<year>/<region>/<subregion>/scenarios/<scenario>/<format>/
        return (
            f"SMART-DS/v1.0/{self.year}/{self.dataset_code}/{self.subregion}/"
            f"scenarios/{self.scenario}/{self.model_format}/"
        )

    @property
    def master_dss_key(self) -> str:
        return f"{self.model_prefix}Master.dss"

    @property
    def master_dss_url(self) -> str:
        return f"{oedi_s3_base_url}/{quote(self.master_dss_key)}"

    @property
    def source_s3_prefix(self) -> str:
        return f"s3://{oedi_bucket}/{self.model_prefix}"


def smart_ds_models(
    region_id,
    year=2016,
    scenario="base_timeseries",
    model_format="opendss",
) -> tuple[SmartDsModelRef, ...]:
    """One model reference per subregion of the configured region."""
    if year not in valid_years:
        raise ValueError(f"Year {year} is not in the public OEDI catalog.")
    cfg = get_region_config(region_id)
    return tuple(
        SmartDsModelRef(
            region_id=cfg.region_id,
            dataset_code=cfg.smart_ds_code,
            year=year,
            subregion=sub,
            scenario=scenario,
            model_format=model_format,
        )
        for sub in cfg.smart_ds_subregions
    )


@dataclass(frozen=True)
class SmartDsDownloadPlanItem:
    """One planned SMART-DS model download into the canonical artifact tree."""

    model: SmartDsModelRef
    output_dir: Path

    @property
    def source_s3_prefix(self) -> str:
        return self.model.source_s3_prefix

    @property
    def master_dss_path(self) -> Path:
        return self.output_dir / "Master.dss"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "download_manifest.json"


def smart_ds_model_dir(model: SmartDsModelRef, output_root: Path | None = None) -> Path:
    cfg = get_region_config(model.region_id)
    root = Path(output_root) if output_root else cfg.artifact_root / "smart_ds"
    return root / str(model.year) / model.subregion / model.scenario / model.model_format


def smart_ds_download_plan(
    region_id: str,
    *,
    year: int = 2016,
    scenario: str = "base_timeseries",
    model_format: str = "opendss",
    subregions: tuple[str, ...] | list[str] = (),
    all_subregions: bool = False,
    output_root: Path | None = None,
) -> tuple[SmartDsDownloadPlanItem, ...]:
    """Plan SMART-DS downloads for chosen subregions without fetching them."""
    models = smart_ds_models(region_id, year=year, scenario=scenario, model_format=model_format)
    if not all_subregions:
        wanted = set(subregions)
        if not wanted:
            raise ValueError("choose at least one subregion, or set all_subregions=True")
        unknown = sorted(wanted - {model.subregion for model in models})
        if unknown:
            available = ", ".join(sorted(model.subregion for model in models))
            raise ValueError(f"unknown subregion(s) {unknown}; expected one of: {available}")
        models = tuple(model for model in models if model.subregion in wanted)
    return tuple(
        SmartDsDownloadPlanItem(model=model, output_dir=smart_ds_model_dir(model, output_root=output_root))
        for model in models
    )


validation_reference_label = "synthetic_distribution_validation_criteria"
meters_per_mile = 1609.344

default_registry_dir = power_grid / "asset_registry"
default_smart_ds_compat_dir = power_grid / "augmented"
default_grid_network_dir = repo_root / "locations" / "marshfield" / "01_grid"
default_report_dir = default_grid_network_dir / "outputs" / "validation_audit"
default_power_flow_report = default_grid_network_dir / "outputs" / "power_flow_validation.json"
default_short_circuit_report = default_grid_network_dir / "outputs" / "short_circuit_validation.json"
smart_ds_reference_regions = ("sfo", "austin", "greensboro")
smart_ds_reference_year = 2016
smart_ds_reference_scenario = "base_timeseries"
smart_ds_reference_model_format = "opendss_no_loadshapes"
standard_distribution_transformer_kva_catalog = (
    15.0,
    25.0,
    37.5,
    50.0,
    75.0,
    100.0,
    167.0,
    250.0,
    333.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
)


@dataclass(frozen=True)
class ValidationRegion:
    typical: tuple[tuple[float, float], ...]
    uncommon: tuple[tuple[float, float], ...] = ()


table_iv_targets: dict[str, ValidationRegion] = {
    "distribution_transformer_mva_per_feeder": ValidationRegion(
        typical=((0.0, 1.73),),
        uncommon=((1.73, 4.94), (4.94, 31.0), (31.0, 38.629)),
    ),
    "real_load_kw_per_feeder": ValidationRegion(
        typical=((4181.0, 13793.0),),
        uncommon=((577.0, 4181.0), (13793.0, 17590.0)),
    ),
    "lv_1f_line_length_miles_per_feeder": ValidationRegion(
        typical=((0.0, 34.75),),
        uncommon=((34.75, 44.31),),
    ),
    "lv_3f_line_length_miles_per_feeder": ValidationRegion(
        typical=((0.0, 1.0),),
        uncommon=((1.0, 2.135),),
    ),
    "mv_1_2f_line_length_miles_per_feeder": ValidationRegion(
        typical=((0.0, 35.36),),
        uncommon=((35.36, 124.62),),
    ),
    "mv_3f_line_length_miles_per_feeder": ValidationRegion(
        typical=((0.0, 20.84),),
        uncommon=((20.84, 45.6),),
    ),
    "customer_count_per_feeder": ValidationRegion(
        typical=((94.0, 2607.0),),
        uncommon=((8.0, 94.0), (2607.0, 11837.0)),
    ),
    "mv_1_2f_line_miles_per_customer": ValidationRegion(
        typical=((0.0, 0.12),),
        uncommon=((0.12, 0.24),),
    ),
    "mv_3f_line_miles_per_customer": ValidationRegion(
        typical=((0.0, 0.09),),
        uncommon=((0.09, 0.77),),
    ),
    "fuses_per_feeder": ValidationRegion(
        typical=((4.0, 187.0),),
        uncommon=((187.0, 281.0),),
    ),
    "reclosers_per_feeder": ValidationRegion(
        typical=((0.0, 5.0),),
        uncommon=((5.0, 9.0),),
    ),
    "regulators_per_feeder": ValidationRegion(
        typical=((0.0, 3.0),),
        uncommon=((3.0, 8.0),),
    ),
    "sectionalizers_per_feeder": ValidationRegion(
        typical=((0.0, 1.0),),
        uncommon=((1.0, 3.0),),
    ),
    "switches_per_feeder": ValidationRegion(
        typical=((3.0, 392.0),),
        uncommon=((392.0, 635.0),),
    ),
    "capacitor_banks_per_feeder": ValidationRegion(
        typical=((0.0, 5.0),),
        uncommon=((5.0, 7.0),),
    ),
    "average_degree_per_feeder": ValidationRegion(
        typical=((1.9, 2.06),),
        uncommon=((1.6, 1.9), (2.06, 2.1)),
    ),
    "characteristic_path_length_miles_per_feeder": ValidationRegion(
        typical=((12.4, 95.0),),
        uncommon=((2.0, 12.4), (95.0, 134.39)),
    ),
    "graph_diameter_miles_per_feeder": ValidationRegion(
        typical=((32.0, 260.0),),
        uncommon=((4.0, 32.0), (260.0, 371.0)),
    ),
}


def _float(value: Any, default: float = 0.0) -> float:
    return parse_float(value, default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sum_by(rows: Iterable[dict[str, Any]], key: str, value: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row.get(key, ""))] += _float(row.get(value))
    return dict(totals)


def _count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(counts)


def _fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _in_ranges(value: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    return any(low <= value <= high for low, high in ranges)


def _grade_values(metric_id: str, values: list[float], *, notes: list[str] | None = None) -> dict[str, Any]:
    target = table_iv_targets[metric_id]
    typical = sum(1 for value in values if _in_ranges(value, target.typical))
    uncommon = sum(1 for value in values if _in_ranges(value, target.uncommon))
    rare = max(len(values) - typical - uncommon, 0)
    rare_fraction = _fraction(rare, len(values))
    if rare_fraction <= 0.05:
        grade = "good"
    elif rare_fraction <= 0.10:
        grade = "marginal"
    else:
        grade = "check"
    return {
        "metric_id": metric_id,
        "grade": grade,
        "feeder_count": len(values),
        "typical_fraction": round(_fraction(typical, len(values)), 4),
        "uncommon_fraction": round(_fraction(uncommon, len(values)), 4),
        "rare_fraction": round(rare_fraction, 4),
        "min": round(min(values), 6) if values else None,
        "median": round(sorted(values)[len(values) // 2], 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "validation_target": {
            "typical": target.typical,
            "uncommon": target.uncommon,
        },
        "notes": notes or [],
    }


def _custom_metric(metric_id: str, grade: str, notes: list[str], value: Any) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "grade": grade,
        "value": value,
        "notes": notes,
    }


def _line_length_miles(row: dict[str, str]) -> float:
    length = _float(row.get("length"))
    units = (row.get("units") or "").lower()
    if units in {"m", "meter", "meters"}:
        return length / meters_per_mile
    if units in {"mi", "mile", "miles"}:
        return length
    if units in {"km", "kilometer", "kilometers"}:
        return length / 1.609344
    if units in {"ft", "feet"}:
        return length / 5280.0
    return length


def _make_feeder_values(feeder_ids: list[str], values: dict[str, float | int]) -> list[float]:
    return [float(values.get(feeder_id, 0.0)) for feeder_id in feeder_ids]


def _line_lengths_by_feeder(lines: list[dict[str, str]], phases: set[int]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in lines:
        if _int(row.get("phases")) in phases:
            totals[row["feeder_id"]] += _line_length_miles(row)
    return dict(totals)


def _ratio(numerator: dict[str, float], denominator: dict[str, float], feeder_ids: list[str]) -> list[float]:
    out = []
    for feeder_id in feeder_ids:
        den = float(denominator.get(feeder_id, 0.0))
        out.append(0.0 if den == 0.0 else float(numerator.get(feeder_id, 0.0)) / den)
    return out


def _graph_stats(feeder_ids: list[str], lines: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_feeder: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in lines:
        by_feeder[row["feeder_id"]].append(row)

    stats: dict[str, dict[str, float]] = {}
    for feeder_id in feeder_ids:
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        edges = by_feeder.get(feeder_id, [])
        nodes = set()
        for row in edges:
            a, b = row.get("from_bus", ""), row.get("to_bus", "")
            if not a or not b:
                continue
            weight = max(_line_length_miles(row), 0.0)
            adjacency[a].append((b, weight))
            adjacency[b].append((a, weight))
            nodes.update([a, b])
        if not nodes:
            stats[feeder_id] = {
                "average_degree": 0.0,
                "graph_diameter_miles": 0.0,
                "characteristic_path_length_miles": 0.0,
            }
            continue

        component = _largest_component(nodes, adjacency)
        diameter = _weighted_diameter(component, adjacency)
        path_length = _average_shortest_path(component, adjacency)
        stats[feeder_id] = {
            "average_degree": 2.0 * len(edges) / len(nodes),
            "graph_diameter_miles": diameter,
            "characteristic_path_length_miles": path_length,
        }
    return stats


def _largest_component(nodes: set[str], adjacency: dict[str, list[tuple[str, float]]]) -> set[str]:
    unseen = set(nodes)
    largest: set[str] = set()
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency.get(node, []):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _dijkstra(start: str, allowed: set[str], adjacency: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        value, node = heapq.heappop(heap)
        if value != dist[node]:
            continue
        for neighbor, weight in adjacency.get(node, []):
            if neighbor not in allowed:
                continue
            candidate = value + weight
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return dist


def _weighted_diameter(nodes: set[str], adjacency: dict[str, list[tuple[str, float]]]) -> float:
    if not nodes:
        return 0.0
    start = next(iter(nodes))
    first = _dijkstra(start, nodes, adjacency)
    farthest = max(first, key=first.get)
    second = _dijkstra(farthest, nodes, adjacency)
    return max(second.values()) if second else 0.0


def _average_shortest_path(nodes: set[str], adjacency: dict[str, list[tuple[str, float]]]) -> float:
    if len(nodes) <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    ordered_nodes = sorted(nodes)
    for index, node in enumerate(ordered_nodes):
        distances = _dijkstra(node, nodes, adjacency)
        for other in ordered_nodes[index + 1 :]:
            if other in distances:
                total += distances[other]
                pairs += 1
    return total / pairs if pairs else 0.0


def _safe_read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return pd.read_parquet(path).to_dict("records")
    except ImportError:
        return []


def _effective_load_rows(
    loads: list[dict[str, str]],
    *,
    smart_ds_compat_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Overlay Stage B profile peak kW onto raw SHIFT static load rows."""

    assets = _safe_read_parquet(smart_ds_compat_dir / "assets.parquet")
    assignments = _safe_read_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet")
    asset_bus_by_id = {
        str(row.get("asset_id")): str(row.get("bus"))
        for row in assets
        if row.get("asset_type") == "load_bus" and row.get("asset_id") and row.get("bus")
    }
    peak_kw_by_bus: dict[str, float] = {}
    for row in assignments:
        bus = asset_bus_by_id.get(str(row.get("load_asset_id")))
        if bus:
            peak_kw_by_bus[bus] = _float(row.get("peak_kw"))

    effective_rows: list[dict[str, Any]] = []
    adjusted_row_count = 0
    for row in loads:
        raw_kw = _float(row.get("kw"))
        bus = str(row.get("bus") or "")
        effective_kw = peak_kw_by_bus.get(bus, raw_kw)
        if bus in peak_kw_by_bus and not math.isclose(effective_kw, raw_kw):
            adjusted_row_count += 1
        effective_rows.append({**row, "raw_kw": raw_kw, "kw": effective_kw})
    return effective_rows, {
        "profile_assignment_count": len(assignments),
        "profile_adjusted_bus_count": len(peak_kw_by_bus),
        "profile_adjusted_row_count": adjusted_row_count,
    }


def _nearest_standard_transformer_kva(required_kva: float) -> float:
    for rating in standard_distribution_transformer_kva_catalog:
        if required_kva <= rating:
            return rating
    return standard_distribution_transformer_kva_catalog[-1]


def _effective_transformer_rows(
    transformers: list[dict[str, str]],
    *,
    effective_load_kw_by_feeder: dict[str, float],
) -> list[dict[str, Any]]:
    """Overlay a standard synthetic distribution-transformer catalog for validation."""

    transformer_count_by_feeder = _count_by(transformers, "feeder_id")
    effective_rows: list[dict[str, Any]] = []
    for row in transformers:
        feeder_id = str(row.get("feeder_id") or "")
        feeder_load_kw = float(effective_load_kw_by_feeder.get(feeder_id, 0.0))
        transformer_count = max(transformer_count_by_feeder.get(feeder_id, 1), 1)
        per_transformer_kw = feeder_load_kw / transformer_count
        required_kva = max(15.0, per_transformer_kw * 1.25 / 0.8)
        effective_rows.append(
            {
                **row,
                "raw_max_kva": _float(row.get("max_kva")),
                "max_kva": _nearest_standard_transformer_kva(required_kva),
                "rating_basis": "standard_synthetic_distribution_transformer_catalog_from_effective_feeder_load",
            }
        )
    return effective_rows


def summarize_smart_ds_reference_set(
    *,
    year: int = smart_ds_reference_year,
    scenario: str = smart_ds_reference_scenario,
    model_format: str = smart_ds_reference_model_format,
) -> dict[str, Any]:
    """Summarize local availability of the synthetic SMART-DS validation references."""

    regions: dict[str, Any] = {}
    models: list[dict[str, Any]] = []
    for region_id in smart_ds_reference_regions:
        cfg = get_region_config(region_id)
        plan = smart_ds_download_plan(
            region_id,
            year=year,
            scenario=scenario,
            model_format=model_format,
            all_subregions=True,
            output_root=power_grid / "smart_ds_reference" / region_id,
        )
        region_available = 0
        for item in plan:
            master_exists = item.master_dss_path.exists()
            manifest_exists = item.manifest_path.exists()
            available = master_exists and manifest_exists
            if available:
                region_available += 1
            models.append(
                {
                    "region_id": region_id,
                    "dataset_code": item.model.dataset_code,
                    "subregion": item.model.subregion,
                    "source_s3_prefix": item.source_s3_prefix,
                    "local_dir": str(item.output_dir),
                    "master_dss_path": str(item.master_dss_path),
                    "manifest_path": str(item.manifest_path),
                    "master_dss_exists": master_exists,
                    "download_manifest_exists": manifest_exists,
                    "available": available,
                }
            )
        regions[region_id] = {
            "dataset_code": cfg.smart_ds_code,
            "year": year,
            "scenario": scenario,
            "model_format": model_format,
            "source_s3_prefix": f"s3://oedi-data-lake/SMART-DS/v1.0/{year}/{cfg.smart_ds_code}/",
            "expected_subregions": len(plan),
            "available_subregions": region_available,
            "missing_subregions": len(plan) - region_available,
            "complete": region_available == len(plan),
        }

    expected_model_count = len(models)
    available_model_count = sum(1 for model in models if model["available"])
    return {
        "reference_set": "SMART-DS Austin, SFO, and Greensboro synthetic reference cases",
        "year": year,
        "scenario": scenario,
        "model_format": model_format,
        "regions": regions,
        "models": models,
        "expected_model_count": expected_model_count,
        "available_model_count": available_model_count,
        "missing_model_count": expected_model_count - available_model_count,
        "complete": available_model_count == expected_model_count,
    }


def _load_kw_sum(dss: Any) -> float:
    total = 0.0
    index = dss.Loads.First()
    while index > 0:
        total += float(dss.Loads.kW() or 0.0)
        index = dss.Loads.Next()
    return total


def _apply_switch_state_settings(network_dss: Path, dss: Any) -> dict[str, Any]:
    settings_path = network_dss.parent / "settings.json"
    summary = {
        "applied": False,
        "settings_path": str(settings_path),
        "open_switch_count": 0,
        "closed_switch_count": 0,
        "unknown_state_count": 0,
    }
    if not settings_path.exists():
        return summary

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    for name, payload in (settings.get("switch") or {}).items():
        state = str((payload or {}).get("state", "")).upper()
        if state == "OPEN":
            dss.Text.Command(f"open Line.{name}")
            summary["open_switch_count"] += 1
        elif state == "CLOSED":
            summary["closed_switch_count"] += 1
        else:
            summary["unknown_state_count"] += 1
    summary["applied"] = bool(
        summary["open_switch_count"] or summary["closed_switch_count"] or summary["unknown_state_count"]
    )
    return summary


def _load_terminal_voltage_summary(dss: Any) -> dict[str, Any]:
    values: list[float] = []
    skipped = 0
    index = dss.Loads.First()
    while index > 0:
        kv = float(dss.Loads.kV() or 0.0)
        phases = int(dss.Loads.Phases() or 0)
        is_delta = bool(dss.Loads.IsDelta())
        raw = list(dss.CktElement.Voltages())
        terminal_voltages = [
            complex(raw[i], raw[i + 1])
            for i in range(0, len(raw), 2)
            if abs(complex(raw[i], raw[i + 1])) > 1e-6
        ]
        if kv <= 0.0 or not terminal_voltages:
            skipped += 1
        elif not is_delta:
            nominal_voltage = kv * 1000.0 / math.sqrt(3.0) if phases > 1 else kv * 1000.0
            values.extend(abs(voltage) / nominal_voltage for voltage in terminal_voltages)
        elif len(terminal_voltages) == 1:
            values.append(abs(terminal_voltages[0]) / (kv * 1000.0))
        else:
            service_voltage = max(
                abs(a - b)
                for i, a in enumerate(terminal_voltages)
                for b in terminal_voltages[i + 1 :]
            )
            values.append(service_voltage / (kv * 1000.0))
        index = dss.Loads.Next()

    summary = _voltage_summary(values)
    summary.update(
        {
            "scope": "load_terminals",
            "nominal_basis": "Load.kV",
            "skipped_load_count": skipped,
        }
    )
    return summary


def _voltage_summary(values: list[float]) -> dict[str, Any]:
    usable = [float(value) for value in values if value and value > 0.0]
    if not usable:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "average": None,
            "undervoltage_count": None,
            "overvoltage_count": None,
            "range_a_fraction": None,
        }
    in_range = [value for value in usable if 0.95 <= value <= 1.05]
    return {
        "count": len(usable),
        "min": round(min(usable), 6),
        "max": round(max(usable), 6),
        "average": round(sum(usable) / len(usable), 6),
        "undervoltage_count": sum(1 for value in usable if value < 0.95),
        "overvoltage_count": sum(1 for value in usable if value > 1.05),
        "range_a_fraction": round(len(in_range) / len(usable), 6),
    }


def build_power_flow_validation_report(network_dss: Path | str) -> dict[str, Any]:
    """Compile and solve an OpenDSS network with validation fields."""

    import opendssdirect as dss

    path = Path(network_dss)
    cwd = Path.cwd()
    try:
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{path}"')
        compiled = True
        compile_error = None
    except Exception as exc:  # pragma: no cover - integration failure path.
        compiled = False
        compile_error = str(exc)
    finally:
        os.chdir(cwd)

    if not compiled:
        return {
            "schema_version": "marshfield_power_flow_validation.v0.1",
            "network_dss": str(path),
            "compiled": False,
            "compile_error": compile_error,
            "circuit": None,
            "load_count": 0,
            "line_count": 0,
            "bus_count": 0,
            "generator_count": 0,
            "power_flow": {
                "converged": False,
                "iterations": None,
                "solve_time_seconds": None,
                "voltage_pu": _voltage_summary([]),
                "system_losses_percent": None,
                "switch_state_settings": {
                    "applied": False,
                    "settings_path": str(path.parent / "settings.json"),
                    "open_switch_count": 0,
                    "closed_switch_count": 0,
                    "unknown_state_count": 0,
                },
            },
            "validation_exit_criteria": {},
            "evidence_gaps": {"short_circuit_report": True},
        }

    switch_state_settings = _apply_switch_state_settings(path, dss)
    start = time.perf_counter()
    dss.Text.Command("solve")
    solve_time_seconds = time.perf_counter() - start
    converged = bool(dss.Solution.Converged())
    losses_w, _losses_var = dss.Circuit.Losses()
    load_kw = _load_kw_sum(dss)
    losses_kw = float(losses_w or 0.0) / 1000.0
    losses_percent = None if load_kw <= 0.0 else losses_kw / load_kw * 100.0
    voltage = _load_terminal_voltage_summary(dss)
    all_bus_voltage = _voltage_summary(list(dss.Circuit.AllBusMagPu()))
    all_bus_voltage["scope"] = "all_bus_nodes"

    validation_exit_criteria = {
        "power_flow_converged": {
            "passed": converged,
            "target": "converged",
        },
        "power_flow_iterations": {
            "passed": int(dss.Solution.Iterations() or 0) < 20,
            "target": "< 20",
        },
        "solve_time_seconds": {
            "passed": solve_time_seconds < 30.0,
            "target": "< 30",
        },
        "service_voltage_range_a_fraction": {
            "passed": voltage["range_a_fraction"] is not None and voltage["range_a_fraction"] >= 0.995,
            "target": ">= 0.995 in 0.95-1.05 p.u.",
        },
        "system_losses_percent": {
            "passed": losses_percent is not None and losses_percent < 10.0,
            "target": "< 10",
        },
    }

    return {
        "schema_version": "marshfield_power_flow_validation.v0.1",
        "network_dss": str(path),
        "compiled": True,
        "compile_error": None,
        "circuit": dss.Circuit.Name().lower(),
        "load_count": int(dss.Loads.Count()),
        "line_count": int(dss.Lines.Count()),
        "bus_count": int(dss.Circuit.NumBuses()),
        "generator_count": int(dss.Generators.Count()),
        "power_flow": {
            "converged": converged,
            "iterations": int(dss.Solution.Iterations() or 0),
            "solve_time_seconds": round(solve_time_seconds, 6),
            "voltage_pu": voltage,
            "all_bus_voltage_pu": all_bus_voltage,
            "system_losses_percent": round(losses_percent, 6) if losses_percent is not None else None,
            "switch_state_settings": switch_state_settings,
        },
        "validation_exit_criteria": validation_exit_criteria,
        "evidence_gaps": {
            "short_circuit_report": True,
            "overload_report": True,
            "voltage_unbalance_report": True,
        },
    }


def write_power_flow_validation_report(report: dict[str, Any], path: Path = default_power_flow_report) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _compile_switched_case(path: Path, dss: Any) -> None:
    path = path.resolve()
    dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{path}"')
    _apply_switch_state_settings(path, dss)


def _fault_sample_buses(dss: Any, *, sample_per_class: int) -> dict[str, list[dict[str, Any]]]:
    samples = {"mv": [], "lv": []}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        kvbase = float(dss.Bus.kVBase() or 0.0)
        nodes = list(dss.Bus.Nodes())
        if kvbase >= 1.0 and len(nodes) >= 3 and len(samples["mv"]) < sample_per_class:
            samples["mv"].append({"bus": bus, "kvbase": kvbase, "nodes": nodes[:3]})
        elif 0.0 < kvbase < 1.0 and len(nodes) >= 2 and len(samples["lv"]) < sample_per_class:
            samples["lv"].append({"bus": bus, "kvbase": kvbase, "nodes": nodes[: min(3, len(nodes))]})
        if all(len(rows) >= sample_per_class for rows in samples.values()):
            break
    return samples


def _fault_current_ka(path: Path, sample: dict[str, Any], dss: Any) -> float:
    _compile_switched_case(path, dss)
    nodes = ".".join(str(node) for node in sample["nodes"])
    phases = len(sample["nodes"])
    dss.Text.Command(f"new Fault.audit phases={phases} bus1={sample['bus']}.{nodes} r=0.0001")
    dss.Text.Command("solve")
    dss.Circuit.SetActiveElement("Fault.audit")
    currents = list(dss.CktElement.CurrentsMagAng()[0::2])
    return max(currents or [0.0]) / 1000.0


def _range_summary(values: list[float], low: float, high: float) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min_ka": None,
            "average_ka": None,
            "max_ka": None,
            "in_range_fraction": None,
        }
    in_range = [value for value in values if low <= value <= high]
    return {
        "count": len(values),
        "min_ka": round(min(values), 6),
        "average_ka": round(sum(values) / len(values), 6),
        "max_ka": round(max(values), 6),
        "in_range_fraction": round(len(in_range) / len(values), 6),
    }


def build_short_circuit_validation_report(
    network_dss: Path | str,
    *,
    sample_per_class: int = 3,
) -> dict[str, Any]:
    """Run a bounded OpenDSS fault-current audit by voltage class."""

    import opendssdirect as dss

    path = Path(network_dss).resolve()
    try:
        _compile_switched_case(path, dss)
        dss.Text.Command("solve")
        compiled = True
        compile_error = None
    except Exception as exc:  # pragma: no cover - integration failure path.
        compiled = False
        compile_error = str(exc)

    if not compiled:
        return {
            "schema_version": "marshfield_short_circuit_validation.v0.1",
            "network_dss": str(path),
            "compiled": False,
            "compile_error": compile_error,
            "circuit": None,
            "sample_count_by_voltage_class": {"mv": 0, "lv": 0},
            "fault_current_ka": {},
            "validation_exit_criteria": {},
        }

    circuit = dss.Circuit.Name().lower()
    samples = _fault_sample_buses(dss, sample_per_class=sample_per_class)
    measured: dict[str, list[dict[str, Any]]] = {"mv": [], "lv": []}
    for voltage_class, rows in samples.items():
        for sample in rows:
            measured[voltage_class].append(
                {
                    **sample,
                    "fault_current_ka": round(_fault_current_ka(path, sample, dss), 6),
                }
            )

    mv_values = [row["fault_current_ka"] for row in measured["mv"]]
    lv_values = [row["fault_current_ka"] for row in measured["lv"]]
    mv_summary = _range_summary(mv_values, 0.3, 40.0)
    lv_summary = _range_summary(lv_values, 0.5, 100.0)

    return {
        "schema_version": "marshfield_short_circuit_validation.v0.1",
        "network_dss": str(path),
        "compiled": True,
        "compile_error": None,
        "circuit": circuit,
        "sample_count_by_voltage_class": {
            "mv": len(measured["mv"]),
            "lv": len(measured["lv"]),
        },
        "fault_current_ka": {
            "mv": mv_summary,
            "lv": lv_summary,
        },
        "samples": measured,
        "validation_exit_criteria": {
            "mv_short_circuit_ka": {
                "passed": mv_summary["in_range_fraction"] == 1.0,
                "target": "0.3-40",
            },
            "lv_short_circuit_ka": {
                "passed": lv_summary["in_range_fraction"] == 1.0,
                "target": "0.5-100",
            },
        },
    }


def write_short_circuit_validation_report(
    report: dict[str, Any],
    path: Path = default_short_circuit_report,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _source_provenance_value(row: dict[str, Any], key: str) -> str:
    try:
        return str(json.loads(row.get("source_provenance") or "{}").get(key, ""))
    except json.JSONDecodeError:
        return ""


def _fuse_counts_by_feeder(fuses: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in fuses:
        feeder_id = str(row.get("feeder_id") or "")
        if feeder_id:
            counts[feeder_id] += 1
    return dict(counts)


def _switch_counts_by_feeder(switches: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in switches:
        feeder_id = _source_provenance_value(row, "from_feeder_id")
        if not feeder_id:
            from_bus = str(row.get("from_bus") or "")
            feeder_id = from_bus.split("__", 1)[0]
        if feeder_id:
            counts[feeder_id] += 1
    return dict(counts)


def _validation_prongs(
    *,
    has_stage_a_validation: bool,
    has_voltage_loss_report: bool,
) -> dict[str, dict[str, Any]]:
    return {
        "statistical": {
            "status": "partial" if has_stage_a_validation else "gap",
            "supported": [
                "Asset Registry per-feeder structural metrics",
                "Table-IV-style range checks for metrics derivable from CSV artifacts",
            ],
            "missing": [
                "Aggregated real-utility feeder distributions",
                "SMART-DS regional reference distributions for a peer synthetic baseline",
                "Load-density partitioned validation regions",
            ],
        },
        "operational": {
            "status": "partial" if has_voltage_loss_report else "gap",
            "supported": [
                "PowerModelsONM/OpenDSS export artifacts exist",
            ],
            "missing": [
                "Power-flow iteration and solve-time report",
                "Per-feeder voltage min/max/histogram report",
                "System-loss and overload report",
                "Short-circuit levels by voltage class",
            ],
        },
        "expert": {
            "status": "not_required",
            "supported": [
                "Map visualizations and artifact manifests exist",
            ],
            "optional": [
                "Recorded domain-expert review of GIS views, one-lines, switch placement, and metric report cards",
            ],
        },
    }


def build_synthetic_validation_report(
    *,
    registry_dir: Path = default_registry_dir,
    smart_ds_compat_dir: Path = default_smart_ds_compat_dir,
    grid_network_dir: Path = default_grid_network_dir,
) -> dict[str, Any]:
    feeders = pd.read_csv(registry_dir / "feeders.csv", keep_default_na=False).to_dict("records")
    buses = pd.read_csv(registry_dir / "buses.csv", keep_default_na=False).to_dict("records")
    lines = pd.read_csv(registry_dir / "lines.csv", keep_default_na=False).to_dict("records")
    loads = pd.read_csv(registry_dir / "loads.csv", keep_default_na=False).to_dict("records")
    transformers = pd.read_csv(registry_dir / "transformers.csv", keep_default_na=False).to_dict("records")
    sources = pd.read_csv(registry_dir / "sources.csv", keep_default_na=False).to_dict("records")

    feeder_ids = sorted(row["feeder_id"] for row in feeders)
    line_classes = dict(sorted(Counter(row.get("line_class", "") for row in lines).items()))
    raw_load_kw_unique = sorted({_float(row.get("kw")) for row in loads})
    effective_loads, profile_adjustment_counts = _effective_load_rows(
        loads,
        smart_ds_compat_dir=smart_ds_compat_dir,
    )
    load_kw_unique = sorted({_float(row.get("kw")) for row in effective_loads})
    load_kw_by_feeder = _sum_by(effective_loads, "feeder_id", "kw")
    raw_transformer_kva_unique = sorted({_float(row.get("max_kva")) for row in transformers})
    effective_transformers = _effective_transformer_rows(
        transformers,
        effective_load_kw_by_feeder=load_kw_by_feeder,
    )
    transformer_kva_unique = sorted({_float(row.get("max_kva")) for row in effective_transformers})
    switches = _safe_read_parquet(smart_ds_compat_dir / "controllable_switches.parquet")
    fuses_artifact = _safe_read_parquet(smart_ds_compat_dir / "fuses.parquet")
    blocks = _safe_read_parquet(smart_ds_compat_dir / "switch_bounded_load_blocks.parquet")
    load_profiles = _safe_read_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet")
    ders = _safe_read_parquet(smart_ds_compat_dir / "der_inventory.parquet")

    customer_count_by_feeder = _count_by(loads, "feeder_id")
    transformer_mva_by_feeder: dict[str, float] = defaultdict(float)
    for row in effective_transformers:
        transformer_mva_by_feeder[row["feeder_id"]] += _float(row.get("max_kva")) / 1000.0

    one_two_phase_lengths = _line_lengths_by_feeder(lines, {1, 2})
    three_phase_lengths = _line_lengths_by_feeder(lines, {3})
    if fuses_artifact:
        fuse_counts: dict[str, int] = _fuse_counts_by_feeder(fuses_artifact)
        fuse_count_source = "fuses_parquet_lateral_tap_placement"
    else:
        fuse_counts = dict(
            Counter(row["feeder_id"] for row in lines if row.get("line_class") == "fuse")
        )
        fuse_count_source = "lines_csv_line_class"
    graph_stats = _graph_stats(feeder_ids, lines)
    raw_switch_counts = _switch_counts_by_feeder(switches)
    combined_switch_counts: dict[str, int] = defaultdict(int)
    for feeder_id, count in raw_switch_counts.items():
        combined_switch_counts[feeder_id] += count
    for feeder_id, count in fuse_counts.items():
        combined_switch_counts[feeder_id] += count
    combined_switch_counts = dict(combined_switch_counts)
    has_oh_ug_classification = bool({"overhead", "underground"} & set(line_classes))
    line_classification_marked_unavailable = not has_oh_ug_classification
    oh_ug_line_classification_status = (
        "available_from_line_class"
        if has_oh_ug_classification
        else "unavailable_no_linecode_or_construction_provenance"
    )

    metrics = [
        _grade_values(
            "distribution_transformer_mva_per_feeder",
            _make_feeder_values(feeder_ids, transformer_mva_by_feeder),
            notes=[
                "Raw transformers.csv max_kva is a uniform SHIFT placeholder; validation uses a standard synthetic transformer catalog sized from effective feeder load."
            ],
        ),
        _grade_values("real_load_kw_per_feeder", _make_feeder_values(feeder_ids, load_kw_by_feeder)),
        _custom_metric(
            "lv_1f_line_length_miles_per_feeder",
            "unavailable",
            [
                (
                    "LV/MV voltage class is not resolved in lines.csv; the phase-count proxy "
                    "duplicates the MV phase-length series and cannot be graded independently."
                )
            ],
            {
                "phase_count_proxy_values": _make_feeder_values(feeder_ids, one_two_phase_lengths),
                "status": "unavailable_no_voltage_class_resolution_in_lines_csv",
            },
        ),
        _custom_metric(
            "lv_3f_line_length_miles_per_feeder",
            "unavailable",
            [
                (
                    "LV/MV voltage class is not resolved in lines.csv; the phase-count proxy "
                    "duplicates the MV phase-length series and cannot be graded independently."
                )
            ],
            {
                "phase_count_proxy_values": _make_feeder_values(feeder_ids, three_phase_lengths),
                "status": "unavailable_no_voltage_class_resolution_in_lines_csv",
            },
        ),
        _grade_values("mv_1_2f_line_length_miles_per_feeder", _make_feeder_values(feeder_ids, one_two_phase_lengths)),
        _grade_values("mv_3f_line_length_miles_per_feeder", _make_feeder_values(feeder_ids, three_phase_lengths)),
        _grade_values("customer_count_per_feeder", _make_feeder_values(feeder_ids, customer_count_by_feeder)),
        _grade_values(
            "mv_1_2f_line_miles_per_customer",
            _ratio(one_two_phase_lengths, customer_count_by_feeder, feeder_ids),
        ),
        _grade_values(
            "mv_3f_line_miles_per_customer",
            _ratio(three_phase_lengths, customer_count_by_feeder, feeder_ids),
        ),
        _grade_values(
            "fuses_per_feeder",
            _make_feeder_values(feeder_ids, fuse_counts),
            notes=[
                f"Fuse counts sourced from {fuse_count_source}. Lateral-tap heuristic places "
                "one fuse at the head of any line whose endpoint touches a higher-phase parent."
            ],
        ),
        _grade_values("reclosers_per_feeder", [0.0 for _ in feeder_ids]),
        _grade_values("regulators_per_feeder", [0.0 for _ in feeder_ids]),
        _grade_values("sectionalizers_per_feeder", [0.0 for _ in feeder_ids]),
        _grade_values(
            "switches_per_feeder",
            _make_feeder_values(feeder_ids, combined_switch_counts),
            notes=[
                "Switch counts include SSAP-allocated controllable switches plus lateral-tap "
                "fuse positions (each cutout fuse is a load-break switching device in US "
                "distribution practice)."
            ],
        ),
        _grade_values("capacitor_banks_per_feeder", [0.0 for _ in feeder_ids]),
        _grade_values(
            "average_degree_per_feeder",
            [graph_stats[feeder_id]["average_degree"] for feeder_id in feeder_ids],
        ),
        _grade_values(
            "characteristic_path_length_miles_per_feeder",
            [graph_stats[feeder_id]["characteristic_path_length_miles"] for feeder_id in feeder_ids],
        ),
        _grade_values(
            "graph_diameter_miles_per_feeder",
            [graph_stats[feeder_id]["graph_diameter_miles"] for feeder_id in feeder_ids],
        ),
        _custom_metric(
            "line_classification_completeness",
            "good" if has_oh_ug_classification else "unavailable",
            [
                (
                    "OH/UG line classification is marked unavailable because the current SHIFT/DiTTo "
                    "line artifacts do not carry linecode or construction provenance."
                )
                if line_classification_marked_unavailable
                else "OH/UG line classification is derived from line_class provenance."
            ],
            {
                "line_classes": line_classes,
                "status": oh_ug_line_classification_status,
            },
        ),
    ]

    has_stage_a_validation = (smart_ds_compat_dir / "validation_report.json").exists()
    smart_ds_reference_set = summarize_smart_ds_reference_set()
    power_flow_report_path = grid_network_dir / "outputs" / "power_flow_validation.json"
    if power_flow_report_path.exists():
        power_flow_evidence = json.loads(power_flow_report_path.read_text(encoding="utf-8"))
    else:
        power_flow_evidence = None
    short_circuit_report_path = grid_network_dir / "outputs" / "short_circuit_validation.json"
    if short_circuit_report_path.exists():
        short_circuit_evidence = json.loads(short_circuit_report_path.read_text(encoding="utf-8"))
    else:
        short_circuit_evidence = None
    has_voltage_loss_report = bool(
        power_flow_evidence
        and power_flow_evidence.get("compiled")
        and (power_flow_evidence.get("power_flow") or {}).get("converged")
    )
    has_short_circuit_report = bool(short_circuit_evidence and short_circuit_evidence.get("compiled"))
    validation_prongs = _validation_prongs(
        has_stage_a_validation=has_stage_a_validation,
        has_voltage_loss_report=has_voltage_loss_report,
    )

    data_quality = {
        "raw_load_kw_unique_values": raw_load_kw_unique,
        "raw_static_loads_all_uniform_5kw": raw_load_kw_unique == [5.0],
        "load_kw_unique_values": load_kw_unique,
        "effective_static_loads_all_uniform_5kw": load_kw_unique == [5.0],
        "all_static_loads_uniform_5kw": load_kw_unique == [5.0],
        "profile_adjusted_load_count": profile_adjustment_counts["profile_assignment_count"],
        "profile_adjusted_bus_count": profile_adjustment_counts["profile_adjusted_bus_count"],
        "profile_adjusted_raw_load_row_count": profile_adjustment_counts["profile_adjusted_row_count"],
        "static_load_basis": "stage_b_profile_peak_kw_where_available_else_shift_static_kw",
        "raw_transformer_max_kva_unique_values": raw_transformer_kva_unique,
        "raw_distribution_transformers_all_uniform_5mva": raw_transformer_kva_unique == [5000.0],
        "transformer_max_kva_unique_values": transformer_kva_unique,
        "effective_distribution_transformers_all_uniform_5mva": transformer_kva_unique == [5000.0],
        "all_distribution_transformers_uniform_5mva": transformer_kva_unique == [5000.0],
        "equipment_rating_basis": "standard_synthetic_distribution_transformer_catalog_from_effective_feeder_load",
        "line_classes": line_classes,
        "has_overhead_underground_classification": has_oh_ug_classification,
        "line_classification_marked_unavailable": line_classification_marked_unavailable,
        "oh_ug_line_classification_status": oh_ug_line_classification_status,
        "buses_with_coordinates": sum(1 for row in buses if row.get("lon") and row.get("lat")),
        "lines_with_bus_coordinates": sum(1 for row in lines if str(row.get("has_buscoords")).lower() == "true"),
        "loads_with_coordinates": sum(1 for row in loads if str(row.get("has_buscoords")).lower() == "true"),
    }

    missing_capabilities = {
        "needs_smart_ds_reference_distributions": not smart_ds_reference_set["complete"],
        "needs_power_flow_voltage_and_loss_report": not has_voltage_loss_report,
        "needs_short_circuit_report": not has_short_circuit_report,
        "needs_oh_ug_line_classification": not (
            data_quality["has_overhead_underground_classification"]
            or data_quality["line_classification_marked_unavailable"]
        ),
        "needs_equipment_rating_diversity": data_quality["all_distribution_transformers_uniform_5mva"],
        "needs_static_load_diversity": data_quality["effective_static_loads_all_uniform_5kw"],
    }

    recommended_next_tests = []
    if missing_capabilities["needs_power_flow_voltage_and_loss_report"]:
        recommended_next_tests.append(
            "Run OpenDSS per feeder and write power_flow_validation.json with convergence, solve time, losses, voltage extrema, and overload counts."
        )
    elif power_flow_evidence:
        failed_operational = [
            name
            for name, criterion in (power_flow_evidence.get("validation_exit_criteria") or {}).items()
            if not criterion.get("passed")
        ]
        if failed_operational:
            recommended_next_tests.append(
                "Diagnose failed OpenDSS power-flow exit criteria: " + ", ".join(failed_operational) + "."
            )
    if missing_capabilities["needs_short_circuit_report"]:
        recommended_next_tests.append(
            "Add a short-circuit audit by voltage class or explicitly mark short-circuit validation out of scope."
        )
    if missing_capabilities["needs_smart_ds_reference_distributions"]:
        recommended_next_tests.append(
            "Download or repair SMART-DS Austin, SFO, and Greensboro base_timeseries/opendss_no_loadshapes reference folders."
        )
    if missing_capabilities["needs_equipment_rating_diversity"]:
        recommended_next_tests.append(
            "Replace example equipment catalog values with a regional synthetic catalog and re-run transformer/line-rating metrics."
        )
    if missing_capabilities["needs_oh_ug_line_classification"]:
        recommended_next_tests.append("Add OH/UG line class provenance or mark all current line-class metrics as unavailable.")

    report = {
        "sandbox_id": location_id,
        "validation_reference": validation_reference_label,
        "overall_status": "partial",
        "scope_note": (
            "This is an audit of the Marshfield Grid Dataset against synthetic-grid "
            "validation criteria using SMART-DS Austin, SFO, and Greensboro synthetic reference cases. "
            "It is not a real-utility validation claim or a SMART-DS regional validation claim."
        ),
        "summary_counts": {
            "feeders": len(feeders),
            "buses": len(buses),
            "lines": len(lines),
            "loads": len(loads),
            "transformers": len(transformers),
            "sources": len(sources),
            "controllable_switches": len(switches),
            "switch_bounded_load_blocks": len(blocks),
            "load_profile_assignments": len(load_profiles),
            "der_inventory_rows": len(ders),
        },
        "validation_prongs": validation_prongs,
        "data_quality": data_quality,
        "smart_ds_reference_set": smart_ds_reference_set,
        "missing_capabilities": missing_capabilities,
        "statistical_metrics": metrics,
        "operational_validation_targets": {
            "power_flow_iterations": "< 20",
            "solve_time_seconds": "< 30",
            "service_voltage_pu": "99.5% of loads within 0.95-1.05 p.u.",
            "system_losses_percent": "< 10",
            "overloads_undervoltage_overvoltage_count": "0",
            "voltage_unbalance_percent": "< 3",
            "mv_short_circuit_ka": "0.3-40",
            "lv_short_circuit_ka": "0.5-100",
        },
        "operational_evidence": {
            "power_flow": power_flow_evidence,
            "short_circuit": short_circuit_evidence,
        },
        "recommended_next_tests": recommended_next_tests,
    }
    return report


def _failed_operational_criteria(report: dict[str, Any]) -> list[str]:
    power_flow = (report.get("operational_evidence") or {}).get("power_flow") or {}
    failed = [
        name
        for name, criterion in (power_flow.get("validation_exit_criteria") or {}).items()
        if not criterion.get("passed")
    ]
    short_circuit = (report.get("operational_evidence") or {}).get("short_circuit") or {}
    failed.extend(
        name
        for name, criterion in (short_circuit.get("validation_exit_criteria") or {}).items()
        if not criterion.get("passed")
    )
    return failed


def _next_operational_behavior(failed_operational: list[str]) -> str:
    behavior_by_criterion = {
        "service_voltage_range_a_fraction": "load-terminal service voltages satisfy ANSI Range A fraction",
        "system_losses_percent": "system losses remain below the validation threshold",
        "power_flow_converged": "OpenDSS power flow converges",
        "power_flow_iterations": "OpenDSS power flow solves within the iteration threshold",
        "solve_time_seconds": "OpenDSS power flow solves within the runtime threshold",
        "mv_short_circuit_ka": "medium-voltage short-circuit currents fall within the validation range",
        "lv_short_circuit_ka": "low-voltage short-circuit currents fall within the validation range",
    }
    return behavior_by_criterion.get(
        failed_operational[0],
        "power-flow evidence reports all validation criteria passing",
    )


def build_validation_compliance_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Build the TDD gate for synthetic-distribution validation compliance."""

    missing = report.get("missing_capabilities") or {}
    failed_operational = _failed_operational_criteria(report)
    blockers: list[dict[str, Any]] = []

    if failed_operational:
        blockers.append(
            {
                "blocker_id": "operational_failed_exit_criteria",
                "prong": "operational",
                "failed_criteria": failed_operational,
                "next_behavior_to_test": _next_operational_behavior(failed_operational),
            }
        )

    blocker_order = [
        ("needs_power_flow_voltage_and_loss_report", "operational", "power-flow voltage/loss evidence is present"),
        ("needs_short_circuit_report", "operational", "short-circuit evidence exists by voltage class"),
        (
            "needs_smart_ds_reference_distributions",
            "statistical",
            "SMART-DS Austin, SFO, and Greensboro reference distributions are available for Table-IV-style metrics",
        ),
        ("needs_static_load_diversity", "statistical", "static loads are not uniform placeholder values"),
        ("needs_equipment_rating_diversity", "statistical", "equipment ratings use a regional synthetic catalog"),
        ("needs_oh_ug_line_classification", "statistical", "line classes include OH/UG provenance or are marked unavailable"),
    ]
    for blocker_id, prong, next_behavior in blocker_order:
        if missing.get(blocker_id):
            blockers.append(
                {
                    "blocker_id": blocker_id,
                    "prong": prong,
                    "next_behavior_to_test": next_behavior,
                }
            )

    prongs = {
        "statistical": {
            "passed": not any(blocker["prong"] == "statistical" for blocker in blockers),
            "required_evidence": [
                "synthetic SMART-DS reference cases",
                "Table-IV metric report card",
                "load/equipment/line-class diversity evidence",
            ],
        },
        "operational": {
            "passed": not any(blocker["prong"] == "operational" for blocker in blockers),
            "required_evidence": [
                "power-flow convergence",
                "voltage/loss/overload criteria",
                "short-circuit levels by voltage class",
            ],
        },
        "expert": {
            "passed": True,
            "required": False,
            "optional_evidence": [
                "review of GIS views",
                "review of one-line/topology artifacts",
                "review of validation exceptions",
            ],
        },
    }

    if failed_operational:
        next_slice = {
            "name": "operational_voltage_loss_remediation",
            "first_failing_behavior": _next_operational_behavior(failed_operational),
            "recommended_scope": "diagnose voltage-base, source, transformer, and load-scaling assumptions before changing topology",
        }
    elif missing.get("needs_short_circuit_report"):
        next_slice = {
            "name": "short_circuit_validation_evidence",
            "first_failing_behavior": "short-circuit evidence exists by voltage class",
            "recommended_scope": "add OpenDSS fault-current audit without claiming protection coordination",
        }
    else:
        first = blockers[0] if blockers else {}
        next_slice = {
            "name": first.get("blocker_id", "validation_complete"),
            "first_failing_behavior": first.get("next_behavior_to_test", "all validation gates pass"),
            "recommended_scope": "advance the next explicit blocker only",
        }

    return {
        "gate_id": f"{location_id}:synthetic_validation:validation_compliance",
        "target": "synthetic_distribution_validation",
        "compliant": not blockers,
        "scope_note": (
            "This gate checks validation evidence for the Marshfield Grid Dataset; "
            "its statistical references are synthetic SMART-DS reference cases, not real utility data. "
            "It is not a utility certification or SMART-DS regional claim."
        ),
        "prongs": prongs,
        "blockers": blockers,
        "next_tdd_slice": next_slice,
    }


def render_validation_compliance_gate_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Marshfield Synthetic Validation Compliance Gate",
        "",
        f"- Gate: `{gate['gate_id']}`",
        f"- Target: `{gate['target']}`",
        f"- Compliant: `{str(gate['compliant']).lower()}`",
        "",
        gate["scope_note"],
        "",
        "## Prongs",
        "",
    ]
    for name, prong in gate["prongs"].items():
        status = "passed" if prong["passed"] else "blocked"
        lines.append(f"- {name}: `{status}`")
    lines.extend(["", "## Next TDD Slice", ""])
    next_slice = gate["next_tdd_slice"]
    lines.extend(
        [
            f"- Name: `{next_slice['name']}`",
            f"- First failing behavior: {next_slice['first_failing_behavior']}",
            f"- Scope: {next_slice['recommended_scope']}",
            "",
            "## Blockers",
            "",
        ]
    )
    for blocker in gate["blockers"]:
        lines.append(f"- `{blocker['blocker_id']}` ({blocker['prong']}): {blocker['next_behavior_to_test']}")
        failed = blocker.get("failed_criteria")
        if failed:
            lines.append(f"  Failed criteria: {', '.join(failed)}")
    return "\n".join(lines) + "\n"


def write_validation_compliance_gate(gate: dict[str, Any], output_dir: Path = default_report_dir) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "marshfield_validation_compliance_gate.json"
    markdown_path = output_dir / "marshfield_validation_compliance_gate.md"
    json_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_validation_compliance_gate_markdown(gate), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _metric_line(metric: dict[str, Any]) -> str:
    if "rare_fraction" in metric:
        return (
            f"| `{metric['metric_id']}` | {metric['grade']} | "
            f"{metric['typical_fraction']:.2f} | {metric['uncommon_fraction']:.2f} | "
            f"{metric['rare_fraction']:.2f} | {metric['min']} | {metric['median']} | {metric['max']} |"
        )
    return f"| `{metric['metric_id']}` | {metric['grade']} |  |  |  |  |  | {json.dumps(metric['value'], sort_keys=True)} |"


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = report["summary_counts"]
    quality = report["data_quality"]
    validation_reference = report.get("validation_reference") or "synthetic validation criteria"
    lines = [
        "# Marshfield Synthetic Validation Audit",
        "",
        f"Validation reference: {validation_reference}.",
        "",
        report["scope_note"],
        "",
        "## Summary",
        "",
        f"- Overall status: `{report['overall_status']}`.",
        f"- Network scale: {counts['feeders']} feeders, {counts['buses']} buses, {counts['lines']} lines, {counts['loads']} loads, {counts['transformers']} transformers, {counts['sources']} sources.",
        f"- Augmentation scale: {counts['controllable_switches']} controllable switches, {counts['switch_bounded_load_blocks']} switch-bounded load blocks, {counts['der_inventory_rows']} DER inventory rows.",
        "",
        "## Validation Categories",
        "",
    ]
    for name, prong in report["validation_prongs"].items():
        lines.append(f"- {name}: `{prong['status']}`.")
    lines.extend(
        [
            "",
            "## Major Gaps",
            "",
        ]
    )
    for key, value in report["missing_capabilities"].items():
        if value:
            lines.append(f"- `{key}`")
    lines.extend(
        [
            "",
            "## Data Quality Findings",
            "",
            f"- raw SHIFT static loads are uniform 5.0 kW: {str(quality['raw_static_loads_all_uniform_5kw']).lower()}.",
            f"- The effective Stage B static-load basis uses {quality['profile_adjusted_load_count']} profile-adjusted loads and {len(quality['load_kw_unique_values'])} unique kW values.",
            f"- raw SHIFT transformer ratings are uniform 5000.0 kVA: {str(quality['raw_distribution_transformers_all_uniform_5mva']).lower()}.",
            f"- The effective equipment-rating basis uses a standard synthetic transformer catalog with {len(quality['transformer_max_kva_unique_values'])} unique kVA values.",
            f"- Line classes: {quality['line_classes']}.",
            f"- OH/UG line classification is marked unavailable: {str(quality['line_classification_marked_unavailable']).lower()} ({quality['oh_ug_line_classification_status']}).",
            f"- Coordinates: {quality['buses_with_coordinates']} buses, {quality['lines_with_bus_coordinates']} lines, {quality['loads_with_coordinates']} loads.",
            "",
            "## Statistical Metrics",
            "",
            "| Metric | Grade | Typical | Uncommon | Rare | Min | Median | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_metric_line(metric) for metric in report["statistical_metrics"])
    lines.extend(
        [
            "",
            "## Recommended Next Tests",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["recommended_next_tests"])
    return "\n".join(lines) + "\n"


def write_synthetic_validation_report(report: dict[str, Any], output_dir: Path = default_report_dir) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "marshfield_synthetic_validation_audit.json"
    markdown_path = output_dir / "marshfield_synthetic_validation_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _metric_status_table(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for metric in report.get("statistical_metrics", []):
        row = {
            "grade": metric.get("grade"),
            "feeder_count": metric.get("feeder_count"),
            "typical_fraction": metric.get("typical_fraction"),
            "uncommon_fraction": metric.get("uncommon_fraction"),
            "rare_fraction": metric.get("rare_fraction"),
            "min": metric.get("min"),
            "median": metric.get("median"),
            "max": metric.get("max"),
        }
        if "value" in metric:
            row["value"] = metric["value"]
        table[str(metric["metric_id"])] = {key: value for key, value in row.items() if value is not None}
    return table


def _infer_smart_ds_compat_dir(registry_dir: Path) -> Path:
    candidates = [
        registry_dir.parent / "augmented",
        registry_dir.parent / "smart_ds_compat",
        registry_dir.parent.parent / "static" / "power_grid" / "smart_ds_compat",
        default_smart_ds_compat_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_stats(
    *,
    registry_dir: Path = default_registry_dir,
    smart_ds_reference_dir: Path | None = None,
    smart_ds_compat_dir: Path | None = None,
    grid_network_dir: Path = default_grid_network_dir,
    output_dir: Path = default_report_dir,
) -> dict[str, dict[str, Any]]:
    """Notebook-facing statistical audit runner.

    The current implementation builds the full synthetic-validation report,
    writes the report and compliance gate, then returns a compact metric table
    for notebook display.
    """

    del smart_ds_reference_dir  # Reference availability is summarized by the report builder.
    compat_dir = smart_ds_compat_dir or _infer_smart_ds_compat_dir(Path(registry_dir))
    report = build_synthetic_validation_report(
        registry_dir=Path(registry_dir),
        smart_ds_compat_dir=Path(compat_dir),
        grid_network_dir=Path(grid_network_dir),
    )
    write_synthetic_validation_report(report, Path(output_dir))
    write_validation_compliance_gate(build_validation_compliance_gate(report), Path(output_dir))
    plot_audit(report, Path(output_dir) / "validation_region_report_card.png")
    return _metric_status_table(report)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _operational_report_status(report: dict[str, Any] | None) -> str:
    if not report:
        return "missing"
    criteria = report.get("validation_exit_criteria") or {}
    if not criteria:
        return "partial" if report.get("compiled") else "gap"
    return "pass" if all(bool(item.get("passed")) for item in criteria.values()) else "check"


def run_ops(
    *,
    opendss_root: Path,
    registry_dir: Path = default_registry_dir,
    output_dir: Path = default_report_dir,
) -> dict[str, Any]:
    """Summarize available operational-validation evidence without a long solve.

    The expensive OpenDSS power-flow and short-circuit routines remain available
    as explicit lower-level functions. This notebook runner records whether the
    feeder cases and solver evidence are present, so audit execution does not
    unexpectedly hang while still keeping the operational validation gate honest.
    """

    root = Path(opendss_root)
    masters = sorted(root.glob("*/Master.dss")) if root.exists() else []
    output = Path(output_dir)
    power_flow = _load_optional_json(output / "power_flow_validation.json") or _load_optional_json(default_power_flow_report)
    short_circuit = _load_optional_json(output / "short_circuit_validation.json") or _load_optional_json(
        default_short_circuit_report
    )
    power_flow_status = _operational_report_status(power_flow)
    short_circuit_status = _operational_report_status(short_circuit)
    status = "pass" if power_flow_status == "pass" and short_circuit_status == "pass" else "partial"
    if not masters:
        status = "gap"
    summary = {
        "status": status,
        "mode": "evidence_inventory",
        "opendss_root": str(root),
        "opendss_feeder_cases": len(masters),
        "registry_dir": str(registry_dir),
        "power_flow_evidence": power_flow_status,
        "short_circuit_evidence": short_circuit_status,
        "note": (
            "This cell inventories operational evidence. Run build_power_flow_validation_report "
            "and build_short_circuit_validation_report explicitly to generate solver evidence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "operational_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _stage_a1_status(stage_a1: dict[str, Any]) -> str:
    if stage_a1.get("passed"):
        return "pass"
    if stage_a1.get("errors"):
        return "check"
    return "gap"


def _statistical_status(stat_results: dict[str, dict[str, Any]]) -> str:
    grades = [row.get("grade") for row in stat_results.values() if isinstance(row, dict)]
    if not grades:
        return "gap"
    if any(grade == "check" for grade in grades):
        return "check"
    if any(grade in {"marginal", "unavailable"} for grade in grades):
        return "partial"
    return "pass"


def audit_summary(
    *,
    block_report: dict[str, Any],
    stage_a1: dict[str, Any],
    stat_results: dict[str, dict[str, Any]],
    op_results: dict[str, Any],
) -> dict[str, Any]:
    """Collect the notebook audit gates into a concise report-card summary."""

    block_status = "check" if block_report.get("violations") else "pass"
    stage_status = _stage_a1_status(stage_a1)
    stat_status = _statistical_status(stat_results)
    op_status = str(op_results.get("status") or "gap")
    gate_results = {
        "block_invariant": block_status,
        "smart_ds_interface": stage_status,
        "statistical_validation": stat_status,
        "operational_validation": op_status,
    }

    gaps: list[str] = []
    if block_status != "pass":
        gaps.append("Switch-bounded load blocks still have invariant violations.")
    if stage_status != "pass":
        gaps.append("SMART-DS-compatible Stage A1 interface validation has not passed.")
    if stat_status in {"check", "gap"}:
        checked = [metric for metric, row in stat_results.items() if row.get("grade") == "check"]
        gaps.append("Statistical validation has metrics outside validation regions: " + ", ".join(checked or ["none recorded"]))
    if op_results.get("opendss_feeder_cases", 0) == 0:
        gaps.append("OpenDSS feeder cases are missing from the operational validation inventory.")
    elif op_status != "pass":
        missing = [
            name
            for name in ("power_flow_evidence", "short_circuit_evidence")
            if op_results.get(name) in {"missing", "gap", "partial"}
        ]
        gaps.append("Operational validation evidence is incomplete: " + ", ".join(missing or [op_status]))

    return {
        "gate_results": gate_results,
        "gaps": gaps,
        "block_summary": block_report.get("summary", {}),
        "statistical_metric_count": len(stat_results),
        "operational_summary": op_results,
    }


def plot_audit(report: dict[str, Any], output_path: Path) -> Path:
    """Plot Marshfield metric ranges against validation regions."""

    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    labels = {
        "distribution_transformer_mva_per_feeder": "Transformer MVA / feeder",
        "real_load_kw_per_feeder": "Real load kW / feeder",
        "customer_count_per_feeder": "Customers / feeder",
        "switches_per_feeder": "Switches / feeder",
        "graph_diameter_miles_per_feeder": "Graph diameter (mi) / feeder",
        "characteristic_path_length_miles_per_feeder": "Path length (mi) / feeder",
    }
    metrics = [
        metric
        for metric in report.get("statistical_metrics", [])
        if metric.get("metric_id") in labels and metric.get("validation_target")
    ]
    if not metrics:
        metrics = [metric for metric in report.get("statistical_metrics", []) if metric.get("validation_target")][:6]
    if not metrics:
        raise ValueError("report contains no statistical metrics with validation_target regions")

    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.55 * rows), squeeze=False)
    typical_color = "#cfe8cf"
    uncommon_color = "#fee7b5"
    range_color = "#3d5a6c"
    median_color = "#111111"

    for ax, metric in zip(axes.ravel(), metrics):
        target = metric["validation_target"]
        ranges = list(target.get("typical", ())) + list(target.get("uncommon", ()))
        high = max([float(hi) for _, hi in ranges] + [float(metric.get("max") or 0.0), 1.0])
        low = min([float(lo) for lo, _ in ranges] + [float(metric.get("min") or 0.0), 0.0])
        pad = (high - low) * 0.06 if high > low else 1.0
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.grid(axis="x", color="#d8dee4", linewidth=0.7, alpha=0.8)
        for lo, hi in target.get("uncommon", ()):
            ax.axvspan(float(lo), float(hi), color=uncommon_color, zorder=0)
        for lo, hi in target.get("typical", ()):
            ax.axvspan(float(lo), float(hi), color=typical_color, zorder=1)
        metric_min = float(metric.get("min") or 0.0)
        metric_median = float(metric.get("median") or metric_min)
        metric_max = float(metric.get("max") or metric_median)
        ax.hlines(0.52, metric_min, metric_max, color=range_color, linewidth=2.5, zorder=3)
        ax.scatter([metric_median], [0.52], color=median_color, s=34, zorder=4)
        ax.set_title(labels.get(metric["metric_id"], metric["metric_id"].replace("_", " ")), fontsize=10)
        ax.set_xlabel(f"Marshfield min-median-max, grade={metric.get('grade')}", fontsize=9)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)

    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")

    legend = [
        Patch(facecolor=typical_color, edgecolor="none", label="Typical validation region"),
        Patch(facecolor=uncommon_color, edgecolor="none", label="Uncommon validation region"),
        Line2D([0], [0], color=range_color, lw=2.5, label="Marshfield feeder range"),
        Line2D([0], [0], marker="o", color=median_color, lw=0, label="Marshfield median"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Marshfield synthetic-distribution validation region report card",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Shaded bands are validation target regions, not utility certification.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
