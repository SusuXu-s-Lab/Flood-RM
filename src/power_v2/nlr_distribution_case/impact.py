"""Flood exposure and native ERAD fragility adapters.

The stochastic state model remains

    X[a,t,m] ~ Bernoulli(F_tau(a)(d[a,t]))

but ``F`` is evaluated through ERAD's native ``FragilityCurve`` /
``ProbabilityFunction`` machinery rather than a local hand-written CDF.  Local
code only maps study asset labels to ERAD ``AssetTypes`` and samples external
SFINCS grids into per-asset depths.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd

from .core import parse_float
from .native import NativeDependencyError, asset_system_from_gdm, require_module, run_erad_hazard


class FragilityModel(Protocol):
    """Small protocol consumed by Stage A2 Monte Carlo state sampling."""

    def probability(self, local_asset_type: str, depth_m: float | None) -> float: ...


@dataclass(frozen=True)
class NativeEradCurve:
    """One ERAD curve bound to a local asset type."""

    local_asset_type: str
    erad_asset_type: Any
    curve: Any

    def probability(self, depth_m: float | None) -> float:
        if depth_m is None:
            return 0.0
        depth = float(depth_m)
        if not math.isfinite(depth) or depth < 0.0:
            return 0.0
        Distance = _distance_quantity()
        value = self.curve.prob_function.prob_model.probability(Distance(depth, "meter"))
        return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class NativeEradFragilityModel:
    """ERAD-backed depth-fragility model for local Grid Dataset asset types."""

    curves_by_local_type: Mapping[str, NativeEradCurve]

    @classmethod
    def from_csv(cls, curves_csv: str | Path, mapping_csv: str | Path) -> "NativeEradFragilityModel":
        """Build ERAD ``FragilityCurve`` objects from reviewed CSV parameters.

        The CSV schema mirrors the previous artifact, but probability evaluation
        is delegated to ERAD's ``ProbabilityFunctionBuilder``/SciPy path.
        """

        mapping = load_asset_type_mapping(mapping_csv)
        by_erad: dict[str, dict[str, str]] = {}
        with Path(curves_csv).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                by_erad[str(row["erad_asset_type"])] = row
        curves: dict[str, NativeEradCurve] = {}
        for local_type, erad_type_name in mapping.items():
            if erad_type_name not in by_erad:
                raise KeyError(f"no curve row for ERAD asset type {erad_type_name!r}")
            row = by_erad[erad_type_name]
            curves[local_type] = _curve_from_row(local_type, erad_type_name, row)
        return cls(curves)

    @classmethod
    def from_default_erad_curves(cls, mapping: Mapping[str, str] | None = None) -> "NativeEradFragilityModel":
        """Bind local asset types to ERAD's packaged default fragility curves."""

        defaults = require_module("erad.default_fragility_curves", package="nrel-erad", purpose="ERAD default fragility curves")
        curve_sets = getattr(defaults, "DEFAULT_FRAGILTY_CURVES", None) or getattr(defaults, "DEFAULT_FRAGILITY_CURVES", None)
        if not curve_sets:
            raise NativeDependencyError("ERAD did not expose DEFAULT_FRAGILTY_CURVES / DEFAULT_FRAGILITY_CURVES")
        local_mapping = dict(mapping or default_local_to_erad_asset_type())
        curves: dict[str, NativeEradCurve] = {}
        for local_type, erad_name in local_mapping.items():
            asset_type = _asset_type_enum(erad_name)
            curve = _find_default_curve(curve_sets, asset_type)
            curves[local_type] = NativeEradCurve(local_type, asset_type, curve)
        return cls(curves)

    def probability(self, local_asset_type: str, depth_m: float | None) -> float:
        key = line_local_asset_type(local_asset_type) if str(local_asset_type).startswith("line_") is False and str(local_asset_type) in {"overhead", "underground", "fuse"} else str(local_asset_type)
        try:
            curve = self.curves_by_local_type[key]
        except KeyError as exc:
            raise KeyError(f"no ERAD fragility curve mapped for local asset type {local_asset_type!r}") from exc
        return curve.probability(depth_m)


def default_local_to_erad_asset_type() -> dict[str, str]:
    """Default local Grid Dataset asset-type mapping to ERAD ``AssetTypes``."""

    return {
        "source": "substation",
        "transformer": "transformer_pad_mount",
        "load_bus": "distribution_junction_box",
        "underground_line_proxy": "distribution_underground_cables",
        "line_underground": "distribution_underground_cables",
        "fuse_proxy": "switch",
        "line_fuse": "switch",
        "overhead_line": "distribution_overhead_lines",
        "line_overhead": "distribution_overhead_lines",
        "line": "distribution_overhead_lines",
        "line_other": "distribution_overhead_lines",
    }


@lru_cache(maxsize=None)
def native_fragility_model(curves_csv: str | Path | None = None, mapping_csv: str | Path | None = None) -> NativeEradFragilityModel:
    """Return the ERAD-native fragility model used by default Stage A2 runs."""

    if curves_csv is not None and mapping_csv is not None:
        return NativeEradFragilityModel.from_csv(curves_csv, mapping_csv)
    return NativeEradFragilityModel.from_default_erad_curves()


def failure_probability(
    local_asset_type: str,
    depth_m: float | None,
    *,
    model: FragilityModel | None = None,
    curves_csv: str | Path | None = None,
    mapping_csv: str | Path | None = None,
) -> float:
    """ERAD-native failure probability for a local Grid Dataset asset type."""

    active = model or native_fragility_model(curves_csv, mapping_csv)
    return active.probability(local_asset_type, depth_m)


def line_local_asset_type(line_class: str | None) -> str:
    value = (line_class or "").strip().lower()
    if value == "underground":
        return "line_underground"
    if value == "fuse":
        return "line_fuse"
    if value == "overhead":
        return "line_overhead"
    return "line_other"


@dataclass(frozen=True)
class AssetPoint:
    asset_id: str
    asset_type: str
    feeder_id: str
    lon: float
    lat: float
    label: str


def load_asset_points(registry_or_augmented_dir: str | Path, *, include_lines: bool = False) -> list[AssetPoint]:
    """Load flood-relevant point assets from Stage A1 Parquet or registry CSVs."""

    root = Path(registry_or_augmented_dir)
    assets_parquet = root / "assets.parquet"
    if assets_parquet.exists():
        rows = pd.read_parquet(assets_parquet).to_dict("records")
        line_types = {"line", "overhead_line", "underground_line_proxy"}
        points = []
        for row in rows:
            lon = parse_float(row.get("lon"))
            lat = parse_float(row.get("lat"))
            asset_type = str(row.get("asset_type") or "")
            if lon is None or lat is None or row.get("coordinate_status") != "valid":
                continue
            if not row.get("is_flood_relevant") and not (include_lines and asset_type in line_types):
                continue
            asset_id = str(row["asset_id"])
            points.append(AssetPoint(asset_id, asset_type, str(row.get("feeder_id") or ""), lon, lat, f"{asset_type} {asset_id}"))
        return sorted(points, key=lambda item: item.asset_id)

    points: list[AssetPoint] = []
    for table, asset_type, x, y, name in [
        ("load_buses.csv", "load_bus", "lon", "lat", "bus"),
        ("transformers.csv", "transformer", "location_lon", "location_lat", "transformer_name"),
        ("sources.csv", "source", "lon", "lat", "source_name"),
    ]:
        path = root / table
        if not path.exists():
            continue
        for row in pd.read_csv(path, keep_default_na=False).to_dict("records"):
            lon, lat = parse_float(row.get(x)), parse_float(row.get(y))
            if lon is None or lat is None:
                continue
            points.append(AssetPoint(str(row[name]), asset_type, str(row.get("feeder_id") or ""), lon, lat, f"{asset_type} {row[name]}"))
    if include_lines and (root / "lines.csv").exists():
        for row in pd.read_csv(root / "lines.csv", keep_default_na=False).to_dict("records"):
            coords = [parse_float(row.get(k)) for k in ("from_lon", "from_lat", "to_lon", "to_lat")]
            if any(v is None for v in coords):
                continue
            points.append(
                AssetPoint(
                    str(row["line_name"]),
                    line_local_asset_type(row.get("line_class")),
                    str(row.get("feeder_id") or ""),
                    (coords[0] + coords[2]) / 2.0,  # type: ignore[operator]
                    (coords[1] + coords[3]) / 2.0,  # type: ignore[operator]
                    f"{row.get('line_class', 'line')} line {row['line_name']}",
                )
            )
    return sorted(points, key=lambda item: item.asset_id)


def compute_asset_impacts(
    event_dir: str | Path,
    *,
    registry_dir: str | Path,
    event_id: str | None = None,
    fragility_model: FragilityModel | None = None,
    probability_threshold: float = 0.50,
    max_sample_distance_m: float = 150.0,
    include_lines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-asset flood exposure and ERAD-native fragility impact rows."""

    assets = load_asset_points(registry_dir, include_lines=include_lines)
    depths, distances = sample_sfincs_peak_depths(event_dir, assets, max_sample_distance_m=max_sample_distance_m)
    model = fragility_model or native_fragility_model()
    rows = []
    for asset, depth_m, distance_m in zip(assets, depths, distances, strict=True):
        probability = model.probability(asset.asset_type, depth_m)
        rows.append({
            "event_id": event_id or Path(event_dir).name,
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "feeder_id": asset.feeder_id,
            "lon": asset.lon,
            "lat": asset.lat,
            "peak_depth_m": depth_m,
            "nearest_grid_distance_m": distance_m,
            "failure_probability": probability,
            "affected_probability": probability,
            "affected": probability >= probability_threshold,
            "affected_probability_threshold": probability_threshold,
            "label": asset.label,
        })
    summary = summarize_impacts(rows, event_id=event_id or Path(event_dir).name)
    summary.update({"event_dir": str(event_dir), "probability_threshold": probability_threshold, "max_sample_distance_m": max_sample_distance_m, "include_lines": include_lines})
    return rows, summary


def summarize_impacts(rows: Iterable[Mapping[str, Any]], *, event_id: str) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    row_list = list(rows)
    for row in row_list:
        asset_type = str(row["asset_type"])
        bucket = by_type.setdefault(asset_type, {"asset_count": 0, "affected_count": 0, "expected_affected_count": 0.0, "max_failure_probability": 0.0, "max_peak_depth_m": 0.0})
        probability = float(row.get("failure_probability") or 0.0)
        depth = float(row.get("peak_depth_m") or 0.0)
        bucket["asset_count"] += 1
        bucket["affected_count"] += int(bool(row.get("affected")))
        bucket["expected_affected_count"] += probability
        bucket["max_failure_probability"] = max(bucket["max_failure_probability"], probability)
        bucket["max_peak_depth_m"] = max(bucket["max_peak_depth_m"], depth)
    return {
        "event_id": event_id,
        "asset_count": len(row_list),
        "affected_count": sum(int(bool(row.get("affected"))) for row in row_list),
        "expected_affected_count": sum(float(row.get("failure_probability") or 0.0) for row in row_list),
        "by_asset_type": by_type,
    }


def run_erad_simulation_from_gdm(distribution_system: Any, hazard_system: Any, *, curve_set: str | None = None) -> Any:
    """Use ERAD's native GDM bridge and hazard simulator."""

    return run_erad_hazard(asset_system_from_gdm(distribution_system), hazard_system, curve_set=curve_set)


def sample_sfincs_peak_depths(
    event_dir: str | Path,
    assets: Iterable[AssetPoint],
    *,
    max_sample_distance_m: float = 150.0,
    model_crs: str = "EPSG:26919",
) -> tuple[list[float | None], list[float]]:
    """Nearest-cell peak-depth sampler for a completed ``sfincs_map.nc``."""

    np, xr, Transformer, cKDTree = _require_geo_stack()
    path = Path(event_dir) / "sfincs_map.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        for name in ["x", "y", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{path} missing {name}")
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        peak_depth = _sfincs_peak_depth_grid(ds)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & active & np.isfinite(peak_depth)
    grid_xy = np.column_stack([x[valid], y[valid]])
    if len(grid_xy) == 0:
        raise RuntimeError(f"{path} has no active finite depth cells")
    tree = cKDTree(grid_xy)
    to_model = Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
    asset_list = list(assets)
    asset_xy = np.asarray([to_model.transform(asset.lon, asset.lat) for asset in asset_list], dtype=float)
    distances, nearest = tree.query(asset_xy, k=1)
    flat_depth = peak_depth[valid].astype(float)
    depths = flat_depth[nearest]
    depths = np.where(distances <= float(max_sample_distance_m), depths, np.nan)
    return [None if not math.isfinite(float(value)) else float(value) for value in depths], [float(value) for value in distances]


def _sfincs_peak_depth_grid(dataset: Any) -> Any:
    np, *_ = _require_geo_stack()
    if "hmax" in dataset:
        return np.asarray(dataset["hmax"].values, dtype=float)
    if "zsmax" in dataset and "zb" in dataset:
        zsmax = np.asarray(dataset["zsmax"].values, dtype=float)
        zb = np.asarray(dataset["zb"].values, dtype=float)
        if zsmax.ndim == 3:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                zsmax = np.nanmax(zsmax, axis=0)
        return np.maximum(zsmax - zb, 0.0)
    if "zs" in dataset and "zb" in dataset:
        depth = np.asarray(dataset["zs"].values, dtype=float) - np.asarray(dataset["zb"].values, dtype=float)[None, :, :]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            return np.nanmax(np.maximum(depth, 0.0), axis=0)
    raise RuntimeError("sfincs_map.nc must contain hmax, zsmax+zb, or zs+zb")


def _require_geo_stack():
    try:
        import numpy as np
        import xarray as xr
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SFINCS sampling requires numpy, xarray, pyproj, and scipy") from exc
    return np, xr, Transformer, cKDTree


def _distance_quantity() -> type[Any]:
    try:
        from gdm.quantities import Distance  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise NativeDependencyError("ERAD depth-fragility evaluation requires grid-data-models Distance quantity") from exc
    return Distance


def _asset_type_enum(name: str) -> Any:
    enums = require_module("erad.enums", package="nrel-erad", purpose="ERAD asset type mapping")
    AssetTypes = enums.AssetTypes
    key = str(name).strip()
    aliases = {
        "distribution_transformer": "transformer_pad_mount",
        "distribution_source": "substation",
        "underground_line": "distribution_underground_cables",
        "fuse_cutout": "switch",
        "load_service": "distribution_junction_box",
        "topology_only": "distribution_overhead_lines",
    }
    key = aliases.get(key, key)
    if hasattr(AssetTypes, key):
        return getattr(AssetTypes, key)
    raise KeyError(f"ERAD AssetTypes has no member {name!r}")


def _curve_from_row(local_type: str, erad_type_name: str, row: Mapping[str, str]) -> NativeEradCurve:
    models = require_module("erad.models.fragility_curve", package="nrel-erad", purpose="ERAD fragility curves")
    Distance = _distance_quantity()
    distribution = str(row.get("distribution") or "lognorm")
    if distribution != "lognorm":
        parameters = [Distance(float(row.get("loc_m", 0.0)), "meter"), float(row["scale_m"])]
    else:
        # scipy.stats.lognorm.cdf(x, s, loc, scale)
        parameters = [float(row["shape"]), Distance(float(row.get("loc_m", 0.0)), "meter"), Distance(float(row["scale_m"]), "meter")]
    asset_type = _asset_type_enum(erad_type_name)
    curve = models.FragilityCurve(
        asset_type=asset_type,
        prob_function=models.ProbabilityFunction(distribution=distribution, parameters=parameters),
    )
    return NativeEradCurve(local_type, asset_type, curve)


def _find_default_curve(curve_sets: Iterable[Any], asset_type: Any) -> Any:
    for curve_set in curve_sets:
        for curve in getattr(curve_set, "curves", []) or []:
            if getattr(curve, "asset_type", None) == asset_type:
                return curve
    raise KeyError(f"ERAD default fragility curves do not include asset type {asset_type!r}")


def load_asset_type_mapping(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {row["local_asset_type"]: row["erad_asset_type"] for row in csv.DictReader(fh)}
