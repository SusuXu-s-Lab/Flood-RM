from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import yaml

from .types import HandoffPoint
from paths import relative_to_or_absolute, resolve_location_path

STREAM_HANDOFF_MODES = {
    "stream_boundary_intersection",
    "sfincs_stream_boundary",
    "boundary_stream_intersection",
    "sfincs_native_river_inflow",
    "sfincs_native_reservoir_boundary_inflow",
}
DEFAULT_Q_STANDARD_NAME = "river_water__volume_flow_rate"


def model_crs(config: dict[str, Any]) -> str:
    return str((config.get("wflow", {}) or {}).get("model_crs") or (config.get("project", {}) or {}).get("model_crs") or "EPSG:4326")


def handoff_location_mode(config: dict[str, Any]) -> str:
    return str(((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}).get("handoff_location", "sfincs_native_river_inflow")).lower()


def accepted_handoff_placements(config: dict[str, Any]) -> set[str]:
    mode = handoff_location_mode(config)
    if mode == "sfincs_native_river_inflow":
        return {"sfincs_native_river_inflow", "sfincs_native_reservoir_boundary_inflow"}
    if mode in {"stream_boundary_intersection", "sfincs_stream_boundary", "boundary_stream_intersection"}:
        return {"stream_boundary_intersection", "sfincs_stream_boundary", "boundary_stream_intersection"}
    return {mode} if mode in STREAM_HANDOFF_MODES else STREAM_HANDOFF_MODES


def candidate_handoff_source_paths(config: dict[str, Any], location_root: str | Path) -> list[Path]:
    location_root = Path(location_root)
    paths: list[Path] = []
    manifest = resolve_location_path(location_root, (config.get("sfincs_domain_set", {}) or {}).get("domain_manifest", "data/sfincs/domains/domain_set.yaml"))
    if manifest.exists():
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        for domain in payload.get("domains", []) or []:
            base = domain.get("base_model_root")
            if base:
                paths.append(resolve_location_path(location_root, base) / "gis" / "wflow_handoff_sources.geojson")
    domains_root = resolve_location_path(location_root, (config.get("sfincs_domain_set", {}) or {}).get("domains_root", "data/sfincs/domains"))
    if domains_root.exists():
        paths.extend(sorted(domains_root.glob("*/base/gis/wflow_handoff_sources.geojson")))
    paths.append(location_root / "data/sfincs/base/gis/wflow_handoff_sources.geojson")
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve() if path.exists() else path.absolute()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def active_sfincs_domain_ids(config: dict[str, Any], location_root: str | Path) -> set[str]:
    configured = (config.get("sfincs_domain_set", {}) or {}).get("include_domain_ids") or []
    active = {str(v) for v in configured if str(v).strip()}
    if active:
        return active
    manifest = resolve_location_path(location_root, (config.get("sfincs_domain_set", {}) or {}).get("domain_manifest", "data/sfincs/domains/domain_set.yaml"))
    if not manifest.exists():
        return set()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return {str(d.get("sfincs_domain_id")) for d in payload.get("domains", []) or [] if d.get("sfincs_domain_id")}


def read_handoff_artifacts(config: dict[str, Any], location_root: str | Path, *, crs: str | None = None) -> gpd.GeoDataFrame:
    """Read reviewed SFINCS stream-boundary handoff artifacts."""
    frames: list[gpd.GeoDataFrame] = []
    accepted = accepted_handoff_placements(config)
    for path in candidate_handoff_source_paths(config, location_root):
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        if frame.empty or "sfincs_handoff_id" not in frame:
            continue
        if "handoff_placement" in frame:
            frame = frame[frame["handoff_placement"].fillna("").astype(str).str.lower().isin(accepted)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(columns=["sfincs_handoff_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    gdf = gdf.drop_duplicates(subset=[c for c in ["sfincs_domain_id", "sfincs_handoff_id"] if c in gdf.columns]).copy()
    active = active_sfincs_domain_ids(config, location_root)
    if active and "sfincs_domain_id" in gdf:
        gdf = gdf[gdf["sfincs_domain_id"].astype(str).isin(active)].copy()
    if crs and not gdf.empty:
        gdf = gdf.to_crs(crs)
    return gdf


def read_handoff_points(config: dict[str, Any], location_root: str | Path, *, crs: str | None = None) -> list[HandoffPoint]:
    gdf = read_handoff_artifacts(config, location_root, crs=crs)
    if gdf.empty:
        return []
    weight_field = "uparea" if "uparea" in gdf and pd.to_numeric(gdf["uparea"], errors="coerce").fillna(0).sum() > 0 else ""
    if weight_field:
        raw = pd.to_numeric(gdf[weight_field], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = raw / raw.sum()
    else:
        weights = np.full(len(gdf), 1.0 / len(gdf), dtype=float)
    return [
        HandoffPoint(
            id=str(row["sfincs_handoff_id"]),
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            weight=float(weights[i]),
            sfincs_domain_id=str(row.get("sfincs_domain_id")) if row.get("sfincs_domain_id") is not None else None,
            wflow_submodel_id=str(row.get("wflow_submodel_id")) if row.get("wflow_submodel_id") is not None else None,
        )
        for i, (_, row) in enumerate(gdf.iterrows())
    ]


def plan_domain(config: dict[str, Any], location_root: str | Path, *, write: bool = True) -> list[dict[str, Any]]:
    """Plan one Wflow submodel per active SFINCS domain from native handoff artifacts."""
    location_root = Path(location_root)
    handoffs = read_handoff_artifacts(config, location_root, crs="EPSG:4326")
    if handoffs.empty:
        raise FileNotFoundError("No reviewed SFINCS wflow_handoff_sources.geojson artifacts were found")
    min_uparea = float((((config.get("wflow", {}) or {}).get("domain_set", {}) or {}).get("min_uparea_km2", 5.0)))
    if "sfincs_domain_id" not in handoffs:
        project = str((config.get("project", {}) or {}).get("name") or location_root.name)
        handoffs["sfincs_domain_id"] = f"{project}_main"
    submodels: list[dict[str, Any]] = []
    for domain_id, group in handoffs.groupby("sfincs_domain_id", sort=True):
        group = group.to_crs("EPSG:4326").sort_values("sfincs_handoff_id")
        points = [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": str(domain_id),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": _finite(row.get("uparea")) or _finite(row.get("uparea_km2")),
            }
            for _, row in group.iterrows()
        ]
        submodels.append(
            {
                "wflow_submodel_id": str(domain_id),
                "region_kind": "subbasin",
                "hydromt_region": _subbasin_region(points, min_uparea_km2=min_uparea),
                "handoff_outlet_region": _subbasin_region(points, min_uparea_km2=min_uparea),
                "sfincs_domain_ids": [str(domain_id)],
                "sfincs_handoff_ids": [p["sfincs_handoff_id"] for p in points],
                "handoff_points": points,
                "gauges_fn": str(write_gauge_file(config, location_root, str(domain_id), points)),
                "observation_gauges_fn": None,
            }
        )
    if write:
        write_domain_manifest(config, location_root, submodels)
    return submodels


def render_hydromt_build_steps(
    config: dict[str, Any],
    location_root: str | Path,
    build_config: str | Path,
    submodel: dict[str, Any],
    *,
    sfincs_snap_to_river: bool = True,
    sfincs_snap_uparea: bool = True,
    obs_snap_to_river: bool = True,
    obs_snap_uparea: bool = True,
) -> list[dict[str, Any]]:
    """Return HydroMT-Wflow workflow steps for one reviewed SFINCS handoff submodel."""
    steps = deepcopy(_workflow_steps(build_config))
    region = submodel.get("hydromt_region") or submodel.get("region")
    if not region:
        raise ValueError(f"{submodel.get('wflow_submodel_id')} has no HydroMT region")
    _replace_basemap_region(steps, region)

    handoff = (config.get("wflow", {}) or {}).get("handoff", {}) or {}
    q_standard_name = str(handoff.get("source_standard_name", DEFAULT_Q_STANDARD_NAME))
    gauges_fn = submodel.get("gauges_fn") or submodel.get("sfincs_gauges_fn")
    if gauges_fn:
        _set_setup_gauges(
            steps,
            resolve_location_path(location_root, gauges_fn),
            basename="sfincs",
            gauge_toml_param=q_standard_name,
            snap_to_river=sfincs_snap_to_river,
            snap_uparea=sfincs_snap_uparea,
            derive_subcatch=True,
            replace_existing_unnamed=True,
        )
    obs_fn = submodel.get("observation_gauges_fn")
    if obs_fn:
        _set_setup_gauges(
            steps,
            resolve_location_path(location_root, obs_fn),
            basename="usgs",
            gauge_toml_param=q_standard_name,
            snap_to_river=obs_snap_to_river,
            snap_uparea=obs_snap_uparea,
            derive_subcatch=False,
        )
    return steps


def write_domain_manifest(config: dict[str, Any], location_root: str | Path, submodels: list[dict[str, Any]] | None = None) -> Path:
    location_root = Path(location_root)
    if submodels is None:
        submodels = plan_domain(config, location_root, write=False)
    path = resolve_location_path(location_root, (config.get("wflow", {}) or {}).get("domain_set_manifest", "data/wflow/domain_set.yaml"))
    payload = {
        "method": "sfincs_handoff_artifact_subbasins",
        "handoff": {
            "target": "sfincs_discharge_forcing",
            "source_standard_name": ((config.get("wflow", {}) or {}).get("handoff", {}) or {}).get("source_standard_name", "river_water__volume_flow_rate"),
        },
        "submodels": [_manifest_submodel(item, location_root) for item in submodels],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# GENERATED FILE - source: wflow_runs.domain\n" + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def domain_submodels(config: dict[str, Any], location_root: str | Path) -> list[dict[str, Any]]:
    submodels = configured_or_manifest_submodels(config, location_root)
    if submodels:
        return submodels
    return plan_domain(config, location_root, write=True)


def configured_or_manifest_submodels(config: dict[str, Any], location_root: str | Path) -> list[dict[str, Any]]:
    """Return configured or manifested Wflow submodels without planning or writing."""
    configured = list(((config.get("wflow", {}) or {}).get("domain_set", {}) or {}).get("submodels", []) or [])
    if configured:
        return configured
    manifest = resolve_location_path(location_root, (config.get("wflow", {}) or {}).get("domain_set_manifest", "data/wflow/domain_set.yaml"))
    if not manifest.exists():
        return []
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return list(payload.get("submodels", []) or [])


def write_gauge_file(config: dict[str, Any], location_root: str | Path, submodel_id: str, points: list[dict[str, Any]]) -> Path:
    root = resolve_location_path(location_root, ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root", "data/wflow/domain_set_gauges"))
    path = root / f"{submodel_id}_sfincs_gauges.geojson"
    rows = []
    for i, point in enumerate(points, start=1):
        rows.append(
            {
                "index": i,
                "name": str(point["sfincs_handoff_id"]),
                "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                "sfincs_domain_id": str(point.get("sfincs_domain_id") or ""),
                "wflow_submodel_id": str(submodel_id),
                "uparea": point.get("uparea_km2"),
                "geometry": Point(float(point["lon"]), float(point["lat"])),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return path


def _subbasin_region(points: list[dict[str, Any]], *, min_uparea_km2: float | None) -> dict[str, Any]:
    if not points:
        raise ValueError("HydroMT subbasin region requires at least one handoff point")
    xs = [float(p["lon"]) for p in points]
    ys = [float(p["lat"]) for p in points]
    region: dict[str, Any] = {"subbasin": [xs[0], ys[0]] if len(points) == 1 else [xs, ys]}
    if min_uparea_km2 is not None and min_uparea_km2 > 0:
        region["uparea"] = float(min_uparea_km2)
    return region


def _workflow_steps(workflow_file: str | Path) -> list[dict[str, Any]]:
    from hydromt.readers import read_workflow_yaml

    try:
        _, _, steps = read_workflow_yaml(str(workflow_file))
        return list(steps or [])
    except RuntimeError as exc:
        if "does not contain a `steps` section" not in str(exc):
            raise
    payload = yaml.safe_load(Path(workflow_file).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"HydroMT workflow file {workflow_file} must contain a mapping")
    return [
        {str(name): deepcopy(value or {})}
        for name, value in payload.items()
        if name not in {"modeltype", "global", "steps"}
    ]


def _replace_basemap_region(steps: list[dict[str, Any]], region: dict[str, Any]) -> None:
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("setup_basemaps"), dict):
            step["setup_basemaps"]["region"] = deepcopy(region)
            return
    steps.insert(0, {"setup_basemaps": {"region": deepcopy(region)}})


def _set_setup_gauges(
    steps: list[dict[str, Any]],
    gauges_fn: str | Path,
    *,
    basename: str,
    gauge_toml_param: str,
    snap_to_river: bool,
    snap_uparea: bool,
    derive_subcatch: bool,
    replace_existing_unnamed: bool = False,
) -> None:
    step = {
        "setup_gauges": {
            "gauges_fn": str(gauges_fn),
            "index_col": "index",
            "snap_to_river": bool(snap_to_river),
            "snap_uparea": bool(snap_uparea),
            "basename": basename,
            "gauge_toml_header": ["Q"],
            "gauge_toml_param": [gauge_toml_param],
            "derive_subcatch": bool(derive_subcatch),
        }
    }
    for i, existing in enumerate(steps):
        if not (isinstance(existing, dict) and isinstance(existing.get("setup_gauges"), dict)):
            continue
        current = existing["setup_gauges"].get("basename")
        if current == basename or (replace_existing_unnamed and current in (None, "", "gauges")):
            steps[i] = step
            return
    steps.append(step)


def _manifest_submodel(item: dict[str, Any], location_root: Path) -> dict[str, Any]:
    out = dict(item)
    for key in ("gauges_fn", "observation_gauges_fn"):
        if out.get(key):
            out[key] = relative_to_or_absolute(out[key], location_root)
    return out


def _finite(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None
