from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from wflow_boundary.paths import location_path


_GENERATED = "# GENERATED FILE — source: wflow_boundary_compat.catalog\n"


def ensure_local_catalog(config: dict[str, Any], location_root: str | Path) -> Path:
    """Write the optional local HydroMT data catalog.

    This is intentionally outside :mod:`wflow_boundary`: projects with curated HydroMT
    catalogs should pass those directly and delete this shim.
    """
    location_root = Path(location_root)
    wflow = config.get("wflow", {}) or {}
    path = location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    if path.exists() and not bool(wflow.get("rewrite_data_catalog", False)):
        return path

    collection = config.get("collection", {}) or {}
    hydro = collection.get("national_hydrography", {}) or {}
    static = config.get("static_sources", {}) or {}
    project = config.get("project", {}) or {}
    crs = str(project.get("reference_crs", "EPSG:4326"))
    events_root = Path(wflow.get("events_root", "data/wflow/events"))

    def abs_uri(value):
        return str(location_path(location_root, value).resolve())

    catalog: dict[str, Any] = {
        "meta": {"roots": [".."], "source": "wflow_boundary_compat.catalog"},
        "event_precip": _raster_xarray(abs_uri(events_root / "<event_id>" / "precip.nc"), "event_forcing"),
        "event_temp_pet": _raster_xarray(abs_uri(events_root / "<event_id>" / "temp_pet.nc"), "event_forcing"),
    }
    if hydro.get("hydromt_basemap"):
        catalog["us_hydrography_basemap"] = _raster_xarray(abs_uri(hydro["hydromt_basemap"]), "hydrography")
    if hydro.get("river_geometry"):
        catalog["nhdplus_hr_river_geometry"] = _geodataframe(abs_uri(hydro["river_geometry"]), crs, "hydrography")
    reservoirs = (hydro.get("reservoirs", {}) or {}).get("output") or hydro.get("reservoirs_output")
    if reservoirs:
        catalog["nhdplus_hr_wflow_reservoirs"] = _geodataframe(abs_uri(reservoirs), crs, "hydrography")
    landcover = ((static.get("wflow_collection_extent", {}) or {}).get("landcover_output") or (static.get("landcover", {}) or {}).get("output"))
    if landcover:
        catalog["esa_worldcover"] = _rasterio(abs_uri(landcover), "landuse")
    soil = hydro.get("wflow_soil_parameters")
    if soil:
        name = (((wflow.get("source_strategy", {}) or {}).get("soils", {}) or {}).get("wflow_parameters") or "ssurgo_wflow_soil_parameters")
        catalog[name] = _raster_xarray(abs_uri(soil), "soils")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_GENERATED + yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def bind_event_catalog_template(data_catalog: str | Path, event_root: str | Path, event_id: str) -> Path:
    text = Path(data_catalog).read_text(encoding="utf-8").replace("<event_id>", str(event_id))
    out = Path(event_root) / "_hydromt_data_catalog.yml"
    out.write_text(_GENERATED + text, encoding="utf-8")
    return out


def write_event_update_workflow(update_config: str | Path, event_root: str | Path, start, end) -> Path:
    payload = yaml.safe_load(Path(update_config).read_text(encoding="utf-8")) or {}
    _set_times(payload, start, end)
    out = Path(event_root) / "_hydromt_update.yml"
    out.write_text(_GENERATED + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def _set_times(payload: dict[str, Any], start, end) -> None:
    import pandas as pd

    def update(node):
        data = node.setdefault("data", {})
        data["time.starttime"] = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S")
        data["time.endtime"] = pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%S")

    if isinstance(payload.get("setup_config"), dict):
        update(payload["setup_config"])
    for step in payload.get("steps", []) or []:
        if isinstance(step, dict) and isinstance(step.get("setup_config"), dict):
            update(step["setup_config"])


def _raster_xarray(uri: str, category: str) -> dict[str, Any]:
    return {"data_type": "RasterDataset", "driver": {"name": "raster_xarray"}, "uri": uri, "metadata": {"category": category}}


def _rasterio(uri: str, category: str) -> dict[str, Any]:
    return {"data_type": "RasterDataset", "driver": {"name": "rasterio"}, "uri": uri, "metadata": {"category": category}}


def _geodataframe(uri: str, crs: str, category: str) -> dict[str, Any]:
    return {"data_type": "GeoDataFrame", "driver": {"name": "pyogrio"}, "uri": uri, "metadata": {"crs": crs, "category": category}}
