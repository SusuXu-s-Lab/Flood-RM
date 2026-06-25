from __future__ import annotations

from pathlib import Path
import os

import yaml

from generated_artifact import write_generated_yaml


def build_static_data_catalog(config, paths):
    """Write a compact HydroMT data catalog for one location workspace."""
    location_root = Path(paths["location_root"])
    catalog_path = Path(paths["data_catalog"])
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_crs = str(config.get("project", {}).get("model_crs", "EPSG:4326"))
    sources = config.get("static_sources", {})
    dem_path = location_root / sources["terrain"]["output"]
    landcover_path = location_root / sources["landcover"]["output"]
    hsg_path = location_root / sources["ssurgo"]["hsg_output"]
    ksat_path = location_root / sources["ssurgo"]["ksat_output"]
    catalog = {
        "dem_region": _raster_entry(
            _relative_to_catalog(catalog_path, dem_path),
            crs=_raster_crs(dem_path, fallback_crs),
            category="topography",
        ),
        "landcover_region": _raster_entry(
            _relative_to_catalog(catalog_path, landcover_path),
            crs=_raster_crs(landcover_path, fallback_crs),
            category="landuse",
        ),
        "hydrologic_soil_group": _raster_entry(
            _relative_to_catalog(catalog_path, hsg_path),
            crs=_raster_crs(hsg_path, fallback_crs),
            category="soils",
        ),
        "saturated_conductivity": _raster_entry(
            _relative_to_catalog(catalog_path, ksat_path),
            crs=_raster_crs(ksat_path, fallback_crs),
            category="soils",
        ),
    }
    if _uses_coastal_wave_catalog_aliases(config):
        catalog["cudem_elv"] = _raster_entry(
            _relative_to_catalog(catalog_path, dem_path),
            crs=_raster_crs(dem_path, fallback_crs),
            category="topography",
        )
        catalog["worldcover"] = _raster_entry(
            _relative_to_catalog(catalog_path, landcover_path),
            crs=_raster_crs(landcover_path, fallback_crs),
            category="landuse",
        )
    streamflow_records = (
        config.get("collection", {})
        .get("usgs_streamgages", {})
        .get("streamflow_records", {})
        .get("output")
    )
    if streamflow_records:
        catalog["usgs_streamflow_records"] = _dataframe_entry(
            _relative_to_catalog(catalog_path, location_root / streamflow_records),
            category="streamflow",
            source="USGS NWIS discharge records",
        )
    for name, category in [
        ("rainfall", "rainfall"),
        ("streamflow", "streamflow_event_members"),
        ("soil_moisture", "soil_moisture"),
    ]:
        value = config.get("event_catalog", {}).get("forcing_members", {}).get(name)
        if value:
            catalog[f"{name}_members"] = _dataframe_entry(
                _relative_to_catalog(catalog_path, location_root / value),
                category=category,
            )
    handoff = config.get("wflow", {}).get("handoff", {}).get("manifest")
    if handoff:
        catalog["wflow_domain_set_handoff"] = _dataframe_entry(
            _relative_to_catalog(catalog_path, location_root / handoff),
            category="wflow_sfincs_handoff",
        )
    write_generated_yaml(catalog_path, catalog, source="static intake (build_static_data_catalog)")
    return catalog_path


def _uses_coastal_wave_catalog_aliases(config):
    return bool(config.get("coastal_waves")) or config.get("flood_setting") == "coastal"


def _raster_entry(uri, *, crs, category):
    return {
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
        "uri": uri,
        "metadata": {
            "crs": crs,
            "category": category,
        },
    }


def _raster_crs(path, fallback):
    try:
        import rasterio

        with rasterio.open(path) as src:
            return str(src.crs or fallback)
    except Exception:
        return fallback


def _dataframe_entry(uri, *, category, source=None):
    metadata = {"category": category}
    if source:
        metadata["source"] = source
    return {
        "data_type": "DataFrame",
        "driver": {
            "name": "pandas",
            "options": {"parse_dates": True},
        },
        "uri": uri,
        "metadata": metadata,
    }


def _relative_to_catalog(catalog_path, target_path):
    return Path(os.path.relpath(Path(target_path).resolve(), Path(catalog_path).resolve().parent)).as_posix()
