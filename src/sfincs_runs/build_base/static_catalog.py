from __future__ import annotations

from pathlib import Path
import os

import yaml


def build_static_data_catalog(config, paths):
    """Write a compact HydroMT data catalog for one location workspace."""
    location_root = Path(paths["location_root"])
    catalog_path = Path(paths["data_catalog"])
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    crs = str(config.get("project", {}).get("model_crs", "EPSG:4326"))
    sources = config.get("static_sources", {})
    catalog = {
        "dem_region": _raster_entry(
            _relative_to_catalog(catalog_path, location_root / sources["terrain"]["output"]),
            crs=crs,
            category="topography",
        ),
        "landcover_region": _raster_entry(
            _relative_to_catalog(catalog_path, location_root / sources["landcover"]["output"]),
            crs=crs,
            category="landuse",
        ),
        "hydrologic_soil_group": _raster_entry(
            _relative_to_catalog(catalog_path, location_root / sources["ssurgo"]["hsg_output"]),
            crs=crs,
            category="soils",
        ),
        "saturated_conductivity": _raster_entry(
            _relative_to_catalog(catalog_path, location_root / sources["ssurgo"]["ksat_output"]),
            crs=crs,
            category="soils",
        ),
    }
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
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return catalog_path


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
