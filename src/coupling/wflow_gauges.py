"""Wflow gauge inputs derived from Wflow-SFINCS coupling geometry."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from paths import location_root_from_paths, resolve_location_path


def write_wflow_crossing_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write HydroMT-Wflow gauges for crossing-derived Wflow-SFINCS handoff points."""
    location_root = location_root_from_paths(paths)
    points = submodel.get("handoff_points")
    if points:
        records = [
            {
                "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                "sfincs_domain_id": str(point.get("sfincs_domain_id", "")),
                "lon": float(point["lon"]),
                "lat": float(point["lat"]),
                "uparea": float(point["uparea_km2"]) if point.get("uparea_km2") is not None else np.nan,
            }
            for point in points
        ]
    else:
        region = submodel.get("region", {}) or {}
        outlet_xy = (submodel.get("outlet_region") or region).get("subbasin") or region.get("subbasin")
        if not outlet_xy:
            raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no crossing outlet for a gauge")
        handoff_id = next(
            (str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value),
            str(submodel.get("wflow_submodel_id")),
        )
        uparea = region.get("uparea")
        records = [
            {
                "sfincs_handoff_id": handoff_id,
                "sfincs_domain_id": next((str(value) for value in submodel.get("sfincs_domain_ids", ()) if value), ""),
                "lon": float(outlet_xy[0]),
                "lat": float(outlet_xy[1]),
                "uparea": float(uparea) if uparea is not None else np.nan,
            }
        ]

    gauges = gpd.GeoDataFrame(
        {
            "index": range(1, len(records) + 1),
            "name": [record["sfincs_handoff_id"] for record in records],
            "uparea": [record["uparea"] for record in records],
            "sfincs_handoff_id": [record["sfincs_handoff_id"] for record in records],
            "wflow_submodel_id": [str(submodel.get("wflow_submodel_id"))] * len(records),
            "sfincs_domain_id": [record["sfincs_domain_id"] for record in records],
            "gauge_location_source": ["sfincs_stream_boundary_intersection"] * len(records),
        },
        geometry=[Point(record["lon"], record["lat"]) for record in records],
        crs="EPSG:4326",
    )
    out_path = resolve_location_path(
        location_root,
        output
        or Path(config.get("wflow", {}).get("gauges", {}).get("root", "data/wflow/domain_set_gauges"))
        / f"{submodel['wflow_submodel_id']}_sfincs_gauges.geojson",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gauges.to_file(out_path, driver="GeoJSON")
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    snap_uparea = bool(gauges["uparea"].notna().all()) and outlet_source not in {
        "boundary_handoff_watershed",
        "stream_boundary_watershed",
        "sfincs_boundary_watershed",
    }
    return {
        "gauges_fn": out_path,
        "gauge_count": int(len(gauges)),
        "snap_to_river": True,
        "snap_uparea": snap_uparea,
        "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
        "sfincs_handoff_ids": tuple(record["sfincs_handoff_id"] for record in records),
    }
