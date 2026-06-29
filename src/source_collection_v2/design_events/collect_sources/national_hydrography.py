from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from design_events.stochastic_boundary.audit import Artifact, resolve, write_artifact

NHDPLUS = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"
LAYERS = {"flowlines": 3, "waterbodies": 9, "catchments": 10}


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    root = resolve(paths, spec.get("raw_dir", "data/sources/national_hydrography"))
    bbox = spec.get("bbox_wgs84")
    if bbox is None and spec.get("search_geometry"):
        bbox = tuple(gpd.read_file(resolve(paths, spec["search_geometry"])).to_crs("EPSG:4326").total_bounds)
    if bbox is None:
        raise ValueError("national_hydrography collection requires bbox_wgs84 or search_geometry")
    wanted = spec.get("layers", ["flowlines", "catchments", "waterbodies"])
    files = {name: root / f"nhdplus_hr_{name}.gpkg" for name in wanted}
    if skip_existing and all(path.exists() for path in files.values()):
        return Artifact("national_hydrography", "nhdplus_hr_layers", None, None, files, {"reused": True})
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, path in files.items():
        gdf = fetch_layer(bbox, LAYERS[name], service_url=spec.get("service_url", NHDPLUS), timeout=int(spec.get("request_timeout_seconds", 120)))
        gdf.to_file(path, driver="GPKG")
        counts[f"{name}_features"] = int(len(gdf))
    artifact = Artifact("national_hydrography", "nhdplus_hr_layers", None, None, files, {"bbox_wgs84": list(map(float, bbox)), **counts})
    write_artifact(paths, artifact)
    return artifact


def fetch_layer(bbox, layer_id: int, *, service_url=NHDPLUS, timeout=120) -> gpd.GeoDataFrame:
    features, offset, limit = [], 0, 2000
    while True:
        response = requests.get(
            f"{service_url.rstrip('/')}/{int(layer_id)}/query",
            params={
                "where": "1=1", "geometry": ",".join(str(float(v)) for v in bbox),
                "geometryType": "esriGeometryEnvelope", "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*", "returnGeometry": "true", "outSR": "4326", "f": "geojson",
                "resultOffset": offset, "resultRecordCount": limit,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json(); batch = payload.get("features", [])
        features.extend(batch)
        if not payload.get("exceededTransferLimit") and len(batch) < limit:
            break
        offset += len(batch)
        if not batch:
            break
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
