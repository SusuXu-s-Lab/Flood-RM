from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests

from source_collection_v2.stochastic_boundary.audit import Artifact, resolve, write_artifact

WFS = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs"
POST = "https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest"
ATTRIBUTES = ["mukey", "hydgrp", "ksat_r", "hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r", "claytotal_r", "dbthirdbar_r", "om_r", "ph1to1h2o_r"]


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    polygons = resolve(paths, spec.get("polygons", "data/sources/ssurgo/mapunitpoly.gpkg"))
    attrs = resolve(paths, spec.get("attributes", "data/sources/ssurgo/mapunit_attributes.csv"))
    bbox = spec.get("bbox_wgs84")
    if bbox is None and spec.get("search_geometry"):
        bbox = tuple(gpd.read_file(resolve(paths, spec["search_geometry"])).to_crs("EPSG:4326").total_bounds)
    if bbox is None:
        raise ValueError("SSURGO collection requires bbox_wgs84 or search_geometry")
    if skip_existing and polygons.exists() and attrs.exists():
        return Artifact("ssurgo", "mapunit_sources", None, None, {"polygons": polygons, "attributes": attrs}, {"reused": True})
    soils = fetch_polygons(bbox, polygons, timeout=int(spec.get("request_timeout_seconds", 120)))
    table = fetch_attributes(soils["mukey"].dropna().astype(str).unique(), attrs, timeout=int(spec.get("request_timeout_seconds", 120)))
    artifact = Artifact("ssurgo", "mapunit_sources", None, None, {"polygons": polygons, "attributes": attrs}, {"mapunits": int(soils["mukey"].nunique()) if "mukey" in soils else 0, "attribute_rows": len(table)})
    write_artifact(paths, artifact)
    return artifact


def fetch_polygons(bbox, out: Path, *, timeout=120) -> gpd.GeoDataFrame:
    west, south, east, north = map(float, bbox)
    params = {"service": "WFS", "version": "1.1.0", "request": "GetFeature", "typename": "mapunitpoly", "srsname": "EPSG:4326", "outputformat": "GML2", "bbox": ",".join(map(str, [west, south, east, north]))}
    response = requests.get(f"{WFS}?{urlencode(params)}", timeout=timeout)
    response.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    gml = out.with_suffix(".gml"); gml.write_bytes(response.content)
    soils = gpd.read_file(gml).to_crs("EPSG:4326")
    soils.to_file(out, driver="GPKG"); gml.unlink(missing_ok=True)
    return soils


def fetch_attributes(mukeys, out: Path, *, timeout=120) -> pd.DataFrame:
    keys = sorted({str(k).replace("'", "''") for k in mukeys if str(k).strip()})
    query = "SELECT mu.mukey, co.hydgrp, ch.ksat_r, ch.hzdept_r, ch.hzdepb_r, ch.sandtotal_r, ch.silttotal_r, ch.claytotal_r, ch.dbthirdbar_r, ch.om_r, ch.ph1to1h2o_r FROM mapunit AS mu INNER JOIN component AS co ON mu.mukey = co.mukey LEFT JOIN chorizon AS ch ON co.cokey = ch.cokey WHERE mu.mukey IN ({}) AND co.majcompflag = 'Yes' ORDER BY mu.mukey, co.comppct_r DESC, ch.hzdept_r".format(", ".join(f"'{k}'" for k in keys)) if keys else ""
    if not query:
        frame = pd.DataFrame(columns=ATTRIBUTES)
    else:
        response = requests.post(POST, json={"query": query, "format": "JSON+COLUMNNAME"}, timeout=timeout)
        response.raise_for_status()
        rows = response.json().get("Table", [])
        frame = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame(columns=ATTRIBUTES)
    out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out, index=False)
    return frame
