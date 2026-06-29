from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from source_collection_v2.stochastic_boundary.audit import Artifact, resolve, write_artifact

ROOT = "https://hydromet.lcra.org/"
ENDPOINT = "api/GetDataForAllSites"


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    out = resolve(paths, spec.get("output", "data/sources/lcra_hydromet/flow_sites.geojson"))
    current = resolve(paths, spec.get("current_output", "data/sources/lcra_hydromet/flow_sites_current.csv"))
    if skip_existing and out.exists() and current.exists():
        return Artifact("lcra_hydromet", "flow_sites", settings["start"], settings["end"], {"candidate_geojson": out, "current_csv": current}, {"reused": True})
    response = requests.get(str(spec.get("root_url", ROOT)).rstrip("/") + "/" + ENDPOINT, timeout=int(spec.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    frame = _frame(response.json(), spec, paths)
    out.parent.mkdir(parents=True, exist_ok=True); current.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(out, driver="GeoJSON"); frame.drop(columns="geometry").to_csv(current, index=False)
    artifact = Artifact("lcra_hydromet", "flow_sites", settings["start"], settings["end"], {"candidate_geojson": out, "current_csv": current}, {"candidate_count": len(frame), "supplemental_source": True})
    write_artifact(paths, artifact)
    return artifact


def _frame(records, spec, paths):
    rows = []
    for r in records or []:
        if r.get("flow") is None or r.get("latitude") in (None, "") or r.get("longitude") in (None, ""):
            continue
        agency = str(r.get("agency", "LCRA")).strip() or "LCRA"
        rows.append({"site_no": f"{agency}:{r.get('siteNumber', '')}", "site_name": r.get("siteName", ""), "agency": agency, "flow_cfs": _num(r.get("flow")), "stage_ft": _num(r.get("stage")), "date_time": r.get("dateTime"), "latitude": _num(r.get("latitude")), "longitude": _num(r.get("longitude")), "review_status": "supplemental_candidate"})
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["longitude"] for r in rows], [r["latitude"] for r in rows]), crs="EPSG:4326") if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    region = spec.get("search_geometry")
    if region and (path := resolve(paths, region)) and path.exists() and not gdf.empty:
        geom = gpd.read_file(path).to_crs("EPSG:4326").geometry.union_all()
        gdf = gdf[gdf.geometry.within(geom)]
    exclude = {str(x).upper() for x in spec.get("exclude_agencies", ["USGS"])}
    return gdf[~gdf.get("agency", pd.Series(dtype=str)).astype(str).str.upper().isin(exclude)].reset_index(drop=True)


def _num(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
