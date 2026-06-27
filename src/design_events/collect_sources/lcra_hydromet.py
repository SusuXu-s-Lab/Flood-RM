from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


LCRA_HYDROMET_ROOT = "https://hydromet.lcra.org/"
LCRA_ALL_SITES_ENDPOINT = "api/GetDataForAllSites"
DEFAULT_OUTPUT = "data/sources/lcra_hydromet/flow_sites.geojson"
DEFAULT_CURRENT_OUTPUT = "data/sources/lcra_hydromet/flow_sites_current.csv"
DEFAULT_REVIEWED_USGS_NETWORK = "data/sources/usgs_streamgages/streamgage_network.geojson"


def _timestamp(value):
    return None if value is None else pd.Timestamp(value).isoformat()


def _relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_source_artifact(paths, source, kind, start=None, end=None, artifacts=None, metadata=None, status="complete"):
    manifest = {
        "study_location": paths["location_name"],
        "source": source,
        "kind": kind,
        "status": status,
        "start": _timestamp(start),
        "end": _timestamp(end),
        "artifacts": {key: _relative_path(value, paths["repo_root"]) for key, value in (artifacts or {}).items()},
        "metadata": metadata or {},
    }
    path = paths["source_artifacts_root"] / f"{source}_{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def collect_lcra_hydromet(settings, skip_existing=False, smoke=False):
    """Collect public LCRA/COA Hydromet flow-site coverage for source review.

    This is intentionally a supplemental discovery layer. It does not replace the
    reviewed USGS historical discharge source until historical Hydromet records
    have been validated for the target period.
    """
    config = settings["config"]
    paths = settings["paths"]
    spec = settings.get("lcra_hydromet", {})
    return discover_lcra_hydromet_flow_sites(
        config,
        paths,
        site_records=None,
        skip_existing=skip_existing,
        smoke=smoke,
        spec=spec,
    )


def discover_lcra_hydromet_flow_sites(
    config,
    paths,
    *,
    site_records=None,
    skip_existing=False,
    smoke=False,
    spec=None,
):
    spec = dict(spec or config.get("collection", {}).get("lcra_hydromet", {}))
    output_path = _location_path(paths, spec.get("output", DEFAULT_OUTPUT))
    current_output_path = _location_path(paths, spec.get("current_output", DEFAULT_CURRENT_OUTPUT))
    if skip_existing and output_path.exists() and current_output_path.exists():
        return {
            "reused": True,
            "candidate_count": len(gpd.read_file(output_path)),
            "candidate_geojson": output_path,
            "current_csv": current_output_path,
        }

    records = site_records if site_records is not None else fetch_lcra_hydromet_site_records(spec)
    frame = lcra_hydromet_flow_site_frame(records, spec, paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(frame.to_json(drop_id=True), encoding="utf-8")
    frame.drop(columns="geometry").to_csv(current_output_path, index=False)
    artifact_json = write_source_artifact(
        paths,
        source="lcra_hydromet",
        kind="flow_sites",
        start=pd.Timestamp(config.get("collection", {}).get("start", settings_start_default())),
        end=pd.Timestamp(config.get("collection", {}).get("end", settings_end_default())),
        artifacts={"candidate_geojson": output_path, "current_csv": current_output_path},
        metadata={
            "candidate_count": int(len(frame)),
            "endpoint": _endpoint_url(spec, LCRA_ALL_SITES_ENDPOINT),
            "search_geometry": spec.get("search_geometry", "data/static/aoi/wflow_nhdplus_watersheds.geojson"),
            "exclude_agencies": sorted(_exclude_agencies(spec)),
            "distinct_from_usgs_network": spec.get("distinct_from_usgs_network", DEFAULT_REVIEWED_USGS_NETWORK),
            "distinct_distance_m": _distinct_distance_m(spec),
            "supplemental_source": True,
            "status": "current_flow_site_discovery_only",
        },
    )
    return {
        "reused": False,
        "candidate_count": int(len(frame)),
        "candidate_geojson": output_path,
        "current_csv": current_output_path,
        "source_artifact_json": artifact_json,
        "smoke": bool(smoke),
    }


def fetch_lcra_hydromet_site_records(spec=None):
    spec = spec or {}
    response = requests.get(
        _endpoint_url(spec, LCRA_ALL_SITES_ENDPOINT),
        timeout=int(spec.get("request_timeout_seconds", 60)),
    )
    response.raise_for_status()
    return response.json()


def lcra_hydromet_flow_site_frame(records, spec, paths):
    rows = []
    for record in records or []:
        row = _flow_site_row(record)
        if row is not None:
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return gpd.GeoDataFrame(columns=_schema_columns(), geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    region = _search_region(spec, paths)
    if region is not None:
        gdf = gdf[gdf.geometry.within(region)].copy()
    gdf = _drop_excluded_agencies(gdf, spec)
    gdf = _drop_usgs_overlaps(gdf, spec, paths)
    return gdf[_schema_columns() + ["geometry"]].sort_values(["latitude", "longitude"], ascending=[False, True])


def summarize_lcra_hydromet_headwater_coverage(config, paths, *, site_records=None, spec=None):
    spec = dict(spec or config.get("collection", {}).get("lcra_hydromet", {}))
    records = site_records if site_records is not None else fetch_lcra_hydromet_site_records(spec)
    frame = lcra_hydromet_flow_site_frame(records, spec, paths)
    region = _search_region(spec, paths)
    if region is None or frame.empty:
        return pd.Series(
            {
                "flow_sites_in_region": int(len(frame)),
                "northwest_flow_sites": 0,
                "supplemental_source": "LCRA Hydromet",
            },
            name="lcra_hydromet_headwater_coverage",
        )
    minx, miny, maxx, maxy = region.bounds
    midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
    northwest = frame[(frame.geometry.x < midx) & (frame.geometry.y > midy)]
    return pd.Series(
        {
            "flow_sites_in_region": int(len(frame)),
            "northwest_flow_sites": int(len(northwest)),
            "northwest_site_names": ", ".join(northwest["site_name"].astype(str)),
            "supplemental_source": "LCRA Hydromet",
            "promotion_status": "requires historical record QA before use in POT/design-event catalog",
        },
        name="lcra_hydromet_headwater_coverage",
    )


def _flow_site_row(record):
    flow = record.get("flow")
    if flow is None:
        return None
    lat = _number(record.get("latitude"))
    lon = _number(record.get("longitude"))
    if lat is None or lon is None:
        return None
    agency = str(record.get("agency", "LCRA")).strip() or "LCRA"
    site_number = str(record.get("siteNumber", "")).strip()
    return {
        "site_no": f"{agency}:{site_number}",
        "site_number": site_number,
        "site_name": str(record.get("siteName", "")).strip(),
        "source_network": "LCRA Hydromet",
        "agency": agency,
        "status": "current",
        "flow_cfs": _number(flow),
        "stage_ft": _number(record.get("stage")),
        "date_time": record.get("dateTime"),
        "latitude": lat,
        "longitude": lon,
        "roles": [],
        "review_status": "supplemental_candidate",
        "review_notes": "Public Hydromet flow site; validate historical records before use as a design-event discharge source.",
    }


def _drop_excluded_agencies(gdf, spec):
    excluded = _exclude_agencies(spec)
    if not excluded or gdf.empty or "agency" not in gdf:
        return gdf
    return gdf[~gdf["agency"].fillna("").astype(str).str.upper().isin(excluded)].copy()


def _drop_usgs_overlaps(gdf, spec, paths):
    if gdf.empty:
        return gdf
    network_value = spec.get("distinct_from_usgs_network", DEFAULT_REVIEWED_USGS_NETWORK)
    if not network_value:
        return gdf
    network_path = _location_path(paths, network_value)
    if not network_path.exists():
        return gdf
    usgs = gpd.read_file(network_path)
    if usgs.empty:
        return gdf
    if usgs.crs is None:
        usgs = usgs.set_crs("EPSG:4326")
    projected_crs = spec.get("distinct_projected_crs", "EPSG:32614")
    hydromet_projected = gdf.to_crs(projected_crs)
    usgs_projected = usgs.to_crs(projected_crs)
    threshold = _distinct_distance_m(spec)
    keep = []
    for _, row in hydromet_projected.iterrows():
        nearest = usgs_projected.distance(row.geometry).min()
        keep.append(bool(pd.isna(nearest) or nearest > threshold))
    return gdf.loc[keep].copy()


def _schema_columns():
    return [
        "site_no",
        "site_number",
        "site_name",
        "source_network",
        "agency",
        "status",
        "flow_cfs",
        "stage_ft",
        "date_time",
        "latitude",
        "longitude",
        "roles",
        "review_status",
        "review_notes",
    ]


def _exclude_agencies(spec):
    value = spec.get("exclude_agencies", ["USGS"])
    if isinstance(value, str):
        value = [value]
    return {str(item).upper() for item in value}


def _distinct_distance_m(spec):
    return float(spec.get("distinct_distance_m", 150.0))


def _search_region(spec, paths):
    value = spec.get("search_geometry", "data/static/aoi/wflow_nhdplus_watersheds.geojson")
    if not value:
        return None
    path = _location_path(paths, value)
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"LCRA Hydromet search geometry is empty: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326").union_all()


def _endpoint_url(spec, path):
    root = str(spec.get("root_url", LCRA_HYDROMET_ROOT)).rstrip("/") + "/"
    return root + path.lstrip("/")


def _location_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(paths["location_root"]) / path


def _number(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def settings_start_default():
    return "1979-01-01"


def settings_end_default():
    return "2022-12-31"
