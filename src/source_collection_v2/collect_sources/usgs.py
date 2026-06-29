from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import requests

from design_events.stochastic_boundary.audit import Artifact, covers, resolve, write_artifact

SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

ALIASES = {
    "station_nm": "site_name", "site_nm": "site_name", "dec_lat_va": "latitude", "lat_va": "latitude",
    "dec_long_va": "longitude", "long_va": "longitude", "drain_area_va": "drainage_area_sqmi",
    "parm_cd": "parameter_cd", "parameterCd": "parameter_cd", "siteStatus": "status",
    "begin_date": "period_start", "end_date": "period_end",
}


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    candidate_out = Path(paths.get("usgs_streamgage_candidates_geojson") or resolve(paths, spec.get("candidate_output", "data/sources/usgs/streamgage_candidates.geojson")))
    records_out = Path(paths.get("usgs_streamflow_records_csv") or resolve(paths, (spec.get("streamflow_records") or {}).get("output", "data/sources/usgs/streamflow_records.csv")))
    if skip_existing and candidate_out.exists() and covers(paths, "usgs", "streamgages", start, end):
        return Artifact("usgs", "streamgages", start, end, {"candidate_geojson": candidate_out, "streamflow_records_csv": records_out}, {"reused": True})
    sites = site_candidates(fetch_site_records(spec, paths), spec)
    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.longitude, sites.latitude), crs="EPSG:4326").to_file(candidate_out, driver="GeoJSON")
    records = pd.DataFrame(columns=["site_no", "time", "discharge_cfs", "source"])
    rec_cfg = spec.get("streamflow_records") or {}
    if rec_cfg.get("collect", False):
        site_nos = rec_cfg.get("site_nos") or _reviewed_site_nos(resolve(paths, spec.get("reviewed_network", "data/sources/usgs/streamgage_network.geojson"))) or sites["site_no"].astype(str).tolist()
        rows = [row for site in site_nos for row in fetch_discharge_records(spec, site, start, end)]
        records = pd.DataFrame(rows, columns=["site_no", "time", "discharge_cfs", "source"])
        records_out.parent.mkdir(parents=True, exist_ok=True); records.to_csv(records_out, index=False)
    artifact = Artifact("usgs", "streamgages", start, end, {"candidate_geojson": candidate_out, "streamflow_records_csv": records_out}, {"candidate_count": len(sites), "record_count": len(records), "parameter_cd": _parameter(spec)})
    write_artifact(paths, artifact)
    return artifact


def fetch_site_records(spec: dict, paths: dict) -> list[dict]:
    params = _site_params(spec, paths)
    response = requests.get((spec.get("discovery") or {}).get("url", SITE_URL), params=params, timeout=int((spec.get("discovery") or {}).get("request_timeout_seconds", 60)))
    response.raise_for_status()
    return _rdb(response.text)


def site_candidates(records: list[dict], spec: dict) -> pd.DataFrame:
    frame = pd.DataFrame(records).rename(columns=lambda c: ALIASES.get(str(c), str(c)))
    if frame.empty:
        return pd.DataFrame(columns=["site_no", "site_name", "status", "drainage_area_sqmi", "period_start", "period_end", "record_years", "longitude", "latitude", "review_status"])
    frame = frame[frame.get("parameter_cd", _parameter(spec)).astype(str).eq(_parameter(spec))] if "parameter_cd" in frame else frame
    if spec.get("active_records_only", True) and "status" in frame:
        frame = frame[frame.status.astype(str).str.lower().eq("active")]
    rows = []
    for site, g in frame.dropna(subset=["longitude", "latitude"]).groupby("site_no"):
        first = g.iloc[0]
        a, b = pd.to_datetime(g.get("period_start"), errors="coerce").min(), pd.to_datetime(g.get("period_end"), errors="coerce").max()
        rows.append({
            "site_no": str(site), "site_name": first.get("site_name", ""), "status": str(first.get("status", "active")).lower(),
            "drainage_area_sqmi": pd.to_numeric(pd.Series([first.get("drainage_area_sqmi")]), errors="coerce").iloc[0],
            "period_start": None if pd.isna(a) else a.date().isoformat(), "period_end": None if pd.isna(b) else b.date().isoformat(),
            "record_years": None if pd.isna(a) or pd.isna(b) else round(max((b - a).days, 0) / 365.25, 2),
            "longitude": float(first.longitude), "latitude": float(first.latitude), "review_status": "candidate",
        })
    return pd.DataFrame(rows)


def fetch_discharge_records(spec: dict, site_no: str, start, end) -> list[dict]:
    rec = spec.get("streamflow_records") or {}
    service = str(rec.get("service", "dv")).lower()
    params = {"format": "rdb", "sites": str(site_no), "parameterCd": _parameter(spec), "startDT": pd.Timestamp(start).date().isoformat(), "endDT": pd.Timestamp(end).date().isoformat(), "siteStatus": _status(spec)}
    if service == "dv":
        params["statCd"] = str(rec.get("stat_cd", "00003"))
    response = requests.get(rec.get("url") or (IV_URL if service == "iv" else DV_URL), params=params, timeout=int(rec.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    frame = pd.DataFrame(_rdb(response.text)).rename(columns=lambda c: ALIASES.get(str(c), str(c)))
    if frame.empty:
        return []
    time_col = next((c for c in ["time", "datetime", "dateTime"] if c in frame), None)
    q_col = next((c for c in frame.columns if "00060" in str(c) and not str(c).endswith(("_cd", "_qualifiers", "_approval_cd"))), None) or ("discharge_cfs" if "discharge_cfs" in frame else None)
    out = pd.DataFrame({"site_no": frame["site_no"].astype(str), "time": pd.to_datetime(frame[time_col], errors="coerce"), "discharge_cfs": pd.to_numeric(frame[q_col], errors="coerce"), "source": f"usgs_{service}"})
    return out.dropna().sort_values("time").to_dict("records")


def _site_params(spec: dict, paths: dict) -> dict:
    d = spec.get("discovery") or {}
    params = {"format": "rdb", "parameterCd": _parameter(spec), "siteStatus": _status(spec), "siteOutput": d.get("site_output", "Expanded")}
    for src, dst in {"sites": "sites", "state_cd": "stateCd", "huc": "huc", "county_cd": "countyCd"}.items():
        if d.get(src):
            params[dst] = ",".join(d[src]) if isinstance(d[src], list) else str(d[src])
    bbox = d.get("bbox") or _bbox(d.get("search_geometry"), paths, d.get("hydrologic_buffer_km"))
    if bbox:
        params["bBox"] = ",".join(str(x) for x in bbox)
    if d.get("data_types"):
        params["hasDataTypeCd"] = ",".join(d["data_types"])
    return params


def _rdb(text: str) -> list[dict]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) > 1 and all(re.fullmatch(r"\d*[snd](\[\d+\])?", x.strip()) for x in lines[1].split("\t")):
        del lines[1]
    return [] if not lines else pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str).to_dict("records")


def _bbox(value, paths, buffer_km=None):
    if not value:
        return None
    path = resolve(paths, value)
    if not path or not path.exists():
        return None
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    if buffer_km:
        crs = gdf.estimate_utm_crs() or "EPSG:3857"
        geom = gdf.to_crs(crs).geometry.union_all().buffer(float(buffer_km) * 1000)
        return [round(float(v), 6) for v in gpd.GeoSeries([geom], crs=crs).to_crs("EPSG:4326").total_bounds]
    return [round(float(v), 6) for v in gdf.total_bounds]


def _reviewed_site_nos(path):
    if path is None or not Path(path).exists():
        return []
    gdf = gpd.read_file(path)
    if "review_status" in gdf:
        gdf = gdf[gdf.review_status.astype(str).str.lower().isin(["accepted", "accepted_with_warning"])]
    return gdf.get("site_no", pd.Series(dtype=str)).astype(str).tolist()


def _parameter(spec):
    return str((spec.get("discovery") or {}).get("parameter_cd", "00060"))


def _status(spec):
    return str((spec.get("discovery") or {}).get("site_status", "active"))
