from __future__ import annotations

from pathlib import Path
import json
import tomllib

import numpy as np
import pandas as pd
import tomli_w
import xarray as xr

from collect_sources.usgs_streamgages import fetch_nwis_discharge_records
from event_streamflow import (
    event_streamflow_records_path,
    finite_float,
    streamflow_member_metadata,
    streamflow_records_path,
)
from wflow_runs.streamflow_readiness import (
    active_wflow_submodel_ids,
    event_catalog_row,
    observation_gauges_path,
)
from wflow_runs.usgs import usgs_instantaneous_streamflow_spec


WFLOW_EXTERNAL_RIVER_INFLOW = "river_water__external_inflow_volume_flow_rate"
WFLOW_EXTERNAL_RIVER_INFLOW_VAR = "river_inflow"
CFS_TO_CMS = 0.028316846592


def prepare_wflow_streamflow_realization_for_event_model(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    event_model_root,
    submodel_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Add event-scaled USGS streamflow as Wflow external river inflow.

    This is a legacy/future adapter for workflows that intentionally inject streamflow.
    ADR-0016 rainfall-driven Wflow-SFINCS events should use `streamflow_readiness`
    instead, because injecting streamflow can double-count rainfall-runoff response.
    """
    import geopandas as gpd

    location_root = Path(location_root)
    event_model_root = Path(event_model_root)
    row = event_catalog_row(location_root, event_id, catalog_path)
    member = streamflow_member_metadata(config, location_root, row)
    gauges_path = observation_gauges_path(config, location_root, submodel_id)
    if not gauges_path.exists():
        raise FileNotFoundError(f"Wflow streamflow realization gauges not found: {gauges_path}")
    gauges = gpd.read_file(gauges_path)
    if "site_no" not in gauges:
        raise ValueError(f"{gauges_path} lacks site_no column for streamflow realization")
    allowed_sites = set(member["site_nos"])
    gauges = gauges[gauges["site_no"].astype(str).isin(allowed_sites)].copy()
    if gauges.empty:
        raise ValueError(
            f"No reviewed Wflow observation gauges for {submodel_id} match streamflow member "
            f"{member['member_id']} sites."
        )

    records, records_metadata = _load_event_streamflow_records(
        config,
        location_root,
        event_id=event_id,
        row=row,
        member=member,
        site_nos=sorted(set(gauges["site_no"].astype(str))),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    series_by_site = _event_site_streamflow_series(
        records,
        row,
        member,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    gauges = gauges[gauges["site_no"].astype(str).isin(series_by_site)].copy()
    if gauges.empty:
        raise ValueError(f"No reviewed Wflow gauges have streamflow records for member {member['member_id']}")

    forcing_path = event_model_root / "inmaps-event.nc"
    if not forcing_path.exists():
        raise FileNotFoundError(forcing_path)
    with xr.open_dataset(forcing_path) as src:
        ds = src.load()
    ydim, xdim = _spatial_dims(ds)
    inflow = np.zeros((ds.sizes["time"], ds.sizes[ydim], ds.sizes[xdim]), dtype="float32")
    placement = []
    gauges_for_grid = _gauges_for_forcing_grid(gauges, ds, ydim, xdim)
    for _, gauge in gauges_for_grid.iterrows():
        site_no = str(gauge["site_no"])
        y_index, x_index = _nearest_grid_cell(ds, ydim, xdim, float(gauge.geometry.y), float(gauge.geometry.x))
        values = series_by_site[site_no].reindex(pd.DatetimeIndex(pd.to_datetime(ds["time"].values))).interpolate("time").ffill().bfill()
        inflow[:, y_index, x_index] += values.to_numpy(dtype="float32")
        placement.append(
            {
                "site_no": site_no,
                "y_index": int(y_index),
                "x_index": int(x_index),
                "peak_m3s": float(values.max()),
            }
        )

    ds[WFLOW_EXTERNAL_RIVER_INFLOW_VAR] = (("time", ydim, xdim), inflow)
    ds[WFLOW_EXTERNAL_RIVER_INFLOW_VAR].attrs.update(
        units="m3 s-1",
        standard_name=WFLOW_EXTERNAL_RIVER_INFLOW,
        source="scaled_usgs_streamflow_event_member",
    )
    tmp = forcing_path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(forcing_path)

    _configure_external_inflow(event_model_root / "wflow_sbm.toml")
    provenance = {
        "event_id": str(event_id),
        "submodel_id": str(submodel_id),
        "streamflow_realization": "wflow_external_river_inflow",
        "wflow_variable": WFLOW_EXTERNAL_RIVER_INFLOW,
        "forcing_variable": WFLOW_EXTERNAL_RIVER_INFLOW_VAR,
        "member_id": member["member_id"],
        "streamflow_scale_factor": streamflow_scale_factor(row, member),
        "source_sites": sorted(series_by_site),
        "placed_sites": placement,
        **records_metadata,
    }
    provenance_path = event_model_root / "streamflow_realization.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "status": "prepared",
        "streamflow_realization": "wflow_external_river_inflow",
        "forcing_path": str(forcing_path),
        "forcing_variable": WFLOW_EXTERNAL_RIVER_INFLOW_VAR,
        "source_site_count": len(series_by_site),
        "placed_site_count": len(placement),
        "provenance": str(provenance_path),
    }


def cache_wflow_event_instantaneous_streamflow(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    overwrite: bool = False,
) -> dict:
    """Fetch and cache USGS instantaneous discharge for one event analog window."""
    import geopandas as gpd

    location_root = Path(location_root)
    row = event_catalog_row(location_root, event_id, catalog_path)
    member = streamflow_member_metadata(config, location_root, row)
    cache_path = event_streamflow_records_path(config, location_root, event_id, member, start, end)
    if cache_path.exists() and not overwrite:
        records = pd.read_csv(cache_path, dtype={"site_no": str}, parse_dates=["time"])
        return {
            "event_id": str(event_id),
            "member_id": member["member_id"],
            "status": "cached",
            "records_path": str(cache_path),
            "record_count": int(len(records)),
            "site_count": int(records["site_no"].astype(str).nunique()) if "site_no" in records else 0,
            "records_source": _record_source_summary(records),
            "records_resolution": _record_resolution_summary(records),
        }

    reviewed_sites: set[str] = set()
    for submodel_id in active_wflow_submodel_ids(config, location_root):
        gauges_path = observation_gauges_path(config, location_root, submodel_id)
        if not gauges_path.exists():
            continue
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges:
            raise ValueError(f"{gauges_path} lacks site_no column for streamflow realization")
        reviewed_sites.update(gauges["site_no"].astype(str))
    source_sites = sorted(set(member["site_nos"]) & reviewed_sites)
    if not source_sites:
        return {
            "event_id": str(event_id),
            "member_id": member["member_id"],
            "status": "no_reviewed_site_overlap",
            "records_path": str(cache_path),
            "record_count": 0,
            "site_count": 0,
            "records_source": "",
            "records_resolution": "",
        }

    records = _fetch_event_instantaneous_records(
        config,
        member=member,
        site_nos=source_sites,
        row=row,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    if records.empty:
        return {
            "event_id": str(event_id),
            "member_id": member["member_id"],
            "status": "no_iv_records",
            "records_path": str(cache_path),
            "record_count": 0,
            "site_count": 0,
            "records_source": "",
            "records_resolution": "",
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(cache_path, index=False)
    return {
        "event_id": str(event_id),
        "member_id": member["member_id"],
        "status": "fetched",
        "records_path": str(cache_path),
        "record_count": int(len(records)),
        "site_count": int(records["site_no"].astype(str).nunique()),
        "records_source": _record_source_summary(records),
        "records_resolution": _record_resolution_summary(records),
    }


def streamflow_scale_factor(row: pd.Series, member: dict) -> float:
    value = finite_float(row.get("streamflow_scale_factor"))
    if value and value > 0:
        return value
    target = finite_float(row.get("streamflow"))
    template = finite_float(row.get("streamflow_template_value")) or member.get("peak_flow_cfs")
    if target and template and template > 0:
        return float(target) / float(template)
    return 1.0


def _configure_external_inflow(toml_path: Path) -> None:
    if not toml_path.exists():
        raise FileNotFoundError(toml_path)
    with toml_path.open("rb") as src:
        cfg = tomllib.load(src)
    cfg.setdefault("input", {})
    cfg["input"].setdefault("forcing", {})
    cfg["input"]["forcing"][WFLOW_EXTERNAL_RIVER_INFLOW] = WFLOW_EXTERNAL_RIVER_INFLOW_VAR
    toml_path.write_bytes(tomli_w.dumps(cfg).encode("utf-8"))


def _load_streamflow_records(config: dict, location_root: Path, site_nos: list[str]) -> pd.DataFrame:
    records_path = streamflow_records_path(config, location_root)
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    records = pd.read_csv(records_path, dtype={"site_no": str}, parse_dates=["time"])
    records = records[records["site_no"].astype(str).isin(set(site_nos))].copy()
    if records.empty:
        raise ValueError(f"No streamflow records for requested Wflow sites in {records_path}")
    return records


def _load_event_streamflow_records(
    config: dict,
    location_root: Path,
    *,
    event_id: str,
    row: pd.Series,
    member: dict,
    site_nos: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    settings = (((config.get("wflow", {}) or {}).get("streamflow_realization", {}) or {}))
    cache_path = event_streamflow_records_path(config, location_root, event_id, member, start, end)
    source_sites = sorted(set(site_nos) & set(member["site_nos"]))

    if cache_path.exists():
        records = pd.read_csv(cache_path, dtype={"site_no": str}, parse_dates=["time"])
        records = records[records["site_no"].astype(str).isin(set(site_nos))].copy()
        if not records.empty:
            return records, {
                "records_path": str(cache_path),
                "records_source": _record_source_summary(records),
                "records_resolution": _record_resolution_summary(records),
            }

    fetch_iv = bool(settings.get("fetch_instantaneous_usgs", False))
    require_iv = bool(settings.get("require_instantaneous_usgs", False))
    if fetch_iv:
        fetched = _fetch_event_instantaneous_records(
            config,
            member=member,
            site_nos=source_sites,
            row=row,
            start=start,
            end=end,
        )
        if not fetched.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fetched.to_csv(cache_path, index=False)
            return fetched, {
                "records_path": str(cache_path),
                "records_source": _record_source_summary(fetched),
                "records_resolution": _record_resolution_summary(fetched),
            }
        if require_iv:
            raise ValueError(
                "No USGS instantaneous streamflow records were fetched for event "
                f"{event_id!r} member {member['member_id']!r}."
            )

    fallback = _load_streamflow_records(config, location_root, site_nos)
    if require_iv and not fallback.get("source", pd.Series(dtype=str)).astype(str).str.contains("usgs_iv").any():
        raise ValueError(
            "USGS instantaneous event-window records are required, but no cached IV records "
            f"exist at {cache_path} and fetch_instantaneous_usgs is disabled or returned no data."
        )
    return fallback, {
        "records_path": str(streamflow_records_path(config, location_root)),
        "records_source": _record_source_summary(fallback),
        "records_resolution": _record_resolution_summary(fallback),
    }


def _fetch_event_instantaneous_records(
    config: dict,
    *,
    member: dict,
    site_nos: list[str],
    row: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not site_nos:
        return pd.DataFrame(columns=["site_no", "time", "discharge_cfs", "source"])
    window_start, window_end = _analog_window(row, member, start=start, end=end)
    spec = usgs_instantaneous_streamflow_spec(config)
    records = []
    for site_no in site_nos:
        records.extend(fetch_nwis_discharge_records(spec, site_no, window_start, window_end))
    frame = pd.DataFrame(records, columns=["site_no", "time", "discharge_cfs", "source"])
    if frame.empty:
        return frame
    frame["site_no"] = frame["site_no"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    return frame.dropna(subset=["site_no", "time", "discharge_cfs"]).sort_values(["site_no", "time"]).reset_index(drop=True)


def _analog_window(row: pd.Series, member: dict, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    reference_time = pd.Timestamp(row.get("event_reference_time"))
    analog_event_time = pd.Timestamp(member["event_time"])
    pre = reference_time - start
    post = end - reference_time
    return analog_event_time - pre, analog_event_time + post


def _record_source_summary(records: pd.DataFrame) -> str:
    if "source" not in records:
        return "unknown"
    counts = records["source"].fillna("unknown").astype(str).value_counts().sort_index()
    return ",".join(f"{source}:{count}" for source, count in counts.items())


def _record_resolution_summary(records: pd.DataFrame) -> str:
    if records.empty:
        return "empty"
    pieces = []
    for site_no, group in records.groupby(records["site_no"].astype(str), sort=True):
        times = pd.Series(pd.to_datetime(group["time"], errors="coerce")).dropna().sort_values()
        diffs = times.diff().dropna().dt.total_seconds() / 3600.0
        if diffs.empty:
            pieces.append(f"{site_no}:single")
        else:
            pieces.append(f"{site_no}:median_dt_h={float(diffs.median()):.3g}")
    return ";".join(pieces)


def _event_site_streamflow_series(
    records: pd.DataFrame,
    row: pd.Series,
    member: dict,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.Series]:
    reference_time = pd.Timestamp(row.get("event_reference_time"))
    analog_event_time = pd.Timestamp(member["event_time"])
    window_start, window_end = _analog_window(row, member, start=start, end=end)
    target_index = pd.date_range(start=start, end=end, freq="h")
    scale = streamflow_scale_factor(row, member)
    out: dict[str, pd.Series] = {}
    selected = records[(records["time"] >= window_start) & (records["time"] <= window_end)].copy()
    for site_no, group in selected.groupby(selected["site_no"].astype(str)):
        series = group.sort_values("time").set_index("time")["discharge_cfs"].astype(float)
        series.index = pd.DatetimeIndex(series.index) - analog_event_time + reference_time
        hourly = (
            series.reindex(series.index.union(target_index))
            .sort_index()
            .interpolate(method="time")
            .reindex(target_index)
            .ffill()
            .bfill()
        )
        if hourly.notna().any() and float(hourly.max()) > 0:
            out[str(site_no)] = (hourly * scale * CFS_TO_CMS).astype(float)
    return out


def _spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    if {"latitude", "longitude"}.issubset(ds.dims):
        return "latitude", "longitude"
    if {"y", "x"}.issubset(ds.dims):
        return "y", "x"
    raise ValueError("Wflow forcing dataset must have latitude/longitude or y/x spatial dimensions")


def _gauges_for_forcing_grid(gauges, ds: xr.Dataset, ydim: str, xdim: str):
    if ydim == "latitude" and xdim == "longitude":
        return gauges.to_crs("EPSG:4326") if gauges.crs is not None else gauges
    crs = None
    if "spatial_ref" in ds.coords:
        crs = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    return gauges.to_crs(crs) if crs and gauges.crs is not None else gauges


def _nearest_grid_cell(ds: xr.Dataset, ydim: str, xdim: str, lat: float, lon: float) -> tuple[int, int]:
    ycoord = np.asarray(ds[ydim].values, dtype=float)
    xcoord = np.asarray(ds[xdim].values, dtype=float)
    y_value = lat if ydim == "latitude" else lat
    x_value = lon if xdim == "longitude" else lon
    y_index = int(np.nanargmin(np.abs(ycoord - y_value)))
    x_index = int(np.nanargmin(np.abs(xcoord - x_value)))
    return y_index, x_index
