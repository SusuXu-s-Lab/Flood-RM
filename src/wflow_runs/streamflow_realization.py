from __future__ import annotations

from pathlib import Path
import json
import tomllib

import numpy as np
import pandas as pd
import tomli_w
import xarray as xr

from wflow_runs.notebook import resolve_location_path


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
    """Add event-scaled USGS streamflow as native Wflow external river inflow.

    The forcing is written into the HydroMT-produced ``inmaps-event.nc`` as a gridded
    ``river_inflow`` variable and referenced from ``wflow_sbm.toml`` using Wflow v1's
    ``river_water__external_inflow_volume_flow_rate`` forcing name.
    """
    import geopandas as gpd

    location_root = Path(location_root)
    event_model_root = Path(event_model_root)
    row = _event_catalog_row(location_root, event_id, catalog_path)
    member = _streamflow_member_metadata(config, location_root, row)
    gauges_path = _observation_gauges_path(config, location_root, submodel_id)
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

    records = _load_streamflow_records(config, location_root, sorted(set(gauges["site_no"].astype(str))))
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
        "streamflow_scale_factor": _streamflow_scale_factor(row, member),
        "source_sites": sorted(series_by_site),
        "placed_sites": placement,
        "records_path": str(_streamflow_records_path(config, location_root)),
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


def validate_wflow_streamflow_realization(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    event_model_root=None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate that the catalog POT/scaled streamflow event is wired into Wflow.

    Dynamic Wflow-to-SFINCS coupling should not accept a discharge handoff unless the
    selected streamflow realization is represented as Wflow external river inflow. The
    catalog streamflow fallback is a valid SFINCS time-series source, but it bypasses
    Wflow routing and is therefore not enough for ``source: wflow_dynamic``.
    """
    location_root = Path(location_root)
    row = _event_catalog_row(location_root, event_id, catalog_path)
    rows = [
        _catalog_streamflow_row(row),
        _streamflow_records_row(config, location_root),
        _streamflow_members_row(config, location_root, row),
    ]
    if event_model_root is not None:
        rows.extend(_event_model_external_inflow_rows(Path(event_model_root)))
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow streamflow realization is not wired into the event model: {details}")
    return report


def require_wflow_external_streamflow_inflow(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    event_model_root,
) -> pd.DataFrame:
    """Fail unless the event model consumes the scaled USGS/POT streamflow event."""
    report = validate_wflow_streamflow_realization(
        config,
        location_root,
        event_id,
        catalog_path=catalog_path,
        event_model_root=event_model_root,
        raise_on_error=True,
    )
    return report


def _configure_external_inflow(toml_path: Path) -> None:
    if not toml_path.exists():
        raise FileNotFoundError(toml_path)
    with toml_path.open("rb") as src:
        cfg = tomllib.load(src)
    cfg.setdefault("input", {})
    cfg["input"].setdefault("forcing", {})
    cfg["input"]["forcing"][WFLOW_EXTERNAL_RIVER_INFLOW] = WFLOW_EXTERNAL_RIVER_INFLOW_VAR
    toml_path.write_bytes(tomli_w.dumps(cfg).encode("utf-8"))


def _observation_gauges_path(config: dict, location_root: Path, submodel_id: str) -> Path:
    root = (
        ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root")
        or "data/wflow/domain_set_gauges"
    )
    root = resolve_location_path(location_root, root)
    return root / f"{submodel_id}_observation_gauges.geojson"


def _load_streamflow_records(config: dict, location_root: Path, site_nos: list[str]) -> pd.DataFrame:
    records_path = _streamflow_records_path(config, location_root)
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    records = pd.read_csv(records_path, dtype={"site_no": str}, parse_dates=["time"])
    records = records[records["site_no"].astype(str).isin(set(site_nos))].copy()
    if records.empty:
        raise ValueError(f"No streamflow records for requested Wflow sites in {records_path}")
    return records


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
    pre = reference_time - start
    post = end - reference_time
    window_start = analog_event_time - pre
    window_end = analog_event_time + post
    target_index = pd.date_range(start=start, end=end, freq="h")
    scale = _streamflow_scale_factor(row, member)
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


def _catalog_streamflow_row(row: pd.Series) -> dict:
    missing = [
        key
        for key in ("streamflow_member_id", "streamflow_scale_factor")
        if row.get(key) is None or pd.isna(row.get(key)) or str(row.get(key)).strip() == ""
    ]
    if missing:
        return {
            "check": "catalog_streamflow_member",
            "status": "failed",
            "message": "missing " + ", ".join(missing),
        }
    return {
        "check": "catalog_streamflow_member",
        "status": "passed",
        "message": f"member={row.get('streamflow_member_id')}; scale={row.get('streamflow_scale_factor')}",
    }


def _streamflow_records_row(config: dict, location_root: Path) -> dict:
    records_path = _streamflow_records_path(config, location_root)
    if not records_path.exists():
        return {"check": "streamflow_records", "status": "failed", "message": f"missing {records_path}"}
    try:
        sample = pd.read_csv(records_path, nrows=1)
    except Exception as exc:
        return {"check": "streamflow_records", "status": "failed", "message": f"unreadable {records_path}: {exc}"}
    required = {"site_no", "time", "discharge_cfs"}
    missing = sorted(required - set(sample.columns))
    return {
        "check": "streamflow_records",
        "status": "passed" if not missing else "failed",
        "message": f"path={records_path}" if not missing else f"missing columns {missing}",
    }


def _event_catalog_row(location_root: Path, event_id: str, catalog_path):
    catalog_path = (
        Path(catalog_path)
        if catalog_path
        else resolve_location_path(location_root, "data/event_catalog/catalog/probability_catalog.csv")
    )
    if not catalog_path.is_absolute():
        catalog_path = location_root / catalog_path
    catalog = pd.read_csv(catalog_path)
    catalog["event_id"] = catalog["event_id"].astype(str)
    match = catalog[catalog["event_id"] == str(event_id)]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not in {catalog_path}")
    return match.iloc[0]


def _streamflow_records_path(config: dict, location_root: Path) -> Path:
    streamflow_cfg = (((config.get("event_catalog", {}) or {}).get("driver_records", {}) or {}).get("streamflow", {}) or {})
    for key in ("records", "records_file", "path"):
        value = streamflow_cfg.get(key)
        if value:
            return resolve_location_path(location_root, value)
    return resolve_location_path(location_root, "data/sources/usgs_streamgages/streamflow_records.csv")


def _streamflow_members_path(config: dict, location_root: Path) -> Path:
    streamflow_cfg = (((config.get("event_catalog", {}) or {}).get("driver_records", {}) or {}).get("streamflow", {}) or {})
    for key in ("members", "members_file"):
        value = streamflow_cfg.get(key)
        if value:
            return resolve_location_path(location_root, value)
    return resolve_location_path(location_root, "data/sources/usgs_streamgages/streamflow_members.csv")


def _streamflow_members_row(config: dict, location_root: Path, row: pd.Series) -> dict:
    members_path = _streamflow_members_path(config, location_root)
    if not members_path.exists():
        return {"check": "streamflow_members", "status": "failed", "message": f"missing {members_path}"}
    try:
        members = pd.read_csv(members_path, dtype={"site_no": str})
    except Exception as exc:
        return {"check": "streamflow_members", "status": "failed", "message": f"unreadable {members_path}: {exc}"}
    member_id = str(row.get("streamflow_member_id"))
    matched = "member_id" in members and bool((members["member_id"].astype(str) == member_id).any())
    return {
        "check": "streamflow_members",
        "status": "passed" if matched else "failed",
        "message": f"member={member_id}" if matched else f"member {member_id!r} not found in {members_path}",
    }


def _streamflow_member_metadata(config: dict, location_root: Path, row: pd.Series) -> dict:
    member_id = str(row.get("streamflow_member_id"))
    site_no = str(row.get("streamflow_member_site_no") or member_id.split("_")[0])
    event_time = row.get("streamflow_member_time") or row.get("event_reference_time")
    peak_flow_cfs = _finite_float(row.get("streamflow_template_value"))
    contributing: list[str] = []

    members_path = _streamflow_members_path(config, location_root)
    if members_path.exists():
        members = pd.read_csv(members_path, dtype={"site_no": str})
        match = members[members["member_id"].astype(str) == member_id]
        if not match.empty:
            mrow = match.iloc[0]
            site_no = str(mrow.get("site_no") or site_no)
            event_time = mrow.get("event_time") or event_time
            peak_flow_cfs = _finite_float(mrow.get("peak_flow_cfs")) or peak_flow_cfs
            contributing = _split_site_list(mrow.get("contributing_site_nos"))

    return {
        "member_id": member_id,
        "site_no": site_no,
        "event_time": event_time,
        "peak_flow_cfs": peak_flow_cfs,
        "site_nos": list(dict.fromkeys([site_no, *contributing])),
    }


def _streamflow_scale_factor(row: pd.Series, member: dict) -> float:
    value = _finite_float(row.get("streamflow_scale_factor"))
    if value and value > 0:
        return value
    target = _finite_float(row.get("streamflow"))
    template = _finite_float(row.get("streamflow_template_value")) or member.get("peak_flow_cfs")
    if target and template and template > 0:
        return float(target) / float(template)
    return 1.0


def _split_site_list(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _event_model_external_inflow_rows(event_model_root: Path) -> list[dict]:
    toml_path = event_model_root / "wflow_sbm.toml"
    forcing_path = event_model_root / "inmaps-event.nc"
    rows = []
    forcing_var = ""
    if not toml_path.exists():
        rows.append({"check": "wflow_external_inflow_config", "status": "failed", "message": f"missing {toml_path}"})
    else:
        with toml_path.open("rb") as src:
            cfg = tomllib.load(src)
        forcing_var = str(((cfg.get("input", {}) or {}).get("forcing", {}) or {}).get(WFLOW_EXTERNAL_RIVER_INFLOW, "")).strip()
        status = "passed" if forcing_var else "failed"
        rows.append(
            {
                "check": "wflow_external_inflow_config",
                "status": status,
                "message": (
                    f"{WFLOW_EXTERNAL_RIVER_INFLOW}={forcing_var}"
                    if forcing_var
                    else f"[input.forcing].{WFLOW_EXTERNAL_RIVER_INFLOW} is not set"
                ),
            }
        )
    if not forcing_path.exists():
        rows.append({"check": "wflow_external_inflow_forcing", "status": "failed", "message": f"missing {forcing_path}"})
    else:
        with xr.open_dataset(forcing_path) as ds:
            variable = forcing_var or WFLOW_EXTERNAL_RIVER_INFLOW_VAR
            present = variable in ds
            dims = tuple(ds[variable].dims) if present else ()
            has_time = present and "time" in dims
            values = ds[variable].load() if present else None
            positive = bool((values > 0).any()) if values is not None else False
        status = "passed" if present and has_time and positive else "failed"
        message = (
            f"variable={variable}; dims={dims}; has_positive={positive}"
            if present
            else f"missing variable {variable!r}"
        )
        rows.append({"check": "wflow_external_inflow_forcing", "status": status, "message": message})
    return rows
