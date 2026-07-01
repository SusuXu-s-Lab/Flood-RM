from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts
from event_streamflow import (
    finite_float,
    streamflow_member_metadata as _streamflow_member_metadata,
    streamflow_records_path as _streamflow_records_path,
)
from paths import resolve_location_path
from wflow_runs.event import (
    legacy_event_catalog_row,
    required_event_value,
)
from wflow_runs.output import gauge_discharge

CFS_TO_CMS = 0.028316846592


def build_discharge_geodataset(
    series_by_handoff: dict[str, pd.Series],
    points_by_handoff: dict[str, tuple[float, float]],
    *,
    crs,
    variable: str = "discharge",
):
    """Assemble per-handoff discharge series into a HydroMT GeoDataset."""
    ds = build_discharge_dataset(series_by_handoff, points_by_handoff, crs=crs)
    if variable != "discharge":
        ds = ds.rename({"discharge": variable})
        ds[variable].attrs.update(units="m3 s-1", standard_name="river_water__volume_flow_rate")
    epsg = _epsg(crs)
    if epsg is not None:
        ds.attrs["crs"] = epsg
    return ds


def merge_submodel_discharge(
    submodel_outputs: list[dict],
    *,
    model_crs,
    out_path: Path,
    handoff_points: dict[str, tuple[float, float]] | None = None,
):
    """Merge per-submodel Wflow gauge discharge into one SFINCS discharge GeoDataset."""
    normalized = [
        {
            **entry,
            "run_model_root": entry.get("run_model_root") or Path(entry["run_output_dir"]).parent,
        }
        for entry in submodel_outputs
    ]
    series: dict[str, pd.Series] = {}
    points: dict[str, tuple[float, float]] = {}
    crs_by_handoff: dict[str, object] = {}
    for item in normalized:
        sub_series, sub_points, sub_crs = gauge_discharge(
            item["run_model_root"],
            item["gauges_geojson"],
            run_output_dir=item.get("run_output_dir"),
        )
        series.update(sub_series)
        points.update(sub_points)
        for key in sub_points:
            crs_by_handoff[key] = sub_crs
    points = _reproject_points(points, crs_by_handoff, model_crs)
    if handoff_points:
        points.update({hid: xy for hid, xy in handoff_points.items() if hid in series})
    ds = build_discharge_dataset(series, points, crs=model_crs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def build_discharge_dataset(
    series_by_handoff: dict[str, pd.Series],
    points_by_handoff: dict[str, tuple[float, float]],
    *,
    crs: str,
) -> xr.Dataset:
    """Assemble handoff discharge series into the SFINCS source GeoDataset schema."""
    ids = [hid for hid in series_by_handoff if hid in points_by_handoff]
    if not ids:
        raise ValueError("no discharge series overlap handoff point coordinates")
    frame = pd.concat(
        {hid: pd.Series(series_by_handoff[hid]).astype(float) for hid in ids},
        axis=1,
    ).sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    ds = xr.Dataset(
        {"discharge": (("index", "time"), frame[ids].to_numpy(dtype=float).T)},
        coords={
            "index": np.arange(1, len(ids) + 1, dtype=int),
            "time": frame.index.values,
            "name": ("index", np.asarray(ids, dtype=object)),
            "x": ("index", np.asarray([points_by_handoff[i][0] for i in ids], dtype=float)),
            "y": ("index", np.asarray([points_by_handoff[i][1] for i in ids], dtype=float)),
        },
        attrs={"crs": str(crs), "featureType": "timeSeries"},
    )
    ds["discharge"].attrs.update(
        units="m3 s-1",
        standard_name="river_water__volume_flow_rate",
    )
    try:
        ds.vector.set_crs(crs)
    except Exception:
        pass
    return ds


def write_event_streamflow_handoff_discharge(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    model_crs,
    out_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """Write SFINCS discharge forcing from the event-catalog streamflow analog."""
    location_root = Path(location_root)
    row = legacy_event_catalog_row(location_root, event_id, catalog_path)
    handoffs = sfincs_handoff_locations(config, location_root, model_crs)
    if handoffs is None or handoffs.empty:
        raise ValueError("no SFINCS handoff source locations available for streamflow discharge forcing")

    total_series, provenance = _event_streamflow_series_cms(config, location_root, row, start=start, end=end)
    distribution_field = _handoff_distribution_field(handoffs)
    provenance["distribution_field"] = distribution_field
    weights = _handoff_distribution_weights(handoffs)
    points_by_handoff = {
        str(record["sfincs_handoff_id"]): (float(record.geometry.x), float(record.geometry.y))
        for _, record in handoffs.iterrows()
    }
    series_by_handoff = {
        handoff_id: total_series * float(weight)
        for handoff_id, weight in weights.items()
        if handoff_id in points_by_handoff
    }
    ds = build_discharge_geodataset(series_by_handoff, points_by_handoff, crs=model_crs)
    ds.attrs.update(
        {
            "discharge_source": "event_streamflow_timeseries",
            "distribution": "uparea_weighted" if distribution_field else "equal",
            "event_streamflow_member_id": provenance["member_id"],
            "streamflow_target_peak_cfs": provenance["target_peak_cfs"],
            "streamflow_template_peak_cfs": provenance["template_peak_cfs"],
            "streamflow_scale_factor": provenance["scale_factor"],
            "source_records": provenance["records_path"],
            "source_sites": ",".join(provenance["source_sites"]),
            "reference_time": str(provenance["reference_time"]),
            "analog_event_time": str(provenance["analog_event_time"]),
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    _write_json(out_path.with_suffix(".provenance.json"), {**provenance, "weights": weights})
    return out_path


def sfincs_handoff_points(config: dict, location_root: Path, model_crs) -> dict[str, tuple[float, float]]:
    """Return vetted SFINCS handoff source coordinates keyed by ``sfincs_handoff_id``."""
    locations = sfincs_handoff_locations(config, location_root, model_crs)
    if locations is None or locations.empty:
        return {}
    points: dict[str, tuple[float, float]] = {}
    for _, row in locations.iterrows():
        handoff_id = str(row["sfincs_handoff_id"])
        points[handoff_id] = (float(row.geometry.x), float(row.geometry.y))
    return points


def sfincs_handoff_locations(config: dict, location_root: Path, model_crs):
    """Return vetted SFINCS handoff source rows projected to ``model_crs``."""
    locations = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=resolve_location_path,
    )
    if locations is None or locations.empty:
        return locations.iloc[0:0] if locations is not None else locations
    active_domain_ids = _active_sfincs_domain_ids(config, location_root)
    if active_domain_ids and "sfincs_domain_id" in locations.columns:
        locations = locations[locations["sfincs_domain_id"].astype(str).isin(active_domain_ids)].copy()
        if locations.empty:
            raise ValueError(
                "SFINCS handoff artifacts exist, but none match active sfincs_domain_set domains: "
                + ", ".join(sorted(active_domain_ids))
            )
    target = locations.to_crs(model_crs) if locations.crs is not None and _epsg(locations.crs) != _epsg(model_crs) else locations
    return target


def _epsg(crs):
    if crs is None:
        return None
    try:
        from pyproj import CRS

        return int(CRS.from_user_input(crs).to_epsg())
    except Exception:
        try:
            return int(crs)
        except (TypeError, ValueError):
            return None


def _active_sfincs_domain_ids(config: dict, location_root: Path) -> set[str]:
    configured = config.get("sfincs_domain_set", {}).get("include_domain_ids") or []
    active = {str(value) for value in configured if str(value).strip()}
    if active:
        return active
    manifest_value = config.get("sfincs_domain_set", {}).get("domain_manifest", "data/sfincs/domains/domain_set.yaml")
    manifest = resolve_location_path(location_root, manifest_value)
    if not manifest.exists():
        return set()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return {
        str(domain.get("sfincs_domain_id"))
        for domain in payload.get("domains", [])
        if domain.get("sfincs_domain_id")
    }


def _handoff_distribution_weights(handoffs) -> dict[str, float]:
    id_col = "sfincs_handoff_id"
    if _handoff_distribution_field(handoffs) == "uparea":
        values = pd.to_numeric(handoffs["uparea"], errors="coerce").fillna(0.0).astype(float)
        return {
            str(handoff_id): float(value / values.sum())
            for handoff_id, value in zip(handoffs[id_col].astype(str), values, strict=False)
        }
    n = len(handoffs)
    if n == 0:
        return {}
    return {str(handoff_id): 1.0 / n for handoff_id in handoffs[id_col].astype(str)}


def _handoff_distribution_field(handoffs) -> str:
    if "uparea" not in handoffs.columns:
        return ""
    values = pd.to_numeric(handoffs["uparea"], errors="coerce").fillna(0.0).astype(float)
    return "uparea" if float(values.sum()) > 0 else ""


def _reproject_points(
    points: dict[str, tuple[float, float]],
    crs_by_handoff: dict[str, object],
    dst_crs,
) -> dict[str, tuple[float, float]]:
    from pyproj import CRS, Transformer

    out: dict[str, tuple[float, float]] = {}
    transformers: dict[str, object] = {}
    for handoff_id, (x, y) in points.items():
        src = crs_by_handoff.get(handoff_id)
        if src is None or CRS.from_user_input(src) == CRS.from_user_input(dst_crs):
            out[handoff_id] = (float(x), float(y))
            continue
        key = str(src)
        transformers.setdefault(key, Transformer.from_crs(src, dst_crs, always_xy=True))
        nx, ny = transformers[key].transform(x, y)
        out[handoff_id] = (float(nx), float(ny))
    return out


def _event_streamflow_series_cms(
    config: dict,
    location_root: Path,
    row: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, dict]:
    records_path = _streamflow_records_path(config, location_root)
    if not records_path.exists():
        raise FileNotFoundError(f"streamflow records not found: {records_path}")
    records = pd.read_csv(records_path, dtype={"site_no": str}, parse_dates=["time"])
    if records.empty:
        raise ValueError(f"streamflow records are empty: {records_path}")

    member = _streamflow_member_metadata(config, location_root, row)
    reference_time = pd.Timestamp(required_event_value(row, "event_reference_time"))
    analog_event_time = pd.Timestamp(member["event_time"])
    pre = reference_time - pd.Timestamp(start)
    post = pd.Timestamp(end) - reference_time
    window_start = analog_event_time - pre
    window_end = analog_event_time + post

    source_sites = [site for site in member["site_nos"] if site in set(records["site_no"].astype(str))]
    selected = records[
        records["site_no"].astype(str).isin(source_sites)
        & (records["time"] >= window_start)
        & (records["time"] <= window_end)
    ].copy()
    if selected.empty:
        raise ValueError(
            "no streamflow records overlap the selected analog window "
            f"{window_start.isoformat()} to {window_end.isoformat()} for member {member['member_id']!r}"
        )

    pivot = (
        selected.pivot_table(index="time", columns="site_no", values="discharge_cfs", aggfunc="mean")
        .sort_index()
        .astype(float)
    )
    template = pivot.max(axis=1).dropna()
    if template.empty or float(template.max()) <= 0:
        raise ValueError(f"streamflow analog member {member['member_id']!r} has no positive discharge records")
    shifted = template.copy()
    shifted.index = pd.DatetimeIndex(shifted.index) - analog_event_time + reference_time
    target_index = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="h")
    hourly = (
        shifted.reindex(shifted.index.union(target_index))
        .sort_index()
        .interpolate(method="time")
        .reindex(target_index)
        .ffill()
        .bfill()
    )
    template_peak_cfs = float(hourly.max())
    target_peak_cfs = _streamflow_target_peak_cfs(row, member)
    scale_factor = target_peak_cfs / template_peak_cfs if template_peak_cfs > 0 else 1.0
    out = (hourly * scale_factor * CFS_TO_CMS).astype(float)
    provenance = {
        "member_id": member["member_id"],
        "member_site_no": member["site_no"],
        "source_sites": sorted(set(source_sites)),
        "records_path": str(records_path),
        "reference_time": reference_time.isoformat(),
        "analog_event_time": analog_event_time.isoformat(),
        "window_start": pd.Timestamp(start).isoformat(),
        "window_end": pd.Timestamp(end).isoformat(),
        "analog_window_start": window_start.isoformat(),
        "analog_window_end": window_end.isoformat(),
        "target_peak_cfs": target_peak_cfs,
        "template_peak_cfs": template_peak_cfs,
        "scale_factor": scale_factor,
        "distribution_field": "",
    }
    return out, provenance


def _streamflow_target_peak_cfs(row: pd.Series, member: dict) -> float:
    for key in ("streamflow", "target_peak_cfs"):
        value = finite_float(row.get(key))
        if value and value > 0:
            return value
    scale = _positive_float(row.get("streamflow_scale_factor"), default=1.0)
    template_peak = finite_float(row.get("streamflow_template_value")) or member.get("peak_flow_cfs")
    if template_peak and template_peak > 0:
        return float(template_peak) * scale
    raise ValueError("event row has no usable streamflow design peak")


def _positive_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not (np.isfinite(out) and out > 0):
        return float(default)
    return out


def _write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
