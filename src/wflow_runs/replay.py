"""Domain-set Wflow event replay → merged SFINCS discharge forcing.

The greensboro-style inland coupling builds one Wflow submodel per encompassing-HUC /
stream-boundary-crossing domain (see ``wflow.domain_set``). A single Event-Catalog event
is replayed by, for each submodel:

  1. resolving the event time window from the catalog ``event_reference_time``,
  2. writing per-event HydroMT update-forcing config + data catalog (so the
     ``event_precip`` / ``event_temp_pet`` pointers resolve to *this* event's forcing),
  3. ``hydromt update wflow_sbm <base>/<submodel> -i <update> -d <catalog> -o events/<event>/<submodel>``,
  4. running the Wflow engine on the produced run config,

then writing a single ``events/<event>/sfincs_discharge.nc`` GeoDataset keyed by
``sfincs_handoff_id``. By default this can merge each submodel's Wflow ``Q`` series at
``gauges_sfincs`` points; locations can also request the catalog streamflow analog as
the SFINCS handoff source, which avoids passing Wflow default-state startup recession to
SFINCS as river inflow. ``SfincsModel.discharge_points.create(timeseries=...)`` reads
that time-series contract while preserving the native SFINCS ``src`` points.

The single-model ``run_wflow_event_replay`` in :mod:`wflow_runs.notebook` only targets one
base; this module is the domain-set generalisation that also produces the merged handoff.

Prerequisites:
  - per-event meteo forcing ``events/<event>/precip.nc`` and ``temp_pet.nc``
    (the ``event_precip`` / ``event_temp_pet`` data-catalog contract; use
    ``build_event_meteo_forcing`` to stage it from the Event Catalog rainfall member),
  - built Wflow submodels (``wflow_sbm.toml`` + ``staticmaps.nc``),
  - a Wflow engine for ``execute=True`` (env ``WFLOW_BIN``, or ``wflow.run.command`` in config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import tomllib

import numpy as np
import pandas as pd
import yaml

from design_events.collect_sources.aorc_event_meteo import (
    aorc_wflow_temp_pet_variables,
    prepare_aorc_temp_pet_for_wflow,
)
from sfincs_runs.hydrology import find_aorc_event_window, prepare_aorc_precip_for_sfincs
from wflow_runs.coupled_handoff import read_stream_boundary_handoff_location_artifacts
from wflow_runs.build_plan import (
    repair_wflow_canopy_parameters,
    repair_wflow_gauge_map,
    repair_wflow_river_width,
    repair_wflow_staticmaps_nodata,
)
from wflow_runs.notebook import (
    _describe_hydromt_command,
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
    resolve_location_path,
)
from wflow_runs.states import prepare_wflow_event_instate
from wflow_runs.streamflow_realization import (
    prepare_wflow_streamflow_realization_for_event_model,
    require_wflow_external_streamflow_inflow,
)

CFS_TO_CMS = 0.028316846592


# ─── pure helpers (unit-tested) ───────────────────────────────────────────────


def resolve_event_window(
    reference_time,
    *,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
    timestep_seconds: int = 3600,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Wflow simulation window bracketing a catalog event's reference time.

    ``pre_event_hours`` is spin-up before the peak; ``post_event_hours`` lets the
    hydrograph recede. Both ends are snapped to the forcing timestep so the window is
    an exact number of steps.
    """
    ref = pd.Timestamp(reference_time)
    if pd.isna(ref):
        raise ValueError(f"event_reference_time is not a valid timestamp: {reference_time!r}")
    step = pd.Timedelta(seconds=int(timestep_seconds))
    start = (ref - pd.Timedelta(hours=float(pre_event_hours))).floor(step)
    end = (ref + pd.Timedelta(hours=float(post_event_hours))).ceil(step)
    return start, end


def build_event_meteo_forcing(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
    overwrite: bool = False,
) -> dict:
    """Stage the per-event Wflow forcing files consumed by the replay update.

    The rainfall field comes from the catalog-selected AORC SST storm window and is
    scaled by ``rainfall_scale_factor``. The companion ``temp_pet.nc`` is written from
    AORC event temperature, pressure, and radiation fields using the native
    HydroMT-Wflow ``setup_temp_pet_forcing`` contract for De Bruin PET.
    """
    location_root = Path(location_root)
    wflow = config.get("wflow", {})
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    event_dir = events_root / str(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)

    row = _event_catalog_row(location_root, event_id, catalog_path)
    start, end = resolve_event_window(
        row["event_reference_time"],
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )

    precip_path = event_dir / "precip.nc"
    temp_pet_path = event_dir / "temp_pet.nc"
    write_precip = overwrite or not precip_path.exists()
    write_temp_pet = overwrite or write_precip or not temp_pet_path.exists()

    rainfall_source_nc = _event_rainfall_source_nc(config, location_root, row)
    scale_factor = _positive_float(row.get("rainfall_scale_factor"), default=1.0)
    if write_precip:
        precip_cfg = (wflow.get("event_forcing", {}) or {}).get("precipitation", {}) or {}
        aorc_cfg = (config.get("collection", {}) or {}).get("aorc_sst", {}) or {}
        prepare_aorc_precip_for_sfincs(
            rainfall_source_nc,
            precip_path,
            t_start=start,
            t_stop=end,
            variable=str(precip_cfg.get("variable", aorc_cfg.get("variable", "APCP_surface"))),
            align_start_to_run=True,
            window_alignment=str(precip_cfg.get("window_alignment", "start")),
            precip_start=_catalog_rainfall_start(row),
            scale_factor=scale_factor,
        )
        _write_json(
            event_dir / "precip_provenance.json",
            {
                "source_nc": str(rainfall_source_nc),
                "output_nc": str(precip_path),
                "time_start": start.isoformat(),
                "time_stop": end.isoformat(),
                "rainfall_scale_factor": scale_factor,
                "hydromt_sfincs_contract": "SfincsPrecipitation.create(cumulative_input=True)",
            },
        )

    if write_temp_pet:
        source_time_start = _catalog_rainfall_start(row) or start
        prepare_aorc_temp_pet_for_wflow(
            rainfall_source_nc,
            temp_pet_path,
            t_start=start,
            t_stop=end,
            precip_template=precip_path,
            variable_candidates=aorc_wflow_temp_pet_variables(config),
            source_time_start=source_time_start,
            provenance_path=event_dir / "temp_pet_provenance.json",
        )

    return {
        "event_id": str(event_id),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "rainfall_source_nc": str(rainfall_source_nc),
        "rainfall_scale_factor": scale_factor,
        "precip_path": str(precip_path),
        "temp_pet_path": str(temp_pet_path),
        "precip_provenance": str(event_dir / "precip_provenance.json"),
        "temp_pet_provenance": str(event_dir / "temp_pet_provenance.json"),
        "precip_written": bool(write_precip),
        "temp_pet_written": bool(write_temp_pet),
    }


def build_discharge_geodataset(
    series_by_handoff: dict[str, pd.Series],
    points_by_handoff: dict[str, tuple[float, float]],
    *,
    crs,
    variable: str = "discharge",
):
    """Assemble per-handoff discharge series into a HydroMT GeoDataset.

    ``series_by_handoff`` maps ``sfincs_handoff_id`` → a discharge ``pd.Series`` on a
    DatetimeIndex; ``points_by_handoff`` maps the same id → ``(x, y)`` in ``crs``. The
    result has dims ``(index, time)`` with a *unique integer* ``index`` (required by
    ``SfincsModel.discharge_points``), the ``sfincs_handoff_id`` carried as a ``name``
    coordinate, and ``x``/``y`` point coordinates — the GeoDataset shape that
    ``discharge_points.create`` reads (pass the opened ``xr.Dataset``, not the path, so
    HydroMT uses the xarray driver rather than the vector fallback).
    """
    import xarray as xr

    handoff_ids = [hid for hid in series_by_handoff if hid in points_by_handoff]
    if not handoff_ids:
        raise ValueError("no handoff points with both a discharge series and a location")

    frame = pd.concat(
        {hid: pd.Series(series_by_handoff[hid]).astype(float) for hid in handoff_ids},
        axis=1,
    ).sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame[handoff_ids]  # stable column order matching handoff_ids

    xs = np.array([float(points_by_handoff[hid][0]) for hid in handoff_ids], dtype=float)
    ys = np.array([float(points_by_handoff[hid][1]) for hid in handoff_ids], dtype=float)

    ds = xr.Dataset(
        {variable: (("index", "time"), frame.to_numpy(dtype=float).T)},
        coords={
            "index": np.arange(1, len(handoff_ids) + 1, dtype=int),
            "time": frame.index.values,
            "name": ("index", np.array(handoff_ids, dtype=object)),
            "x": ("index", xs),
            "y": ("index", ys),
        },
    )
    ds[variable].attrs.update(units="m3 s-1", standard_name="river_water__volume_flow_rate")
    epsg = _epsg(crs)
    if epsg is not None:
        ds.attrs["crs"] = epsg
        try:  # hydromt's GeoDataset vector accessor stamps a spatial_ref the reader honours
            # set_crs mutates ds in place and returns the CRS — don't reassign ds
            ds.vector.set_crs(epsg)
        except Exception:
            pass
    return ds


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


def read_submodel_gauge_discharge(
    run_output_dir: Path,
    gauges_geojson: Path,
    *,
    csv_name: str | None = None,
):
    """Read one submodel's Wflow gauge discharge + handoff locations.

    Returns ``(series_by_handoff, points_by_handoff)`` in the gauges' native CRS. Wflow
    writes per-gauge columns ``Q_<index>`` (one per ``gauges_sfincs`` row) to its output
    CSV; each gauge row carries the ``sfincs_handoff_id`` and geometry we key on.
    """
    import geopandas as gpd

    gauges = gpd.read_file(gauges_geojson)
    if "sfincs_handoff_id" not in gauges or "index" not in gauges:
        raise ValueError(f"{gauges_geojson} lacks sfincs_handoff_id/index columns")

    csv_path = _resolve_wflow_output_csv(Path(run_output_dir), csv_name)
    table = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    series_by_handoff: dict[str, pd.Series] = {}
    points_by_handoff: dict[str, tuple[float, float]] = {}
    for _, gauge in gauges.iterrows():
        handoff_id = str(gauge["sfincs_handoff_id"])
        column = _match_gauge_column(table.columns, gauge["index"])
        if column is None:
            continue
        series_by_handoff[handoff_id] = table[column]
        points_by_handoff[handoff_id] = (float(gauge.geometry.x), float(gauge.geometry.y))
    if not series_by_handoff:
        raise ValueError(f"no Q_<index> gauge columns matched in {csv_path}")
    return series_by_handoff, points_by_handoff


def _resolve_wflow_output_csv(run_output_dir: Path, csv_name: str | None) -> Path:
    if csv_name:
        candidate = run_output_dir / csv_name
        if candidate.exists():
            return candidate
    candidates = sorted(run_output_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no Wflow output CSV under {run_output_dir}")
    named = [c for c in candidates if c.name in {"output.csv", "output_scalar.csv"}]
    return named[0] if named else candidates[0]


def _match_gauge_column(columns, index_value) -> str | None:
    try:
        index_text = str(int(float(index_value)))
    except (TypeError, ValueError):
        index_text = str(index_value)
    exact = {f"Q_{index_text}", f"Q_gauges_sfincs_{index_text}", f"Q_{index_value}"}
    for column in columns:
        if column in exact:
            return column
    for column in columns:  # fall back to a suffix match (Q_*_<index>)
        if str(column).startswith("Q") and str(column).endswith(f"_{index_text}"):
            return column
    return None


def merge_submodel_discharge(
    submodel_outputs: list[dict],
    *,
    model_crs,
    out_path: Path,
    handoff_points: dict[str, tuple[float, float]] | None = None,
):
    """Merge per-submodel gauge discharge into one ``sfincs_discharge.nc`` GeoDataset.

    Each entry in ``submodel_outputs`` is ``{"run_output_dir", "gauges_geojson"}``. The
    merged series are reprojected to ``model_crs`` (the SFINCS grid CRS) so the handoff
    points land on the SFINCS ``src`` locations.
    """
    import geopandas as gpd

    series_by_handoff: dict[str, pd.Series] = {}
    native_points: dict[str, tuple[float, float]] = {}
    native_crs = None
    for entry in submodel_outputs:
        gauges_path = Path(entry["gauges_geojson"])
        native_crs = native_crs or gpd.read_file(gauges_path).crs
        series, points = read_submodel_gauge_discharge(entry["run_output_dir"], gauges_path)
        series_by_handoff.update(series)
        native_points.update(points)

    points_by_handoff = _reproject_points(native_points, native_crs, model_crs)
    if handoff_points:
        points_by_handoff.update(
            {
                handoff_id: point
                for handoff_id, point in handoff_points.items()
                if handoff_id in series_by_handoff
            }
        )
    ds = build_discharge_geodataset(series_by_handoff, points_by_handoff, crs=model_crs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


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
    """Write SFINCS discharge forcing from the event-catalog streamflow analog.

    This is the native SFINCS time-series handoff file: locations come from the vetted
    SFINCS handoff source artifact, while the hydrograph comes from the selected USGS
    analog records, scaled to the catalog design peak and distributed to handoff points
    by upstream area when that field is available.
    """
    row = _event_catalog_row(Path(location_root), event_id, catalog_path)
    handoffs = _sfincs_handoff_locations_for_replay(config, Path(location_root), model_crs)
    if handoffs is None or handoffs.empty:
        raise ValueError("no SFINCS handoff source locations available for streamflow discharge forcing")

    total_series, provenance = _event_streamflow_series_cms(config, Path(location_root), row, start=start, end=end)
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


def _reproject_points(points_by_handoff, src_crs, dst_crs):
    if not points_by_handoff or _epsg(src_crs) == _epsg(dst_crs) or src_crs is None:
        return points_by_handoff
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = {}
    for handoff_id, (x, y) in points_by_handoff.items():
        nx, ny = transformer.transform(x, y)
        out[handoff_id] = (float(nx), float(ny))
    return out


def _sfincs_handoff_points_for_replay(config: dict, location_root: Path, model_crs) -> dict[str, tuple[float, float]]:
    """Return vetted SFINCS handoff source coordinates keyed by ``sfincs_handoff_id``."""
    locations = _sfincs_handoff_locations_for_replay(config, location_root, model_crs)
    if locations is None or locations.empty:
        return {}
    points: dict[str, tuple[float, float]] = {}
    for _, row in locations.iterrows():
        handoff_id = str(row["sfincs_handoff_id"])
        points[handoff_id] = (float(row.geometry.x), float(row.geometry.y))
    return points


def _sfincs_handoff_locations_for_replay(config: dict, location_root: Path, model_crs):
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
    reference_time = pd.Timestamp(_required_event_value(row, "event_reference_time"))
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


def _streamflow_member_metadata(config: dict, location_root: Path, row: pd.Series) -> dict:
    member_id = str(_required_event_value(row, "streamflow_member_id"))
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

    site_nos = list(dict.fromkeys([site_no, *contributing]))
    return {
        "member_id": member_id,
        "site_no": site_no,
        "event_time": event_time,
        "peak_flow_cfs": peak_flow_cfs,
        "site_nos": site_nos,
    }


def _split_site_list(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _streamflow_target_peak_cfs(row: pd.Series, member: dict) -> float:
    for key in ("streamflow", "target_peak_cfs"):
        value = _finite_float(row.get(key))
        if value and value > 0:
            return value
    scale = _positive_float(row.get("streamflow_scale_factor"), default=1.0)
    template_peak = _finite_float(row.get("streamflow_template_value")) or member.get("peak_flow_cfs")
    if template_peak and template_peak > 0:
        return float(template_peak) * scale
    raise ValueError("event row has no usable streamflow design peak")


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _discharge_handoff_source(config: dict) -> str:
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    return str(forcing.get("source", "wflow_replay")).strip().lower()


# ─── orchestration (subprocess steps gated behind execute=True) ───────────────


@dataclass(frozen=True)
class ReplayStep:
    submodel_id: str
    update_command: str
    run_command: str
    run_output_dir: str
    gauges_geojson: str


def replay_inland_domain_set(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = False,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
) -> pd.DataFrame:
    """Replay every Wflow submodel for one event and write the merged SFINCS discharge.

    With ``execute=False`` (default) this plans the work — resolves the window, checks
    prerequisites, and returns the per-submodel HydroMT-update + Wflow-run commands —
    without spawning anything. With ``execute=True`` it runs the updates + Wflow engine
    and writes ``events/<event>/sfincs_discharge.nc``.
    """
    location_root = Path(location_root)
    wflow = config.get("wflow", {})
    base_root = resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base"))
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    update_cfg = resolve_location_path(location_root, wflow.get("update_forcing_config", "wflow_update_forcing.yml"))
    data_catalog = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    model_crs = wflow.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))

    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    start, end = resolve_event_window(reference_time, pre_event_hours=pre_event_hours, post_event_hours=post_event_hours)

    event_dir = events_root / event_id
    if execute:
        # Per-event meteo forcing is only consumed by the actual HydroMT update; a dry run
        # still plans (and writes the runnable -i/-d configs) without it.
        _require_event_forcing(event_dir)
    submodels = _domain_set_submodels(config, location_root)
    if not submodels:
        raise ValueError("wflow.domain_set has no submodels to replay")

    event_dir.mkdir(parents=True, exist_ok=True)
    per_event_catalog = _write_per_event_data_catalog(data_catalog, event_dir, event_id)
    per_event_update = _write_per_event_update_config(update_cfg, event_dir, start, end)
    run_command_template = _wflow_run_command(config)
    discharge_source = _discharge_handoff_source(config)

    steps: list[ReplayStep] = []
    apply_repairs = _wflow_replay_repairs_enabled(config)
    for submodel in submodels:
        submodel_id = str(submodel["wflow_submodel_id"])
        submodel_base = base_root / submodel_id
        if not (submodel_base / "wflow_sbm.toml").exists():
            raise FileNotFoundError(f"Wflow submodel base not built: {submodel_base}")
        if execute:
            repair_wflow_staticmaps_nodata(submodel_base)
            repair_wflow_canopy_parameters(submodel_base)
        if execute and apply_repairs:
            repair_wflow_river_width(submodel_base)
            repair_wflow_gauge_map(submodel_base)
        out_dir = event_dir / submodel_id
        if execute:
            _prepare_replay_submodel_output_dir(event_dir, out_dir)
        update_command = (
            f"hydromt update wflow_sbm {submodel_base} "
            f"-i {per_event_update} -d {per_event_catalog} -o {out_dir} -vvv"
        )
        run_config = out_dir / "wflow_sbm.toml"
        run_command = run_command_template.format(run_config=run_config)
        steps.append(
            ReplayStep(
                submodel_id=submodel_id,
                update_command=update_command,
                run_command=run_command,
                run_output_dir=str(out_dir / "run_event"),
                gauges_geojson=str(submodel_base / "staticgeoms" / "gauges_sfincs.geojson"),
            )
        )

    rows = []
    for step in steps:
        status = "planned"
        resolved_update_command, hydromt_runner_status, hydromt_runner_issue = _describe_hydromt_command(
            step.update_command,
            location_root,
        )
        if execute:
            _run(_resolve_hydromt_command(step.update_command, location_root), cwd=location_root)
            prepare_wflow_event_instate(event_dir / step.submodel_id, base_root / step.submodel_id)
            if discharge_source == "wflow_dynamic":
                prepare_wflow_streamflow_realization_for_event_model(
                    config,
                    location_root,
                    event_id,
                    catalog_path=catalog_path,
                    event_model_root=event_dir / step.submodel_id,
                    submodel_id=step.submodel_id,
                    start=start,
                    end=end,
                )
                require_wflow_external_streamflow_inflow(
                    config,
                    location_root,
                    event_id,
                    catalog_path=catalog_path,
                    event_model_root=event_dir / step.submodel_id,
                )
            repair_wflow_staticmaps_nodata(event_dir / step.submodel_id)
            repair_wflow_canopy_parameters(event_dir / step.submodel_id)
            if apply_repairs:
                repair_wflow_river_width(event_dir / step.submodel_id)
                repair_wflow_gauge_map(event_dir / step.submodel_id)
            _prepare_wflow_run_output_dir(event_dir / step.submodel_id / "wflow_sbm.toml")
            _run(shlex.split(step.run_command), cwd=location_root)
            status = "completed"
        rows.append(
            {
                "event_id": event_id,
                "submodel_id": step.submodel_id,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "update_command": step.update_command,
                "resolved_update_command": resolved_update_command,
                "hydromt_runner_status": hydromt_runner_status,
                "hydromt_runner_issue": hydromt_runner_issue,
                "run_command": step.run_command,
                "run_output_dir": step.run_output_dir,
                "status": status,
            }
        )

    discharge_path = event_dir / "sfincs_discharge.nc"
    if execute:
        if discharge_source in {"event_streamflow", "event_streamflow_timeseries", "catalog_streamflow"}:
            write_event_streamflow_handoff_discharge(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                model_crs=model_crs,
                out_path=discharge_path,
                start=start,
                end=end,
            )
        else:
            merge_submodel_discharge(
                [{"run_output_dir": s.run_output_dir, "gauges_geojson": s.gauges_geojson} for s in steps],
                model_crs=model_crs,
                out_path=discharge_path,
                handoff_points=_sfincs_handoff_points_for_replay(config, location_root, model_crs),
            )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    report["sfincs_discharge_source"] = discharge_source
    return report


# ─── orchestration helpers ────────────────────────────────────────────────────


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
    if "event_reference_time" not in match:
        raise ValueError(f"{catalog_path} has no event_reference_time column")
    return match.iloc[0]


def _event_reference_time(location_root: Path, event_id: str, catalog_path):
    return _event_catalog_row(location_root, event_id, catalog_path)["event_reference_time"]


def _event_rainfall_source_nc(config: dict, location_root: Path, row: pd.Series) -> Path:
    rainfall_member_file = _required_event_value(row, "rainfall_member_file")
    rainfall_member_file = resolve_location_path(location_root, rainfall_member_file)
    precip_cfg = (
        (config.get("wflow", {}) or {})
        .get("event_forcing", {})
        .get("precipitation", {})
        or {}
    )
    event_windows_dir = precip_cfg.get("event_windows_dir") or (rainfall_member_file.parent / "event_windows")
    event_windows_dir = resolve_location_path(location_root, event_windows_dir)
    return find_aorc_event_window(
        event_windows_dir,
        member_id=str(_required_event_value(row, "rainfall_member_id")),
        storm_start=_required_event_value(row, "rainfall_member_time"),
    )


def _required_event_value(row: pd.Series, key: str):
    value = row.get(key)
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Event Catalog row is missing required Wflow forcing field: {key}")
    return value


def _catalog_rainfall_start(row: pd.Series):
    reference = row.get("event_reference_time")
    offset = row.get("rainfall_start_offset_hours")
    if reference is None or offset is None or pd.isna(reference) or pd.isna(offset):
        return None
    return pd.Timestamp(reference) + pd.Timedelta(hours=float(offset))


def _positive_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not (np.isfinite(out) and out > 0):
        return float(default)
    return out


def _write_neutral_temp_pet(
    out_path: Path,
    precip_path: Path,
    *,
    temp_c: float = 15.0,
    press_msl_hpa: float = 1013.25,
) -> Path:
    import xarray as xr

    if not Path(precip_path).exists():
        raise FileNotFoundError(f"precip.nc must be written before temp_pet.nc: {precip_path}")
    with xr.open_dataset(precip_path) as precip_ds:
        template = precip_ds["precip"].transpose("time", "y", "x")
        coords = {dim: template.coords[dim].values for dim in ("time", "y", "x")}
        shape = template.shape

    def filled(value: float):
        return np.full(shape, float(value), dtype=np.float32)

    ds = xr.Dataset(
        {
            "temp": (("time", "y", "x"), filled(temp_c)),
            "press_msl": (("time", "y", "x"), filled(press_msl_hpa)),
            "kin": (("time", "y", "x"), filled(0.0)),
            "kout": (("time", "y", "x"), filled(0.0)),
        },
        coords=coords,
        attrs={
            "crs": "EPSG:4326",
            "source": "neutral short-event Wflow PET companion for scaled AORC rainfall replay",
        },
    )
    ds["temp"].attrs.update(units="degree C")
    ds["press_msl"].attrs.update(units="hPa")
    ds["kin"].attrs.update(units="W m-2")
    ds["kout"].attrs.update(units="W m-2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


def _write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _require_event_forcing(event_dir: Path) -> None:
    missing = [name for name in ("precip.nc", "temp_pet.nc") if not (event_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "per-event Wflow forcing is not built (the event_precip/event_temp_pet contract): "
            + ", ".join(str(event_dir / name) for name in missing)
            + ". Run build_event_meteo_forcing before replaying."
        )


def _domain_set_submodels(config: dict, location_root: Path) -> list[dict]:
    submodels = list(config.get("wflow", {}).get("domain_set", {}).get("submodels", []) or [])
    if submodels:
        return submodels
    manifest = resolve_location_path(
        location_root, config.get("wflow", {}).get("domain_set_manifest", "data/wflow/domain_set.yaml")
    )
    if manifest.exists():
        return list((yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("submodels", []) or [])
    return []


def _write_per_event_data_catalog(data_catalog: Path, event_dir: Path, event_id: str) -> Path:
    """Materialise the data catalog with ``<event_id>`` placeholders bound to this event."""
    text = Path(data_catalog).read_text(encoding="utf-8").replace("<event_id>", str(event_id))
    out = event_dir / "_replay_data_catalog.yml"
    out.write_text(text, encoding="utf-8")
    return out


def _write_per_event_update_config(update_cfg: Path, event_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Copy the update-forcing config with the event window substituted into setup_config."""
    workflow = yaml.safe_load(Path(update_cfg).read_text(encoding="utf-8")) or {}
    if "setup_config" in workflow:
        data = workflow["setup_config"].setdefault("data", {})
        data["time.starttime"] = start.strftime("%Y-%m-%dT%H:%M:%S")
        data["time.endtime"] = end.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        for step in workflow.get("steps", []):
            if "setup_config" in step:
                data = step["setup_config"].setdefault("data", {})
                data["time.starttime"] = start.strftime("%Y-%m-%dT%H:%M:%S")
                data["time.endtime"] = end.strftime("%Y-%m-%dT%H:%M:%S")
    out = event_dir / "_wflow_update_forcing.yml"
    out.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    return out


def _prepare_replay_submodel_output_dir(event_dir: Path, out_dir: Path) -> None:
    """Remove a generated event-submodel directory before HydroMT rewrites it."""
    event_dir = Path(event_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == event_dir or event_dir not in out_dir.parents:
        raise ValueError(f"refusing to clean replay output outside event dir: {out_dir}")
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()


def _prepare_wflow_run_output_dir(run_config: Path) -> None:
    """Remove the generated Wflow run-output directory before solver execution."""
    run_config = Path(run_config).resolve()
    if not run_config.exists():
        return
    with run_config.open("rb") as src:
        cfg = tomllib.load(src)
    dir_output = str(cfg.get("dir_output", "")).strip()
    if not dir_output:
        return
    model_root = run_config.parent
    out_dir = Path(dir_output)
    if not out_dir.is_absolute():
        out_dir = model_root / out_dir
    out_dir = out_dir.resolve()
    if out_dir == model_root or model_root not in out_dir.parents:
        raise ValueError(f"refusing to clean Wflow output outside model dir: {out_dir}")
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()


def _wflow_run_command(config: dict) -> str:
    """Resolve the Wflow engine command template (``{run_config}`` placeholder)."""
    run_cfg = config.get("wflow", {}).get("run", {}) or {}
    command = run_cfg.get("command") or os.environ.get(run_cfg.get("bin_env", "WFLOW_BIN"), "")
    if command:
        return command if "{run_config}" in command else f"{command} {{run_config}}"
    # Default to the Wflow.jl CLI convention used by the reference coupling workflow.
    return "wflow_cli {run_config}"


def _wflow_replay_repairs_enabled(config: dict) -> bool:
    replay_cfg = (config.get("wflow", {}) or {}).get("replay", {}) or {}
    return bool(replay_cfg.get("apply_legacy_repairs", False))


def _run(command_parts, *, cwd) -> None:
    if not command_parts:
        raise ValueError("empty command")
    try:
        subprocess.run(command_parts, cwd=Path(cwd), check=True, env=_hydromt_subprocess_env(Path(cwd)))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"executable not found for replay step: {shlex.join([str(p) for p in command_parts])}"
        ) from exc
