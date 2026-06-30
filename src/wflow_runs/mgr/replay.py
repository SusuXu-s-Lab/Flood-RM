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
    ``build_meteo`` to stage it from the Event Catalog rainfall member),
  - built Wflow submodels (``wflow_sbm.toml`` + ``staticmaps.nc``),
  - a Wflow engine for ``execute=True`` (env ``WFLOW_BIN``, or ``wflow.run.command`` in config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shlex
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

from collect_sources.aorc_event_meteo import (
    aorc_wflow_temp_pet_variables,
    prepare_aorc_temp_pet_for_wflow,
)
from sfincs_runs.hydrology import prepare_aorc_precip_for_sfincs
from wflow_runs.handoff_locations import read_stream_boundary_handoff_location_artifacts
from wflow_runs.build_plan import (
    repair_wflow_canopy_parameters,
    repair_wflow_gauge_map,
    repair_wflow_river_width,
    repair_wflow_staticmaps_nodata,
    validate_staticmaps,
)
from wflow_runs.notebook import (
    _describe_hydromt_command,
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
    resolve_location_path,
)
from wflow_runs.states import prepare_wflow_event_instate
from wflow_runs.streamflow_realization import (
    _finite_float,
    _split_site_list,
    _streamflow_members_path,
    _streamflow_records_path,
    apply_same_frequency_amplification,
)
from wflow_runs.event import (
    build_discharge_dataset as _v2_build_discharge_dataset,
    configured_event_window_hours as _v2_configured_event_window_hours,
    event_window as _v2_event_window,
    gauge_discharge as _v2_gauge_discharge,
    match_gauge_column as _v2_match_gauge_column,
    merge_submodel_discharge as _v2_merge_submodel_discharge,
    _output_csv as _v2_output_csv,
    catalog_rainfall_start as _v2_catalog_rainfall_start,
    clean_legacy_replay_submodel_output_dir,
    legacy_event_catalog_row,
    required_event_value as _v2_required_event_value,
    write_legacy_replay_data_catalog,
    write_legacy_replay_update_config,
)
from wflow_runs.domain import configured_or_manifest_submodels as _v2_configured_or_manifest_submodels
from wflow_runs.runner import clean_output_dir as _v2_clean_output_dir
from wflow_runs.runner import wflow_run_command as _v2_wflow_run_command
from wflow_runs.runner import zero_event_forcing as _zero_event_forcing

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
    return _v2_event_window(
        reference_time,
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
        timestep_seconds=timestep_seconds,
    )


def configured_event_window_hours(
    config: dict,
    *,
    default_pre_event_hours: float = 48.0,
    default_post_event_hours: float = 72.0,
) -> tuple[float, float]:
    """Return Wflow event-window hours, including configured SFINCS drain-down.

    Wflow dynamic handoff needs enough post-rain time for routed discharge to
    recede before SFINCS stops. Locations may override this with
    ``wflow.event_window``. Otherwise, reuse the standard Wflow 72-hour
    post-event forcing window and add the Location's SFINCS drain-down buffer.
    """

    return _v2_configured_event_window_hours(
        config,
        default_pre_event_hours=default_pre_event_hours,
        default_post_event_hours=default_post_event_hours,
    )


def build_meteo(
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
    location_root = Path(location_root).resolve()
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
    precip_provenance = event_dir / "precip_provenance.json"
    temp_pet_provenance = event_dir / "temp_pet_provenance.json"
    write_precip = overwrite or not precip_path.exists() or not _provenance_window_matches(precip_provenance, start, end)
    write_temp_pet = (
        overwrite
        or write_precip
        or not temp_pet_path.exists()
        or not _provenance_window_matches(temp_pet_provenance, start, end)
    )

    rainfall_source_nc = _event_rainfall_source_nc(config, location_root, row)
    scale_factor = _positive_float(row.get("rainfall_scale_factor"), default=1.0)
    if write_temp_pet:
        _require_event_meteo_variables(config, rainfall_source_nc, event_id=event_id)
    if write_precip:
        precip_cfg = (wflow.get("event_forcing", {}) or {}).get("precipitation", {}) or {}
        aorc_cfg = (config.get("collection", {}) or {}).get("aorc_sst", {}) or {}
        prepare_aorc_precip_for_sfincs(
            rainfall_source_nc,
            precip_path,
            t_start=start,
            t_stop=end,
            variable=str(precip_cfg.get("variable", aorc_cfg.get("variable", "APCP_surface"))),
            window_alignment=str(precip_cfg.get("window_alignment", "start")),
            precip_start=_catalog_rainfall_start(row),
            scale_factor=scale_factor,
        )
        _write_json(
            precip_provenance,
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
            provenance_path=temp_pet_provenance,
        )

    return {
        "event_id": str(event_id),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "rainfall_source_nc": str(rainfall_source_nc),
        "rainfall_scale_factor": scale_factor,
        "precip_path": str(precip_path),
        "temp_pet_path": str(temp_pet_path),
        "precip_provenance": str(precip_provenance),
        "temp_pet_provenance": str(temp_pet_provenance),
        "precip_written": bool(write_precip),
        "temp_pet_written": bool(write_temp_pet),
    }


def resolve_event_rainfall_source_nc(config: dict, location_root, event_id: str, *, catalog_path=None) -> Path:
    """Resolve the catalog-selected AORC event-window file used by Wflow replay."""
    location_root = Path(location_root).resolve()
    row = _event_catalog_row(location_root, event_id, catalog_path)
    return _event_rainfall_source_nc(config, location_root, row)


def _require_event_meteo_variables(config: dict, source_nc: Path, *, event_id: str) -> None:
    import xarray as xr

    candidates_by_target = aorc_wflow_temp_pet_variables(config)
    missing: dict[str, list[str]] = {}
    with xr.open_dataset(source_nc) as ds:
        available = set(ds.data_vars)
        for target, candidates in candidates_by_target.items():
            if not any(candidate in available and _data_array_has_finite(ds[candidate]) for candidate in candidates):
                missing[target] = list(candidates)
    if not missing:
        return

    meteo_cfg = ((config.get("collection", {}) or {}).get("aorc_sst", {}) or {}).get("event_meteo", {}) or {}
    config_hint = ""
    if not bool(meteo_cfg.get("enabled", False)):
        config_hint = " Set collection.aorc_sst.event_meteo.enabled: true in the location Wflow config."
    missing_text = "; ".join(f"{target}: tried {candidates}" for target, candidates in missing.items())
    raise RuntimeError(
        f"Wflow event meteo forcing for {event_id} cannot be built because the selected AORC "
        f"event-window file is rainfall-only or stale: {source_nc}. Missing variables: {missing_text}."
        f"{config_hint} Rerun 02_flood/02_collect_sources.ipynb from the AORC SST Event Windows "
        "cell so event windows are regenerated with AORC event_meteo variables, then rerun the "
        "dynamic handoff notebook."
    )


def _data_array_has_finite(da) -> bool:
    return bool(np.isfinite(da).any().compute().item())


def build_discharge_geodataset(
    series_by_handoff: dict[str, pd.Series],
    points_by_handoff: dict[str, tuple[float, float]],
    *,
    crs,
    variable: str = "discharge",
):
    """Assemble per-handoff discharge series into a HydroMT GeoDataset."""
    ds = _v2_build_discharge_dataset(series_by_handoff, points_by_handoff, crs=crs)
    if variable != "discharge":
        ds = ds.rename({"discharge": variable})
        ds[variable].attrs.update(units="m3 s-1", standard_name="river_water__volume_flow_rate")
    epsg = _epsg(crs)
    if epsg is not None:
        ds.attrs["crs"] = epsg
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
    """Read one submodel's Wflow gauge discharge + handoff locations."""
    if csv_name is None:
        series, points, _crs = _v2_gauge_discharge(
            Path(run_output_dir).parent,
            gauges_geojson,
            run_output_dir=run_output_dir,
        )
        return series, points

    import geopandas as gpd

    gauges = gpd.read_file(gauges_geojson)
    table = pd.read_csv(_resolve_wflow_output_csv(Path(run_output_dir), csv_name), index_col=0, parse_dates=True)
    series_by_handoff: dict[str, pd.Series] = {}
    points_by_handoff: dict[str, tuple[float, float]] = {}
    for _, gauge in gauges.iterrows():
        handoff_id = str(gauge["sfincs_handoff_id"])
        column = _match_gauge_column(table.columns, gauge["index"])
        if column is not None:
            series_by_handoff[handoff_id] = pd.to_numeric(table[column], errors="coerce").astype(float)
            points_by_handoff[handoff_id] = (float(gauge.geometry.x), float(gauge.geometry.y))
    if not series_by_handoff:
        raise ValueError(f"no Q_<index> gauge columns matched in {_resolve_wflow_output_csv(Path(run_output_dir), csv_name)}")
    return series_by_handoff, points_by_handoff


def _resolve_wflow_output_csv(run_output_dir: Path, csv_name: str | None) -> Path:
    if csv_name:
        candidate = run_output_dir / csv_name
        if candidate.exists():
            return candidate
    return _v2_output_csv(run_output_dir)


def _match_gauge_column(columns, index_value) -> str | None:
    return _v2_match_gauge_column(columns, index_value)


def merge_submodel_discharge(
    submodel_outputs: list[dict],
    *,
    model_crs,
    out_path: Path,
    handoff_points: dict[str, tuple[float, float]] | None = None,
):
    """Merge per-submodel gauge discharge into one ``sfincs_discharge.nc`` GeoDataset."""
    normalized = [
        {
            **entry,
            "run_model_root": entry.get("run_model_root") or Path(entry["run_output_dir"]).parent,
        }
        for entry in submodel_outputs
    ]
    return _v2_merge_submodel_discharge(
        normalized,
        model_crs=model_crs,
        out_path=out_path,
        handoff_points=handoff_points,
    )


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
    location_root = Path(location_root).resolve()
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
            # Event models run with rainfall + antecedent moisture only; frequency provenance
            # is applied as a single-K Same-Frequency Amplification on the merged output below.
            repair_wflow_staticmaps_nodata(event_dir / step.submodel_id)
            repair_wflow_canopy_parameters(event_dir / step.submodel_id)
            if apply_repairs:
                repair_wflow_river_width(event_dir / step.submodel_id)
                repair_wflow_gauge_map(event_dir / step.submodel_id)
            validate_staticmaps(event_dir / step.submodel_id)
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
            # One Same-Frequency Amplification K applied uniformly to the handoff hydrographs.
            # No-op (K=1) until the catalog provides a per-event target and a
            # primary_reference_gage is configured.
            apply_same_frequency_amplification(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                discharge_nc=discharge_path,
                submodel_runs=[{"run_output_dir": s.run_output_dir, "gauges_geojson": s.gauges_geojson} for s in steps],
            )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    report["sfincs_discharge_source"] = discharge_source
    return report


def run_zero_rain_control(
    config: dict,
    location_root,
    event_id: str,
    *,
    execute: bool = False,
) -> pd.DataFrame:
    """Run a Wflow startup/baseflow control with event rainfall and inflow set to zero.

    The control reuses the already materialised event Wflow model folders. It copies
    them below ``events/<event>/_zero_rain/<submodel>``, zeros the dynamic forcing
    variables in ``inmaps-event.nc``, runs Wflow, and writes
    ``events/<event>/_zero_rain/sfincs_discharge.nc`` for dynamic-handoff QA.
    """
    location_root = Path(location_root).resolve()
    wflow = config.get("wflow", {}) or {}
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    model_crs = wflow.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    event_dir = events_root / str(event_id)
    zero_root = event_dir / "_zero_rain"
    run_command_template = _wflow_run_command(config)
    rows = []
    outputs = []
    for submodel in _domain_set_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        source_model = event_dir / submodel_id
        control_model = zero_root / submodel_id
        if not (source_model / "wflow_sbm.toml").exists():
            raise FileNotFoundError(
                f"Zero-rain control requires the event Wflow model first: {source_model}"
            )
        status = "planned"
        if execute:
            if control_model.exists():
                shutil.rmtree(control_model)
            control_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_model, control_model)
            _zero_event_forcing(control_model / "inmaps-event.nc")
            _prepare_wflow_run_output_dir(control_model / "wflow_sbm.toml")
            _run(shlex.split(run_command_template.format(run_config=control_model / "wflow_sbm.toml")), cwd=location_root)
            status = "completed"
        run_output_dir = control_model / "run_event"
        gauges_geojson = control_model / "staticgeoms" / "gauges_sfincs.geojson"
        outputs.append({"run_output_dir": run_output_dir, "gauges_geojson": gauges_geojson})
        rows.append(
            {
                "event_id": str(event_id),
                "submodel_id": submodel_id,
                "control_model_root": str(control_model),
                "run_output_dir": str(run_output_dir),
                "status": status,
            }
        )
    discharge_path = zero_root / "sfincs_discharge.nc"
    if execute:
        merge_submodel_discharge(
            outputs,
            model_crs=model_crs,
            out_path=discharge_path,
            handoff_points=_sfincs_handoff_points_for_replay(config, location_root, model_crs),
        )
        (zero_root / "zero_rain_control.provenance.json").write_text(
            json.dumps(
                {
                    "event_id": str(event_id),
                    "control": "zero_event_forcing",
                    "zeroed_variables": ["precip"],
                    "purpose": "dynamic_handoff_startup_baseflow_qa",
                    "sfincs_discharge": str(discharge_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    return report


# ─── orchestration helpers ────────────────────────────────────────────────────


def _event_catalog_row(location_root: Path, event_id: str, catalog_path):
    return legacy_event_catalog_row(location_root, event_id, catalog_path)


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
    # Deferred: sfincs_runs.scenarios imports wflow_runs.dynamic_handoff, which imports
    # this module, so a module-level import here would deadlock the wflow_runs <->
    # sfincs_runs circular dependency at import time.
    from sfincs_runs.scenarios.event_forcing import _find_aorc_event_window

    return _find_aorc_event_window(
        event_windows_dir,
        member_id=str(_required_event_value(row, "rainfall_member_id")),
        storm_start=_required_event_value(row, "rainfall_member_time"),
    )


def _required_event_value(row: pd.Series, key: str):
    return _v2_required_event_value(row, key)


def _catalog_rainfall_start(row: pd.Series):
    return _v2_catalog_rainfall_start(row)


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


def _provenance_window_matches(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not Path(path).exists():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        existing_start = pd.Timestamp(payload.get("time_start"))
        existing_end = pd.Timestamp(payload.get("time_stop"))
    except Exception:
        return False
    return existing_start == pd.Timestamp(start) and existing_end == pd.Timestamp(end)


def _require_event_forcing(event_dir: Path) -> None:
    missing = [name for name in ("precip.nc", "temp_pet.nc") if not (event_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "per-event Wflow forcing is not built (the event_precip/event_temp_pet contract): "
            + ", ".join(str(event_dir / name) for name in missing)
            + ". Run build_meteo before replaying."
        )


def _domain_set_submodels(config: dict, location_root: Path) -> list[dict]:
    return _v2_configured_or_manifest_submodels(config, location_root)


def _write_per_event_data_catalog(data_catalog: Path, event_dir: Path, event_id: str) -> Path:
    return write_legacy_replay_data_catalog(data_catalog, event_dir, event_id)


def _write_per_event_update_config(update_cfg: Path, event_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return write_legacy_replay_update_config(update_cfg, event_dir, start, end)


def _prepare_replay_submodel_output_dir(event_dir: Path, out_dir: Path) -> None:
    clean_legacy_replay_submodel_output_dir(event_dir, out_dir)


def _prepare_wflow_run_output_dir(run_config: Path) -> None:
    """Remove the generated Wflow run-output directory before solver execution."""
    _v2_clean_output_dir(run_config)


def _wflow_run_command(config: dict) -> str:
    """Resolve the Wflow engine command template (``{run_config}`` placeholder)."""
    return _v2_wflow_run_command(config)


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
