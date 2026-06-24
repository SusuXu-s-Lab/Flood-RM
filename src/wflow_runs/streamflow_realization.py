from __future__ import annotations

from pathlib import Path
import json
import re
import tomllib

import numpy as np
import pandas as pd
import tomli_w
import xarray as xr
import yaml

from design_events.collect_sources.usgs_streamgages import fetch_nwis_discharge_records
from wflow_runs.notebook import resolve_location_path


WFLOW_EXTERNAL_RIVER_INFLOW = "river_water__external_inflow_volume_flow_rate"
WFLOW_EXTERNAL_RIVER_INFLOW_VAR = "river_inflow"
CFS_TO_CMS = 0.028316846592


def apply_same_frequency_amplification(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    discharge_nc,
    submodel_runs,
) -> dict:
    """ADR-0016 Same-Frequency Amplification of Wflow-generated handoff discharge.

    Scales every handoff hydrograph in ``discharge_nc`` by a single event-level factor
    ``K = streamflow_target / Q_wflow_peak`` evaluated at the Primary Reference Gage, so the
    routed flow honors the observed streamflow-frequency target while preserving hydrograph
    shape, timing, and inter-tributary structure. This is an empirical bias/frequency
    correction on model output, never a prescribed boundary forcing.

    No-op (``K=1``) unless a ``primary_reference_gage`` is configured and the catalog row
    carries a streamflow target and the reference gage's simulated peak can be read. Writes
    ``sfincs_discharge.amplification.json`` provenance next to ``discharge_nc``.
    """
    location_root = Path(location_root)
    discharge_nc = Path(discharge_nc)
    amp_cfg = (((config.get("inland_coupling", {}) or {}).get("amplification", {})) or {})
    provenance = {
        "event_id": str(event_id),
        "method": "same_frequency_amplification",
        "K": 1.0,
        "status": "disabled",
        "reference_gage": None,
        "target_cms": None,
        "wflow_peak_cms": None,
        "k_band": list(amp_cfg.get("k_band", [])) or None,
    }
    provenance_path = discharge_nc.with_name("sfincs_discharge.amplification.json")

    reference_gage = amp_cfg.get("primary_reference_gage") or (
        ((config.get("inland_coupling", {}) or {}).get("primary_reference_gage"))
    )
    if not amp_cfg.get("enabled", True):
        _write_amplification_provenance(provenance_path, provenance)
        return provenance

    # Response-based (ADR-0016): discharge is the Wflow response; K is an empirical
    # calibration BIAS correction, not a per-event streamflow return period. A configured
    # ``k_calibration`` constant (derived from Wflow calibration/validation against the
    # Primary Reference Gage) is applied uniformly; absent that, K=1 (unbiased) unless the
    # catalog carries an explicit per-event ``streamflow_target_cfs``.
    k_calibration = _finite_float(amp_cfg.get("k_calibration"))
    if k_calibration and k_calibration > 0 and not amp_cfg.get("prefer_per_event_target", False):
        k = float(k_calibration)
        band = amp_cfg.get("k_band")
        in_band = True if not band else (float(band[0]) <= k <= float(band[1]))
        provenance.update(K=k, status="calibration_constant" if in_band else "calibration_constant_out_of_band", reference_gage=str(reference_gage) if reference_gage else None)
        _scale_discharge_in_place(discharge_nc, k, reference_gage)
        _write_amplification_provenance(provenance_path, provenance)
        return provenance

    if not reference_gage:
        provenance["status"] = "response_based_unbiased"
        _write_amplification_provenance(provenance_path, provenance)
        return provenance

    try:
        row = _event_catalog_row(location_root, event_id, catalog_path)
        target_cms = _event_streamflow_target_cms(row)
        wflow_peak_cms = _reference_gage_simulated_peak_cms(
            config, location_root, reference_gage, submodel_runs
        )
    except Exception as exc:  # defensive: amplification must never break the handoff
        provenance.update(status=f"skipped:{type(exc).__name__}", reference_gage=str(reference_gage))
        _write_amplification_provenance(provenance_path, provenance)
        return provenance

    provenance.update(reference_gage=str(reference_gage), target_cms=target_cms, wflow_peak_cms=wflow_peak_cms)
    if not target_cms or not wflow_peak_cms or wflow_peak_cms <= 0:
        provenance["status"] = "no_target" if not target_cms else "no_wflow_peak"
        _write_amplification_provenance(provenance_path, provenance)
        return provenance

    k = float(target_cms) / float(wflow_peak_cms)
    band = amp_cfg.get("k_band")
    in_band = True if not band else (float(band[0]) <= k <= float(band[1]))
    provenance.update(K=k, status="applied" if in_band else "applied_out_of_band")
    _scale_discharge_in_place(discharge_nc, k, reference_gage)
    _write_amplification_provenance(provenance_path, provenance)
    return provenance


def _scale_discharge_in_place(discharge_nc: Path, k: float, reference_gage) -> None:
    if not float(k) or float(k) == 1.0:
        return
    with xr.open_dataset(discharge_nc) as src:
        ds = src.load()
    if "discharge" not in ds:
        return
    ds["discharge"] = ds["discharge"] * float(k)
    ds["discharge"].attrs["same_frequency_amplification_K"] = float(k)
    ds.attrs["same_frequency_amplification_K"] = float(k)
    if reference_gage:
        ds.attrs["amplification_reference_gage"] = str(reference_gage)
    tmp = discharge_nc.with_suffix(".amp.tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(discharge_nc)


def _write_amplification_provenance(path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _event_streamflow_target_cms(row: pd.Series) -> float | None:
    """Per-event streamflow target peak at the Primary Reference Gage, in m3/s.

    Reads ONLY the ADR-0016 catalog column ``streamflow_target_cfs`` — the POT target at the
    Primary Reference Gage produced by 03's in-domain anchor. The legacy ``streamflow_cfs`` /
    ``streamflow`` columns are deliberately NOT used: they hold the retired cross-basin
    max-envelope magnitude (~mainstem scale), which would massively over-amplify. Until 03 is
    migrated, this returns None and amplification safely no-ops (status ``no_target``).
    """
    value = _finite_float(row.get("streamflow_target_cfs"))
    if value and value > 0:
        return float(value) * CFS_TO_CMS
    return None


def _reference_gage_simulated_peak_cms(
    config: dict, location_root: Path, reference_gage: str, submodel_runs
) -> float | None:
    """Peak simulated Wflow discharge (m3/s) at the Primary Reference Gage's output cell.

    Locates the reference gage in each submodel's observation-gauge set, matches its Wflow
    ``Q_<index>`` output column, and returns the peak. Returns None if the gage is not an
    output point in any submodel (then amplification no-ops rather than guesses).
    """
    import geopandas as gpd

    from wflow_runs.replay import _match_gauge_column, _resolve_wflow_output_csv

    reference_gage = str(reference_gage)
    for entry in submodel_runs or []:
        run_output_dir = Path(entry["run_output_dir"])
        # The output dir is .../events/<event>/<submodel_id>/run_event; submodel id is its parent name.
        submodel_id = run_output_dir.parent.name
        gauges_path = _observation_gauges_path(config, location_root, submodel_id)
        if not gauges_path.exists():
            continue
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges or "index" not in gauges:
            continue
        match = gauges[gauges["site_no"].astype(str) == reference_gage]
        if match.empty:
            continue
        try:
            csv_path = _resolve_wflow_output_csv(run_output_dir, None)
            table = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except (FileNotFoundError, ValueError):
            continue
        column = _match_gauge_column(table.columns, match.iloc[0]["index"])
        if column is None:
            continue
        peak = float(pd.to_numeric(table[column], errors="coerce").max())
        if np.isfinite(peak):
            return peak
    return None


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
        "streamflow_scale_factor": _streamflow_scale_factor(row, member),
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


def validate_wflow_streamflow_realization(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    event_model_root=None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """ADR-0016: validate the rainfall-driven inland event is ready for Wflow generation.

    Discharge is the Wflow *response*, not an injected streamflow member, so readiness now
    checks the rainfall-driven design contract: a rainfall member is wired, and the
    Same-Frequency Amplification / baseflow Primary Reference Gage is configured. When an
    event model is supplied it confirms precip forcing is present and that NO external river
    inflow is wired (legacy injection would double-count the gauged rainfall-runoff response).
    """
    location_root = Path(location_root)
    row = _event_catalog_row(location_root, event_id, catalog_path)
    rows = [
        _catalog_rainfall_row(row),
        _amplification_reference_row(config),
    ]
    if event_model_root is not None:
        rows.extend(_event_model_rainfall_forcing_rows(Path(event_model_root)))
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Inland rainfall-driven event is not ready for Wflow generation: {details}")
    return report


def _catalog_rainfall_row(row: pd.Series) -> dict:
    missing = [
        key
        for key in ("rainfall_member_id", "rainfall_member_file")
        if row.get(key) is None or pd.isna(row.get(key)) or str(row.get(key)).strip() == ""
    ]
    if missing:
        return {"check": "catalog_rainfall_member", "status": "failed", "message": "missing " + ", ".join(missing)}
    return {
        "check": "catalog_rainfall_member",
        "status": "passed",
        "message": f"member={row.get('rainfall_member_id')}; scale={row.get('rainfall_scale_factor')}",
    }


def _amplification_reference_row(config: dict) -> dict:
    gage = (((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {}).get("primary_reference_gage"))
    if not gage:
        return {
            "check": "amplification_reference_gage",
            "status": "review_required",
            "message": "inland_coupling.amplification.primary_reference_gage unset (single-K/baseflow validation anchor)",
        }
    return {"check": "amplification_reference_gage", "status": "passed", "message": f"primary_reference_gage={gage}"}


def _event_model_rainfall_forcing_rows(event_model_root: Path) -> list[dict]:
    forcing_path = event_model_root / "inmaps-event.nc"
    rows: list[dict] = []
    if not forcing_path.exists():
        rows.append({"check": "wflow_event_precip_forcing", "status": "failed", "message": f"missing {forcing_path}"})
        return rows
    with xr.open_dataset(forcing_path) as ds:
        has_precip = "precip" in ds
        has_inflow = WFLOW_EXTERNAL_RIVER_INFLOW_VAR in ds
    rows.append(
        {
            "check": "wflow_event_precip_forcing",
            "status": "passed" if has_precip else "failed",
            "message": "precip present" if has_precip else "precip missing from inmaps-event.nc",
        }
    )
    rows.append(
        {
            "check": "wflow_no_external_inflow",
            "status": "passed" if not has_inflow else "review_required",
            "message": (
                "no external river_inflow (rainfall-driven, ADR-0016)"
                if not has_inflow
                else "legacy external river_inflow present — rainfall-runoff double-count risk"
            ),
        }
    )
    return rows


def wflow_streamflow_gage_overlap(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    submodel_ids: list[str] | None = None,
) -> dict:
    """Describe whether an event's streamflow member can force reviewed Wflow gauges."""
    import geopandas as gpd

    location_root = Path(location_root)
    row = _event_catalog_row(location_root, event_id, catalog_path)
    member = _streamflow_member_metadata(config, location_root, row)
    member_sites = {str(site) for site in member["site_nos"]}
    submodel_ids = submodel_ids or _active_wflow_submodel_ids(config, location_root)
    reviewed_sites: set[str] = set()
    gauge_paths: list[str] = []
    missing_paths: list[str] = []
    for submodel_id in submodel_ids:
        gauges_path = _observation_gauges_path(config, location_root, submodel_id)
        if not gauges_path.exists():
            missing_paths.append(str(gauges_path))
            continue
        gauge_paths.append(str(gauges_path))
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges:
            raise ValueError(f"{gauges_path} lacks site_no column for streamflow realization")
        reviewed_sites.update(gauges["site_no"].astype(str))

    overlap = sorted(member_sites & reviewed_sites)
    compatible = bool(overlap)
    if compatible:
        message = (
            f"streamflow member {member['member_id']} overlaps reviewed Wflow gauges: "
            + ", ".join(overlap)
        )
    else:
        message = (
            f"streamflow member {member['member_id']} sites do not overlap reviewed Wflow observation gauges "
            f"for active submodels {submodel_ids or 'none'}."
        )
        if missing_paths:
            message += " Missing gauge files: " + ", ".join(missing_paths)
    return {
        "event_id": str(event_id),
        "member_id": member["member_id"],
        "member_sites": sorted(member_sites),
        "submodel_ids": list(submodel_ids),
        "reviewed_site_count": len(reviewed_sites),
        "overlap_site_nos": overlap,
        "compatible": compatible,
        "gauge_paths": gauge_paths,
        "message": message,
    }


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
    """Fetch and cache USGS instantaneous discharge for one event analog window.

    The cache is the production input used by ``prepare_wflow_streamflow_realization``.
    It is intentionally separate from model execution so local preflight can download
    event-window IV hydrographs before inputs are synced to the cluster.
    """
    import geopandas as gpd

    location_root = Path(location_root)
    row = _event_catalog_row(location_root, event_id, catalog_path)
    member = _streamflow_member_metadata(config, location_root, row)
    cache_path = _event_streamflow_records_path(config, location_root, event_id, member, start, end)
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
    for submodel_id in _active_wflow_submodel_ids(config, location_root):
        gauges_path = _observation_gauges_path(config, location_root, submodel_id)
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


def _active_wflow_submodel_ids(config: dict, location_root: Path) -> list[str]:
    domain_set = ((config.get("wflow", {}) or {}).get("domain_set", {}) or {})
    configured = [
        str(item["wflow_submodel_id"])
        for item in domain_set.get("submodels", []) or []
        if item.get("wflow_submodel_id")
    ]
    if configured:
        return configured

    manifest_path = resolve_location_path(
        location_root,
        (config.get("wflow", {}) or {}).get("domain_set_manifest", "data/wflow/domain_set.yaml"),
    )
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifested = [
            str(item["wflow_submodel_id"])
            for item in manifest.get("submodels", []) or []
            if item.get("wflow_submodel_id")
        ]
        if manifested:
            return manifested

    gauges_root = (
        ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root")
        or "data/wflow/domain_set_gauges"
    )
    gauges_root = resolve_location_path(location_root, gauges_root)
    suffix = "_observation_gauges.geojson"
    return sorted(path.name[: -len(suffix)] for path in gauges_root.glob(f"*{suffix}"))


def _load_streamflow_records(config: dict, location_root: Path, site_nos: list[str]) -> pd.DataFrame:
    records_path = _streamflow_records_path(config, location_root)
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
    cache_path = _event_streamflow_records_path(config, location_root, event_id, member, start, end)
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
        "records_path": str(_streamflow_records_path(config, location_root)),
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
    spec = dict(((config.get("collection", {}) or {}).get("usgs_streamgages", {}) or {}))
    records_cfg = dict(spec.get("streamflow_records", {}) or {})
    records_cfg["service"] = "iv"
    records_cfg.pop("stat_cd", None)
    spec["streamflow_records"] = records_cfg
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


def _event_streamflow_records_path(
    config: dict,
    location_root: Path,
    event_id: str,
    member: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    settings = (((config.get("wflow", {}) or {}).get("streamflow_realization", {}) or {}))
    root = resolve_location_path(
        location_root,
        settings.get("event_records_root", "data/sources/usgs_streamgages/event_streamflow_iv"),
    )
    token = "_".join(
        [
            str(event_id),
            str(member["member_id"]),
            pd.Timestamp(start).strftime("%Y%m%dT%H%M%S"),
            pd.Timestamp(end).strftime("%Y%m%dT%H%M%S"),
        ]
    )
    return root / f"{_safe_filename_token(token)}.csv"


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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
