from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr

from event_streamflow import finite_float
from paths import resolve_location_path
from wflow_runs.event import legacy_event_catalog_row
from wflow_runs.output import match_gauge_column, resolve_wflow_output_csv

CFS_TO_CMS = 0.028316846592


def apply_same_frequency_amplification(
    config: dict,
    location_root,
    event_id: str | None = None,
    *,
    catalog_path=None,
    discharge_nc,
    submodel_runs=None,
    event=None,
    write_provenance: bool = True,
) -> dict:
    """Apply one Same-Frequency Amplification factor to a merged SFINCS discharge artifact."""
    location_root = Path(location_root)
    discharge_nc = Path(discharge_nc)
    if event is not None:
        return _apply_design_event_same_frequency_amplification(
            config,
            location_root,
            event,
            discharge_nc=discharge_nc,
            submodel_runs=submodel_runs or [],
            write_provenance=write_provenance,
        )
    if event_id is None:
        raise ValueError("event_id is required when event is not provided")

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
        _write_json(provenance_path, provenance)
        return provenance

    k_calibration = finite_float(amp_cfg.get("k_calibration"))
    if k_calibration and k_calibration > 0 and not amp_cfg.get("prefer_per_event_target", False):
        k = float(k_calibration)
        band = amp_cfg.get("k_band")
        in_band = True if not band else (float(band[0]) <= k <= float(band[1]))
        provenance.update(
            K=k,
            status="calibration_constant" if in_band else "calibration_constant_out_of_band",
            reference_gage=str(reference_gage) if reference_gage else None,
        )
        _scale_discharge_in_place(discharge_nc, k, reference_gage)
        _write_json(provenance_path, provenance)
        return provenance

    if not reference_gage:
        provenance["status"] = "response_based_unbiased"
        _write_json(provenance_path, provenance)
        return provenance

    try:
        row = legacy_event_catalog_row(location_root, event_id, catalog_path)
        target_cms = _event_streamflow_target_cms(row)
        wflow_peak_cms = _reference_gage_simulated_peak_cms(
            config,
            location_root,
            reference_gage,
            submodel_runs,
        )
    except Exception as exc:
        provenance.update(status=f"skipped:{type(exc).__name__}", reference_gage=str(reference_gage))
        _write_json(provenance_path, provenance)
        return provenance

    provenance.update(reference_gage=str(reference_gage), target_cms=target_cms, wflow_peak_cms=wflow_peak_cms)
    if not target_cms or not wflow_peak_cms or wflow_peak_cms <= 0:
        provenance["status"] = "no_target" if not target_cms else "no_wflow_peak"
        _write_json(provenance_path, provenance)
        return provenance

    k = float(target_cms) / float(wflow_peak_cms)
    band = amp_cfg.get("k_band")
    in_band = True if not band else (float(band[0]) <= k <= float(band[1]))
    provenance.update(K=k, status="applied" if in_band else "applied_out_of_band")
    _scale_discharge_in_place(discharge_nc, k, reference_gage)
    _write_json(provenance_path, provenance)
    return provenance


def _apply_design_event_same_frequency_amplification(
    config: dict,
    location_root: Path,
    event,
    *,
    discharge_nc: Path,
    submodel_runs,
    write_provenance: bool,
) -> dict:
    amp_cfg = ((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {})
    reference_gage = amp_cfg.get("primary_reference_gage") or (
        config.get("inland_coupling", {}) or {}
    ).get("primary_reference_gage")
    band = amp_cfg.get("k_band", [0.25, 4.0])
    provenance = {
        "method": "same_frequency_amplification",
        "equation": "K = clip(Q*_omega(g*) / max_t Q^W_omega(g*,t), K_min, K_max)",
        "K": 1.0,
        "status": "response_based_unbiased",
        "reference_gage": reference_gage,
        "target_cms": event.q_target_cms,
        "wflow_peak_cms": None,
        "k_band": band,
    }
    if not amp_cfg.get("enabled", True):
        provenance["status"] = "disabled"
        _maybe_write_amplification_json(discharge_nc, provenance, write_provenance)
        return provenance
    k_cal = finite_float(amp_cfg.get("k_calibration"))
    if k_cal and k_cal > 0 and not amp_cfg.get("prefer_per_event_target", False):
        k = _clip(float(k_cal), band)
        _scale_discharge_in_place(discharge_nc, k, reference_gage)
        provenance.update(K=k, status="calibration_constant")
        _maybe_write_amplification_json(discharge_nc, provenance, write_provenance)
        return provenance
    if not reference_gage or not event.q_target_cms:
        _maybe_write_amplification_json(discharge_nc, provenance, write_provenance)
        return provenance
    peak = _reference_gage_simulated_peak_cms(config, location_root, str(reference_gage), submodel_runs)
    provenance["wflow_peak_cms"] = peak
    if not peak or peak <= 0:
        provenance["status"] = "no_wflow_peak"
        _maybe_write_amplification_json(discharge_nc, provenance, write_provenance)
        return provenance
    k = _clip(float(event.q_target_cms) / float(peak), band)
    _scale_discharge_in_place(discharge_nc, k, reference_gage)
    provenance.update(K=k, status="applied")
    _maybe_write_amplification_json(discharge_nc, provenance, write_provenance)
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


def _clip(value: float, band) -> float:
    if not band:
        return float(value)
    return float(np.clip(float(value), float(band[0]), float(band[1])))


def _maybe_write_amplification_json(discharge_nc: Path, provenance: dict, enabled: bool) -> None:
    if enabled:
        _write_json(discharge_nc.with_name("sfincs_discharge.amplification.json"), provenance)


def _event_streamflow_target_cms(row: pd.Series) -> float | None:
    value = finite_float(row.get("streamflow_target_cfs"))
    if value and value > 0:
        return float(value) * CFS_TO_CMS
    return None


def _reference_gage_simulated_peak_cms(
    config: dict,
    location_root: Path,
    reference_gage: str,
    submodel_runs,
) -> float | None:
    import geopandas as gpd

    reference_gage = str(reference_gage)
    for entry in submodel_runs or []:
        run_output_dir = Path(entry["run_output_dir"])
        submodel_id = run_output_dir.parent.name
        gauges_path = _observation_gauges_path(config, location_root, submodel_id)
        if not gauges_path.exists():
            continue
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges or "index" not in gauges:
            continue
        match = gauges[gauges["site_no"].astype(str).str.zfill(8) == reference_gage.zfill(8)]
        if match.empty:
            continue
        try:
            csv_path = resolve_wflow_output_csv(run_output_dir, None)
            table = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except (FileNotFoundError, ValueError):
            continue
        column = match_gauge_column(table.columns, match.iloc[0]["index"])
        if column is None:
            continue
        peak = float(pd.to_numeric(table[column], errors="coerce").max())
        if np.isfinite(peak):
            return peak
    return None


def _observation_gauges_path(config: dict, location_root: Path, submodel_id: str) -> Path:
    root = (
        ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root")
        or "data/wflow/domain_set_gauges"
    )
    root = resolve_location_path(location_root, root)
    return root / f"{submodel_id}_observation_gauges.geojson"


def _write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
