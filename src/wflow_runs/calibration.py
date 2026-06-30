from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_sources.usgs_streamgages import fetch_nwis_discharge_records
from wflow_runs.dynamic_handoff import plan_handoff
from wflow_runs.dynamic_handoff_batch import run_handoffs
from wflow_runs.replay import _event_reference_time, resolve_event_window
from wflow_runs.usgs import usgs_instantaneous_streamflow_spec


cfs_to_cms = 0.028316846592


@dataclass(frozen=True)
class ValidationEvents:
    summary: pd.Series
    scenarios: pd.DataFrame
    event_ids: list[str]


@dataclass(frozen=True)
class WflowValidationAudit:
    plan: pd.DataFrame
    report: pd.DataFrame


@dataclass(frozen=True)
class WflowCalibration:
    summary: pd.DataFrame
    artifacts: pd.Series
    patch: dict


def select_validation_events(
    config: dict,
    *,
    scenario_catalog_path,
    readiness_path,
    joint_worklist_path,
    n: int = 5,
) -> ValidationEvents:
    """Choose historical-like events for Wflow validation against observed USGS IV."""
    catalog = pd.read_csv(scenario_catalog_path, dtype={"event_id": str})
    readiness = _read_optional_csv(readiness_path, ["event_id", "status"])
    joint_worklist = _read_optional_csv(joint_worklist_path, ["event_id"])

    explicit_ids = (config.get("wflow", {}).get("validation", {}) or {}).get("historical_event_ids") or []
    if explicit_ids:
        scenarios = catalog[catalog["event_id"].astype(str).isin([str(e) for e in explicit_ids])].copy()
        basis = "config wflow.validation.historical_event_ids"
    elif "rainfall_scale_factor" in catalog:
        pool = catalog.copy()
        pool["historical_proximity"] = (pd.to_numeric(pool["rainfall_scale_factor"], errors="coerce") - 1.0).abs()
        scenarios = pool.nsmallest(n, "historical_proximity").sort_values("event_id").reset_index(drop=True)
        basis = "nearest-to-historical (rainfall_scale_factor approx 1.0)"
    else:
        scenarios = catalog.sort_values("sample_rp_years", ascending=False).head(n).reset_index(drop=True)
        basis = "largest sample_rp_years (fallback)"

    event_ids = scenarios["event_id"].astype(str).tolist()
    summary = pd.Series(
        {
            "catalog_rows": len(catalog),
            "readiness_rows": len(readiness),
            "joint_worklist_rows": len(joint_worklist),
            "validation_event_selection": basis,
            "validation_event_count": len(scenarios),
        },
        name="wflow_validation_inputs",
    )
    return ValidationEvents(summary=summary, scenarios=scenarios, event_ids=event_ids)


def cache_validation_iv_records(
    config: dict,
    location_root,
    event_ids: list[str],
    *,
    scenario_catalog_path,
    events_root,
    wflow_base_root,
    event_streamflow_iv_root,
    rerun: bool = False,
    fetch: bool = True,
    submodel_id: str | None = None,
) -> pd.DataFrame:
    """Fetch/cache observed USGS IV records aligned to Wflow output timesteps."""
    root = Path(event_streamflow_iv_root)
    if not fetch:
        report = pd.DataFrame({"event_id": event_ids, "status": "fetch_disabled"})
    else:
        rows = [
            cache_validation_event_iv_records(
                config,
                location_root,
                event_id,
                scenario_catalog_path=scenario_catalog_path,
                events_root=events_root,
                wflow_base_root=wflow_base_root,
                event_streamflow_iv_root=root,
                rerun=rerun,
                submodel_id=submodel_id,
            )
            for event_id in event_ids
        ]
        report = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    report.to_csv(root.parent / "event_streamflow_iv_report.csv", index=False)
    return report


def cache_validation_event_iv_records(
    config: dict,
    location_root,
    event_id: str,
    *,
    scenario_catalog_path,
    events_root,
    wflow_base_root,
    event_streamflow_iv_root,
    rerun: bool = False,
    submodel_id: str | None = None,
) -> dict:
    location_root = Path(location_root)
    event_streamflow_iv_root = Path(event_streamflow_iv_root)
    reference_time = _event_reference_time(location_root, event_id, scenario_catalog_path)
    start, end = resolve_event_window(reference_time)
    wflow_times = validation_wflow_output_times(event_id, events_root=events_root, submodel_id=submodel_id)
    times_source = "wflow_output"
    if wflow_times.empty:
        wflow_times = pd.date_range(pd.Timestamp(start) + pd.Timedelta(hours=1), pd.Timestamp(end), freq="1h")
        times_source = "expected_hourly_event_window"
    cache_path = validation_iv_cache_path(event_streamflow_iv_root, event_id, start, end, wflow_times)
    site_nos = validation_gauge_sites(
        location_root,
        event_id,
        events_root=events_root,
        wflow_base_root=wflow_base_root,
        submodel_id=submodel_id,
    )

    common = {
        "event_id": str(event_id),
        "records_path": str(cache_path),
        "event_window_start": start,
        "event_window_end": end,
        "wflow_step": validation_wflow_step_label(wflow_times),
        "times_source": times_source,
        "site_nos": ",".join(site_nos),
    }
    if cache_path.exists() and not rerun:
        records = pd.read_csv(cache_path, dtype={"site_no": str}, parse_dates=["time"])
        return {
            **common,
            "status": "cached",
            "record_count": int(len(records)),
            "site_count": int(records["site_no"].astype(str).nunique()) if "site_no" in records else 0,
        }
    if not site_nos:
        return {**common, "status": "no_validation_gauge_sites", "record_count": 0, "site_count": 0}

    spec = usgs_instantaneous_streamflow_spec(config)
    records = []
    for site_no in site_nos:
        records.extend(fetch_nwis_discharge_records(spec, site_no, start, end))

    records_frame = align_iv_records_to_wflow_times(records, wflow_times)
    event_streamflow_iv_root.mkdir(parents=True, exist_ok=True)
    records_frame.to_csv(cache_path, index=False)
    return {
        **common,
        "status": "fetched" if not records_frame.empty else "no_iv_records",
        "record_count": int(len(records_frame)),
        "site_count": int(records_frame["site_no"].astype(str).nunique()) if not records_frame.empty else 0,
    }


def run_wflow_validation_audit(
    config: dict,
    location_root,
    event_ids: list[str],
    *,
    scenario_catalog_path,
    rerun: bool = False,
    execute: bool = False,
) -> WflowValidationAudit:
    plan = pd.DataFrame(
        [plan_handoff(config, location_root, event_id, catalog_path=scenario_catalog_path).to_dict() for event_id in event_ids]
    )
    if not execute:
        report = pd.DataFrame(
            {
                "event_id": event_ids,
                "status": "not_run",
                "message": (
                    "Set EXECUTE_WFLOW_AUDIT=True to run the production dynamic Wflow handoff "
                    "for the historical validation events."
                ),
            }
        )
        return WflowValidationAudit(plan=plan, report=report)

    reports = [
        run_handoffs(
            config,
            location_root,
            catalog_path=scenario_catalog_path,
            event_ids=[event_id],
            status="all",
            execute=True,
            force=rerun,
            overwrite_meteo=rerun,
        )
        for event_id in event_ids
    ]
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    return WflowValidationAudit(plan=plan, report=report)


def build_usgs_wflow_calibration(
    config: dict,
    location_root,
    event_ids: list[str],
    *,
    events_root,
    wflow_base_root,
    event_streamflow_iv_root,
    active_wflow_submodel_id: str | None = None,
) -> WflowCalibration:
    """Score Wflow against observed USGS IV and write the single-K calibration patch."""
    import yaml

    location_root = Path(location_root)
    tables = [
        usgs_calibration_table(
            event_id,
            events_root=events_root,
            wflow_base_root=wflow_base_root,
            event_streamflow_iv_root=event_streamflow_iv_root,
            submodel_id=active_wflow_submodel_id,
        )
        for event_id in event_ids
    ]
    summary = pd.concat([table for table in tables if not table.empty], ignore_index=True) if any(
        not table.empty for table in tables
    ) else pd.DataFrame()
    calibration_root = location_root / "data/wflow/calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    summary_path = calibration_root / "usgs_wflow_calibration_summary.csv"
    summary.to_csv(summary_path, index=False)

    patch, status, suggested_k, event_count = _calibration_patch(config, location_root, summary, summary_path)
    json_path = calibration_root / "wflow_calibration_patch.json"
    yaml_path = calibration_root / "wflow_calibration_patch.yaml"
    json_path.write_text(json.dumps(patch, indent=2), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")

    artifacts = pd.Series(
        {
            "calibration_summary_csv": str(summary_path),
            "calibration_patch_yaml": str(yaml_path),
            "calibration_status": status,
            "suggested_k_calibration": suggested_k,
            "validation_event_count": event_count,
        },
        name="wflow_calibration_artifacts",
    )
    return WflowCalibration(summary=summary, artifacts=artifacts, patch=patch)


def validation_gauge_sites(location_root, event_id, *, events_root, wflow_base_root, submodel_id=None, layer="gauges_usgs") -> list[str]:
    gauges = _read_gauge_layer(event_id, events_root=events_root, wflow_base_root=wflow_base_root, layer=layer, submodel_id=submodel_id)
    if gauges.empty or "site_no" not in gauges:
        return []
    return sorted(gauges["site_no"].dropna().astype(str).unique())


def validation_wflow_output_times(event_id, *, events_root, submodel_id=None) -> pd.DatetimeIndex:
    submodel_id = submodel_id or _first_submodel_id(events_root, event_id)
    if not submodel_id:
        return pd.DatetimeIndex([])
    output_path = Path(events_root) / str(event_id) / submodel_id / "run_event" / "output.csv"
    if not output_path.exists():
        return pd.DatetimeIndex([])
    times = pd.read_csv(output_path, usecols=["time"], parse_dates=["time"])["time"]
    return pd.DatetimeIndex(times.drop_duplicates()).sort_values()


def validation_wflow_step_label(wflow_times: pd.DatetimeIndex) -> str:
    if len(wflow_times) < 2:
        return "unknown_step"
    step = pd.Series(wflow_times).diff().dropna().median()
    seconds = int(pd.Timedelta(step).total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def validation_iv_cache_path(root, event_id, start, end, wflow_times: pd.DatetimeIndex) -> Path:
    start_label = pd.Timestamp(start).strftime("%Y%m%dT%H%M%S")
    end_label = pd.Timestamp(end).strftime("%Y%m%dT%H%M%S")
    return Path(root) / f"{event_id}_gauges_usgs_wflow_timestep_{validation_wflow_step_label(wflow_times)}_{start_label}_{end_label}.csv"


def align_iv_records_to_wflow_times(records, wflow_times: pd.DatetimeIndex) -> pd.DataFrame:
    columns = ["site_no", "time", "discharge_cfs", "source"]
    if not records or wflow_times.empty:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)
    frame["site_no"] = frame["site_no"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    frame = frame.dropna(subset=["site_no", "time", "discharge_cfs"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    step = pd.Series(wflow_times).diff().dropna().median() if len(wflow_times) > 1 else pd.Timedelta(hours=1)
    aligned_frames = []
    for site_no, group in frame.groupby("site_no"):
        series = group.set_index("time").sort_index()["discharge_cfs"].astype(float).groupby(level=0).mean()
        resampled = series.resample(pd.Timedelta(step), origin=wflow_times[0]).mean()
        aligned = resampled.reindex(resampled.index.union(wflow_times)).sort_index().interpolate("time").reindex(wflow_times)
        site_frame = aligned.dropna().rename("discharge_cfs").reset_index().rename(columns={"index": "time"})
        site_frame["site_no"] = str(site_no)
        site_frame["source"] = "usgs_iv_aligned_to_wflow"
        aligned_frames.append(site_frame[columns])
    return pd.concat(aligned_frames, ignore_index=True).sort_values(["site_no", "time"]) if aligned_frames else pd.DataFrame(columns=columns)


def read_wflow_output_csv(event_id, *, events_root, submodel_id=None, control: bool = False) -> pd.DataFrame:
    submodel_id = submodel_id or _first_submodel_id(events_root, event_id)
    if not submodel_id:
        return pd.DataFrame()
    event_root = Path(events_root) / str(event_id)
    model_root = event_root / "_zero_rain" / submodel_id if control else event_root / submodel_id
    csv_path = model_root / "run_event" / "output.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def gauge_output_map(event_id, *, events_root, wflow_base_root, layer="gauges_usgs", submodel_id=None) -> pd.DataFrame:
    gauges = _read_gauge_layer(event_id, events_root=events_root, wflow_base_root=wflow_base_root, layer=layer, submodel_id=submodel_id)
    if gauges.empty:
        return gauges
    gauges["q_column"] = "Q_" + gauges["index"].astype(int).astype(str)
    return pd.DataFrame(gauges.drop(columns="geometry"))


def event_iv_records(event_id, event_streamflow_iv_root) -> pd.DataFrame:
    root = Path(event_streamflow_iv_root)
    files = sorted(root.glob(f"{event_id}_*wflow_timestep*.csv")) or sorted(root.glob(f"{event_id}_*.csv"))
    if not files:
        return pd.DataFrame(columns=["site_no", "time", "discharge_cfs", "source"])
    records = pd.concat(
        [pd.read_csv(path, dtype={"site_no": str}, parse_dates=["time"]) for path in files],
        ignore_index=True,
    ).drop_duplicates(subset=["site_no", "time"])
    records["source"] = records.get("source", pd.Series("usgs_iv", index=records.index)).fillna("usgs_iv")
    return records.sort_values(["site_no", "time"])


def observed_event_site_flow(event_id, site_no, target_index, *, event_streamflow_iv_root) -> pd.Series:
    target_index = pd.DatetimeIndex(target_index).sort_values()
    records = event_iv_records(event_id, event_streamflow_iv_root)
    selected = records[records["site_no"].astype(str).eq(str(site_no))].copy()
    if selected.empty or target_index.empty:
        return pd.Series(dtype=float, name=str(site_no))
    observed = selected.set_index("time").sort_index()["discharge_cfs"].astype(float) * cfs_to_cms
    observed = observed.groupby(level=0).mean()
    observed = observed.loc[(observed.index >= target_index.min()) & (observed.index <= target_index.max())]
    if observed.empty:
        return pd.Series(dtype=float, name=str(site_no))
    aligned = observed.reindex(observed.index.union(target_index)).sort_index().interpolate("time").reindex(target_index)
    return aligned.rename(str(site_no))


def hydrograph_scores(simulated: pd.Series, observed: pd.Series) -> dict:
    joined = pd.concat([simulated.rename("sim"), observed.rename("obs")], axis=1).dropna()
    if joined.empty:
        return {"n": 0, "nse": np.nan, "kge": np.nan, "peak_bias_fraction": np.nan, "volume_bias_fraction": np.nan}
    obs = joined["obs"].to_numpy(dtype=float)
    sim = joined["sim"].to_numpy(dtype=float)
    denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((sim - obs) ** 2) / denom if denom else np.nan
    r = np.corrcoef(sim, obs)[0, 1] if len(joined) > 1 and np.std(sim) and np.std(obs) else np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) else np.nan
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite([r, alpha, beta]).all() else np.nan
    return {
        "n": int(len(joined)),
        "nse": float(nse) if np.isfinite(nse) else np.nan,
        "kge": float(kge) if np.isfinite(kge) else np.nan,
        "peak_bias_fraction": float((sim.max() - obs.max()) / obs.max()) if obs.max() else np.nan,
        "volume_bias_fraction": float((sim.sum() - obs.sum()) / obs.sum()) if obs.sum() else np.nan,
    }


def usgs_calibration_table(event_id, *, events_root, wflow_base_root, event_streamflow_iv_root, submodel_id=None) -> pd.DataFrame:
    sim = read_wflow_output_csv(event_id, events_root=events_root, submodel_id=submodel_id)
    gauges = gauge_output_map(event_id, events_root=events_root, wflow_base_root=wflow_base_root, layer="gauges_usgs", submodel_id=submodel_id)
    records = event_iv_records(event_id, event_streamflow_iv_root)
    if sim.empty or gauges.empty or records.empty:
        return pd.DataFrame()
    rows = []
    for _, gauge in gauges.iterrows():
        q_col = str(gauge["q_column"])
        if q_col not in sim:
            continue
        obs = observed_event_site_flow(event_id, gauge["site_no"], sim.index, event_streamflow_iv_root=event_streamflow_iv_root)
        scores = hydrograph_scores(sim[q_col].astype(float), obs)
        if scores["n"] > 0:
            rows.append({"event_id": event_id, "site_no": str(gauge["site_no"]), "q_column": q_col, **scores})
    return pd.DataFrame(rows)


def _calibration_patch(config: dict, location_root: Path, calibration_summary: pd.DataFrame, summary_path: Path) -> tuple[dict, str, float | None, int]:
    min_overlap_steps = 12
    min_events_for_ship = 2
    default_k_band = [0.25, 4.0]
    amp_cfg = ((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {})
    primary_reference_gage = amp_cfg.get("primary_reference_gage") or (config.get("inland_coupling", {}) or {}).get("primary_reference_gage")

    valid = calibration_summary.copy()
    if primary_reference_gage and not valid.empty and "site_no" in valid:
        valid = valid[valid["site_no"].astype(str).eq(str(primary_reference_gage))].copy()
    if not valid.empty and {"n", "peak_bias_fraction"}.issubset(valid.columns):
        for column in ["n", "peak_bias_fraction", "volume_bias_fraction", "kge", "nse"]:
            if column in valid:
                valid[column] = pd.to_numeric(valid[column], errors="coerce")
        valid = valid[valid["n"].ge(min_overlap_steps) & valid["peak_bias_fraction"].gt(-0.95)].copy()
        valid["k_peak"] = 1.0 / (1.0 + valid["peak_bias_fraction"])
        if "volume_bias_fraction" in valid:
            valid["k_volume"] = 1.0 / (1.0 + valid["volume_bias_fraction"])
        valid = valid[np.isfinite(valid["k_peak"])]

    suggested_k = float(valid["k_peak"].median()) if not valid.empty and "k_peak" in valid else None
    event_count = int(valid["event_id"].nunique()) if not valid.empty and "event_id" in valid else 0
    status = "ready_to_ship" if suggested_k and event_count >= min_events_for_ship else "insufficient_validation_events"
    patch = {
        "location": location_root.name,
        "status": status,
        "method": "single-K peak-bias correction from Wflow-vs-USGS IV validation events",
        "primary_reference_gage": str(primary_reference_gage) if primary_reference_gage else None,
        "event_count": event_count,
        "row_count": int(len(valid)),
        "k_calibration": suggested_k,
        "k_band": amp_cfg.get("k_band", default_k_band),
        "source_summary_csv": str(summary_path.relative_to(location_root)),
        "runtime_config_patch": {
            "inland_coupling": {
                "amplification": {
                    "enabled": True,
                    "primary_reference_gage": str(primary_reference_gage) if primary_reference_gage else None,
                    "k_calibration": suggested_k,
                    "k_band": amp_cfg.get("k_band", default_k_band),
                }
            }
        },
    }
    return patch, status, suggested_k, event_count


def _read_optional_csv(path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame(columns=columns)


def _read_gauge_layer(event_id, *, events_root, wflow_base_root, layer: str, submodel_id=None) -> pd.DataFrame:
    import geopandas as gpd

    submodel_id = submodel_id or _first_submodel_id(events_root, event_id) or _first_base_submodel_id(wflow_base_root)
    if not submodel_id:
        return pd.DataFrame()
    gauges_path = Path(events_root) / str(event_id) / submodel_id / "staticgeoms" / f"{layer}.geojson"
    if not gauges_path.exists():
        gauges_path = Path(wflow_base_root) / submodel_id / "staticgeoms" / f"{layer}.geojson"
    if not gauges_path.exists():
        return pd.DataFrame()
    return gpd.read_file(gauges_path)


def _first_submodel_id(events_root, event_id) -> str | None:
    event_root = Path(events_root) / str(event_id)
    if not event_root.exists():
        return None
    for child in sorted(event_root.iterdir()):
        if child.is_dir() and (child / "run_event").exists():
            return child.name
    return None


def _first_base_submodel_id(wflow_base_root) -> str | None:
    root = Path(wflow_base_root)
    if not root.exists():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "staticgeoms").exists():
            return child.name
    return None
