from __future__ import annotations

from pathlib import Path

import pandas as pd

from collect_sources.usgs_streamgages import fetch_nwis_discharge_records
from wflow_runs.event_catalog import event_reference_time, event_window
from wflow_runs.output import first_event_submodel_id, read_wflow_gauge_layer
from wflow_runs.usgs import usgs_instantaneous_streamflow_spec

cfs_to_cms = 0.028316846592


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
    reference_time = event_reference_time(location_root, event_id, scenario_catalog_path)
    start, end = event_window(reference_time)
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


def validation_gauge_sites(location_root, event_id, *, events_root, wflow_base_root, submodel_id=None, layer="gauges_usgs") -> list[str]:
    gauges = read_wflow_gauge_layer(event_id, events_root=events_root, wflow_base_root=wflow_base_root, layer=layer, submodel_id=submodel_id)
    if gauges.empty or "site_no" not in gauges:
        return []
    return sorted(gauges["site_no"].dropna().astype(str).unique())


def validation_wflow_output_times(event_id, *, events_root, submodel_id=None) -> pd.DatetimeIndex:
    submodel_id = submodel_id or first_event_submodel_id(events_root, event_id)
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
