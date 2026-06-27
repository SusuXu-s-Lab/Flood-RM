import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from sfincs_runs.config import load_runtime, parse_sfincs_inp
m_to_ft = 3.280839895013123
km2_to_acres = 247.10538146716534


def event_id(x):
    text = str(x).strip()
    if text.startswith("evt_") and text[4:].isdigit():
        return f"evt_{int(text[4:]):04d}"
    if text.isdigit() and int(text) > 0:
        return f"evt_{int(text):04d}"
    if text and "/" not in text and "\\" not in text and text not in {".", ".."}:
        return text
    raise ValueError(f"Bad event id: {x!r}")


def event_dirs(root, ids=None, limit=None):
    selected = sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("."))
    if ids:
        wanted = {event_id(x) for x in ids}
        selected = [p for p in selected if p.name in wanted]
        missing = wanted - {p.name for p in selected}
        if missing:
            raise FileNotFoundError(f"Missing events: {', '.join(sorted(missing))}")
    return selected[:limit] if limit is not None else selected


def _self_contained_run_output(event_dir):
    event_dir = Path(event_dir)
    return all((event_dir / name).exists() for name in ["sfincs_map.nc", "sfincs.inp", "sfincs.bzs", "forcing_manifest.json"])


def completed_event_inventory(scenarios_root, storage_root, ids=None, limit=None):
    scenario_events = event_dirs(scenarios_root)
    storage_events = event_dirs(storage_root)
    if ids:
        wanted = {event_id(x) for x in ids}
        scenario_events = [p for p in scenario_events if p.name in wanted]
        storage_events = [p for p in storage_events if p.name in wanted]
        missing = wanted - {p.name for p in [*scenario_events, *storage_events]}
        if missing:
            raise FileNotFoundError(f"Missing events: {', '.join(sorted(missing))}")
    scenario_completed = [d for d in scenario_events if (Path(storage_root) / d.name / "sfincs_map.nc").exists()]
    storage_completed = [d for d in storage_events if _self_contained_run_output(d)]

    use_storage_outputs = len(storage_completed) > len(scenario_completed)
    all_events = storage_events if use_storage_outputs else scenario_events
    completed_events = storage_completed if use_storage_outputs else scenario_completed
    if limit is not None:
        completed_events = completed_events[:limit]

    return {
        "scenario_events": scenario_events,
        "storage_events": storage_events,
        "scenario_completed": scenario_completed,
        "storage_completed": storage_completed,
        "all_events": all_events,
        "completed_events": completed_events,
        "event_source_root": Path(storage_root) if use_storage_outputs else Path(scenarios_root),
        "use_storage_outputs": use_storage_outputs,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Write per-event SFINCS flood statistics.")
    p.add_argument("--config", default=None, help="optional config overlay yaml")
    p.add_argument("--event-id", action="append", dest="event_ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--scenarios-dir", type=Path, default=None)
    p.add_argument("--storage-dir", type=Path, default=None)
    p.add_argument("--stats-dir", type=Path, default=None)
    p.add_argument("--coastline-geojson", type=Path, default=None, help="Unused; kept for old command lines.")
    p.add_argument("--land-threshold-m", type=float, default=0.0)
    p.add_argument("--huthresh-m", type=float, default=0.01)
    p.add_argument("--impact-threshold-m", type=float, default=0.10)
    p.add_argument("--plots-dir", type=Path)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    _, runtime_paths = load_runtime(args.config)
    args.scenarios_dir = args.scenarios_dir or runtime_paths["scenarios_root"]
    args.storage_dir = args.storage_dir or runtime_paths["storage_root"]
    args.stats_dir = args.stats_dir or runtime_paths["stats_root"]
    args.design_outputs_root = runtime_paths["design_outputs_root"]
    return args


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def first(*values):
    for x in values:
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            return x
    return None


def events_dir(root, scenario):
    return Path(root) / ("events" if scenario == "base" else f"events_{scenario}")


def _default_design_outputs_root(config_path=None):
    return load_runtime(config_path)[1]["design_outputs_root"]


def load_scenario_build(scenarios_dir, *, design_outputs_root=None):
    # Scenario build files are the bridge from SFINCS folders back to design_events.
    summary = read_json(Path(scenarios_dir) / "scenario_summary.json")
    if not summary.get("design_outputs_root"):
        summary["design_outputs_root"] = str(design_outputs_root or _default_design_outputs_root())
    summary.setdefault("design_scenario", "base")
    report_path = Path(scenarios_dir) / "scenario_build_report.csv"
    report = pd.read_csv(report_path).set_index("event_id").to_dict("index") if report_path.exists() else {}
    return summary, report


def load_design_events(summary, *, design_outputs_root=None):
    # Read design_events outputs directly so stats rows keep SLR/return-period context.
    root = Path(summary.get("design_outputs_root") or design_outputs_root or _default_design_outputs_root())
    scenario = summary.get("design_scenario", "base")
    sampled = root / "catalog" / "sampled_peaks.csv"
    if not sampled.exists():
        return {}, {"scenario_name": scenario, "slr_offset_m": None}

    df = pd.read_csv(sampled)
    if "event_id" not in df:
        df.insert(0, "event_id", [f"evt_{i + 1:04d}" for i in range(len(df))])
    df["event_id"] = df["event_id"].astype(str)

    member_summary = events_dir(root, scenario) / "surge_event_members_summary.csv"
    if member_summary.exists():
        df = df.merge(pd.read_csv(member_summary), on="event_id", how="left", suffixes=("", "_member"))

    attrs = {"scenario_name": scenario, "slr_offset_m": None}
    member_nc = events_dir(root, scenario) / "surge_event_members.nc"
    if member_nc.exists():
        with xr.open_dataset(member_nc) as ds:
            attrs["scenario_name"] = ds.attrs.get("scenario_name", scenario)
            attrs["slr_offset_m"] = ds.attrs.get("slr_offset_m")
    return df.set_index("event_id").to_dict("index"), attrs


def finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def metric(a, fn):
    values = finite(a)
    return float(fn(values)) if values.size else None


def area(mask, cell_area_m2):
    return None if cell_area_m2 <= 0 else float(np.count_nonzero(mask) * cell_area_m2 / 1_000_000.0)


def cell_area_m2(event_dir, inp):
    cell_area = float(inp.get("dx", "nan")) * float(inp.get("dy", "nan"))
    if np.isfinite(cell_area) and cell_area > 0:
        return cell_area

    qtrfile = inp.get("qtrfile")
    if not qtrfile:
        return 0.0
    qtr_path = Path(event_dir) / qtrfile
    if not qtr_path.exists():
        return 0.0

    with xr.open_dataset(qtr_path, decode_times=False) as ds:
        dx = float(ds.attrs.get("dx", "nan"))
        dy = float(ds.attrs.get("dy", "nan"))
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
            return 0.0
        levels = np.asarray(ds["level"].values) if "level" in ds else np.asarray([])
        if levels.size and np.nanmax(levels) > np.nanmin(levels):
            return 0.0
        return dx * dy


def to_ft(x):
    return None if x is None else float(x * m_to_ft)


def to_acres(x):
    return None if x is None else float(x * km2_to_acres)


def parse_time(ds, inp):
    if "time" not in ds:
        return [], None
    raw = np.asarray(ds["time"].values)
    if np.issubdtype(raw.dtype, np.datetime64):
        stamps = pd.DatetimeIndex(pd.to_datetime(raw))
    else:
        t0 = pd.to_datetime(inp.get("tstart", "20000101 000000"), format="%Y%m%d %H%M%S")
        stamps = pd.DatetimeIndex([t0 + pd.to_timedelta(float(x), unit="s") for x in raw])
    hours = np.asarray([(t - stamps[0]) / pd.Timedelta(hours=1) for t in stamps], float)
    dt = float(np.median(np.diff(hours))) if len(hours) > 1 else None
    return hours.tolist(), dt


def bzs_stats(path):
    if not path.exists() or path.stat().st_size == 0:
        return {"n_boundary_points": 0, "bzs_t0_mean_m": None, "bzs_t0_max_m": None, "bzs_peak_max_m": None}
    arr = np.loadtxt(path)
    arr = arr[None, :] if arr.ndim == 1 else arr
    values = arr[:, 1:]
    return {
        "n_boundary_points": int(values.shape[1]),
        "bzs_t0_mean_m": float(np.nanmean(values[0])),
        "bzs_t0_max_m": float(np.nanmax(values[0])),
        "bzs_peak_max_m": float(np.nanmax(values)),
    }


def solver_progress(event_id, scenario_dir, storage_dir):
    for path in [storage_dir / event_id / "sfincs_log.txt", scenario_dir / "sfincs_log.txt", storage_dir / event_id / "sfincs.log", scenario_dir / "sfincs.log"]:
        if path.exists():
            found = [int(x) for x in re.findall(r"(\d+)% complete", path.read_text(errors="ignore"))]
            return str(path), len(found), max(found) if found else None
    return None, 0, None


def map_path(event_id, scenario_dir, storage_dir):
    for path in [storage_dir / event_id / "sfincs_map.nc", scenario_dir / "sfincs_map.nc"]:
        if path.exists():
            return path
    raise FileNotFoundError(f"No sfincs_map.nc for {event_id}")


def peak_index(a):
    if not np.isfinite(a).any():
        return None, None, None
    flat = int(np.nanargmax(a))
    y, x = np.unravel_index(flat, a.shape)
    return float(a[y, x]), int(y), int(x)


def event_stats(event_dir, storage_dir, land_threshold_m, huthresh_m, impact_threshold_m, scenario_summary, scenario_rows, design_rows, design_attrs):
    event = event_dir.name
    inp = parse_sfincs_inp(event_dir / "sfincs.inp")
    manifest = read_json(event_dir / "forcing_manifest.json")
    scenario_row = scenario_rows.get(event, {})
    design_row = design_rows.get(event, {})
    log_path, n_progress, progress = solver_progress(event, event_dir, storage_dir)
    mpath = map_path(event, event_dir, storage_dir)

    with xr.open_dataset(mpath, decode_times=False) as ds:
        for name in ["zs", "zb", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{mpath} missing {name}")

        zs = np.asarray(ds["zs"].values, np.float32)
        zb = np.asarray(ds["zb"].values, np.float32)
        active = np.asarray(ds["msk"].values, float) > 0
        hours, dt = parse_time(ds, inp)

        depth = np.where(active[None, :, :], zs - zb[None, :, :], np.nan)
        wet = np.isfinite(depth) & (depth > huthresh_m)
        land = active & np.isfinite(zb) & (zb > land_threshold_m)
        ocean = active & ~land

        baseline = np.where(land, np.maximum(depth[0], 0.0), np.nan)
        baseline_wet = land & np.isfinite(baseline) & (baseline > impact_threshold_m)
        dry_t0 = land & ~baseline_wet
        incremental = np.where(land[None, :, :], np.maximum(depth - np.nan_to_num(baseline, nan=0.0)[None, :, :], 0.0), np.nan)
        impact = np.isfinite(incremental) & (incremental > impact_threshold_m)
        newly_wet = impact & dry_t0[None, :, :]

        cell_area = cell_area_m2(event_dir, inp)

        peak_map = np.full(land.shape, np.nan, float)
        impacted_cells = np.any(impact, axis=0)
        if np.any(impacted_cells):
            peak_map[impacted_cells] = np.nanmax(np.where(impact, incremental, np.nan)[:, impacted_cells], axis=0)
        peak_depth, peak_y, peak_x = peak_index(peak_map)
        impact_counts = impact.sum(axis=(1, 2))
        newly_counts = newly_wet.sum(axis=(1, 2))
        duration = np.where(np.isfinite(peak_map), impact.sum(axis=0) * dt, np.nan) if dt else np.full(peak_map.shape, np.nan)

        out = {
            "event_id": event,
            "scenario_dir": str(event_dir),
            "map_path": str(mpath),
            "log_path": log_path,
            "design_outputs_root": scenario_summary.get("design_outputs_root"),
            "design_scenario": first(manifest.get("design_scenario"), scenario_row.get("design_scenario"), design_attrs.get("scenario_name"), scenario_summary.get("design_scenario")),
            "design_slr_offset_m": first(manifest.get("design_slr_offset_m"), scenario_row.get("design_slr_offset_m"), design_attrs.get("slr_offset_m")),
            "source_event_index": first(manifest.get("source_event_index"), scenario_row.get("source_event_index")),
            "sample_rp_years": first(manifest.get("sample_rp_years"), scenario_row.get("sample_rp_years"), design_row.get("sample_rp_years")),
            "probability_weight": first(manifest.get("probability_weight"), scenario_row.get("probability_weight"), design_row.get("probability_weight")),
            "template_id": first(manifest.get("template_id"), scenario_row.get("template_id"), design_row.get("template_id")),
            "template_peak_time": first(manifest.get("template_peak_time"), scenario_row.get("template_peak_time"), design_row.get("template_peak_time")),
            "tail_morph_factor": first(manifest.get("tail_morph_factor"), scenario_row.get("tail_morph_factor"), design_row.get("tail_morph_factor")),
            "precip_mode": manifest.get("precip_mode"),
            "forcing_variable": first(manifest.get("forcing_variable"), scenario_row.get("forcing_variable"), scenario_summary.get("forcing_variable")),
            "driver_h_magnitude": first(manifest.get("drivers", {}).get("h_magnitude"), manifest.get("driver_h_magnitude"), scenario_row.get("driver_h_magnitude"), design_row.get("absolute_peak_m"), design_row.get("peak_m"), design_row.get("peak")),
            "driver_p_magnitude": first(manifest.get("drivers", {}).get("p_magnitude"), manifest.get("driver_p_magnitude"), scenario_row.get("driver_p_magnitude"), design_row.get("p")),
            "expected_zsini_m": first(manifest.get("expected", {}).get("zsini_m"), manifest.get("expected_zsini_m"), scenario_row.get("expected_zsini_m")),
            "expected_bzs_t0_mean_m": first(manifest.get("expected", {}).get("bzs_t0_mean_m"), manifest.get("expected_bzs_t0_mean_m"), scenario_row.get("expected_bzs_t0_mean_m")),
            "expected_bzs_peak_max_m": first(manifest.get("expected", {}).get("bzs_peak_max_m"), manifest.get("expected_bzs_peak_max_m"), scenario_row.get("expected_bzs_peak_max_m")),
            "zsini_m": float(inp["zsini"]) if "zsini" in inp else None,
            "available_time_steps": int(zs.shape[0]),
            "dt_hours": dt,
            "last_output_hour": hours[-1] if hours else None,
            "progress_steps_found": n_progress,
            "max_progress_percent": progress,
            "wet_dry_threshold_m": huthresh_m,
            "impact_threshold_m": impact_threshold_m,
            "land_mask_source": f"bed_elevation_gt_{land_threshold_m:.2f}m",
            "active_cells": int(active.sum()),
            "ocean_t0_cells": int(ocean.sum()),
            "land_t0_cells": int(land.sum()),
            "baseline_wet_land_t0_cells": int(baseline_wet.sum()),
            "dry_land_t0_cells": int(dry_t0.sum()),
            "cell_area_m2": cell_area or None,
            "raw_peak_domain_zs_m": metric(np.where(active[None, :, :], zs, np.nan), np.max),
            "peak_ocean_zs_m": metric(np.where(wet & ocean[None, :, :], zs, np.nan), np.max),
            "baseline_t0_flooded_area_km2": area(baseline_wet, cell_area),
            "dry_land_t0_area_km2": area(dry_t0, cell_area),
            "peak_incremental_land_depth_m": peak_depth,
            "mean_peak_incremental_land_depth_m": metric(peak_map, np.mean),
            "median_peak_incremental_land_depth_m": metric(peak_map, np.median),
            "p90_peak_incremental_land_depth_m": metric(peak_map, lambda x: np.percentile(x, 90)),
            "peak_land_grid_y": peak_y,
            "peak_land_grid_x": peak_x,
            "peak_incremental_flooded_area_km2": area(impact[int(np.argmax(impact_counts))] if impact_counts.size else np.zeros_like(land), cell_area),
            "anytime_incremental_flooded_area_km2": area(np.any(impact, axis=0), cell_area),
            "peak_newly_flooded_area_km2": area(newly_wet[int(np.argmax(newly_counts))] if newly_counts.size else np.zeros_like(land), cell_area),
            "anytime_newly_flooded_area_km2": area(np.any(newly_wet, axis=0), cell_area),
            "longest_incremental_flood_duration_h": metric(duration, np.max),
            "mean_incremental_flood_duration_h": metric(duration, np.mean),
            "median_incremental_flood_duration_h": metric(duration, np.median),
            "p90_incremental_flood_duration_h": metric(duration, lambda x: np.percentile(x, 90)),
            "cumulative_incremental_flooded_area_km2h": float(impact_counts.sum() * cell_area * dt / 1_000_000.0) if dt and cell_area else None,
        }

    out.update(bzs_stats(event_dir / "sfincs.bzs"))
    for key in [k for k in list(out) if k.endswith("_m")]:
        out[f"{key[:-2]}_ft"] = to_ft(out[key])
    for key in [k for k in list(out) if k.endswith("_km2")]:
        out[f"{key[:-4]}_acres"] = to_acres(out[key])
    for hours in [6, 12, 24]:
        out[f"area_incremental_flooded_ge_{hours:02d}h_km2"] = area(np.isfinite(duration) & (duration >= hours), cell_area)
    return out


_EVENT_STATS_CTX = {}


def _event_stats_init(storage_dir, thresholds, scenario_summary, scenario_rows, design_rows, design_attrs):
    _EVENT_STATS_CTX.update(
        storage_dir=storage_dir,
        thresholds=thresholds,
        scenario_summary=scenario_summary,
        scenario_rows=scenario_rows,
        design_rows=design_rows,
        design_attrs=design_attrs,
    )


def _event_stats_worker(event_dir):
    c = _EVENT_STATS_CTX
    land_threshold_m, huthresh_m, impact_threshold_m = c["thresholds"]
    return event_stats(
        event_dir, c["storage_dir"], land_threshold_m, huthresh_m, impact_threshold_m,
        c["scenario_summary"], c["scenario_rows"], c["design_rows"], c["design_attrs"],
    )


def event_stats_table(
    completed_events,
    storage_dir,
    *,
    land_threshold_m,
    huthresh_m,
    impact_threshold_m,
    scenario_summary,
    scenario_rows,
    design_rows,
    design_attrs,
    workers=None,
):
    """Build the per-event stats table, reading the (heavy 3-D) SFINCS maps in parallel.

    ``event_stats`` loads each event's full ``zs`` time series, so for hundreds of events the
    serial loop is I/O bound. This fans the per-event reads across processes (default: up to
    16 / ``os.cpu_count()``). The shared per-scenario metadata is sent once per worker via the
    pool initializer, so only the event Path is pickled per task. Set ``workers=1`` to force
    the serial path (e.g. for debugging). Returns a DataFrame (one row per event).
    """
    events = list(completed_events)
    if not events:
        return pd.DataFrame()
    if workers is None:
        workers = min(os.cpu_count() or 1, 16)
    thresholds = (land_threshold_m, huthresh_m, impact_threshold_m)
    if workers <= 1 or len(events) == 1:
        _event_stats_init(storage_dir, thresholds, scenario_summary, scenario_rows, design_rows, design_attrs)
        return pd.DataFrame([_event_stats_worker(d) for d in events])
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_event_stats_init,
        initargs=(storage_dir, thresholds, scenario_summary, scenario_rows, design_rows, design_attrs),
    ) as executor:
        rows = list(executor.map(_event_stats_worker, events, chunksize=4))
    return pd.DataFrame(rows)


def series_summary(df, col):
    s = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return {"count": int(len(s)), "mean": float(s.mean()) if len(s) else None, "median": float(s.median()) if len(s) else None, "p90": float(s.quantile(0.9)) if len(s) else None, "max": float(s.max()) if len(s) else None}


def top_events(df, col, n=10):
    if col not in df:
        return []
    return df[["event_id", col]].assign(**{col: pd.to_numeric(df[col], errors="coerce")}).dropna().sort_values(col, ascending=False).head(n).to_dict("records")


def write_plots(df, plots_dir):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/surge-mpl-config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for col in ["peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2", "longest_incremental_flood_duration_h"]:
        s = pd.to_numeric(df.get(col), errors="coerce").dropna()
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(s, bins=min(40, max(10, int(np.sqrt(len(s))))), color="#2878b5", edgecolor="white")
        ax.set(title=col, xlabel=col, ylabel="event count")
        fig.tight_layout()
        out = plots_dir / f"hist_{col}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out.name)
    return written


def write_report(df, aggregate, path):
    lines = [
        "# SFINCS flood statistics",
        "",
        f"- Event count: {len(df)}",
        f"- CSV: `{aggregate['scenario_stats_csv']}`",
        f"- Design outputs: `{aggregate.get('design_outputs_root')}`",
        f"- Design scenarios: {', '.join(aggregate.get('design_scenarios', [])) or 'unknown'}",
        f"- SLR offsets (m): {aggregate.get('design_slr_offsets_m', [])}",
        "",
    ]
    for col, label in [
        ("peak_incremental_land_depth_m", "Peak incremental inland depth"),
        ("peak_incremental_flooded_area_km2", "Peak incremental flood extent"),
        ("longest_incremental_flood_duration_h", "Longest incremental duration"),
    ]:
        s = aggregate["metric_summaries"][col]
        lines.append(f"- {label}: mean={s['mean']}, max={s['max']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    inventory = completed_event_inventory(args.scenarios_dir, args.storage_dir, args.event_ids, args.limit)
    selected = inventory["completed_events"]
    if not selected:
        print("No completed event directories matched.")
        return 1

    scenario_summary, scenario_rows = load_scenario_build(
        args.scenarios_dir,
        design_outputs_root=args.design_outputs_root,
    )
    design_rows, design_attrs = load_design_events(scenario_summary)

    print(f"Processing {len(selected)} events with {args.workers} workers ...")
    print(f"Event source: {inventory['event_source_root']}")
    print(f"Design scenario: {first(design_attrs.get('scenario_name'), scenario_summary.get('design_scenario'))}")
    print(f"Design SLR offset: {design_attrs.get('slr_offset_m')} m")
    rows, failures, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                event_stats,
                d,
                args.storage_dir,
                args.land_threshold_m,
                args.huthresh_m,
                args.impact_threshold_m,
                scenario_summary,
                scenario_rows,
                design_rows,
                design_attrs,
            ): d
            for d in selected
        }
        for i, future in enumerate(as_completed(futures), 1):
            d = futures[future]
            try:
                row = future.result()
                rows.append(row)
                out = args.stats_dir / d.name
                out.mkdir(parents=True, exist_ok=True)
                (out / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(f"[{i:4d}/{len(selected)}] {d.name} elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
            except Exception as exc:
                failures.append(f"{d.name}: {exc}")
                print(f"[{i:4d}/{len(selected)}] FAILED {d.name}: {exc}", file=sys.stderr, flush=True)

    if not rows:
        return 1
    df = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    csv_path = args.stats_dir / "scenario_stats.csv"
    df.to_csv(csv_path, index=False)

    metric_cols = ["zsini_m", "baseline_t0_flooded_area_km2", "peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2", "anytime_incremental_flooded_area_km2", "longest_incremental_flood_duration_h", "mean_incremental_flood_duration_h", "area_incremental_flooded_ge_24h_km2"]
    by_design = {}
    if "design_scenario" in df:
        for name, group in df.groupby("design_scenario", dropna=False):
            by_design[str(name)] = {col: series_summary(group, col) for col in metric_cols}
    aggregate = {
        "event_count": len(df),
        "event_ids": df["event_id"].tolist(),
        "scenario_stats_csv": str(csv_path),
        "design_outputs_root": scenario_summary.get("design_outputs_root"),
        "design_scenarios": sorted(str(x) for x in df.get("design_scenario", pd.Series(dtype=object)).dropna().unique()),
        "design_slr_offsets_m": sorted(float(x) for x in pd.to_numeric(df.get("design_slr_offset_m", pd.Series(dtype=float)), errors="coerce").dropna().unique()),
        "metric_summaries": {col: series_summary(df, col) for col in metric_cols},
        "metric_summaries_by_design_scenario": by_design,
        "top_events": {
            "peak_incremental_land_depth_m": top_events(df, "peak_incremental_land_depth_m"),
            "peak_incremental_flooded_area_km2": top_events(df, "peak_incremental_flooded_area_km2"),
            "area_incremental_flooded_ge_24h_km2": top_events(df, "area_incremental_flooded_ge_24h_km2"),
        },
        "failures": failures,
    }
    if not args.no_plots:
        plots_dir = args.plots_dir or args.stats_dir / "plots"
        aggregate["plots_dir"] = str(plots_dir)
        aggregate["plot_files"] = write_plots(df, plots_dir)
    (args.stats_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_report(df, aggregate, args.stats_dir / "report.md")
    print(f"Done. CSV -> {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
