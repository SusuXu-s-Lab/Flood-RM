"""
sfincs_runs.diagnostics
~~~~~~~~~~~~~~~~~~~~~~~
Diagnostic plots for SFINCS single-use-case notebook 04.

All matplotlib / animation / basemap code lives here so the notebook
cells can stay to one-liners (import + call + optional display).

Public API
----------
plot_forcing_qa_standard      Pre-run 6-panel QA for surge + rain builds
plot_forcing_qa_waves         Pre-run 6-panel QA for surge + rain + SnapWave builds
plot_flood_animation          Static peak-depth summary + flood/ocean mp4 (coastal)
plot_inland_flood_animation   Static peak-depth summary + flood/discharge mp4 (inland)
plot_postrun_diagnostics      3-panel post-run check (soil frac / his.nc zs / precip)
plot_precip_animation         Spatiotemporal AORC precip mp4
plot_runup_overtopping        Runup gauge map + per-gauge crest overtopping screen
"""
from __future__ import annotations

import json
from pathlib import Path

import contextily as ctx
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display
from shapely.geometry import LineString


# ─── private helpers ──────────────────────────────────────────────────────────


def _axis_message(ax, message: str) -> None:
    """Write a centred placeholder message and hide ticks."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _spatial_time_stats(da) -> pd.DataFrame:
    """Collapse all non-time dims to (mean, max) time series."""
    dims = tuple(d for d in da.dims if d != "time")
    if not dims:
        return pd.DataFrame({"mean": da.to_pandas(), "max": da.to_pandas()})
    return pd.DataFrame(
        {
            "mean": da.mean(dims, skipna=True).to_pandas(),
            "max":  da.max(dims, skipna=True).to_pandas(),
        }
    )


def _snapwave_stats(path: Path, run_start: pd.Timestamp) -> pd.DataFrame:
    """Read a whitespace-delimited SnapWave boundary file → (mean, min, max) Series."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    times  = pd.DatetimeIndex(
        run_start + pd.to_timedelta(frame.iloc[:, 0].astype(float), unit="s")
    )
    values = frame.iloc[:, 1:].astype(float)
    return pd.DataFrame(
        {
            "mean": values.mean(axis=1).to_numpy(),
            "min":  values.min(axis=1).to_numpy(),
            "max":  values.max(axis=1).to_numpy(),
        },
        index=times,
    )


def _day_boundary_ticks(times: pd.DatetimeIndex) -> list[int]:
    """
    Return bar-chart integer indices at midnight boundaries.
    Falls back to ~8 evenly spaced indices for sub-daily events.
    """
    idx = [i for i, t in enumerate(times) if t.hour == 0 and t.minute == 0]
    if len(idx) < 2:
        step = max(1, len(times) // 8)
        idx  = list(range(0, len(times), step))
    return idx


def _apply_day_ticks(ax, times: pd.DatetimeIndex, color: str = "black") -> None:
    """Apply midnight-only x-ticks to a bar chart whose x is integer indices."""
    day_idx = _day_boundary_ticks(times)
    ax.set_xticks(day_idx)
    ax.set_xticklabels(
        [times[i].strftime("%b %-d") for i in day_idx],
        rotation=30, ha="right", fontsize=7, color=color,
    )
    ax.tick_params(axis="x", length=4)


def _parse_sfincs_datetime(value) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    try:
        return pd.Timestamp(text)
    except ValueError:
        return None


def _read_sfincs_inp_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    target = key.lower()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in raw_line:
            continue
        raw_key, value = raw_line.split("=", 1)
        if raw_key.strip().lower() == target:
            return value.strip()
    return None


def _resolve_run_start(run_root: Path, fallback=None) -> pd.Timestamp:
    manifest_path = Path(run_root) / "forcing_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_start = _parse_sfincs_datetime(manifest.get("run_start"))
        if run_start is not None:
            return run_start

    inp_start = _parse_sfincs_datetime(
        _read_sfincs_inp_value(Path(run_root) / "sfincs.inp", "tstart")
    )
    if inp_start is not None:
        return inp_start

    fallback_start = _parse_sfincs_datetime(fallback)
    if fallback_start is not None:
        return fallback_start
    raise ValueError(f"Cannot resolve SFINCS run start from {run_root}")


def _format_datetime_axis(ax, times: pd.DatetimeIndex) -> None:
    if len(times) == 0:
        return
    span_days = max((times.max() - times.min()) / pd.Timedelta(days=1), 0.0)
    if span_days >= 2:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _read_json_if_exists(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(run_root: Path, value) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute() or "://" in str(value):
        return path
    for base in (run_root, *run_root.parents):
        if (base / "config.yaml").exists():
            return base / path
        if base.name == "data" and base.parent.exists():
            return base.parent / path
    return run_root / path


def _dataset_time_index(data_array) -> pd.DatetimeIndex | None:
    if "time" not in data_array.coords:
        return None
    try:
        return pd.DatetimeIndex(pd.to_datetime(data_array["time"].values))
    except Exception:
        return None


def _plot_missing_panel(ax, title: str, message: str) -> None:
    ax.set_title(title)
    _axis_message(ax, message)


def plot_inland_coupled_forcing_qa(
    *,
    forcing_manifest,
    out_dir=None,
    event_id=None,
    event_label=None,
):
    """Pre-run QA for an inland Wflow-SFINCS staged scenario.

    The inland coupled path is fluvial/pluvial: Wflow produces a discharge NetCDF
    consumed by SFINCS, and optional direct rainfall is described in the forcing
    manifest. The figure intentionally avoids coastal water-level panels used by
    Marshfield and instead audits discharge handoff, event-driver metadata, and
    staged static hydrology files.
    """
    forcing_manifest = Path(forcing_manifest)
    manifest = _read_json_if_exists(forcing_manifest)
    run_root = forcing_manifest.parent
    event_id = str(event_id or manifest.get("event_id") or run_root.name)
    event_label = str(event_label or event_id)
    out_dir = Path(out_dir) if out_dir is not None else run_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    discharge_path = _resolve_manifest_path(run_root, manifest.get("wflow_discharge_forcing"))

    summary = {
        "event": event_label,
        "run_root": str(run_root),
        "forcing_mode": manifest.get("forcing_mode"),
        "direct_rainfall_enabled": manifest.get("direct_rainfall_enabled"),
        "wflow_discharge_forcing": None if discharge_path is None else str(discharge_path),
        "rainfall_member_id": manifest.get("rainfall_member_id"),
        "soil_moisture_member_id": manifest.get("soil_moisture_member_id"),
    }
    display(pd.Series(summary, name="inland_forcing_summary"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    if discharge_path is not None and discharge_path.exists():
        with xr.open_dataset(discharge_path) as ds:
            names = list(ds.data_vars)
            variable = "discharge" if "discharge" in ds else names[0]
            data = ds[variable].load()
        if "time" in data.dims:
            dims = tuple(dim for dim in data.dims if dim != "time")
            frame = pd.DataFrame(
                {
                    "mean": data.mean(dims, skipna=True).to_pandas() if dims else data.to_pandas(),
                    "max": data.max(dims, skipna=True).to_pandas() if dims else data.to_pandas(),
                }
            )
            frame.plot(ax=axes[0], color=["#3182bd", "#08519c"], linewidth=1.8)
            _format_datetime_axis(axes[0], pd.DatetimeIndex(frame.index))
            axes[0].set_ylabel(f"{variable}")
            axes[0].set_title("Wflow discharge handoff to SFINCS")
            axes[0].grid(True, alpha=0.25)
        else:
            values = np.asarray(data.values, dtype=float).ravel()
            axes[0].hist(values[np.isfinite(values)], bins=40, color="#3182bd", alpha=0.75)
            axes[0].set_title("Wflow discharge handoff distribution")
            axes[0].set_xlabel(variable)
    else:
        _plot_missing_panel(
            axes[0],
            "Wflow discharge handoff to SFINCS",
            "No sfincs_discharge.nc yet\nrun Wflow replay first",
        )

    if discharge_path is not None and discharge_path.exists():
        with xr.open_dataset(discharge_path) as ds:
            variable = "discharge" if "discharge" in ds else next(iter(ds.data_vars))
            data = ds[variable]
            snapshot = data.max("time", skipna=True) if "time" in data.dims else data
            if {"x", "y"} & set(snapshot.coords):
                snapshot.plot(ax=axes[1], cmap="Blues", cbar_kwargs=dict(shrink=0.8, label=variable))
                axes[1].set_aspect("equal", adjustable="datalim")
                axes[1].set_title("Peak Wflow discharge forcing")
            else:
                axes[1].axis("off")
                axes[1].table(
                    cellText=[[name, str(value)] for name, value in summary.items()],
                    colLabels=["field", "value"],
                    loc="center",
                )
                axes[1].set_title("Forcing manifest summary")
    else:
        axes[1].axis("off")
        axes[1].table(
            cellText=[[name, "" if value is None else str(value)] for name, value in summary.items()],
            colLabels=["field", "value"],
            loc="center",
        )
        axes[1].set_title("Forcing manifest summary")

    smax = np.fromfile(run_root / "sfincs.smax", dtype="<f4") if (run_root / "sfincs.smax").exists() else np.array([])
    seff = np.fromfile(run_root / "sfincs.seff", dtype="<f4") if (run_root / "sfincs.seff").exists() else np.array([])
    if smax.size and seff.size:
        valid = np.isfinite(smax) & np.isfinite(seff) & (smax > 0)
        frac = seff[valid] / smax[valid] if valid.any() else np.array([])
        axes[2].hist(frac, bins=40, color="#6baed6", edgecolor="white", linewidth=0.4)
        if frac.size:
            axes[2].axvline(float(np.median(frac)), color="#08519c", linestyle="--", label=f"median={float(np.median(frac)):.2f}")
            axes[2].legend(fontsize=8)
        axes[2].set_xlabel("seff / smax")
        axes[2].set_title("Initial SFINCS soil saturation")
    else:
        _plot_missing_panel(axes[2], "Initial SFINCS soil saturation", "No smax/seff files staged")

    ks = np.fromfile(run_root / "sfincs.ks", dtype="<f4") if (run_root / "sfincs.ks").exists() else np.array([])
    if ks.size:
        finite_ks = ks[np.isfinite(ks) & (ks > 0)]
        axes[3].hist(finite_ks, bins=50, color="#8c6d31", alpha=0.75)
        if finite_ks.size:
            p50, p95 = np.percentile(finite_ks, [50, 95])
            axes[3].set_title(f"Ksat (p50={p50:.1f}, p95={p95:.1f})")
        else:
            axes[3].set_title("Infiltration hydraulic conductivity")
        axes[3].set_xlabel("mm/hr")
    else:
        _plot_missing_panel(axes[3], "Infiltration hydraulic conductivity", "No sfincs.ks file staged")

    for ax in axes:
        if ax.has_data():
            ax.grid(True, alpha=0.25)
    out_path = out_dir / f"{event_id}_inland_forcing_qa.png"
    fig.savefig(out_path, dpi=160)
    plt.show()
    print("Saved inland forcing QA plot:", out_path)
    return out_path


def plot_inland_coupled_postrun_diagnostics(
    *,
    run_root,
    event_label=None,
    out_dir=None,
):
    """Post-run inland coupled diagnostics for SFINCS outputs when present."""
    run_root = Path(run_root)
    event_label = str(event_label or run_root.name)
    out_dir = Path(out_dir) if out_dir is not None else run_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    map_path = run_root / "sfincs_map.nc"
    if map_path.exists():
        with xr.open_dataset(map_path, decode_times=False) as ds:
            if {"zs", "zb"}.issubset(ds.data_vars):
                depth = (ds["zs"] - ds["zb"]).where(ds["zs"] > ds["zb"])
                peak = depth.max("time", skipna=True) if "time" in depth.dims else depth
                peak.plot(ax=axes[0], cmap="Blues", cbar_kwargs=dict(shrink=0.8, label="depth [m]"))
                axes[0].set_title("Peak SFINCS flood depth")
                axes[0].set_aspect("equal", adjustable="datalim")
            else:
                _plot_missing_panel(axes[0], "Peak SFINCS flood depth", "sfincs_map.nc has no zs/zb")
    else:
        _plot_missing_panel(axes[0], "Peak SFINCS flood depth", "sfincs_map.nc not found")

    his_path = run_root / "sfincs_his.nc"
    if his_path.exists():
        run_start = _resolve_run_start(run_root)
        with xr.open_dataset(his_path, decode_times=False) as his:
            times = pd.DatetimeIndex(run_start + pd.to_timedelta(his["time"].values.astype(float), unit="s"))
            variable = "point_zs" if "point_zs" in his.data_vars else next(iter(his.data_vars), None)
            if variable is not None:
                values = his[variable].values
                series = np.asarray(values[:, 0] if values.ndim > 1 else values, dtype=float)
                axes[1].plot(times, np.where(np.isfinite(series), series, np.nan), color="#2171b5", linewidth=1.8)
                axes[1].set_title(f"{variable} hydrograph")
                axes[1].set_ylabel("water level [m]")
                _format_datetime_axis(axes[1], times)
            else:
                _plot_missing_panel(axes[1], "SFINCS hydrograph", "sfincs_his.nc has no variables")
    else:
        _plot_missing_panel(axes[1], "SFINCS hydrograph", "sfincs_his.nc not found")

    manifest = _read_json_if_exists(run_root / "forcing_manifest.json")
    axes[2].axis("off")
    rows = [
        ["event_id", manifest.get("event_id", run_root.name)],
        ["forcing_mode", manifest.get("forcing_mode", "")],
        ["wflow_source_variable", manifest.get("wflow_source_variable", "")],
        ["direct_rainfall_enabled", manifest.get("direct_rainfall_enabled", "")],
        ["wflow_discharge_forcing", manifest.get("wflow_discharge_forcing", "")],
    ]
    axes[2].table(cellText=rows, colLabels=["field", "value"], loc="center")
    axes[2].set_title("Coupling manifest")

    for ax in axes[:2]:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{event_label} inland coupled post-run diagnostics", fontsize=11, y=1.02)
    out_path = out_dir / f"{run_root.name}_inland_postrun_diagnostics.png"
    fig.savefig(out_path, dpi=160)
    plt.show()
    print("Saved inland post-run diagnostics:", out_path)
    return out_path


# ─── plot_forcing_qa_standard ─────────────────────────────────────────────────


def plot_forcing_qa_standard(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    h_series: pd.Series,
    staged_manifest: dict,
    precip_manifest: dict | None,
    include_precip: bool,
    hydrology_inputs: dict | None = None,
) -> Path:
    """
    Pre-run forcing QA panel for surge + rainfall example runs.

    Panels
    ------
    [0] Boundary water level
    [1] AORC precipitation intervals (mean + max domain)
    [2] SFINCS netampr precipitation
    [3] Soil storage histogram (smax / seff)
    [4] Infiltration hydraulic conductivity (Ksat)
    [5] Manifest summary text
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_precip = None
    netampr_precip  = None
    if include_precip and precip_manifest:
        with xr.open_dataset(precip_manifest["prepared_precip"]) as ds:
            var_name = "precip" if "precip" in ds.data_vars else next(iter(ds.data_vars))
            prepared_precip = _spatial_time_stats(ds[var_name].load())
        netampr_path = run_root / precip_manifest["netamprfile"]
        if netampr_path.exists():
            with xr.open_dataset(netampr_path) as ds:
                var_name = "Precipitation" if "Precipitation" in ds.data_vars else next(iter(ds.data_vars))
                netampr_precip = _spatial_time_stats(ds[var_name].load())

    summary = {
        "event":                  event_label,
        "run_root":               str(run_root),
        "modeled_hours":          staged_manifest["run_duration_hours"],
        "water_level_samples":    len(h_series),
        "rainfall_source":        None if not precip_manifest else Path(precip_manifest["rainfall_source_nc"]).name,
        "soil_moisture_fraction": None if not precip_manifest else precip_manifest.get("initial_soil_moisture_fraction"),
    }
    display(pd.Series(summary, name="forcing_summary"))

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()

    # [0] Boundary water level
    h_series.plot(ax=axes[0], color="#0b6e99", linewidth=2)
    axes[0].set_title("Boundary water level")
    axes[0].set_ylabel("m MSL")
    axes[0].grid(True, alpha=0.25)

    # [1] AORC precipitation
    if prepared_precip is not None:
        prepared_precip.plot(ax=axes[1], color=["#2ca25f", "#006d2c"], linewidth=1.8)
        axes[1].set_title("AORC precipitation intervals")
        axes[1].set_ylabel("mm per interval")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].set_title("AORC precipitation intervals")
        _axis_message(axes[1], "No precipitation staged")

    # [2] netampr precipitation
    if netampr_precip is not None:
        netampr_precip.plot(ax=axes[2], color=["#756bb1", "#54278f"], linewidth=1.8)
        axes[2].set_title("SFINCS netampr precipitation")
        axes[2].set_ylabel("model forcing units")
        axes[2].grid(True, alpha=0.25)
    else:
        axes[2].set_title("SFINCS netampr precipitation")
        _axis_message(axes[2], "No netampr file staged")

    # [3] Soil storage histogram
    smax = np.fromfile(run_root / "sfincs.smax", dtype="<f4") if (run_root / "sfincs.smax").exists() else np.array([])
    seff = np.fromfile(run_root / "sfincs.seff", dtype="<f4") if (run_root / "sfincs.seff").exists() else np.array([])
    if smax.size and seff.size:
        valid   = np.isfinite(smax) & np.isfinite(seff) & (smax > 0)
        wetness = float(np.nanmedian(seff[valid] / smax[valid])) if valid.any() else np.nan
        axes[3].hist(smax[valid], bins=40, alpha=0.55, label="smax")
        axes[3].hist(seff[valid], bins=40, alpha=0.55, label="seff")
        axes[3].set_title(f"Soil storage and initial wetness (median seff/smax={wetness:.2f})")
        axes[3].set_xlabel("m")
        axes[3].legend()
    else:
        axes[3].set_title("Soil storage and initial wetness")
        _axis_message(axes[3], "No smax/seff files staged")

    # [4] Ksat
    ks = np.fromfile(run_root / "sfincs.ks", dtype="<f4") if (run_root / "sfincs.ks").exists() else np.array([])
    if ks.size:
        finite_ks = ks[np.isfinite(ks) & (ks > 0)]
        axes[4].hist(finite_ks, bins=50, color="#8c6d31", alpha=0.75)
        if finite_ks.size:
            p50, p95      = np.percentile(finite_ks, [50, 95])
            cap_fraction  = float(np.mean(np.isclose(finite_ks, float(np.nanmax(finite_ks)))))
            axes[4].set_title(f"Ksat (p50={p50:.1f}, p95={p95:.1f}, cap={cap_fraction:.1%})")
        else:
            axes[4].set_title("Infiltration hydraulic conductivity")
        axes[4].set_xlabel("mm/hr")
    else:
        axes[4].set_title("Infiltration hydraulic conductivity")
        _axis_message(axes[4], "No sfincs.ks file staged")

    # [5] Manifest text
    axes[5].axis("off")
    if precip_manifest:
        manifest_items = {
            "prepared_precip":            Path(precip_manifest["prepared_precip"]).name,
            "netamprfile":                precip_manifest.get("netamprfile"),
            "sefffile":                   precip_manifest.get("sefffile"),
            "rainfall_window_alignment":  precip_manifest.get("rainfall_window_alignment"),
        }
        axes[5].text(0.0, 1.0, pd.Series(manifest_items).to_string(),
                     ha="left", va="top", family="monospace")

    out_path = out_dir / f"{event_id}_forcing_qa.png"
    fig.savefig(out_path, dpi=160)
    plt.show()
    print("Saved forcing QA plot:", out_path)
    return out_path


# ─── plot_forcing_qa_waves ────────────────────────────────────────────────────


def plot_forcing_qa_waves(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    h_series: pd.Series,
    run_start: pd.Timestamp,
    staged_manifest: dict,
    precip_manifest: dict | None,
    include_precip: bool,
) -> Path:
    """
    Pre-run forcing QA panel for surge + rain + SnapWave example runs.

    Panels
    ------
    [0] Boundary water level with event-reference axvline
    [1] AORC precipitation intervals
    [2] Initial soil storage (bar chart, wetness fraction)
    [3] Ksat histogram
    [4] SnapWave Hs + Tp at boundary
    [5] SnapWave wave direction + directional spread
    """
    from sfincs_runs.snapwave_setup import unwrap_direction_degrees

    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_precip = None
    if include_precip and precip_manifest:
        with xr.open_dataset(precip_manifest["prepared_precip"]) as ds:
            var_name = "precip" if "precip" in ds.data_vars else next(iter(ds.data_vars))
            prepared_precip = _spatial_time_stats(ds[var_name].load())

    snapwave = {
        "Hs (m)":       _snapwave_stats(run_root / "snapwave.bhs", run_start),
        "Tp (s)":       _snapwave_stats(run_root / "snapwave.btp", run_start),
        "Dir (deg)":    _snapwave_stats(run_root / "snapwave.bwd", run_start),
        "Spread (deg)": _snapwave_stats(run_root / "snapwave.bds", run_start),
    }

    event_reference_time = pd.Timestamp(staged_manifest.get("event_reference_time", run_start))
    coastal_windows = [
        w for w in staged_manifest.get("driver_windows", []) if w.get("driver") == "coastal"
    ]
    coastal_start = (
        event_reference_time + pd.Timedelta(hours=float(coastal_windows[0]["start_offset_hours"]))
        if coastal_windows
        else run_start
    )
    boundary_times  = pd.date_range(coastal_start, periods=len(h_series), freq="h")
    boundary_series = pd.Series(h_series.to_numpy(dtype=float), index=boundary_times, name="water_level_msl")

    summary = {
        "event":                  event_label,
        "run_root":               str(run_root),
        "timing_policy":          staged_manifest.get("timing_policy"),
        "modeled_hours":          staged_manifest["run_duration_hours"],
        "water_level_samples":    len(h_series),
        "rainfall_source":        None if not precip_manifest else Path(precip_manifest["rainfall_source_nc"]).name,
        "soil_moisture_fraction": None if not precip_manifest else precip_manifest.get("initial_soil_moisture_fraction"),
    }
    display(pd.Series(summary, name="forcing_summary"))

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    axes = axes.ravel()

    # [0] Boundary water level + event reference
    boundary_series.plot(ax=axes[0], color="#0b6e99", linewidth=2)
    axes[0].axvline(event_reference_time, color="#d95f02", linestyle="--",
                    linewidth=1.2, label="event peak/reference")
    axes[0].set_title("Boundary water level (catalog event window)")
    axes[0].set_ylabel("m MSL")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    # [1] AORC precipitation
    if prepared_precip is not None:
        prepared_precip.plot(ax=axes[1], color=["#2ca25f", "#006d2c"], linewidth=1.8)
        axes[1].set_title("AORC precipitation intervals")
        axes[1].set_ylabel("mm per interval")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].set_title("AORC precipitation intervals")
        _axis_message(axes[1], "No precipitation staged")

    # [2] Initial soil storage bar chart
    smax = np.fromfile(run_root / "sfincs.smax", dtype="<f4") if (run_root / "sfincs.smax").exists() else np.array([])
    seff = np.fromfile(run_root / "sfincs.seff", dtype="<f4") if (run_root / "sfincs.seff").exists() else np.array([])
    if smax.size and seff.size:
        valid   = np.isfinite(smax) & np.isfinite(seff) & (smax > 0)
        wvals   = seff[valid] / smax[valid] if valid.any() else np.array([])
        wetness = float(np.nanmedian(wvals)) if wvals.size else np.nan
        remaining = max(0.0, 1.0 - wetness) if np.isfinite(wetness) else np.nan
        smax_p50  = float(np.nanmedian(smax[valid])) if valid.any() else np.nan
        seff_p50  = float(np.nanmedian(seff[valid])) if valid.any() else np.nan
        axes[2].barh(["storage fraction"], [wetness],   color="#3182bd", label="initial wetness")
        axes[2].barh(["storage fraction"], [remaining], left=[wetness], color="#c7e9c0", label="remaining capacity")
        axes[2].set_xlim(0, 1)
        axes[2].set_xlabel("fraction of SCS storage")
        axes[2].set_title(
            f"Initial soil storage: {wetness:.2f} full "
            f"(median smax={smax_p50:.2f} m, seff={seff_p50:.2f} m)"
        )
        axes[2].legend(loc="lower right", fontsize=8)
        axes[2].grid(True, axis="x", alpha=0.25)
    else:
        axes[2].set_title("Initial soil storage")
        _axis_message(axes[2], "No smax/seff files staged")

    # [3] Ksat histogram
    ks = np.fromfile(run_root / "sfincs.ks", dtype="<f4") if (run_root / "sfincs.ks").exists() else np.array([])
    if ks.size:
        finite_ks = ks[np.isfinite(ks) & (ks > 0)]
        axes[3].hist(finite_ks, bins=50, color="#8c6d31", alpha=0.75)
        if finite_ks.size:
            p50, p95     = np.percentile(finite_ks, [50, 95])
            cap_fraction = float(np.mean(np.isclose(finite_ks, float(np.nanmax(finite_ks)))))
            axes[3].set_title(f"Ksat (p50={p50:.1f}, p95={p95:.1f}, cap={cap_fraction:.1%})")
        else:
            axes[3].set_title("Infiltration hydraulic conductivity")
        axes[3].set_xlabel("mm/hr")
    else:
        axes[3].set_title("Infiltration hydraulic conductivity")
        _axis_message(axes[3], "No sfincs.ks file staged")

    # [4] SnapWave Hs + Tp
    for label, color in [("Hs (m)", "#0571b0"), ("Tp (s)", "#ca0020")]:
        stats = snapwave[label]
        axes[4].plot(stats.index, stats["mean"], label=label, color=color, linewidth=1.8)
        axes[4].fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.15)
    axes[4].set_title("SnapWave height and peak period")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend()

    # [5] SnapWave direction + spread
    direction = snapwave["Dir (deg)"].copy()
    direction["mean_unwrapped"] = unwrap_direction_degrees(direction["mean"])
    axes[5].plot(direction.index, direction["mean_unwrapped"],
                 label="Dir (deg, unwrapped)", color="#7b3294", linewidth=1.8)
    spread = snapwave["Spread (deg)"]
    axes[5].plot(spread.index, spread["mean"], label="Spread (deg)", color="#008837", linewidth=1.8)
    axes[5].fill_between(spread.index, spread["min"], spread["max"], color="#008837", alpha=0.15)
    axes[5].set_title("SnapWave direction and spread")
    axes[5].set_ylabel("degrees")
    axes[5].grid(True, alpha=0.25)
    axes[5].legend()

    out_path = out_dir / f"{event_id}_forcing_qa.png"
    fig.savefig(out_path, dpi=160)
    plt.show()
    print("Saved forcing QA plot:", out_path)
    return out_path


# ─── plot_flood_animation ─────────────────────────────────────────────────────


def plot_flood_animation(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    h_series: pd.Series,
    t_start: pd.Timestamp,
    zsini: float,
    huthresh: float = 0.02,
    display_depth_threshold: float = 0.05,
    bmap_zoom: int = 12,
) -> Path:
    """
    Static peak-depth summary (shown inline) + flood/ocean mp4 animation (saved).

    The static panel shows the boundary water level and peak flood depth map.
    The animation frames show land flood depth and ocean surface anomaly over time.

    Parameters
    ----------
    display_depth_threshold : float
        Minimum flood depth (m) rendered in the animation.  Filters numerical
        noise without changing SFINCS's internal wet/dry threshold (HUTHRESH).
    huthresh : float
        SFINCS HUTHRESH value — used only in the diagnostic print statement.

    Returns
    -------
    Path to the saved mp4 file.
    """
    from sfincs_runs.scenarios.io import parse_sfincs_inp

    out_dir.mkdir(parents=True, exist_ok=True)

    inp   = parse_sfincs_inp(run_root / "sfincs.inp")
    epsg  = int(inp.get("epsg", 26919))
    tstart = pd.to_datetime(str(inp["tstart"]), format="%Y%m%d %H%M%S")

    with xr.open_dataset(run_root / "sfincs_map.nc", decode_times=False) as ds:
        x      = ds["x"].values.astype(float)
        y      = ds["y"].values.astype(float)
        time_s = ds["time"].values.astype(float)
        zb     = ds["zb"].values.astype(float)
        msk    = ds["msk"].values.astype(float)
        zs     = ds["zs"].values.astype(float)
        has_uv = "u" in ds and "v" in ds
        u      = ds["u"].values.astype(float) if has_uv else None
        v      = ds["v"].values.astype(float) if has_uv else None

    timestamps = [tstart + pd.Timedelta(seconds=float(t)) for t in time_s]
    n_steps    = len(timestamps)
    active     = np.isfinite(msk) & (msk > 0)
    land_mask  = active & np.isfinite(zb) & (zb >= 0.0)
    ocean_mask = active & ~land_mask

    depth_all      = zs - zb[None, :, :]
    land_depth     = np.where(land_mask[None, :, :] & (depth_all > display_depth_threshold), depth_all, np.nan)
    peak_land_depth = np.nanmax(land_depth, axis=0)
    ocean_eta      = np.where(ocean_mask[None, :, :], zs - zs[0:1, :, :], np.nan)

    land_vals  = peak_land_depth[np.isfinite(peak_land_depth)]
    depth_vmax = float(np.ceil(np.nanpercentile(land_vals, 99) * 2) / 2) if land_vals.size else 2.0
    depth_vmax = max(depth_vmax, 0.5)
    all_eta    = np.abs(ocean_eta[np.isfinite(ocean_eta)])
    eta_vmax   = float(np.percentile(all_eta, 99)) if all_eta.size else 1.0
    eta_vmax   = max(eta_vmax, 0.25)

    peak_t_idx  = int(np.nanargmax(np.nanmean(np.where(np.isfinite(land_depth), land_depth, 0.0), axis=(1, 2))))
    peak_time   = timestamps[peak_t_idx]
    peak_wet_km2 = float(np.sum(np.isfinite(land_depth[peak_t_idx])) * 90 * 90 / 1e6)

    flood_norm_anim = mcolors.Normalize(vmin=0, vmax=depth_vmax)
    ocean_norm_anim = mcolors.TwoSlopeNorm(vmin=-eta_vmax, vcenter=0.0, vmax=eta_vmax)
    x_min, x_max    = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max    = float(np.nanmin(y)), float(np.nanmax(y))

    print(f"Frames: {n_steps}")
    print(f"Peak flooded area: {peak_wet_km2:.2f} km² at {peak_time}")
    print(f"Display depth threshold: {display_depth_threshold*100:.0f} cm  (HUTHRESH={huthresh*100:.0f} cm)")

    # FIX: rebuild h with a proper datetime index so axvline(peak_time) shares the same
    # scale.  Without this matplotlib interprets the integer index as days-since-1970.
    h_dt = pd.Series(h_series.values, index=pd.date_range(t_start, periods=len(h_series), freq="h"))

    # Static summary panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h_dt.plot(ax=axes[0], color="#0b6e99", linewidth=2, label="surge_absolute")
    axes[0].axhline(zsini, color="#d95f02", linestyle="--", linewidth=1.5, label=f"zsini = {zsini:.2f} m")
    axes[0].axvline(peak_time, color="#444444", linestyle=":", linewidth=1.5, label="peak flood time")
    axes[0].set_title(f"{event_label} boundary water level")
    axes[0].set_ylabel("Water level (m MSL)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    mesh = axes[1].pcolormesh(
        x, y, np.ma.masked_invalid(peak_land_depth),
        cmap="Blues", norm=flood_norm_anim, shading="auto", rasterized=True,
    )
    axes[1].set_title(f"Peak flood depth ({peak_time:%Y-%m-%d %H:%M})")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.04, label="Flood depth (m)")
    plt.tight_layout()
    plt.show()

    # Animation
    BMAP_SOURCE = ctx.providers.Esri.WorldImagery

    def _setup_axes(figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        try:
            ctx.add_basemap(ax, crs=f"EPSG:{epsg}", source=BMAP_SOURCE,
                            zoom=bmap_zoom, attribution=False)
        except Exception as exc:
            print(f"Satellite basemap unavailable: {exc}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Easting (m)", color="white")
        ax.set_ylabel("Northing (m)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_aspect("equal")
        return fig, ax

    fig_anim, ax_anim = _setup_axes()
    ocean_mesh = ax_anim.pcolormesh(
        x, y, np.ma.masked_invalid(ocean_eta[0]),
        cmap="RdBu_r", norm=ocean_norm_anim, shading="auto",
        rasterized=True, alpha=0.38, zorder=1,
    )
    flood_mesh = ax_anim.pcolormesh(
        x, y, np.ma.masked_invalid(land_depth[0]),
        cmap="Blues", norm=flood_norm_anim, shading="auto",
        rasterized=True, alpha=0.72, zorder=2,
    )
    title_txt = ax_anim.set_title("", color="white", fontsize=10, pad=8)

    def _update(i):
        ocean_mesh.set_array(np.ma.masked_invalid(ocean_eta[i]).ravel())
        flood_mesh.set_array(np.ma.masked_invalid(land_depth[i]).ravel())
        wet_km2 = float(np.sum(np.isfinite(land_depth[i])) * 90 * 90 / 1e6)
        title_txt.set_text(
            f"{event_label} | {timestamps[i]:%Y-%m-%d %H:%M} | "
            f"t+{int(time_s[i] // 3600):3d}h | flooded={wet_km2:.2f} km²"
        )
        return ocean_mesh, flood_mesh, title_txt

    ani = animation.FuncAnimation(fig_anim, _update, frames=n_steps, interval=120, blit=False)
    out_mp4 = out_dir / f"{event_id}_flood_ocean_animation.mp4"
    ani.save(
        str(out_mp4),
        writer=animation.FFMpegWriter(
            fps=10, bitrate=1800, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        ),
        dpi=140,
        savefig_kwargs={"facecolor": fig_anim.get_facecolor()},
    )
    plt.close(fig_anim)
    print("Saved animation:", out_mp4)
    return out_mp4


def _grid_cell_area_m2(x: np.ndarray, y: np.ndarray, fallback: float = 100.0) -> float:
    """Median |Δx|·|Δy| of a (regular) SFINCS grid in metres², for flooded-area sums."""
    def _spacing(arr, axis):
        if arr.ndim > 1:
            diffs = np.abs(np.diff(arr, axis=axis))
        else:
            diffs = np.abs(np.diff(arr))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        return float(np.median(diffs)) if diffs.size else fallback

    dx = _spacing(x, axis=-1)
    dy = _spacing(y, axis=0)
    area = dx * dy
    return area if np.isfinite(area) and area > 0 else fallback * fallback


def plot_inland_flood_animation(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    discharge,
    t_start: pd.Timestamp,
    huthresh: float = 0.02,
    display_depth_threshold: float = 0.05,
    bmap_zoom: int = 12,
) -> Path:
    """
    Inland (fluvial/pluvial) counterpart to ``plot_flood_animation``.

    Static peak-depth summary (shown inline) + flood/discharge mp4 (saved).

    The coastal version renders an ocean surface anomaly and a boundary
    water-level panel; neither exists for an inland Wflow→SFINCS run. This version
    animates land flood depth over a satellite basemap with a synced discharge
    hydrograph (the inflow fed to the Wflow→SFINCS ``src`` points) carrying a
    moving time cursor.

    Parameters
    ----------
    discharge : pandas Series or DataFrame
        Inflow discharge fed to the SFINCS ``src`` points. A DataFrame (time ×
        src) is summed to a total-inflow line; a Series is used directly. If the
        index is not datetime it is rebuilt hourly from ``t_start``.
    display_depth_threshold : float
        Minimum flood depth (m) rendered in the animation. Filters numerical
        noise without changing SFINCS's wet/dry threshold (HUTHRESH).
    huthresh : float
        SFINCS HUTHRESH value — used only in the diagnostic print statement.

    Returns
    -------
    Path to the saved mp4 file.
    """
    from sfincs_runs.scenarios.io import parse_sfincs_inp

    out_dir.mkdir(parents=True, exist_ok=True)

    inp = parse_sfincs_inp(run_root / "sfincs.inp")
    epsg = int(inp.get("epsg", 26919))
    tstart = pd.to_datetime(str(inp["tstart"]), format="%Y%m%d %H%M%S")

    with xr.open_dataset(run_root / "sfincs_map.nc", decode_times=False) as ds:
        x = ds["x"].values.astype(float)
        y = ds["y"].values.astype(float)
        time_s = ds["time"].values.astype(float)
        zb = ds["zb"].values.astype(float)
        msk = ds["msk"].values.astype(float)
        zs = ds["zs"].values.astype(float)

    timestamps = [tstart + pd.Timedelta(seconds=float(t)) for t in time_s]
    n_steps = len(timestamps)
    active = np.isfinite(msk) & (msk > 0)

    depth_all = zs - zb[None, :, :]
    land_depth = np.where(active[None, :, :] & (depth_all > display_depth_threshold), depth_all, np.nan)
    peak_land_depth = np.nanmax(land_depth, axis=0)

    land_vals = peak_land_depth[np.isfinite(peak_land_depth)]
    depth_vmax = float(np.ceil(np.nanpercentile(land_vals, 99) * 2) / 2) if land_vals.size else 2.0
    depth_vmax = max(depth_vmax, 0.5)

    px_area = _grid_cell_area_m2(x, y)
    wet_km2 = np.array([float(np.sum(np.isfinite(land_depth[i])) * px_area / 1e6) for i in range(n_steps)])
    peak_t_idx = int(np.nanargmax(np.nan_to_num(np.nanmean(np.where(np.isfinite(land_depth), land_depth, 0.0), axis=(1, 2)))))
    peak_time = timestamps[peak_t_idx]

    # Discharge → total-inflow line on a datetime axis sharing the map clock.
    disch = discharge.sum(axis=1) if isinstance(discharge, pd.DataFrame) else pd.Series(discharge)
    if not isinstance(disch.index, pd.DatetimeIndex):
        disch = pd.Series(disch.to_numpy(), index=pd.date_range(t_start, periods=len(disch), freq="h"))
    disch = disch.sort_index()

    flood_norm = mcolors.Normalize(vmin=0, vmax=depth_vmax)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    print(f"Frames: {n_steps}")
    print(f"Peak flooded area: {wet_km2[peak_t_idx]:.2f} km² at {peak_time}")
    print(f"Peak inflow: {float(disch.max()):.1f} m³/s")
    print(f"Display depth threshold: {display_depth_threshold*100:.0f} cm  (HUTHRESH={huthresh*100:.0f} cm)")

    # Static summary panel: discharge hydrograph + peak flood depth map.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    disch.plot(ax=axes[0], color="#08519c", linewidth=2, label="total inflow")
    axes[0].axvline(peak_time, color="#444444", linestyle=":", linewidth=1.5, label="peak flood time")
    axes[0].set_title(f"{event_label} Wflow→SFINCS inflow discharge")
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    mesh = axes[1].pcolormesh(
        x, y, np.ma.masked_invalid(peak_land_depth),
        cmap="Blues", norm=flood_norm, shading="auto", rasterized=True,
    )
    axes[1].set_title(f"Peak flood depth ({peak_time:%Y-%m-%d %H:%M})")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.04, label="Flood depth (m)")
    plt.tight_layout()
    plt.show()

    # Animation: flood depth over satellite imagery + synced discharge hydrograph.
    BMAP_SOURCE = ctx.providers.Esri.WorldImagery

    fig_anim = plt.figure(figsize=(10, 10))
    fig_anim.patch.set_facecolor("#1a1a2e")
    grid = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.18, figure=fig_anim)
    ax_map = fig_anim.add_subplot(grid[0])
    ax_hyd = fig_anim.add_subplot(grid[1])

    ax_map.set_facecolor("#1a1a2e")
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    try:
        ctx.add_basemap(ax_map, crs=f"EPSG:{epsg}", source=BMAP_SOURCE, zoom=bmap_zoom, attribution=False)
    except Exception as exc:
        print(f"Satellite basemap unavailable: {exc}")
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_xlabel("Easting (m)", color="white")
    ax_map.set_ylabel("Northing (m)", color="white")
    ax_map.tick_params(colors="white")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("white")
    ax_map.set_aspect("equal")

    flood_mesh = ax_map.pcolormesh(
        x, y, np.ma.masked_invalid(land_depth[0]),
        cmap="Blues", norm=flood_norm, shading="auto",
        rasterized=True, alpha=0.78, zorder=2,
    )
    cbar = fig_anim.colorbar(flood_mesh, ax=ax_map, fraction=0.04, pad=0.02, label="Flood depth (m)")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    title_txt = ax_map.set_title("", color="white", fontsize=10, pad=8)

    ax_hyd.set_facecolor("#1a1a2e")
    ax_hyd.plot(disch.index, disch.to_numpy(), color="#6baed6", linewidth=1.8)
    ax_hyd.set_ylabel("Inflow (m³/s)", color="white", fontsize=9)
    ax_hyd.tick_params(colors="white", labelsize=8)
    for spine in ax_hyd.spines.values():
        spine.set_edgecolor("white")
    ax_hyd.grid(True, alpha=0.2)
    _format_datetime_axis(ax_hyd, pd.DatetimeIndex(disch.index))
    cursor = ax_hyd.axvline(timestamps[0], color="#fd8d3c", linewidth=1.6)

    def _update(i):
        flood_mesh.set_array(np.ma.masked_invalid(land_depth[i]).ravel())
        cursor.set_xdata([timestamps[i], timestamps[i]])
        title_txt.set_text(
            f"{event_label} | {timestamps[i]:%Y-%m-%d %H:%M} | "
            f"t+{int(time_s[i] // 3600):3d}h | flooded={wet_km2[i]:.2f} km²"
        )
        return flood_mesh, cursor, title_txt

    ani = animation.FuncAnimation(fig_anim, _update, frames=n_steps, interval=120, blit=False)
    out_mp4 = out_dir / f"{event_id}_flood_discharge_animation.mp4"
    ani.save(
        str(out_mp4),
        writer=animation.FFMpegWriter(
            fps=10, bitrate=1800, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        ),
        dpi=140,
        savefig_kwargs={"facecolor": fig_anim.get_facecolor()},
    )
    plt.close(fig_anim)
    print("Saved animation:", out_mp4)
    return out_mp4


# ─── plot_postrun_diagnostics ─────────────────────────────────────────────────


def plot_postrun_diagnostics(
    *,
    run_root: Path,
    event_label: str,
    t_start: pd.Timestamp | None = None,
    run_start: pd.Timestamp | None = None,
    hydrology_inputs: dict | None = None,
    gauge_names: list[str] | None = None,
    has_rugfile: bool = True,
) -> None:
    """
    3-panel post-run check (shown inline, not saved).

    Panels
    ------
    [0] Initial soil saturation fraction histogram (seff/smax)
    [1] Runup gauge water surface from sfincs_his.nc
         – waves builds: runup_gauge_zs variable, labelled from gauge_names
         – standard builds: falls back to point_zs (obs station)
    [2] Domain-mean AORC precipitation rate bar chart

    Parameters
    ----------
    gauge_names
        Ordered list of gauge name strings for the legend.  Pass
        ``[t[0] for t in transects]`` from waves notebooks.  If None
        gauge labels default to "gauge 0", "gauge 1", …
    has_rugfile
        False for standard (non-wave) builds that have no rugfile and
        therefore no ``runup_gauge_zs`` variable in sfincs_his.nc.
    """
    run_root = Path(run_root)
    fallback_start = run_start if run_start is not None else t_start
    sfincs_run_start = _resolve_run_start(run_root, fallback=fallback_start)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # [0] Soil saturation fraction
    smax = np.fromfile(run_root / "sfincs.smax", dtype="<f4") if (run_root / "sfincs.smax").exists() else np.array([])
    seff = np.fromfile(run_root / "sfincs.seff", dtype="<f4") if (run_root / "sfincs.seff").exists() else np.array([])
    if smax.size and seff.size:
        valid = np.isfinite(smax) & np.isfinite(seff) & (smax > 0)
        frac  = seff[valid] / smax[valid]
        axes[0].hist(frac, bins=40, color="#6baed6", edgecolor="white", linewidth=0.4)
        axes[0].axvline(float(np.median(frac)), color="#08519c", linestyle="--", linewidth=1.5,
                        label=f"median = {float(np.median(frac)):.2f}")
        sm = (hydrology_inputs or {}).get("soil_moisture_summary") or {}
        title_extra = f"  (NWM mean = {sm.get('mean_soil_moisture', float('nan')):.2f})" if sm else ""
        axes[0].set_title(f"Initial soil saturation (seff/smax){title_extra}")
        axes[0].set_xlabel("Fraction")
        axes[0].set_ylabel("Cell count")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_title("Initial soil saturation")
        axes[0].text(0.5, 0.5, "smax/seff not staged", ha="center", va="center",
                     transform=axes[0].transAxes)

    # [1] Runup gauge / obs station zs
    his_path = run_root / "sfincs_his.nc"
    if his_path.exists():
        with xr.open_dataset(his_path, decode_times=False) as his:
            his_ts = pd.DatetimeIndex(
                sfincs_run_start + pd.to_timedelta(his["time"].values.astype(float), unit="s")
            )
            if has_rugfile and "runup_gauge_zs" in his.data_vars:
                rug_zs = his["runup_gauge_zs"].values.astype(float)
                n_rug  = rug_zs.shape[1]
                for gi in range(n_rug):
                    vals = np.where(rug_zs[:, gi] > -900.0, rug_zs[:, gi], np.nan)
                    lbl  = gauge_names[gi] if gauge_names and gi < len(gauge_names) else f"gauge {gi}"
                    axes[1].plot(his_ts, vals, linewidth=1.8, label=lbl)
                if not np.any(rug_zs > -900.0):
                    axes[1].text(0.5, 0.5, "all -999 fill\n(gauge outside active cells)",
                                 ha="center", va="center", transform=axes[1].transAxes, fontsize=9)
                elif n_rug:
                    axes[1].legend(fontsize=8)
            elif "point_zs" in his.data_vars:
                pt_zs = his["point_zs"].values[:, 0].astype(float)
                axes[1].plot(his_ts, np.where(np.isfinite(pt_zs), pt_zs, np.nan),
                             linewidth=1.8, color="#2171b5", label="obs station zs")
                axes[1].legend(fontsize=8)
            else:
                axes[1].text(0.5, 0.5, "no gauge / obs output in his.nc",
                             ha="center", va="center", transform=axes[1].transAxes, fontsize=9)
            _format_datetime_axis(axes[1], his_ts)
    else:
        axes[1].text(0.5, 0.5, "sfincs_his.nc not found", ha="center", va="center",
                     transform=axes[1].transAxes)
    axes[1].set_title("Runup gauge / obs station zs (sfincs_his.nc)")
    axes[1].set_ylabel("zs (m MSL)")
    axes[1].grid(True, alpha=0.25)

    # [2] Domain-mean precip rate
    precip_nc = run_root / "aorc_precip_for_sfincs.nc"
    if precip_nc.exists():
        with xr.open_dataset(precip_nc) as dp:
            var         = "precip" if "precip" in dp.data_vars else next(iter(dp.data_vars))
            precip_times = pd.DatetimeIndex(dp["time"].values)
            precip_mean  = float(dp[var].mean(skipna=True).values)
            precip_ts    = dp[var].mean(
                dim=[d for d in dp[var].dims if d != "time"], skipna=True
            ).values
        axes[2].bar(range(len(precip_times)), precip_ts, color="#74c476",
                    edgecolor="white", linewidth=0.4)
        _apply_day_ticks(axes[2], precip_times)
        axes[2].set_title(f"Domain-mean precip rate (mean={precip_mean:.2f} mm/hr)")
        axes[2].set_ylabel("mm / hr")
        axes[2].grid(True, axis="y", alpha=0.25)
    else:
        axes[2].set_title("Domain-mean precip rate")
        axes[2].text(0.5, 0.5, "aorc_precip_for_sfincs.nc not found", ha="center", va="center",
                     transform=axes[2].transAxes)

    plt.suptitle(f"{event_label} post-run diagnostics", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


# ─── plot_precip_animation ────────────────────────────────────────────────────


def plot_precip_animation(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    domain_gdf=None,
) -> Path | None:
    """
    Spatiotemporal AORC precipitation mp4 animation.

    Two-panel layout: satellite basemap with precipitation overlay (top)
    + domain-mean bar chart with animated cursor (bottom).

    Parameters
    ----------
    domain_gdf
        Optional GeoDataFrame (any CRS) whose boundary is drawn on the spatial
        panel as a yellow SFINCS-domain outline.  Pass ``None`` to omit.

    Returns
    -------
    Path to the saved mp4, or None if the precip file was not found.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    precip_nc = run_root / "aorc_precip_for_sfincs.nc"

    if not precip_nc.exists():
        print(f"Precip file not found: {precip_nc}")
        return None

    with xr.open_dataset(precip_nc) as dp:
        pvar         = "precip" if "precip" in dp.data_vars else next(iter(dp.data_vars))
        precip_arr   = dp[pvar].values.astype(float)
        lon          = dp["x"].values
        lat          = dp["y"].values
        precip_times = pd.DatetimeIndex(dp["time"].values)
        precip_units = dp[pvar].attrs.get("units", "mm")

    n_pr_frames = len(precip_times)
    LON, LAT    = np.meshgrid(lon, lat)

    wet     = precip_arr[precip_arr > 0.0]
    pr_vmax = float(np.percentile(wet, 95)) if wet.size else 1.0
    pr_vmax = max(pr_vmax, 1.0)

    domain_mean = np.nanmean(precip_arr, axis=(1, 2))
    event_total = float(np.nansum(precip_arr))

    print(f"Frames      : {n_pr_frames}")
    print(f"Colour max  : {pr_vmax:.2f} {precip_units}  (95th pct of wet pixels)")
    print(f"Domain total: {event_total:.0f} mm·px (sum over grid and time)")

    fig_pr = plt.figure(figsize=(10, 10), facecolor="#1a1a2e")
    gs     = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.08, figure=fig_pr)
    ax_map = fig_pr.add_subplot(gs[0])
    ax_ts  = fig_pr.add_subplot(gs[1])
    for ax in (ax_map, ax_ts):
        ax.set_facecolor("#1a1a2e")

    ax_map.set_xlim(lon.min(), lon.max())
    ax_map.set_ylim(lat.min(), lat.max())
    try:
        ctx.add_basemap(ax_map, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
                        zoom=10, attribution=False)
    except Exception as exc:
        print(f"Satellite basemap unavailable: {exc}")
    ax_map.set_xlim(lon.min(), lon.max())
    ax_map.set_ylim(lat.min(), lat.max())

    pr_cmap    = plt.cm.YlGnBu
    pr_norm    = mcolors.Normalize(vmin=0, vmax=pr_vmax)
    pr_masked0 = np.ma.masked_where(precip_arr[0] <= 0.0, precip_arr[0])
    pm = ax_map.pcolormesh(
        LON, LAT, pr_masked0,
        cmap=pr_cmap, norm=pr_norm, shading="auto",
        alpha=0.80, rasterized=True, zorder=2,
    )
    cbar = fig_pr.colorbar(pm, ax=ax_map, fraction=0.030, pad=0.02)
    cbar.set_label(f"Precip ({precip_units})", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if domain_gdf is not None:
        domain_gdf.to_crs("EPSG:4326").boundary.plot(
            ax=ax_map, color="#ffcc33", linewidth=2.0, zorder=4, label="SFINCS domain"
        )
        ax_map.legend(loc="upper right", fontsize=8,
                      facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    ax_map.set_xlabel("Longitude", color="white", fontsize=8)
    ax_map.set_ylabel("Latitude",  color="white", fontsize=8)
    ax_map.tick_params(colors="white", labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("white")

    title_pr = ax_map.set_title("", color="white", fontsize=10, pad=6)

    bar_colors = [pr_cmap(pr_norm(v)) for v in domain_mean]
    ax_ts.bar(range(n_pr_frames), domain_mean, color=bar_colors, width=0.7, zorder=2)
    ax_ts.set_xlim(-0.5, n_pr_frames - 0.5)
    _apply_day_ticks(ax_ts, precip_times, color="white")
    ax_ts.set_ylabel(f"Domain mean\n({precip_units})", color="white", fontsize=8)
    ax_ts.tick_params(colors="white", labelsize=7)
    ax_ts.yaxis.set_label_position("right")
    ax_ts.yaxis.tick_right()
    for sp in ax_ts.spines.values():
        sp.set_edgecolor("#555566")
    ax_ts.set_facecolor("#1a1a2e")
    ax_ts.grid(axis="y", color="#555566", linewidth=0.5, zorder=1)

    cursor_line = ax_ts.axvline(0, color="white", linewidth=1.5, zorder=3)

    def _update_pr(i):
        pr_frame = np.ma.masked_where(precip_arr[i] <= 0.0, precip_arr[i])
        pm.set_array(pr_frame.ravel())
        cursor_line.set_xdata([i, i])
        title_pr.set_text(
            f"{event_label} · AORC precip | {precip_times[i]:%Y-%m-%d %H:%M UTC} | "
            f"domain mean = {domain_mean[i]:.2f} {precip_units}"
        )
        return pm, cursor_line, title_pr

    ani_pr = animation.FuncAnimation(fig_pr, _update_pr, frames=n_pr_frames, interval=400, blit=False)
    out_precip_mp4 = out_dir / f"{event_id}_precip_animation.mp4"
    ani_pr.save(
        str(out_precip_mp4),
        writer=animation.FFMpegWriter(
            fps=3, bitrate=1200, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        ),
        dpi=140,
        savefig_kwargs={"facecolor": fig_pr.get_facecolor()},
    )
    plt.close(fig_pr)
    print("Saved:", out_precip_mp4)
    return out_precip_mp4


# ─── plot_runup_overtopping ───────────────────────────────────────────────────


def plot_runup_overtopping(
    *,
    run_root: Path,
    out_dir: Path,
    event_id: str,
    event_label: str,
    wave_cfg: dict,
    config: dict,
    model_crs: str,
    run_start: pd.Timestamp,
) -> Path | None:
    """
    Runup gauge location map + per-gauge crest overtopping screen (waves builds).

    Layout
    ------
    Left  : OSM basemap with point markers at gauge midpoints + structure lines.
    Right : One subplot row per gauge (shared x-axis), showing the gauge zs time
            series, dashed crest line, overtopping fill, and Hs on a secondary
            y-axis.

    Overtopping convention: crest-exceedance screen for configured runup gauges.

    Returns
    -------
    Path to the saved PNG, or None if no gauges are configured / his.nc absent.
    """
    import geopandas as gpd

    runup_cfg       = wave_cfg.get("runup_gauges") or config.get("scenario_build", {}).get("runup_gauges", {})
    runup_transects = runup_cfg.get("transects", [])
    runup_crs       = runup_cfg.get("crs", model_crs)

    structure_paths = [
        (run_root / "gis" / "weir.geojson",    "weir",     "#c44e52"),
        (run_root / "gis" / "thd.geojson",     "thin_dam", "#8172b2"),
    ]

    def _read_layer(path, kind):
        if not path.exists():
            return gpd.GeoDataFrame(
                columns=["kind", "name", "z", "geometry"], geometry="geometry", crs=model_crs
            )
        layer = gpd.read_file(path).to_crs(model_crs)
        layer["kind"] = kind
        if "name" not in layer.columns:
            layer["name"] = [f"{kind}_{i}" for i in range(len(layer))]
        if "z" not in layer.columns:
            layer["z"] = np.nan
        return layer[["kind", "name", "z", "geometry"]]

    structures = gpd.GeoDataFrame(
        pd.concat([_read_layer(p, k) for p, k, _ in structure_paths], ignore_index=True),
        geometry="geometry", crs=model_crs,
    )
    structures = structures[~structures.geometry.is_empty & structures.geometry.notna()].copy()

    transect_records = [
        {"gauge": rec["name"],
         "geometry": LineString([(rec["x0"], rec["y0"]), (rec["x1"], rec["y1"])])}
        for rec in runup_transects
    ]
    transect_gdf = (
        gpd.GeoDataFrame(transect_records, geometry="geometry", crs=runup_crs).to_crs(model_crs)
        if transect_records
        else gpd.GeoDataFrame(columns=["gauge", "geometry"], geometry="geometry", crs=model_crs)
    )

    his_path = run_root / "sfincs_his.nc"
    if transect_gdf.empty:
        print("No runup gauges are configured for this event.")
        return None
    if not his_path.exists():
        print(f"Missing {his_path}; run the model before plotting runup gauge diagnostics.")
        return None

    with xr.open_dataset(his_path, decode_times=False) as his_ds:
        runup_var = next((n for n in ("runup_gauge_zs", "rug_zs", "zs") if n in his_ds), None)
        if runup_var is None:
            raise KeyError(f"No runup gauge variable found in {his_path}")
        time_var     = "time" if "time" in his_ds else list(his_ds[runup_var].dims)[0]
        his_seconds  = np.asarray(his_ds[time_var].values, dtype=float)
        runup_values = np.asarray(his_ds[runup_var].values, dtype=float)

    if runup_values.ndim == 1:
        runup_values = runup_values[:, np.newaxis]
    if runup_values.shape[0] != len(his_seconds):
        runup_values = np.moveaxis(
            runup_values,
            np.where(np.asarray(runup_values.shape) == len(his_seconds))[0][0],
            0,
        )
    runup_values = np.where(runup_values <= -900.0, np.nan, runup_values)
    runup_times  = pd.DatetimeIndex(run_start + pd.to_timedelta(his_seconds, unit="s"))
    dt_hours     = float(np.nanmedian(np.diff(his_seconds)) / 3600.0) if len(his_seconds) > 1 else np.nan

    wave_summary = None
    bhs_path = run_root / "snapwave.bhs"
    if bhs_path.exists():
        bhs = pd.read_csv(bhs_path, sep=r"\s+", header=None)
        if bhs.shape[1] > 1:
            wave_times   = pd.DatetimeIndex(
                run_start + pd.to_timedelta(bhs.iloc[:, 0].astype(float), unit="s")
            )
            wave_summary = pd.Series(
                bhs.iloc[:, 1:].mean(axis=1).to_numpy(dtype=float),
                index=wave_times, name="mean_Hs_m",
            )

    def _nearest_structure(geom):
        if structures.empty:
            return None, np.nan
        distances = structures.geometry.distance(geom)
        idx = distances.idxmin()
        return structures.loc[idx], float(distances.loc[idx])

    def _nearest_kind_distance(geom, kind):
        subset = structures[structures["kind"] == kind]
        return np.nan if subset.empty else float(subset.geometry.distance(geom).min())

    n_gauges = len(transect_gdf)
    _tab10   = plt.cm.tab10.colors

    # Figure: left = map (full height), right = stacked per-gauge subplots
    _n_rows = max(n_gauges, 1)
    fig = plt.figure(figsize=(17, max(4.5, 3.8 * _n_rows)))
    gs_fig = gridspec.GridSpec(
        _n_rows, 2, width_ratios=[1, 2.4], hspace=0.12, wspace=0.15, figure=fig
    )
    ax_map = fig.add_subplot(gs_fig[:, 0])

    ax_ts_list: list = []
    for _idx in range(_n_rows):
        _sharex = ax_ts_list[0] if _idx > 0 else None
        ax_ts_list.append(fig.add_subplot(gs_fig[_idx, 1], sharex=_sharex))

    # Map — point markers at transect midpoints (no line drawn)
    struct_wm   = structures.to_crs(epsg=3857)
    transect_wm = transect_gdf.to_crs(epsg=3857)
    all_wm = gpd.GeoDataFrame(
        geometry=list(struct_wm.geometry) + list(transect_wm.geometry), crs=3857
    )
    xmin, ymin, xmax, ymax = all_wm.total_bounds
    buf = 500
    ax_map.set_xlim(xmin - buf, xmax + buf)
    ax_map.set_ylim(ymin - buf, ymax + buf)

    for path, kind, color in structure_paths:
        layer = struct_wm[struct_wm["kind"] == kind]
        if not layer.empty:
            layer.plot(ax=ax_map, color=color, linewidth=3.0, label=kind.replace("_", " "), zorder=3)

    for i, (_, row) in enumerate(transect_wm.reset_index(drop=True).iterrows()):
        gc    = _tab10[i % len(_tab10)]
        midpt = row.geometry.interpolate(0.5, normalized=True)
        ax_map.scatter(midpt.x, midpt.y, color=gc, s=100, marker="o",
                       edgecolors="white", linewidths=0.9, zorder=5)

    ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto", alpha=0.9)
    ax_map.set_title("Runup gauge locations", fontsize=10, fontweight="bold")
    ax_map.set_axis_off()

    struct_handles = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=k.replace("_", " "))
        for _, k, c in structure_paths
        if not structures[structures["kind"] == k].empty
    ] + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_tab10[i % len(_tab10)], markeredgecolor="white",
                   markersize=8, label=row["gauge"])
        for i, (_, row) in enumerate(transect_gdf.reset_index(drop=True).iterrows())
    ]
    ax_map.legend(handles=struct_handles, loc="lower right", fontsize=7.5, framealpha=0.85)

    # Time-series subplots
    summary_rows = []
    for i, row in transect_gdf.reset_index(drop=True).iterrows():
        gc    = _tab10[i % len(_tab10)]
        ax_ts = ax_ts_list[i]
        ax_hs = ax_ts.twinx() if wave_summary is not None else None

        vals = (runup_values[:, i] if i < runup_values.shape[1]
                else np.full(len(runup_times), np.nan))
        nearest, distance_m = _nearest_structure(row.geometry)
        nearest_weir_m      = _nearest_kind_distance(row.geometry, "weir")
        nearest_thin_dam_m  = _nearest_kind_distance(row.geometry, "thin_dam")
        crest = (
            float(nearest["z"])
            if nearest is not None and pd.notna(nearest.get("z", np.nan))
            else np.nan
        )

        ax_ts.plot(runup_times, vals, color=gc, linewidth=2.2, zorder=4)

        if np.isfinite(crest):
            ax_ts.axhline(crest, color=gc, linestyle="--", linewidth=1.5, alpha=0.75, zorder=3)
            ax_ts.annotate(f"  crest {crest:.2f} m", xy=(1.0, crest),
                           xycoords=("axes fraction", "data"),
                           va="center", fontsize=7.5, color=gc, clip_on=False)
            ax_ts.fill_between(runup_times, crest, vals, where=vals > crest,
                               color=gc, alpha=0.25, interpolate=True, zorder=2)
            excess_arr = vals - crest
            peak_idx   = int(np.nanargmax(excess_arr))
            peak_exc   = float(excess_arr[peak_idx])
            _lbl = (f"+{peak_exc:.2f} m above crest" if peak_exc > 0
                    else f"freeboard {-peak_exc:.2f} m")
            ax_ts.text(runup_times[peak_idx], vals[peak_idx], f"  {_lbl}",
                       fontsize=7.5, color=gc, va="bottom", ha="left", zorder=6)
            exceedance_samples = int(np.nansum(excess_arr > 0.0))
            exceedance_hours   = exceedance_samples * dt_hours if np.isfinite(dt_hours) else np.nan
            max_excess         = float(np.nanmax(excess_arr)) if np.isfinite(excess_arr).any() else np.nan
        else:
            exceedance_samples = 0
            exceedance_hours   = np.nan
            max_excess         = np.nan

        if wave_summary is not None and ax_hs is not None:
            ax_hs.plot(wave_summary.index, wave_summary.values,
                       color="0.55", linestyle=":", linewidth=1.5, label="mean Hs (SnapWave bnd)")
            ax_hs.set_ylabel("Hs (m)", color="0.55", fontsize=8)
            ax_hs.tick_params(axis="y", colors="0.55", labelsize=7)
            ax_hs.set_ylim(bottom=0)
            if i == 0:
                ax_hs.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        ax_ts.set_ylabel("zs (m MSL)", fontsize=9)
        ax_ts.set_title(row["gauge"], fontsize=9, loc="left", color=gc, fontweight="bold", pad=4)
        ax_ts.grid(True, alpha=0.25)
        if i < n_gauges - 1:
            plt.setp(ax_ts.get_xticklabels(), visible=False)

        summary_rows.append({
            "gauge":              row["gauge"],
            "nearest_structure":  None if nearest is None else nearest["kind"],
            "distance_m":         distance_m,
            "nearest_weir_m":     nearest_weir_m,
            "nearest_thin_dam_m": nearest_thin_dam_m,
            "crest_m_msl":        crest,
            "max_runup_m_msl":    float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan,
            "max_excess_m":       max_excess,
            "exceedance_samples": exceedance_samples,
            "exceedance_hours":   exceedance_hours,
        })

    if ax_ts_list:
        fig.autofmt_xdate(rotation=20, ha="right")

    fig.suptitle(f"{event_label}  ·  runup and crest overtopping screen",
                 fontsize=12, fontweight="bold", y=1.01)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_id}_runup_structure_qa.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()

    display(pd.DataFrame(summary_rows))
    print(f"Saved runup/structure QA figure → {out_path}")
    return out_path
