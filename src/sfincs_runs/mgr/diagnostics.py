"""SFINCS diagnostic tables, probability products, and compatibility exports.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

M_TO_FT = 3.28084
SLR_BENCHMARK_RETURN_PERIODS = (10, 100, 500)

_PLOTTING_EXPORTS = {
    'plot_flood_response_diagnostics',
    'plot_driver_response_matrix',
    'plot_driver_outcome_matrix',
    'plot_forcing',
    'plot_wave_forcing',
    'plot_standard_animation',
    'plot_animation',
    'plot_standard_diagnostics',
    'plot_precip_animation',
    'plot_runup',
    'plot_slr_depth_comparison',
}

__all__ = [
    'FloodResponseDiagnostics',
    'DriverResponseDiagnostics',
    'SlrDepthComparison',
    'flood_response_diagnostics',
    'driver_response_diagnostics',
    'masked_sfincs_depth',
    'weighted_standardized_associations',
    'slr_scenario_storage_roots',
    'common_completed_events',
    'scenario_paths',
    'completed_runs_by_scenario',
    'health_check_table',
    'select_benchmark_events',
    'slr_event_depth_comparison',
    *_PLOTTING_EXPORTS,
]

@dataclass(frozen=True)
class FloodResponseDiagnostics:
    """Return-period, storm-family, and largest-response summaries."""

    flood: pd.DataFrame
    scope: pd.Series
    rp_band_stats: pd.DataFrame
    storm_type_stats: pd.DataFrame
    top_flood_events: pd.DataFrame


@dataclass(frozen=True)
class DriverResponseDiagnostics:
    """Weighted driver/flood-response association table."""

    associations: pd.DataFrame
    associations_path: Path
    driver_columns: list[str]
    outcome_columns: list[str]

    @property
    def top_associations(self) -> pd.DataFrame:
        if self.associations.empty:
            return self.associations
        return self.associations.sort_values(
            "standardized_wls_coefficient",
            key=lambda s: s.abs(),
            ascending=False,
        ).head(30)


# ─── private helpers ──────────────────────────────────────────────────────────


def flood_response_diagnostics(catalogue_csv) -> FloodResponseDiagnostics:
    """Summarize how sampled event severity maps to flood depth, area, and duration."""
    path = Path(catalogue_csv)
    flood = pd.read_csv(path)
    numeric_cols = [
        "sample_rp_years", "coastal_absolute_peak_m", "coastal_peak_m", "rainfall_metric_mm", "bzs_peak_max_m",
        "peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2",
        "anytime_incremental_flooded_area_km2", "longest_incremental_flood_duration_h",
        "mean_incremental_flood_duration_h", "cumulative_incremental_flooded_area_km2h",
    ]
    for col in [c for c in numeric_cols if c in flood]:
        flood[col] = pd.to_numeric(flood[col], errors="coerce")

    rp_breaks = [0, 2, 10, 50, 100, 500, np.inf]
    rp_labels = ["<2 yr", "2-10 yr", "10-50 yr", "50-100 yr", "100-500 yr", ">500 yr"]
    flood["rp_band"] = pd.cut(flood["sample_rp_years"], bins=rp_breaks, labels=rp_labels, right=False)
    flood["has_incremental_flood"] = flood["anytime_incremental_flooded_area_km2"] > 0

    def p90(values):
        return values.quantile(0.90)

    rp_band_stats = (
        flood.groupby(["rp_band", "severity_band"], observed=True)
        .agg(
            event_count=("event_id", "nunique"),
            median_rp_years=("sample_rp_years", "median"),
            median_peak_boundary_m=("bzs_peak_max_m", "median"),
            median_rainfall_mm=("rainfall_metric_mm", "median"),
            flood_hit_rate=("has_incremental_flood", "mean"),
            median_peak_depth_m=("peak_incremental_land_depth_m", "median"),
            p90_peak_depth_m=("peak_incremental_land_depth_m", p90),
            median_peak_area_km2=("peak_incremental_flooded_area_km2", "median"),
            p90_anytime_area_km2=("anytime_incremental_flooded_area_km2", p90),
            p90_longest_duration_h=("longest_incremental_flood_duration_h", p90),
        )
        .reset_index()
    )
    storm_type_stats = (
        flood.groupby(["storm_type", "severity_band"], observed=True)
        .agg(
            event_count=("event_id", "nunique"),
            median_rp_years=("sample_rp_years", "median"),
            median_peak_boundary_m=("bzs_peak_max_m", "median"),
            median_rainfall_mm=("rainfall_metric_mm", "median"),
            median_peak_area_km2=("peak_incremental_flooded_area_km2", "median"),
            p90_anytime_area_km2=("anytime_incremental_flooded_area_km2", p90),
            p90_longest_duration_h=("longest_incremental_flood_duration_h", p90),
        )
        .reset_index()
    )
    top_cols = [
        "event_id", "storm_type", "severity_band", "sample_rp_years",
        "coastal_absolute_peak_m", "rainfall_metric_mm", "bzs_peak_max_m",
        "peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2",
        "anytime_incremental_flooded_area_km2", "longest_incremental_flood_duration_h",
    ]
    top_flood_events = flood.sort_values(
        ["anytime_incremental_flooded_area_km2", "peak_incremental_land_depth_m"],
        ascending=False,
    )[[c for c in top_cols if c in flood]].head(15)
    scope = pd.Series(
        {
            "catalogue_csv": str(path),
            "events": int(len(flood)),
            "storm_types": ", ".join(sorted(flood["storm_type"].dropna().astype(str).unique())),
            "max_anytime_incremental_area_km2": flood["anytime_incremental_flooded_area_km2"].max(),
            "max_peak_incremental_depth_m": flood["peak_incremental_land_depth_m"].max(),
        },
        name="flood_catalogue_scope",
    )
    return FloodResponseDiagnostics(flood, scope, rp_band_stats, storm_type_stats, top_flood_events)


def driver_response_diagnostics(
    outcomes: pd.DataFrame,
    *,
    outdir,
    min_rows: int = 8,
) -> DriverResponseDiagnostics:
    """Estimate weighted diagnostic associations between event drivers and flood outcomes."""
    drivers = [
        "coastal_water_level", "coastal_absolute_peak_m", "rainfall", "rainfall_metric_mm",
        "soil_moisture_metric", "coastal_water_level_scale_factor", "rainfall_scale_factor",
        "rainfall_pairing_lag_hours",
    ]
    response = [
        "peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2",
        "anytime_incremental_flooded_area_km2", "longest_incremental_flood_duration_h",
    ]
    drivers = _usable_numeric_columns(outcomes, drivers)
    response = _usable_numeric_columns(outcomes, response, min_unique=1)
    associations = weighted_standardized_associations(outcomes, drivers=drivers, outcomes=response, min_rows=min_rows)
    out_path = Path(outdir) / "driver_flood_response_diagnostic_associations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    associations.to_csv(out_path, index=False)
    return DriverResponseDiagnostics(associations, out_path, drivers, response)


def _usable_numeric_columns(data: pd.DataFrame, columns: list[str], *, min_unique: int = 2) -> list[str]:
    usable = []
    for column in columns:
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if len(values) and values.nunique() >= int(min_unique):
            usable.append(column)
    return usable


def masked_sfincs_depth(
    map_path,
    *,
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
    depth_kind: str = "incremental",
) -> dict:
    """Masked SFINCS flood depth (ft) on land, with flood extent defined consistently.

    Extent is always the cells that rose above their t0 baseline (the storm footprint),
    which is NaN-safe for cells dry at t0 (SFINCS writes ``zs`` NaN there) — the previous
    ``(zsmax - zs0) > huthresh`` gate dropped exactly those newly-inundated cells. The
    returned ``depth_kind`` is either:

    - ``"incremental"`` (default): water above the t0 antecedent level (mirrors
      ``scenario_stats.event_stats``; storm-controlled, ~SLR-invariant).
    - ``"total"``: absolute inundation depth above ground (``zsmax - zb`` = Total Water
      Level minus bed), the standard coastal flood-depth that rises with sea-level rise.
    """
    with xr.open_dataset(map_path) as ds:
        zsmax = ds["zsmax"].max("timemax") if "zsmax" in ds and "timemax" in ds["zsmax"].dims else ds["zsmax"]
        zb = ds["zb"]
        depth_m = zsmax - zb
        baseline_depth = (ds["zs"].isel(time=0) - zb).clip(min=0.0).fillna(0.0)
        incremental_m = depth_m - baseline_depth
        flooded = incremental_m > huthresh_m
        if land_min_elev_m is not None:
            flooded = flooded & (zb >= land_min_elev_m)
        value = depth_m if depth_kind == "total" else incremental_m
        return {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
            "depth_ft": np.asarray((value.where(flooded) * M_TO_FT).values, dtype=float),
        }


def weighted_standardized_associations(
    data: pd.DataFrame,
    drivers: list[str],
    outcomes: list[str],
    *,
    weight_col: str = "probability_weight",
    group_col: str = "storm_type",
    min_rows: int = 8,
) -> pd.DataFrame:
    """Weighted standardized driver/outcome associations for diagnostic review."""
    rows = []
    groups = [("all", data)]
    if group_col in data:
        groups.extend((str(name), group) for name, group in data.groupby(group_col, dropna=False))
    for group_name, group in groups:
        available_drivers = [d for d in drivers if d in group]
        for outcome in [o for o in outcomes if o in group]:
            cols = [outcome, *available_drivers]
            if weight_col in group:
                cols.append(weight_col)
            sub = group[cols].copy()
            for col in [outcome, *available_drivers, weight_col]:
                if col in sub:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[outcome, *available_drivers])
            if len(sub) < int(min_rows) or not available_drivers:
                continue
            weights = sub[weight_col].to_numpy(dtype=float) if weight_col in sub else np.ones(len(sub), dtype=float)
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
            if weights.sum() <= 0:
                weights = np.ones(len(sub), dtype=float)
            y = _weighted_zscore(sub[outcome].to_numpy(dtype=float), weights)
            xcols = []
            used_drivers = []
            for driver in available_drivers:
                z = _weighted_zscore(sub[driver].to_numpy(dtype=float), weights)
                if np.isfinite(z).all() and np.nanstd(z) > 0:
                    xcols.append(z)
                    used_drivers.append(driver)
            if not xcols:
                continue
            x = np.column_stack([np.ones(len(sub)), *xcols])
            root_w = np.sqrt(weights / weights.mean())
            try:
                beta = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0][1:]
            except np.linalg.LinAlgError:
                continue
            for driver, coefficient in zip(used_drivers, beta):
                rows.append(
                    {
                        "storm_type": group_name,
                        "outcome": outcome,
                        "driver": driver,
                        "standardized_wls_coefficient": float(coefficient),
                        "weighted_correlation": float(_weighted_corr(sub[driver].to_numpy(dtype=float), sub[outcome].to_numpy(dtype=float), weights)),
                        "n_events": int(len(sub)),
                        "interpretation": "diagnostic association, not causal attribution",
                    }
                )
    columns = [
        "storm_type",
        "outcome",
        "driver",
        "standardized_wls_coefficient",
        "weighted_correlation",
        "n_events",
        "interpretation",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["outcome", "storm_type", "driver"]).reset_index(drop=True)


def _weighted_zscore(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    sd = np.sqrt(var)
    return (values - mean) / sd if sd > 0 else np.zeros_like(values)


def _weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    xz = _weighted_zscore(x, weights)
    yz = _weighted_zscore(y, weights)
    return float(np.average(xz * yz, weights=weights))


_PRETTY_AXIS_LABELS = {
    "coastal_water_level": "Coastal Water Level (m)",
    "coastal_peak_m": "Coastal NTR / Surge Peak (m)",
    "coastal_absolute_peak_m": "Coastal Absolute Peak (m)",
    "rainfall": "Rainfall (mm)",
    "rainfall_metric_mm": "72h Mean Rainfall (mm)",
    "soil_moisture_metric": "Soil Moisture Metric",
    "initial_soil_moisture_fraction": "Initial Soil Moisture Fraction",
    "rainfall_pairing_lag_hours": "Rainfall Pairing Lag (hours)",
    "peak_incremental_land_depth_m": "Peak Incremental Land Depth (m)",
    "peak_incremental_flooded_area_km2": "Peak Incremental Flooded Area (km²)",
    "anytime_incremental_flooded_area_km2": "Anytime Incremental Flooded Area (km²)",
}


@dataclass(frozen=True)
class SlrDepthComparison:
    """Per-scenario masked flood depth (ft) and depth deltas vs base for one event."""

    event_id: str
    return_period_years: float
    scenarios: tuple
    offsets_m: dict
    x: np.ndarray
    y: np.ndarray
    depth_ft: dict
    delta_ft: dict


def slr_scenario_storage_roots(storage_root, scenarios, *, base_key: str = "base") -> dict:
    """Map ``scenarios:`` keys to their per-scenario SFINCS run-output directories.

    Convention: the base scenario reuses ``storage_root`` (``run_outputs``); every
    other key uses the sibling ``<storage_root>_<key>`` (e.g. ``run_outputs_noaa_int_2050``).
    Only existing directories are returned.
    """
    storage_root = Path(storage_root)
    roots = {}
    for key in scenarios:
        root = storage_root if key == base_key else storage_root.with_name(f"{storage_root.name}_{key}")
        if root.exists():
            roots[key] = root
    return roots


def common_completed_events(scenario_roots: dict) -> set:
    sets = []
    for root in scenario_roots.values():
        sets.append({p.parent.name for p in Path(root).glob("*/sfincs_map.nc")})
    return set.intersection(*sets) if sets else set()


def scenario_paths(paths: dict, scenario: str, *, base_key: str = "base") -> dict:
    """Per-scenario copy of the notebook ``paths`` dict for one SLR projection.

    Overrides ``storage_root`` and ``scenarios_root`` to the scenario's sibling dirs
    (``<root>_<scenario>``) and isolates ``stats_root`` under a ``<scenario>`` subdir so
    per-scenario outputs don't clobber base. The shared ``design_outputs_root`` (catalog
    drivers are scenario-independent — the same events re-run under a fixed MSL offset) is
    kept as-is. The base scenario returns ``paths`` unchanged.
    """
    if scenario == base_key:
        return dict(paths)
    out = dict(paths)
    storage_root = Path(paths["storage_root"])
    scenarios_root = Path(paths["scenarios_root"])
    out["storage_root"] = storage_root.with_name(f"{storage_root.name}_{scenario}")
    out["scenarios_root"] = scenarios_root.with_name(f"{scenarios_root.name}_{scenario}")
    out["stats_root"] = Path(paths["stats_root"]) / scenario
    return out


def completed_runs_by_scenario(scenario_roots: dict) -> pd.DataFrame:
    """Every completed SFINCS run (has ``sfincs_map.nc``) across all scenario storage roots.

    One row per (scenario, event): ``design_scenario``, ``event_id``, ``run_dir``,
    ``map_path``. The single enumeration the notebook iterates so QA and evaluation span
    base + all SLR projections without per-scenario boilerplate.
    """
    rows = []
    for scenario, root in scenario_roots.items():
        for map_path in sorted(Path(root).glob("*/sfincs_map.nc")):
            rows.append(
                {
                    "design_scenario": scenario,
                    "event_id": map_path.parent.name,
                    "run_dir": map_path.parent,
                    "map_path": map_path,
                }
            )
    return pd.DataFrame(rows, columns=["design_scenario", "event_id", "run_dir", "map_path"])


def _health_check_one(item):
    design_scenario, event_id, run_dir, map_path = item
    rec = {"design_scenario": design_scenario, "event_id": event_id, "open_error": ""}
    meta_path = Path(run_dir) / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        rec["returncode"] = int(meta.get("returncode", -1))
        rec["duration_min"] = float(meta.get("duration_sec", float("nan"))) / 60.0
    else:
        rec["returncode"] = None
        rec["duration_min"] = float("nan")
    map_path = Path(map_path)
    rec["map_mb"] = map_path.stat().st_size / (1024 * 1024) if map_path.exists() else 0.0
    try:
        with xr.open_dataset(map_path, decode_times=False) as ds:
            zs_last = ds["zs"].isel(time=-1).values
            rec["zs_finite_frac"] = float(np.isfinite(zs_last).mean())
            rec["zs_max_last_m"] = float(np.nanmax(zs_last)) if np.any(np.isfinite(zs_last)) else float("nan")
            rec["n_timesteps"] = int(ds.sizes.get("time", 0))
    except Exception as exc:
        rec["zs_finite_frac"] = float("nan")
        rec["zs_max_last_m"] = float("nan")
        rec["n_timesteps"] = 0
        rec["open_error"] = str(exc)[:120]
    return rec


def health_check_table(runs: pd.DataFrame, *, workers=None) -> pd.DataFrame:
    """Per-run completion health across scenarios, reading maps in parallel.

    For every completed run (a row of :func:`completed_runs_by_scenario`) record the solver
    ``returncode``, map file size, final-timestep ``zs`` finite fraction, timestep count, and
    wall-clock duration. Each check opens one NetCDF, so for hundreds of runs the serial loop
    is I/O bound; this fans the opens across processes (default: up to 16 / ``os.cpu_count()``;
    ``workers=1`` forces serial). Returns a DataFrame (one row per run).
    """
    items = list(runs[["design_scenario", "event_id", "run_dir", "map_path"]].itertuples(index=False, name=None))
    if not items:
        return pd.DataFrame()
    if workers is None:
        workers = min(os.cpu_count() or 1, 16)
    if workers <= 1 or len(items) == 1:
        return pd.DataFrame([_health_check_one(it) for it in items])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_health_check_one, items, chunksize=8))
    return pd.DataFrame(rows)


def select_benchmark_events(
    catalogue,
    return_periods=SLR_BENCHMARK_RETURN_PERIODS,
    *,
    rp_column: str = "sample_rp_years",
    rank_column: str = "peak_incremental_land_depth_m",
    rank_ascending: bool = False,
    id_column: str = "event_id",
    eligible_ids=None,
    log_band: float = 0.35,
) -> pd.DataFrame:
    """Representative design event near each benchmark return period.

    Restricts to events within a ``log_band`` neighborhood of each target RP (in log
    space), then ranks by ``rank_column`` to choose the representative event. The default
    ``peak_incremental_land_depth_m`` (descending) selects the **most consequential** event
    near each RP, which is what makes a usable SLR depth-comparison figure — in a
    tail-enriched catalog, ranking by ``probability_weight`` ("most-likely") systematically
    favors mild, near-dry combinations. Pass ``rank_column="probability_weight"`` for the
    most-likely event instead. Falls back to the RP-closest event when ``rank_column`` is
    absent. ``eligible_ids`` (e.g. events completed in every scenario) restricts the pool.
    """
    df = catalogue.copy()
    if eligible_ids is not None:
        df = df[df[id_column].astype(str).isin({str(i) for i in eligible_ids})]
    df[rp_column] = pd.to_numeric(df[rp_column], errors="coerce")
    df = df.dropna(subset=[rp_column, id_column])
    df = df[df[rp_column] > 0].drop_duplicates(id_column)
    if df.empty:
        return pd.DataFrame(columns=["return_period_years", "event_id", "sample_rp_years"])
    rows = []
    for rp in return_periods:
        dist = (np.log(df[rp_column]) - np.log(float(rp))).abs()
        scored = df.assign(_dist=dist).sort_values("_dist")
        band = scored[scored["_dist"] <= float(log_band)]
        if not band.empty and rank_column in band and band[rank_column].notna().any():
            pick = band.sort_values(rank_column, ascending=rank_ascending).iloc[0]
        else:
            pick = scored.iloc[0]
        rows.append(
            {
                "return_period_years": float(rp),
                "event_id": str(pick[id_column]),
                "sample_rp_years": float(pick[rp_column]),
                "rank_value": float(pd.to_numeric(pd.Series([pick.get(rank_column)]), errors="coerce").iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def slr_event_depth_comparison(
    event_id,
    scenario_roots,
    *,
    offsets_m=None,
    base_key: str = "base",
    return_period_years: float = float("nan"),
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
    depth_kind: str = "incremental",
) -> SlrDepthComparison:
    """Load one event's masked flood depth (ft) per scenario and the deltas vs base.

    Deltas treat non-flooded cells as 0 depth so newly-inundated area appears as a
    positive change (mirrors the notebook's incremental-depth convention and Fig. 10).
    ``depth_kind="total"`` plots absolute inundation depth (rises with SLR); the default
    ``"incremental"`` plots storm depth above the antecedent level.
    """
    offsets_m = offsets_m or {}
    depth, coords = {}, None
    for key, root in scenario_roots.items():
        map_path = Path(root) / str(event_id) / "sfincs_map.nc"
        if not map_path.exists():
            continue
        data = masked_sfincs_depth(map_path, huthresh_m=huthresh_m, land_min_elev_m=land_min_elev_m, depth_kind=depth_kind)
        depth[key] = np.asarray(data["depth_ft"], dtype=float)
        if coords is None:
            coords = (np.asarray(data["x"], dtype=float), np.asarray(data["y"], dtype=float))
    if base_key not in depth:
        raise ValueError(f"base scenario '{base_key}' has no completed map for event {event_id}")
    base = np.nan_to_num(depth[base_key], nan=0.0)
    delta = {}
    for key, d in depth.items():
        if key == base_key:
            continue
        scn = np.nan_to_num(d, nan=0.0)
        union_wet = (scn > 0) | (base > 0)
        delta[key] = np.where(union_wet, scn - base, np.nan)
    ordered = [base_key] + [k for k in depth if k != base_key]
    x, y = coords
    return SlrDepthComparison(
        event_id=str(event_id),
        return_period_years=float(return_period_years),
        scenarios=tuple(ordered),
        offsets_m={k: float(offsets_m.get(k, float("nan"))) for k in ordered},
        x=x,
        y=y,
        depth_ft=depth,
        delta_ft=delta,
    )

def __getattr__(name):
    if name in _PLOTTING_EXPORTS:
        from sfincs_runs import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
