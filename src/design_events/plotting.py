from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

from .fit_history.extreme_value import (
    fit_distribution,
    get_frozen_dist,
    observed_return_periods,
    score_fit,
)
from .build_events.selection import assign_severity_bands

def _finish(fig):
    fig.tight_layout()
    return fig


def _axis(ax, figsize):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return None, ax


def _finish_created(fig):
    if fig is not None:
        fig.tight_layout()
    return fig


# Stage 1.1: SFINCS offshore boundary + chosen CORA snap node centroid.
def plot_boundary_and_node(paths, config):
    crs = config["collection"]["cora"].get("boundary_points_crs", "EPSG:26919")
    boundary = pd.read_csv(paths["sfincs_boundary_file"], sep=r"\s+", header=None, usecols=[0, 1])
    boundary = gpd.GeoDataFrame(
        boundary,
        geometry=gpd.points_from_xy(boundary[0], boundary[1]),
        crs=crs,
    )
    centroid = gpd.GeoSeries([boundary.geometry.union_all().centroid], crs=crs).to_crs(4326).iloc[0]
    boundary = boundary.to_crs(4326)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(boundary.geometry.x, boundary.geometry.y, "-o", color="steelblue", lw=2, ms=4, label="SFINCS boundary nodes")
    ax.scatter(centroid.x, centroid.y, color="crimson", s=120, marker="*", zorder=3,
               label=f"boundary centroid ({centroid.x:.4f}, {centroid.y:.4f})")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Stage 1.1 — SFINCS boundary and CORA snap target")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 2.1: raw hourly water level: sample window + full distribution.
def plot_raw_waterlevel(waterlevel, window_slice=("2018-01-01", "2018-03-31")):
    window = waterlevel.loc[window_slice[0]:window_slice[1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(window.index, window.values, lw=1.0)
    axes[0].set_title(f"Raw CORA hourly water level ({window_slice[0]} to {window_slice[1]})")
    axes[0].set_ylabel("water level [m+MSL]")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(waterlevel.values, bins=60, color="0.4", alpha=0.85)
    axes[1].set_title(f"Full record (n={len(waterlevel):,})")
    axes[1].set_xlabel("water level [m+MSL]")
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

# Stage 2.2: linear MSL trend used to detrend hourly water level to a reference epoch.
def plot_detrending(waterlevel, detrend_meta):
    series = waterlevel.dropna().sort_index()
    annual = series.resample("YS").mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(annual.index.year, annual.values, "o-", color="0.3", label="CORA annual mean")
    if detrend_meta.get("applied"):
        slope = float(detrend_meta["slope_m_per_year"])
        ref_year = float(detrend_meta["reference_epoch_year"])
        years = annual.index.year.to_numpy(dtype=float)
        ref_mask = annual.index.year == int(ref_year)
        anchor = float(annual[ref_mask].mean()) if ref_mask.any() else float(annual.mean())
        ax.plot(years, anchor + slope * (years - ref_year), "--", color="crimson", lw=2,
                label=f"linear trend = {slope*1000:+.2f} mm/yr (ref={ref_year:.0f})")
        ax.axvline(ref_year, color="black", ls=":", lw=1, label=f"reference epoch = {ref_year:.0f}")
        source = detrend_meta.get("slope_source", "")
        ax.text(0.02, 0.95, f"slope source: {source}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.text(0.02, 0.95, "detrending disabled",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("year")
    ax.set_ylabel("annual-mean water level [m+MSL]")
    ax.set_title("Detrending: secular MSL trend at the CORA boundary node")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 2.3: POT thresholding on a sample window + magnitude histogram.
def plot_pot_extraction(waterlevel, peaks, threshold_m,
                        window_slice=("2018-01-01", "2018-03-31")):
    window = waterlevel.loc[window_slice[0]:window_slice[1]]
    peak_window = peaks.loc[window_slice[0]:window_slice[1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(window.index, window.values, lw=1.0, label="raw hourly")
    axes[0].scatter(peak_window.index, peak_window.values, color="crimson", s=22,
                    zorder=3, label="POT peaks")
    axes[0].axhline(threshold_m, color="black", ls="--", lw=1.0,
                    label=f"threshold = {threshold_m:.2f} m")
    axes[0].set_ylabel("water level [m+MSL]")
    axes[0].set_title("POT extraction (sample window)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].hist(peaks.dropna().values, bins=35, color="steelblue", alpha=0.85)
    axes[1].axvline(threshold_m, color="black", ls="--", lw=1.0)
    axes[1].set_title(f"Historical peak magnitudes (n={peaks.dropna().size})")
    axes[1].set_xlabel("peak [m+MSL_ref]")
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

# Stage 2.4: candidate distributions (Exp vs GPD) with AIC scores. Use the
# same fitting and scoring path as production so the figure cannot disagree
# with the selected marginal.
def plot_aic_model_selection(peaks, marginal):
    values = peaks.dropna().to_numpy(dtype=float)
    sorted_vals = np.sort(values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    # Left: empirical CDF vs fitted CDFs, with AIC printed in the legend.
    axes[0].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)),
                 "o", color="0.3", alpha=0.7, label=f"empirical CDF ({len(sorted_vals):,} peaks)")
    grid = np.linspace(values.min(), values.max() * 1.05, 400)
    fits = {}
    for dist in ("exp", "gpd"):
        params = fit_distribution(values, dist)
        aic = score_fit(values, params, dist, "AIC")
        fits[dist] = (params, aic)
    chosen_dist = min(fits, key=lambda key: fits[key][1])
    for dist in ("exp", "gpd"):
        params, aic = fits[dist]
        chosen = (dist == chosen_dist)
        axes[0].plot(grid, get_frozen_dist(params, dist).cdf(grid),
                     "-" if chosen else "--", lw=2.2 if chosen else 1.3,
                     color="crimson" if chosen else "steelblue",
                     label=f"{dist.upper()} fit (AIC={aic:.1f}{'  ←AIC pick' if chosen else ''})")
    axes[0].set_xlabel("peak height h [m+MSL_ref]")
    axes[0].set_ylabel("F(h) = P(peak ≤ h)")
    axes[0].set_title("Which distribution best matches the historical peaks?")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: log-survival diagnostic. The Exponential is a straight line on this scale;
    # empirical points on the line are evidence that no heavier-tailed (GPD) model is needed.
    n = len(sorted_vals)
    emp_surv = 1.0 - (np.arange(1, n + 1) - 0.5) / n
    axes[1].semilogy(sorted_vals, emp_surv, "o", color="0.3", alpha=0.7,
                     label="empirical P(peak > h)")
    for dist in ("exp", "gpd"):
        params, _ = fits[dist]
        chosen = (dist == chosen_dist)
        axes[1].semilogy(grid, get_frozen_dist(params, dist).sf(grid),
                         "-" if chosen else "--", lw=2.2 if chosen else 1.3,
                         color="crimson" if chosen else "steelblue",
                         label=f"{dist.upper()} fit{' ←AIC pick' if chosen else ''}")
    axes[1].set_xlabel("peak height h [m+MSL_ref]")
    axes[1].set_ylabel("P(peak > h)  (log scale)")
    axes[1].set_title("Tail diagnostic — Exp = straight line on log-survival axis")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")
    return _finish(fig)

# Stage 2.5: the 3-step conversion chain that turns a fitted distribution
# into a return-period curve. Each panel is one substitution:
# per-peak probability: expected exceedances per year:  return period.
# The 100-yr peak is highlighted in every panel so a reviewer can trace it
# horizontally across the figure.
def plot_return_curve_with_ci(peaks, marginal, bootstrap, rps=None,
                              highlight_rp=100.0):
    if rps is None:
        rps = np.array([2, 5, 10, 25, 50, 100, 250, 500])
    values = peaks.dropna().to_numpy(dtype=float)
    rate = float(marginal.extremes_rate)
    h_grid = np.linspace(values.min(), values.max() * 1.18, 500)
    rp_at_h = marginal.return_period(h_grid)
    annual_at_h = 1.0 / rp_at_h
    surv_at_h = annual_at_h / rate
    h_star = float(marginal.magnitude(highlight_rp))
    p_star = 1.0 / (rate * highlight_rp)
    rate_star = 1.0 / highlight_rp
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    # Step 1: per-peak survival probability from the fitted Exponential.
    ax = axes[0]
    ax.semilogy(h_grid, surv_at_h, "-", color="crimson", lw=2)
    ax.axvline(h_star, ls=":", color="black")
    ax.axhline(p_star, ls=":", color="black")
    ax.plot([h_star], [p_star], "o", color="black", ms=7, zorder=5)
    ax.set_xlabel("peak height h [m+MSL_ref]")
    ax.set_ylabel("P(peak > h)  per peak  (log)")
    ax.set_title("Step 1 — Exp gives per-peak probability")
    ax.text(0.04, 0.06,
            f"P(peak > {h_star:.2f}) = 1 / {1/p_star:,.0f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.grid(True, alpha=0.3, which="both")
    # Step 2: multiply by the peaks-per-year rate.
    ax = axes[1]
    ax.semilogy(h_grid, annual_at_h, "-", color="crimson", lw=2)
    ax.axvline(h_star, ls=":", color="black")
    ax.axhline(rate_star, ls=":", color="black")
    ax.plot([h_star], [rate_star], "o", color="black", ms=7, zorder=5)
    ax.set_xlabel("peak height h [m+MSL_ref]")
    ax.set_ylabel("expected exceedances / year  (log)")
    ax.set_title(f"Step 2 — × {rate:.1f} peaks / year")
    ax.text(0.04, 0.06,
            f"{rate:.1f} × P(peak > {h_star:.2f})\n= {rate_star:.4f} / yr",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.grid(True, alpha=0.3, which="both")
    # Step 3: invert the annual rate into a return period; overlay the
    # historical peaks and the bootstrap confidence band on the inverse view.
    ax = axes[2]
    ax.plot(observed_return_periods(values, rate), np.sort(values),
            "o", color="0.3", alpha=0.55, label="historical peaks")
    ax.plot(rps, marginal.magnitude(rps), "-", color="crimson", lw=2,
            label=f"{marginal.dist_name.upper()} fit")
    if bootstrap is not None:
        cl_pct = int(round(100 * bootstrap["confidence_level"]))
        ax.fill_between(bootstrap["rps"], bootstrap["lo"], bootstrap["hi"],
                        color="crimson", alpha=0.15,
                        label=f"{cl_pct}% bootstrap CI ({bootstrap['n_succeeded']} reps)")
    ax.axvline(highlight_rp, ls=":", color="black")
    ax.axhline(h_star, ls=":", color="black")
    ax.plot([highlight_rp], [h_star], "o", color="black", ms=8, zorder=5)
    ax.annotate(f"{int(highlight_rp)}-yr peak = {h_star:.2f} m",
                xy=(highlight_rp, h_star),
                xytext=(highlight_rp * 0.05, h_star + 0.07),
                fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("return period [yr]  (log)")
    ax.set_ylabel("peak height h [m+MSL_ref]")
    ax.set_title("Step 3 — RP = 1 / annual rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(
        f"From fitted distribution to return-period curve  "
        f"(per-peak P  →  × {rate:.1f}/yr  →  invert)",
        y=1.02, fontsize=11,
    )
    return _finish(fig)

# Stage 2.6: peak series + Theil-Sen line + Mann-Kendall p-value annotation.
def plot_stationarity(peaks, report):
    series = peaks.dropna()
    t = (series.index - series.index.min()) / pd.Timedelta(days=365.25)
    t = np.asarray(t, dtype=float)
    y = series.to_numpy(dtype=float)
    ts = stats.theilslopes(y, t, 0.95)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(series.index, y, "o", color="0.3", alpha=0.7, label="peaks")
    line_t = np.array([t.min(), t.max()])
    ax.plot([series.index.min(), series.index.max()],
            ts.intercept + ts.slope * line_t, "-", color="crimson", lw=2,
            label=f"Theil-Sen = {ts.slope*1000:+.2f} mm/yr "
                  f"[{ts.low_slope*1000:+.2f}, {ts.high_slope*1000:+.2f}]")
    mk_p = float(report.get("mann_kendall_p", float("nan")))
    note = "stationary" if mk_p >= 0.05 else "trend detected"
    ax.set_title(f"Stationarity diagnostic: Mann-Kendall p={mk_p:.3f} ({note})")
    ax.set_ylabel("peak [m+MSL_ref]")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 3.1: hybrid splice between empirical body and parametric tail.
def plot_hybrid_splice(historical_peaks, sampled_peaks, splice_q):
    historical_peaks = np.asarray(historical_peaks, dtype=float)
    splice_peak = float(np.quantile(historical_peaks, splice_q))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(historical_peaks, bins=35, alpha=0.65, density=True,
                 label=f"historical (n={len(historical_peaks)})", color="steelblue")
    axes[0].hist(sampled_peaks["peak_m"].to_numpy(), bins=60, alpha=0.55, density=True,
                 label=f"synthetic targets (n={len(sampled_peaks):,})", color="orange")
    axes[0].axvline(splice_peak, color="black", ls="--",
                    label=f"splice q={splice_q:.2f} → {splice_peak:.2f} m")
    axes[0].set_xlabel("peak [m+MSL_ref]")
    axes[0].set_ylabel("density")
    axes[0].set_title("Stage 3.1 — Hybrid splice: empirical body ↔ parametric tail")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    rp_sorted = sampled_peaks["sample_rp_years"].sort_values().to_numpy()
    axes[1].plot(np.arange(1, len(rp_sorted) + 1), rp_sorted)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sample rank")
    axes[1].set_ylabel("coastal driver return period [yr]")
    axes[1].set_title("Synthetic coastal-driver RP coverage (log)")
    axes[1].grid(True, alpha=0.3, which="both")
    return _finish(fig)

def _annual_chance_label(return_period_years):
    aep = 100.0 / float(return_period_years)
    return f"{aep:g}% annual chance"

def _return_period_axis_context(catalog):
    columns = set(catalog.columns)
    if columns & {"basis_site_no", "peak_flow_cfs", "streamflow_member_id", "streamflow_source"}:
        return {
            "kind": "streamflow",
            "axis_label": "streamgage-network return period [years]",
            "title_label": "streamgage-network benchmark coverage",
            "joint_label": "Streamgage-network return period",
        }
    if columns & {"coastal_peak_m", "coastal_member_id", "coastal_analog_id", "coastal_source"}:
        return {
            "kind": "coastal",
            "axis_label": "coastal driver return period [years]",
            "title_label": "coastal-driver benchmark coverage",
            "joint_label": "Coastal driver return period",
        }
    return {
        "kind": "generic",
        "axis_label": "event-driver return period [years]",
        "title_label": "event-driver benchmark coverage",
        "joint_label": "Event-driver return period",
    }

def _short_event_label(value, *, max_length=28):
    text = "" if pd.isna(value) else str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"

def _catalog_display_label(catalog, *, default="Catalog"):
    if catalog is None or "catalog_role" not in catalog:
        return default
    roles = {
        str(role).strip().lower()
        for role in catalog["catalog_role"].dropna().unique()
        if str(role).strip()
    }
    if roles == {"design"}:
        return "Selected Design Catalog"
    if roles == {"probability"}:
        return "Probability Catalog"
    if "design" in roles:
        return "Design Catalog"
    if "probability" in roles:
        return "Probability Catalog"
    return default

def _candidate_pool_band_counts(catalog, bands):
    counts = pd.Series(0.0, index=list(bands), dtype=float)
    if catalog is None or "severity_band" not in catalog:
        return counts, 0
    keyed = catalog.copy()
    keyed["severity_band"] = keyed["severity_band"].astype(str)
    if "pool_band_support" in keyed:
        counts = (
            pd.to_numeric(keyed["pool_band_support"], errors="coerce")
            .groupby(keyed["severity_band"])
            .max()
            .reindex(bands)
            .fillna(0.0)
        )
    elif {"pool_band_probability", "candidate_pool_count"}.issubset(keyed.columns):
        pool_count = pd.to_numeric(keyed["candidate_pool_count"], errors="coerce").dropna()
        total = float(pool_count.iloc[0]) if len(pool_count) else float(len(keyed))
        probabilities = (
            pd.to_numeric(keyed["pool_band_probability"], errors="coerce")
            .groupby(keyed["severity_band"])
            .max()
            .reindex(bands)
            .fillna(0.0)
        )
        counts = (probabilities * total).round()
    else:
        counts = keyed["severity_band"].value_counts().reindex(bands).fillna(0.0)

    total_column = pd.to_numeric(catalog.get("candidate_pool_count", pd.Series(dtype=float)), errors="coerce").dropna()
    total = int(round(float(total_column.iloc[0]))) if len(total_column) else int(round(float(counts.sum())))
    return counts.astype(float), total

def _format_return_period_label(value):
    value = float(value)
    number = f"{int(round(value))}" if value.is_integer() else f"{value:g}"
    return f"{number}-yr"

def nearest_benchmark_events(catalog, *, benchmarks=None):
    benchmarks = benchmarks or [10, 50, 100, 500]
    frame = catalog.copy()
    frame["sample_rp_years"] = pd.to_numeric(frame["sample_rp_years"], errors="coerce")
    valid = frame.dropna(subset=["sample_rp_years"]).copy()
    valid = valid[valid["sample_rp_years"] > 0]
    if valid.empty:
        return pd.DataFrame()
    log_rp = np.log(valid["sample_rp_years"])
    rows = []
    for benchmark in benchmarks:
        benchmark = float(benchmark)
        index = (log_rp - np.log(benchmark)).abs().idxmin()
        row = valid.loc[index]
        rows.append(
            {
                "benchmark_return_period_years": int(benchmark) if benchmark.is_integer() else benchmark,
                "annual_chance_label": _annual_chance_label(benchmark),
                "event_id": row.get("event_id", pd.NA),
                "sample_rp_years": float(row["sample_rp_years"]),
                "severity_band": row.get("severity_band", pd.NA),
                "probability_weight": row.get("probability_weight", pd.NA),
                "coastal_peak_m": row.get("coastal_peak_m", row.get("peak_m", pd.NA)),
                "basis_site_no": row.get("basis_site_no", pd.NA),
                "peak_flow_cfs": row.get("peak_flow_cfs", pd.NA),
                "absolute_log_error": float(abs(np.log(row["sample_rp_years"]) - np.log(benchmark))),
            }
        )
    return pd.DataFrame(rows)

def _compound_lag_frame(catalog):
    lag_column = next(
        (column for column in ["rainfall_peak_offset_hours", "rainfall_pairing_lag_hours"] if column in catalog),
        None,
    )
    has_lag_column = lag_column is not None
    if lag_column is None:
        lag_column = "rainfall_peak_offset_hours"
        frame = catalog.assign(**{lag_column: np.nan}).copy()
    else:
        frame = catalog.copy()

    numeric_columns = [
        lag_column,
        "rainfall_start_offset_hours",
        "rainfall_end_offset_hours",
        "coastal_water_level",
        "coastal_peak_m",
        "rainfall_metric_mm",
        "rainfall",
        "sample_rp_years",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, lag_column, has_lag_column

def _compound_lag_driver_panels(frame):
    rainfall_column = next(
        (column for column in ["rainfall_metric_mm", "rainfall"] if column in frame and frame[column].notna().any()),
        "rainfall_metric_mm",
    )
    coastal_column = next(
        (
            column for column in ["coastal_water_level", "coastal_peak_m"]
            if column in frame and frame[column].notna().any()
        ),
        None,
    )
    panels = [
        (rainfall_column, "Rainfall event total (mm)"),
    ]
    if coastal_column is not None:
        panels.insert(0, (coastal_column, "Coastal peak NTR / residual (m)"))
    return [
        (column, label)
        for column, label in panels
        if column in frame and frame[column].notna().any()
    ]

def _first_nonnull(frame, column, default=pd.NA):
    if column not in frame:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]

def compound_lag_diagnostic_summary(catalog):
    frame, lag_column, has_lag_column = _compound_lag_frame(catalog)
    finite = frame[frame[lag_column].notna()].copy()
    driver_panels = _compound_lag_driver_panels(finite)
    timing_columns = {"rainfall_start_offset_hours", "rainfall_end_offset_hours"}
    has_timing_window = timing_columns.issubset(finite.columns) and finite[list(timing_columns)].notna().all(axis=1).any()
    ready = has_lag_column and not finite.empty and len(driver_panels) >= 2
    summary = {
        "diagnostic_status": "ready" if ready else "insufficient_lag_metadata",
        "lag_basis": f"{lag_column}: rainfall peak relative to coastal peak",
        "plot_design": "intensity-vs-lag scatter panels plus rainfall timing strips",
        "finite_lag_rows": len(finite),
        "available_driver_panels": len(driver_panels),
        "has_rainfall_window_offsets": bool(has_timing_window),
    }
    if not finite.empty:
        summary.update({
            "median_peak_lag_hours": float(finite[lag_column].median()),
            "lag_5th_percentile_hours": float(finite[lag_column].quantile(0.05)),
            "lag_95th_percentile_hours": float(finite[lag_column].quantile(0.95)),
        })
    return pd.Series(summary, name="compound_lag_diagnostic")

def compound_lag_analog_reuse_summary(catalog):
    frame, lag_column, has_lag_column = _compound_lag_frame(catalog)
    index_column = "empirical_lag_analog_index"
    if index_column not in frame or frame[index_column].dropna().empty:
        return pd.Series(
            {
                "diagnostic_status": "missing_empirical_lag_analog_index",
                "finite_lag_rows": int(frame[lag_column].notna().sum()) if has_lag_column else 0,
            },
            name="compound_lag_analog_reuse",
        )

    analog = frame[index_column].dropna()
    counts = analog.astype(str).value_counts()
    total = int(counts.sum())
    ess = float(total * total / np.square(counts.to_numpy(dtype=float)).sum()) if total else np.nan
    top_key = counts.index[0] if len(counts) else None
    top_rows = frame[frame[index_column].astype(str).eq(str(top_key))] if top_key is not None else pd.DataFrame()
    summary = {
        "diagnostic_status": "ready",
        "pairing_policy": _first_nonnull(frame, "compound_pairing_policy"),
        "finite_lag_rows": int(frame[lag_column].notna().sum()) if has_lag_column else 0,
        "unique_lag_analogues": int(len(counts)),
        "effective_lag_analogues_ess": ess,
        "ess_fraction_of_rows": float(ess / total) if total else np.nan,
        "max_analogue_reuse_count": int(counts.iloc[0]) if len(counts) else 0,
        "max_analogue_reuse_fraction": float(counts.iloc[0] / total) if total and len(counts) else np.nan,
        "dominant_analogue_index": top_key,
    }
    if not top_rows.empty:
        summary["dominant_analogue_event_time"] = _first_nonnull(top_rows, "empirical_lag_analog_event_time")
        summary["dominant_analogue_storm_type"] = _first_nonnull(top_rows, "empirical_lag_analog_storm_type")
        summary["dominant_analogue_lag_hours"] = float(pd.to_numeric(top_rows[lag_column], errors="coerce").median())
    return pd.Series(summary, name="compound_lag_analog_reuse")

def plot_compound_lag_sampling_diagnostic(catalog, paired_observations=None, *, top_n=15):
    frame, lag_column, has_lag_column = _compound_lag_frame(catalog)
    if not has_lag_column or frame[lag_column].dropna().empty:
        return None

    sampled_lag = pd.to_numeric(frame[lag_column], errors="coerce").dropna().to_numpy(dtype=float)
    observed_lag = np.array([], dtype=float)
    if paired_observations is not None:
        from design_events.build_events.compound_timing import observed_compound_lag_pool

        pool = observed_compound_lag_pool(paired_observations)
        if not pool.empty:
            observed_lag = pd.to_numeric(pool["observed_lag_hours"], errors="coerce").dropna().to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    axis = axes[0]
    if observed_lag.size:
        x = np.sort(observed_lag)
        y = np.arange(1, len(x) + 1) / len(x)
        axis.step(x, y, where="post", color="0.35", linewidth=2.0, label=f"observed pool (n={len(x)})")
    x = np.sort(sampled_lag)
    y = np.arange(1, len(x) + 1) / len(x)
    axis.step(x, y, where="post", color="tab:blue", linewidth=2.0, label=f"sampled catalogue (n={len(x)})")
    axis.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Rainfall peak lag from coastal NTR peak (hours)")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("Observed vs sampled compound lag")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)

    axis = axes[1]
    if "empirical_lag_analog_index" in frame and frame["empirical_lag_analog_index"].notna().any():
        counts = frame["empirical_lag_analog_index"].dropna().astype(str).value_counts().head(int(top_n)).sort_values()
        axis.barh(counts.index, counts.to_numpy(dtype=float), color="tab:blue", alpha=0.82)
        total = int(frame["empirical_lag_analog_index"].notna().sum())
        ess_summary = compound_lag_analog_reuse_summary(frame)
        ess = float(ess_summary.get("effective_lag_analogues_ess", np.nan))
        max_fraction = float(ess_summary.get("max_analogue_reuse_fraction", np.nan))
        title = "Lag analogue reuse"
        if np.isfinite(ess) and np.isfinite(max_fraction):
            title = f"{title} (ESS={ess:.1f}, max={100 * max_fraction:.1f}%)"
        axis.set_title(title)
        axis.set_xlabel("Catalogue rows using analogue")
        axis.set_ylabel("Observed analogue index")
        axis.grid(True, axis="x", alpha=0.3)
        if total:
            axis.axvline(0.05 * total, color="0.35", linestyle=":", linewidth=1.2, label="5% of rows")
            axis.legend(loc="lower right", fontsize=8)
    else:
        axis.text(0.5, 0.5, "lag analogue reuse\nmetadata not available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()

    fig.suptitle("Conditional empirical lag analogue diagnostic", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig

def plot_compound_lag_diagnostic(catalog):
    frame, lag_column, has_lag_column = _compound_lag_frame(catalog)
    frame = frame[frame[lag_column].notna()].copy()
    driver_panels = _compound_lag_driver_panels(frame)
    if not has_lag_column or frame.empty or len(driver_panels) < 2:
        return None

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [1, 1, 1.3]},
    )
    role_column = (
        "compound_pairing_role"
        if "compound_pairing_role" in frame and frame["compound_pairing_role"].notna().any()
        else None
    )
    roles = sorted(frame[role_column].dropna().astype(str).unique()) if role_column is not None else []
    base_palette = plt.cm.tab10.colors
    palette = {role: base_palette[index % len(base_palette)] for index, role in enumerate(roles)}

    for axis, (driver_column, driver_label) in zip(axes[:2], driver_panels):
        if role_column is None:
            axis.scatter(
                frame[driver_column], frame[lag_column],
                s=24, alpha=0.72, edgecolor="white", linewidth=0.35,
            )
        else:
            for role in roles:
                role_rows = frame[frame[role_column].astype(str).eq(role)]
                axis.scatter(
                    role_rows[driver_column], role_rows[lag_column],
                    s=24, alpha=0.72, edgecolor="white", linewidth=0.35,
                    label=role, color=palette[role],
                )
        axis.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        axis.set_xlabel(driver_label)
        axis.set_ylabel("Rainfall peak lag from coastal peak (hours)")
        axis.set_title(driver_label.split(" (")[0])

    timing_axis = axes[2]
    if {"rainfall_start_offset_hours", "rainfall_end_offset_hours"}.issubset(frame.columns):
        timing_rows = frame.dropna(subset=["rainfall_start_offset_hours", lag_column, "rainfall_end_offset_hours"]).copy()
    else:
        timing_rows = pd.DataFrame()

    if not timing_rows.empty:
        timing_rows = timing_rows.sort_values(lag_column)
        if len(timing_rows) > 11:
            pick_positions = np.linspace(0, len(timing_rows) - 1, 11).round().astype(int)
            timing_rows = timing_rows.iloc[pick_positions].drop_duplicates(subset=["event_id"])
        y_positions = np.arange(len(timing_rows))
        timing_axis.hlines(
            y_positions,
            timing_rows["rainfall_start_offset_hours"],
            timing_rows["rainfall_end_offset_hours"],
            color="0.55", linewidth=2.2,
        )
        timing_axis.scatter(timing_rows[lag_column], y_positions, color="tab:blue", s=28, zorder=3)
        timing_axis.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        timing_axis.set_yticks(y_positions)
        timing_axis.set_yticklabels(timing_rows["event_id"].astype(str), fontsize=8)
        timing_axis.set_xlabel("Hours from coastal peak")
        timing_axis.set_title("Representative rainfall windows")
    else:
        timing_axis.text(
            0.5, 0.5,
            "rainfall start/end offsets\nnot available",
            ha="center", va="center", transform=timing_axis.transAxes,
        )
        timing_axis.set_axis_off()

    fig.suptitle("Compound rainfall-coastal lag diagnostic", fontsize=13)
    if roles:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Pairing role",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 5),
            fontsize=8,
            title_fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=(0.0, 0.18 if roles else 0.0, 1.0, 0.94))
    return fig

def plot_return_period_benchmark_coverage(catalog, stress_catalog=None, *, benchmarks=None):
    benchmarks = benchmarks or [10, 50, 100, 500]
    axis = _return_period_axis_context(catalog)
    catalog_label = _catalog_display_label(catalog, default="Catalog")
    frame = catalog.copy()
    frame["sample_rp_years"] = pd.to_numeric(frame["sample_rp_years"], errors="coerce")
    rp = frame["sample_rp_years"].replace([np.inf, -np.inf], np.nan).dropna()
    rp = rp[rp > 0]
    nearest = nearest_benchmark_events(frame, benchmarks=benchmarks)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    if len(rp):
        lower = max(0.01, float(rp.min()))
        upper = max(float(rp.max()), max(benchmarks))
        bins = np.geomspace(lower, upper, 32)
        axes[0].hist(rp, bins=bins, color="#4c78a8", alpha=0.70, label=catalog_label)
        if stress_catalog is not None and "sample_rp_years" in stress_catalog:
            stress_rp = pd.to_numeric(stress_catalog["sample_rp_years"], errors="coerce").dropna()
            stress_rp = stress_rp[stress_rp > 0]
            if len(stress_rp):
                y = np.full(len(stress_rp), max(1, len(rp) * 0.004))
                axes[0].scatter(stress_rp, y, marker="|", s=45, color="#d62728", alpha=0.65,
                                label="Stress/Training Set")
        axes[0].set_xscale("log")
        axes[0].set_xlim(lower, upper * 1.05)
    ymax = axes[0].get_ylim()[1] or 1
    for benchmark in benchmarks:
        axes[0].axvline(benchmark, color="black", ls=":", lw=1)
        axes[0].text(
            benchmark,
            ymax * 0.92,
            f"{int(benchmark)}-yr\n{_annual_chance_label(benchmark).split()[0]}",
            ha="center",
            va="top",
            fontsize=8,
            rotation=0,
        )
    axes[0].set_xlabel(axis["axis_label"])
    axes[0].set_ylabel("event count")
    axes[0].set_title(f"10/50/100/500-year {axis['title_label']}")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[1].axis("off")
    if not nearest.empty:
        table_columns = [
            "benchmark_return_period_years",
            "annual_chance_label",
            "event_id",
        ]
        labels = ["target RP", "AEP", "nearest event"]
        col_widths = [0.15, 0.18, 0.27]
        if axis["kind"] == "streamflow":
            table_columns.extend(["basis_site_no", "peak_flow_cfs"])
            labels.extend(["basis gage", "peak cfs"])
            col_widths.extend([0.14, 0.13])
        elif axis["kind"] == "coastal":
            table_columns.append("coastal_peak_m")
            labels.append("peak m")
            col_widths.append(0.12)
        table_columns.extend(["sample_rp_years", "severity_band"])
        labels.extend(["event RP", "band"])
        col_widths.extend([0.12, 0.14])
        table = nearest[table_columns].copy()
        table["benchmark_return_period_years"] = table["benchmark_return_period_years"].map(_format_return_period_label)
        table["annual_chance_label"] = table["annual_chance_label"].str.replace(" annual chance", " AEP", regex=False)
        table["event_id"] = table["event_id"].map(_short_event_label)
        table["sample_rp_years"] = table["sample_rp_years"].map(lambda value: f"{value:.1f}")
        if "peak_flow_cfs" in table:
            table["peak_flow_cfs"] = pd.to_numeric(table["peak_flow_cfs"], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )
        if "coastal_peak_m" in table:
            table["coastal_peak_m"] = pd.to_numeric(table["coastal_peak_m"], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
        cell_text = table.to_numpy().tolist()
        rendered = axes[1].table(cellText=cell_text, colLabels=labels, colWidths=col_widths, loc="center")
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8)
        rendered.scale(1, 1.45)
    axes[1].set_title("Nearest catalog rows used for benchmark slices")
    return _finish(fig)

# Stage 4.1: normalized historical templates + shape-diversity scatter.
def plot_template_bank(template_frame, n_show=8):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sample = template_frame.sample(min(n_show, len(template_frame)), random_state=0)
    for _, row in sample.iterrows():
        axes[0].plot(row["surge_template"], alpha=0.7)
    axes[0].set_xlabel("relative-hour index")
    axes[0].set_ylabel("normalized surge anomaly")
    axes[0].set_title(f"Stage 4.1 — {len(sample)} normalized historical templates")
    axes[0].grid(True, alpha=0.3)
    sc = axes[1].scatter(template_frame["peak_m"],
                         template_frame["duration_above_50pct_peak"],
                         c=template_frame["asymmetry_ratio"], cmap="viridis",
                         s=18, alpha=0.85)
    plt.colorbar(sc, ax=axes[1], label="asymmetry (fall/rise)")
    axes[1].set_xlabel("template peak [m]")
    axes[1].set_ylabel("duration > 50% peak [hr]")
    axes[1].set_title(f"Template diversity (n={len(template_frame)})")
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

# Stage 4.2: tail-morph time-stretch as a function of target peak magnitude.
def plot_tail_morph(historical_peaks, settings):
    historical_peaks = np.asarray(historical_peaks, dtype=float)
    historical_peaks = historical_peaks[np.isfinite(historical_peaks)]
    max_peak = float(np.nanmax(historical_peaks))
    trigger_q = float(settings.get("tail_morph_trigger_quantile", 0.95))
    max_factor = float(settings.get("tail_morph_max_factor", 1.30))
    trigger_peak = float(np.nanquantile(historical_peaks, trigger_q))
    denom = max(0.05, max_peak - trigger_peak)
    targets = np.linspace(historical_peaks.min(), max_peak * 1.4, 400)
    factors = np.where(
        targets <= max_peak, 1.0,
        1.0 + np.minimum(max_factor - 1.0,
                         (max_factor - 1.0) * (targets - max_peak) / denom),
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(targets, factors, "-", color="crimson", lw=2)
    ax.axvline(max_peak, ls="--", color="black", label=f"max historical = {max_peak:.2f} m")
    ax.axvline(trigger_peak, ls=":", color="0.3", label=f"trigger q={trigger_q:.2f}")
    ax.axhline(max_factor, ls=":", color="crimson", alpha=0.5,
               label=f"cap = {max_factor:.2f}× duration")
    ax.set_xlabel("target peak [m]")
    ax.set_ylabel("time-stretch factor")
    ax.set_title("Stage 4.2 — Tail-morph factor for out-of-sample peaks")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 4.3: Gaussian kernel weights for one target peak + reuse distribution.
def plot_template_matching(template_frame, summary, target_peak, settings):
    historical_peaks = template_frame["peak_m"].to_numpy(dtype=float)
    pool_size = int(min(settings.get("nearest_pool_size", 75), len(historical_peaks)))
    sigma_scale = float(settings.get("kernel_sigma_scale", 0.50))
    sigma_min = float(settings.get("kernel_sigma_min_m", 0.03))
    sigma_max = float(settings.get("kernel_sigma_max_m", 0.20))
    order = np.argsort(np.abs(historical_peaks - target_peak))
    pool_idx = order[:pool_size]
    pool_peaks = historical_peaks[pool_idx]
    sigma = float(np.clip(sigma_scale * np.nanstd(pool_peaks), sigma_min, sigma_max))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].scatter(historical_peaks, np.zeros_like(historical_peaks),
                    color="0.7", s=10, label=f"all templates ({len(historical_peaks)})")
    axes[0].scatter(pool_peaks, np.full_like(pool_peaks, 0.05),
                    color="steelblue", s=22, label=f"nearest pool ({pool_size})")
    axes[0].axvline(target_peak, color="crimson", ls="--", lw=2,
                    label=f"target = {target_peak:.2f} m")
    grid = np.linspace(historical_peaks.min(), historical_peaks.max(), 300)
    kernel = np.exp(-0.5 * ((grid - target_peak) / sigma) ** 2)
    axes[0].plot(grid, kernel, color="crimson", alpha=0.6,
                 label=f"Gaussian kernel σ={sigma:.3f} m")
    axes[0].set_xlabel("template peak [m]")
    axes[0].set_yticks([])
    axes[0].set_title("Stage 4.3 — Analogue selection for one target peak")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    template_use = summary["template_id"].value_counts()
    axes[1].hist(template_use.values, bins=30, color="steelblue", alpha=0.85)
    axes[1].set_xlabel("times each template was used")
    axes[1].set_ylabel("template count")
    axes[1].set_title(
        f"Reuse distribution (max = {int(template_use.max())} of "
        f"{len(summary):,} events; reuse-penalty λ"
        f"={settings.get('reuse_penalty_lambda', 1.0):.2f})"
    )
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

def _circular_day_gap(a, b):
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.minimum(diff, 366.0 - diff)

def _catalog_time_series(catalog, columns):
    for column in columns:
        if column in catalog:
            return pd.to_datetime(catalog[column], errors="coerce"), column
    return pd.Series(pd.NaT, index=catalog.index, dtype="datetime64[ns]"), None

def _normalize_pairing_strategy(strategy, forcing=None):
    if strategy == "inland_rainfall_pairing_priority":
        return "seasonal_window_permutation"
    if strategy == "inland_antecedent_moisture_pairing":
        return "antecedent_to_forcing"
    return strategy

def _seasonal_pairing_values(catalog, forcing):
    reference_time, reference_column = _catalog_time_series(
        catalog,
        [
            "coastal_template_peak_time",
            "event_reference_time",
            "template_peak_time",
            "event_time",
            "event_date",
        ],
    )
    member_time, _ = _catalog_time_series(catalog, [f"{forcing}_member_time"])
    mask = reference_time.notna() & member_time.notna()
    reference_doy = reference_time[mask].dt.dayofyear.to_numpy(dtype=float)
    member_doy = member_time[mask].dt.dayofyear.to_numpy(dtype=float)
    diff = _circular_day_gap(reference_doy, member_doy)
    member_id = catalog.get(f"{forcing}_member_id")
    if member_id is None:
        member_id = pd.Series(["<missing>"] * len(catalog), index=catalog.index)
    members = member_id.loc[mask].astype(str)
    return reference_doy, member_doy, diff, members, reference_column

def seasonal_pairing_diagnostics(catalog, forcing, *, window_days=None):
    _, member_doy, diff, members, _ = _seasonal_pairing_values(catalog, forcing)
    reuse = members.value_counts()
    window = float(window_days) if window_days is not None else np.nan
    in_window = int((diff <= window).sum()) if np.isfinite(window) else int(len(diff))
    return pd.DataFrame(
        [
            {
                "forcing": forcing,
                "paired_rows": int(len(diff)),
                "unique_members": int(members.nunique()) if len(members) else 0,
                "unique_member_days": int(pd.Series(member_doy).nunique()) if len(member_doy) else 0,
                "in_window_rows": in_window,
                "in_window_fraction": float(in_window / len(diff)) if len(diff) else np.nan,
                "median_gap_days": float(np.nanmedian(diff)) if len(diff) else np.nan,
                "p90_gap_days": float(np.nanquantile(diff, 0.90)) if len(diff) else np.nan,
                "max_gap_days": float(np.nanmax(diff)) if len(diff) else np.nan,
                "max_member_reuse": int(reuse.max()) if len(reuse) else 0,
                "p95_member_reuse": float(reuse.quantile(0.95)) if len(reuse) else np.nan,
            }
        ]
    )

def antecedent_pairing_diagnostics(catalog, forcing):
    member_time = pd.to_datetime(catalog.get(f"{forcing}_member_time"), errors="coerce")
    reference_time = pd.to_datetime(catalog.get(f"{forcing}_pairing_reference_time"), errors="coerce")
    configured_lag = pd.to_numeric(
        catalog.get(f"{forcing}_pairing_lag_hours"),
        errors="coerce",
    )
    mask = member_time.notna() & reference_time.notna()
    lag_hours = (reference_time[mask] - member_time[mask]).dt.total_seconds() / 3600.0
    configured = configured_lag[mask]
    if configured.notna().any():
        expected = float(configured.dropna().median())
        on_lag = int(np.isclose(lag_hours.to_numpy(dtype=float), expected, atol=1e-6).sum())
    else:
        expected = np.nan
        on_lag = 0
    return pd.DataFrame(
        [
            {
                "forcing": forcing,
                "policy": "antecedent_to_forcing",
                "paired_rows": int(mask.sum()),
                "reference_rows": int(reference_time.notna().sum()),
                "configured_lag_hours": expected,
                "median_lag_hours": float(np.nanmedian(lag_hours)) if len(lag_hours) else np.nan,
                "min_lag_hours": float(np.nanmin(lag_hours)) if len(lag_hours) else np.nan,
                "max_lag_hours": float(np.nanmax(lag_hours)) if len(lag_hours) else np.nan,
                "on_lag_rows": on_lag,
                "on_lag_fraction": float(on_lag / len(lag_hours)) if len(lag_hours) else np.nan,
            }
        ]
    )

def forcing_pairing_diagnostics(catalog, forcings=None, policies=None):
    if forcings is None:
        forcings = [
            prefix.removesuffix("_pairing_policy")
            for prefix in catalog.columns
            if prefix.endswith("_pairing_policy")
        ]
    policies = policies or {}
    frames = []
    for forcing in forcings:
        policy = policies.get(forcing, {})
        strategy = policy.get("strategy")
        if strategy is None and f"{forcing}_pairing_policy" in catalog:
            values = catalog[f"{forcing}_pairing_policy"].dropna().astype(str)
            strategy = values.iloc[0] if len(values) else None
        strategy = _normalize_pairing_strategy(strategy, forcing)
        if strategy == "seasonal_window_permutation":
            window = policy.get("window_days")
            if window is None and f"{forcing}_pairing_window_days" in catalog:
                window_values = pd.to_numeric(catalog[f"{forcing}_pairing_window_days"], errors="coerce").dropna()
                window = float(window_values.iloc[0]) if len(window_values) else None
            frame = seasonal_pairing_diagnostics(catalog, forcing, window_days=window)
            frame.insert(1, "policy", strategy)
            frames.append(frame)
        elif strategy == "antecedent_to_forcing":
            frames.append(antecedent_pairing_diagnostics(catalog, forcing))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def wave_analog_diagnostics(catalog):
    policy = catalog.get("snapwave_pairing_policy", pd.Series(dtype=object)).dropna().astype(str)
    policy_label = policy.mode().iloc[0] if len(policy) else "not_configured"
    required = ["coastal_analog_id", "snapwave_member_id", "snapwave_member_file"]
    present = [column for column in required if column in catalog]
    if len(present) != len(required):
        paired = pd.Series(False, index=catalog.index)
    else:
        values = catalog[required].fillna("").astype(str)
        paired = values.ne("").all(axis=1)
    if {"coastal_analog_id", "snapwave_member_id"}.issubset(catalog.columns):
        same_analog = (
            catalog["coastal_analog_id"].fillna("").astype(str)
            == catalog["snapwave_member_id"].fillna("").astype(str)
        ) & paired
    else:
        same_analog = pd.Series(False, index=catalog.index)
    return pd.DataFrame(
        [
            {
                "forcing": "coastal_waves",
                "policy": policy_label,
                "paired_rows": int(paired.sum()),
                "missing_rows": int((~paired).sum()),
                "same_analog_rows": int(same_analog.sum()),
            }
        ]
    )

def plot_antecedent_pairing(catalog, forcing, *, ax=None):
    member_time = pd.to_datetime(catalog.get(f"{forcing}_member_time"), errors="coerce")
    reference_time = pd.to_datetime(catalog.get(f"{forcing}_pairing_reference_time"), errors="coerce")
    lag_hours = (reference_time - member_time).dt.total_seconds() / 3600.0
    lag_hours = lag_hours[np.isfinite(lag_hours)]
    expected = pd.to_numeric(catalog.get(f"{forcing}_pairing_lag_hours"), errors="coerce").dropna()
    expected = float(expected.median()) if len(expected) else np.nan
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    if len(lag_hours):
        ax.hist(lag_hours, bins=min(40, max(5, int(np.sqrt(len(lag_hours))))), color="steelblue", alpha=0.85)
    if np.isfinite(expected):
        ax.axvline(expected, color="crimson", ls="--", lw=2, label=f"configured lag = {expected:g} h")
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("reference time minus member time [hours]")
    ax.set_ylabel("event count")
    ax.set_title(f"Stage 5.2 — Antecedent pairing: {forcing} (n={len(lag_hours):,})")
    ax.grid(True, alpha=0.3, axis="y")
    return _finish(fig)

def plot_configured_pairing(catalog, forcing, *, policy=None, ax=None):
    policy = policy or {}
    strategy = policy.get("strategy")
    if strategy is None and f"{forcing}_pairing_policy" in catalog:
        values = catalog[f"{forcing}_pairing_policy"].dropna().astype(str)
        strategy = values.iloc[0] if len(values) else None
    strategy = _normalize_pairing_strategy(strategy, forcing)
    if strategy == "antecedent_to_forcing":
        return plot_antecedent_pairing(catalog, forcing, ax=ax)
    window = policy.get("window_days")
    if window is None and f"{forcing}_pairing_window_days" in catalog:
        window_values = pd.to_numeric(catalog[f"{forcing}_pairing_window_days"], errors="coerce").dropna()
        window = float(window_values.iloc[0]) if len(window_values) else None
    return plot_seasonal_pairing(catalog, forcing, window_days=window, ax=ax)

def plot_rainfall_member_distribution(members):
    depth_column = next(
        (
            column
            for column in [
                "mean_precip_mm",
                "max_precip_mm",
                "min_precip_mm",
                # Backward-compatible legacy names. AORC APCP values are mm, not inches.
                "mean_precip_in",
                "max_precip_in",
                "min_precip_in",
            ]
            if column in members
        ),
        None,
    )
    if depth_column is None:
        raise KeyError("rainfall members need a precipitation depth column")
    frame = members.copy()
    frame[depth_column] = pd.to_numeric(frame[depth_column], errors="coerce")
    if "rank" in frame:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.sort_values("rank")
        x = frame["rank"]
    else:
        frame = frame.sort_values(depth_column, ascending=False).reset_index(drop=True)
        x = np.arange(1, len(frame) + 1)
    storm_time = pd.to_datetime(frame.get("storm_start"), errors="coerce")
    month_counts = storm_time.dt.month.value_counts().reindex(range(1, 13)).fillna(0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(x, frame[depth_column], "o-", color="steelblue", ms=3, lw=1)
    axes[0].set_xlabel("SST member rank")
    axes[0].set_ylabel(depth_column)
    axes[0].set_title(f"AORC SST rainfall member depths (n={len(frame):,})")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(month_counts.index, month_counts.values, color="0.45", alpha=0.85)
    axes[1].set_xlabel("storm-start month")
    axes[1].set_ylabel("member count")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("AORC SST member seasonality")
    axes[1].grid(True, alpha=0.3, axis="y")
    return _finish(fig)

def _shade_circular_window(ax, window):
    window = float(window)
    if window >= 365:
        ax.axhspan(1, 366, color="steelblue", alpha=0.12, label=f"±{int(window)}-day window")
        return
    x = np.linspace(1, 366, 400)
    ax.fill_between(
        x,
        np.clip(x - window, 1, 366),
        np.clip(x + window, 1, 366),
        color="steelblue",
        alpha=0.12,
        label=f"±{int(window)}-day window",
    )
    left = x[x <= window]
    if len(left):
        ax.fill_between(left, 366 - window + left, 366, color="steelblue", alpha=0.12)
    right = x[x >= 366 - window]
    if len(right):
        ax.fill_between(right, 1, right + window - 366, color="steelblue", alpha=0.12)

# Stage 5.2: seasonal-window pairing diagnostic. Day-of-year (DOY) of each
# catalog reference event vs. DOY of the paired member, with the configured
# circular ±window_days band overlaid. The reference event is coastal for
# Marshfield and streamgage-network based for inland locations.
def plot_seasonal_pairing(catalog, forcing, *, window_days=None, ax=None):
    reference_doy, member_doy, diff, _, reference_column = _seasonal_pairing_values(catalog, forcing)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    if window_days is not None:
        _shade_circular_window(ax, float(window_days))
    ax.scatter(reference_doy, member_doy, c=diff, cmap="viridis", s=10, alpha=0.65)
    ax.plot([1, 366], [1, 366], ls=":", color="black", lw=0.8)
    reference_label = (reference_column or "event reference").replace("_", " ")
    ax.set_xlabel(f"{reference_label} DOY")
    ax.set_ylabel(f"paired {forcing} member DOY")
    in_window = int((diff <= float(window_days)).sum()) if window_days is not None else len(diff)
    title = f"Stage 5.2 — Seasonal-window pairing: {forcing} (n={len(diff):,})"
    if window_days is not None:
        title += f"  |  in-window={in_window}/{len(diff)}"
    ax.set_title(title)
    ax.set_xlim(0, 367)
    ax.set_ylim(0, 367)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 6.1: tail-enrichment audit. Sampling-region split (body vs tail) and
# the per-row sampling_weight that records simulation-budget enrichment.
def plot_sampling_weights(catalog):
    region = catalog["sampling_region"].astype(str)
    weight = pd.to_numeric(catalog["sampling_weight"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    counts = region.value_counts().reindex(["body", "tail"]).fillna(0).astype(int)
    axes[0].bar(counts.index, counts.values, color=["steelblue", "crimson"], alpha=0.85)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("event count")
    axes[0].set_title(f"Stage 6.1 — Sampling region (n={len(catalog):,})")
    axes[0].grid(True, alpha=0.3, axis="y")
    for label, color in [("body", "steelblue"), ("tail", "crimson")]:
        sub = weight[region == label].dropna()
        if len(sub):
            axes[1].hist(sub, bins=30, color=color, alpha=0.65, label=f"{label} (n={len(sub):,})")
    axes[1].set_xlabel("sampling_weight (body/tail enrichment correction)")
    axes[1].set_ylabel("event count")
    axes[1].set_title("Per-row sampling-budget weights")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

def _catalog_with_severity(catalog, *, severity_bands=None):
    frame = catalog.copy()
    if "severity_band" not in frame:
        if "sample_rp_years" not in frame:
            raise ValueError("catalog needs severity_band or sample_rp_years")
        frame["severity_band"] = assign_severity_bands(frame["sample_rp_years"], severity_bands)
    return frame

def _sampling_weight_series(catalog):
    if "sampling_weight" in catalog:
        return pd.to_numeric(catalog["sampling_weight"], errors="coerce").fillna(0.0)
    return pd.Series(np.ones(len(catalog), dtype=float), index=catalog.index)

def severity_band_distribution(catalog, *, band_order=None, severity_bands=None):
    band_order = band_order or ["mild", "common", "significant", "rare", "extreme", "beyond_design"]
    catalog = _catalog_with_severity(catalog, severity_bands=severity_bands)
    counts = catalog["severity_band"].value_counts()
    weights = _sampling_weight_series(catalog)
    weighted = weights.groupby(catalog["severity_band"].astype(str)).sum()
    if "probability_weight" in catalog:
        probability_weights = pd.to_numeric(catalog["probability_weight"], errors="coerce")
        has_probability_weight = bool(probability_weights.notna().any())
        probability = probability_weights.fillna(0.0).groupby(catalog["severity_band"].astype(str)).sum()
    else:
        probability = pd.Series(dtype=float)
        has_probability_weight = False
    bands = [band for band in band_order if band in counts.index]
    distribution = pd.DataFrame(
        {
            "severity_band": bands,
            "event_count": counts.reindex(bands).fillna(0).astype(int).to_numpy(),
            "weighted_mass": weighted.reindex(bands).fillna(0.0).astype(float).to_numpy(),
            "probability_mass": probability.reindex(bands).fillna(0.0).astype(float).to_numpy(),
        }
    )
    distribution.attrs["has_probability_weight"] = has_probability_weight
    return distribution

# Stage 6.2: severity-band coverage. Counts show model-budget allocation;
# probability mass shows the distribution used by response summaries when
# probability_weight is present.
def plot_severity_bands(catalog, *, band_order=None):
    distribution = severity_band_distribution(catalog, band_order=band_order)
    has_probability_weight = bool(distribution.attrs.get("has_probability_weight", False))
    right_column = "probability_mass" if has_probability_weight else "weighted_mass"
    right_ylabel = "probability mass" if has_probability_weight else "weighted pseudo-count"
    right_title = "Probability-weighted mass" if has_probability_weight else "Sampling-weight corrected pseudo-count"
    colors = plt.get_cmap("YlOrRd")(np.linspace(0.25, 0.9, len(distribution)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    bars = axes[0].bar(
        distribution["severity_band"],
        distribution["event_count"],
        color=colors,
        edgecolor="0.3",
    )
    for bar, value in zip(bars, distribution["event_count"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("event count")
    axes[0].set_title(f"Unweighted event count (n={int(distribution['event_count'].sum()):,})")
    axes[0].grid(True, alpha=0.3, axis="y")

    bars = axes[1].bar(
        distribution["severity_band"],
        distribution[right_column],
        color=colors,
        edgecolor="0.3",
    )
    for bar, value in zip(bars, distribution[right_column]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2g}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel(right_ylabel)
    axes[1].set_title(right_title)
    axes[1].grid(True, alpha=0.3, axis="y")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Severity-band coverage", y=1.03)
    return _finish(fig)

def plot_catalog_set_severity_comparison(probability_catalog, stress_catalog, *, band_order=None):
    band_order = band_order or ["mild", "common", "significant", "rare", "extreme", "beyond_design"]
    catalog_label = _catalog_display_label(probability_catalog, default="Catalog")
    probability = severity_band_distribution(probability_catalog, band_order=band_order)
    stress = severity_band_distribution(stress_catalog, band_order=band_order)
    bands = [band for band in band_order if band in set(probability["severity_band"]) | set(stress["severity_band"])]
    probability_counts = probability.set_index("severity_band")["event_count"].reindex(bands).fillna(0)
    stress_counts = stress.set_index("severity_band")["event_count"].reindex(bands).fillna(0)
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.18, probability_counts, width=0.36, color="#4c78a8", alpha=0.75,
           label=f"{catalog_label} (n={int(probability_counts.sum()):,})")
    ax.bar(x + 0.18, stress_counts, width=0.36, color="#d62728", alpha=0.75,
           label=f"Resilience Stress/Training Set (n={int(stress_counts.sum()):,})")
    for xpos, value in zip(x - 0.18, probability_counts):
        if value:
            ax.text(xpos, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + 0.18, stress_counts):
        if value:
            ax.text(xpos, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, bands, rotation=20, ha="right")
    ax.set_ylabel("event count")
    ax.set_title(f"Stage 6.3 — {catalog_label} vs Resilience Stress/Training Set")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    return _finish(fig)

def plot_original_vs_design_severity(original_catalog, design_catalog, *, band_order=None, severity_bands=None):
    band_order = band_order or ["mild", "common", "significant", "rare", "extreme", "beyond_design"]
    original = severity_band_distribution(
        original_catalog,
        band_order=band_order,
        severity_bands=severity_bands,
    )
    design = severity_band_distribution(
        design_catalog,
        band_order=band_order,
        severity_bands=severity_bands,
    )
    bands = [band for band in band_order if band in set(original["severity_band"]) | set(design["severity_band"])]
    original_counts = original.set_index("severity_band")["event_count"].reindex(bands).fillna(0)
    design_counts = design.set_index("severity_band")["event_count"].reindex(bands).fillna(0)
    original_total = max(float(original_counts.sum()), 1.0)
    design_total = max(float(design_counts.sum()), 1.0)
    original_mass = original.set_index("severity_band")["probability_mass"].reindex(bands).fillna(0.0)
    design_mass = design.set_index("severity_band")["probability_mass"].reindex(bands).fillna(0.0)
    if float(original_mass.sum()) <= 0:
        original_mass = original_counts / original_total
    if float(design_mass.sum()) <= 0:
        design_mass = design_counts / design_total
    x = np.arange(len(bands))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(x - 0.18, original_counts / original_total, width=0.36, color="#4c78a8", alpha=0.75,
                label=f"Original POT members (n={int(original_counts.sum()):,})")
    axes[0].bar(x + 0.18, design_counts / design_total, width=0.36, color="#d62728", alpha=0.75,
                label=f"Design catalog count (n={int(design_counts.sum()):,})")
    axes[0].set_ylabel("fraction of rows")
    axes[0].set_title("Stage 6.2 — Original vs design row distribution")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x - 0.18, original_mass, width=0.36, color="#4c78a8", alpha=0.75,
                label="Original POT mass")
    axes[1].bar(x + 0.18, design_mass, width=0.36, color="#d62728", alpha=0.75,
                label="Design probability mass")
    axes[1].set_ylabel("probability mass")
    axes[1].set_title("Probability-weighted distribution")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.set_xticks(x, bands, rotation=20, ha="right")
    return _finish(fig)

# Stage 5.5: pairing-policy sensitivity. Overlay seasonal-window vs independent permutation
# for one forcing. Independent permutation is the baseline sensitivity case, not production.
def plot_independent_vs_seasonal(catalog_seasonal, catalog_independent, forcing, *, window_days=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    plot_seasonal_pairing(catalog_seasonal, forcing, window_days=window_days, ax=axes[0])
    plot_seasonal_pairing(catalog_independent, forcing, window_days=window_days, ax=axes[1])
    for ax, label, catalog in [
        (axes[0], "seasonal_window_permutation", catalog_seasonal),
        (axes[1], "independent_permutation (sensitivity)", catalog_independent),
    ]:
        stats = seasonal_pairing_diagnostics(catalog, forcing, window_days=window_days).iloc[0]
        ax.set_title(
            f"{label} — {forcing}\n"
            f"in-window={int(stats['in_window_rows']):,}/{int(stats['paired_rows']):,}; "
            f"median gap={stats['median_gap_days']:.0f} d"
        )
    fig.suptitle("Stage 5.5 — Pairing-policy sensitivity (in-memory comparison)", y=1.02)
    return _finish(fig)

def _forcing_value_column(members, forcing, value_column=None):
    if value_column is not None:
        if value_column not in members:
            raise ValueError(f"{value_column!r} is not present in {forcing} members")
        return value_column
    candidates_by_forcing = {
        "rainfall": [
            "mean_precip_mm",
            "max_precip_mm",
            "total_precip_mm",
            "rainfall_depth_mm",
            # Backward-compatible legacy names. AORC APCP values are mm, not inches.
            "mean_precip_in",
            "total_precip_in",
            "precip_in",
            "depth_in",
            "mean",
            "max",
        ],
        "soil_moisture": ["soil_moisture_mean", "SOIL_M", "soil_moisture", "mean"],
        "streamflow": ["streamflow", "flow", "discharge", "Q", "mean"],
        "snapwave": ["hs", "Hm0", "wave_height", "tp", "peak_period"],
        "waves": ["hs", "Hm0", "wave_height", "tp", "peak_period"],
    }
    candidates = candidates_by_forcing.get(str(forcing), []) + ["value", "magnitude"]
    column = next((candidate for candidate in candidates if candidate in members), None)
    if column is None:
        numeric = [
            column
            for column in members.columns
            if column != "member_id" and pd.api.types.is_numeric_dtype(members[column])
        ]
        column = numeric[0] if numeric else None
    if column is None:
        raise ValueError(f"could not infer a numeric value column for {forcing} members")
    return column

def _plot_member_table(members, forcing):
    frame = members.copy()
    if "member_id" in frame:
        return frame
    soil_value_column = "SOILSAT_TOP" if "SOILSAT_TOP" in frame else "SOIL_M"
    if str(forcing) == "soil_moisture" and {"time", soil_value_column}.issubset(frame.columns):
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame.dropna(subset=["time"])
        grouped = frame.groupby("time", as_index=False).agg(
            soil_moisture_mean=(soil_value_column, "mean"),
            soil_moisture_min=(soil_value_column, "min"),
            soil_moisture_max=(soil_value_column, "max"),
        )
        grouped["member_id"] = "soil_moisture_" + grouped["time"].dt.strftime("%Y%m%dT%H%M%S")
        return grouped
    time_column = next((column for column in ["storm_date", "storm_start", "time"] if column in frame), None)
    if time_column is not None:
        times = pd.to_datetime(frame[time_column], errors="coerce").dt.strftime("%Y%m%dT%H%M%S")
        frame["member_id"] = str(forcing) + "_" + times.fillna(pd.Series(range(len(frame)), index=frame.index).astype(str))
        return frame
    raise ValueError("members must include a 'member_id' column")

def _probability_series(catalog):
    if "probability_weight" in catalog:
        probability = pd.to_numeric(catalog["probability_weight"], errors="coerce")
        if probability.notna().any():
            return probability.fillna(0.0)
    sampling = pd.to_numeric(catalog.get("sampling_weight", 1.0), errors="coerce").fillna(0.0)
    total = float(sampling.sum())
    if np.isfinite(total) and total > 0:
        return sampling / total
    return pd.Series(np.full(len(catalog), 1.0 / max(len(catalog), 1)), index=catalog.index)

def forcing_selection_frame(catalog, members, forcing, *, value_column=None):
    member_id_column = f"{forcing}_member_id"
    if member_id_column not in catalog:
        raise ValueError(f"{member_id_column!r} is not present in catalog")

    members = _plot_member_table(members, forcing)
    value_column = _forcing_value_column(members, forcing, value_column)
    source = members[["member_id", value_column]].copy()
    source["member_id"] = source["member_id"].astype(str)
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce")

    selected = pd.DataFrame(
        {
            "member_id": catalog[member_id_column].astype(str),
            "sampling_weight": pd.to_numeric(catalog.get("sampling_weight", 1.0), errors="coerce").fillna(0.0),
            "probability_weight": _probability_series(catalog),
        }
    )
    selected = selected[selected["member_id"].notna() & (selected["member_id"] != "<NA>")]
    grouped = selected.groupby("member_id", as_index=False).agg(
        selected_count=("member_id", "size"),
        selected_sampling_mass=("sampling_weight", "sum"),
        selected_probability_mass=("probability_weight", "sum"),
    )
    out = source.merge(grouped, on="member_id", how="left")
    out[["selected_count", "selected_sampling_mass", "selected_probability_mass"]] = out[
        ["selected_count", "selected_sampling_mass", "selected_probability_mass"]
    ].fillna(0.0)
    out["selected_count"] = out["selected_count"].astype(int)
    out = out.rename(columns={value_column: "member_value"})
    out["value_column"] = value_column
    out["forcing"] = forcing
    return out

def _selected_forcing_values(catalog, members, forcing, *, value_column=None):
    member_id_column = f"{forcing}_member_id"
    members = _plot_member_table(members, forcing)
    value_column = _forcing_value_column(members, forcing, value_column)
    source = members[["member_id", value_column]].copy()
    source["member_id"] = source["member_id"].astype(str)
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce")
    selected = catalog.copy()
    selected["member_id"] = selected[member_id_column].astype(str)
    selected["probability_weight"] = _probability_series(selected)
    joined = selected.merge(source, on="member_id", how="left")
    joined = joined.rename(columns={value_column: "member_value"})
    joined["value_column"] = value_column
    return joined

def plot_forcing_marginal_comparison(catalog, members, forcing, *, value_column=None):
    selection = forcing_selection_frame(catalog, members, forcing, value_column=value_column)
    selected_values = np.repeat(
        selection["member_value"].to_numpy(dtype=float),
        selection["selected_count"].to_numpy(dtype=int),
    )
    source_values = selection["member_value"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if source_values.size:
        ax.hist(source_values, bins=24, alpha=0.45, color="0.45", label="source members")
    if selected_values.size:
        ax.hist(selected_values, bins=24, alpha=0.55, color="steelblue", label="selected count")
    weighted = selection[selection["selected_probability_mass"] > 0]
    if len(weighted):
        ax.scatter(
            weighted["member_value"],
            np.zeros(len(weighted)),
            s=600 * weighted["selected_probability_mass"].to_numpy(dtype=float),
            color="crimson",
            alpha=0.55,
            label="selected probability mass",
            zorder=3,
        )
    value_label = selection["value_column"].iloc[0] if len(selection) else "member value"
    ax.set_xlabel(value_label.replace("_", " "))
    ax.set_ylabel("member count")
    ax.set_title(f"Stage 5.4 — {forcing} marginal comparison")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

def _streamflow_member_frame(members):
    frame = members.copy()
    required = {"site_no", "event_time", "peak_flow_cfs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("streamflow members missing required columns: " + ", ".join(sorted(missing)))
    frame["site_no"] = frame["site_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame["peak_flow_cfs"] = pd.to_numeric(frame["peak_flow_cfs"], errors="coerce")
    if "sample_rp_years" in frame:
        frame["sample_rp_years"] = pd.to_numeric(frame["sample_rp_years"], errors="coerce")
    else:
        frame["sample_rp_years"] = np.nan
    if "sampling_region" not in frame:
        frame["sampling_region"] = "body"
    return frame.dropna(subset=["event_time", "peak_flow_cfs"]).copy()

def _streamflow_record_frame(records):
    frame = records.copy()
    frame = frame.rename(
        columns={
            "datetime": "time",
            "dateTime": "time",
            "value": "discharge_cfs",
            "flow_cfs": "discharge_cfs",
            "00060": "discharge_cfs",
        }
    )
    required = {"site_no", "time", "discharge_cfs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("streamflow records missing required columns: " + ", ".join(sorted(missing)))
    frame["site_no"] = frame["site_no"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    return frame.dropna(subset=["site_no", "time", "discharge_cfs"]).sort_values(["site_no", "time"])

def _streamflow_plot_site(records, members, site_no=None):
    if site_no is not None:
        return str(site_no)
    counts = members["site_no"].astype(str).value_counts()
    if not counts.empty:
        return str(counts.index[0])
    return str(records["site_no"].astype(str).iloc[0])

def _streamflow_site_threshold(records, members, site_no, threshold_quantile):
    member_thresholds = pd.to_numeric(
        members.loc[members["site_no"].astype(str) == str(site_no), "site_threshold_cfs"]
        if "site_threshold_cfs" in members
        else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    if len(member_thresholds):
        return float(member_thresholds.median())
    site_records = records.loc[records["site_no"].astype(str) == str(site_no), "discharge_cfs"]
    if site_records.empty:
        return np.nan
    return float(site_records.quantile(float(threshold_quantile)))

def plot_streamflow_pot_extraction(records, members, *, threshold_quantile=0.98, site_no=None, window_slice=None):
    records = _streamflow_record_frame(records)
    members = _streamflow_member_frame(members)
    selected_site = _streamflow_plot_site(records, members, site_no=site_no)
    site_records = records[records["site_no"].astype(str) == selected_site].set_index("time")["discharge_cfs"].sort_index()
    site_peaks = members[members["site_no"].astype(str) == selected_site].set_index("event_time")["peak_flow_cfs"].sort_index()
    if window_slice is None:
        if len(site_peaks):
            center = site_peaks.sort_values(ascending=False).index[0]
        else:
            center = site_records.idxmax()
        window_slice = (
            pd.Timestamp(center) - pd.Timedelta(days=45),
            pd.Timestamp(center) + pd.Timedelta(days=45),
        )
    window = site_records.loc[window_slice[0]:window_slice[1]]
    peak_window = site_peaks.loc[window_slice[0]:window_slice[1]]
    threshold = _streamflow_site_threshold(records, members, selected_site, threshold_quantile)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(window.index, window.values, lw=1.0, label="raw discharge")
    axes[0].scatter(peak_window.index, peak_window.values, color="crimson", s=22, zorder=3, label="POT peaks")
    if np.isfinite(threshold):
        axes[0].axhline(threshold, color="black", ls="--", lw=1.0, label=f"threshold = {threshold:,.0f} cfs")
    axes[0].set_ylabel("discharge [cfs]")
    axes[0].set_title(f"Stage 2.3 — Streamflow POT extraction ({selected_site})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    values = site_peaks.dropna().values
    if len(values):
        axes[1].hist(values, bins=min(35, max(5, int(np.sqrt(len(values)) * 2))), color="steelblue", alpha=0.85)
    if np.isfinite(threshold):
        axes[1].axvline(threshold, color="black", ls="--", lw=1.0)
    axes[1].set_title(f"Historical streamflow peak magnitudes (n={len(values)})")
    axes[1].set_xlabel("peak discharge [cfs]")
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)

def plot_streamflow_pot_members(members):
    frame = _streamflow_member_frame(members)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for site_no, group in frame.groupby("site_no", sort=True):
        ax.scatter(group["event_time"], group["peak_flow_cfs"], s=18, alpha=0.55, label=site_no)
    ax.set_ylabel("peak discharge [cfs]")
    ax.set_title("USGS streamflow POT members by reviewed gage")
    ax.grid(True, alpha=0.3)
    if frame["site_no"].nunique() <= 10:
        ax.legend(loc="best", fontsize=8)
    return _finish(fig)

def plot_streamflow_return_period_distribution(members):
    frame = _streamflow_member_frame(members).dropna(subset=["sample_rp_years"])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = frame["sampling_region"].astype(str).map({"tail": "#b91c1c", "body": "#2563eb"}).fillna("#6b7280")
    ax.scatter(
        frame["sample_rp_years"],
        frame["peak_flow_cfs"],
        s=24,
        alpha=0.65,
        c=colors,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample return period [years]")
    ax.set_ylabel("peak discharge [cfs]")
    ax.set_title("Return-period ranked streamflow members")
    ax.grid(True, alpha=0.3)
    return _finish(fig)

def plot_coastal_forcing_joint(catalog, members, forcing, *, value_column=None):
    axis = _return_period_axis_context(catalog)
    joined = _selected_forcing_values(catalog, members, forcing, value_column=value_column)
    rp = pd.to_numeric(joined["sample_rp_years"], errors="coerce")
    value = pd.to_numeric(joined["member_value"], errors="coerce")
    probability = pd.to_numeric(joined["probability_weight"], errors="coerce").fillna(0.0)
    sizes = 20 + 800 * probability.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(rp, value, s=sizes, alpha=0.65, color="teal", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(axis["axis_label"])
    value_label = joined["value_column"].iloc[0] if len(joined) else "member value"
    ax.set_ylabel(value_label.replace("_", " "))
    ax.set_title(f"Stage 5.5 — {axis['joint_label']} vs {forcing} forcing")
    ax.grid(True, alpha=0.3)
    return _finish(fig)

# Stage 4.4: synthetic-vs-historical descriptor distributions for QC acceptance.
def plot_acceptance_descriptors(template_frame, summary):
    # duration_above_50pct_peak and asymmetry_ratio are degenerate here: templates
    # are cut on a fixed symmetric window so asymmetry is constant 1.0, and
    # synthetic events reuse template duration by construction.
    columns = ["peak", "volume"]
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for ax, column in zip(axes, columns):
        h = pd.to_numeric(template_frame[column], errors="coerce").dropna()
        s = pd.to_numeric(summary[column], errors="coerce").dropna()
        ax.hist(h, bins=30, alpha=0.6, density=True, label="historical", color="steelblue")
        ax.hist(s, bins=30, alpha=0.6, density=True, label="synthetic", color="orange")
        ax.set_title(column.replace("_", " "))
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Stage 4.4 — Acceptance: synthetic descriptors vs historical templates", y=1.02)
    return _finish(fig)

def plot_distinct_oscillatory_proxies(
    member_dataset,
    summary,
    template_frame,
    waterlevel,
    *,
    random_seed=0,
    candidate_n=250,
    pick_n=5,
    oscillation_keep_fraction=0.4,
):
    rng = np.random.default_rng(random_seed)
    axis = member_dataset["relative_hour"].to_numpy().astype(int)
    summary_idx = summary.set_index("event_id")
    template_idx = template_frame.set_index("template_id")
    candidate_ids = pd.Index(
        rng.choice(
            summary_idx.index.to_numpy(dtype=str),
            size=min(candidate_n, len(summary_idx)),
            replace=False,
        )
    )
    proxy_rows = []
    score_rows = []
    for event_id in candidate_ids:
        meta = summary_idx.loc[event_id]
        tpl = template_idx.loc[meta["template_id"]]
        hist_time = tpl["peak_time"] + pd.to_timedelta(axis, unit="h")
        hist_total = waterlevel.reindex(hist_time).to_numpy(dtype=float)
        baseline = float(tpl["baseline_m"])
        scale = float(meta["peak"]) / max(float(tpl["peak_m"]), 1e-6)
        proxy = baseline + (hist_total - baseline) * scale
        proxy_rows.append(proxy)
        dy = np.diff(np.nan_to_num(proxy, nan=baseline))
        sign = np.sign(dy)
        sign_changes = int(np.sum((sign[1:] * sign[:-1]) < 0)) if sign.size > 1 else 0
        roughness = float(np.nansum(np.abs(np.diff(proxy, n=2)))) if proxy.size > 2 else 0.0
        score_rows.append((event_id, sign_changes, roughness))
    proxy_df = pd.DataFrame(proxy_rows, index=candidate_ids, columns=axis)
    score_df = pd.DataFrame(
        score_rows,
        columns=["event_id", "sign_changes", "roughness"],
    ).set_index("event_id")
    score_df["oscillation_score"] = score_df["sign_changes"] + 0.05 * score_df["roughness"]
    keep_n = max(pick_n, int(np.ceil(len(score_df) * oscillation_keep_fraction)))
    candidate_ids = score_df.sort_values("oscillation_score", ascending=False).head(keep_n).index
    proxy_df = proxy_df.loc[candidate_ids]
    matrix = np.nan_to_num(proxy_df.to_numpy(dtype=float), nan=0.0)
    seed = int(rng.integers(matrix.shape[0]))
    picked = [seed]
    while len(picked) < min(pick_n, matrix.shape[0]):
        remaining = [i for i in range(matrix.shape[0]) if i not in picked]
        min_dist = [min(np.linalg.norm(matrix[i] - matrix[j]) for j in picked) for i in remaining]
        picked.append(remaining[int(np.argmax(min_dist))])
    picked_ids = pd.Index(candidate_ids)[picked]
    picked_rows = summary_idx.loc[picked_ids]
    picked_proxy_df = proxy_df.loc[picked_ids]
    y_min = float(np.nanmin(picked_proxy_df.to_numpy(dtype=float)))
    y_max = float(np.nanmax(picked_proxy_df.to_numpy(dtype=float)))
    y_pad = 0.05 * max(y_max - y_min, 1e-6)
    fig, axes = plt.subplots(len(picked_ids), 1, figsize=(11, 2.4 * len(picked_ids)), sharex=True)
    for ax, event_id in zip(np.atleast_1d(axes), picked_ids):
        series = proxy_df.loc[event_id].dropna()
        meta = picked_rows.loc[event_id]
        ax.plot(series.index, series.values, lw=1.6)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel("m+MSL")
        ax.set_title(
            f"{event_id} | peak={meta['peak']:.2f} m | "
            f"rp={meta['sample_rp_years']:.1f} y | template={meta['template_id']}"
        )
        ax.grid(True, alpha=0.3)
    axes = np.atleast_1d(axes)
    axes[-1].set_xlabel("relative hour")
    fig.suptitle("Stage 4.5 — Distinct oscillatory water-level proxies", y=1.01, fontsize=11)
    selected_columns = [
        "template_id",
        "sample_rp_years",
        "peak",
        "volume",
        "duration_above_50pct_peak",
        "asymmetry_ratio",
    ]
    selected_columns = [column for column in selected_columns if column in picked_rows]
    selected = picked_rows[selected_columns].join(
        score_df[["sign_changes", "roughness", "oscillation_score"]]
    )
    return _finish(fig), selected

def plot_msl_shift_scenario_comparison(
    scenario_datasets,
    marginal_ci,
    marginal_params,
    *,
    scenario_colors=None,
    example_event_index=1000,
):
    scenario_names = list(scenario_datasets)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scenario_colors = scenario_colors or {
        name: default_colors[i % len(default_colors)] for i, name in enumerate(scenario_names)
    }
    scenario_offsets = {
        name: float(ds.attrs.get("slr_offset_m", 0.0)) for name, ds in scenario_datasets.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for name in scenario_names:
        ds = scenario_datasets[name]
        peaks = pd.to_numeric(ds["peak"].to_series(), errors="coerce").dropna() + scenario_offsets[name]
        ax.hist(
            peaks,
            bins=min(40, max(5, len(peaks))),
            alpha=0.55,
            label=f"{name} (+{scenario_offsets[name]:.2f} m)",
            color=scenario_colors[name],
        )
    ax.set_xlabel("absolute peak [m+MSL_ref]")
    ax.set_ylabel("count")
    ax.set_title("Synthetic peak distribution per scenario")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    base_ds = scenario_datasets[scenario_names[0]]
    event_ids = base_ds["event_id"].to_numpy()
    example_event = str(event_ids[min(example_event_index, len(event_ids) - 1)])
    for name in scenario_names:
        ds = scenario_datasets[name]
        series = ds["surge_absolute"].sel(event_id=example_event).to_numpy()
        rel = ds["relative_hour"].to_numpy()
        finite = np.isfinite(series)
        ax.plot(
            rel[finite],
            series[finite],
            color=scenario_colors[name],
            label=f"{name} (+{scenario_offsets[name]:.2f} m)",
            linewidth=1.6,
        )
    ax.set_xlabel("relative hour")
    ax.set_ylabel("absolute water level [m+MSL_ref]")
    ax.set_title(f"Event {example_event}: rigid translation under SLR")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    rps = marginal_ci["rps"].to_numpy()
    for name in scenario_names:
        offset = scenario_offsets[name]
        ax.plot(
            rps,
            marginal_ci["h_point"] + offset,
            color=scenario_colors[name],
            label=f"{name} (+{offset:.2f} m)",
            linewidth=1.8,
        )
        ax.fill_between(
            rps,
            marginal_ci["h_lo"] + offset,
            marginal_ci["h_hi"] + offset,
            color=scenario_colors[name],
            alpha=0.18,
        )
    ax.set_xscale("log")
    ax.set_xlabel("coastal driver return period [years]")
    ax.set_ylabel("absolute peak [m+MSL_ref]")
    ax.set_title("Coastal-driver return-period curve per scenario (95% bootstrap CI)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    params = marginal_params if isinstance(marginal_params, pd.DataFrame) else pd.DataFrame(marginal_params)
    ref_epoch = float(params["detrend_reference_epoch_year"].dropna().iloc[0])
    fig.suptitle(f"Stage 7 — MSL-shift scenarios | reference epoch {ref_epoch:.1f}", y=1.02, fontsize=11)
    return _finish(fig)

# --- Copula-Joint compound-dependence figures ------------------------------------------
# These visualize the production copula_joint method: the paired co-occurrence sample,
# the fitted vine, AND joint-exceedance isolines, the tail-enrichment budget, and the
# field-preserving realization. Heavy imports are local to avoid import cycles.

def plot_driver_cooccurrence(paired, x, y, *, ax=None):
    """Scatter the two-sided POT co-occurrence sample, colored by conditioning driver."""
    from scipy.stats import kendalltau

    fig, ax = _axis(ax, (6.5, 5.5))
    for cond, color in zip(sorted(paired["conditioned_on"].unique()), ["#4c78a8", "#f58518", "#54a24b"]):
        sub = paired[paired["conditioned_on"] == cond]
        ax.scatter(sub[x], sub[y], s=16, alpha=0.55, color=color, label=f"conditioned on {cond} (n={len(sub)})")
    tau, p = kendalltau(paired[x], paired[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Two-sided POT co-occurrence — Kendall τ={tau:.2f} (p={p:.1e})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish_created(fig)

def plot_copula_fit_diagnostics(model, paired, *, n=5000, seed=11):
    """Observed vs simulated dependence on the uniform (pseudo-observation) scale."""
    import pyvinecopulib as pv

    names = list(model.driver_names)
    observed = pv.to_pseudo_obs(np.asfortranarray(paired[names].to_numpy(dtype=float)), ties_method="random", seeds=[int(seed)])
    simulated = np.asarray(model.vine.simulate(int(n), qrng=True, seeds=[int(seed)]), dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.hexbin(
        simulated[:, 0],
        simulated[:, 1],
        gridsize=34,
        extent=(0, 1, 0, 1),
        mincnt=1,
        cmap="Greys",
        alpha=0.65,
        label=f"fitted vine sample (n={len(simulated):,})",
    )
    ax.scatter(observed[:, 0], observed[:, 1], s=22, alpha=0.80, color="#d62728", label=f"observed pseudo-obs (n={len(observed):,})")
    families = ", ".join(sorted({str(f) for row in model.vine.families for f in row})) if hasattr(model.vine, "families") else ""
    ax.set_xlabel(f"u[{names[0]}]")
    ax.set_ylabel(f"u[{names[1]}]")
    ax.set_title(f"Vine copula fit (semiparametric, simulated n={len(simulated):,})\nfamilies: {families}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish(fig)

def plot_and_joint_isolines(
    model,
    *,
    return_periods=(10, 50, 100, 500),
    n_sample=4000,
    grid=60,
    seed=11,
    ax=None,
    paired=None,
    catalog=None,
):
    """AND joint-exceedance isolines over two drivers."""
    from design_events.build_events.probability.exceedance import and_return_period, and_survival_from_cdf

    if model.dim != 2:
        raise ValueError("AND isoline plot supports exactly two drivers")
    names = list(model.driver_names)
    u = np.asarray(model.vine.simulate(int(n_sample), qrng=True, seeds=[int(seed)]), dtype=float)
    x = np.asarray(model.marginals[0].ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9)), dtype=float)
    y = np.asarray(model.marginals[1].ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9)), dtype=float)
    _, rp_points = and_return_period(and_survival_from_cdf(u, model.vine.cdf), model.event_rate)

    xs = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.999), grid)
    ys = np.linspace(np.quantile(y, 0.01), np.quantile(y, 0.999), grid)
    xx, yy = np.meshgrid(xs, ys)
    ug = np.column_stack([
        np.asarray(model.marginals[0].cdf(xx.ravel()), dtype=float),
        np.asarray(model.marginals[1].cdf(yy.ravel()), dtype=float),
    ])
    _, rp_grid = and_return_period(and_survival_from_cdf(ug, model.vine.cdf), model.event_rate)
    rp_grid = rp_grid.reshape(xx.shape)
    # The vine CDF is a QMC estimate, so the raw RP surface is noisy; smooth in log space
    # to give clean, single-segment isolines (removes jaggedness and duplicate labels).
    from scipy.ndimage import gaussian_filter

    rp_cap = 10.0 * float(max(return_periods))
    finite_grid = rp_grid[np.isfinite(rp_grid)]
    fill = float(finite_grid.max()) if finite_grid.size else rp_cap
    rp_filled = np.clip(np.where(np.isfinite(rp_grid), rp_grid, fill), 1e-3, rp_cap)
    rp_smooth = 10.0 ** gaussian_filter(np.log10(rp_filled), sigma=1.2)

    fig, ax = _axis(ax, (7.0, 5.6))
    finite = np.isfinite(rp_points)
    sc = ax.scatter(x[finite], y[finite], c=np.log10(np.clip(rp_points[finite], 1e-3, None)), s=10, alpha=0.5, cmap="viridis")
    contour = ax.contour(xx, yy, rp_smooth, levels=sorted(return_periods), colors="crimson", linewidths=1.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt=lambda v: f"{v:g}-yr AND")
    if paired is not None and all(name in paired for name in names):
        ax.scatter(
            paired[names[0]],
            paired[names[1]],
            s=14,
            facecolors="none",
            edgecolors="black",
            alpha=0.28,
            linewidths=0.8,
            label=f"observed POT pairs (n={len(paired):,})",
        )
    if catalog is not None and all(name in catalog for name in names):
        catalog_label = _catalog_display_label(catalog, default="Selected Catalog")
        ax.scatter(
            catalog[names[0]],
            catalog[names[1]],
            s=20,
            color="#d62728",
            marker="x",
            linewidths=0.8,
            alpha=0.72,
            label=f"{catalog_label} (n={len(catalog):,})",
        )
    cbar = (fig or ax.figure).colorbar(sc, ax=ax)
    cbar.set_label("log10 AND joint return period [yr]")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("AND joint-exceedance isolines with observed and selected events")
    if paired is not None or catalog is not None:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _finish_created(fig)

def plot_tide_ntr_decomposition(components, window_slice=("2018-01-01", "2018-03-31")):
    """CORA total water level split into MSL + tide + non-tidal residual (Fix 2)."""
    sub = components.loc[slice(*window_slice)] if window_slice else components
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sub.index, sub["wl"], color="#4c78a8", lw=0.7, label="total water level")
    axes[0].plot(sub.index, sub["msl"] + sub["tide"], color="#f58518", lw=0.7, alpha=0.85, label="MSL + astronomical tide")
    axes[0].set_ylabel("m (MSL datum)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("CORA total water level vs reconstructed MSL + tide (utide harmonic analysis)")
    axes[1].plot(sub.index, sub["ntr"], color="#e45756", lw=0.7, label="non-tidal residual (surge)")
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_ylabel("NTR (m)")
    axes[1].set_xlabel("time")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Non-tidal residual = storm surge — the coastal copula axis and scaled quantity")
    return _finish(fig)

def plot_storm_type_cooccurrence(paired, x, y, *, ax=None):
    """Scatter the co-occurrence sample colored by storm-type population (Fix 3)."""
    fig, ax = _axis(ax, (6.5, 5.5))
    palette = {"nor_easter": "#4c78a8", "other_non_tropical": "#54a24b", "tc": "#e45756", "unresolved": "#bab0ac"}
    for storm_type in [s for s in palette if s in set(paired["storm_type"])]:
        sub = paired[paired["storm_type"] == storm_type]
        ax.scatter(sub[x], sub[y], s=20, alpha=0.65, color=palette[storm_type], label=f"{storm_type} (n={len(sub)})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Compound drivers by storm-type population")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish_created(fig)

def plot_population_copula_fits(model, paired, *, n=4000, seed=11):
    """Per-population observed vs fitted dependence on the uniform scale (small multiples)."""
    import pyvinecopulib as pv

    names = list(model.driver_names)
    pops = list(model.populations)
    fig, axes = plt.subplots(1, len(pops), figsize=(5.2 * len(pops), 5.0), squeeze=False)
    for ax, pop in zip(axes[0], pops):
        sub = paired[paired["storm_type"] == pop.storm_type]
        simulated = np.asarray(pop.vine.simulate(int(n), qrng=True, seeds=[int(seed)]), dtype=float)
        ax.hexbin(simulated[:, 0], simulated[:, 1], gridsize=30, extent=(0, 1, 0, 1), mincnt=1, cmap="Greys", alpha=0.65)
        if len(sub) >= 2:
            observed = pv.to_pseudo_obs(np.asfortranarray(sub[names].to_numpy(dtype=float)), ties_method="random", seeds=[int(seed)])
            ax.scatter(observed[:, 0], observed[:, 1], s=22, alpha=0.85, color="#d62728")
        flag = "  [low-confidence fit]" if pop.low_confidence else ""
        ax.set_title(f"{pop.storm_type}  (n={pop.n_events}, λ={pop.rate:.2f}/yr){flag}", fontsize=10)
        ax.set_xlabel(f"u[{names[0]}]")
        ax.set_ylabel(f"u[{names[1]}]")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-storm-type vine fits: observed pseudo-obs (red) vs fitted sample (grey)")
    return _finish(fig)

def plot_combined_and_isolines(model, *, return_periods=(10, 50, 100, 500), grid=60, n_sample=4000, seed=11, paired=None, catalog=None, ax=None):
    """Combined AND isolines across storm-type populations."""
    from scipy.ndimage import gaussian_filter

    from design_events.build_events.probability.exceedance import combined_return_period

    names = list(model.driver_names)
    # mixture pool for axis ranges and the scatter
    phys_parts, type_parts = [], []
    for k, pop in enumerate(model.populations):
        n_pop = max(50, int(round(int(n_sample) * pop.rate / model.total_rate)))
        u = np.asarray(pop.vine.simulate(n_pop, qrng=True, seeds=[int(seed) + k]), dtype=float)
        phys_parts.append(np.column_stack([np.asarray(m.ppf(np.clip(u[:, j], 1e-9, 1 - 1e-9)), dtype=float) for j, m in enumerate(pop.marginals)]))
        type_parts.append(np.array([pop.storm_type] * n_pop))
    phys = np.vstack(phys_parts)
    _, rp_points = combined_return_period(phys, model.populations)

    xs = np.linspace(np.quantile(phys[:, 0], 0.01), np.quantile(phys[:, 0], 0.999), grid)
    ys = np.linspace(np.quantile(phys[:, 1], 0.01), np.quantile(phys[:, 1], 0.999), grid)
    xx, yy = np.meshgrid(xs, ys)
    _, rp_grid = combined_return_period(np.column_stack([xx.ravel(), yy.ravel()]), model.populations)
    rp_grid = rp_grid.reshape(xx.shape)
    rp_cap = 10.0 * float(max(return_periods))
    finite = rp_grid[np.isfinite(rp_grid)]
    fill = float(finite.max()) if finite.size else rp_cap
    rp_smooth = 10.0 ** gaussian_filter(np.log10(np.clip(np.where(np.isfinite(rp_grid), rp_grid, fill), 1e-3, rp_cap)), sigma=1.2)
    fig, ax = _axis(ax, (7.0, 5.6))
    palette = {"nor_easter": "#4c78a8", "other_non_tropical": "#54a24b", "tc": "#e45756", "unresolved": "#bab0ac"}
    pool_type = np.concatenate(type_parts)
    for storm_type in [s for s in palette if s in set(pool_type)]:
        m = pool_type == storm_type
        ax.scatter(phys[m, 0], phys[m, 1], s=8, alpha=0.30, color=palette[storm_type], label=f"{storm_type} pool")
    contour = ax.contour(xx, yy, rp_smooth, levels=sorted(return_periods), colors="crimson", linewidths=1.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt=lambda v: f"{v:g}-yr AND")
    if catalog is not None and all(name in catalog for name in names):
        ax.scatter(catalog[names[0]], catalog[names[1]], s=16, color="black", marker="x", linewidths=0.7, alpha=0.6, label=f"selected catalog (n={len(catalog):,})")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Combined AND isolines across storm-type populations")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _finish_created(fig)

def plot_joint_tail_budget(catalog, stress_settings, *, severity_bands=None, band_order=None):
    """Compare the fitted candidate pool against the selected design/stress set."""
    from design_events.build_events.probability.dependence import check_stress_budget

    catalog_label = _catalog_display_label(catalog, default="Catalog")
    report = check_stress_budget(catalog, stress_settings or {}, severity_bands=severity_bands, raise_on_shortfall=False)
    order = band_order or ["mild", "common", "significant", "rare", "extreme", "beyond_design"]
    report = report.set_index("severity_band").reindex([b for b in order if b in report["severity_band"].values]).dropna(how="all").reset_index()
    mass = catalog.groupby("severity_band")["probability_weight"].sum()
    pool_counts, pool_total = _candidate_pool_band_counts(catalog, report["severity_band"].astype(str).tolist())
    report["candidate_pool_count"] = report["severity_band"].map(pool_counts).fillna(0.0).to_numpy(dtype=float)
    keep = (
        (report["candidate_pool_count"] > 0)
        | (pd.to_numeric(report["catalog_count"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(report["stress_budget_count"], errors="coerce").fillna(0) > 0)
    )
    report = report[keep].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    pos = np.arange(len(report))
    pool_label = f"candidate pool (n={pool_total:,})" if pool_total else "candidate pool"
    selected_label = f"{catalog_label} / stress set (n={len(catalog):,})"
    axes[0].bar(pos - 0.24, report["candidate_pool_count"], width=0.24, color="#9ecae9", label=pool_label)
    axes[0].bar(pos, report["catalog_count"], width=0.24, color="#4c78a8", label=selected_label)
    axes[0].bar(pos + 0.24, report["stress_budget_count"], width=0.24, color="#e45756", label="configured stress budget")
    axes[0].set_xticks(pos, report["severity_band"], rotation=25, ha="right")
    axes[0].set_ylabel("events")
    axes[0].set_yscale("log")
    positive_counts = report[["candidate_pool_count", "catalog_count", "stress_budget_count"]].to_numpy(dtype=float)
    positive_counts = positive_counts[positive_counts > 0]
    if positive_counts.size:
        axes[0].set_ylim(max(0.8, positive_counts.min() * 0.6), positive_counts.max() * 1.8)
    axes[0].set_title(f"Candidate pool to selected stress set ({pool_total:,} -> {len(catalog):,})")
    axes[0].legend(fontsize=9)
    flags = report["meets_budget"].astype(bool).all()
    axes[0].text(0.02, 0.95, "budget MET" if flags else "BUDGET SHORTFALL", transform=axes[0].transAxes,
                 color=("#2a7" if flags else "#c33"), fontsize=10, va="top")

    selected = pd.to_numeric(report["catalog_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    pool = report["candidate_pool_count"].to_numpy(dtype=float)
    selection_rate = np.divide(selected, pool, out=np.zeros_like(selected), where=pool > 0)
    axes[1].bar(pos, 100.0 * selection_rate, color="#72b7b2")
    axes[1].set_xticks(pos, report["severity_band"], rotation=25, ha="right")
    axes[1].set_ylabel("selected from pool [%]")
    axes[1].set_title("Selection rate by severity band")
    axes[1].grid(True, alpha=0.3, axis="y")

    band_mass = [float(mass.get(b, 0.0)) for b in report["severity_band"]]
    axes[2].bar(pos, band_mass, color="#f58518")
    axes[2].set_xticks(pos, report["severity_band"], rotation=25, ha="right")
    axes[2].set_ylabel("probability mass")
    axes[2].set_title("Fitted probability mass by band")
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    return _finish(fig)

def plot_realization_scaling(catalog, driver, *, ax=None):
    """Field-Preserving Realization diagnostics: scale-factor spread + analog diversity."""
    scale_col, member_col = f"{driver}_scale_factor", f"{driver}_template_member_id"
    scales = pd.to_numeric(catalog[scale_col], errors="coerce").dropna()
    fig, ax = _axis(ax, (6.5, 4.5))
    ax.hist(scales, bins=40, color="#4c78a8", alpha=0.8)
    ax.axvline(1.0, color="crimson", ls="--", lw=1.5, label="K=1 (no scaling)")
    distinct = int(catalog[member_col].nunique()) if member_col in catalog else 0
    ax.set_xlabel(f"{driver} scale factor K = target / observed")
    ax.set_ylabel("events")
    ax.set_title(f"{driver}: field-preserving realization\n{distinct} distinct observed analogs used (n={len(scales)})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _finish_created(fig)

def plot_storm_loading_pattern(catalog, *, ax=None):
    """Inland Storm Timing Descriptor: where storms peak within their accumulation window.

    Histogram of normalized peak position (0=onset, 1=window end) with front/center/back
    tercile bands — a diversity axis for the Resilience Stress/Training Set (ADR-0019).
    """
    position = pd.to_numeric(catalog.get("storm_loading_position"), errors="coerce").dropna()
    fig, ax = _axis(ax, (6.5, 4.5))
    ax.hist(position, bins=np.linspace(0, 1, 25), color="#4c78a8", alpha=0.85)
    for edge in (1 / 3, 2 / 3):
        ax.axvline(edge, color="0.4", ls="--", lw=1.0)
    ymax = ax.get_ylim()[1]
    for x, label in ((1 / 6, "front"), (0.5, "center"), (5 / 6, "back")):
        ax.text(x, ymax * 0.92, label, ha="center", fontsize=9, color="0.3")
    if "storm_loading_pattern" in catalog:
        counts = catalog["storm_loading_pattern"].value_counts().to_dict()
        subtitle = "  ".join(f"{k}={v}" for k, v in counts.items() if k != "unresolved")
    else:
        subtitle = ""
    ax.set_xlabel("normalized rainfall-peak position in storm window")
    ax.set_ylabel("events")
    ax.set_title(f"Storm loading pattern (n={len(position)})\n{subtitle}", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    return _finish_created(fig)


def plot_observed_basin_lag(basin_lag_frame):
    """Observed catchment basin lag at the Primary Reference Gage.

    Left: distribution of observed-discharge-peak minus rainfall-peak lag (the observed
    reference for the Wflow Readiness peak-timing check). Right: lag vs antecedent soil
    moisture colored by season (the observed side of the Soil-Moisture Modulation
    Diagnostic).
    """
    frame = basin_lag_frame.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lag = pd.to_numeric(frame.get("basin_lag_hours"), errors="coerce").dropna()
    axes[0].hist(lag, bins=30, color="#54a24b", alpha=0.85)
    if len(lag):
        axes[0].axvline(float(lag.median()), color="crimson", ls="--", lw=1.5,
                        label=f"median {lag.median():.0f} h")
        axes[0].legend(loc="best", fontsize=9)
    axes[0].set_xlabel("observed basin lag (h): discharge peak − rainfall peak")
    axes[0].set_ylabel("storms")
    axes[0].set_title(f"Observed basin lag @ reference gage (n={len(lag)})", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    soil = pd.to_numeric(frame.get("antecedent_soil_moisture"), errors="coerce")
    if soil.notna().any():
        for season, sub in frame.assign(_soil=soil).dropna(subset=["_soil", "basin_lag_hours"]).groupby("season"):
            axes[1].scatter(sub["_soil"], sub["basin_lag_hours"], s=22, alpha=0.6, label=str(season))
        axes[1].set_xlabel("antecedent soil moisture at onset")
        axes[1].legend(loc="best", fontsize=8, title="season")
    else:
        axes[1].text(0.5, 0.5, "no antecedent soil moisture joined", ha="center", va="center",
                     transform=axes[1].transAxes, color="0.5")
        axes[1].set_xlabel("antecedent soil moisture at onset")
    axes[1].set_ylabel("observed basin lag (h)")
    axes[1].set_title("Lag vs antecedent moisture (response-side modulation)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    return _finish(fig)


def plot_timing_seasonality(seasonality_frame):
    """Rainfall-peak month and hour distributions (convective vs frontal/tropical split)."""
    frame = seasonality_frame.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    months = pd.to_numeric(frame.get("month"), errors="coerce").dropna().astype(int)
    axes[0].hist(months, bins=np.arange(0.5, 13.5, 1), color="#e45756", alpha=0.85)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xlabel("rainfall-peak month")
    axes[0].set_ylabel("storms")
    axes[0].set_title("Peak seasonality", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    hours = pd.to_numeric(frame.get("hour"), errors="coerce").dropna().astype(int)
    axes[1].hist(hours, bins=np.arange(-0.5, 24.5, 1), color="#f58518", alpha=0.85)
    axes[1].set_xlabel("rainfall-peak hour (UTC)")
    axes[1].set_ylabel("storms")
    axes[1].set_title("Peak diurnal timing", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    return _finish(fig)


# Short notebook-facing plot names.
plot_rainfall = plot_rainfall_member_distribution
plot_return_periods = plot_streamflow_return_period_distribution
plot_tail_budget = plot_joint_tail_budget
plot_scaling = plot_realization_scaling
plot_loading = plot_storm_loading_pattern
plot_basin_lag = plot_observed_basin_lag
plot_seasonality = plot_timing_seasonality
