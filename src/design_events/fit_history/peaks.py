from __future__ import annotations
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from .extreme_value import (
    bootstrap_return_values,
    eva_block_maxima,
    eva_peaks_over_threshold,
    normalize_time_frequency,
    plot_return_values,
    rps_default,
)
from .return_curve import (
    from_eva_dataset,
    marginal_params_frame,
    marginal_rps_frame,
    write_historical_peak_marginal,
)

def load_hourly_waterlevel(path):
    # Step 1: Cora data collector writes one hourly value column.
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return pd.to_numeric(frame["value"], errors="coerce").sort_index()

def _run_eva(waterlevel, peak_settings, fit_rps):
    # Single EVA pass: peak extraction + marginal fit. Factored out so the
    # detrend workflow can call it twice -- once on raw data to estimate
    # the secular trend, then again on detrended data to produce the
    # production fit.
    method = str(peak_settings.get("method", "pot")).strip().lower()
    period = normalize_time_frequency(peak_settings.get("hydrological_year_start", "YS-AUG"))
    selection = str(peak_settings.get("selection_criterion", "AIC"))
    hourly = waterlevel.dropna()
    da_hourly = xr.DataArray(
        hourly.values,
        dims="time",
        coords={"time": hourly.index},
        attrs={"units": "m+MSL"},
    )
    if method == "pot":
        pot_settings = peak_settings.get("pot", {})
        pot_distributions = list(pot_settings.get("distributions", ["exp", "gpd"]))
        threshold_quantile = float(pot_settings.get("threshold_quantile", 0.98))
        ds_eva = eva_peaks_over_threshold(
            da_hourly,
            qthresh=threshold_quantile,
            min_dist=int(pot_settings.get("min_peak_distance_hours", 72)),
            period=period,
            distribution=None if len(pot_distributions) > 1 else pot_distributions[0],
            rps=fit_rps,
            criterium=selection,
        )
    else:
        threshold_quantile = float("nan")
        ds_eva = eva_block_maxima(
            da_hourly,
            period=period,
            distribution=None,
            rps=fit_rps,
            criterium=selection,
        )
    marginal = from_eva_dataset(ds_eva, method=method, threshold_quantile=threshold_quantile)
    peaks = ds_eva["peaks"].dropna("time").to_series().rename("h")
    return peaks, marginal

def _years_since(index, ref_year):
    # Convert a DatetimeIndex into floating years past a reference epoch.
    # Daily fraction (dayofyear / 365.25) is sufficient resolution for a
    # secular linear trend (mm-per-year scale).
    years = index.year.to_numpy(dtype=float) + (index.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    return years - float(ref_year)

def detrend_hourly_to_reference_epoch(waterlevel, slope, ref_year):
    # Translate the hourly water level to its reference-epoch equivalent
    # by subtracting a linear secular trend.
    #
    #   H_detrended(t) = H(t) - slope * (year(t) - reference_epoch_year)
    #
    # After this transformation, the record has zero linear trend in
    # expectation and represents what hourly water level WOULD have looked
    # like across 1979-2022 if mean sea level had been pinned to its
    # reference-epoch value the whole time. Storm anomalies are preserved
    # because the trend is removed uniformly across all hours.
    if slope == 0.0:
        return waterlevel
    series = waterlevel.dropna()
    delta = float(slope) * _years_since(series.index, ref_year)
    return waterlevel.subtract(pd.Series(delta, index=series.index), fill_value=0.0)

def _resolve_reference_epoch(detrend_cfg, waterlevel):
    # "midpoint" (default) -> auto-derive from record start/end. Anything
    # else is parsed as an integer year. Auto-derivation keeps the choice
    # data-dependent and removes a magic number.
    raw = detrend_cfg.get("reference_epoch", "midpoint")
    if isinstance(raw, str) and raw.strip().lower() == "midpoint":
        idx = waterlevel.dropna().index
        return float((idx.min().year + idx.max().year) / 2.0)
    return float(raw)

def _theil_sen_slope(peaks_series):
    series = peaks_series.dropna()
    if len(series) < 3:
        return 0.0
    t = _years_since(series.index, series.index.min().year)
    return float(stats.theilslopes(series.to_numpy(dtype=float), t, 0.95).slope)

def _boundary_cora_annual_mean_slope(waterlevel):
    # Estimate secular MSL trend from the same CORA boundary-node series
    # used as forcing, not from storm peaks and not from a distant gauge.
    # Annual means suppress tide phase and storm weather enough for a
    # transparent local trend estimate.
    series = waterlevel.dropna().sort_index()
    if len(series) < 3 or not isinstance(series.index, pd.DatetimeIndex):
        return 0.0, 0
    annual = series.resample("YS").agg(["mean", "count"])
    min_hours = int(0.75 * 365.25 * 24)
    annual = annual[annual["count"] >= min_hours]
    if len(annual) < 3:
        return 0.0, int(len(annual))
    years = annual.index.year.to_numpy(dtype=float) + 0.5
    t = years - float(years.min())
    slope = stats.theilslopes(annual["mean"].to_numpy(dtype=float), t, 0.95).slope
    return float(slope), int(len(annual))

def _resolve_detrend_slope(detrend_cfg, raw_peaks, waterlevel=None):
    # Source the slope: theil_sen (default) or external (regulator-supplied).
    method = str(detrend_cfg.get("method", "theil_sen")).strip().lower()
    if method == "none":
        return 0.0, "none", {}
    if method == "external":
        external = detrend_cfg.get("external_slope_m_per_year")
        if external is None:
            raise ValueError(
                "extremes.detrend.method = 'external' requires "
                "extremes.detrend.external_slope_m_per_year to be set"
            )
        return float(external), "external", {}
    if method in {"boundary_cora", "boundary_cora_annual_mean", "cora_annual_mean", "cora_waterlevel"}:
        if waterlevel is None:
            raise ValueError("boundary CORA detrending requires hourly waterlevel series")
        slope, n_years = _boundary_cora_annual_mean_slope(waterlevel)
        return slope, "boundary_cora_annual_mean", {"annual_mean_year_count": n_years}
    return _theil_sen_slope(raw_peaks), "theil_sen", {}

def fit_historical_peaks(config, waterlevel):
    # Step 2a: select historical peaks and fit their return-period curve.
    #
    # If MSL detrending is enabled (extremes.detrend.enabled = true), a
    # two-pass workflow runs:
    #   pass 1: extract peaks on RAW hourly to estimate the secular slope.
    #   detrend: subtract slope * (year - reference_epoch) from hourly.
    #   pass 2: re-extract on DETRENDED hourly and fit the production curve.
    # The marginal returned represents peaks at the reference epoch, so
    # MSL-shift scenario offsets in config.yaml are unambiguously
    # relative to that epoch.
    peak_settings = config.get("extremes", {})
    detrend_cfg = peak_settings.get("detrend", {}) or {}
    rps = np.array(peak_settings.get("return_periods", rps_default), dtype=float)
    fit_rps = np.hstack([[1.1], rps])

    detrend_enabled = bool(detrend_cfg.get("enabled", False))
    if not detrend_enabled:
        peaks, marginal = _run_eva(waterlevel, peak_settings, fit_rps)
        detrend_meta = {
            "applied": False, "slope_m_per_year": 0.0,
            "reference_epoch_year": float("nan"), "slope_source": "none",
        }
        return peaks, marginal, fit_rps, detrend_meta

    # Pass 1: raw peaks to estimate the slope.
    raw_peaks, _ = _run_eva(waterlevel, peak_settings, fit_rps)
    slope, slope_source, slope_meta = _resolve_detrend_slope(detrend_cfg, raw_peaks, waterlevel)
    ref_year = _resolve_reference_epoch(detrend_cfg, waterlevel)

    # Detrend and re-extract on the same waterlevel transformed to the
    # reference epoch.
    detrended = detrend_hourly_to_reference_epoch(waterlevel, slope, ref_year)
    peaks, marginal = _run_eva(detrended, peak_settings, fit_rps)
    detrend_meta = {
        "applied": True,
        "slope_m_per_year": float(slope),
        "reference_epoch_year": float(ref_year),
        "slope_source": slope_source,
        "raw_peak_count": int(raw_peaks.dropna().size),
        "detrended_peak_count": int(peaks.dropna().size),
        **slope_meta,
    }
    return peaks, marginal, fit_rps, detrend_meta

def stationarity_report(peaks_series, detrend_meta=None):
    # Diagnose whether the historical peak record looks stationary.
    #
    # Regulator-facing rationale:
    # - The hybrid sampler treats every historical peak as exchangeable: the
    #   bootstrap body resamples uniformly from 1979-onward, and the GPD/exp
    #   tail fit pools all selected peaks. Both assumptions require
    #   stationarity of the peak distribution. Sea-level rise alone (Boston
    #   tide gauge ~3 mm/yr) injects ~13 cm of trend over a 44-year record,
    #   which is 5-8% of the typical Marshfield peak range and material to
    #   the fitted return-period curve.
    # - This function does not detrend or modify the record. Detrending is
    #   a scientific decision (which trend? linear in time? regressed on a
    #   reanalyzed MSL series?) that must be made by a human reviewer. We
    #   only surface the test statistics so the reviewer can decide.
    #
    # Tests reported (each on the selected peak series, indexed by event
    # time, not on hourly water level):
    # - Mann-Kendall: distribution-free monotonic trend test, computed via
    #   Kendall tau on (event_time_in_years, peak_height). Two-sided p-value.
    # - Theil-Sen: robust median-of-pairwise-slopes estimator with 95% CI.
    #   Reported in m/yr; compare against the local SLR rate as a sanity check.
    # - OLS slope: classical least-squares slope; reported only as a
    #   reference value, since OLS is sensitive to extremes by construction.
    #
    # Interpretation rule of thumb (NOT a gate):
    #   Mann-Kendall p < 0.05 -> trend is statistically detectable. The
    #   reviewer should compare the Theil-Sen slope to the local SLR rate
    #   and decide whether to (a) accept the trend as physical and detrend,
    #   (b) shorten the record to a stationary window, or (c) accept the
    #   trend as small relative to peak variance and document the residual.
    series = peaks_series.dropna()
    if len(series) < 3 or not isinstance(series.index, pd.DatetimeIndex):
        return {
            "n_peaks": int(len(series)),
            "note": "insufficient data or non-datetime index; stationarity test skipped",
        }
    t_years = (series.index - series.index.min()) / pd.Timedelta(days=365.25)
    t = np.asarray(t_years, dtype=float)
    y = series.to_numpy(dtype=float)
    mk = stats.kendalltau(t, y)
    ts = stats.theilslopes(y, t, 0.95)
    ols = stats.linregress(t, y)
    out = {
        "n_peaks": int(len(series)),
        "record_start": str(series.index.min().date()),
        "record_end": str(series.index.max().date()),
        "record_years": float(t.max()),
        "mann_kendall_tau": float(mk.statistic),
        "mann_kendall_p": float(mk.pvalue),
        "theil_sen_slope_m_per_year": float(ts.slope),
        "theil_sen_lo_m_per_year": float(ts.low_slope),
        "theil_sen_hi_m_per_year": float(ts.high_slope),
        "ols_slope_m_per_year": float(ols.slope),
        "ols_p": float(ols.pvalue),
        "interpretation": _interpret_trend(mk.pvalue, ts.slope, detrend_meta),
    }
    if detrend_meta:
        out["detrend"] = {
            "applied": bool(detrend_meta.get("applied", False)),
            "slope_m_per_year": float(detrend_meta.get("slope_m_per_year", 0.0)),
            "reference_epoch_year": float(detrend_meta.get("reference_epoch_year", float("nan"))),
            "slope_source": str(detrend_meta.get("slope_source", "none")),
            "annual_mean_year_count": int(detrend_meta.get("annual_mean_year_count", 0) or 0),
        }
    return out

def _interpret_trend(p, slope, detrend_meta=None):
    detrended = bool(detrend_meta and detrend_meta.get("applied"))
    if p >= 0.05:
        return (
            "No statistically detectable residual trend in detrended peaks; "
            "fit at reference epoch is defensible."
            if detrended
            else "No statistically detectable trend at alpha=0.05; pooled "
            "stationary fit is defensible."
        )
    if not detrended:
        if slope > 0:
            return (
                "Trend detectable at alpha=0.05 with positive slope; pooled "
                "stationary fit may UNDERSTATE forward risk (averages early "
                "and late MSL regimes). Review against local SLR rate and "
                "decide on detrending or restricting to a recent window."
            )
        return (
            "Trend detectable at alpha=0.05 with negative slope; pooled "
            "stationary fit may OVERSTATE forward risk (recent peaks lower "
            "than pooled mean). Review against local storm climatology."
        )
    # Detrending was applied but a residual trend remains. The most
    # common cause is that the slope source (e.g. theil_sen on POT peaks)
    # underestimates the true MSL trend at the gauge. Switching to an
    # external slope (e.g. NOAA published Boston Harbor mean trend)
    # typically eliminates the residual.
    direction = "positive" if slope > 0 else "negative"
    return (
        f"Residual {direction} trend detectable at alpha=0.05 after "
        f"detrending (source = {detrend_meta.get('slope_source')}). The "
        "applied slope appears to underestimate the true MSL trend; "
        "consider switching extremes.detrend.method to 'external' with a "
        "published gauge SLR rate to fully remove the secular component."
    )

def write_stationarity_report(paths, peaks_series, detrend_meta=None):
    # Persist the stationarity diagnostic next to the marginal fit.
    # The file is intentionally JSON, not embedded in marginal_params.csv,
    # to keep the test-statistic reporting separate from the fitted curve
    # parameters that downstream samplers actually consume. When
    # detrending has been applied, the report runs on the DETRENDED peak
    # series and exists to verify the detrend was effective.
    report = stationarity_report(peaks_series, detrend_meta)
    paths["stationarity_report_json"].parent.mkdir(parents=True, exist_ok=True)
    with paths["stationarity_report_json"].open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

def write_catalog(paths, peaks, marginal, fit_rps, detrend_meta=None):
    # Step 2b: save selected peaks and fitted curve for event building.
    # historical_peaks.csv stores the DETRENDED peaks (i.e. peaks at the
    # reference epoch), to match the marginal that the sampler will load.
    paths["catalog_root"].mkdir(parents=True, exist_ok=True)
    peaks.rename_axis("time").to_frame().to_csv(paths["historical_peaks_csv"])
    write_historical_peak_marginal(marginal, paths["marginal_params_csv"], detrend_meta)
    marginal_rps_frame(marginal, fit_rps).round(4).to_csv(paths["marginal_rps_csv"])

def write_return_value_plot(paths, peaks, marginal, rps, bootstrap=None):
    # Diagnostic plot: historical peaks vs fitted return-period curve.
    # If a bootstrap result is supplied, overlay a shaded confidence band
    # around the fitted curve so a reviewer can see distributional
    # uncertainty alongside the point estimate.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = peaks.dropna().values.astype(float)
    plot_return_values(
        x=x,
        params=np.asarray(marginal.params, dtype=float),
        distribution=marginal.dist_name,
        rps=rps,
        ax=ax,
        extremes_rate=marginal.extremes_rate,
    )
    if bootstrap is not None:
        cl_pct = int(round(100 * bootstrap["confidence_level"]))
        ax.fill_between(
            bootstrap["rps"], bootstrap["lo"], bootstrap["hi"],
            color="k", alpha=0.15, label=f"{cl_pct}% bootstrap CI",
        )
        ax.legend(loc="upper left")
    ax.set_ylabel("Water level [m+MSL]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths["marginal_plot_png"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["marginal_plot_png"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths["marginal_plot_png"]

def write_marginal_rps_ci(paths, marginal, rps, bootstrap):
    # Write the return-period table side-by-side with bootstrap CI bounds.
    # Schema: rps (years), h_point (fitted curve), h_lo / h_median / h_hi
    # (bootstrap percentiles at the configured confidence level). The
    # point estimate stays the value the sampler uses; lo/hi are review-only.
    rps = np.asarray(rps, dtype=float)
    df = pd.DataFrame(
        {
            "h_point": marginal.magnitude(rps),
            "h_lo": bootstrap["lo"],
            "h_median": bootstrap["median"],
            "h_hi": bootstrap["hi"],
        },
        index=pd.Index(rps, name="rps"),
    )
    paths["marginal_rps_ci_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(paths["marginal_rps_ci_csv"])

def write_marginal_bootstrap_meta(paths, bootstrap):
    # Persist non-tabular bootstrap metadata: confidence level, replicate
    # counts, and the empirical distribution-selection frequencies. The
    # last item answers "did AIC actually prefer the same family across
    # resamples, or is the family choice itself uncertain?" -- which a
    # reviewer should weigh when reading the CI.
    payload = {
        "confidence_level": bootstrap["confidence_level"],
        "n_replicates": bootstrap["n_replicates"],
        "n_succeeded": bootstrap["n_succeeded"],
        "distribution_counts": bootstrap["distribution_counts"],
    }
    paths["marginal_bootstrap_json"].parent.mkdir(parents=True, exist_ok=True)
    with paths["marginal_bootstrap_json"].open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _bootstrap_settings(config, marginal):
    # Resolve bootstrap settings + the candidate distribution list that
    # MUST match what the production fit used, so the bootstrap exercises
    # the same model-selection step rather than locking in the chosen family.
    extremes = config.get("extremes", {})
    bootstrap_cfg = extremes.get("bootstrap", {})
    if marginal.method == "pot":
        candidates = list(extremes.get("pot", {}).get("distributions", ["exp", "gpd"]))
    else:
        candidates = list(extremes.get("bm", {}).get("distributions", ["gumb", "gev"]))
    distribution = None if len(candidates) > 1 else candidates[0]
    return {
        "ev_type": marginal.method,
        "distribution": distribution,
        "criterium": str(extremes.get("selection_criterion", "AIC")),
        "n_replicates": int(bootstrap_cfg.get("n_replicates", 1000)),
        "confidence_level": float(bootstrap_cfg.get("confidence_level", 0.95)),
        "seed": int(bootstrap_cfg.get("seed", 42)),
    }

def build_catalog(config, paths):
    # step 2c: hourly water level -> historical peaks -> return-period curve.
    waterlevel = load_hourly_waterlevel(paths["waterlevel_csv"])
    peaks, marginal, fit_rps, detrend_meta = fit_historical_peaks(config, waterlevel)
    write_catalog(paths, peaks, marginal, fit_rps, detrend_meta)
    if detrend_meta.get("applied"):
        print(
            f"[detrend] applied {detrend_meta['slope_source']} slope = "
            f"{detrend_meta['slope_m_per_year']*1000:+.2f} mm/yr, "
            f"reference epoch = {detrend_meta['reference_epoch_year']:.1f}"
        )
    # Stationarity diagnostic: runs on the DETRENDED peak series and is
    # the validation that detrending worked. After successful detrending,
    # Mann-Kendall p should rise above 0.05 (no remaining trend).
    report = write_stationarity_report(paths, peaks, detrend_meta)
    # Distributional uncertainty: nonparametric bootstrap of the return-
    # period curve. Runs the SAME model-selection step inside each
    # replicate so the resulting CI captures both parameter variance and
    # family-selection variance. Sampler is unchanged (still uses the
    # point estimate); the CI is a review artifact.
    bs_cfg = _bootstrap_settings(config, marginal)
    bootstrap = bootstrap_return_values(
        peaks.dropna().to_numpy(dtype=float),
        bs_cfg["ev_type"],
        fit_rps,
        marginal.extremes_rate,
        distribution=bs_cfg["distribution"],
        criterium=bs_cfg["criterium"],
        n_replicates=bs_cfg["n_replicates"],
        confidence_level=bs_cfg["confidence_level"],
        seed=bs_cfg["seed"],
    )
    if bootstrap is not None:
        write_marginal_rps_ci(paths, marginal, fit_rps, bootstrap)
        write_marginal_bootstrap_meta(paths, bootstrap)
    if report.get("mann_kendall_p", 1.0) < 0.05:
        slope_mm = report["theil_sen_slope_m_per_year"] * 1000
        if detrend_meta.get("applied"):
            # Detrending ran but a residual trend persists: the slope
            # source underestimated the true MSL trend.
            print(
                f"[stationarity] WARNING: residual trend after detrend = "
                f"{slope_mm:+.2f} mm/yr (Mann-Kendall p="
                f"{report['mann_kendall_p']:.4f}). Slope source = "
                f"{detrend_meta['slope_source']!r} appears to underestimate "
                "the true MSL trend. Switch extremes.detrend.method to "
                "'external' with a published gauge SLR rate to fully remove "
                "the secular component."
            )
        else:
            direction = (
                "understate forward risk (positive trend; pooled fit averages "
                "early and late MSL regimes)"
                if slope_mm > 0
                else "overstate forward risk (negative trend; recent peaks are "
                "lower than the pooled mean)"
            )
            print(
                f"[stationarity] WARNING: Mann-Kendall p={report['mann_kendall_p']:.4f} "
                f"on {report['n_peaks']} peaks; Theil-Sen slope = "
                f"{slope_mm:.2f} mm/yr. Pooled-stationary fit may "
                f"{direction}; review against local SLR rate before "
                "defending the dataset."
            )
    fig_path = write_return_value_plot(
        paths,
        peaks,
        marginal,
        np.array(config.get("extremes", {}).get("return_periods", rps_default), dtype=float),
        bootstrap=bootstrap,
    )
    return {
        "waterlevel": waterlevel,
        "historical_peaks": peaks,
        "marginal": marginal,
        "figure": fig_path,
        "stationarity": report,
        "bootstrap": bootstrap,
    }

def build_threshold_model_sensitivity(config, paths):
    # Review-only sensitivity table. It varies POT threshold, peak spacing,
    # and tail model, but does not write production catalog/event artifacts.
    waterlevel = load_hourly_waterlevel(paths["waterlevel_csv"])
    extremes = config.get("extremes", {})
    q_values = extremes.get("sensitivity_threshold_quantiles", [0.95, 0.97, 0.98, 0.99])
    distance_values = extremes.get("sensitivity_min_peak_distance_hours", [48, 72, 96])
    model_values = extremes.get("sensitivity_models", ["exp", "gpd", "aic"])
    report_rps = np.array(extremes.get("return_periods", rps_default), dtype=float)

    rows = []
    for q in q_values:
        for min_dist in distance_values:
            for model in model_values:
                cfg = deepcopy(config)
                pot = cfg.setdefault("extremes", {}).setdefault("pot", {})
                pot["threshold_quantile"] = float(q)
                pot["min_peak_distance_hours"] = int(min_dist)
                pot["distributions"] = ["exp", "gpd"] if str(model).lower() == "aic" else [str(model)]
                peaks, marginal, _, detrend_meta = fit_historical_peaks(cfg, waterlevel)
                stationarity = stationarity_report(peaks, detrend_meta)
                row = {
                    "threshold_quantile": float(q),
                    "min_peak_distance_hours": int(min_dist),
                    "model_setting": str(model),
                    "selected_distribution": marginal.dist_name,
                    "peak_count": int(marginal.peak_count),
                    "extremes_rate": float(marginal.extremes_rate),
                    "detrend_slope_m_per_year": float(detrend_meta.get("slope_m_per_year", 0.0)),
                    "residual_mann_kendall_p": float(stationarity.get("mann_kendall_p", np.nan)),
                    "residual_theil_sen_slope_m_per_year": float(stationarity.get("theil_sen_slope_m_per_year", np.nan)),
                }
                for rp, h in zip(report_rps, marginal.magnitude(report_rps)):
                    row[f"h_{int(rp)}yr_m"] = float(h)
                rows.append(row)

    df = pd.DataFrame(rows)
    paths["sensitivity_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(paths["sensitivity_csv"], index=False)
    return df
