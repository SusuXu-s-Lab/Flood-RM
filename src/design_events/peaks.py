from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from design_events.records import (
    marginal_rps_frame,
    write_historical_peak_marginal,
)

def load_hourly_waterlevel(path):
    # Step 1: Cora data collector writes one hourly value column.
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return pd.to_numeric(frame["value"], errors="coerce").sort_index()

def stationarity_report(peaks_series, detrend_meta=None):
    # Interpretation rule of thumb (not a gate): Mann-Kendall p < 0.05 -> trend is
    # statistically detectable; the reviewer compares the Theil-Sen slope to the local SLR rate.
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
    # Detrending applied but a residual trend remains -- usually the slope source
    # underestimates the true MSL trend; an external published slope typically removes it.
    direction = "positive" if slope > 0 else "negative"
    return (
        f"Residual {direction} trend detectable at alpha=0.05 after "
        f"detrending (source = {detrend_meta.get('slope_source')}). The "
        "applied slope appears to underestimate the true MSL trend; "
        "consider switching extremes.detrend.method to 'external' with a "
        "published gauge SLR rate to fully remove the secular component."
    )

def write_catalog(paths, peaks, marginal, fit_rps, detrend_meta=None):
    # Step 2b: save selected peaks and fitted curve for event building.
    # historical_peaks.csv stores DETRENDED peaks (at the reference epoch) to match the marginal.
    paths["catalog_root"].mkdir(parents=True, exist_ok=True)
    peaks.rename_axis("time").to_frame().to_csv(paths["historical_peaks_csv"])
    write_historical_peak_marginal(marginal, paths["marginal_params_csv"], detrend_meta)
    marginal_rps_frame(marginal, fit_rps).round(4).to_csv(paths["marginal_rps_csv"])

