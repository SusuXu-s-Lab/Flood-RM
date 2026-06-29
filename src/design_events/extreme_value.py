"""Extreme-value fitting: POT / block-maxima tails, AIC/BIC selection, return values.

The low-level extreme-value layer under the ``Marginal`` classes (ADR-0021): fit an
AIC/BIC-selected tail (Exp/GPD for POT, Gumbel/GEV for block maxima), convert between
return period and magnitude, and bootstrap return-value confidence bands. Pure
scipy/numpy/pandas; ``plot_return_values`` imports matplotlib lazily so headless runs
(catalog builds, audits) never pull in a display backend.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.signal import find_peaks

# Return periods to report on the fitted curve, in years.
rps_default = np.array([2, 5, 10, 25, 50, 100, 250, 500])

# Candidate tail distributions by extreme-value method.
# POT = peaks over threshold: exponential or generalized Pareto.
# BM = block maxima: Gumbel or generalized extreme value.
candidate_distributions = {
    "pot": ["exp", "gpd"],
    "bm": ["gumb", "gev"],
}

# exp: exponential tail, represented as genpareto with zero shape
# gpd: generalized Pareto distribution
# gumb: Gumbel distribution for block maxima
# gev: generalized extreme value distribution
scipy_name = {
    "exp": "genpareto",
    "gpd": "genpareto",
    "gumb": "gumbel_r",
    "gev": "genextreme",
}

def normalize_time_frequency(freq):
    # Pandas renamed some annual aliases; keep old config files working.
    if not isinstance(freq, str):
        return freq
    aliases = {"AS": "YS", "A": "YE"}
    base, sep, suffix = freq.partition("-")
    return f"{aliases.get(base, base)}{sep}{suffix}" if sep else aliases.get(base, base)

def eva_peaks_over_threshold(da, qthresh=0.9, min_dist=0, period="365.25D",
                             distribution=None, rps=rps_default, criterium="AIC",
                             min_sample_size=0):
    # POT: keep local peaks above a high historical quantile.
    peaks = peak_series(da, min_dist)
    threshold = float(da.quantile(qthresh))
    peaks = peaks.where(peaks > threshold)
    return fit_peak_dataset(peaks, "pot", period, distribution, rps, criterium)

def eva_block_maxima(da, period="365.25D", distribution=None, rps=rps_default,
                     criterium="AIC", min_dist=0, min_sample_size=0):
    # Block maxima: keep one largest value per period.
    series = da.to_series().dropna()
    peaks = series.resample(normalize_time_frequency(period)).max()
    peaks = xr.DataArray(peaks, dims="time", coords={"time": peaks.index}, name="peaks")
    return fit_peak_dataset(peaks, "bm", period, distribution, rps, criterium)

def peak_series(da, min_dist):
    series = da.to_series().dropna()
    peaks, _ = find_peaks(series.to_numpy(), distance=max(1, int(min_dist)))
    out = pd.Series(np.nan, index=series.index, name="peaks")
    out.iloc[peaks] = series.iloc[peaks]
    return xr.DataArray(out, dims="time", coords={"time": out.index}, name="peaks")

def fit_peak_dataset(peaks, ev_type, period, distribution, rps, criterium):
    # One dataset goes downstream: selected peaks, fitted parameters, return values.
    values = peaks.to_series().dropna().to_numpy(dtype=float)
    years = max(1, n_periods(peaks["time"].to_index(), period))
    extremes_rate = len(values) / years
    params, dist_name = fit_best_distribution(values, ev_type, distribution, criterium)
    params_da = xr.DataArray(
        params,
        dims="dparams",
        coords={
            "dparams": ["shape", "loc", "scale"],
            "distribution": dist_name,
            "extremes_rate": extremes_rate,
        },
        name="parameters",
    )
    rv_da = xr.DataArray(
        return_values(params, dist_name, rps, extremes_rate),
        dims="rps",
        coords={"rps": rps},
        name="return_values",
    )
    peaks = peaks.assign_coords(extremes_rate=extremes_rate)
    return xr.merge([peaks, params_da, rv_da], compat="override")

def n_periods(time_index, period):
    period = normalize_time_frequency(period)
    if len(time_index) == 0:
        return 1
    return len(pd.Series(1, index=time_index).resample(period).first().dropna())

def fit_best_distribution(values, ev_type, distribution=None, criterium="AIC"):
    choices = [distribution] if distribution else candidate_distributions[ev_type]
    fits = [(fit_distribution(values, dist), dist) for dist in choices]
    return min(fits, key=lambda item: score_fit(values, item[0], item[1], criterium))

def fit_distribution(values, dist_name):
    # Store every distribution as shape, loc, scale so CSV output stays simple.
    if dist_name == "exp":
        loc, scale = stats.expon.fit(values)
        return np.array([0.0, loc, scale])
    if dist_name == "gumb":
        loc, scale = stats.gumbel_r.fit(values)
        return np.array([0.0, loc, scale])
    return np.array(get_dist(dist_name).fit(values), dtype=float)

def score_fit(values, params, dist_name, criterium):
    # AIC/BIC reward fit quality but penalize extra parameters.
    k = len(params)
    loglik = float(np.sum(get_frozen_dist(params, dist_name).logpdf(values)))
    if criterium.upper() == "BIC":
        return k * np.log(len(values)) - 2 * loglik
    return 2 * k - 2 * loglik

def get_dist(dist_name):
    return getattr(stats, scipy_name.get(dist_name, dist_name))

def get_frozen_dist(params, dist_name):
    shape, loc, scale = np.asarray(params, dtype=float)
    if dist_name == "gumb":
        return stats.gumbel_r(loc=loc, scale=scale)
    return get_dist(dist_name)(shape, loc=loc, scale=scale)

def return_values(params, dist_name, rps=rps_default, extremes_rate=1.0):
    # return period uses annual exceedance rate, not plain quantile.
    q = 1 / np.asarray(rps, dtype=float) / extremes_rate
    return get_frozen_dist(params, dist_name).isf(q)

def bootstrap_return_values(values, ev_type, rps, extremes_rate, *,
                            distribution=None, criterium="AIC",
                            n_replicates=1000, confidence_level=0.95, seed=0):
    rng = np.random.default_rng(int(seed))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rps = np.asarray(rps, dtype=float)
    n = len(values)
    if n < 3 or n_replicates <= 0:
        return None
    rvs = np.full((int(n_replicates), len(rps)), np.nan, dtype=float)
    dist_counts = {}
    for b in range(int(n_replicates)):
        sample = values[rng.integers(0, n, size=n)]
        try:
            params, dist_name = fit_best_distribution(sample, ev_type, distribution, criterium)
            rvs[b] = return_values(params, dist_name, rps, extremes_rate)
            dist_counts[dist_name] = dist_counts.get(dist_name, 0) + 1
        except Exception:
            # Some bootstrap samples can produce degenerate fits (e.g. all
            # identical values, scipy convergence failure). Drop the
            # replicate rather than corrupting the percentile estimates.
            continue
    alpha = 1.0 - float(confidence_level)
    finite_mask = np.all(np.isfinite(rvs), axis=1)
    rvs_ok = rvs[finite_mask]
    if len(rvs_ok) == 0:
        return None
    return {
        "rps": rps,
        "lo": np.quantile(rvs_ok, alpha / 2.0, axis=0),
        "median": np.quantile(rvs_ok, 0.5, axis=0),
        "hi": np.quantile(rvs_ok, 1.0 - alpha / 2.0, axis=0),
        "confidence_level": float(confidence_level),
        "n_replicates": int(n_replicates),
        "n_succeeded": int(len(rvs_ok)),
        "distribution_counts": dist_counts,
    }

def observed_return_periods(values, extremes_rate=1.0):
    ranks = len(values) + 1 - stats.rankdata(values, method="average")
    exceedance_rate = ranks / (len(values) + 1) * extremes_rate
    return 1 / exceedance_rate

def plot_return_values(x, params, distribution, ax=None, rps=rps_default,
                       color="k", extremes_rate=1.0, **kwargs):
    import matplotlib.pyplot as plt # always ensures compatibility
    observed = np.sort(x[np.isfinite(x)])
    fitted = return_values(params, distribution, rps, extremes_rate)
    observed_rps = observed_return_periods(observed, extremes_rate)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(observed_rps, observed, color=color, marker="o", lw=0, label="historical peaks")
    ax.plot(rps, fitted, color=color, ls="--", label=f"{distribution.upper()} fit")
    ax.set_xscale("log")
    ax.set_xlabel("Return period [years]")
    ax.set_ylabel("Return value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return ax
