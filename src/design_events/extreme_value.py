"""Extreme-value fitting: POT / block-maxima tails, AIC/BIC selection, return values.
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

def peak_series(da, min_dist):
    series = da.to_series().dropna()
    peaks, _ = find_peaks(series.to_numpy(), distance=max(1, int(min_dist)))
    out = pd.Series(np.nan, index=series.index, name="peaks")
    out.iloc[peaks] = series.iloc[peaks]
    return xr.DataArray(out, dims="time", coords={"time": out.index}, name="peaks")

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

