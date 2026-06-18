from __future__ import annotations
from functools import cached_property
import numpy as np
import pandas as pd

from .extreme_value import get_frozen_dist
clip_eps = 1e-9

class HistoricalPeakMarginal:
    """Fitted curve that converts peak height <-> return period."""
    def __init__(self, dist_name, params, extremes_rate, method, threshold_quantile, peak_count):
        self.dist_name = str(dist_name)
        self.params = tuple(float(p) for p in params)
        self.extremes_rate = float(extremes_rate)
        self.method = str(method)
        self.threshold_quantile = float(threshold_quantile)
        self.peak_count = int(peak_count)
        if not (np.isfinite(self.extremes_rate) and self.extremes_rate > 0):
            raise ValueError(f"extremes_rate must be finite and > 0, got {self.extremes_rate!r}")
        if not all(np.isfinite(p) for p in self.params):
            raise ValueError(f"params must all be finite, got {self.params!r}")
        get_frozen_dist(self.params, self.dist_name)

    @cached_property
    def _frozen_dist(self):
        return get_frozen_dist(self.params, self.dist_name)

    def magnitude(self, return_period):
        # Return period -> annual exceedance probability -> peak height.
        arr = np.asarray(return_period, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(arr > 0, 1.0 / arr / self.extremes_rate, np.nan)
        q = np.clip(q, clip_eps, 1 - clip_eps)
        out = np.asarray(self._frozen_dist.isf(q), dtype=float)
        out = np.where(np.isnan(arr) | (arr <= 0), np.nan, out)
        return out.item() if np.ndim(return_period) == 0 else out

    def return_period(self, magnitude):
        # Peak height -> annual exceedance probability -> return period.
        arr = np.asarray(magnitude, dtype=float)
        p = np.asarray(self._frozen_dist.cdf(arr), dtype=float)
        q = np.clip(1.0 - p, clip_eps, 1.0)
        out = 1.0 / (q * self.extremes_rate)
        out = np.where(np.isnan(arr), np.nan, out)
        return out.item() if np.ndim(magnitude) == 0 else out

    def cdf(self, magnitude):
        # Marginal CDF F(x); the uniform Driver Probability Index used by the copula stage.
        arr = np.asarray(magnitude, dtype=float)
        out = np.asarray(self._frozen_dist.cdf(arr), dtype=float)
        return out.item() if np.ndim(magnitude) == 0 else out

    def ppf(self, u):
        # Marginal quantile F^-1(u); maps a copula uniform back to a physical magnitude.
        arr = np.clip(np.asarray(u, dtype=float), clip_eps, 1.0 - clip_eps)
        out = np.asarray(self._frozen_dist.ppf(arr), dtype=float)
        return out.item() if np.ndim(u) == 0 else out

    def pdf(self, magnitude):
        # Marginal density f(x).
        arr = np.asarray(magnitude, dtype=float)
        out = np.asarray(self._frozen_dist.pdf(arr), dtype=float)
        return out.item() if np.ndim(magnitude) == 0 else out

def _scalar_coord(da, name, default):
    # Xarray stores fitted metadata as coordinates; unwrap one scalar value.
    coord = da.coords.get(name)
    values = np.asarray(default if coord is None else coord.values)
    return default if values.size == 0 else values.reshape(-1)[0].item()

def from_eva_dataset(ds_eva, *, method, threshold_quantile=float("nan")):
    # Convert extreme-value output into the small object used downstream.
    params = ds_eva["parameters"]
    return HistoricalPeakMarginal(
        dist_name=str(_scalar_coord(params, "distribution", "gev")),
        params=tuple(float(v) for v in np.asarray(params.values).reshape(-1)),
        extremes_rate=float(_scalar_coord(params, "extremes_rate", 1.0)),
        method=method,
        threshold_quantile=float(threshold_quantile),
        peak_count=int(ds_eva["peaks"].notnull().sum().item()),
    )

def marginal_params_frame(marginal, detrend_meta=None):
    # Single source of truth for the marginal-params CSV schema. Used by
    # both the on-disk writer and the notebook display path so they stay
    # in lockstep when fields are added (e.g. detrend metadata).
    p = list(marginal.params) + [np.nan] * max(0, 3 - len(marginal.params))
    row = {
        "dist": marginal.dist_name,
        "shape": float(p[0]),
        "loc": float(p[1]),
        "scale": float(p[2]),
        "extremes_method": marginal.method,
        "extremes_rate": marginal.extremes_rate,
        "threshold_quantile": marginal.threshold_quantile,
        "peak_count": marginal.peak_count,
    }
    meta = detrend_meta or {}
    row.update({
        "detrend_applied": bool(meta.get("applied", False)),
        "detrend_slope_m_per_year": float(meta.get("slope_m_per_year", 0.0)),
        "detrend_reference_epoch_year": float(meta.get("reference_epoch_year", float("nan"))),
        "detrend_slope_source": str(meta.get("slope_source", "none")),
        "detrend_annual_mean_year_count": int(meta.get("annual_mean_year_count", 0) or 0),
    })
    return pd.DataFrame([row], index=["h"])

def marginal_rps_frame(marginal, rps):
    # Return-period table as a DataFrame indexed by RP, used by both the
    # on-disk writer and the notebook display path.
    rps = np.asarray(rps, dtype=float)
    return pd.DataFrame({"h": marginal.magnitude(rps)}, index=pd.Index(rps, name="rps"))

def write_historical_peak_marginal(marginal, path, detrend_meta=None):
    # Save one row so the sampler can reload the fitted curve later.
    # Detrend metadata is persisted on the same row so a reviewer can
    # reconstruct exactly what secular trend was removed before the fit
    # and what reference epoch the curve represents.
    path.parent.mkdir(parents=True, exist_ok=True)
    marginal_params_frame(marginal, detrend_meta).to_csv(path)

def load_historical_peak_marginal(path):
    # Rebuild the return-period curve from the saved CSV.
    df = pd.read_csv(path, index_col=0)
    row = df.loc["h"]
    params = tuple(float(v) for v in row[["shape", "loc", "scale"]].dropna().values)
    return HistoricalPeakMarginal(
        dist_name=str(row["dist"]),
        params=params,
        extremes_rate=float(row["extremes_rate"]),
        method=str(row["extremes_method"]),
        threshold_quantile=float(row.get("threshold_quantile", float("nan"))),
        peak_count=int(row.get("peak_count", 0) or 0),
    )


class EmpiricalMarginal:
    """Bounded empirical-CDF marginal for state/antecedent drivers (e.g. soil moisture).

    Unlike a POT exp/GPD tail, this does not extrapolate beyond the observed sample: the
    quantile function saturates at the observed [min, max]. Appropriate for a bounded
    conditioning/antecedent variable such as soil saturation fraction, where an unbounded
    extreme-value tail produces unphysical values (>1 saturation). Jane et al. (2020) fit
    non-conditioning compound-flood drivers to bounded distributions, not a GPD tail.
    """

    dist_name = "empirical"

    def __init__(self, values):
        v = np.asarray(values, dtype=float)
        v = np.sort(v[np.isfinite(v)])
        if v.size < 2:
            raise ValueError("EmpiricalMarginal needs at least 2 finite values")
        self.values = v
        self.peak_count = int(v.size)
        # Weibull plotting positions: strictly interior so cdf/ppf invert cleanly.
        self._p = np.arange(1, v.size + 1) / (v.size + 1.0)

    def cdf(self, x):
        arr = np.asarray(x, dtype=float)
        out = np.interp(arr, self.values, self._p, left=0.0, right=1.0)
        return out.item() if np.ndim(x) == 0 else out

    def ppf(self, q):
        arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        out = np.interp(arr, self._p, self.values, left=self.values[0], right=self.values[-1])
        return out.item() if np.ndim(q) == 0 else out

    def pdf(self, x):
        arr = np.asarray(x, dtype=float)
        density = np.gradient(self._p, self.values)
        out = np.interp(arr, self.values, density, left=0.0, right=0.0)
        return out.item() if np.ndim(x) == 0 else out
