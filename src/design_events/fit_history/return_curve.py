"""Fitted marginal curves (peak height <-> return period) and their on-disk schema.

The marginal classes are the single source of truth in ``design_events_v2.records``
(ADR-0021): this module re-exports them so production keeps one implementation while
``design_events_v2`` stays standalone. The EVA-dataset adapter and the params/RP CSV
schema (consumed by ``fit_history.peaks`` and ``build_events.coastal``) stay here.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from design_events_v2.records import EmpiricalMarginal, HistoricalPeakMarginal

clip_eps = 1e-9

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
    # Save one row so the sampler can reload the fitted curve; detrend metadata rides on the
    # same row so a reviewer sees what trend was removed and at what reference epoch.
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


__all__ = [
    "EmpiricalMarginal",
    "HistoricalPeakMarginal",
    "from_eva_dataset",
    "marginal_params_frame",
    "marginal_rps_frame",
    "write_historical_peak_marginal",
    "load_historical_peak_marginal",
]
