"""
From real driver records to fitted marginals, the paired POT sample, and member libraries.

1. **Marginals** ``F_j``: AIC-selected POT/BM extreme-value tails for the stochastic
   forcing drivers, a bounded empirical CDF for state/antecedent drivers.
   
2. **Two-Sided Conditional POT Co-occurrence Sample**: condition on each driver, decluster
   its peaks, pair the concurrent maxima of the others within a window; carry the
   *distinct-storm* rate so ``T = 1/(rate*S)`` is not inflated by the two-sided row count.
   
3. **Member libraries**: per-timestamp/per-peak field pointers the realization scales.

Coastal water level enters as **non-tidal residual** (NTR/surge), never total water level:
the copula axis and the realization use NTR while the astronomical tide is added back
unscaled downstream.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

clip_eps = 1e-9

# --------------------------------------------------------------------------- #
# Marginals F_j and their extreme-value fit                                  #
# --------------------------------------------------------------------------- #
# The extreme-value fit is the single source of truth in design_events.extreme_value
# and the Marginal classes below build on it.
from design_events.extreme_value import fit_best_distribution, get_frozen_dist


class HistoricalPeakMarginal:
    """Fitted extreme-value curve: peak height <-> return period, plus cdf/ppf/pdf (F_j)."""

    dist_name: str

    def __init__(self, dist_name, params, extremes_rate, method="pot", threshold_quantile=float("nan"), peak_count=0):
        self.dist_name = str(dist_name)
        self.params = tuple(float(p) for p in params)
        self.extremes_rate = float(extremes_rate)
        self.method = str(method)
        self.threshold_quantile = float(threshold_quantile)
        self.peak_count = int(peak_count)
        if not (np.isfinite(self.extremes_rate) and self.extremes_rate > 0):
            raise ValueError(f"extremes_rate must be finite and > 0, got {self.extremes_rate!r}")
        get_frozen_dist(self.params, self.dist_name)

    @cached_property
    def _dist(self):
        return get_frozen_dist(self.params, self.dist_name)

    def magnitude(self, return_period):  # RP -> AEP -> height
        arr = np.asarray(return_period, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(arr > 0, 1.0 / arr / self.extremes_rate, np.nan)
        q = np.clip(q, clip_eps, 1 - clip_eps)
        out = np.where(np.isnan(arr) | (arr <= 0), np.nan, np.asarray(self._dist.isf(q), dtype=float))
        return out.item() if np.ndim(return_period) == 0 else out

    def return_period(self, magnitude):  # height -> AEP -> RP
        arr = np.asarray(magnitude, dtype=float)
        q = np.clip(1.0 - np.asarray(self._dist.cdf(arr), dtype=float), clip_eps, 1.0)
        out = np.where(np.isnan(arr), np.nan, 1.0 / (q * self.extremes_rate))
        return out.item() if np.ndim(magnitude) == 0 else out

    def cdf(self, magnitude):
        out = np.asarray(self._dist.cdf(np.asarray(magnitude, dtype=float)), dtype=float)
        return out.item() if np.ndim(magnitude) == 0 else out

    def ppf(self, u):
        arr = np.clip(np.asarray(u, dtype=float), clip_eps, 1.0 - clip_eps)
        out = np.asarray(self._dist.ppf(arr), dtype=float)
        return out.item() if np.ndim(u) == 0 else out

    def pdf(self, magnitude):
        out = np.asarray(self._dist.pdf(np.asarray(magnitude, dtype=float)), dtype=float)
        return out.item() if np.ndim(magnitude) == 0 else out


class EmpiricalMarginal:
    """Bounded empirical-CDF marginal for a state/antecedent driver (e.g. soil moisture).

    The quantile function saturates at the observed [min, max] — no unphysical tail
    extrapolation (>1 saturation). Weibull plotting positions keep cdf/ppf invertible.
    """

    dist_name = "empirical"

    def __init__(self, values):
        v = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
        if v.size < 2:
            raise ValueError("EmpiricalMarginal needs at least 2 finite values")
        self.values = v
        self.peak_count = int(v.size)
        self._p = np.arange(1, v.size + 1) / (v.size + 1.0)

    def cdf(self, x):
        out = np.interp(np.asarray(x, dtype=float), self.values, self._p, left=0.0, right=1.0)
        return out.item() if np.ndim(x) == 0 else out

    def ppf(self, q):
        arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        out = np.interp(arr, self._p, self.values, left=self.values[0], right=self.values[-1])
        return out.item() if np.ndim(q) == 0 else out

    def pdf(self, x):
        density = np.gradient(self._p, self.values)
        out = np.interp(np.asarray(x, dtype=float), self.values, density, left=0.0, right=0.0)
        return out.item() if np.ndim(x) == 0 else out

    def return_period(self, x):
        q = np.clip(1.0 - np.asarray(self.cdf(x), dtype=float), clip_eps, 1.0)
        return 1.0 / q  # rate-free; state drivers are not used for univariate RP


def fit_marginal(values, *, extremes_rate, kind="pot", ev_type="pot", criterium="AIC"):
    """Fit a role-aware marginal ``F_j`` for one Driver Probability Index.

    ``kind="pot"`` -> AIC-selected Exp/GPD tail (forcing drivers: rainfall, NTR, discharge).
    ``kind="empirical"`` -> bounded empirical CDF (state drivers: soil saturation fraction).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        raise ValueError("need at least 3 finite values to fit a driver marginal")
    if kind == "empirical":
        return EmpiricalMarginal(v)
    if kind != "pot":
        raise ValueError(f"unknown marginal kind {kind!r}; use 'pot' or 'empirical'")
    params, dist_name = fit_best_distribution(v, ev_type, criterium=criterium)
    return HistoricalPeakMarginal(dist_name, params, extremes_rate, method=ev_type, peak_count=int(v.size))


# --------------------------------------------------------------------------- #
# Coastal NTR (surge) split                                                   #
# --------------------------------------------------------------------------- #
def coastal_components(waterlevel, *, latitude, msl_window="30D", msl_min_periods=200):
    """Split total water level into mean sea level + astronomical tide + NTR (surge).

    ``msl`` = 30-day centered moving average (removes secular MSL trend + seasonal cycle);
    ``tide`` = utide harmonic reconstruction of the MSL-removed anomaly; ``ntr = wl-msl-tide``.
    The copula axis and realization use ``ntr``; the tide is preserved unscaled downstream.
    """
    import utide

    s = pd.Series(waterlevel).dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("waterlevel must have a DatetimeIndex")
    msl = s.rolling(msl_window, center=True, min_periods=msl_min_periods).mean()
    anomaly = (s - msl).dropna()
    coef = utide.solve(anomaly.index.values, anomaly.to_numpy(dtype=float),
                       lat=float(latitude), conf_int="none", trend=False, verbose=False)
    tide = pd.Series(utide.reconstruct(anomaly.index.values, coef, verbose=False).h, index=anomaly.index)
    ntr = anomaly - tide
    return pd.DataFrame({"wl": s.reindex(anomaly.index), "msl": msl.reindex(anomaly.index), "tide": tide, "ntr": ntr})


def non_tidal_residual(waterlevel, *, latitude, **kwargs):
    """Just the NTR (surge) Series — the coastal copula axis and realization index."""
    return coastal_components(waterlevel, latitude=latitude, **kwargs)["ntr"]


# --------------------------------------------------------------------------- #
# Two-sided conditional POT co-occurrence sample                             #
# --------------------------------------------------------------------------- #
def _scalar_at_time(series, time):
    value = pd.Series(series).loc[pd.Timestamp(time)]
    if isinstance(value, pd.Series):
        return float(pd.to_numeric(value, errors="coerce").max())
    return float(value)


def declustered_pot_peaks(series, *, threshold=None, threshold_quantile=0.98, min_separation_hours=120.0):
    """Declustered peaks-over-threshold: greedily keep the largest exceedances no closer
    than ``min_separation_hours``. Returns a frame with ``time`` and ``value``."""
    s = pd.Series(series).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    s = s.sort_index()
    thr = float(threshold) if threshold is not None else float(s.quantile(threshold_quantile))
    exceedances = s[s > thr]
    if exceedances.empty:
        return pd.DataFrame({"time": pd.to_datetime([]), "value": pd.Series([], dtype=float)})
    separation = pd.Timedelta(hours=float(min_separation_hours))
    selected = []
    for time, _ in exceedances.sort_values(ascending=False, kind="stable").items():
        if all(abs(time - chosen) >= separation for chosen in selected):
            selected.append(time)
    selected = sorted(selected)
    return pd.DataFrame({"time": selected, "value": [_scalar_at_time(exceedances, t) for t in selected]})


def calibrate_threshold_for_rate(series, target_rate_per_year, *, min_separation_hours=120.0, search_quantile=0.90):
    """Pick the POT threshold whose declustered peaks occur at ~``target_rate_per_year``.
    Returns ``(threshold, peaks_frame, record_years)``."""
    s = pd.Series(series).dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    record_years = (s.index[-1] - s.index[0]).total_seconds() / (365.25 * 86400.0)
    target_count = max(1, int(round(float(target_rate_per_year) * record_years)))
    candidates = s[s > s.quantile(search_quantile)].sort_values(ascending=False, kind="stable")
    separation = pd.Timedelta(hours=float(min_separation_hours))
    times, values = [], []
    for time, value in candidates.items():
        if all(abs(time - chosen) >= separation for chosen in times):
            times.append(time)
            values.append(float(value))
            if len(times) >= target_count:
                break
    threshold = values[-1] if values else float(s.quantile(search_quantile))
    peaks = pd.DataFrame({"time": times, "value": values}).sort_values("time").reset_index(drop=True)
    return threshold, peaks, record_years


def distinct_event_rate(peak_times_by_driver, *, min_separation_hours, record_years):
    """Rate of distinct storms across conditioning drivers (events/yr): the union of
    conditioning peaks with peaks within ``min_separation_hours`` merged into one event."""
    all_times = sorted(t for times in peak_times_by_driver.values() for t in times)
    if not all_times or not (record_years and record_years > 0):
        return float("nan")
    separation = pd.Timedelta(hours=float(min_separation_hours))
    distinct, last = 0, None
    for t in all_times:
        if last is None or (t - last) >= separation:
            distinct += 1
            last = t
    return distinct / record_years


def build_paired_observations(drivers, *, driver_names=None, condition_on=None, target_rate_per_year=None,
                              threshold_quantiles=0.98, decluster_window_hours=120.0,
                              pairing_window_hours=72.0, dropna=True):
    """Two-sided conditional POT co-occurrence sample over multiple drivers.

    ``drivers`` = DataFrame (DatetimeIndex, one column per driver) or name->Series mapping.
    Condition on each driver in turn; every declustered peak is paired with the concurrent
    maximum of the others within ``pairing_window_hours``. Records the distinct-storm rate on
    ``out.attrs['base_event_rate_per_year']`` for ``T = 1/(rate*S)``.
    """
    if isinstance(drivers, pd.DataFrame):
        names = list(driver_names) if driver_names is not None else list(drivers.columns)
        series = {name: pd.Series(drivers[name]).dropna().sort_index() for name in names}
    else:
        series = {name: pd.Series(values).dropna().sort_index() for name, values in drivers.items()}
        names = list(driver_names) if driver_names is not None else list(series.keys())
    for name in names:
        if not isinstance(series[name].index, pd.DatetimeIndex):
            raise ValueError(f"driver {name!r} must have a DatetimeIndex")

    conditioning = list(condition_on) if condition_on is not None else list(names)

    def quantile_for(name):
        if isinstance(threshold_quantiles, dict):
            return float(threshold_quantiles.get(name, 0.98))
        return float(threshold_quantiles)

    pairing = pd.Timedelta(hours=float(pairing_window_hours))
    rows = []
    peak_times_by_driver, record_years = {}, float("nan")
    for cond in conditioning:
        if target_rate_per_year is not None:
            _thr, peaks, record_years = calibrate_threshold_for_rate(
                series[cond], target_rate_per_year, min_separation_hours=decluster_window_hours)
        else:
            peaks = declustered_pot_peaks(
                series[cond], threshold_quantile=quantile_for(cond), min_separation_hours=decluster_window_hours)
        peak_times_by_driver[cond] = [pd.Timestamp(t) for t in peaks["time"]]
        for _, peak in peaks.iterrows():
            event_time = pd.Timestamp(peak["time"])
            row = {"event_time": event_time, "conditioned_on": cond,
                   cond: float(peak["value"]), f"{cond}_time": event_time}
            for other in names:
                if other == cond:
                    continue
                window = series[other].loc[event_time - pairing : event_time + pairing]
                if len(window):
                    other_time = pd.Timestamp(window.idxmax())
                    row[other] = _scalar_at_time(window, other_time)
                    row[f"{other}_time"] = other_time
                else:
                    row[other] = np.nan
                    row[f"{other}_time"] = pd.NaT
            rows.append(row)

    out = pd.DataFrame(rows, columns=["event_time", "conditioned_on", *names, *[f"{n}_time" for n in names]])
    if dropna and not out.empty:
        out = out.dropna(subset=names).reset_index(drop=True)
    out.attrs["base_event_rate_per_year"] = distinct_event_rate(
        peak_times_by_driver, min_separation_hours=decluster_window_hours, record_years=record_years)
    out.attrs["record_years"] = float(record_years)
    out.attrs["target_rate_per_year"] = float(target_rate_per_year) if target_rate_per_year is not None else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Driver series + member libraries (IO)                                      #
# --------------------------------------------------------------------------- #
def _dependence(config):
    if "dependence" in config:
        return dict(config.get("dependence") or {})
    return dict(((config.get("event_catalog") or {}).get("dependence") or {}))


def _is_config(obj):
    return isinstance(obj, dict) and bool({"records", "dependence", "event_catalog"} & set(obj))


def _record_specs(config):
    specs = dict(config.get("records") or {})
    if specs:
        return specs
    specs = dict(_dependence(config).get("driver_records") or {})
    if not specs:
        raise ValueError("driver record specs are required in runtime_config.records or event_catalog.dependence")
    return specs


def _pairing_kwargs(config):
    dependence = _dependence(config)
    cooc = dict(dependence.get("cooccurrence") or {})
    out = {
        "threshold_quantiles": cooc.get("threshold_quantiles", cooc.get("threshold_quantile", 0.98)),
        "decluster_window_hours": float(cooc.get("decluster_window_hours", 120.0)),
        "pairing_window_hours": float(cooc.get("pairing_window_hours", 72.0)),
        "dropna": bool(cooc.get("dropna", True)),
    }
    if cooc.get("target_rate_per_year") is not None:
        out["target_rate_per_year"] = float(cooc["target_rate_per_year"])
    condition_on = dependence.get("condition_on") or cooc.get("condition_on")
    if condition_on:
        out["condition_on"] = list(condition_on)
    return out


def _driver_vector(config, records=None):
    drivers = list(_dependence(config).get("driver_vector") or [])
    if drivers:
        return drivers
    if records is not None:
        return list(records)
    raise ValueError("dependence.driver_vector is required")


def load_records(config_or_specs, *, location_root=None, sites=None):
    """Load configured driver records into source-agnostic time series."""
    if _is_config(config_or_specs):
        root = config_or_specs.get("location_root", location_root) if location_root is None else location_root
        return load_driver_series(_record_specs(config_or_specs), location_root=root, sites=sites)
    return load_driver_series(dict(config_or_specs or {}), location_root=location_root, sites=sites)


def paired_pot(records_or_config, config=None, *, location_root=None, sites=None, **overrides):
    """Build the paired POT co-occurrence sample from records or runtime config.

    Carries ``base_event_rate_per_year`` for ``T = 1/(lambda * S_and(F(x)))``.
    """
    if config is None and _is_config(records_or_config):
        config = records_or_config
        records = load_records(config, location_root=location_root, sites=sites)
    else:
        records = records_or_config
    params = {**(_pairing_kwargs(config) if config is not None else {}), **overrides}
    driver_names = list(params.pop("driver_names", _driver_vector(config, records) if config is not None else records))
    missing = [driver for driver in driver_names if driver not in records]
    if missing:
        raise ValueError(f"driver records missing: {missing}")
    return build_paired_observations(
        {driver: records[driver] for driver in driver_names},
        driver_names=driver_names,
        **params,
    )


def load_driver_series(record_specs, *, location_root=None, sites=None):
    """Read each ``{path, time_column, value_column, aggregate?, group_column?, transform?}``
    spec into a clean ``pd.Series`` on a DatetimeIndex. ``transform="ntr"`` replaces a total
    water-level record with its non-tidal residual (surge) via utide."""
    series = {}
    for driver, spec in record_specs.items():
        path = Path(spec["path"])
        if not path.is_absolute() and location_root is not None:
            path = Path(location_root) / path
        if not path.exists():
            raise FileNotFoundError(f"{driver} record not found: {path}")
        frame = pd.read_csv(path)
        group_column = spec.get("group_column")
        if group_column and sites and group_column in frame:
            frame = frame[frame[group_column].astype(str).isin({str(s) for s in sites})]
        missing = [c for c in (spec["time_column"], spec["value_column"]) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{driver} record is missing configured column(s) {missing}: {path}. "
                "Refresh the source artifact or update the driver_records spec so the "
                "dependence sample uses the intended driver timestamp and value."
            )
        time = pd.to_datetime(frame[spec["time_column"]], errors="coerce")
        value = pd.to_numeric(frame[spec["value_column"]], errors="coerce")
        clean = pd.Series(value.to_numpy(dtype=float), index=pd.DatetimeIndex(time)).dropna().sort_index()
        if spec.get("aggregate"):
            clean = getattr(clean.groupby(level=0), spec["aggregate"])()
        if spec.get("transform") == "ntr":
            clean = non_tidal_residual(clean, latitude=float(spec["latitude"])).dropna().sort_index()
        if clean.empty:
            raise ValueError(f"{driver} record produced no usable values: {path}")
        series[driver] = clean
    return series


def member_library_from_records(records, *, value_column, time_column, index_column,
                                aggregate="mean", id_prefix, member_file):
    """Per-timestamp member library (field pointers): one member per timestamp, each row
    carrying ``member_id``/``member_file``/``time`` plus the realization index column."""
    frame = records.copy()
    frame["_time"] = pd.to_datetime(frame[time_column], errors="coerce")
    grouped = frame.dropna(subset=["_time"]).groupby("_time", as_index=False)[value_column].agg(aggregate)
    return pd.DataFrame({
        "member_id": id_prefix + "_" + grouped["_time"].dt.strftime("%Y%m%dT%H%M%S"),
        "member_file": str(member_file),
        "time": grouped["_time"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        index_column: grouped[value_column].to_numpy(dtype=float),
    })

# --------------------------------------------------------------------------------------
# EVA-dataset adapter + marginal params/RP CSV schema (moved from the legacy
# return-curve helpers). Consumed by peak-fitting and coastal builders.
# --------------------------------------------------------------------------------------


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
    "fit_best_distribution", "HistoricalPeakMarginal", "EmpiricalMarginal", "fit_marginal",
    "coastal_components", "non_tidal_residual",
    "declustered_pot_peaks", "calibrate_threshold_for_rate", "distinct_event_rate",
    "build_paired_observations", "load_records", "paired_pot", "load_driver_series", "member_library_from_records",
    "from_eva_dataset", "marginal_params_frame", "marginal_rps_frame",
    "write_historical_peak_marginal", "load_historical_peak_marginal",
]
