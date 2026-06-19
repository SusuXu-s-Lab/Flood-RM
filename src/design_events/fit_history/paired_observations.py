"""Two-sided conditional POT co-occurrence sampling of compound flood drivers.

Assembles the paired-observation sample that the vine fit (`build_events.probability.dependence`)
consumes: a set of historical events where the Driver Probability Indices are observed
*together*. The method uses two-sided conditional sampling:

- condition on each driver in turn; take its declustered peaks-over-threshold,
- for every conditioning peak, record the concurrent maximum of the other drivers
  within a pairing window,
- concatenate the conditional samples into one paired-observation table.

This is the data step (and the main publication risk): it must run on aligned
historical driver time series (e.g. detrended water level / NTR, basin-average AORC
rainfall, USGS discharge, NWM antecedent soil moisture). The engine here is source-
agnostic — it takes driver Series on a shared DatetimeIndex — so the location-specific
plumbing is just reading those records into Series before calling it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def declustered_pot_peaks(series, *, threshold=None, threshold_quantile=0.98, min_separation_hours=120.0):
    """Declustered peaks-over-threshold of one driver series.

    Greedily selects the largest exceedances such that no two peaks fall within
    ``min_separation_hours``. Returns a frame with ``time`` and ``value``.
    """
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
    for time, value in exceedances.sort_values(ascending=False, kind="stable").items():
        if all(abs(time - chosen) >= separation for chosen in selected):
            selected.append(time)
    selected = sorted(selected)
    return pd.DataFrame({"time": selected, "value": [float(exceedances.loc[t]) for t in selected]})


def calibrate_threshold_for_rate(series, target_rate_per_year, *, min_separation_hours=120.0, search_quantile=0.90):
    """Pick the POT threshold whose declustered peaks occur at ~``target_rate_per_year``.

    This selects a threshold to obtain a target exceedance rate, rather than fixing a
    quantile and inheriting whatever rate falls out.
    Because greedy declustering visits exceedances in descending order, the first ``N``
    selected peaks are exactly the ``N`` largest declustered peaks, so we stop once
    ``N = round(target_rate * record_years)`` are found and read the threshold off the
    smallest of them. Returns ``(threshold, peaks_frame, record_years)``.
    """
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


def _distinct_event_rate(peak_times_by_driver, *, min_separation_hours, record_years):
    """Rate of distinct storms across the conditioning drivers (events/yr).

    Two-sided sampling lists the same storm once per conditioning driver, so the base
    point process behind the copula is the *union* of conditioning peaks with peaks
    within ``min_separation_hours`` merged into one event. ``T = 1/(rate * S)`` then uses
    this distinct-storm rate, not the inflated row count.
    """
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


def build_paired_observations(
    drivers,
    *,
    driver_names=None,
    condition_on=None,
    target_rate_per_year=None,
    threshold_quantiles=0.98,
    decluster_window_hours=120.0,
    pairing_window_hours=72.0,
    dropna=True,
):
    """Two-sided conditional POT co-occurrence sample over multiple drivers.

    ``drivers`` is a DataFrame (DatetimeIndex, one column per driver) or a mapping of
    name -> Series. Conditioning on each driver in turn, every declustered peak is paired
    with the concurrent maximum of the other drivers within ``pairing_window_hours``.
    Returns a frame with ``event_time``, ``conditioned_on``, and one column per driver —
    the Driver Probability Index matrix for ``fit_driver_dependence``.

    ``condition_on`` restricts which drivers seed conditioning peaks (the extreme forcing
    drivers; a bounded antecedent state is paired but never conditioned). When
    ``target_rate_per_year`` is set, each conditioning threshold is calibrated to that rate
    instead of a fixed quantile, and the realized distinct-storm rate is
    recorded on ``out.attrs['base_event_rate_per_year']`` for the ``T = 1/(rate*S)`` step.
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
                series[cond], target_rate_per_year, min_separation_hours=decluster_window_hours
            )
        else:
            peaks = declustered_pot_peaks(
                series[cond], threshold_quantile=quantile_for(cond), min_separation_hours=decluster_window_hours
            )
        peak_times_by_driver[cond] = [pd.Timestamp(t) for t in peaks["time"]]
        for _, peak in peaks.iterrows():
            event_time = pd.Timestamp(peak["time"])
            row = {
                "event_time": event_time,
                "conditioned_on": cond,
                cond: float(peak["value"]),
                f"{cond}_time": event_time,
            }
            for other in names:
                if other == cond:
                    continue
                window = series[other].loc[event_time - pairing : event_time + pairing]
                if len(window):
                    other_time = pd.Timestamp(window.idxmax())
                    row[other] = float(window.loc[other_time])
                    row[f"{other}_time"] = other_time
                else:
                    row[other] = np.nan
                    row[f"{other}_time"] = pd.NaT
            rows.append(row)

    out = pd.DataFrame(rows, columns=["event_time", "conditioned_on", *names, *[f"{name}_time" for name in names]])
    if dropna and not out.empty:
        out = out.dropna(subset=names).reset_index(drop=True)
    out.attrs["base_event_rate_per_year"] = _distinct_event_rate(
        peak_times_by_driver, min_separation_hours=decluster_window_hours, record_years=record_years
    )
    out.attrs["record_years"] = float(record_years)
    out.attrs["target_rate_per_year"] = float(target_rate_per_year) if target_rate_per_year is not None else float("nan")
    return out
