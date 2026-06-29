"""Coastal-only helpers for the ADR-0020 reference bundle.

The v2 coastal contract is narrow: use NTR/surge as the stochastic Driver
Probability Index and preserve mean sea level + astronomical tide unscaled in metadata
and audit checks. This module does not write SFINCS boundary forcing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, wasserstein_distance

from design_events.records import (
    coastal_components,
    load_historical_peak_marginal,
    non_tidal_residual,
)
from design_events.peaks import load_hourly_waterlevel
from design_events.probability import assign_severity_bands


def tide_preserving_total_water_level(components, *, ntr_scale_factor=1.0, msl_shift=0.0):
    """Rebuild total water level as ``MSL + tide + K * NTR + msl_shift``.

    Only the NTR/surge component is scaled. Tide remains unchanged by construction.
    """

    frame = pd.DataFrame(components)
    required = {"msl", "tide", "ntr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("coastal components missing columns: " + ", ".join(sorted(missing)))
    return (
        pd.to_numeric(frame["msl"], errors="coerce")
        + float(msl_shift)
        + pd.to_numeric(frame["tide"], errors="coerce")
        + float(ntr_scale_factor) * pd.to_numeric(frame["ntr"], errors="coerce")
    )


def coastal_realization_metadata(events, drivers, components=None, config=None):
    """Return coastal realization audit metadata from v2 bundle tables."""

    driver_frame = pd.DataFrame(drivers)
    coastal = driver_frame[driver_frame.get("driver", pd.Series(dtype=str)).astype(str).isin(["coastal_ntr", "coastal_water_level"])]
    scale = pd.to_numeric(coastal.get("scale_factor", pd.Series(dtype=float)), errors="coerce")
    metadata = {
        "coastal_driver": "coastal_ntr" if "coastal_ntr" in set(coastal.get("driver", [])) else "coastal_water_level",
        "tide_preserved_unscaled": True,
        "scaled_component": "non_tidal_residual",
        "ntr_scale_factor_min": float(scale.min()) if scale.notna().any() else None,
        "ntr_scale_factor_max": float(scale.max()) if scale.notna().any() else None,
        "realized_member_count": int(coastal.get("member_id", pd.Series(dtype=object)).nunique()),
    }

    if components is not None:
        frame = pd.DataFrame(components).copy()
        if {"wl", "msl", "tide", "ntr"}.issubset(frame.columns):
            recon = frame["msl"] + frame["tide"] + frame["ntr"]
            metadata["component_reconstruction_max_abs_error"] = float((frame["wl"] - recon).abs().max())
        elif {"msl", "tide", "ntr"}.issubset(frame.columns):
            metadata["component_reconstruction_max_abs_error"] = 0.0
        metadata["component_rows"] = int(len(frame))
    else:
        metadata["component_reconstruction_max_abs_error"] = None
        metadata["component_rows"] = 0

    return metadata


# --------------------------------------------------------------------------------------
# Production coastal hybrid sampler + surge hydrograph templates/members (moved from the
# legacy nested coastal builders; ADR-0021). NTR/tide contract preserved: the copula
# and sampler use NTR; tide rides back unscaled in the realized total water level.
# --------------------------------------------------------------------------------------


def sample_return_periods(n, settings, seed):
    # Spread rare events across return-period space.
    rng = np.random.default_rng(seed)
    rp_min = float(settings.get("return_period_min_years", 1.5))
    rp_max = float(settings.get("return_period_max_years", 250.0))
    p = (np.arange(n) + rng.random(n)) / max(1, n)
    if settings.get("spacing", "log") == "linear":
        samples = rp_min + (rp_max - rp_min) * p
    else:
        samples = np.exp(np.log(rp_min) + (np.log(rp_max) - np.log(rp_min)) * p)
    return rng.permutation(samples)

def bootstrap_body_sample(data, n_samples, seed):
    # Body sampler: bootstrap with replacement from historical peaks. Each sample is an exact
    # observed peak (no interpolation/smoothing/blending), so the synthetic body's empirical
    # marginal and variance converge to the input record's and every sample is traceable.
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0 or n_samples <= 0:
        return np.empty(0, dtype=float)
    return rng.choice(data, size=int(n_samples), replace=True)


def _round_to_total(fractions, total):
    raw = {name: float(frac) * total for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = int(total) - sum(counts.values())
    for name in sorted(raw, key=lambda value: raw[value] - counts[value], reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def _select_candidate_pool_by_band(pool, n_samples, settings, seed):
    fractions = dict(settings.get("catalog_band_fractions") or {})
    if not fractions:
        return pool.sample(n=int(n_samples), replace=len(pool) < int(n_samples), random_state=int(seed)).reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    target_counts = _round_to_total(fractions, int(n_samples))
    selected = []
    support = pool["severity_band"].astype(str).value_counts().to_dict()
    mass = pool.groupby(pool["severity_band"].astype(str))["probability_weight"].sum().to_dict()
    pool_total = max(len(pool), 1)
    for band, n_band in target_counts.items():
        n_band = int(n_band)
        if n_band <= 0:
            continue
        members = pool[pool["severity_band"].astype(str).eq(str(band))]
        if members.empty:
            raise ValueError(f"candidate pool has no events in required severity band {band!r}")
        picks = rng.choice(members.index.to_numpy(), size=n_band, replace=len(members) < n_band)
        rows = pool.loc[picks].copy()
        band_mass = float(mass.get(str(band), 0.0))
        target_fraction = n_band / float(n_samples)
        rows["sampling_weight"] = band_mass / target_fraction if target_fraction > 0 else np.nan
        rows["probability_weight"] = band_mass / n_band if n_band > 0 else np.nan
        rows["pool_band_support"] = int(support.get(str(band), 0))
        rows["pool_band_probability"] = rows["pool_band_support"] / float(pool_total)
        selected.append(rows)
    if not selected:
        raise ValueError("no catalog events selected from candidate pool")
    out = pd.concat(selected, ignore_index=True)
    out["probability_weight"] = out["probability_weight"] / float(out["probability_weight"].sum())
    out["catalog_role"] = "design"
    out["sampling_scheme"] = "band_stratified_importance_from_candidate_pool"
    return out.sample(frac=1.0, random_state=int(seed) + 1).reset_index(drop=True)


def _probability_weights(rows, marginal, *, splice_q, body_count, tail_count, settings):
    weights = np.zeros(len(rows), dtype=float)
    if len(rows) == 0:
        return weights
    if body_count and tail_count:
        target_body_fraction = float(splice_q)
        target_tail_fraction = float(1.0 - splice_q)
    else:
        target_body_fraction = 1.0 if body_count else 0.0
        target_tail_fraction = 1.0 if tail_count else 0.0

    region = rows["sampling_region"].astype(str).to_numpy()
    body_mask = region == "body"
    tail_mask = region == "tail"
    if body_mask.any():
        weights[body_mask] = target_body_fraction / float(body_mask.sum())
    if tail_mask.any():
        tail_rp = np.asarray(marginal.return_period(rows.loc[tail_mask, "peak_m"].to_numpy(dtype=float)), dtype=float)
        tail_rp = np.where(np.isfinite(tail_rp) & (tail_rp > 0), tail_rp, np.nan)
        if str(settings.get("spacing", "log")) == "linear":
            tail_scores = 1.0 / np.square(tail_rp)
        else:
            tail_scores = 1.0 / tail_rp
        if not np.isfinite(tail_scores).any() or float(np.nansum(tail_scores)) <= 0.0:
            tail_scores = np.ones(tail_mask.sum(), dtype=float)
        tail_scores = np.nan_to_num(tail_scores, nan=0.0, posinf=0.0, neginf=0.0)
        weights[tail_mask] = target_tail_fraction * tail_scores / float(tail_scores.sum())
    return weights


def hybrid_peak_sample(peaks, n_samples, settings, marginal, seed):
    return hybrid_peak_sample_frame(peaks, n_samples, settings, marginal, seed)["peak_m"].to_numpy(dtype=float)


def hybrid_peak_sample_frame(peaks, n_samples, settings, marginal, seed):
    # Common peaks are bootstrap resamples below the splice threshold; rare peaks come from the
    # fitted return-period curve, drawn uniformly in log-RP space above it. Sampling weights
    # preserve the body/tail probability mass implied by the splice threshold.
    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if len(peaks) == 0:
        raise RuntimeError("no historical peaks available for hybrid sampling")
    candidate_pool_count = int(settings.get("candidate_pool_count", 0) or 0)
    if candidate_pool_count > int(n_samples) and settings.get("catalog_band_fractions"):
        pool_settings = dict(settings)
        pool_settings.pop("candidate_pool_count", None)
        pool_settings.pop("catalog_band_fractions", None)
        pool_settings.pop("tail_sample_fraction", None)
        pool = hybrid_peak_sample_frame(peaks, candidate_pool_count, pool_settings, marginal, seed)
        pool["sample_rp_years"] = marginal.return_period(pool["peak_m"].to_numpy(dtype=float))
        pool["severity_band"] = assign_severity_bands(pool["sample_rp_years"], settings.get("severity_bands"))
        selected = _select_candidate_pool_by_band(pool, n_samples, settings, seed + 17)
        selected["candidate_pool_count"] = candidate_pool_count
        return selected
    splice_q = float(np.clip(settings.get("hybrid_splice_quantile", 0.95), 0.5, 0.999))
    threshold = np.quantile(peaks, splice_q)
    rp_min = float(settings.get("return_period_min_years", 1.5))
    rp_max = float(settings.get("return_period_max_years", 250.0))
    splice_rp = float(marginal.return_period(threshold))
    body_candidates = peaks[peaks <= threshold]
    body_feasible = len(body_candidates) > 0
    tail_feasible = rp_max >= max(rp_min, splice_rp)
    if body_feasible and tail_feasible:
        tail_fraction = settings.get("tail_sample_fraction", 1 - splice_q)
        tail_count = int(round(n_samples * float(tail_fraction)))
        tail_count = min(max(tail_count, 1), n_samples - 1) if n_samples > 1 else 0
    elif tail_feasible:
        tail_count = n_samples
    else:
        tail_count = 0
    body_count = n_samples - tail_count
    # Synthetic body marginal == empirical marginal of the truncated record (in expectation):
    # the only body-region assumption is "future common storms resemble past common storms".
    body = bootstrap_body_sample(body_candidates, body_count, seed)
    rows = pd.DataFrame(
        {
            "peak_m": body,
            "sampling_region": "body",
            "event_origin": "historical_bootstrap_body",
        }
    )
    if tail_count:
        # Tail: start no smaller than the return period at the splice threshold.
        tail_settings = dict(settings)
        tail_settings["return_period_min_years"] = max(rp_min, splice_rp)
        tail_settings["return_period_max_years"] = rp_max
        tail = marginal.magnitude(sample_return_periods(tail_count, tail_settings, seed + 1))
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    {
                        "peak_m": tail,
                        "sampling_region": "tail",
                        "event_origin": "synthetic_tail",
                    }
                ),
            ],
            ignore_index=True,
        )
    sample_body_fraction = body_count / n_samples if n_samples else 0.0
    sample_tail_fraction = tail_count / n_samples if n_samples else 0.0
    if body_count and tail_count:
        target_body_fraction = splice_q
        target_tail_fraction = 1.0 - splice_q
    else:
        target_body_fraction = 1.0 if body_count else 0.0
        target_tail_fraction = 1.0 if tail_count else 0.0
    rows["sampling_weight"] = np.where(
        rows["sampling_region"] == "tail",
        target_tail_fraction / sample_tail_fraction if sample_tail_fraction else np.nan,
        target_body_fraction / sample_body_fraction if sample_body_fraction else np.nan,
    )
    rows["probability_weight"] = _probability_weights(
        rows,
        marginal,
        splice_q=splice_q,
        body_count=body_count,
        tail_count=tail_count,
        settings=settings,
    )
    configured_tail_fraction = settings.get("tail_sample_fraction")
    natural_tail_fraction = 1.0 - splice_q
    rows["catalog_role"] = "probability"
    rows["sampling_scheme"] = (
        "probability_proportional_hybrid"
        if configured_tail_fraction is None or np.isclose(float(configured_tail_fraction), natural_tail_fraction)
        else "tail_enriched_hybrid"
    )
    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(len(rows))
    return rows.iloc[order].reset_index(drop=True)

def build_sampled_peaks(config, paths):
    # Write target peak heights for the event-member builder.
    event_count = int(config.get("events", {}).get("target_event_count", 2500))
    settings = config.get("sampling", {})
    seed = int(config.get("template_assignment", {}).get("random_seed", 42))
    peaks = pd.read_csv(paths["historical_peaks_csv"], parse_dates=["time"], index_col="time")
    marginal = load_historical_peak_marginal(paths["marginal_params_csv"])
    sample = hybrid_peak_sample_frame(peaks["h"].dropna(), event_count, settings, marginal, seed)
    actual_count = len(sample)
    df = pd.DataFrame({
        "event_id": [f"evt_{i:04d}" for i in range(1, actual_count + 1)],
        "peak_m": sample["peak_m"].to_numpy(dtype=float),
        "sample_rp_years": marginal.return_period(sample["peak_m"].to_numpy(dtype=float)),
        "sampling_region": sample["sampling_region"].to_numpy(),
        "sampling_weight": sample["sampling_weight"].to_numpy(dtype=float),
        "probability_weight": sample["probability_weight"].to_numpy(dtype=float),
        "event_origin": sample["event_origin"].to_numpy(),
        "catalog_role": sample["catalog_role"].to_numpy(),
        "sampling_scheme": sample["sampling_scheme"].to_numpy(),
    })
    for optional in ["candidate_pool_count", "pool_band_support", "pool_band_probability"]:
        if optional in sample:
            df[optional] = sample[optional].to_numpy()
    df["severity_band"] = assign_severity_bands(df["sample_rp_years"], settings.get("severity_bands"))
    paths["sampled_peaks_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["sampled_peaks_csv"], index=False, float_format="%.10g")
    return df


# Surge hydrograph templates and event-member artifacts


template_columns = [
    "peak_time",
    "baseline_m",
    "threshold_m",
    "absolute_peak_m",
    "peak_m",
    "volume",
    "duration_above_50pct_peak",
    "rise_time_to_peak",
    "fall_time_from_peak",
    "asymmetry_ratio",
    "n_secondary_peaks",
    "valid_start_hour",
    "valid_end_hour",
]

member_columns = [
    "sample_rp_years",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "template_id",
    "template_peak_m",
    "template_peak_time",
    "tail_morph_factor",
    "peak",
    "volume",
    "duration_above_50pct_peak",
    "rise_time_to_peak",
    "fall_time_from_peak",
    "asymmetry_ratio",
    "n_secondary_peaks",
    "valid_start_hour",
    "valid_end_hour",
]

def _ensure_event_ids(df_events):
    df = df_events.copy()
    if "event_id" not in df.columns:
        df.insert(0, "event_id", [f"evt_{i:04d}" for i in range(1, len(df) + 1)])
    df["event_id"] = df["event_id"].astype(str)
    return df

def _find_contiguous_event_bounds(values, peak_idx, threshold):
    # Walk left and right from the peak until the surge drops below threshold.
    start = peak_idx
    end = peak_idx
    while start > 0 and np.isfinite(values[start - 1]) and values[start - 1] >= threshold:
        start -= 1
    while end < values.size - 1 and np.isfinite(values[end + 1]) and values[end + 1] >= threshold:
        end += 1
    return start, end

def _local_peaks(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, -np.inf)
    peaks, _ = find_peaks(values)
    return peaks[np.isfinite(values[peaks])]

def _enforce_duration_bounds(start, end, *, peak_idx, size, min_event_hours, max_event_hours):
    # Keep events long enough to include storm shape, but not so long they include calm water.
    min_samples = max(2, int(min_event_hours) + 1)
    max_samples = max(min_samples, int(max_event_hours) + 1)
    while (end - start + 1) < min_samples:
        grow_left = start > 0 and (peak_idx - start) <= (end - peak_idx)
        grow_right = end < size - 1
        if grow_left:
            start -= 1
        elif grow_right:
            end += 1
        elif start > 0:
            start -= 1
        else:
            break
    if (end - start + 1) > max_samples:
        half = max_samples // 2
        start = max(0, peak_idx - half)
        end = min(size - 1, start + max_samples - 1)
        start = max(0, end - max_samples + 1)
    return start, end

def _count_secondary_peaks(values, peak_idx):
    # Count meaningful side peaks. Multi-peak storms are less useful as templates.
    if np.isfinite(values).sum() < 3:
        return 0
    arr = np.where(np.isfinite(values), values, -np.inf)
    peaks = _local_peaks(arr)
    peaks = peaks[peaks != peak_idx]
    cutoff = 0.25 * float(arr[peak_idx])
    return int(np.sum(arr[peaks] >= cutoff)) if peaks.size else 0

def _dominant_peak_ok(values, peak_idx, ratio_max):
    # Keep templates where the central peak clearly dominates nearby bumps.
    if np.isfinite(values).sum() < 3:
        return False
    arr = np.where(np.isfinite(values), values, -np.inf)
    main_value = float(arr[peak_idx])
    if not np.isfinite(main_value) or main_value <= 0.0:
        return False
    peaks = _local_peaks(arr)
    secondary = arr[peaks[peaks != peak_idx]]
    return not len(secondary) or float(np.nanmax(secondary)) / main_value <= float(ratio_max)

def _compute_descriptors(rel_hours, values, peak_idx):
    # These simple descriptors let us compare real and synthetic event shapes.
    peak = float(values[peak_idx])
    positive = np.clip(values, 0.0, None)
    start_hour = float(rel_hours[np.where(np.isfinite(values))[0][0]])
    end_hour = float(rel_hours[np.where(np.isfinite(values))[0][-1]])
    duration_above_half = int(np.sum(values >= 0.5 * peak))
    rise_time = float(abs(start_hour))
    fall_time = float(end_hour)
    asymmetry = float(fall_time / rise_time) if rise_time > 0 else np.nan
    return {
        "peak": peak,
        "volume": float(np.nansum(positive)),
        "duration_above_50pct_peak": duration_above_half,
        "rise_time_to_peak": rise_time,
        "fall_time_from_peak": fall_time,
        "asymmetry_ratio": asymmetry,
        "n_secondary_peaks": _count_secondary_peaks(values, peak_idx),
        "valid_start_hour": start_hour,
        "valid_end_hour": end_hour,
    }

def _series_to_axis(rel_hours, values, axis):
    # Put each event on the same relative-hour axis centered at peak time.
    out = np.full(axis.shape, np.nan, dtype=float)
    if rel_hours.size == 0:
        return out
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return out
    x = rel_hours[finite].astype(float)
    y = values[finite].astype(float)
    in_range = (axis >= x.min()) & (axis <= x.max())
    if np.unique(x).size == 1:
        out[axis == x[0]] = y[0]
        return out
    out[in_range] = np.interp(axis[in_range].astype(float), x, y)
    return out

def _stretch_on_axis(values, axis, factor):
    # Rare out-of-sample peaks can last slightly longer, but keep the same shape family.
    if factor <= 1.000001:
        return values.copy()
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if finite.sum() < 2:
        return values.copy()
    x = axis[finite].astype(float)
    y = values[finite].astype(float)
    query = axis.astype(float) / factor
    in_range = (query >= x.min()) & (query <= x.max())
    out[in_range] = np.interp(query[in_range], x, y)
    return out

def extract_historical_templates(h_series, peak_times, settings):
    # Cut real Cora storms around historical peaks and normalize their shapes.
    baseline_hours = int(settings.get("pre_event_baseline_hours", 24))
    threshold_fraction = float(settings.get("event_threshold_fraction", 0.10))
    threshold_min_m = float(settings.get("event_threshold_min_m", 0.05))
    min_event_hours = int(settings.get("min_event_hours", 12))
    max_event_hours = int(settings.get("max_event_hours", 168))
    tide_half_window_hours = int(settings.get("tide_resolving_half_window_hours", min(72, max_event_hours)))
    tide_half_window_hours = max(1, min(tide_half_window_hours, max_event_hours))
    dominant_peak_ratio_max = float(settings.get("dominant_peak_ratio_max", 0.90))
    rows = []
    search_hours = int(max_event_hours + baseline_hours + 24)
    full_axis = np.arange(-int(max_event_hours), int(max_event_hours) + 1, dtype=int)
    for template_id, peak_time in enumerate(pd.DatetimeIndex(peak_times), start=1):
        # Start with a wide window so the full storm is available.
        if peak_time not in h_series.index:
            continue
        window = h_series.loc[
            peak_time - pd.Timedelta(hours=search_hours): peak_time + pd.Timedelta(hours=search_hours)
        ].copy()
        if window.empty or peak_time not in window.index:
            continue
        # First baseline: water level before the peak.
        peak_idx = int(window.index.get_loc(peak_time))
        baseline0_lo = max(0, peak_idx - int(baseline_hours))
        if peak_idx > baseline0_lo:
            baseline0 = float(window.iloc[baseline0_lo:peak_idx].median())
        else:
            baseline0 = float(window.iloc[max(0, peak_idx - 1): peak_idx + 1].median())
        anomaly0 = window.astype(float) - baseline0
        peak0 = float(anomaly0.iloc[peak_idx])
        if not np.isfinite(peak0) or peak0 <= 0.0:
            continue
        threshold0 = max(float(threshold_fraction) * peak0, float(threshold_min_m))
        start0, end0 = _find_contiguous_event_bounds(anomaly0.to_numpy(dtype=float), peak_idx, threshold0)
        # Second baseline: water level before the event starts.
        baseline1_lo = max(0, start0 - int(baseline_hours))
        baseline1 = float(window.iloc[baseline1_lo:start0].median()) if start0 > baseline1_lo else baseline0
        # Anomaly hydrograph = absolute water level minus pre-event baseline.
        anomaly = window.astype(float) - baseline1
        peak = float(anomaly.iloc[peak_idx])
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        threshold = max(float(threshold_fraction) * peak, float(threshold_min_m))
        start, end = _find_contiguous_event_bounds(anomaly.to_numpy(dtype=float), peak_idx, threshold)
        start, end = _enforce_duration_bounds(
            start,
            end,
            peak_idx=peak_idx,
            size=anomaly.size,
            min_event_hours=min_event_hours,
            max_event_hours=max_event_hours,
        )
        segment = anomaly.iloc[start:end + 1]
        rel_hours = ((segment.index - peak_time) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        values = segment.to_numpy(dtype=float)
        local_peak_idx = int(np.argmax(values))
        if rel_hours[local_peak_idx] != 0:
            zero_idx = np.where(rel_hours == 0)[0]
            if zero_idx.size:
                local_peak_idx = int(zero_idx[0])
        if not _dominant_peak_ok(values, local_peak_idx, dominant_peak_ratio_max):
            continue
        # Normalize by peak so the template stores shape, not magnitude.
        normalized = values / max(float(values[local_peak_idx]), 1e-6)
        padded = _series_to_axis(rel_hours, normalized, full_axis)
        absolute = _series_to_axis(rel_hours, values, full_axis)
        tide_segment = h_series.loc[
            peak_time - pd.Timedelta(hours=tide_half_window_hours):
            peak_time + pd.Timedelta(hours=tide_half_window_hours)
        ]
        tide_rel_hours = ((tide_segment.index - peak_time) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        water_level_total_template = _series_to_axis(
            tide_rel_hours,
            tide_segment.to_numpy(dtype=float),
            full_axis,
        )
        descriptors = _compute_descriptors(rel_hours.astype(float), values, local_peak_idx)
        rows.append(
            {
                "template_id": f"tpl_{template_id:04d}",
                "peak_time": peak_time,
                "baseline_m": baseline1,
                "threshold_m": threshold,
                "absolute_peak_m": float(window.loc[peak_time]),
                "peak_m": descriptors["peak"],
                "surge_template": padded,
                "surge_absolute": absolute,
                "water_level_total_template": water_level_total_template,
                **descriptors,
            }
        )

    if not rows:
        raise RuntimeError("no dominant historical templates were extracted from CORA hourly water levels")
    return pd.DataFrame(rows).sort_values("peak_m").reset_index(drop=True)

def template_bank_to_dataset(template_frame, axis):
    # Store all historical templates in one file.
    template_ids = template_frame["template_id"].astype(str).to_numpy()
    ds = xr.Dataset(
        data_vars={
            "surge_template": (("template_id", "relative_hour"), np.stack(template_frame["surge_template"].to_numpy())),
            "surge_absolute": (("template_id", "relative_hour"), np.stack(template_frame["surge_absolute"].to_numpy())),
            "water_level_total_template": (
                ("template_id", "relative_hour"),
                np.stack(template_frame["water_level_total_template"].to_numpy()),
            ),
        },
        coords={
            "template_id": template_ids,
            "relative_hour": axis.astype(int),
        },
    )
    for column in template_columns:
        ds[column] = ("template_id", template_frame[column].to_numpy())
    ds.attrs.update(
        source="CORA single-node historical template bank",
        signal="event anomaly hydrograph",
        units="m",
    )
    return ds

def _tail_morph_factor(target_peak, historical_peaks, config):
    # If a sampled peak is beyond history, stretch time modestly instead of inventing a new shape.
    max_peak = float(np.nanmax(historical_peaks))
    if target_peak <= max_peak:
        return 1.0
    trigger_q = float(config.get("tail_morph_trigger_quantile", 0.95))
    max_factor = float(config.get("tail_morph_max_factor", 1.30))
    trigger_peak = float(np.nanquantile(historical_peaks, trigger_q))
    denom = max(0.05, max_peak - trigger_peak)
    excess = max(0.0, target_peak - max_peak)
    factor = 1.0 + min(max_factor - 1.0, (max_factor - 1.0) * excess / denom)
    return float(np.clip(factor, 1.0, max_factor))

def build_surge_event_members(df_events, template_frame, axis, settings):
    # Match each target peak to nearby real storm shapes.
    rng = np.random.default_rng(int(settings.get("random_seed", 42)))
    nearest_pool_size = int(settings.get("nearest_pool_size", 75))
    sigma_scale = float(settings.get("kernel_sigma_scale", 0.50))
    sigma_min = float(settings.get("kernel_sigma_min_m", 0.03))
    sigma_max = float(settings.get("kernel_sigma_max_m", 0.20))
    reuse_penalty_lambda = float(settings.get("reuse_penalty_lambda", 0.15))
    historical_peaks = template_frame["peak_m"].to_numpy(dtype=float)
    historical_templates = np.stack(template_frame["surge_template"].to_numpy())
    historical_total_templates = np.stack(template_frame["water_level_total_template"].to_numpy())
    template_ids = template_frame["template_id"].astype(str).to_numpy()
    usage_count = np.zeros(template_ids.size, dtype=int)
    pool_size = max(1, min(nearest_pool_size, historical_peaks.size))
    member_arrays = []
    total_member_arrays = []
    summary_rows = []

    for row in df_events.itertuples(index=False):
        event_id = str(getattr(row, "event_id"))
        target_peak = float(getattr(row, "peak_m"))

        # Start with templates closest in peak height.
        order = np.argsort(np.abs(historical_peaks - target_peak))
        pool_idx = order[:pool_size]
        if target_peak > float(np.nanmax(historical_peaks)):
            pool_idx = np.argsort(historical_peaks)[-pool_size:]
        pool_peaks = historical_peaks[pool_idx]
        sigma = float(np.clip(sigma_scale * np.nanstd(pool_peaks), sigma_min, sigma_max))
        sigma = sigma if np.isfinite(sigma) and sigma > 0 else sigma_min

        # Favor similar peaks, but allow reuse when needed.
        distance_weights = np.exp(-0.5 * ((pool_peaks - target_peak) / sigma) ** 2)
        reuse_weights = np.exp(-reuse_penalty_lambda * usage_count[pool_idx])
        weights = distance_weights * reuse_weights
        if not np.isfinite(weights).any() or float(weights.sum()) <= 0.0:
            weights = np.ones(pool_idx.size, dtype=float)
        weights = weights / weights.sum()

        # Draw one real template, then scale its normalized shape to the target peak.
        selected_local = int(rng.choice(np.arange(pool_idx.size), p=weights))
        selected_idx = int(pool_idx[selected_local])
        usage_count[selected_idx] += 1
        template_norm = historical_templates[selected_idx].astype(float)
        template_total = historical_total_templates[selected_idx].astype(float)
        template_baseline = float(template_frame.iloc[selected_idx]["baseline_m"])
        template_peak = max(float(historical_peaks[selected_idx]), 1e-6)
        morph_factor = _tail_morph_factor(target_peak, historical_peaks, settings)
        morphed = _stretch_on_axis(template_norm, axis.astype(float), morph_factor)
        scaled = morphed * target_peak
        total_scale = target_peak / template_peak
        water_level_total = template_baseline + (template_total - template_baseline) * total_scale
        finite = np.isfinite(scaled)
        if not finite.any():
            raise RuntimeError(f"{event_id} produced an empty synthetic hydrograph")
        peak_idx = int(np.nanargmax(scaled))
        descriptors = _compute_descriptors(axis.astype(float), scaled, peak_idx)
        total_finite = np.flatnonzero(np.isfinite(water_level_total))
        if total_finite.size:
            descriptors["valid_start_hour"] = float(axis[int(total_finite[0])])
            descriptors["valid_end_hour"] = float(axis[int(total_finite[-1])])
        summary_rows.append(
            {
                "event_id": event_id,
                "sample_rp_years": float(getattr(row, "sample_rp_years", np.nan)),
                "sampling_region": getattr(row, "sampling_region", pd.NA),
                "sampling_weight": float(getattr(row, "sampling_weight", np.nan)),
                "probability_weight": float(getattr(row, "probability_weight", np.nan)),
                "template_id": template_ids[selected_idx],
                "template_peak_m": float(historical_peaks[selected_idx]),
                "template_peak_time": template_frame.iloc[selected_idx]["peak_time"],
                "tail_morph_factor": morph_factor,
                **descriptors,
            }
        )
        member_arrays.append(scaled)
        total_member_arrays.append(water_level_total)
    summary = pd.DataFrame(summary_rows).sort_values("event_id").reset_index(drop=True)
    member_matrix = np.stack(member_arrays)
    total_member_matrix = np.stack(total_member_arrays)
    ds = xr.Dataset(
        data_vars={
            "surge": (("event_id", "relative_hour"), member_matrix),
            "water_level_total": (("event_id", "relative_hour"), total_member_matrix),
            "valid_mask": (("event_id", "relative_hour"), np.isfinite(total_member_matrix)),
            "surge_valid_mask": (("event_id", "relative_hour"), np.isfinite(member_matrix)),
        },
        coords={
            "event_id": summary["event_id"].astype(str).to_numpy(),
            "relative_hour": axis.astype(int),
        },
    )
    for column in member_columns:
        ds[column] = ("event_id", summary[column].to_numpy())
    ds.attrs.update(
        source="Synthetic CORA surge event members",
        signal="event anomaly hydrograph plus tide-resolving analog total-water-level member",
        units="m",
    )
    return ds, summary

def build_acceptance_report(summary, template_frame):
    # Small report for checking event quality before using outputs downstream.
    report = {
        "event_count": int(len(summary)),
        "valid_event_count": int(summary["peak"].notna().sum()),
        "all_qc_descriptors_populated": bool(summary.notna().all(axis=0).all()),
    }

    template_use = summary["template_id"].value_counts(normalize=True)
    report["max_template_reuse_fraction"] = float(template_use.max()) if not template_use.empty else np.nan
    report["tail_morph_fraction_gt_1p15"] = float((summary["tail_morph_factor"] > 1.15).mean()) if len(summary) else np.nan

    diagnostics = {}
    for column in ["peak", "volume", "duration_above_50pct_peak", "asymmetry_ratio"]:
        hist = pd.to_numeric(template_frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        syn = pd.to_numeric(summary[column], errors="coerce").dropna().to_numpy(dtype=float)
        diagnostics[column] = {
            "ks": float(ks_2samp(hist, syn).statistic) if ks_2samp and len(hist) and len(syn) else None,
            "wasserstein": float(wasserstein_distance(hist, syn)) if wasserstein_distance and len(hist) and len(syn) else None,
        }
    report["distribution_diagnostics"] = diagnostics

    report["checks"] = {
        "at_least_2500_valid_events": bool(report["valid_event_count"] >= 2500),
        "tail_morph_fraction_lt_0p05": bool(report["tail_morph_fraction_gt_1p15"] < 0.05),
        "max_template_reuse_fraction_lte_0p02": bool(report["max_template_reuse_fraction"] <= 0.02),
        "all_qc_descriptors_populated": bool(report["all_qc_descriptors_populated"]),
    }
    report["passed"] = bool(all(report["checks"].values()))
    return report

def write_overview_plot(output_path, template_frame, summary, ds_members):
    # One quick visual check: shape descriptors and a few hydrographs.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.ravel()
    sample_cols = ["peak", "duration_above_50pct_peak", "volume", "asymmetry_ratio"]
    for i, column in enumerate(sample_cols[:3]):
        ax[i].hist(pd.to_numeric(template_frame[column], errors="coerce"), bins=30, alpha=0.6, label="historical")
        ax[i].hist(pd.to_numeric(summary[column], errors="coerce"), bins=30, alpha=0.6, label="synthetic")
        ax[i].set_title(column.replace("_", " "))
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].legend()
    sample_event_ids = summary["event_id"].head(6).tolist()
    for event_id in sample_event_ids:
        values = ds_members["surge"].sel(event_id=event_id).to_numpy().astype(float)
        ax[3].plot(ds_members["relative_hour"].values, values, alpha=0.7)
    ax[3].set_title("sample synthetic hydrographs")
    ax[3].set_xlabel("relative hour")
    ax[3].set_ylabel("surge anomaly [m]")
    ax[3].grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path

def _apply_msl_shift(ds_members, summary, scenario):
    # Storm intensification under climate change is NOT modeled: shapes
    # and magnitudes are stationary; only the absolute reference level
    # moves.
    offset = float(scenario["slr_offset_m"])
    ds = ds_members
    if "water_level_total" in ds:
        ds["water_level_total"] = ds["water_level_total"] + offset
        ds["water_level_total"].attrs.update(
            long_name="tide-resolving coastal water level under scenario MSL",
            units="m",
            note="historical CORA analog total water level with scaled event anomaly and scenario MSL offset; SFINCS boundary forcing reads this by default",
        )
    ds = ds.assign(surge_absolute=ds["surge"] + offset)
    ds["surge_absolute"].attrs.update(
        long_name="legacy compact coastal water level under scenario MSL",
        units="m",
        note="surge_anomaly + slr_offset_m; retained for compatibility with compact-template workflows",
    )
    ds["surge"].attrs.update(
        long_name="event surge anomaly above pre-event baseline",
        units="m",
        note="scenario-invariant; identical across MSL-shift scenarios",
    )
    ds.attrs.update(
        scenario_name=scenario["name"],
        scenario_description=scenario["description"],
        slr_offset_m=offset,
    )
    for key in [
        "source",
        "source_url",
        "source_dataset",
        "source_location_basis",
        "source_baseline_year",
        "source_accessed",
        "scenario_family",
        "projection_year",
    ]:
        if key in scenario:
            ds.attrs[f"scenario_{key}"] = scenario[key]
    summary = summary.copy()
    summary["scenario_name"] = scenario["name"]
    summary["slr_offset_m"] = offset
    if "water_level_total" in ds:
        summary["absolute_peak_m"] = ds["water_level_total"].max("relative_hour", skipna=True).to_pandas().reindex(
            summary["event_id"].astype(str)
        ).to_numpy(dtype=float)
    else:
        summary["absolute_peak_m"] = summary["peak"].astype(float) + offset
    return ds, summary

def build_surge_event_artifacts(config, paths):
    # Step 4: turn target peaks into full surge hydrographs.
    design_cfg = config.get("design_events", {})
    template_cfg = config.get("template_assignment", {})
    settings = {**design_cfg, **template_cfg}
    h_series = load_hourly_waterlevel(paths["waterlevel_csv"]).dropna().sort_index()
    df_peaks = pd.read_csv(paths["historical_peaks_csv"], parse_dates=["time"], index_col="time")
    peak_times = pd.DatetimeIndex(df_peaks["h"].dropna().index)
    template_frame = extract_historical_templates(h_series, peak_times, settings)
    if config.get("coastal_waves", False):
        # SnapWave reuses the template's historical hour, so keep only analogs covered by ERA5 waves.
        collection_cfg = config.get("collection", {})
        wave_cfg = collection_cfg.get("era5_waves", {})
        wave_start_value = wave_cfg.get("start_date") or collection_cfg.get("start")
        wave_end_value = wave_cfg.get("end_date") or collection_cfg.get("end")
        if wave_start_value or wave_end_value:
            wave_start = pd.Timestamp(wave_start_value) if wave_start_value else None
            wave_end = pd.Timestamp(wave_end_value) if wave_end_value else None
            if isinstance(wave_end_value, str) and len(wave_end_value) == 10:
                wave_end = wave_end + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
            peak_time = pd.to_datetime(template_frame["peak_time"])
            template_start = peak_time + pd.to_timedelta(template_frame["valid_start_hour"], unit="h")
            template_end = peak_time + pd.to_timedelta(template_frame["valid_end_hour"], unit="h")
            keep = pd.Series(True, index=template_frame.index)
            if wave_start is not None:
                keep &= template_start >= wave_start
            if wave_end is not None:
                keep &= template_end <= wave_end
            template_frame = template_frame.loc[keep].reset_index(drop=True)
            if template_frame.empty:
                raise RuntimeError("no historical templates overlap the configured ERA5 wave collection window")
    axis = np.arange(
        -int(design_cfg.get("max_event_hours", 168)),
        int(design_cfg.get("max_event_hours", 168)) + 1,
        dtype=int,
    )
    ds_templates = template_bank_to_dataset(template_frame, axis)
    df_events = _ensure_event_ids(pd.read_csv(paths["sampled_peaks_csv"]))
    ds_members, summary = build_surge_event_members(df_events, template_frame, axis, settings)
    acceptance = build_acceptance_report(summary, template_frame)
    scenario = paths.get("scenario", {"name": "base", "slr_offset_m": 0.0, "description": ""})
    ds_members, summary = _apply_msl_shift(ds_members, summary, scenario)
    acceptance["scenario_name"] = scenario["name"]
    acceptance["slr_offset_m"] = float(scenario["slr_offset_m"])

    return {
        "template_frame": template_frame,
        "template_dataset": ds_templates,
        "member_dataset": ds_members,
        "member_summary": summary,
        "acceptance": acceptance,
    }

def write_event_artifacts(paths, artifacts):
    paths["events_root"].mkdir(parents=True, exist_ok=True)
    _write_netcdf_replace(artifacts["template_dataset"], paths["template_bank_nc"])
    _write_netcdf_replace(artifacts["member_dataset"], paths["event_members_nc"])
    artifacts["member_summary"].to_csv(paths["event_summary_csv"], index=False)
    paths["event_acceptance_json"].write_text(
        json.dumps(artifacts["acceptance"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    pd.Series({"h": 0.0}, name="lag_days").rename_axis("driver").to_csv(paths["lagtimes_csv"])
    write_overview_plot(
        paths["event_overview_png"],
        artifacts["template_frame"],
        artifacts["member_summary"],
        artifacts["member_dataset"],
    )

def _write_netcdf_replace(dataset, path):
    # Avoid corrupting an existing NetCDF if a write fails halfway through.
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        dataset.to_netcdf(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

__all__ = [
    "coastal_components",
    "coastal_realization_metadata",
    "non_tidal_residual",
    "tide_preserving_total_water_level",
    "sample_return_periods",
    "bootstrap_body_sample",
    "hybrid_peak_sample",
    "hybrid_peak_sample_frame",
    "build_sampled_peaks",
    "extract_historical_templates",
    "template_bank_to_dataset",
    "build_surge_event_members",
    "build_acceptance_report",
    "write_overview_plot",
    "build_surge_event_artifacts",
    "write_event_artifacts",
    "template_columns",
    "member_columns",
]
