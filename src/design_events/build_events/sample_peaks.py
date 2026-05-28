from __future__ import annotations
from importlib import import_module
import numpy as np
import pandas as pd

history_curve = import_module("design_events.fit_history.return_curve")
load_historical_peak_marginal = history_curve.load_historical_peak_marginal
assign_severity_bands = import_module(
    "design_events.build_events.event_distribution"
).assign_severity_bands

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
    # Body sampler: bootstrap with replacement from historical peaks.
    # - Every body sample is an exact unmodified observed historical peak.
    #   No interpolation, no smoothing, no convex blending of neighbors.
    # - The empirical marginal of the synthetic body converges to the
    #   empirical marginal of the input peaks as n_samples grows. Sample
    #   variance matches the historical record's variance (no shrinkage
    #   toward neighborhood means, which a k-NN convex blend would induce).
    # - All samples lie in [min(data), max(data)]. A reviewer can trace any
    #   body sample back to a specific historical storm in the catalog.
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0 or n_samples <= 0:
        return np.empty(0, dtype=float)
    return rng.choice(data, size=int(n_samples), replace=True)


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
    # common peaks are bootstrap resamples of the historical record below
    # the splice threshold; rare peaks come from the fitted return-period
    # curve, drawn uniformly in log-RP space above the splice threshold.
    # If the tail is oversampled, sampling weights preserve the target
    # body/tail probability mass implied by the splice threshold.
    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if len(peaks) == 0:
        raise RuntimeError("no historical peaks available for hybrid sampling")
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
    # Synthetic body marginal == empirical marginal
    # of the truncated record (in expectation), so the only modeling
    # assumption in the body region is "future common storms resemble
    # past common storms" 
    body = bootstrap_body_sample(body_candidates, body_count, seed)
    rows = pd.DataFrame({"peak_m": body, "sampling_region": "body"})
    if tail_count:
        # Tail: start no smaller than the return period at the splice threshold.
        tail_settings = dict(settings)
        tail_settings["return_period_min_years"] = max(rp_min, splice_rp)
        tail_settings["return_period_max_years"] = rp_max
        tail = marginal.magnitude(sample_return_periods(tail_count, tail_settings, seed + 1))
        rows = pd.concat(
            [rows, pd.DataFrame({"peak_m": tail, "sampling_region": "tail"})],
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
    df = pd.DataFrame({
        "event_id": [f"evt_{i:04d}" for i in range(1, event_count + 1)],
        "peak_m": sample["peak_m"].to_numpy(dtype=float),
        "sample_rp_years": marginal.return_period(sample["peak_m"].to_numpy(dtype=float)),
        "sampling_region": sample["sampling_region"].to_numpy(),
        "sampling_weight": sample["sampling_weight"].to_numpy(dtype=float),
        "probability_weight": sample["probability_weight"].to_numpy(dtype=float),
    })
    df["severity_band"] = assign_severity_bands(df["sample_rp_years"], settings.get("severity_bands"))
    paths["sampled_peaks_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["sampled_peaks_csv"], index=False, float_format="%.10g")
    return df
