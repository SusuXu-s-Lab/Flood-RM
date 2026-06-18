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
    # common peaks are bootstrap resamples of the historical record below
    # the splice threshold; rare peaks come from the fitted return-period
    # curve, drawn uniformly in log-RP space above the splice threshold.
    # If the tail is oversampled, sampling weights preserve the target
    # body/tail probability mass implied by the splice threshold.
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
    # Synthetic body marginal == empirical marginal
    # of the truncated record (in expectation), so the only modeling
    # assumption in the body region is "future common storms resemble
    # past common storms" 
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
