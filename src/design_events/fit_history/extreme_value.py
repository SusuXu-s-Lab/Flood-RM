"""Extreme-value fitting (POT / block-maxima, AIC/BIC, return values).

The implementation is the single source of truth in ``design_events_v2.extreme_value``
(ADR-0021); this module re-exports it so ``design_events.fit_history.extreme_value`` keeps
working while ``design_events`` is hollowed into a compatibility shim over v2.
"""

from __future__ import annotations

from design_events_v2.extreme_value import (
    bootstrap_return_values,
    candidate_distributions,
    eva_block_maxima,
    eva_peaks_over_threshold,
    fit_best_distribution,
    fit_distribution,
    fit_peak_dataset,
    get_dist,
    get_frozen_dist,
    n_periods,
    normalize_time_frequency,
    observed_return_periods,
    peak_series,
    plot_return_values,
    return_values,
    rps_default,
    score_fit,
    scipy_name,
)

__all__ = [
    "bootstrap_return_values",
    "candidate_distributions",
    "eva_block_maxima",
    "eva_peaks_over_threshold",
    "fit_best_distribution",
    "fit_distribution",
    "fit_peak_dataset",
    "get_dist",
    "get_frozen_dist",
    "n_periods",
    "normalize_time_frequency",
    "observed_return_periods",
    "peak_series",
    "plot_return_values",
    "return_values",
    "rps_default",
    "score_fit",
    "scipy_name",
]
