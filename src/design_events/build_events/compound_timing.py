"""Compatibility shim — compound rainfall timing now lives in design_events_v2.timing.

ADR-0021 convergence: the production wide-catalog compound-timing functions moved to
``design_events_v2.timing``. This module re-exports them so notebook and builder imports
of ``design_events.build_events.compound_timing`` keep resolving.
"""
from __future__ import annotations

from design_events_v2.timing import (
    attach_empirical_rainfall_lags,
    enrich_rainfall_member_timing,
    observed_compound_lag_pool,
    weighted_observed_lag_analog,
)

__all__ = [
    "attach_empirical_rainfall_lags",
    "enrich_rainfall_member_timing",
    "observed_compound_lag_pool",
    "weighted_observed_lag_analog",
]
