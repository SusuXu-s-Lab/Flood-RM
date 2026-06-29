"""Compatibility shim — inland rainfall timing now lives in design_events_v2.timing.

ADR-0021 convergence: the inland (Wflow-coupled) timing descriptors and observed
diagnostics moved to ``design_events_v2.timing``. This module re-exports them so notebook
and builder imports of ``design_events.build_events.inland_timing`` keep resolving.
"""
from __future__ import annotations

from design_events_v2.timing import (
    attach_inland_rainfall_timing,
    observed_basin_lag,
    storm_loading_pattern,
    timing_seasonality,
)

__all__ = [
    "attach_inland_rainfall_timing",
    "observed_basin_lag",
    "storm_loading_pattern",
    "timing_seasonality",
]
