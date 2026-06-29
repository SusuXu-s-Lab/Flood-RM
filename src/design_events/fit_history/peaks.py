"""Compatibility shim — historical peak EVA workflow now lives in design_events_v2.peaks.

ADR-0021 convergence: hourly-water-level loading, peak extraction, MSL detrending, the
return-period fit, stationarity diagnostics, and catalog/plot writers moved to
``design_events_v2.peaks``. This module re-exports them so notebook and builder imports
of ``design_events.fit_history.peaks`` keep resolving.
"""
from __future__ import annotations

from design_events_v2.peaks import (
    build_catalog,
    build_threshold_model_sensitivity,
    detrend_hourly_to_reference_epoch,
    fit_historical_peaks,
    load_hourly_waterlevel,
    stationarity_report,
    write_catalog,
    write_marginal_bootstrap_meta,
    write_marginal_rps_ci,
    write_return_value_plot,
    write_stationarity_report,
)

__all__ = [
    "build_catalog",
    "build_threshold_model_sensitivity",
    "detrend_hourly_to_reference_epoch",
    "fit_historical_peaks",
    "load_hourly_waterlevel",
    "stationarity_report",
    "write_catalog",
    "write_marginal_bootstrap_meta",
    "write_marginal_rps_ci",
    "write_return_value_plot",
    "write_stationarity_report",
]
