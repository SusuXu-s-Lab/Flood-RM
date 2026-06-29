"""Compatibility shim — marginal curves + params/RP schema now live in design_events_v2.records.

ADR-0021 convergence: the marginal classes, EVA-dataset adapter, and the params/RP CSV
schema are single-sourced in ``design_events_v2.records``. This module re-exports them so
``fit_history.peaks``, ``build_events.coastal``, and notebook imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.records import (
    EmpiricalMarginal,
    HistoricalPeakMarginal,
    from_eva_dataset,
    load_historical_peak_marginal,
    marginal_params_frame,
    marginal_rps_frame,
    write_historical_peak_marginal,
)

clip_eps = 1e-9

__all__ = [
    "EmpiricalMarginal",
    "HistoricalPeakMarginal",
    "from_eva_dataset",
    "marginal_params_frame",
    "marginal_rps_frame",
    "write_historical_peak_marginal",
    "load_historical_peak_marginal",
]
