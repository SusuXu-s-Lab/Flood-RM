"""Compatibility shim — coastal hybrid sampler + surge templates now live in design_events_v2.coastal.

ADR-0021 convergence: the hybrid body/tail peak sampler, surge hydrograph template
extraction, synthetic event-member construction, and the tide-resolving MSL-shift staging
moved to ``design_events_v2.coastal``. The NTR/tide contract is unchanged. This module
re-exports them so notebook and builder imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.coastal import (
    bootstrap_body_sample,
    build_acceptance_report,
    build_sampled_peaks,
    build_surge_event_artifacts,
    build_surge_event_members,
    extract_historical_templates,
    hybrid_peak_sample,
    hybrid_peak_sample_frame,
    member_columns,
    sample_return_periods,
    template_bank_to_dataset,
    template_columns,
    write_event_artifacts,
    write_overview_plot,
)

__all__ = [
    "bootstrap_body_sample",
    "build_acceptance_report",
    "build_sampled_peaks",
    "build_surge_event_artifacts",
    "build_surge_event_members",
    "extract_historical_templates",
    "hybrid_peak_sample",
    "hybrid_peak_sample_frame",
    "member_columns",
    "sample_return_periods",
    "template_bank_to_dataset",
    "template_columns",
    "write_event_artifacts",
    "write_overview_plot",
]
