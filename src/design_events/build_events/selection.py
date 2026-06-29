"""Compatibility shim — stress/training selection now lives in design_events_v2.selection.

ADR-0021 convergence: resilience stress/training selection, compound stress pairing,
antecedent soil-moisture stamping, and event-distribution summaries moved to
``design_events_v2.selection``. Severity bands stay sourced from ``design_events_v2.probability``.
This module re-exports them so notebook and builder imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.probability import assign_severity_bands, default_severity_bands
from design_events_v2.selection import (
    apply_compound_stress_pairing,
    attach_antecedent_soil_moisture,
    default_benchmark_return_period_years,
    default_stress_training_severity_fractions,
    select_training,
    summarize_event_distribution,
    write_event_distribution_artifacts,
    write_resilience_stress_training_artifacts,
    # private helpers retained for the selection test net
    _compound_roles,
    _rainfall_offsets,
    _select_compound_rainfall_member,
)

__all__ = [
    "assign_severity_bands",
    "default_severity_bands",
    "apply_compound_stress_pairing",
    "attach_antecedent_soil_moisture",
    "default_benchmark_return_period_years",
    "default_stress_training_severity_fractions",
    "select_training",
    "summarize_event_distribution",
    "write_event_distribution_artifacts",
    "write_resilience_stress_training_artifacts",
]
