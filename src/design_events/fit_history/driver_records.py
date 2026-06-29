"""Compatibility shim — driver-record assembly now lives in design_events_v2.driver_records.

ADR-0021 convergence: config-driven paired-observation assembly and member-library
construction (including the same-analog wave-coupling special case) moved to
``design_events_v2.driver_records``. This module re-exports them so notebook and builder
imports of ``design_events.fit_history.driver_records`` keep resolving.
"""
from __future__ import annotations

from design_events_v2.driver_records import (
    assemble_paired_observations,
    assemble_paired_observations_from_config,
    build_member_libraries,
    cooccurrence_params_from_config,
    record_specs_from_config,
)
from design_events_v2.records import load_driver_series, member_library_from_records

__all__ = [
    "assemble_paired_observations",
    "assemble_paired_observations_from_config",
    "build_member_libraries",
    "cooccurrence_params_from_config",
    "record_specs_from_config",
    "load_driver_series",
    "member_library_from_records",
]
