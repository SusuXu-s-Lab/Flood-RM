"""Two-sided conditional POT co-occurrence sampling of compound flood drivers.

The implementation is the single source of truth in ``design_events_v2.records``
(ADR-0021); this module re-exports it so ``design_events.fit_history.paired_observations``
keeps working while ``design_events`` is hollowed into a compatibility shim over v2.
"""

from __future__ import annotations

from design_events_v2.records import (
    build_paired_observations,
    calibrate_threshold_for_rate,
    declustered_pot_peaks,
    distinct_event_rate,
)

__all__ = [
    "build_paired_observations",
    "calibrate_threshold_for_rate",
    "declustered_pot_peaks",
    "distinct_event_rate",
]
