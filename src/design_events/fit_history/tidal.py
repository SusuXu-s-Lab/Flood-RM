"""Tide / non-tidal-residual split of the coastal water-level record.

The implementation is the single source of truth in ``design_events_v2.records``
(ADR-0021); this module re-exports it so notebooks importing
``design_events.fit_history.tidal`` keep working while ``design_events`` is hollowed
out into a compatibility shim over v2.
"""

from __future__ import annotations

from design_events_v2.records import coastal_components, non_tidal_residual

__all__ = ["coastal_components", "non_tidal_residual"]
