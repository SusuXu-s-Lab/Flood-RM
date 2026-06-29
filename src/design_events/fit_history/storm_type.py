"""Storm-type classification of historical compound events.

The implementation is the single source of truth in ``design_events_v2.storm_type``
(ADR-0021); this module re-exports it so ``design_events.fit_history.storm_type`` keeps
working while ``design_events`` is hollowed into a compatibility shim over v2.
"""

from __future__ import annotations

from design_events_v2.storm_type import classify_from_config, classify_storm_type

__all__ = ["classify_storm_type", "classify_from_config"]
