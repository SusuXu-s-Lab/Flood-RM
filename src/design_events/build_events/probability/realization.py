"""Field-Preserving Realization bridge: sampled Driver Probability Index -> observed field.

Layer 2 of the two-layer framework. The single source of truth is
``design_events_v2.realization`` (ADR-0021); this module re-exports it under the production
names while ``design_events`` is hollowed into a compatibility shim over v2.
"""

from __future__ import annotations

from design_events_v2.realization import (
    attach_field_preserving_realization,
    draw_lags as draw_relative_lags,
    select_analog as select_analog_realization,
)

clip_eps = 1e-12

__all__ = [
    "attach_field_preserving_realization",
    "draw_relative_lags",
    "select_analog_realization",
    "clip_eps",
]
