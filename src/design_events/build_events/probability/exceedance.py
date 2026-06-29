"""AND joint-exceedance primitives — single-sourced in design_events_v2 (ADR-0021).

The exact AND-survival math, the return-period conversion, the seeded (reproducible) vine
CDF, and most-likely AND-isoline design-event selection live in
``design_events_v2.probability``; this module re-exports them for back-compat callers. The
old production-only label/dispatcher/mixture wrappers were removed once
``build_joint_catalog``/``build_tail`` moved onto the v2 seam.
"""

from __future__ import annotations

from design_events_v2.probability import (
    and_return_period,
    and_survival_empirical,
    and_survival_from_cdf,
    combined_and_frequency,
    combined_return_period,
    seeded_cdf,
    select_design_events_on_and_isolines as select_most_likely_design_events,
)

__all__ = [
    "and_return_period",
    "and_survival_empirical",
    "and_survival_from_cdf",
    "combined_and_frequency",
    "combined_return_period",
    "seeded_cdf",
    "select_most_likely_design_events",
]
