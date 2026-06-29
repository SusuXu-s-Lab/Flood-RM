"""Compound-driver dependence — superseded by the v2 seam (ADR-0021).

The vine fit, AND-labeled importance sampling, and storm-type mixture now live in
``design_events_v2`` (``workflow.fit_law`` / ``workflow.sample_catalog``, ``mixture``,
``probability``); ``design_catalog.build_joint_catalog`` composes that seam. This module
re-exports the stress-budget gate for back-compat callers.
"""

from __future__ import annotations

from design_events_v2.probability import check_stress_budget

__all__ = ["check_stress_budget"]
