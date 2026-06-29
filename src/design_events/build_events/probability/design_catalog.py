"""Compatibility shim — copula-joint design-catalog builder now lives in design_events_v2.build.

ADR-0021 convergence: ``build_joint_catalog`` / ``build_tail`` / ``fit_index_marginal`` and the
``JointCatalogResult`` moved to ``design_events_v2.build``. Re-exported here so notebook and
builder imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.build import (
    JointCatalogResult,
    build_joint_catalog,
    build_tail,
    default_realization_specs,
    fit_index_marginal,
)

__all__ = [
    "JointCatalogResult",
    "build_joint_catalog",
    "build_tail",
    "default_realization_specs",
    "fit_index_marginal",
]
