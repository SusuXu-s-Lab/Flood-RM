"""Compatibility shim — inland design-catalog builder now lives in design_events_v2.build.

ADR-0021 convergence: ``build_inland_catalog`` / ``build_inland_historical_tail_catalog`` /
``fit_reference_streamflow_pot`` and ``InlandDesignCatalogResult`` moved to
``design_events_v2.build``. Re-exported here so notebook and builder imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.build import (
    InlandDesignCatalogResult,
    build_inland_catalog,
    build_inland_historical_tail_catalog,
    fit_reference_streamflow_pot,
)

__all__ = [
    "InlandDesignCatalogResult",
    "build_inland_catalog",
    "build_inland_historical_tail_catalog",
    "fit_reference_streamflow_pot",
]
