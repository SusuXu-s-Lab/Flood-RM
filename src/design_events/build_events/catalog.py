"""Compatibility shim — wide Event Catalog assembly now lives in design_events_v2.catalog.

ADR-0021 convergence: catalog assembly, forcing pairing, and validation moved to
``design_events_v2.catalog``. Re-exported here so notebook and builder imports keep
resolving.
"""
from __future__ import annotations

from design_events_v2.catalog import (
    attach_forcing_members,
    build_event_catalog,
    catalog_columns,
    rebuild_forcing_pairing,
    validate_event_catalog,
    write_event_catalog_audit,
)

__all__ = [
    "attach_forcing_members",
    "build_event_catalog",
    "catalog_columns",
    "rebuild_forcing_pairing",
    "validate_event_catalog",
    "write_event_catalog_audit",
]
