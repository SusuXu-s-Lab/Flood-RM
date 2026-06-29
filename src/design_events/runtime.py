"""Compatibility shim — design-event runtime paths now live in design_events_v2.runtime.

ADR-0021 convergence: ``build_paths``, ``resolve_scenario``, and the ``load_runtime``
(config, paths) loader moved to ``design_events_v2.runtime``. Re-exported here so notebook
and module imports of ``design_events.runtime`` keep resolving.
"""
from __future__ import annotations

from design_events_v2.runtime import build_paths, load_runtime, resolve_scenario

__all__ = ["build_paths", "load_runtime", "resolve_scenario"]
