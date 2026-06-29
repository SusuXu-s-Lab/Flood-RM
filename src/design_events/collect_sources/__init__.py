"""Compatibility shim package — source collection moved to top-level ``collect_sources/``.

ADR-0021 convergence: ``design_events.collect_sources`` is now the standalone
``collect_sources`` package (irreducible source-specific IO, not science). This shim
re-exports the public surface and aliases every submodule so existing imports such as
``import design_events.collect_sources.workflow`` and
``from design_events.collect_sources.usgs_streamgages import ...`` keep resolving.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys

import collect_sources as _cs
from collect_sources import *  # noqa: F401,F403

# Alias each collect_sources submodule under this package path so submodule imports of
# design_events.collect_sources.<name> resolve to the one moved module object.
for _info in pkgutil.iter_modules(_cs.__path__):
    _module = importlib.import_module(f"collect_sources.{_info.name}")
    sys.modules[f"{__name__}.{_info.name}"] = _module
    setattr(sys.modules[__name__], _info.name, _module)
