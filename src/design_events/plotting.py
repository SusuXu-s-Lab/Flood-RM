"""Compatibility shim — Event Catalog review figures now live in design_events_v2.plotting.

ADR-0021 convergence: the single-file plotting module moved to
``design_events_v2.plotting``. This shim aliases that module so notebook imports
(``from design_events import plotting as P``) resolve every public and private name.
"""
from __future__ import annotations

import sys

from design_events_v2 import plotting as _plotting

sys.modules[__name__] = _plotting
