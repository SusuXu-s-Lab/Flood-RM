"""design_events — reference prototype for the Event Catalog.

A small, auditable re-implementation of the probability law + Field-Preserving
Realization framework whose code can be read alongside
``docs/CONTEXT.md``. It is the compatibility-preserving replacement prototype
and peer-review reference: it emits ``events.csv`` / ``drivers.csv`` /
``audit.json`` from existing Source Artifacts and Location Configuration, not
SFINCS/Wflow forcing.

This package does not import production ``design_events`` modules.
It may use shared repository infrastructure such as ``study_location`` for YAML
resolution. The production package stays frozen and authoritative for notebooks
until v2 reconciliation coverage supports a deliberate cutover.

Module map (target; built incrementally):

    probability.py   F_j, C, S_and, AEP, T, severity bands, importance weights
    records.py       driver series, paired POT observations, member libraries
    realization.py   analog selection, scale factors, field pointers
    timing.py        rainfall peak timing, coastal-rainfall lag, inland descriptors
    catalog.py       canonical event/driver catalog contract + audit
    coastal.py       coastal-only hydrograph/NTR/tide realization helpers
    inland.py        inland rainfall design + Wflow response handoff
    diagnostics.py   plots/tables/reports only
    runtime.py       Location YAML + Source Artifact manifests -> runtime_config
    workflow.py      notebook-facing orchestration only
"""

from __future__ import annotations

from design_events import catalog, coastal, diagnostics, inland, probability, records, realization, runtime, timing, workflow

__all__ = [
    "catalog",
    "coastal",
    "diagnostics",
    "inland",
    "probability",
    "records",
    "realization",
    "runtime",
    "timing",
    "workflow",
]
