# Flood-RM Agent Instructions

This repository values scientific correctness, notebook traceability, and human-readable backend code over maximal abstraction or compatibility layering.

The goal of refactoring is not to make the code look more “enterprise.” The goal is to make the code smaller, flatter, easier to trace, and more idiomatic for scientific Python.

A successful cleanup usually:
- deletes redundant code,
- replaces hand-rolled plumbing with pandas/numpy/scipy/geopandas/shapely/pathlib/matplotlib idioms,
- separates core scientific logic from file/path/config/plotting plumbing,
- preserves notebook-facing behavior,
- reduces the number of concepts a new reader must understand.
- treats helper-module extraction as a temporary safety move, not the final architecture.

A failed cleanup usually:
- adds new manager/factory/adapter layers,
- creates compatibility wrappers without removing old paths,
- moves code around without reducing complexity,
- introduces abstract base classes for one implementation,
- converts obvious procedural scientific code into opaque object hierarchies,
- keeps every old function “just in case,”
- passes tests but leaves the code harder to understand.

When in doubt, prefer boring, direct, idiomatic Python.

Thin wrappers are not automatically worth keeping because they have call sites. Preserve
the domain contract, not the wrapper. For example, Stable Grid IDs are a real Grid
Dataset contract, but one-line ID helpers, generic parquet writers, optional hash
wrappers, permissive value parsers, and per-domain path mini-systems should be removed,
inlined, or replaced with direct scientific-library calls once tests characterize the
artifact behavior.

See `docs/plumbing_reduction_directive.md` before continuing any pass that touches
`src/` path, config, manifest, parser, hash, artifact, or compatibility-wrapper
plumbing. The rule is repository-wide; the deleted `power.artifacts` and
`power.notebook` facades are only the first obvious examples.
